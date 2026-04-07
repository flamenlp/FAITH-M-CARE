import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer, util
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import gc

CONFIG = {
    # Two different models
    "explainer_model_id": "Qwen/Qwen3-4B-Instruct-2507", 
    "classifier_model_id": "Qwen/Qwen3-4B-Instruct-2507",
    
    "embedding_model": "all-MiniLM-L6-v2",
    
    # PATHS
    "train_csv_folder": "./train",
    "val_csv_folder":   "./val",
    "test_csv_folder":  "./test",
    
    # JSON PATHS (Set to None to use defaults/auto-generate)
    "train_json": None,
    "val_json":   None,
    "test_json":  None,
    
    "output_dir": "./outputs",
    
    # RESUME
    "resume_checkpoint": None, 
    
    # HYPERPARAMETERS
    "batch_size": 16, 
    "grad_accum": 4, 
    "lr": 5e-5,
    "epochs": 10,
    "max_len": 1536,  
    "context_window": 4,
    "top_k": 3, 
    
    "labels": [
        "Non-Judgmental Language", "Warmth and Encouragement", 
        "Respect for Autonomy", "Active Listening", 
        "Reflecting Feelings", "Situational Appropriateness"
    ]
}

DEFAULT_RAG_DIR = None

if CONFIG["train_json"] is None: CONFIG["train_json"] = os.path.join(DEFAULT_RAG_DIR, "train_processed.json")
if CONFIG["val_json"] is None:   CONFIG["val_json"]   = os.path.join(DEFAULT_RAG_DIR, "val_processed.json")
if CONFIG["test_json"] is None:  CONFIG["test_json"]  = os.path.join(DEFAULT_RAG_DIR, "test_processed.json")

# Create directories if they don't exist
if not os.path.exists(CONFIG["output_dir"]):
    os.makedirs(CONFIG["output_dir"])

# Safely create directory for JSONs
json_dir = os.path.dirname(CONFIG["train_json"])
if json_dir and not os.path.exists(json_dir):
    os.makedirs(json_dir)

# Mapping
LABEL_TO_IDX = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
IDX_TO_LABEL = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}
NUM_CLASSES = 5

dtype_str = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
torch_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using Device: {device} | Precision: {dtype_str}")


def clean_label_column(series):
    def clean_val(x):
        try:
            s = str(x).strip().replace('`', '').replace("'", "").replace('"', '')
            val = int(float(s))
            if val < -2 or val > 2: return 0
            return val
        except:
            return 0
    return series.apply(clean_val)

def prepare_context(df, m=3):
    df['ID'] = df['ID'].astype(str)
    df['ConvID'] = df['ID'].apply(lambda x: x.split('_')[0])
    contexts = []
    
    for _, group in df.groupby('ConvID', sort=False):
        utterances = group['Utterance'].tolist()
        types = group['Type'].tolist()
        for i in range(len(group)):
            start = max(0, i - m)
            ctx_utts = utterances[start:i]
            ctx_types = types[start:i]
            ctx_str = ""
            for t, u in zip(ctx_types, ctx_utts):
                role = "Therapist" if t == 'T' else "Patient"
                ctx_str += f"{role}: {u}\n"
            contexts.append(ctx_str.strip())
    df['Context'] = contexts
    return df

def load_csv_data(folder_path):
    print(f"Loading CSVs from {folder_path}...")
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip()
            for lbl in CONFIG["labels"]:
                if lbl in df.columns:
                    df[lbl] = clean_label_column(df[lbl])
                else:
                    df[lbl] = 0
            dfs.append(df)
        except: pass
            
    if not dfs: return pd.DataFrame()
    full_df = pd.concat(dfs, ignore_index=True)
    full_df = prepare_context(full_df, m=CONFIG["context_window"])
    return full_df[full_df['Type'] == 'T'].dropna(subset=['Utterance']).reset_index(drop=True)

def create_ideal_sets(df, embed_model):
    """Creates disjoint sets of positive/negative examples for RAG."""
    print("Creating Ideal Sets for Retrieval...")
    # Embed everything
    utterances = df['Utterance'].astype(str).tolist()
    embeddings = embed_model.encode(utterances, convert_to_tensor=True, show_progress_bar=True)
    
    ideal_sets = {lbl: {'Pos': [], 'Neg': []} for lbl in CONFIG["labels"]}
    used_indices = set()
    
    # Sorted candidates
    candidates = {}
    for label in CONFIG["labels"]:
        vals = df[label]
        pos_idxs = vals[vals > 0].sort_values(ascending=False).index.tolist()
        neg_idxs = vals[vals < 0].abs().sort_values(ascending=False).index.tolist()
        candidates[label] = {'Pos': pos_idxs, 'Neg': neg_idxs}

    # Round-Robin Allocation
    keep_going = True
    while keep_going:
        added_this_round = False
        for label in CONFIG["labels"]:
            # Pos
            while candidates[label]['Pos']:
                idx = candidates[label]['Pos'].pop(0)
                if idx not in used_indices:
                    ideal_sets[label]['Pos'].append(idx)
                    used_indices.add(idx)
                    added_this_round = True
                    break
            # Neg
            while candidates[label]['Neg']:
                idx = candidates[label]['Neg'].pop(0)
                if idx not in used_indices:
                    ideal_sets[label]['Neg'].append(idx)
                    used_indices.add(idx)
                    added_this_round = True
                    break
        if not added_this_round: keep_going = False
        
    return ideal_sets, embeddings

def generate_explanations_for_split(df, split_name, ideal_sets, train_embeddings, train_df_ref, output_path):
    """Uses Explainer Model to generate explanations using RAG."""
    if os.path.exists(output_path):
        print(f"Explanations for {split_name} already exist at {output_path}. Skipping generation.")
        return

    print(f"\n--- Generating Explanations for {split_name} ---")
    
    # 1. Load Explainer Components
    print(f"Loading Explainer Model: {CONFIG['explainer_model_id']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["explainer_model_id"], trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["explainer_model_id"], 
        device_map="auto", 
        torch_dtype=torch.float16, 
        trust_remote_code=True
    ).eval()
    
    emb_model = SentenceTransformer(CONFIG["embedding_model"], device=device)
    
    # Embed current split
    curr_embs = emb_model.encode(df['Utterance'].astype(str).tolist(), convert_to_tensor=True, show_progress_bar=True)
    
    # Prepare Batches
    all_prompts = []
    meta_data = [] 
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building Prompts"):
        curr_vec = curr_embs[idx]
        for label in CONFIG["labels"]:
            # Retrieve
            pos_idxs = ideal_sets[label]['Pos']
            neg_idxs = ideal_sets[label]['Neg']
            
            def get_rag_text(pool_idxs):
                if not pool_idxs: return []
                pool_vecs = train_embeddings[pool_idxs]
                scores = util.cos_sim(curr_vec, pool_vecs)[0]
                k = min(CONFIG["top_k"], len(pool_idxs))
                _, top_k = torch.topk(scores, k=k)
                g_idxs = [pool_idxs[i] for i in top_k.cpu().numpy()]
                return train_df_ref.iloc[g_idxs]['Utterance'].tolist()

            pos_txt = "\n".join([f"- {t}" for t in get_rag_text(pos_idxs)])
            neg_txt = "\n".join([f"- {t}" for t in get_rag_text(neg_idxs)])
            
            # --- IMPROVED RUBRIC PROMPT ---
            prompt = (
                f"Context:\n{row['Context']}\n\n"
                f"Therapist Utterance: \"{row['Utterance']}\"\n\n"
                f"Trait: {label}\n"
                f"Reference High Scores (+1/+2):\n{pos_txt}\n"
                f"Reference Low Scores (-1/-2):\n{neg_txt}\n\n"
                "Scoring Rubric:\n"
                "-2: Strong Negative Correspondence (Active violation or harmful behavior)\n"
                "-1: Mild Negative Correspondence (Missed opportunity or slight insensitivity)\n"
                " 0: Neutral Correspondence (Standard interaction, neither good nor bad)\n"
                "+1: Mild Positive Correspondence (Good adherence, helpful but generic)\n"
                "+2: Strong Positive Correspondence (Exceptional adherence, deep empathy/skill)\n\n"
                "Task: Explain concisely which score level fits best and why, comparing with references."
            )
            
            msgs = [{"role": "user", "content": prompt}]
            full_prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            all_prompts.append(full_prompt)
            meta_data.append((idx, label))

    # Batch Generation
    print("Running Generation...")
    batch_size = 64
    generated_texts = []
    
    for i in tqdm(range(0, len(all_prompts), batch_size)):
        batch = all_prompts[i : i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, temperature=0.2, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        
        # Decode
        inp_len = inputs.input_ids.shape[1]
        decoded = tokenizer.batch_decode(out[:, inp_len:], skip_special_tokens=True)
        generated_texts.extend(decoded)
        
    # Reassemble
    results_map = {i: df.iloc[i].to_dict() for i in range(len(df))}
    for i in results_map: results_map[i]['Explanations'] = {}
    
    for (r_idx, lbl), txt in zip(meta_data, generated_texts):
        results_map[r_idx]['Explanations'][lbl] = txt
        
    def clean(obj):
        if isinstance(obj, (np.int64, np.int32)): return int(obj)
        return obj
    
    clean_res = [{k: clean(v) for k, v in x.items()} for x in list(results_map.values())]
    
    with open(output_path, 'w') as f:
        json.dump(clean_res, f)
        
    # Cleanup memory
    del model, tokenizer, emb_model
    torch.cuda.empty_cache()
    gc.collect()

def ensure_explanations_exist():
    """Orchestrates the generation if JSONs are missing."""
    if os.path.exists(CONFIG["train_json"]) and os.path.exists(CONFIG["val_json"]) and os.path.exists(CONFIG["test_json"]):
        print("All explanation JSONs found. Proceeding to training.")
        return

    print("\n[INFO] Explanations missing. Starting generation phase...")
    
    # Load raw data
    train_df = load_csv_data(CONFIG["train_csv_folder"])
    val_df = load_csv_data(CONFIG["val_csv_folder"])
    test_df = load_csv_data(CONFIG["test_csv_folder"])
    
    # Create Retrieval Base from Train Data
    emb_model = SentenceTransformer(CONFIG["embedding_model"], device=device)
    ideal_sets, train_embeddings = create_ideal_sets(train_df, emb_model)
    del emb_model
    
    # Generate
    generate_explanations_for_split(train_df, "Train", ideal_sets, train_embeddings, train_df, CONFIG["train_json"])
    generate_explanations_for_split(val_df, "Val", ideal_sets, train_embeddings, train_df, CONFIG["val_json"])
    generate_explanations_for_split(test_df, "Test", ideal_sets, train_embeddings, train_df, CONFIG["test_json"])
    
    print("[INFO] Explanation generation complete.\n")

def load_explanations_map(json_path):
    print(f"Loading Explanations from {json_path}...")
    with open(json_path, 'r') as f:
        data_list = json.load(f)
    mapping = {}
    for item in data_list:
        if 'ID' in item: mapping[str(item['ID'])] = item.get('Explanations', {})
    return mapping

# ==========================================
# 3. WEIGHTS & DATASET
# ==========================================

def compute_dual_weights(df):
    print("Computing Stable Dual Weights...")
    class_weights_list = []
    binary_weights_list = []
    
    for lbl in CONFIG["labels"]:
        vals = [LABEL_TO_IDX[int(x)] for x in df[lbl].tolist()]
        counts = np.bincount(vals, minlength=NUM_CLASSES)
        total = len(vals)
        
        # 1. Main Class Weights (Log Smoothing)
        safe_counts = np.maximum(counts, 1)
        log_weights = 1.0 + np.log(total / safe_counts)
        norm_w = log_weights / np.mean(log_weights)
        norm_w = np.clip(norm_w, 0.5, 3.0) 
        class_weights_list.append(torch.tensor(norm_w, dtype=torch.float32))
        
        # 2. Binary Weights
        is_negative = np.array(vals) < 2
        num_neg = np.sum(is_negative)
        num_pos = total - num_neg
        
        if num_neg > 0:
            raw_bin_w = num_pos / num_neg
            bin_w = np.sqrt(raw_bin_w) 
        else:
            bin_w = 1.0
        bin_w = min(bin_w, 5.0) 
        binary_weights_list.append(torch.tensor(bin_w, dtype=torch.float32))
        
    return (torch.stack(class_weights_list).to(device), torch.stack(binary_weights_list).to(device))

class RobustClassificationDataset(Dataset):
    def __init__(self, csv_folder, json_path, tokenizer, max_len):
        self.df = load_csv_data(csv_folder)
        self.expl_map = load_explanations_map(json_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        row_id = str(row['ID'])
        explanations = self.expl_map.get(row_id, {})
        
        analysis = ""
        for lbl in CONFIG["labels"]:
            expl_text = str(explanations.get(lbl, "")).strip() or "No info."
            analysis += f"{lbl}: {expl_text}\n"

        text_input = (
            f"Context:\n{row['Context']}\n"
            f"Therapist: \"{row['Utterance']}\"\n"
            f"Analysis:\n{analysis}\n"
            "Classify the clinical traits."
        )

        enc = self.tokenizer(
            text_input,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        targets = [LABEL_TO_IDX[int(row[lbl])] for lbl in CONFIG["labels"]]
            
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(targets, dtype=torch.long)
        }

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, last_hidden_state, attention_mask):
        w = self.attention(last_hidden_state)
        w[attention_mask == 0] = float('-inf')
        weights = torch.softmax(w, dim=1)
        pooled = torch.sum(last_hidden_state * weights, dim=1)
        return pooled

class QwenHierarchicalClassifier(nn.Module):
    def __init__(self, model_id, class_weights=None, binary_weights=None):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch_dtype, 
            trust_remote_code=True
        )
        
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, 
            r=16, lora_alpha=32, lora_dropout=0.1, 
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        self.backbone = get_peft_model(base_model, peft_config)
        self.pooler = AttentionPooling(self.config.hidden_size)
        self.norm = nn.LayerNorm(self.config.hidden_size)
        
        self.main_heads = nn.ModuleList()
        self.binary_heads = nn.ModuleList()
        
        for _ in range(6):
            self.main_heads.append(nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.config.hidden_size, 64),
                nn.ReLU(),
                nn.LayerNorm(64), 
                nn.Linear(64, NUM_CLASSES)
            ))
            self.binary_heads.append(nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.config.hidden_size, 64),
                nn.ReLU(),
                nn.LayerNorm(64),
                nn.Linear(64, 1) 
            ))
            
        self._init_head_weights()
        self.pooler.to(device=device, dtype=torch.float32)
        self.norm.to(device=device, dtype=torch.float32)
        self.main_heads.to(device=device, dtype=torch.float32)
        self.binary_heads.to(device=device, dtype=torch.float32)
        
        self.class_weights = class_weights
        self.binary_weights = binary_weights

    def _init_head_weights(self):
        for head in self.main_heads:
            nn.init.normal_(head[-1].weight, std=0.01)
            nn.init.zeros_(head[-1].bias)
        for head in self.binary_heads:
            nn.init.normal_(head[-1].weight, std=0.01)
            nn.init.zeros_(head[-1].bias)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        pooled = self.pooler(last_hidden.to(dtype=torch.float32), attention_mask)
        pooled = self.norm(pooled)
        
        main_logits_list = []
        binary_logits_list = []
        for i in range(6):
            main_logits_list.append(self.main_heads[i](pooled))
            binary_logits_list.append(self.binary_heads[i](pooled))
            
        main_logits = torch.stack(main_logits_list, dim=1)
        binary_logits = torch.stack(binary_logits_list, dim=1).squeeze(-1)
        
        loss = None
        if labels is not None:
            total_loss = 0
            binary_targets = (labels < 2).float()
            for i in range(6):
                w_c = self.class_weights[i] if self.class_weights is not None else None
                loss_main = nn.CrossEntropyLoss(weight=w_c)(main_logits[:, i, :], labels[:, i])
                w_b = self.binary_weights[i] if self.binary_weights is not None else None
                loss_bin = nn.BCEWithLogitsLoss(pos_weight=w_b)(binary_logits[:, i], binary_targets[:, i])
                total_loss += loss_main + loss_bin
            loss = total_loss / 6.0
            
        return {"loss": loss, "logits": main_logits, "binary_logits": binary_logits}

def train():
    # 1. Generate Explanations if missing
    ensure_explanations_exist()
    
    print("\n--- Initializing Classification Training ---")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["classifier_model_id"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load Datasets using the JSONs (now guaranteed to exist)
    train_ds = RobustClassificationDataset(CONFIG["train_csv_folder"], CONFIG["train_json"], tokenizer, CONFIG["max_len"])
    val_ds = RobustClassificationDataset(CONFIG["val_csv_folder"], CONFIG["val_json"], tokenizer, CONFIG["max_len"])
    
    c_weights, b_weights = compute_dual_weights(train_ds.df)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"])
    
    model = QwenHierarchicalClassifier(CONFIG["classifier_model_id"], class_weights=c_weights, binary_weights=b_weights)
    
    # --- RESUME LOGIC ---
    if CONFIG["resume_checkpoint"] and os.path.exists(CONFIG["resume_checkpoint"]):
        print(f"Resuming training from: {CONFIG['resume_checkpoint']}")
        state_dict = torch.load(CONFIG["resume_checkpoint"], map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("Starting training from scratch.")
        
    model.to(device)
    model.backbone.print_trainable_parameters()
    
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=CONFIG["lr"], weight_decay=0.01)
    
    num_steps = len(train_loader) * CONFIG["epochs"] // CONFIG["grad_accum"]
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(num_steps*0.1), num_training_steps=num_steps)
    
    best_qwk = -1.0
    
    print("\nStarting Hierarchical Training...")
    
    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_loss = 0
        current_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        optimizer.zero_grad()
        
        for step, batch in enumerate(progress):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            out = model(input_ids, mask, labels)
            
            if torch.isnan(out['loss']):
                print("NaN Loss detected. Skipping.")
                optimizer.zero_grad()
                continue
            
            loss = out['loss'] / CONFIG["grad_accum"]
            loss.backward()
            
            if (step + 1) % CONFIG["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            current_loss += loss.item() * CONFIG["grad_accum"]
            train_loss += loss.item() * CONFIG["grad_accum"]
            
            if step % 10 == 0:
                progress.set_postfix(loss=current_loss / 10)
                current_loss = 0
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_preds = []
        val_trues = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                out = model(input_ids, mask, labels)
                preds_idx = torch.argmax(out['logits'], dim=2)
                val_preds.extend(preds_idx.cpu().numpy())
                val_trues.extend(labels.cpu().numpy())
        
        val_preds = np.array(val_preds)
        val_trues = np.array(val_trues)
        
        qwks = []
        for i in range(6):
            q = cohen_kappa_score(val_trues[:, i], val_preds[:, i], weights='quadratic')
            qwks.append(q)
        avg_qwk = np.mean(qwks)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Avg QWK: {avg_qwk:.4f}")
        
        if avg_qwk > best_qwk:
            best_qwk = avg_qwk
            torch.save(model.state_dict(), os.path.join(CONFIG["output_dir"], "best_classifier.pt"))
            print("--> Saved Best Model")

def evaluate():
    print("\n--- Running Final Test ---")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["classifier_model_id"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    test_ds = RobustClassificationDataset(CONFIG["test_csv_folder"], CONFIG["test_json"], tokenizer, CONFIG["max_len"])
    test_loader = DataLoader(test_ds, batch_size=8)
    
    model = QwenHierarchicalClassifier(CONFIG["classifier_model_id"]) 
    model.load_state_dict(torch.load(os.path.join(CONFIG["output_dir"], "best_classifier.pt")))
    model.to(device)
    model.eval()
    
    all_preds_idx = []
    all_trues_idx = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            out = model(input_ids, mask)
            preds = torch.argmax(out['logits'], dim=2)
            
            all_preds_idx.extend(preds.cpu().numpy())
            all_trues_idx.extend(labels.cpu().numpy())
            
    all_preds_idx = np.array(all_preds_idx)
    all_trues_idx = np.array(all_trues_idx)
    
    all_preds_real = np.vectorize(IDX_TO_LABEL.get)(all_preds_idx)
    all_trues_real = np.vectorize(IDX_TO_LABEL.get)(all_trues_idx)
    
    metrics = {}
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    labels_range = [-2, -1, 0, 1, 2]
    
    print("\nResults:")
    for i, lbl in enumerate(CONFIG["labels"]):
        p = all_preds_real[:, i]
        t = all_trues_real[:, i]
        qwk = cohen_kappa_score(t, p, weights='quadratic')
        acc = accuracy_score(t, p)
        metrics[lbl] = {"QWK": qwk, "Acc": acc}
        print(f"{lbl:<30} | QWK: {qwk:.3f} | Acc: {acc:.3f}")
        
        cm = confusion_matrix(t, p, labels=labels_range)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=labels_range, yticklabels=labels_range)
        axes[i].set_title(f"{lbl}\nQWK: {qwk:.3f}")
        
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "class_confusion_matrices.png"))
    with open(os.path.join(CONFIG["output_dir"], "test_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    train()
    evaluate() 
