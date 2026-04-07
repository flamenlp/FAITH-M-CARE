import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import gc
import glob
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from sentence_transformers import SentenceTransformer, util
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. CONFIGURATION
# ==========================================

CONFIG = {
    # --- MODEL IDENTIFIERS ---
    "explainer_model_id": "mistralai/Mistral-7B-Instruct-v0.2",
    "classifier_model_id": "Qwen/Qwen3-4B-Instruct-2507",
    "embedding_model": "all-MiniLM-L6-v2",
    
    "test_dataset_path": "test.csv",
    
    "train_csv_folder": "./train",

    "checkpoint_path": None,
    
    # --- OUTPUT ---
    "output_dir": "./inference_outputs",
    
    # --- PARAMS ---
    "batch_size": 16,
    "max_len": 1536,
    "context_window": 4, # Must match training context length
    "top_k": 2,          # Number of RAG examples to retrieve
    
    # Mappings
    "abbrev_map": {
        "NJ": "Non-Judgmental Language",
        "WE": "Warmth and Encouragement",
        "RA": "Respect for Autonomy",
        "AL": "Active Listening",
        "RF": "Reflecting Feelings",
        "SA": "Situational Appropriateness"
    }
}

# Standard Constants
LABELS = list(CONFIG["abbrev_map"].values())
LABEL_TO_IDX = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
IDX_TO_LABEL = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}
NUM_CLASSES = 5

# Setup Device
dtype_str = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
torch_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
device = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists(CONFIG["output_dir"]):
    os.makedirs(CONFIG["output_dir"])


def clean_label_column(series):
    """Robustly cleans labels to integers."""
    def clean_val(x):
        try:
            s = str(x).strip().replace('`', '').replace("'", "").replace('"', '')
            val = int(float(s))
            if val < -2 or val > 2: return 0
            return val
        except: return 0
    return series.apply(clean_val)

def prepare_context(df, m=3):
    """Builds conversation history context."""
    # Handle different ID formats
    if 'conv_id' in df.columns:
        # For new test set format
        df['ConvID'] = df['conv_id'].astype(str)
        if 'turn_index' in df.columns:
            df = df.sort_values(by=['ConvID', 'turn_index'])
        text_col = 'utterance'
        speaker_col = 'speaker'
    else:
        # For original train set format
        df['ID'] = df['ID'].astype(str)
        df['ConvID'] = df['ID'].apply(lambda x: x.split('_')[0])
        text_col = 'Utterance'
        speaker_col = 'Type'

    contexts = []
    for _, group in df.groupby('ConvID', sort=False):
        utts = group[text_col].tolist()
        speakers = group[speaker_col].tolist()
        
        for i in range(len(group)):
            start = max(0, i - m)
            ctx_utts = utts[start:i]
            ctx_s = speakers[start:i]
            
            ctx_str = ""
            for s, u in zip(ctx_s, ctx_utts):
                role = "Therapist" if s == 'T' else "Patient"
                ctx_str += f"{role}: {u}\n"
            contexts.append(ctx_str.strip())
            
    df['Context'] = contexts
    return df

def load_train_reference_data(folder_path):
    """Loads and prepares the TRAINING data to be used as Knowledge Base."""
    print(f"Loading Reference Training Data from {folder_path}...")
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # Standardize columns for training data
            if 'utterance' in df.columns: df.rename(columns={'utterance': 'Utterance'}, inplace=True)
            if 'speaker' in df.columns: df.rename(columns={'speaker': 'Type'}, inplace=True)
            
            # Clean Labels
            for lbl in LABELS:
                if lbl in df.columns: df[lbl] = clean_label_column(df[lbl])
                else: df[lbl] = 0
            dfs.append(df)
        except: pass
            
    full_df = pd.concat(dfs, ignore_index=True)
    full_df = prepare_context(full_df, m=CONFIG["context_window"])
    
    # Filter for Therapist
    return full_df[full_df['Type'] == 'T'].reset_index(drop=True)

def load_test_data(csv_path):
    """Loads the NEW Test dataset."""
    print(f"Loading Test Data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Rename abbreviations
    df = df.rename(columns=CONFIG["abbrev_map"])
    
    # Prep context
    df = prepare_context(df, m=CONFIG["context_window"])
    
    # Clean Labels
    for lbl in LABELS:
        if lbl in df.columns: df[lbl] = clean_label_column(df[lbl])
        else: df[lbl] = 0
            
    # Filter Therapist
    df = df[df['speaker'] == 'T'].copy().reset_index(drop=True)
    # Ensure standard column name
    df.rename(columns={'utterance': 'Utterance'}, inplace=True)
    
    return df


def build_ideal_sets(train_df, embed_model):
    """Creates the Reference Knowledge Base from Training Data."""
    print("Building Ideal Sets from Training Data...")
    
    # 1. Embed All Training Utterances
    utterances = train_df['Utterance'].astype(str).tolist()
    embeddings = embed_model.encode(utterances, convert_to_tensor=True, show_progress_bar=True)
    
    ideal_sets = {lbl: {'Pos': [], 'Neg': []} for lbl in LABELS}
    used_indices = set()
    
    # 2. Select Disjoint Best Examples
    candidates = {}
    for label in LABELS:
        vals = train_df[label]
        pos_idxs = vals[vals > 0].sort_values(ascending=False).index.tolist()
        neg_idxs = vals[vals < 0].abs().sort_values(ascending=False).index.tolist()
        candidates[label] = {'Pos': pos_idxs, 'Neg': neg_idxs}

    keep_going = True
    while keep_going:
        added_this_round = False
        for label in LABELS:
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
        
    print("Ideal Sets Constructed.")
    return ideal_sets, embeddings


def generate_explanations(test_df, train_df, ideal_sets, train_embeddings):
    print("\n--- Phase 1: Generating Explanations (RAG) ---")
    
    # Load Explainer Model
    print("Loading Explainer LLM...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["explainer_model_id"], trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["explainer_model_id"], device_map="auto", torch_dtype=torch_dtype, trust_remote_code=True
    ).eval()
    
    # Embed Test Data
    print("Embedding Test Data...")
    emb_model = SentenceTransformer(CONFIG["embedding_model"], device=device)
    test_embs = emb_model.encode(test_df['Utterance'].astype(str).tolist(), convert_to_tensor=True, show_progress_bar=True)
    
    prompts = []
    meta = []
    
    print("Constructing Prompts...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        curr_vec = test_embs[idx]
        for label in LABELS:
            # RAG Retrieval
            pos_idxs = ideal_sets[label]['Pos']
            neg_idxs = ideal_sets[label]['Neg']
            
            def get_examples(pool_idxs):
                if not pool_idxs: return []
                scores = util.cos_sim(curr_vec, train_embeddings[pool_idxs])[0]
                _, topk = torch.topk(scores, k=min(CONFIG["top_k"], len(pool_idxs)))
                g_idxs = [pool_idxs[i] for i in topk.cpu().numpy()]
                return train_df.iloc[g_idxs]['Utterance'].tolist()

            pos_txt = "\n".join([f"- {t}" for t in get_examples(pos_idxs)])
            neg_txt = "\n".join([f"- {t}" for t in get_examples(neg_idxs)])
            
            # Prompt (Explanation Only)
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
            txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            prompts.append(txt)
            meta.append((idx, label))
            
    print("Generating Explanations...")
    generated_texts = []
    for i in tqdm(range(0, len(prompts), CONFIG["batch_size"])):
        batch = prompts[i : i + CONFIG["batch_size"]]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, temperature=0.6, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        
        new_toks = out[:, inputs.input_ids.shape[1]:]
        decoded = tokenizer.batch_decode(new_toks, skip_special_tokens=True)
        generated_texts.extend(decoded)
        
    explanations = [{} for _ in range(len(test_df))]
    for (r_idx, lbl), txt in zip(meta, generated_texts):
        explanations[r_idx][lbl] = txt
        
    del model, tokenizer, emb_model, train_embeddings
    torch.cuda.empty_cache()
    gc.collect()
    
    return explanations

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
        return torch.sum(last_hidden_state * weights, dim=1)

class QwenHierarchicalClassifier(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch_dtype, trust_remote_code=True)
        peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        self.backbone = get_peft_model(base_model, peft_config)
        self.pooler = AttentionPooling(self.config.hidden_size)
        self.norm = nn.LayerNorm(self.config.hidden_size)
        self.main_heads = nn.ModuleList()
        self.binary_heads = nn.ModuleList()
        for _ in range(6):
            self.main_heads.append(nn.Sequential(nn.Dropout(0.2), nn.Linear(self.config.hidden_size, 64), nn.ReLU(), nn.LayerNorm(64), nn.Linear(64, NUM_CLASSES)))
            self.binary_heads.append(nn.Sequential(nn.Dropout(0.2), nn.Linear(self.config.hidden_size, 64), nn.ReLU(), nn.LayerNorm(64), nn.Linear(64, 1)))
        self.pooler.to(device=device, dtype=torch.float32)
        self.norm.to(device=device, dtype=torch.float32)
        self.main_heads.to(device=device, dtype=torch.float32)
        self.binary_heads.to(device=device, dtype=torch.float32)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        pooled = self.pooler(outputs.hidden_states[-1].to(dtype=torch.float32), attention_mask)
        pooled = self.norm(pooled)
        logits = [head(pooled) for head in self.main_heads]
        return torch.stack(logits, dim=1)

class InferenceDataset(Dataset):
    def __init__(self, df, explanations, tokenizer, max_len):
        self.df = df
        self.explanations = explanations
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        expls = self.explanations[idx]
        analysis = ""
        for lbl in LABELS: analysis += f"{lbl}: {expls.get(lbl, '')}\n"
        text = f"Context:\n{row['Context']}\nTherapist: \"{row['Utterance']}\"\nAnalysis:\n{analysis}\nClassify the clinical traits."
        enc = self.tokenizer(text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        targets = [LABEL_TO_IDX[row[lbl]] for lbl in LABELS]
        return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0), "labels": torch.tensor(targets, dtype=torch.long)}

def predict_and_evaluate(test_df, explanations):
    print("\n--- Phase 2: Classification ---")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["classifier_model_id"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading Checkpoint: {CONFIG['checkpoint_path']}")
    model = QwenHierarchicalClassifier(CONFIG["classifier_model_id"])
    state_dict = torch.load(CONFIG["checkpoint_path"], map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    ds = InferenceDataset(test_df, explanations, tokenizer, CONFIG["max_len"])
    loader = DataLoader(ds, batch_size=CONFIG["batch_size"], shuffle=False)
    
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for batch in tqdm(loader):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            logits = model(ids, mask)
            preds = torch.argmax(logits, dim=2)
            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(batch['labels'].numpy())
            
    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)
    
    preds_real = np.vectorize(IDX_TO_LABEL.get)(all_preds)
    trues_real = np.vectorize(IDX_TO_LABEL.get)(all_trues)
    
    metrics = {}
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    axes = axes.flatten()
    
    print("\n--- RESULTS ---")
    for i, lbl in enumerate(LABELS):
        p, t = preds_real[:, i], trues_real[:, i]
        qwk = cohen_kappa_score(t, p, weights='quadratic')
        acc = accuracy_score(t, p)
        metrics[lbl] = {"QWK": qwk, "Acc": acc}
        print(f"{lbl:<30} | QWK: {qwk:.4f} | Acc: {acc:.4f}")
        
        cm = confusion_matrix(t, p, labels=[-2, -1, 0, 1, 2])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], xticklabels=[-2,-1,0,1,2], yticklabels=[-2,-1,0,1,2])
        axes[i].set_title(f"{lbl}\nQWK: {qwk:.3f}")
        
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "generalizability_confusion.png"))
    
    # Save CSV
    res_df = test_df.copy()
    for i, lbl in enumerate(LABELS):
        res_df[f"Pred_{lbl}"] = preds_real[:, i]
        res_df[f"True_{lbl}"] = trues_real[:, i]
        res_df[f"Expl_{lbl}"] = [e[lbl] for e in explanations]
        
    res_df.to_csv(os.path.join(CONFIG["output_dir"], "predictions.csv"), index=False)
    print(f"Done. Results saved to {CONFIG['output_dir']}")

if __name__ == "__main__":
    train_df = load_train_reference_data(CONFIG["train_csv_folder"])
    test_df = load_test_data(CONFIG["test_dataset_path"])
    
    emb_model = SentenceTransformer(CONFIG["embedding_model"], device=device)
    ideal_sets, train_embeddings = build_ideal_sets(train_df, emb_model)
    del emb_model
    
    explanations = generate_explanations(test_df, train_df, ideal_sets, train_embeddings)
    
    predict_and_evaluate(test_df, explanations)