# CARE
**Measuring What Matters: Assessing Therapeutic Principles in Mental-Health Conversation**

<div align="center">

[![ACL 2026](https://img.shields.io/badge/ACL-2026%20Oral-brightgreen?style=flat-square)](https://arxiv.org/pdf/2604.05795)
[![arXiv](https://img.shields.io/badge/arXiv-2604.05795-red?style=flat-square)](https://arxiv.org/pdf/2604.05795)

</div>

---

## 📖 About

A framework for evaluating therapist responses across clinically grounded therapeutic principles using:

- **🎯 Context-aware modeling**
- **🔍 Retrieval-Augmented Generation (RAG)**
- **🧠 Knowledge-Distilled Chain-of-Thought (KD-CoT)**
- **📊 Multi-label ordinal classification**

---

## 🎨 Overview

CARE evaluates therapist responses across **6 therapeutic dimensions**:

| Dimension | Description |
|-----------|-------------|
| 🚫 Non-Judgmental Language | Acceptance without judgment |
| 💛 Warmth and Encouragement | Supportive tone |
| 🤝 Respect for Autonomy | Patient independence |
| 👂 Active Listening | Attentive engagement |
| 💭 Reflecting Feelings | Emotional understanding |
| 🎯 Situational Appropriateness | Context relevance |

**Scoring Scale**: -2 (Strong Negative) → +2 (Strong Positive)

---

## ⚙️ Pipeline

```
Conversation Context
        ↓
   Retrieval (RAG)
        ↓
KD-CoT Explanation Generation
        ↓
  Cross-Attention Fusion
        ↓
 Ordinal Classification
```

---

## 📁 Repository Structure

```
.
├── CARE.py                      # Training + explanation generation pipeline
├── Inference.py                 # Inference + evaluation pipeline
├── train/                       # Training CSVs (provided upon request)
├── val/                         # Validation CSVs
├── test/                        # Test CSVs
├── outputs/                     # Model checkpoints + results
├── 1679_Measuring_What_Matters_As.pdf  # Paper PDF
└── README.md                    # This file
```

---

## 📊 Dataset Access

> ⚠️ **Note**: Due to data sensitivity and ethical considerations, the dataset is not publicly hosted.

**To request access**, please fill out the form:
👉 [https://forms.gle/jac2KMsYbeNZBrUU9](https://forms.gle/jac2KMsYbeNZBrUU9)

After approval, you will receive:
- `train/` - Training CSVs with annotated therapist conversations
- `val/` - Validation CSVs
- `test/` - Test CSVs

### Data Format

Each CSV includes:
- **ID**: Unique identifier
- **Utterance**: Text content
- **Type**: T (Therapist) or P (Patient)
- **Six therapeutic labels** with ordinal scores (-2 to +2)

---

## 🚀 Installation

```bash
conda create -n care python=3.10
conda activate care
pip install torch transformers sentence-transformers peft scikit-learn tqdm matplotlib seaborn pandas
```

---

## 💻 Usage

### Training

Run the full CARE pipeline:

```bash
python CARE.py
```

This will:
- ✅ Generate KD-CoT explanations (cached if already computed)
- ✅ Train the hierarchical classifier
- ✅ Save the best model to `outputs/best_classifier.pt`

### Inference & Evaluation

```bash
python Inference.py
```

This will:
- ✅ Generate explanations for test data
- ✅ Run classification
- ✅ Save outputs to `inference_outputs/`:
  - `predictions.csv`
  - `confusion matrices`
  - `metrics`

---

## 🔬 Method Details

### Knowledge-Distilled Chain-of-Thought (KD-CoT)
- Teacher models (GPT-4o, Mistral) generate reasoning explanations
- Explanations guide the student model (Qwen-based)
- ⚡ Teachers generate explanations only; final predictions are student-only

### Retrieval-Augmented Reasoning
- Uses **SentenceTransformers** for semantic similarity
- Retrieves top-k similar examples
- Provides contrastive signals (positive vs negative)

### Cross-Attention Fusion
Integrates three components:
- Context representation
- Distilled knowledge (explanations)
- Utterance embeddings

### Hierarchical Classification
- **Multi-label prediction** across 6 dimensions
- **Dual objective**:
  - Ordinal classification (5 classes)
  - Binary polarity prediction

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **QWK** | Quadratic Weighted Kappa |
| **Accuracy** | Classification accuracy |
| **Confusion Matrices** | Per-dimension analysis |

---

## ⚡ Key Features

✨ **Clinically grounded evaluation**  
✨ **Explanation-guided learning**  
✨ **Retrieval-enhanced reasoning**  
✨ **Multi-task ordinal modeling**  
✨ **Interpretable outputs**  

---

## ⚙️ Configuration

Modify parameters in `CARE.py`:

```python
CONFIG = {
    "batch_size": 16,
    "epochs": 10,
    "lr": 5e-5,
    "context_window": 4,
    "top_k": 3
}
```

---

## 💡 Notes

- ⏱️ Explanation generation is computationally intensive
- 🖥️ GPU recommended for optimal performance
- 💾 Ensure sufficient memory for LLM inference

---

## 📚 Citation

If you use CARE in your research, please cite:

```bibtex
@misc{mazhar2026measuringmattersassessingtherapeutic,
      title={Measuring What Matters!! Assessing Therapeutic Principles in Mental-Health Conversation}, 
      author={Abdullah Mazhar and Het Riteshkumar Shah and Aseem Srivastava and Smriti Joshi and Md Shad Akhtar},
      year={2026},
      eprint={2604.05795},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.05795}
}
```

---

<div align="center">
        
[📄 arXiv Paper](https://arxiv.org/pdf/2604.05795) • [📮 Request Dataset](https://forms.gle/jac2KMsYbeNZBrUU9)

</div>
