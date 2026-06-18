<div align="center">

# 🏥 CARE

## Clinical Assessment and Response Evaluation Framework

[![ACL 2026](https://img.shields.io/badge/ACL-2026_Oral-blue?style=flat-square&logo=arxiv)](https://arxiv.org/pdf/2604.05795)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**Evaluating therapist responses through clinically grounded therapeutic principles**

[📄 Read the Paper](https://arxiv.org/pdf/2604.05795) • [🔗 Project](https://github.com) • [📧 Contact](#contact)

</div>

---

## 🎯 Overview

**CARE** is an intelligent framework that evaluates therapist responses across **six clinically grounded therapeutic dimensions** using advanced NLP techniques:

| Dimension | Focus |
|-----------|-------|
| 🗣️ **Non-Judgmental Language** | Acceptance without judgment |
| 💝 **Warmth and Encouragement** | Supportive and affirming tone |
| 🔓 **Respect for Autonomy** | Supporting patient independence |
| 👂 **Active Listening** | Attentive engagement |
| ❤️ **Reflecting Feelings** | Emotional understanding and validation |
| 🎯 **Situational Appropriateness** | Context-relevant responses |

Each response is scored on an **ordinal scale**: **-2** (Strong Negative) → **+2** (Strong Positive)

---

## ✨ Key Techniques

<div align="center">

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

</div>

- 🔍 **Context-aware Modeling** - Understands conversation flow and context
- 🔄 **Retrieval-Augmented Generation (RAG)** - Retrieves relevant examples for enhanced reasoning
- 🎓 **Knowledge-Distilled Chain-of-Thought (KD-CoT)** - Generates interpretable explanations
- 📊 **Multi-label Ordinal Classification** - Predicts multi-dimensional ordinal scores

---

## ⚡ Highlights

✅ Clinically grounded therapeutic evaluation  
✅ Explanation-guided learning with interpretability  
✅ Retrieval-enhanced reasoning for better context awareness  
✅ Multi-task ordinal modeling for nuanced predictions  
✅ Comprehensive evaluation with QWK metrics  

---

## 📁 Repository Structure

```
.
├── CARE.py                    # Training + explanation generation pipeline
├── Inference.py               # Inference + evaluation pipeline
├── train/                     # Training datasets (provided upon request)
├── val/                       # Validation datasets
├── test/                      # Test datasets
├── outputs/                   # Model checkpoints + results
└── README.md                  # This file
```

---

## 🔐 Dataset Access

> Due to data sensitivity and ethical considerations, the dataset is **not publicly hosted**.

To request access to the training and validation data, please fill out this form:
👉 [Request Dataset Access](https://forms.gle/jac2KMsYbeNZBrUU9)

After approval, you'll receive:
- `train/` - Training data with annotated therapist conversations
- `val/` - Validation data
- `test/` - Test data

### 📋 Data Format

Each CSV file includes:

| Field | Description |
|-------|-------------|
| **ID** | Unique identifier |
| **Utterance** | Therapist or patient response |
| **Type** | T (Therapist) or P (Patient) |
| **6 Labels** | Scores for each therapeutic dimension (-2 to +2) |

---

## 🚀 Quick Start

### 1️⃣ Installation

```bash
# Create environment
conda create -n care python=3.10
conda activate care

# Install dependencies
pip install torch transformers sentence-transformers peft scikit-learn tqdm matplotlib seaborn pandas
```

### 2️⃣ Training

Run the complete CARE pipeline:

```bash
python CARE.py
```

**This will:**
- Generate KD-CoT explanations (cached if already generated)
- Train the hierarchical classifier
- Save the best model to `outputs/best_classifier.pt`

### 3️⃣ Inference & Evaluation

```bash
python Inference.py
```

**This will:**
- Generate explanations for test data
- Run classification
- Save results to `inference_outputs/`:
  - `predictions.csv`
  - `confusion_matrices/`
  - `metrics.json`

---

## 🔬 Method Details

### 🎓 Knowledge-Distilled Chain-of-Thought (KD-CoT)

- Teacher models (GPT-4o, Mistral) generate reasoning explanations
- Explanations guide the student model (Qwen-based)
- **Note:** Teachers only generate explanations, not final predictions

### 🔍 Retrieval-Augmented Reasoning

- Uses **SentenceTransformers** for semantic similarity
- Retrieves top-k similar examples from knowledge base
- Provides contrastive signals (positive vs negative examples)

### 🔗 Cross-Attention Fusion

CARE integrates three key components:
1. Context representation
2. Distilled knowledge (explanations)
3. Utterance embeddings

These are fused via **cross-attention** before classification.

### 📊 Hierarchical Classification

- **Multi-label prediction** across 6 dimensions
- **Dual objective:**
  - Ordinal classification (5 classes)
  - Binary polarity prediction

### 📈 Evaluation Metrics

- **Quadratic Weighted Kappa (QWK)** - Measures inter-rater agreement accounting for magnitude of disagreement
- **Accuracy** - Standard classification accuracy
- **Confusion Matrices** - Detailed error analysis

---

## ⚙️ Configuration

Customize training parameters in `CARE.py`:

```python
CONFIG = {
    "batch_size": 16,           # Batch size for training
    "epochs": 10,               # Number of training epochs
    "lr": 5e-5,                 # Learning rate
    "context_window": 4,        # Conversation context window
    "top_k": 3                  # Top-k retrieved examples for RAG
}
```

---

## 💡 Important Notes

⚠️ **Computational Requirements:**
- Explanation generation is computationally intensive
- GPU recommended (NVIDIA CUDA-compatible GPU)
- Ensure sufficient memory for LLM inference (16GB+ recommended)

---

## 📚 Citation

If you use CARE in your research, please cite:

```bibtex
@inproceedings{care2026,
  title={CARE: Clinical Assessment and Response Evaluation Framework},
  author={[Your Authors]},
  booktitle={Proceedings of ACL 2026},
  year={2026},
  url={https://arxiv.org/pdf/2604.05795}
}
```
---
