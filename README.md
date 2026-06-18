# CARE

**📄 Paper**: This work has been accepted at **ACL 2026** main conference for an **oral presentation**. Read the paper: [https://arxiv.org/pdf/2604.05795](https://arxiv.org/pdf/2604.05795)

---

This repository contains the official implementation of **CARE**, a framework for evaluating therapist responses across clinically grounded therapeutic principles using:

- Context-aware modeling
- Retrieval-Augmented Generation (RAG)
- Knowledge-Distilled Chain-of-Thought (KD-CoT)
- Multi-label ordinal classification

---

## Overview

CARE evaluates therapist responses across six therapeutic dimensions:

- Non-Judgmental Language  
- Warmth and Encouragement  
- Respect for Autonomy  
- Active Listening  
- Reflecting Feelings  
- Situational Appropriateness  

Each response is scored on an **ordinal scale**:

-2 (Strong Negative) → +2 (Strong Positive)


---

## Pipeline


Conversation Context
↓
Retrieval (RAG)
↓
KD-CoT Explanation Generation
↓
Cross-Attention Fusion
↓
Ordinal Classification


---

## Repository Structure

``` id="care_struct"
.
├── CARE.py              # Training + explanation generation pipeline
├── Inference.py         # Inference + evaluation pipeline
├── train/               # Training CSVs (provided upon request)
├── val/                 # Validation CSVs
├── test/                # Test CSVs
├── outputs/             # Model checkpoints + results

--> Dataset Access

Due to data sensitivity and ethical considerations, the dataset is not publicly hosted.

To request access, please fill out the form:

[https://forms.gle/jac2KMsYbeNZBrUU9]

After approval, you will receive:

train/
val/
test/

Each containing CSV files with annotated therapist conversations.

--> Data Format

Each CSV includes:

ID
Utterance
Type (T = Therapist, P = Patient)
Six therapeutic labels:
Label	Description
Non-Judgmental Language	Acceptance without judgment
Warmth and Encouragement	Supportive tone
Respect for Autonomy	Patient independence
Active Listening	Attentive engagement
Reflecting Feelings	Emotional understanding
Situational Appropriateness	Context relevance

--> Installation
conda create -n care python=3.10
conda activate care

pip install torch transformers sentence-transformers peft scikit-learn tqdm matplotlib seaborn pandas

--> Training

Run the full CARE pipeline:

python CARE.py

This will:

Generate KD-CoT explanations (if not already cached)
Train the hierarchical classifier
Save the best model to:
outputs/best_classifier.pt

--> Inference & Evaluation
python Inference.py

This will:

Generate explanations for test data
Run classification
Save outputs:
inference_outputs/
├── predictions.csv
├── confusion matrices
├── metrics

--> Method Details
KD-CoT (Knowledge Distillation)
Teacher models (e.g., GPT-4o, Mistral) generate reasoning explanations
These explanations guide the student model (Qwen-based)

--> Important:
Teacher models are only used for generating explanations, not final predictions.

Retrieval-Augmented Reasoning

Uses SentenceTransformers
Retrieves top-k similar examples
Provides contrastive signals (positive vs negative)

Cross-Attention Fusion

--> CARE integrates:

Context representation
Distilled knowledge (explanations)
Utterance embedding

via cross-attention before classification.

--> Hierarchical Classification

Multi-label prediction (6 dimensions)
Dual objective:
Ordinal classification (5 classes)
Binary polarity prediction

--> Evaluation Metrics
Quadratic Weighted Kappa (QWK)
Accuracy
Confusion Matrices
⚡ Key Features
✔ Clinically grounded evaluation
✔ Explanation-guided learning
✔ Retrieval-enhanced reasoning
✔ Multi-task ordinal modeling
✔ Interpretable outputs

--> Configuration

Modify parameters in CARE.py:

CONFIG = {
    "batch_size": 16,
    "epochs": 10,
    "lr": 5e-5,
    "context_window": 4,
    "top_k": 3
}

--> Notes
Explanation generation is computationally intensive
GPU recommended
Ensure sufficient memory for LLM inference


