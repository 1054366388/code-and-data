Prot-STAR: Neuro-Symbolic Omics Framework

This repository contains the official PyTorch implementation of Prot-STAR.

Prot-STAR is a neuro-symbolic framework designed for omics data analysis. It integrates Structure-Grounded Pruning (utilizing Knowledge Graphs) with Adaptive Semantic Quantization to transform continuous proteomic data into semantic prompts for Large Language Model (LLM) instruction tuning.

📂 Repository Structure

.
├── main.py                   # Entry point for the pipeline
├── src/
│   ├── config.py             # Hyperparameters and path configurations
│   ├── core_modules.py       # Core logic: Pruning (Eq.4), Quantization (Eq.2-3), Prompting (Eq.5)
│   ├── data_loader.py        # Data ingestion for heterogeneous omics formats
│   └── model_engine.py       # LLM initialization and LoRA training engine
└── requirements.txt          # Python dependencies


🛠️ Installation

Clone the repository:

git clone <anonymous_link>
cd Prot-STAR


Create a virtual environment (Recommended):

conda create -n protstar python=3.10
conda activate protstar


Install dependencies:

pip install torch transformers peft datasets pandas openpyxl scikit-learn


(Note: Ensure you have a version of PyTorch compatible with your CUDA version).

📊 Data Preparation

To reproduce the experiments, organize your data in the following structure within the project root. The framework expects heterogeneous Excel files as described below.

data/
├── knowledge_graph/
│   └── S.xlsx                # Adjacency matrix for the Knowledge Graph (Proteins/Genes)
└── COADREAD/                 # Cohort Data Example
    ├── feature_name.xlsx     # List of protein/gene names (Header: Row 1)
    ├── samples_protein.xlsx  # Expression matrix (Rows=Proteins, Cols=Samples)
    └── lables.xlsx           # Classification targets (e.g., Subtypes)


IMPORTANT: > - feature_name.xlsx: Header is in row 1, data starts in row 2.

samples_protein.xlsx: Raw expression data with no header.

lables.xlsx: Target labels with no header.

🚀 Usage

1. Configuration

Before running, ensure src/config.py uses relative paths or points to your data directory.

Anonymity Note: Please ensure absolute paths containing user names (e.g., C:\Users\Name\...) are removed from src/config.py before submission.

2. Run the Pipeline

Execute the main script to start the full workflow (Data Loading -> Pruning -> Quantization -> Prompt Construction -> Training).

python main.py


3. Workflow Description

The main.py script executes the following phases corresponding to the paper's methodology:

Structure-Grounded Pruning (Eq. 4): Selects informative features using the Knowledge Graph (S.xlsx) and data variance.

Code: core.structure_grounded_pruning(X)

Adaptive Semantic Quantization (Eq. 2 & 3): Converts continuous expression values into semantic tokens (e.g., "High", "Low") using Empirical Cumulative Distribution Functions (ECDF).

Code: core.adaptive_semantic_quantization(X_selected)

Instruction Tuning (Eq. 5): Constructs clinical prompts and fine-tunes the Llama-3-8B model using LoRA (Low-Rank Adaptation).

Code: engine.run_training(prompts)

⚠️ Notes on LLM Access & Hardware

This framework is configured to use meta-llama/Meta-Llama-3-8B-Instruct.

HuggingFace Login: You must accept the Llama-3 license on HuggingFace and log in via CLI:

huggingface-cli login


Hardware:

A GPU with at least 16GB VRAM is recommended for training with the specified LoRA configuration.

If running on CPU or smaller GPUs, consider reducing BATCH_SIZE in src/config.py.

⚙️ Requirements

Python >= 3.8

PyTorch >= 2.0

pandas

numpy

transformers

peft

datasets

openpyxl

scikit-learn

accelerate
