# 🌟 Prot-STAR: Framework for Omics Data Analysis

> A robust, end-to-end Machine Learning pipeline designed specifically for advanced omics data analysis.

Prot-STAR is a powerful framework that intelligently integrates **Knowledge Graphs** with **Adaptive Semantic Quantization** to transform continuous proteomic data into structured semantic prompts, perfectly suited for Large Language Model (LLM) instruction tuning and robust downstream classification.

## 📂 Repository Structure

```text
📦 Prot-STAR
 ┣ 📜 main.py                # Entry point for the full execution pipeline
 ┣ 📂 src                    # Core module components
 ┃ ┣ 📜 config.py            # Global configurations, random seed & Optuna hyperparameter space
 ┃ ┣ 📜 core_modules.py      # Abstracted operations: Feature Pruning, Quantization, Prompt generation
 ┃ ┣ 📜 data_loader.py       # Data ingestion, processing, and Train/Test splits
 ┃ ┗ 📜 model_engine.py      # LoRA integration, training loops, and LLM initialization
 ┗ 📜 requirements.txt       # Python package dependencies
```

## 🛠️ Installation

**1. Clone the repository:**
```bash
git clone <repository_url>
cd Prot-STAR
```

**2. Create a virtual environment** *(Highly Recommended)*:
```bash
conda create -n protstar python=3.10 -y
conda activate protstar
```

**3. Install dependencies:**
```bash
pip install torch transformers peft datasets pandas openpyxl scikit-learn optuna
```

## 📊 Data Preparation

To successfully run the framework, structure your data directory exactly as follows within the project root. The system expects these heterogeneous Excel formats:

```text
data/
 ┣ 📂 knowledge_graph/
 ┃ ┗ 📊 S.xlsx                # Adjacency Matrix (Interactions across Proteins/Genes)
 ┗ 📂 COADREAD/               # Example Cohort Data Directory
   ┣ 📄 feature_name.xlsx     # Protein/Gene names (Header must be on Row 1)
   ┣ 📄 samples_protein.xlsx  # Raw Continuous Expression Matrix (Rows=Proteins, Cols=Samples)
   ┗ 📄 lables.xlsx           # Target Classification Labels (e.g., Subtypes)
```

> **Warning**  
> - **`feature_name.xlsx`**: Include headers in row 1; data starts in row 2.
> - **`samples_protein.xlsx`**: Must not contain any headers.
> - **`lables.xlsx`**: Must not contain any headers.

## 🚀 Pipeline Usage

### ⚙️ Step 1: Configuration Validation

Validate the paths and hyperparameter search spaces in `src/config.py`. Update the `BASE_DIR` parameter to strictly point to your local working directory.

### ▶️ Step 2: Execution

Launch the overarching execution pipeline. This autonomously triggers **Data Loading & Splitting $\rightarrow$ Knowledge Graph Pruning $\rightarrow$ Value Quantization $\rightarrow$ Semantic Prompt Construction $\rightarrow$ Parameter tuning using Optuna $\rightarrow$ Model Training**:

```bash
python main.py
```

## 🧠 Workflow Components & Core Mechanisms

1. **Structure-Grounded Pruning (`core.structure_grounded_pruning`)**  
   Intelligently filters out noisy signals to select highly informative features using a harmonic balance between numerical data variance and structural connectivity derived from the Knowledge Graph (`S.xlsx`).

2. **Adaptive Semantic Quantization (`core.adaptive_semantic_quantization`)**  
   Elegantly maps continuous numerical values into standardized categorical tokens (e.g., `"High"`, `"Low"`) using Non-parametric Empirical Cumulative Distribution Functions.

3. **Instruction Dataset Tuning & Hyperparameter Optimization (`engine.run_training`)**  
   Dynamically wraps the LLM training engine in an **Optuna** parameter search to iteratively determine the minimum convergence loss and optimally fine-tune using **LoRA** (Low-Rank Adaptation).

## ⚠️ Notes on LLM Execution & Hardware

This framework is natively configured to deploy `meta-llama/Meta-Llama-3-8B-Instruct`. 

- **Access Constraints**: Users **must** authenticate via the HuggingFace CLI to download the weights:
  ```bash
  huggingface-cli login
  ```
- **Hardware Prerequisites**: 
  - A CUDA-enabled GPU packing a minimum of **16GB VRAM** is recommended for default LoRA configurations.
  - For hardware-constrained environments, modify the Optuna `batch_size` bounds dynamically within `src/config.py`.

## ⚙️ Technical Requirements
- Python $\geq$ 3.8
- PyTorch $\geq$ 2.0
- `transformers`, `peft`, `datasets`, `optuna`
- `pandas`, `numpy`, `scikit-learn`, `openpyxl`, `accelerate`

---
*Built tightly for computational efficiency and modular resilience.*
