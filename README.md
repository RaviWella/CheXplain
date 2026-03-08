<div align="center">

# CheXplain
### Explainable AI for Chest X-Ray Diagnostics

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange?style=flat-square)](https://gradio.app/)
[![Kaggle](https://img.shields.io/badge/Platform-Kaggle-20BEFF?style=flat-square&logo=kaggle)](https://www.kaggle.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

*A dual-level explainable AI system that diagnoses chest X-ray pathologies and generates clinical and patient-friendly explanations using DenseNet-121, Grad-CAM, and Qwen2.5-3B.*

</div>

---

## Overview

**CheXplain** is a final year research project for BSc (Hons) Artificial Intelligence & Data Science at Robert Gordon University, Aberdeen. The system detects 14 chest pathologies from X-ray images and provides two levels of AI-generated explanation:

- **Clinician-level** - structured radiological report with anatomical analysis
- **Patient-level** - simple, warm, jargon-free explanation

The pipeline combines three AI components:

| Component | Technology | Purpose |
|---|---|---|
| Computer Vision | DenseNet-121 (AUROC: 0.8247) | Multi-label pathology classification |
| Explainability (XAI) | Grad-CAM + Behavior Extractor | Attention heatmaps + anatomical region mapping |
| Natural Language | Qwen2.5-3B-Instruct | Dual-level explanation generation |

---

## Repository Structure

```
CheXplain/
│
├── 📁 notebooks/
│   ├── 📁 01-model-training/        ← Train CV models from scratch
│   │   ├── chexpert-densenet.ipynb
│   │   ├── chexpert-resnet50.ipynb
│   │   ├── efficientnet-b3.ipynb
│   │   └── vit-b16.ipynb
│   │
│   ├── 📁 02-evaluation/            ← Compare and evaluate models
│   │   ├── cv-auroc-evaluation.ipynb
│   │   ├── full-xai-evaluation.ipynb
│   │   └── nlp-model-comparison.ipynb
│   │
│   └── 📁 03-pipeline/              ← Full end-to-end systems
│       ├── full-pipeline-densenet-qwen.ipynb   ← Core pipeline
│       ├── full-pipeline-chatbot.ipynb          ← Pipeline + chatbot
│       └── full-pipeline-ui.ipynb               ← Pipeline + Gradio UI 
│
├── 📁 src_v2/                       ← Python source modules
│   ├── config.py
│   ├── xai_enhanced.py
│   ├── behavior_extractor.py
│   ├── llm_explainer.py
│   ├── prompt_utils.py
│   ├── pipeline.py
│   ├── verbose_pipeline.py
│   └── visualization.py
│
├── 📁 test-samples/                 ← Sample chest X-ray images for testing
├── 📁 models/                       ← Model weights (download separately)
│   └── README.md                   ← Download instructions
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Key Results

| Model | Macro AUROC |
|---|---|
| **DenseNet-121** | **0.8247** |
| ResNet-50 | 0.8027 |
| EfficientNet-B3 | 0.7927 |
| ViT-B/16 | ~0.79 |

| XAI Method | Faithfulness Score | Speed |
|---|---|---|
| Grad-CAM | 0.6353 | ~487ms |
| Grad-CAM++ | 0.5602 | ~452ms |
| LIME | 0.5589 | ~102,000ms |
| Occlusion Sensitivity | 0.7677 | ~56,000ms |

> Grad-CAM was selected for the pipeline - best balance of faithfulness and real-time speed.

---

## Prerequisites

- A **Kaggle** account (free) with **GPU T4 or P100 enabled**
- The **CheXplain Kaggle Dataset** (contains `src_v2/` + model weights)
- Basic familiarity with running Jupyter notebooks

> ⚠️ All pipeline notebooks are designed to run on **Kaggle** due to GPU and storage requirements. Local execution is possible only if you have 16GB+ VRAM.

---

## Quick Start - Running the Full UI (Recommended)

This is the fastest way to see the complete CheXplain system working.

### Step 1 - Download the Trained Model Dataset

1. Go to the Kaggle dataset: [XAI Chest X-Ray Project – Final Year](https://www.kaggle.com/datasets/ravinduwellalage2/xai-chest-x-ray-project-final-year)
2. Click **"Copy & Edit"** or add it to your Kaggle notebook as a dataset

### Step 2 - Open the UI Notebook on Kaggle

1. Go to [Kaggle](https://www.kaggle.com) → **"Create"** → **"New Notebook"**
2. Upload `notebooks/03-pipeline/full-pipeline-ui.ipynb` from this repo
3. Or open it directly if already on Kaggle

### Step 3 - Configure the Kaggle Environment

In your Kaggle notebook settings (right panel):
- **Accelerator** → GPU T4 x1 (or P100)
- **Internet** → On
- **Dataset** → Add `ravinduwellalage2/xai-chest-x-ray-project-final-year`

### Step 4 - Run the Notebook (Clean Version)

Run these cells **in order**:

| Cell | Action | What it does |
|---|---|---|
| **Cell 1** | Run | Installs all dependencies |
| **Cell 2** | Run | Verifies GPU is available |
| **Cell 3** | Run | Copies `src/` files and sets model path |
| **Cell 4** | Run | Imports core modules |
| **Cell 6.5** | Run | Patches Gaussian filter compatibility fix |
| **Cell 6.6** | Run | Reloads the patched module |
| **Cell 7** | Run | Loads DenseNet-121 (~80MB, takes ~10s) |
| **Cell 13** | Run | Loads Qwen2.5-3B (~6GB GPU, takes ~2min) |
| **Final Cell** | Run | Launches the **Gradio UI** |

> **Total startup time:** approximately 3–4 minutes on P100 GPU.

### Step 5 - Use the Gradio Interface

Once the final cell runs, you will see a **public URL** in the output like:
```
Running on public URL: https://xxxxxxxx.gradio.live
```

1. Click the public URL to open the UI in your browser
2. **Upload** any chest X-ray image (use samples from `test-samples/`)
3. Select your role: **Clinician** or **Patient**
4. The system will automatically:
   - Diagnose the pathology
   - Generate a Grad-CAM heatmap
   - Map attention to anatomical lung regions
   - Generate a full AI explanation
5. Click **"Start Chat"** to ask follow-up questions about the scan

---

## 🧪 Test Samples

Ready-to-use chest X-ray images are in the `test-samples/` folder:

```
test-samples/
├── sample-1.png
├── sample-2.jpg    ← Used in all pipeline demos (Atelectasis 89.8%)
├── sample-3.jpg
├── sample-4.jpg
├── sample-5.jpg
├── sample-6.jpg
└── 7.jpg
```

---

## Running Other Notebooks

### Model Training (Reproducing from scratch)

> ⚠️ Requires the full **CheXpert dataset** (~439K images). Training takes ~5 hours per model on Kaggle GPU.

1. Add the [CheXpert dataset](https://www.kaggle.com/datasets/ashery/chexpert) to your Kaggle notebook
2. Open the relevant training notebook from `notebooks/01-model-training/`
3. Run all cells sequentially
4. Best checkpoint saves automatically to `/kaggle/working/`

### Model Evaluation

1. Open `notebooks/02-evaluation/cv-auroc-evaluation.ipynb`
2. Update the checkpoint paths to point to your saved `.pth` files
3. Run all cells - generates AUROC curves, comparison bar charts, and CSV results

### XAI Methods Comparison

1. Open `notebooks/02-evaluation/full-xai-evaluation.ipynb`
2. Ensure a trained model checkpoint is available
3. Run all cells - compares Grad-CAM, Grad-CAM++, LIME, and Occlusion Sensitivity

### NLP Model Comparison

1. Open `notebooks/02-evaluation/nlp-model-comparison.ipynb`
2. Run all cells - evaluates Flan-T5, BioGPT, Phi-2, Qwen2, and LLaMA-3 on readability, speed, and medical accuracy

---

## Model Weights

Model weights are **not stored in this repository** due to file size constraints.

| Model | Size | Download |
|---|---|---|
| `denseNet121_v2.pth` | 80.6 MB | [Kaggle Dataset](https://www.kaggle.com/datasets/ravinduwellalage2/xai-chest-x-ray-project-final-year) |
| `efficientNet.pth` | 269.8 MB | [Kaggle Dataset](https://www.kaggle.com/datasets/ravinduwellalage2/xai-chest-x-ray-project-final-year) |

After downloading, place `.pth` files in the `models/` folder if running locally.

---

## Local Setup (Optional)

> ⚠️ Requires NVIDIA GPU with 16GB+ VRAM for Qwen2.5-3B. CPU-only mode will be extremely slow.

```bash
# 1. Clone the repository
git clone https://github.com/RaviWella/CheXplain.git
cd CheXplain

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download model weights from Kaggle and place in models/

# 5. Run a notebook locally
jupyter notebook notebooks/03-pipeline/full-pipeline-ui.ipynb
```

---

## System Architecture

```
Input X-Ray Image
       │
       ▼
┌─────────────────────┐
│   DenseNet-121      │  ← Multi-label classification (14 pathologies)
│   AUROC: 0.8247     │
└────────┬────────────┘
         │  Diagnosis + Probabilities
         ▼
┌─────────────────────┐
│   Grad-CAM XAI      │  ← Attention heatmap generation
│   Behavior          │  ← Anatomical region mapping (6 lung zones)
│   Extractor         │
└────────┬────────────┘
         │  Heatmap + Region Scores + Spatial Analysis
         ▼
┌─────────────────────┐
│   Qwen2.5-3B        │  ← Dual-level explanation generation
│   Instruct          │     -  Clinician: structured radiology report
│                     │     -  Patient: warm, simple explanation
└─────────────────────┘
         │
         ▼
   Gradio UI + Chat
```

---

## Research Paper

This project is documented in the accompanying research paper:

> **Wellalage, R.** (2026). *CheXplain: Explainable Chest X-ray Diagnosis with Saliency and LLM Narratives.* BSc Final Year Project, Robert Gordon University, Aberdeen.

---

## Author

**Ravindu Wellalage**
BSc (Hons) Artificial Intelligence & Data Science
Robert Gordon University, Aberdeen, UK
Student ID: 2313077

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
<i>Built with ❤️ for making AI in healthcare more transparent and explainable.</i>
</div>
