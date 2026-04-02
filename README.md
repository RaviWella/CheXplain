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

- **Clinician-level** — structured radiological report with anatomical analysis and differential considerations
- **Patient-level** — simple, warm, jargon-free explanation with reassuring tone

The pipeline combines three AI components:

| Component | Technology | Purpose |
|---|---|---|
| Computer Vision | DenseNet-121 (AUROC: **0.8305**) | Multi-label pathology classification |
| Explainability (XAI) | Grad-CAM + Behavior Extractor | Attention heatmaps + 7-zone anatomical region mapping |
| Natural Language | Qwen2.5-3B-Instruct | Dual-level explanation generation + follow-up chat |

> A related conference paper was accepted at **IVPAI 2026**: *"Explainable Chest X-Ray Diagnosis Using Visual Saliency and LLM-Generated Dual-Audience Clinical Narratives"*

---

## Repository Structure

```
CheXplain/
│
├── 📁 notebooks/
│   ├── 📁 01-Classification-model/      ← CV model training
│   │   ├── 📁 Hyper-parameter-tunning/  ← 36-trial random HP search per model
│   │   │   ├── hyp-t-densenet121.ipynb
│   │   │   ├── hyp-t-efficientnet-b3.ipynb
│   │   │   ├── hyp-t-resnet50.ipynb
│   │   │   └── hyp-t-vit-b16.ipynb
│   │   │
│   │   └── 📁 Model-training/           ← Final training with optimal HP config
│   │       ├── final-train-densenet121.ipynb
│   │       ├── final-train-efficientnet-b3.ipynb
│   │       ├── final-train-resnet50.ipynb
│   │       └── final-train-vit-b16.ipynb
│   │
│   ├── 📁 02-evaluation/                ← Benchmarking and evaluation
│   │   ├── full-cv-auroc-evaluation.ipynb   ← Compare all 4 CV models
│   │   ├── full-xai-evaluation.ipynb        ← Insertion/Deletion AUC for XAI methods
│   │   └── nlp-model-comparison.ipynb       ← 5-LLM composite score benchmark
│   │
│   └── 📁 03-pipeline/                  ← Full end-to-end systems
│       ├── full-pipeline-densenet-qwen.ipynb  ← Core pipeline (no UI)
│       ├── full-pipeline-chatbot.ipynb         ← Pipeline + chatbot only
│       └── full-pipeline-ui.ipynb              ← Complete system with Gradio UI START HERE
│
├── 📁 src/                              ← Python source modules
│   ├── __init__.py
│   ├── config.py                        ← Model paths, thresholds, zone definitions
│   ├── behavior_extractor.py            ← Novel 7-zone anatomical mapping module
│   ├── llm_explainer.py                 ← Dual-prompt LLM generation logic
│   ├── pipeline.py                      ← End-to-end pipeline orchestrator
│   └── visualization.py                 ← Grad-CAM overlay + heatmap rendering
│
├── 📁 docs/                             ← Project documentation
├── 📁 test-samples/                     ← Sample chest X-ray images for testing
├── 📁 models/                           ← Place downloaded .pth files here
│   └── README.md                        ← Model download instructions
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Key Results

### CV Model Benchmarking (CheXpert Official Validation — 234 images, 11 classes)

| Model | Macro AUROC | Strong-band Classes | Selected |
|---|---|---|---|
| **DenseNet-121** | **0.8305** | **9 / 11** | Yes |
| ResNet-50 | 0.8247 | 8 / 11 | |
| EfficientNet-B3 | 0.7890 | 6 / 11 | |
| ViT-B/16 | 0.7960 | 7 / 11 | |

### XAI Method Benchmarking (Faithfulness vs. Speed)

| XAI Method | Faithfulness Score | Speed | Selected |
|---|---|---|---|
| **Grad-CAM** | **0.6353** | **~487 ms** | Yes |
| Grad-CAM++ | 0.5602 | ~452 ms | |
| Occlusion Sensitivity | 0.7677 | ~56,000 ms  | |
| LIME | 0.5589 | ~102,000 ms  | |

> Grad-CAM was selected — best balance of faithfulness within the 500 ms real-time constraint.

### NLP Model Benchmarking (Composite Score)

| Model | Composite Score | Flesch Gap (Clinician vs Patient) | Selected |
|---|---|---|---|
| **Qwen2.5-3B-Instruct** | **0.587** | **63.9 pts** |  Yes |
| LLaMA-3-8B | 0.541 | 48.2 pts | |
| Phi-2 | 0.498 | 31.4 pts | |
| BioGPT-Large | 0.463 | 18.7 pts | |
| Flan-T5-Base | 0.391 | 9.1 pts | |

---

## Prerequisites

Before you begin, make sure you have:

- A free **[Kaggle](https://www.kaggle.com)** account
- **GPU enabled** on Kaggle (T4 x1 or P100) — Settings → Accelerator
- **Internet enabled** on Kaggle — Settings → Internet → On
- Basic familiarity with running Jupyter notebook cells

> All pipeline notebooks are designed to run on **Kaggle** due to GPU and storage requirements.
> Local execution requires a GPU with **16GB+ VRAM** for Qwen2.5-3B.

---

## Quick Start — Run the Full UI (Recommended)

Follow these 5 steps to have the complete CheXplain system running in under 5 minutes.

---

### Step 1 — Clone or Download This Repository

```bash
git clone https://github.com/RaviWella/CheXplain.git
```

Or click **Code → Download ZIP** on GitHub and extract it.

---

### Step 2 — Add the Trained Model Dataset on Kaggle

1. Go to: [XAI Chest X-Ray Project — Final Year (Kaggle Dataset)](https://www.kaggle.com/datasets/ravinduwellalage2/xai-chest-x-ray-project-final-year)
2. Click **"Copy & Edit"** — this attaches the dataset to a new notebook automatically
3. Or if creating a new notebook manually: go to the right panel → **Add Data** → search `ravinduwellalage2/xai-chest-x-ray-project-final-year`

This dataset contains:
- `denseNet121_v2.pth` — DenseNet-121 trained weights (80.6 MB)
- All `src/` module files pre-packaged

---

### Step 3 — Upload the UI Notebook to Kaggle

1. Go to [Kaggle](https://www.kaggle.com) → **"Create"** → **"New Notebook"**
2. Click the **three-dot menu (⋮)** → **"Import Notebook"**
3. Upload `notebooks/03-pipeline/full-pipeline-ui.ipynb` from this repo
4. Confirm settings in the right panel:
   - **Accelerator** → GPU T4 x1 (minimum) or P100 (recommended)
   - **Internet** → On
   - **Dataset** → confirm `ravinduwellalage2/xai-chest-x-ray-project-final-year` is attached

---

### Step 4 — Run the Notebook Cells in Order

Click **"Run All"** or run each cell manually in this sequence:

| Cell | What it does | Expected time |
|---|---|---|
| **Cell 1** | Installs all pip dependencies (gradio, transformers, etc.) | ~60 s |
| **Cell 2** | Verifies GPU is available and prints device info | ~5 s |
| **Cell 3** | Copies `src/` files from dataset and sets model path | ~5 s |
| **Cell 4** | Imports all core modules | ~10 s |
| **Cell 6.5** | Applies Gaussian filter compatibility patch | ~2 s |
| **Cell 6.6** | Reloads patched module | ~2 s |
| **Cell 7** | Loads DenseNet-121 checkpoint (~80 MB) | ~10 s |
| **Cell 13** | Loads Qwen2.5-3B-Instruct model (~6 GB GPU) | ~2 min |
| **Final Cell** | Launches the Gradio web interface | ~10 s |

>  **Total startup time: approximately 3–4 minutes on P100 GPU.**

---

### Step 5 — Use the Gradio Interface

Once the final cell completes, you will see output like:

```
Running on public URL: https://xxxxxxxxxxxxxxxx.gradio.live
```

1. **Click the public URL** to open the interface in your browser
2. **Upload** a chest X-ray image — use any image from the `test-samples/` folder
3. **Select your role**: Clinician or Patient
4. **Click Analyse** — the system will automatically:
   - Classify the pathology and produce confidence scores
   - Generate a Grad-CAM saliency heatmap
   - Map attention to 7 anatomical lung zones via the Behavior Extractor
   - Generate a full dual-audience AI explanation
5. **Click "Start Chat"** to ask follow-up questions grounded in that specific scan

---

## Test Samples

Ready-to-use chest X-ray images are available in the `test-samples/` folder:

```
test-samples/
├── sample-1.png
├── sample-2.jpg    ← Primary demo image (Atelectasis — 89.8% confidence)
├── sample-3.jpg
├── sample-4.jpg
├── sample-5.jpg
├── sample-6.jpg
└── 7.jpg
```

> `sample-2.jpg` is the image used in all thesis demonstrations and produces the Atelectasis case documented in Chapter 7.

---

## Running Other Notebooks

### Hyperparameter Search (Optional — already done)

> Only run these if you want to reproduce the HP search from scratch. Each run performs 36 random trials and takes ~3 hours on Kaggle GPU.

| Model | HP Search Notebook (Kaggle) |
|---|---|
| DenseNet-121 | [hyp-t-densenet121](https://www.kaggle.com/code/d1nushi/hyp-t-densenet121) |
| ResNet-50 | [hyp-t-resnet50](https://www.kaggle.com/code/d1nushi/hyp-t-resnet50) |
| EfficientNet-B3 | [hyp-t-efficientnet-b3](https://www.kaggle.com/code/d1nushi/hyp-t-efficientnet-b3) |
| ViT-B/16 | [hyp-t-vit-b16](https://www.kaggle.com/code/d1nushi/hyp-t-vit-b16) |

---

### Model Training (Reproduce final models)

> Requires the full [CheXpert dataset](https://www.kaggle.com/datasets/ashery/chexpert) (~439K images). Training takes ~5 hours per model on Kaggle GPU.

| Model | Final Training Notebook (Kaggle) | Macro-AUROC |
|---|---|---|
| ResNet-50 | [final-train-resnet50](https://www.kaggle.com/code/d1nushi/final-train-resnet50) | 0.8247 |
| DenseNet-121 | [final-train-densenet121](https://www.kaggle.com/code/d1nushi/final-train-densenet121) | 0.8305 |
| EfficientNet-B3 | [final-train-efficientnet-b3](https://www.kaggle.com/code/ravinduwellalage2/final-train-efficientnet-b3) | 0.7890 |
| ViT-B/16 | [final-train-VIT-B16](https://www.kaggle.com/code/ravinduwellalage2/final-train-vit-b16) | 0.7960 |

Steps:
1. Open the relevant notebook link above on Kaggle
2. Add the [CheXpert dataset](https://www.kaggle.com/datasets/ashery/chexpert) as a data source
3. Set **Accelerator → GPU T4** and **Internet → On**
4. Click **Run All** — best checkpoint saves automatically to `/kaggle/working/`

---

### Model Evaluation

1. Open `notebooks/02-evaluation/full-cv-auroc-evaluation.ipynb`
2. Attach the model weights dataset
3. Run all cells — generates per-class AUROC bar charts, macro-AUROC comparison, and CSV summary

---

### XAI Methods Comparison

1. Open `notebooks/02-evaluation/full-xai-evaluation.ipynb`
2. Ensure a trained model checkpoint is available
3. Run all cells — compares Grad-CAM, Grad-CAM++, LIME, and Occlusion Sensitivity on Insertion/Deletion AUC and inference speed

---

### NLP Model Comparison

1. Open `notebooks/02-evaluation/nlp-model-comparison.ipynb`
2. Run all cells — evaluates all 5 LLMs (Flan-T5, BioGPT, Phi-2, Qwen2.5-3B, LLaMA-3-8B) on the composite quality score formula

---

## Model Weights

Model weights are **not stored in this repository** due to file size.

| Model | File | Size | Download |
|---|---|---|---|
| DenseNet-121 *(pipeline model)* | `denseNet121_v2.pth` | 80.6 MB | [Kaggle Dataset](https://www.kaggle.com/datasets/ravinduwellalage2/xai-chest-x-ray-project-final-year) |
| ResNet-50 | `resnet50_final.pth` | ~94 MB | [Kaggle Dataset](https://www.kaggle.com/datasets/ravinduwellalage2/xai-chest-x-ray-project-final-year) |
| EfficientNet-B3 | `efficientnet_b3_final.pth` | ~269 MB | [Kaggle Dataset](https://www.kaggle.com/datasets/ravinduwellalage2/xai-chest-x-ray-project-final-year) |
| ViT-B/16 | `vit_b16_final.pth` | ~330 MB | [Kaggle Dataset](https://www.kaggle.com/datasets/ravinduwellalage2/xai-chest-x-ray-project-final-year) |

After downloading, place `.pth` files in the `models/` folder when running locally.

---

## Local Setup (Optional)

> Requires NVIDIA GPU with **16GB+ VRAM** for Qwen2.5-3B. CPU-only mode is extremely slow and not recommended.

```bash
# 1. Clone the repository
git clone https://github.com/RaviWella/CheXplain.git
cd CheXplain

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download model weights from Kaggle and place in models/

# 5. Launch the pipeline notebook
jupyter notebook notebooks/03-pipeline/full-pipeline-ui.ipynb
```

---

## System Architecture

```
Input: Chest X-Ray Image
        │
        ▼
┌──────────────────────────┐
│   STAGE 1 — CV           │  DenseNet-121 · 14-class sigmoid output
│   DenseNet-121           │  Macro-AUROC: 0.8305 on CheXpert validation
└───────────┬──────────────┘
            │  Top class + confidence score
            ▼
┌──────────────────────────┐
│   STAGE 2 — XAI          │  Grad-CAM hook on features.denseblock4
│   Grad-CAM               │  7×7 heatmap → upscaled to 224×224
└───────────┬──────────────┘
            │  Saliency heatmap (224×224)
            ▼
┌──────────────────────────┐
│   STAGE 3 — Behavior     │  ★ Novel contribution
│   Extractor              │  7 anatomical lung zones · threshold 0.3
│                          │  Output: ranked zone list + focal/diffuse pattern
└───────────┬──────────────┘
            │  Structured anatomical context
            ▼
┌──────────────────────────┐
│   STAGE 4 — NLP          │  Qwen2.5-3B-Instruct · Float16 · dual prompt
│   Qwen2.5-3B             │  → Clinician report (technical, radiological)
│                          │  → Patient report (plain language, empathetic)
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│   STAGE 5 — Interface    │  Gradio web UI · domain-restricted chatbot
│   Gradio + Chatbot       │  4-exchange memory · scan-specific context only
└──────────────────────────┘
```

---

## Research Publication

> Wellalage, R.V. and Athuraliya, B. (2026). *Explainable Chest X-Ray Diagnosis Using Visual Saliency and LLM-Generated Dual-Audience Clinical Narratives.* Accepted at **IVPAI 2026** (International Visual Intelligence and Perception in Artificial Intelligence Conference). *(Not yet published)*

---

## Author

**Ravindu Vihela Wellalage**
BSc (Hons) Artificial Intelligence & Data Science
Robert Gordon University, Aberdeen, UK | Informatics Institute of Technology, Sri Lanka
Student ID: 2313077

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
<i>Built with ❤️ for making AI in healthcare more transparent and explainable.</i>
</div>
