"""
Enhanced Configuration for V2 Architecture
"""

import torch
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ============================================================================
# DISEASE LABELS (14 CheXpert/ChestX-ray14 diseases)
# ============================================================================
LABEL_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax"
]

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
ACTIVE_MODEL = "resnet"  # or "densenet"

MODEL_CONFIGS = {
    "resnet": {
        "architecture": "resnet50",
        "num_classes": 14,
        "pretrained": True,
        "checkpoint": MODEL_DIR / "efficientNet.pth"  # Your trained model
    },
    "densenet": {
        "architecture": "densenet121",
        "num_classes": 14,
        "pretrained": True,
        "checkpoint": MODEL_DIR / "densenet121_chexpert.pth"
    }
}

# ============================================================================
# GRAD-CAM CONFIGURATION
# ============================================================================
GRADCAM_LAYER = {
    "resnet": "backbone.layer4",
    "densenet": "backbone.features.denseblock4"
}

# ============================================================================
# PREDICTION THRESHOLDS
# ============================================================================
CONFIDENCE_THRESHOLD = 0.5  # Diseases above this are reported
LOW_CONFIDENCE = 0.3        # Below this = likely negative
HIGH_CONFIDENCE = 0.7       # Above this = high certainty

# ============================================================================
# LLM CONFIGURATION
# ============================================================================
# LLM Configuration - FLAN-T5 for Local Development
LLM_CONFIG = {
    "model_name": "google/flan-t5-large",  # CPU-friendly, 780MB
    
    # CPU Configuration
    "quantization": False,
    "load_in_4bit": False,
    "device_map": "cpu",
    
    # Generation Parameters (optimized for medical text)
    "generation_params": {
        "max_new_tokens": 300,
        "temperature": 0.8,
        "top_p": 0.95,
        "do_sample": True,
        "num_return_sequences": 1,
        "repetition_penalty": 1.2,
        "length_penalty": 1.0
    }
}


# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================
VISUALIZATION = {
    "colormap": "jet",  # For heatmaps
    "alpha": 0.4,       # Overlay transparency
    "negative_color": "cool",  # Blue for no findings
    "figsize": (14, 7)
}

# ============================================================================
# ANATOMICAL REGIONS (for spatial analysis)
# ============================================================================
LUNG_REGIONS = {
    "upper_right": {"x": [0.5, 1.0], "y": [0.0, 0.33]},
    "middle_right": {"x": [0.5, 1.0], "y": [0.33, 0.66]},
    "lower_right": {"x": [0.5, 1.0], "y": [0.66, 1.0]},
    "upper_left": {"x": [0.0, 0.5], "y": [0.0, 0.33]},
    "middle_left": {"x": [0.0, 0.5], "y": [0.33, 0.66]},
    "lower_left": {"x": [0.0, 0.5], "y": [0.66, 1.0]},
    "cardiac": {"x": [0.3, 0.7], "y": [0.4, 0.8]},
}

# ============================================================================
# PROMPT TEMPLATES
# ============================================================================
CLINICIAN_PROMPT_TEMPLATE = """You are a radiologist AI assistant providing technical explanations.

**X-Ray Analysis:**
- Primary Finding: {disease}
- Confidence: {confidence}%
- Spatial Focus: {regions}
- Key Features Detected: {features}
- Model Attention: {attention_summary}

Generate a concise clinician-level explanation (150-200 words) that:
1. Describes the radiological findings
2. Explains what features the AI detected
3. Notes the anatomical regions of interest
4. Provides differential diagnosis considerations
5. Includes confidence interpretation

Use technical medical terminology."""

PATIENT_PROMPT_TEMPLATE = """You are a medical AI explaining X-ray results to patients in simple, reassuring language.

**What the AI Found:**
- Finding: {disease}
- How confident: {confidence}%
- Where it looked: {regions}
- What it saw: {features_simplified}

Generate a patient-friendly explanation (120-150 words) that:
1. Explains what was found in simple terms
2. Uses analogies or comparisons to everyday things
3. Reassures and provides context
4. Explains next steps
5. Avoids medical jargon

Write at an 8th-grade reading level. Be warm and supportive."""

# ============================================================================
# MEDICAL DOMAIN KNOWLEDGE
# ============================================================================
DISEASE_INFO = {
    "Atelectasis": {
        "description": "Partial lung collapse",
        "severity": "moderate",
        "common_causes": ["mucus plug", "tumor", "post-surgery"],
        "typical_location": ["lower lobes"]
    },
    "Cardiomegaly": {
        "description": "Enlarged heart",
        "severity": "moderate",
        "common_causes": ["hypertension", "heart failure"],
        "typical_location": ["cardiac silhouette"]
    },
    "Pneumonia": {
        "description": "Lung infection with consolidation",
        "severity": "high",
        "common_causes": ["bacterial", "viral", "aspiration"],
        "typical_location": ["any lobe"]
    },
    # Add more as needed...
}

print("Configuration loaded successfully!")
