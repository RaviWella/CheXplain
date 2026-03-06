"""
Standalone Visualization Utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple
import cv2


def add_region_overlay(
    image: np.ndarray,
    regions: List[Dict],
    alpha: float = 0.2
) -> np.ndarray:
    """
    Add colored overlays for anatomical regions
    
    Args:
        image: Base image
        regions: List of regions with attention scores
        alpha: Overlay transparency
        
    Returns:
        Image with region overlays
    """
    overlay = image.copy()
    h, w = image.shape[:2]
    
    # Color map for regions
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
    ]
    
    for idx, region in enumerate(regions[:5]):
        # This would need actual region coordinates
        # For now, just annotate
        pass
    
    return overlay


def create_confidence_bar(
    predictions: Dict[str, float],
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Create horizontal bar chart of predictions
    
    Args:
        predictions: Dict of {disease: confidence}
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by confidence
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    diseases = [d for d, _ in sorted_preds]
    confidences = [c * 100 for _, c in sorted_preds]
    
    # Create bars
    colors = ['red' if c > 70 else 'orange' if c > 50 else 'yellow' for c in confidences]
    bars = ax.barh(diseases, confidences, color=colors, alpha=0.7)
    
    # Add confidence values
    for i, (disease, conf) in enumerate(sorted_preds):
        ax.text(conf * 100 + 2, i, f'{conf*100:.1f}%', va='center', fontsize=10)
    
    ax.set_xlabel('Confidence (%)', fontsize=12, fontweight='bold')
    ax.set_title('AI Predictions', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def save_visualization(
    fig: plt.Figure,
    filepath: str,
    dpi: int = 300
):
    """
    Save figure to file
    
    Args:
        fig: Matplotlib figure
        filepath: Output path
        dpi: Resolution
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f" Saved visualization to: {filepath}")


def create_heatmap_colorbar_legend() -> plt.Figure:
    """
    Create a standalone colorbar legend explaining heatmap colors
    
    Returns:
        Small figure with colorbar and explanation
    """
    fig, ax = plt.subplots(figsize=(6, 2))
    
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap='jet')
    ax.set_yticks([])
    ax.set_xticks([0, 64, 128, 192, 255])
    ax.set_xticklabels(['Low\nAttention', '', 'Medium', '', 'High\nAttention'])
    ax.set_title('Heatmap Color Guide', fontweight='bold')
    
    plt.tight_layout()
    return fig
