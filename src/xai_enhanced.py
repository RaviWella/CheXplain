"""
Enhanced XAI Module - Improved Grad-CAM with Advanced Visualizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import gaussian_filter


from .config import GRADCAM_LAYER, VISUALIZATION, DEVICE


class EnhancedGradCAM:
    """
    Enhanced Grad-CAM with better gradient capture and visualization
    """
    
    def __init__(self, model: nn.Module, target_layer: str = None):
        """
        Initialize Enhanced Grad-CAM
        
        Args:
            model: PyTorch model
            target_layer: Layer name to extract gradients from
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Get target layer
        target_module = self._get_target_layer()
        
        if target_module is not None:
            target_module.register_forward_hook(forward_hook)
            target_module.register_full_backward_hook(backward_hook)
            print(f" Hooks registered on: {self.target_layer}")
        else:
            print(f" Warning: Could not find layer {self.target_layer}")
    
    def _get_target_layer(self):
        """Get the target layer module"""
        if self.target_layer is None:
            return None
        
        # Parse layer path (e.g., "backbone.layer4")
        parts = self.target_layer.split('.')
        module = self.model
        
        try:
            for part in parts:
                module = getattr(module, part)
            return module
        except AttributeError:
            return None
    
    def generate_cam(
        self,
        image_tensor: torch.Tensor,
        target_class: int,
        device: str = DEVICE
    ) -> np.ndarray:
        """
        Generate Class Activation Map
        
        Args:
            image_tensor: Input image tensor (C, H, W) or (1, C, H, W)
            target_class: Target class index
            device: Device to run on
            
        Returns:
            Heatmap as numpy array (H, W)
        """
        self.model.eval()
        
        # Ensure batch dimension
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        image_tensor = image_tensor.to(device)
        image_tensor.requires_grad = True
        
        # Forward pass
        logits = self.model(image_tensor)
        
        # Get target score
        target_score = logits[0, target_class]
        
        # Backward pass
        self.model.zero_grad()
        target_score.backward(retain_graph=True)
        
        # Check if gradients were captured
        if self.activations is None or self.gradients is None:
            print(f" Warning: No gradients captured for class {target_class}")
            return np.zeros((7, 7))
        
        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        
        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Convert to numpy
        cam = cam.cpu().detach().numpy()

        # Normalize to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            print(f"   Warning: Flat heatmap for class {target_class}")

        # ADD SMOOTHING HERE
        from scipy.ndimage import gaussian_filter
        cam = gaussian_filter(cam, sigma=2)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # Re-normalize

        return cam

    

    
    def generate_multiple_cams(
        self,
        image_tensor: torch.Tensor,
        target_classes: List[int],
        device: str = DEVICE
    ) -> Dict[int, np.ndarray]:
        """
        Generate CAMs for multiple classes
        
        Returns:
            Dict mapping class_idx -> heatmap
        """
        cams = {}
        for class_idx in target_classes:
            cam = self.generate_cam(image_tensor, class_idx, device)
            cams[class_idx] = cam
        return cams


    def generate_cam_verbose(self, input_tensor, target_class, device):
        """Generate CAM with real-time verbose output"""
        print("\nGRAD-CAM GENERATION")
        print(f"   Target layer: {self.target_layer}")
        print(f"   Target class: {target_class}")
        
        # Show actual computation
        print("\n   Computing gradients...")
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        print(f"   • Logits: {output[0, target_class].item():.3f}")
        
        # Backward pass
        output[0, target_class].backward()
        
        # Show actual shapes
        print(f"   • Activations shape: {self.activations.shape}")
        print(f"   • Gradients shape: {self.gradients.shape}")
        
        # Show actual weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        print(f"   • Max channel weight: {weights.max().item():.3f}")
        print(f"   • Min channel weight: {weights.min().item():.3f}")
        
        # Generate CAM
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam / cam.max()

        # ADD SMOOTHING HERE
        cam_np = cam.cpu().numpy()
        from scipy.ndimage import gaussian_filter
        cam_np = gaussian_filter(cam_np, sigma=2)
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)

        # Show heatmap stats (after smoothing)
        print(f"\n   Heatmap statistics:")
        print(f"   • Peak: {cam_np.max():.3f}")
        print(f"   • Mean: {cam_np.mean():.3f}")
        print(f"   • Coverage (>0.5): {(cam_np > 0.5).sum() / cam_np.size * 100:.1f}%")

        return cam_np





class XAIVisualizer:
    """
    Advanced visualization for XAI results
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize visualizer
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VISUALIZATION
    
    def create_heatmap_overlay(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: str = 'jet'
    ) -> np.ndarray:
        """
        Create heatmap overlay on original image
        
        Args:
            image: Original image (H, W, 3) RGB, uint8
            heatmap: Heatmap (H, W) normalized [0, 1]
            alpha: Overlay transparency
            colormap: Matplotlib colormap name
            
        Returns:
            Overlaid image (H, W, 3) RGB, uint8
        """
        # Resize heatmap to match image
        h, w = image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend with original
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlay.astype(np.uint8)
    
    def create_negative_overlay(
        self,
        image: np.ndarray,
        alpha: float = 0.3
    ) -> np.ndarray:
        """
        Create BLUE overlay for negative findings
        
        Args:
            image: Original image (H, W, 3) RGB
            alpha: Overlay transparency
            
        Returns:
            Blue-tinted image indicating normal/negative
        """
        h, w = image.shape[:2]
        
        # Create blue overlay
        blue_overlay = np.zeros_like(image)
        blue_overlay[:, :, 2] = 200  # Blue channel
        
        # Blend
        result = cv2.addWeighted(image, 1 - alpha, blue_overlay, alpha, 0)
        
        return result.astype(np.uint8)
    
    def create_side_by_side(
        self,
        original: np.ndarray,
        heatmap_overlay: np.ndarray,
        titles: Tuple[str, str] = ("Original X-Ray", "AI Focus Areas"),
        bbox: Optional[Dict] = None,
        regions: Optional[List[Dict]] = None
    ) -> plt.Figure:
        """
        Create side-by-side comparison view
        
        Args:
            original: Original image
            heatmap_overlay: Heatmap overlay image
            titles: Tuple of (left_title, right_title)
            bbox: Bounding box dict with keys: top, left, bottom, right
            regions: List of affected regions with attention scores
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=self.config['figsize'])
        
        # Left: Original
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title(titles[0], fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Right: Heatmap overlay
        axes[1].imshow(heatmap_overlay)
        axes[1].set_title(titles[1], fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Add bounding box if provided
        if bbox is not None:
            rect = patches.Rectangle(
                (bbox['left'], bbox['top']),
                bbox['right'] - bbox['left'],
                bbox['bottom'] - bbox['top'],
                linewidth=2,
                edgecolor='lime',
                facecolor='none',
                linestyle='--'
            )
            axes[1].add_patch(rect)
        
        # Add region annotations if provided
        if regions is not None:
            info_text = "Top Focus Regions:\n"
            for i, region in enumerate(regions[:3], 1):
                name = region['name'].replace('_', ' ').title()
                score = region['attention_score']
                info_text += f"{i}. {name} ({score*100:.1f}%)\n"
            
            axes[1].text(
                0.02, 0.98,
                info_text,
                transform=axes[1].transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        
        plt.tight_layout()
        return fig
    
    def create_comparison_grid(
        self,
        original: np.ndarray,
        findings: Dict[str, Tuple[np.ndarray, float]],
        max_cols: int = 2
    ) -> plt.Figure:
        """
        Create grid showing original + multiple findings
        
        Args:
            original: Original X-ray
            findings: Dict of {disease_name: (overlay_image, confidence)}
            max_cols: Maximum columns in grid
            
        Returns:
            Matplotlib figure
        """
        n_findings = len(findings)
        n_cols = min(max_cols, n_findings + 1)  # +1 for original
        n_rows = (n_findings + 1 + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 7*n_rows))
        axes = np.array(axes).flatten() if n_findings > 0 else [axes]
        
        # First subplot: Original
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original X-Ray', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Remaining subplots: Findings
        for idx, (disease, (overlay, confidence)) in enumerate(findings.items(), start=1):
            axes[idx].imshow(overlay)
            axes[idx].set_title(
                f'{disease}\nConfidence: {confidence*100:.1f}%',
                fontsize=12,
                fontweight='bold'
            )
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(n_findings + 1, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_interactive_view(
        self,
        original: np.ndarray,
        heatmap: np.ndarray,
        predictions: Dict[str, float],
        behavior: Dict,
        disease_name: str
    ) -> plt.Figure:
        """
        Create comprehensive interactive view with all information
        
        Args:
            original: Original X-ray
            heatmap: Raw heatmap
            predictions: Disease predictions
            behavior: Model behavior dict from behavior_extractor
            disease_name: Name of detected disease
            
        Returns:
            Rich matplotlib figure with multiple panels
        """
        fig = plt.figure(figsize=(18, 10))
        
        # Create grid
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Top left: Original
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(original, cmap='gray')
        ax1.set_title('Original X-Ray', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Top middle: Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        im = ax2.imshow(heatmap, cmap='jet')
        ax2.set_title('Attention Heatmap', fontsize=14, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        
        # Top right: Overlay
        ax3 = fig.add_subplot(gs[0, 2])
        overlay = self.create_heatmap_overlay(original, heatmap)
        ax3.imshow(overlay)
        ax3.set_title('Focus Overlay', fontsize=14, fontweight='bold')
        ax3.axis('off')
        
        # Add bounding box if available
        if 'spatial_analysis' in behavior and behavior['spatial_analysis']['bounding_box']:
            bbox = behavior['spatial_analysis']['bounding_box']
            h, w = original.shape[:2]
            # Scale bbox to image size
            rect = patches.Rectangle(
                (bbox['left'] * w / heatmap.shape[1], bbox['top'] * h / heatmap.shape[0]),
                (bbox['right'] - bbox['left']) * w / heatmap.shape[1],
                (bbox['bottom'] - bbox['top']) * h / heatmap.shape[0],
                linewidth=3,
                edgecolor='lime',
                facecolor='none',
                linestyle='--'
            )
            ax3.add_patch(rect)
        
        # Bottom: Model Behavior Summary
        ax4 = fig.add_subplot(gs[1, :])
        ax4.axis('off')
        
        # Build summary text
        summary_text = f" Model Behavior Analysis for: {disease_name}\n\n"
        
        # Confidence
        conf = predictions.get(disease_name, 0)
        summary_text += f"Confidence: {conf*100:.1f}%\n\n"
        
        # Spatial analysis
        if 'spatial_analysis' in behavior:
            spatial = behavior['spatial_analysis']
            summary_text += f" Spatial Focus:\n"
            summary_text += f"  • Peak location: ({spatial['peak_location']['x']}, {spatial['peak_location']['y']})\n"
            summary_text += f"  • Attention coverage: {spatial['attention_percentage']:.1f}%\n"
            summary_text += f"  • Pattern: {'Focal' if spatial['is_focal'] else 'Diffuse'}\n\n"
        
        # Anatomical regions
        if 'anatomical_regions' in behavior and behavior['anatomical_regions']:
            regions = behavior['anatomical_regions'][:3]
            summary_text += f" Affected Regions:\n"
            for i, region in enumerate(regions, 1):
                name = region['name'].replace('_', ' ').title()
                score = region['attention_score']
                summary_text += f"  {i}. {name}: {score*100:.1f}% attention\n"
            summary_text += "\n"
        
        # Decision factors
        if 'decision_factors' in behavior:
            factors = behavior['decision_factors']
            summary_text += f" Decision Factors:\n"
            summary_text += f"  • {factors.get('primary_focus', 'N/A')}\n"
            summary_text += f"  • {factors.get('pattern_description', 'N/A')}\n"
            summary_text += f"  • {factors.get('certainty_level', 'N/A')}\n"
        
        ax4.text(
            0.05, 0.95,
            summary_text,
            transform=ax4.transAxes,
            fontsize=11,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
        )
        
        return fig


# Convenience functions
def create_overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Quick overlay creation"""
    visualizer = XAIVisualizer()
    return visualizer.create_heatmap_overlay(image, heatmap, alpha)


def create_negative_view(image: np.ndarray) -> np.ndarray:
    """Quick blue overlay for negatives"""
    visualizer = XAIVisualizer()
    return visualizer.create_negative_overlay(image)


def visualize_side_by_side(original: np.ndarray, overlay: np.ndarray, bbox: Dict = None):
    """Quick side-by-side visualization"""
    visualizer = XAIVisualizer()
    fig = visualizer.create_side_by_side(original, overlay, bbox=bbox)
    plt.show()
    return fig
