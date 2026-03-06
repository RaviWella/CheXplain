"""
Complete Integration Pipeline
Connects all modules into end-to-end system
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path

from .config import DEVICE, LABEL_NAMES, CONFIDENCE_THRESHOLD
from .behavior_extractor import ModelBehaviorExtractor
from .xai_enhanced import EnhancedGradCAM, XAIVisualizer
from .llm_explainer import LLMExplainer


class CheXplainPipeline:
    """
    Complete end-to-end pipeline for explainable chest X-ray diagnosis
    """
    
    def __init__(
        self,
        model,
        use_llm: bool = True,
        device: str = DEVICE
    ):
        """
        Initialize complete pipeline
        
        Args:
            model: Trained disease detection model
            use_llm: Whether to use LLM for explanations
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.use_llm = use_llm
        
        print("🔧 Initializing CheXplain Pipeline...")
        
        # Initialize components
        self.behavior_extractor = ModelBehaviorExtractor()
        self.gradcam = EnhancedGradCAM(model)
        self.visualizer = XAIVisualizer()
        
        # Initialize LLM (optional)
        if use_llm:
            try:
                print("   Loading LLM for explanations...")
                self.llm_explainer = LLMExplainer(use_quantization=True)
                print("   ✅ LLM ready")
            except Exception as e:
                print(f"   ⚠️ LLM failed to load: {e}")
                print("   → Falling back to template explanations")
                self.llm_explainer = None
        else:
            self.llm_explainer = None
        
        print("✅ Pipeline initialized successfully!\n")
    
    def process_image(
        self,
        image_path: str,
        confidence_threshold: float = CONFIDENCE_THRESHOLD
    ) -> Dict:
        """
        Complete processing of a chest X-ray image
        
        Args:
            image_path: Path to X-ray image
            confidence_threshold: Confidence threshold for positive findings
            
        Returns:
            Dict containing all results
        """
        print(f"📊 Processing: {Path(image_path).name}")
        print("=" * 60)
        
        # Step 1: Load and preprocess image
        print("1️⃣  Loading image...")
        image, image_tensor = self._load_image(image_path)
        print("   ✅ Image loaded")
        
        # Step 2: Get predictions
        print("2️⃣  Running disease detection...")
        predictions, all_probabilities = self._get_predictions(image_tensor)
        positive_findings = {
            disease: conf for disease, conf in predictions.items()
            if conf >= confidence_threshold
        }
        print(f"   ✅ Found {len(positive_findings)} positive finding(s)")
        
        # Step 3: Generate Grad-CAMs for positive findings
        print("3️⃣  Generating visual explanations (Grad-CAM)...")
        heatmaps = {}
        overlays = {}
        behaviors = {}
        
        if len(positive_findings) > 0:
            for disease, confidence in positive_findings.items():
                disease_idx = LABEL_NAMES.index(disease)
                
                # Generate heatmap
                heatmap = self.gradcam.generate_cam(
                    image_tensor, disease_idx, self.device
                )
                heatmaps[disease] = heatmap
                
                # Create overlay
                overlay = self.visualizer.create_heatmap_overlay(
                    image, heatmap, alpha=0.4
                )
                overlays[disease] = overlay
                
                # Extract behavior
                behavior = self.behavior_extractor.extract_complete_behavior(
                    {disease: confidence},
                    heatmap,
                    all_probabilities
                )
                behaviors[disease] = behavior
                
                print(f"   ✅ {disease}: Grad-CAM generated")
        else:
            print("   ℹ️  No positive findings - creating negative view")
            overlay = self.visualizer.create_negative_overlay(image)
            overlays["negative"] = overlay
        
        # Step 4: Generate text explanations
        print("4️⃣  Generating text explanations...")
        explanations = {}
        
        if len(positive_findings) > 0 and self.llm_explainer:
            for disease, confidence in positive_findings.items():
                behavior = behaviors[disease]
                
                explanations[disease] = {
                    "clinician": self.llm_explainer.generate_explanation(
                        disease, confidence, behavior, "clinician"
                    ),
                    "patient": self.llm_explainer.generate_explanation(
                        disease, confidence, behavior, "patient"
                    )
                }
                print(f"   ✅ {disease}: Explanations generated")
        elif len(positive_findings) > 0:
            # Use fallback explanations
            explanations = self._generate_fallback_explanations(
                positive_findings, behaviors
            )
            print(f"   ⚠️  Using template explanations (LLM unavailable)")
        else:
            explanations = self._generate_negative_explanations()
            print(f"   ℹ️  Generated negative finding explanation")
        
        # Step 5: Create visualizations
        print("5️⃣  Creating visualizations...")
        figures = self._create_visualizations(
            image, overlays, positive_findings, behaviors
        )
        print("   ✅ Visualizations ready")
        
        print("\n✅ Processing complete!")
        print("=" * 60)
        
        # Return complete results
        return {
            "image": image,
            "predictions": predictions,
            "positive_findings": positive_findings,
            "all_probabilities": all_probabilities,
            "heatmaps": heatmaps,
            "overlays": overlays,
            "behaviors": behaviors,
            "explanations": explanations,
            "figures": figures
        }
    
    def _load_image(self, image_path: str) -> Tuple[np.ndarray, torch.Tensor]:
        """Load and preprocess image"""
        from torchvision import transforms
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        
        # Preprocess for model
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        image_tensor = transform(image)
        
        return image_array, image_tensor
    
    def _get_predictions(
        self, image_tensor: torch.Tensor
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """Get model predictions"""
        self.model.eval()
        
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            logits = self.model(image_tensor)
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Create predictions dict
        predictions = {
            disease: float(prob)
            for disease, prob in zip(LABEL_NAMES, probabilities)
        }
        
        return predictions, probabilities
    
    def _create_visualizations(
        self,
        image: np.ndarray,
        overlays: Dict[str, np.ndarray],
        findings: Dict[str, float],
        behaviors: Dict[str, Dict]
    ) -> Dict[str, plt.Figure]:
        """Create all visualization figures"""
        figures = {}
        
        if len(findings) > 0:
            # Side-by-side for each finding
            for disease, overlay in overlays.items():
                if disease != "negative":
                    behavior = behaviors.get(disease, {})
                    bbox = behavior.get("spatial_analysis", {}).get("bounding_box")
                    regions = behavior.get("anatomical_regions", [])
                    
                    fig = self.visualizer.create_side_by_side(
                        image, overlay,
                        titles=("Original X-Ray", f"{disease} - AI Focus"),
                        bbox=bbox,
                        regions=regions
                    )
                    figures[f"side_by_side_{disease}"] = fig
            
            # Comparison grid
            overlay_dict = {
                disease: (overlay, findings[disease])
                for disease, overlay in overlays.items()
                if disease != "negative"
            }
            fig_grid = self.visualizer.create_comparison_grid(
                image, overlay_dict
            )
            figures["comparison_grid"] = fig_grid
        else:
            # Negative finding view
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            axes[0].imshow(image, cmap='gray')
            axes[0].set_title('Original X-Ray', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(overlays["negative"])
            axes[1].set_title('No Significant Findings', fontsize=14, fontweight='bold')
            axes[1].axis('off')
            
            plt.tight_layout()
            figures["negative_view"] = fig
        
        return figures
    
    def _generate_fallback_explanations(
        self, findings: Dict[str, float], behaviors: Dict[str, Dict]
    ) -> Dict[str, Dict[str, str]]:
        """Generate template-based explanations if LLM unavailable"""
        explanations = {}
        
        for disease, confidence in findings.items():
            behavior = behaviors.get(disease, {})
            decision_factors = behavior.get("decision_factors", {})
            
            # Clinician explanation
            clinician_text = (
                f"The AI model detected {disease} with {confidence*100:.1f}% confidence. "
                f"{decision_factors.get('primary_focus', 'Model focused on multiple regions')}. "
                f"{decision_factors.get('pattern_description', 'Pattern analysis performed')}. "
                f"Further clinical correlation and radiologist review is recommended."
            )
            
            # Patient explanation
            patient_text = (
                f"The AI detected signs that might indicate {disease.lower()}. "
                f"The computer is {confidence*100:.0f}% confident about this finding. "
                f"Your doctor will review these results carefully and discuss them with you. "
                f"Remember, this is an AI analysis to help your doctor - they will make the final diagnosis."
            )
            
            explanations[disease] = {
                "clinician": clinician_text,
                "patient": patient_text
            }
        
        return explanations
    
    def _generate_negative_explanations(self) -> Dict:
        """Generate explanations for negative findings"""
        return {
            "summary": {
                "clinician": (
                    "No significant pathological findings detected by the AI model "
                    "above the confidence threshold. Routine radiologist review recommended "
                    "to confirm negative findings."
                ),
                "patient": (
                    "The AI did not detect any significant issues in your chest X-ray. "
                    "Your doctor will perform a complete review to confirm this assessment."
                )
            }
        }
    
    def display_results(self, results: Dict, show_plots: bool = True):
        """
        Display results in a formatted way
        
        Args:
            results: Results dict from process_image()
            show_plots: Whether to display matplotlib figures
        """
        print("\n" + "=" * 80)
        print("📋 CHEXPLAIN ANALYSIS RESULTS")
        print("=" * 80)
        
        # Predictions summary
        print("\n🔍 PREDICTIONS:")
        print("-" * 80)
        positive = results["positive_findings"]
        if len(positive) > 0:
            for disease, conf in sorted(positive.items(), key=lambda x: x[1], reverse=True):
                print(f"  • {disease}: {conf*100:.1f}%")
        else:
            print("  No significant findings above threshold")
        
        # Explanations
        explanations = results["explanations"]
        
        if len(positive) > 0:
            print("\n" + "=" * 80)
            print("👨‍⚕️ CLINICIAN EXPLANATIONS")
            print("=" * 80)
            for disease in positive.keys():
                if disease in explanations:
                    print(f"\n{disease}:")
                    print("-" * 80)
                    print(explanations[disease]["clinician"])
            
            print("\n" + "=" * 80)
            print("🧑 PATIENT EXPLANATIONS")
            print("=" * 80)
            for disease in positive.keys():
                if disease in explanations:
                    print(f"\n{disease}:")
                    print("-" * 80)
                    print(explanations[disease]["patient"])
        else:
            print("\n" + "=" * 80)
            print("📄 SUMMARY")
            print("=" * 80)
            print("\nClinician:")
            print("-" * 80)
            print(explanations["summary"]["clinician"])
            print("\nPatient:")
            print("-" * 80)
            print(explanations["summary"]["patient"])
        
        print("\n" + "=" * 80)
        
        # Show visualizations
        if show_plots:
            for name, fig in results["figures"].items():
                plt.show()
    
    def save_results(
        self,
        results: Dict,
        output_dir: str,
        image_name: str = "result"
    ):
        """
        Save all results to directory
        
        Args:
            results: Results from process_image()
            output_dir: Output directory path
            image_name: Base name for output files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving results to: {output_path}")
        
        # Save figures
        for name, fig in results["figures"].items():
            save_path = output_path / f"{image_name}_{name}.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved: {save_path.name}")
        
        # Save explanations as text
        if len(results["positive_findings"]) > 0:
            report_path = output_path / f"{image_name}_report.txt"
            with open(report_path, 'w') as f:
                f.write("CHEXPLAIN ANALYSIS REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("DETECTED FINDINGS:\n")
                f.write("-" * 80 + "\n")
                for disease, conf in results["positive_findings"].items():
                    f.write(f"{disease}: {conf*100:.1f}%\n")
                
                f.write("\n\nCLINICIAN EXPLANATIONS:\n")
                f.write("=" * 80 + "\n")
                for disease, expl in results["explanations"].items():
                    f.write(f"\n{disease}:\n")
                    f.write("-" * 80 + "\n")
                    f.write(expl["clinician"] + "\n")
                
                f.write("\n\nPATIENT EXPLANATIONS:\n")
                f.write("=" * 80 + "\n")
                for disease, expl in results["explanations"].items():
                    f.write(f"\n{disease}:\n")
                    f.write("-" * 80 + "\n")
                    f.write(expl["patient"] + "\n")
            
            print(f"   ✅ Saved: {report_path.name}")
        
        print("✅ All results saved!\n")


# Convenience function
def create_pipeline(model, use_llm: bool = True):
    """Create a CheXplain pipeline"""
    return CheXplainPipeline(model, use_llm=use_llm)
