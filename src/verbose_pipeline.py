"""
Verbose Pipeline - Shows real-time processing steps
"""
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .config import *
from .xai_enhanced import EnhancedGradCAM
from .behavior_extractor import ModelBehaviorExtractor
from .llm_explainer import LLMExplainer
import matplotlib.pyplot as plt
import cv2


class VerbosePipeline:
    """
    Complete pipeline with verbose real-time output
    Shows actual computations, not fake progress
    """
    
    def __init__(self, model, use_llm=True, device='cpu'):
        print("🔧 Initializing Verbose Pipeline...")
        
        self.model = model
        self.device = device
        self.use_llm = use_llm
        
        # Initialize components
        self.gradcam = EnhancedGradCAM(model, target_layer="backbone.layer4")
        self.behavior_extractor = ModelBehaviorExtractor()
        
        if use_llm:
            self.llm_explainer = LLMExplainer(LLM_CONFIG)
        else:
            self.llm_explainer = None
        
        print("Verbose Pipeline Ready...\n")
    
    def process_image_verbose(self, image_path, confidence_threshold=0.5):
        """
        Process image with complete verbose output
        Shows REAL values and computations
        """
        results = {}
        
        print("="*70)
        print("PROCESSING X-RAY (VERBOSE MODE)")
        print("="*70)
        
        # ═══════════════════════════════════════════════════════════
        # STEP 1: IMAGE LOADING
        # ═══════════════════════════════════════════════════════════
        print("\nIMAGE LOADING")
        print(f"   • File: {image_path}")
        
        img = Image.open(image_path)
        original_size = img.size
        print(f"   • Original size: {original_size}")
        print(f"   • Mode: {img.mode}")
        
        # Convert grayscale to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            print(f"   • Converted to RGB")
        
        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        print(f"   • Tensor shape: {img_tensor.shape}")
        print(f"   ✓ Image loaded and preprocessed")
        
        results['image'] = img
        
        # ═══════════════════════════════════════════════════════════
        # STEP 2: MODEL INFERENCE
        # ═══════════════════════════════════════════════════════════
        print("\n" + "─"*70)
        print("\nMODEL INFERENCE")
        print(f"   • Architecture: {ACTIVE_MODEL}")
        print(f"   • Device: {self.device}")
        print(f"   • Forward pass...")
        
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Show RAW logits (top 5)
        print(f"\n   RAW LOGITS (top 5):")
        top_indices = np.argsort(probs)[::-1][:5]
        for idx in top_indices:
            disease = LABEL_NAMES[idx]
            logit_val = logits[0, idx].item()
            print(f"   • {disease:20s}: {logit_val:7.3f}")
        
        # Show probabilities
        print(f"\n   SIGMOID PROBABILITIES:")
        positive_count = 0
        for idx in top_indices:
            disease = LABEL_NAMES[idx]
            conf = probs[idx] * 100
            status = "✓" if conf > confidence_threshold * 100 else "✗"
            if conf > confidence_threshold * 100:
                positive_count += 1
            print(f"   • {disease:20s}: {conf:5.1f}%  [{status}]")
        
        print(f"\n   ✓ Detected {positive_count} positive findings (threshold: {confidence_threshold*100:.0f}%)")
        
        results['predictions'] = {LABEL_NAMES[i]: float(probs[i]) for i in range(len(LABEL_NAMES))}
        results['probabilities'] = probs
        
        # ═══════════════════════════════════════════════════════════
        # STEP 3: GRAD-CAM GENERATION (VERBOSE)
        # ═══════════════════════════════════════════════════════════
        print("\n" + "─"*70)
        
        gradcams = {}
        behaviors = {}
        explanations = {}
        
        for idx in top_indices:
            if probs[idx] > confidence_threshold:
                disease = LABEL_NAMES[idx]
                
                # Use VERBOSE Grad-CAM method
                heatmap = self.gradcam.generate_cam_verbose(
                    img_tensor, idx, self.device
                )
                gradcams[disease] = heatmap
                
                # ═══════════════════════════════════════════════════════════
                # STEP 4: BEHAVIOR EXTRACTION (VERBOSE)
                # ═══════════════════════════════════════════════════════════
                print("\n" + "─"*70)
                
                behavior = self.behavior_extractor.extract_complete_behavior_verbose(
                    predictions={disease: float(probs[idx])},
                    heatmap=heatmap,
                    all_probabilities=probs
                )
                behaviors[disease] = behavior
                
                # ═══════════════════════════════════════════════════════════
                # STEP 5: LLM EXPLANATION (VERBOSE)
                # ═══════════════════════════════════════════════════════════
                if self.llm_explainer:
                    print("\n" + "─"*70)
                    
                    # Clinician
                    clinician_exp = self.llm_explainer.generate_explanation_verbose(
                        disease=disease,
                        confidence=float(probs[idx]),
                        behavior=behavior,
                        layer="clinician"
                    )
                    
                    # Patient
                    patient_exp = self.llm_explainer.generate_explanation_verbose(
                        disease=disease,
                        confidence=float(probs[idx]),
                        behavior=behavior,
                        layer="patient"
                    )
                    
                    explanations[disease] = {
                        'clinician': clinician_exp,
                        'patient': patient_exp
                    }
        
        results['gradcams'] = gradcams
        results['behaviors'] = behaviors
        results['explanations'] = explanations
        
        print("\n" + "="*70)
        print("PROCESSING COMPLETE")
        print("="*70)
        
        return results
    

    def display_results(self, results, show_plots=True):
        """
        Display the results from verbose processing
        
        Args:
            results: Dict from process_image_verbose()
            show_plots: Whether to show matplotlib visualizations
        """
        
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        
        # PREDICTIONS SUMMARY
        print("\nDETECTED FINDINGS:")
        predictions = results.get('predictions', {})
        
        for disease, conf in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
            if conf > 0.5:
                status = "✓ POSITIVE"
                print(f"   {status:12s} | {disease:20s} | {conf*100:5.1f}%")
        
        # BEHAVIOR SUMMARY
        if 'behaviors' in results and results['behaviors']:
            print("\nMODEL BEHAVIOR:")
            for disease, behavior in results['behaviors'].items():
                print(f"\n   Disease: {disease}")
                
                # Spatial info
                spatial = behavior.get('spatial_analysis', {})
                print(f"   • Focus: {spatial.get('attention_percentage', 0):.1f}% of area")
                print(f"   • Pattern: {'Focal' if spatial.get('is_focal') else 'Diffuse'}")
                
                # Anatomical regions
                regions = behavior.get('anatomical_regions', [])
                if regions:
                    top_region = regions[0]['name'].replace('_', ' ').title()
                    print(f"   • Primary region: {top_region}")
        
        # EXPLANATIONS
        if 'explanations' in results and results['explanations']:
            print("\nGENERATED EXPLANATIONS:")
            for disease, exps in results['explanations'].items():
                print(f"\n   === {disease} ===")
                
                print("\n   CLINICIAN LEVEL:")
                print(f"   {exps['clinician'][:150]}...")
                
                print("\n   PATIENT LEVEL:")
                print(f"   {exps['patient'][:150]}...")
        
        # VISUALIZATIONS
        if show_plots and 'gradcams' in results and results['gradcams']:
            print("\nGenerating visualizations...")
            
            num_findings = len(results['gradcams'])
            fig, axes = plt.subplots(num_findings, 3, figsize=(15, 5*num_findings))
            
            if num_findings == 1:
                axes = axes.reshape(1, -1)
            
            for idx, (disease, heatmap) in enumerate(results['gradcams'].items()):
                # Original image
                img = results['image']
                axes[idx, 0].imshow(img, cmap='gray')
                axes[idx, 0].set_title(f'{disease}\nOriginal', fontsize=12, fontweight='bold')
                axes[idx, 0].axis('off')
                
                # Heatmap
                axes[idx, 1].imshow(heatmap, cmap='jet')
                axes[idx, 1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
                axes[idx, 1].axis('off')
                
                # Overlay WITH BOUNDING BOXES
                img_array = np.array(img.resize((224, 224)))
                if len(img_array.shape) == 2:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

                heatmap_resized = cv2.resize(heatmap, (224, 224))

                # EXTRACT BOUNDING BOXES FROM HEATMAP
                threshold = 0.6  # Attention threshold
                binary_mask = (heatmap_resized > threshold).astype(np.uint8)

                # Find contours
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Draw boxes on image
                img_with_boxes = img_array.copy()
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 100:  # Filter small regions
                        x, y, w, h = cv2.boundingRect(contour)
                        # Draw green box
                        cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Create heatmap overlay
                heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(img_with_boxes, 0.6, heatmap_colored, 0.4, 0)

                axes[idx, 2].imshow(overlay)
                axes[idx, 2].set_title('Overlay + Bounding Boxes', fontsize=12, fontweight='bold')
                axes[idx, 2].axis('off')

            
            plt.tight_layout()
            plt.show()
            
            print("✓ Visualizations displayed")
        
        print("\n" + "="*70)
        print("RESULTS DISPLAY COMPLETE")
        print("="*70)

