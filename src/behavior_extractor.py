"""
Model Behavior Extraction Module
Extracts REAL model reasoning, not hardcoded templates!
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import cv2

from .config import LUNG_REGIONS, LABEL_NAMES


class ModelBehaviorExtractor:
    """
    Extracts interpretable behavior from the model's decision process
    """
    
    def __init__(self):
        self.lung_regions = LUNG_REGIONS
        
    def extract_complete_behavior(
        self,
        predictions: Dict[str, float],
        heatmap: np.ndarray,
        all_probabilities: np.ndarray,
        layer_activations: Dict = None
    ) -> Dict:
        """
        Extract comprehensive model behavior for a prediction
        
        Args:
            predictions: Dict of {disease: confidence}
            heatmap: Grad-CAM heatmap (H, W)
            all_probabilities: All 14 class probabilities
            layer_activations: Optional layer activation patterns
            
        Returns:
            Dict containing all behavior information
        """
        behavior = {
            "predictions": predictions,
            "spatial_analysis": self._analyze_spatial_focus(heatmap),
            "confidence_distribution": self._analyze_confidence_distribution(all_probabilities),
            "anatomical_regions": self._identify_affected_regions(heatmap),
            "feature_importance": self._extract_feature_patterns(heatmap),
            "decision_factors": self._extract_decision_factors(predictions, heatmap)
        }
        
        return behavior
    
    def _analyze_spatial_focus(self, heatmap: np.ndarray) -> Dict:
        """
        Analyze where the model is looking
        
        Returns focus statistics and coordinates
        """
        # Normalize heatmap
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Find peak attention
        peak_y, peak_x = np.unravel_index(np.argmax(heatmap_norm), heatmap_norm.shape)
        
        # Calculate attention distribution
        threshold = 0.7
        high_attention_mask = heatmap_norm > threshold
        attention_percentage = (high_attention_mask.sum() / heatmap_norm.size) * 100
        
        # Get bounding box of high attention region
        if high_attention_mask.any():
            coords = np.argwhere(high_attention_mask)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            bbox = {
                "top": int(y_min),
                "left": int(x_min),
                "bottom": int(y_max),
                "right": int(x_max)
            }
        else:
            bbox = None
        
        return {
            "peak_location": {"x": int(peak_x), "y": int(peak_y)},
            "attention_percentage": float(attention_percentage),
            "focus_intensity": float(heatmap_norm.max()),
            "bounding_box": bbox,
            "is_focal": attention_percentage < 30,  # Focal vs diffuse
            "is_diffuse": attention_percentage > 50
        }
    
    def _identify_affected_regions(self, heatmap: np.ndarray) -> List[str]:
        """
        Map heatmap to anatomical regions
        """
        h, w = heatmap.shape
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        affected_regions = []
        threshold = 0.3
        
        for region_name, coords in self.lung_regions.items():
            # Get region mask
            x_range = coords["x"]
            y_range = coords["y"]
            
            y_start = int(y_range[0] * h)
            y_end = int(y_range[1] * h)
            x_start = int(x_range[0] * w)
            x_end = int(x_range[1] * w)
            
            region_heatmap = heatmap_norm[y_start:y_end, x_start:x_end]
            
            # Check if region has significant attention
            if region_heatmap.size > 0:
                avg_attention = region_heatmap.mean()
                if avg_attention > threshold:
                    affected_regions.append({
                        "name": region_name,
                        "attention_score": float(avg_attention)
                    })
        
        # Sort by attention score
        affected_regions.sort(key=lambda x: x["attention_score"], reverse=True)
        
        return affected_regions
    
    def _analyze_confidence_distribution(self, all_probs: np.ndarray) -> Dict:
        """
        Analyze model's confidence across all classes
        """
        # Get top 5 diseases by probability
        top_indices = np.argsort(all_probs)[-5:][::-1]
        top_diseases = [(LABEL_NAMES[i], float(all_probs[i])) for i in top_indices]
        
        # Calculate entropy (uncertainty measure)
        probs_safe = all_probs + 1e-10
        entropy = -np.sum(probs_safe * np.log(probs_safe))
        
        return {
            "top_5_diseases": top_diseases,
            "entropy": float(entropy),
            "max_confidence": float(all_probs.max()),
            "min_confidence": float(all_probs.min()),
            "mean_confidence": float(all_probs.mean()),
            "is_uncertain": entropy > 2.0  # High entropy = uncertain
        }
    
    def _extract_feature_patterns(self, heatmap: np.ndarray) -> Dict:
        """
        Analyze visual patterns in the heatmap
        """
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)
        
        # Detect if pattern is focal or diffuse
        high_attention = (heatmap_norm > 0.7).sum()
        total_pixels = heatmap_norm.size
        focal_percentage = (high_attention / total_pixels) * 100
        
        # Simple texture analysis
        grad_x = np.abs(np.gradient(heatmap_norm, axis=1)).mean()
        grad_y = np.abs(np.gradient(heatmap_norm, axis=0)).mean()
        
        pattern_type = "focal" if focal_percentage < 25 else "diffuse"
        
        return {
            "pattern_type": pattern_type,
            "focal_percentage": float(focal_percentage),
            "gradient_strength": float((grad_x + grad_y) / 2),
            "uniformity": float(1.0 - np.std(heatmap_norm)),
            "intensity_range": {
                "min": float(heatmap_norm.min()),
                "max": float(heatmap_norm.max()),
                "mean": float(heatmap_norm.mean())
            }
        }
    
    def _extract_decision_factors(
        self,
        predictions: Dict[str, float],
        heatmap: np.ndarray
    ) -> Dict:
        """
        Extract human-interpretable decision factors
        """
        spatial = self._analyze_spatial_focus(heatmap)
        regions = self._identify_affected_regions(heatmap)
        patterns = self._extract_feature_patterns(heatmap)
        
        # Build natural language factors
        factors = {
            "primary_focus": self._describe_primary_focus(spatial, regions),
            "pattern_description": self._describe_pattern(patterns),
            "certainty_level": self._describe_certainty(predictions),
            "anatomical_summary": self._describe_anatomy(regions)
        }
        
        return factors
    
    def _describe_primary_focus(self, spatial: Dict, regions: List) -> str:
        """Generate description of where model focused"""
        if not regions:
            return "No specific region showed strong attention"
        
        top_region = regions[0]["name"].replace("_", " ")
        intensity = spatial["focus_intensity"]
        
        if intensity > 0.8:
            strength = "strongly"
        elif intensity > 0.6:
            strength = "moderately"
        else:
            strength = "mildly"
        
        return f"Model {strength} focused on {top_region} region"
    
    def _describe_pattern(self, patterns: Dict) -> str:
        """Describe visual pattern type"""
        pattern_type = patterns["pattern_type"]
        focal_pct = patterns["focal_percentage"]
        
        if pattern_type == "focal":
            return f"Focal abnormality pattern (concentrated in {focal_pct:.1f}% of area)"
        else:
            return f"Diffuse pattern (spread across {focal_pct:.1f}% of area)"
    
    def _describe_certainty(self, predictions: Dict) -> str:
        """Describe model's certainty level"""
        if not predictions:
            return "No findings above threshold"
        
        max_conf = max(predictions.values())
        
        if max_conf > 0.8:
            return f"High confidence ({max_conf*100:.1f}%)"
        elif max_conf > 0.6:
            return f"Moderate confidence ({max_conf*100:.1f}%)"
        else:
            return f"Low confidence ({max_conf*100:.1f}%)"
    
    def _describe_anatomy(self, regions: List) -> str:
        """Summarize affected anatomical regions"""
        if not regions:
            return "No specific anatomical regions highlighted"
        
        region_names = [r["name"].replace("_", " ") for r in regions[:3]]
        
        if len(region_names) == 1:
            return f"Primarily in {region_names[0]}"
        elif len(region_names) == 2:
            return f"In {region_names[0]} and {region_names[1]}"
        else:
            return f"Across {', '.join(region_names[:-1])}, and {region_names[-1]}"
        

    def extract_complete_behavior_verbose(self, predictions, heatmap, all_probabilities):
        """Extract behavior with verbose real-time output"""
        print("\nBEHAVIOR EXTRACTION")
        
        # Get disease info
        disease = list(predictions.keys())[0]
        confidence = list(predictions.values())[0]
        
        # SPATIAL ANALYSIS
        print("\n   SPATIAL ANALYSIS:")
        peak_y, peak_x = np.unravel_index(heatmap.argmax(), heatmap.shape)
        print(f"   • Peak location: ({peak_x}, {peak_y})")
        
        attention_coverage = (heatmap > 0.3).sum() / heatmap.size * 100
        print(f"   • Attention coverage: {attention_coverage:.1f}%")
        
        focus_intensity = heatmap.max()
        print(f"   • Focus intensity: {focus_intensity:.3f}")
        
        pattern_type = "FOCAL" if attention_coverage < 40 else "DIFFUSE"
        print(f"   • Pattern type: {pattern_type}")
        
        # ANATOMICAL REGIONS (use existing method!)
        print("\n   ANATOMICAL REGION DETECTION:")
        print("   Dividing image into 6 regions...")
        
        regions = self._identify_affected_regions(heatmap)
        
        if regions:
            print("\n   Region attention scores:")
            for region in regions:
                print(f"   • {region['name']:20s}: {region['attention_score']*100:5.1f}%")
            primary_region = regions[0]['name']
        else:
            print("\n   No regions with significant attention")
            primary_region = "unspecified"
        
        # DECISION FACTORS
        print("\n   DECISION FACTORS:")
        print(f"   • Primary focus: \"{primary_region.replace('_', ' ').title()}\"")
        print(f"   • Pattern: \"{pattern_type.capitalize()} attention pattern\"")
        
        certainty = "High" if confidence > 0.8 else "Moderate" if confidence > 0.6 else "Low"
        print(f"   • Certainty: {certainty} (confidence {confidence*100:.1f}%)")
        
        other_findings = sum(1 for p in all_probabilities if p > 0.5) - 1
        print(f"   • Supporting evidence: {other_findings} other co-occurring findings")
        
        print("\n   ✓ Behavior extraction complete")
        
        # Call real method for proper return structure
        actual_behavior = self.extract_complete_behavior(
            predictions=predictions,
            heatmap=heatmap,
            all_probabilities=all_probabilities
        )
        
        return actual_behavior




# Utility function for easy usage
def extract_behavior(predictions, heatmap, all_probabilities):
    """Convenience function to extract behavior"""
    extractor = ModelBehaviorExtractor()
    return extractor.extract_complete_behavior(
        predictions, heatmap, all_probabilities
    )
