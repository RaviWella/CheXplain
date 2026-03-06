"""
LLM-Based Explanation Generator
Uses LLaMA 3 to generate dynamic, context-aware explanations
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from .config import LLM_CONFIG, CLINICIAN_PROMPT_TEMPLATE, PATIENT_PROMPT_TEMPLATE, DISEASE_INFO


class LLMExplainer:
    """
    Generate natural language explanations using LLaMA 3
    """
    
    def __init__(
        self,
        model_name: str = None,
        use_quantization: bool = True,
        device: str = "auto"
    ):
        """
        Initialize LLM Explainer
        
        Args:
            model_name: HuggingFace model name (default from config)
            use_quantization: Use 4-bit quantization to save memory
            device: Device to load model on
        """
        self.model_name = model_name or LLM_CONFIG["model_name"]
        self.use_quantization = use_quantization
        self.device = device
        
        print(f" Loading LLM: {self.model_name}")
        print(f"   Quantization: {'Enabled (4-bit)' if use_quantization else 'Disabled'}")
        
        # Load model and tokenizer
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        self._load_model()
        
        print(" LLM loaded successfully!")


    
    def _load_model(self):
        """Load the LLM with appropriate configuration"""
        
        try:
            # Check if we have GPU
            has_cuda = torch.cuda.is_available()
            device = "cuda" if has_cuda else "cpu"
            
            print(f"   Device: {device.upper()}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Set padding token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Check model type for appropriate loading
            if "t5" in self.model_name.lower():
                # T5 models use Seq2Seq architecture
                from transformers import AutoModelForSeq2SeqLM
                
                print(f"   Loading T5 model on {device.upper()}...")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    device_map=device,
                    torch_dtype=torch.float32 if device == "cpu" else torch.float16,
                    low_cpu_mem_usage=True
                )
                
                # Create text2text generation pipeline
                self.pipeline = pipeline(
                    "text2text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=-1 if device == "cpu" else 0,
                    max_length=300
                )
                
            else:
                # Causal LM models (GPT, LLaMA, Mistral, etc.)
                print(f"   Loading causal LM on {device.upper()}...")
                
                if self.use_quantization and has_cuda:
                    # Use quantization only on GPU
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        quantization_config=quantization_config,
                        device_map=self.device,
                        trust_remote_code=True,
                        torch_dtype=torch.float16
                    )
                else:
                    # No quantization (CPU or user preference)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        device_map=device,
                        trust_remote_code=True,
                        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
                        low_cpu_mem_usage=True
                    )
                
                # Create text generation pipeline
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=-1 if device == "cpu" else 0,
                    **LLM_CONFIG["generation_params"]
                )
            
            print(f"Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading LLM: {e}")
            print("\nTrying lighter fallback models...")
            self._load_fallback_model()


    
    def _load_fallback_model(self):
        """Load a smaller fallback model if LLaMA fails"""
        
        # multiple fallback options
        fallback_models = [
            "google/flan-t5-large",      # Good quality, no auth needed
            "microsoft/BioGPT-Large",    # Medical domain
            "gpt2-medium",               # Always available
        ]
        
        for fallback_model in fallback_models:
            try:
                print(f"   Trying fallback: {fallback_model}...")
                
                self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                
                # Use appropriate model class based on architecture
                if "t5" in fallback_model.lower():
                    from transformers import AutoModelForSeq2SeqLM
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        fallback_model,
                        device_map=self.device
                    )
                    # T5 uses text2text generation
                    self.pipeline = pipeline(
                        "text2text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        max_length=300
                    )
                else:
                    # GPT-style models use causal LM
                    self.model = AutoModelForCausalLM.from_pretrained(
                        fallback_model,
                        device_map=self.device
                    )
                    self.pipeline = pipeline(
                        "text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        max_length=300
                    )
                
                print(f"Fallback model loaded: {fallback_model}")
                return
                
            except Exception as e:
                print(f"   Failed: {e}")
                continue
        
        # If all fallback models fail, set pipeline to None (use template fallback)
        print("All LLM models failed. Using template-based explanations.")
        self.pipeline = None

    
    def generate_explanation(
        self,
        disease: str,
        confidence: float,
        behavior: Dict,
        layer: str = "clinician"
    ) -> str:
        """
        Generate explanation for a disease detection
        
        Args:
            disease: Detected disease name
            confidence: Prediction confidence (0-1)
            behavior: Model behavior dict from behavior_extractor
            layer: "clinician" or "patient"
            
        Returns:
            Generated explanation text
        """
        if self.pipeline is None:
            return self._generate_fallback_explanation(disease, confidence, layer)
        
        # Build prompt
        prompt = self._build_prompt(disease, confidence, behavior, layer)
        
        # Generate
        try:
            outputs = self.pipeline(
                prompt,
                max_new_tokens=LLM_CONFIG["generation_params"]["max_new_tokens"],
                do_sample=True,
                temperature=LLM_CONFIG["generation_params"]["temperature"],
                top_p=LLM_CONFIG["generation_params"]["top_p"]
            )
            
            # Extract generated text
            generated_text = outputs[0]["generated_text"]
            
            # Remove prompt from output (LLaMA includes it)
            explanation = generated_text.replace(prompt, "").strip()
            
            # Clean up
            explanation = self._clean_output(explanation)
            
            return explanation
            
        except Exception as e:
            print(f" Generation error: {e}")
            return self._generate_fallback_explanation(disease, confidence, layer)
    
    def _build_prompt(
        self,
        disease: str,
        confidence: float,
        behavior: Dict,
        layer: str
    ) -> str:
        """
        Build prompt for LLM from model behavior
        
        Args:
            disease: Disease name
            confidence: Confidence score
            behavior: Extracted model behavior
            layer: Target audience layer
            
        Returns:
            Formatted prompt string
        """
        # Extract relevant info from behavior
        regions = behavior.get("anatomical_regions", [])
        spatial = behavior.get("spatial_analysis", {})
        decision_factors = behavior.get("decision_factors", {})
        
        # Format regions
        if regions:
            region_list = ", ".join([
                f"{r['name'].replace('_', ' ')} ({r['attention_score']*100:.0f}%)"
                for r in regions[:3]
            ])
        else:
            region_list = "No specific regions highlighted"
        
        # Format features
        features = []
        if spatial.get("is_focal"):
            features.append("focal abnormality pattern")
        elif spatial.get("is_diffuse"):
            features.append("diffuse pattern")
        
        if spatial.get("focus_intensity", 0) > 0.7:
            features.append("high intensity")
        
        features_text = ", ".join(features) if features else "subtle changes"
        
        # Get attention summary
        attention_summary = decision_factors.get("primary_focus", "Multiple regions examined")
        
        # Choose template
        if layer == "clinician":
            template = CLINICIAN_PROMPT_TEMPLATE
        else:
            template = PATIENT_PROMPT_TEMPLATE
            # Simplify features for patients
            features_text = self._simplify_features(features_text, disease)
        
        # Fill template
        prompt = template.format(
            disease=disease,
            confidence=f"{confidence*100:.1f}",
            regions=region_list,
            features=features_text,
            features_simplified=features_text,
            attention_summary=attention_summary
        )
        
        return prompt
    
    def _simplify_features(self, technical_features: str, disease: str) -> str:
        """
        Convert technical features to patient-friendly language
        
        Args:
            technical_features: Technical description
            disease: Disease name
            
        Returns:
            Simplified description
        """
        simplifications = {
            "focal abnormality pattern": "a specific area that looks different",
            "diffuse pattern": "changes spread across the lungs",
            "high intensity": "strong signal",
            "consolidation": "fluid or infection in the air sacs",
            "opacity": "cloudy area",
            "infiltration": "fluid buildup",
        }
        
        simplified = technical_features
        for tech, simple in simplifications.items():
            simplified = simplified.replace(tech, simple)
        
        return simplified
    
    def _clean_output(self, text: str) -> str:
        """
        Clean up generated text
        
        Args:
            text: Raw generated text
            
        Returns:
            Cleaned text
        """
        # Remove common artifacts
        text = text.strip()
        
        # Remove markdown if present
        text = text.replace("**", "").replace("##", "")
        
        # Remove incomplete sentences at the end
        if text and text[-1] not in '.!?':
            last_period = text.rfind('.')
            if last_period > len(text) * 0.7:  # Keep if >70% complete
                text = text[:last_period + 1]
        
        return text
    
    def _generate_fallback_explanation(
        self,
        disease: str,
        confidence: float,
        layer: str
    ) -> str:
        """
        Generate basic template-based explanation if LLM fails
        
        This is a safety fallback - better than nothing!
        """
        if layer == "clinician":
            return (
                f"The AI model detected {disease} with {confidence*100:.1f}% confidence. "
                f"This finding is based on analysis of radiographic features consistent with "
                f"the disease pattern. Further clinical correlation and radiologist review "
                f"is recommended. The confidence level "
                f"{'suggests high certainty' if confidence > 0.7 else 'indicates moderate certainty' if confidence > 0.5 else 'is relatively low'}."
            )
        else:  # patient
            disease_info = DISEASE_INFO.get(disease, {})
            description = disease_info.get("description", disease.lower())
            
            return (
                f"The AI detected signs of {description} in your X-ray. "
                f"The computer is {confidence*100:.0f}% confident about this finding. "
                f"This means the AI noticed patterns in your chest X-ray that might indicate {disease.lower()}. "
                f"Your doctor will review these results and discuss them with you. "
                f"Remember, this is just an AI analysis - your doctor will make the final diagnosis."
            )
    
    def generate_batch_explanations(
        self,
        findings: Dict[str, float],
        behaviors: Dict[str, Dict]
    ) -> Dict[str, Dict[str, str]]:
        """
        Generate explanations for multiple findings
        
        Args:
            findings: Dict of {disease: confidence}
            behaviors: Dict of {disease: behavior_dict}
            
        Returns:
            Dict of {disease: {"clinician": text, "patient": text}}
        """
        explanations = {}
        
        for disease, confidence in findings.items():
            behavior = behaviors.get(disease, {})
            
            explanations[disease] = {
                "clinician": self.generate_explanation(
                    disease, confidence, behavior, "clinician"
                ),
                "patient": self.generate_explanation(
                    disease, confidence, behavior, "patient"
                )
            }
        
        return explanations
    
    def generate_summary(
        self,
        findings: Dict[str, float],
        layer: str = "clinician"
    ) -> str:
        """
        Generate overall summary of all findings
        
        Args:
            findings: Dict of all findings
            layer: Target audience
            
        Returns:
            Summary text
        """
        if not findings:
            if layer == "clinician":
                return (
                    "No significant pathological findings detected by the AI model above "
                    "the confidence threshold. Routine radiologist review recommended to "
                    "confirm negative findings."
                )
            else:
                return (
                    "The AI did not detect any significant issues in your chest X-ray. "
                    "Your doctor will perform a complete review to confirm this assessment."
                )
        
        # Build summary prompt
        findings_list = [f"{disease} ({conf*100:.0f}%)" for disease, conf in findings.items()]
        
        if layer == "clinician":
            prompt = (
                f"Provide a brief clinical summary of these AI-detected findings: "
                f"{', '.join(findings_list)}. "
                f"Include recommendations for further evaluation."
            )
        else:
            prompt = (
                f"Explain these X-ray findings to a patient in simple terms: "
                f"{', '.join(findings_list)}. "
                f"Be reassuring and explain what happens next."
            )
        
        if self.pipeline:
            try:
                outputs = self.pipeline(prompt, max_new_tokens=200)
                summary = outputs[0]["generated_text"].replace(prompt, "").strip()
                return self._clean_output(summary)
            except:
                pass
        
        # Fallback summary
        if layer == "clinician":
            return (
                f"The AI detected {len(findings)} finding(s): {', '.join(findings_list)}. "
                f"Comprehensive radiologist review recommended for correlation with "
                f"clinical history and additional imaging if needed."
            )
        else:
            return (
                f"The AI found {len(findings)} area(s) that need attention. "
                f"Your doctor will explain what this means for you and discuss next steps."
            )

    def generate_explanation_verbose(self, disease, confidence, behavior, layer):
        """Generate with verbose thinking"""
        print(f"\nLLM EXPLANATION: {layer.upper()}")
        
        # Build prompt
        prompt = self._build_prompt(disease, confidence, behavior, layer)
        
        # Show actual prompt
        print("\n   PROMPT SENT:")
        print("   " + "─"*50)
        for line in prompt.split('\n'):
            print(f"   {line}")
        print("   " + "─"*50)
        
        # Tokenize
        print(f"\nTokenizing...")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        print(f"   • Input tokens: {inputs['input_ids'].shape[1]}")
        
        # Generate
        print(f"Generating response...")
        
        # If possible, show token-by-token (T5 doesn't support this easily)
        outputs = self.pipeline(prompt, max_length=300)
        
        explanation = outputs[0]['generated_text']
        
        print(f"\n   GENERATED EXPLANATION:")
        print("   " + "─"*50)
        print(f"   {explanation}")
        print("   " + "─"*50)
        print(f"   • Output tokens: ~{len(explanation.split())}")
        
        return explanation




# Convenience function
def create_explainer(use_quantization: bool = True) -> LLMExplainer:
    """
    Create and return an LLM explainer instance
    
    Args:
        use_quantization: Whether to use 4-bit quantization
        
    Returns:
        Initialized LLMExplainer
    """
    return LLMExplainer(use_quantization=use_quantization)
