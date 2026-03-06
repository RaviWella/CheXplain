"""
Advanced Prompt Engineering Utilities
"""

from typing import Dict, List


class PromptBuilder:
    """
    Advanced prompt construction for medical explanations
    """
    
    @staticmethod
    def build_context_aware_prompt(
        disease: str,
        confidence: float,
        behavior: Dict,
        patient_history: str = None,
        layer: str = "clinician"
    ) -> str:
        """
        Build context-aware prompt with additional medical knowledge
        
        Args:
            disease: Disease name
            confidence: Confidence score
            behavior: Model behavior
            patient_history: Optional patient context
            layer: Target layer
            
        Returns:
            Enhanced prompt
        """
        base_prompt = f"""
You are an expert radiologist AI assistant.

**Case Information:**
- Finding: {disease}
- AI Confidence: {confidence*100:.1f}%
- Spatial Focus: {behavior.get('decision_factors', {}).get('anatomical_summary', 'N/A')}
- Pattern: {behavior.get('decision_factors', {}).get('pattern_description', 'N/A')}
"""
        
        if patient_history:
            base_prompt += f"\n- Clinical Context: {patient_history}\n"
        
        if layer == "clinician":
            base_prompt += """
Provide a technical radiological explanation including:
1. Radiographic findings
2. Anatomical localization
3. Differential diagnosis
4. Recommended follow-up
"""
        else:
            base_prompt += """
Explain this finding to a patient using:
1. Simple, clear language
2. Everyday analogies
3. Reassurance where appropriate
4. Next steps
"""
        
        return base_prompt
    
    @staticmethod
    def build_comparative_prompt(
        primary_finding: str,
        differential_diagnoses: List[str],
        confidence_scores: Dict[str, float]
    ) -> str:
        """
        Build prompt for differential diagnosis explanation
        
        Args:
            primary_finding: Main detected disease
            differential_diagnoses: List of alternative diagnoses
            confidence_scores: Confidence for each
            
        Returns:
            Comparative prompt
        """
        prompt = f"""
Primary AI Detection: {primary_finding} ({confidence_scores[primary_finding]*100:.1f}%)

Differential Considerations:
"""
        for disease in differential_diagnoses:
            conf = confidence_scores.get(disease, 0)
            prompt += f"- {disease}: {conf*100:.1f}%\n"
        
        prompt += """
Explain why the AI favored the primary finding and what features distinguish it 
from the differentials. Be concise and technical.
"""
        
        return prompt
    
    @staticmethod
    def build_negative_finding_prompt(layer: str = "clinician") -> str:
        """
        Prompt for explaining negative (normal) findings
        
        Args:
            layer: Target audience
            
        Returns:
            Prompt for negative findings
        """
        if layer == "clinician":
            return """
The AI analysis did not detect significant pathological findings above the confidence threshold.

Provide a brief technical statement explaining:
1. What the AI examined
2. Why no abnormalities were flagged
3. Recommendations for radiologist review

Keep it professional and concise (3-4 sentences).
"""
        else:
            return """
The AI did not find concerning signs in the chest X-ray.

Explain to a patient:
1. What this means in simple terms
2. Why the doctor still needs to review
3. Provide reassurance

Use warm, simple language (3-4 sentences).
"""


def enhance_prompt_with_medical_knowledge(
    base_prompt: str,
    disease: str,
    medical_db: Dict = None
) -> str:
    """
    Enhance prompt with medical domain knowledge
    
    Args:
        base_prompt: Base prompt
        disease: Disease name
        medical_db: Optional medical knowledge database
        
    Returns:
        Enhanced prompt
    """
    if medical_db and disease in medical_db:
        info = medical_db[disease]
        enhancement = f"""
        
**Medical Context:**
- Typical presentation: {info.get('description', 'N/A')}
- Common locations: {', '.join(info.get('typical_location', []))}
- Severity: {info.get('severity', 'varies')}
"""
        return base_prompt + enhancement
    
    return base_prompt
