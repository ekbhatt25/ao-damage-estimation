"""
LLM Integration - Sprint 3
Gemini 3 Flash with all 4 requirements
"""

import google.generativeai as genai
import json
import os
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self):
        """Initialize Gemini client"""
        
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def process_claim(self, cv_output, cost_output, vehicle_info):
        """Main LLM processing function"""
        
        # Calculate confidence
        confidence = self._calculate_confidence(cv_output, cost_output)
        
        # Determine STP eligibility
        total_cost = sum(cost_output['total_cost_range']) / 2
        severity_levels = [p['severity'] for p in cost_output['damaged_parts']]
        total_loss = cost_output.get('total_loss', False)
        stp_decision = self._decide_stp(total_cost, confidence, severity_levels, total_loss)
        
        # Generate explanation
        explanation = self._generate_explanation(
            cost_output,
            vehicle_info,
            confidence,
            stp_decision
        )
        
        # Auto-escalation: force manual review if confidence too low
        requires_review = (
            not stp_decision['stp_eligible'] or
            confidence < 0.60 or
            cost_output.get('total_loss', False)
        )

        # Build final output
        output = {
            'damaged_parts': cost_output['damaged_parts'],
            'total_cost_range': cost_output['total_cost_range'],
            'explanation': explanation,
            'confidence_score': confidence,
            'stp_eligible': stp_decision['stp_eligible'],
            'stp_reasoning': stp_decision['stp_reasoning'],
            'requires_adjuster_review': requires_review,
            'total_loss': cost_output.get('total_loss', False),
            'override_allowed': True,   # adjuster can always override STP decision
            'model_version': '1.0.0',
        }
        
        # Validate structure
        self._validate_output(output)
        
        return output
    
    def _calculate_confidence(self, cv_output, cost_output):
        """Confidence scoring"""
        
        # Average CV confidence
        cv_confidences = [d.get('confidence', 0.5) for d in cv_output['damaged_parts']]
        avg_cv = sum(cv_confidences) / len(cv_confidences) if cv_confidences else 0.5
        
        # Cost certainty
        cost_range = cost_output['total_cost_range']
        if cost_range[1] > 0:
            range_ratio = (cost_range[1] - cost_range[0]) / cost_range[1]
            cost_certainty = 1.0 - min(range_ratio, 1.0)
        else:
            cost_certainty = 0.5
        
        # Combined
        combined = (avg_cv * 0.7) + (cost_certainty * 0.3)
        
        return round(combined, 2)
    
    def _decide_stp(self, total_cost, confidence, severity_levels, total_loss=False):
        """STP recommendation"""

        cost_ok       = total_cost < 1500
        confidence_ok = confidence > 0.80
        severity_ok   = 'major' not in severity_levels
        not_total_loss = not total_loss

        stp_eligible = cost_ok and confidence_ok and severity_ok and not_total_loss

        if stp_eligible:
            reasoning = (
                f"Claim eligible for auto-approval: "
                f"cost ${total_cost:.0f} under $1,500 threshold, "
                f"{confidence:.0%} confidence meets requirement, "
                f"no major damage detected, not a total loss."
            )
        else:
            reasons = []
            if total_loss:
                reasons.append("repair cost exceeds 70% of vehicle ACV — total loss")
            if not cost_ok:
                reasons.append(f"cost ${total_cost:.0f} exceeds $1,500")
            if not confidence_ok:
                reasons.append(f"{confidence:.0%} confidence below 80%")
            if not severity_ok:
                reasons.append("major damage requires review")

            reasoning = f"Manual adjuster review required: {', '.join(reasons)}."

        return {
            'stp_eligible': stp_eligible,
            'stp_reasoning': reasoning
        }
    
    def _generate_explanation(self, cost_output, vehicle_info, confidence, stp_decision):
        """Explanation generation via Gemini"""
        
        # Build parts summary
        parts_summary = []
        for p in cost_output['damaged_parts']:
            parts_summary.append(
                f"{p['part']}: {p['damage_type']} ({p['severity']}) - "
                f"${p['cost_range'][0]}-${p['cost_range'][1]} to {p['action']}"
            )
        
        labor_rates = cost_output.get('labor_rates', {})
        labor_note  = f"body: ${labor_rates.get('body', 58)}/hr" if labor_rates else "national average rates"

        prompt = f"""You are an auto insurance AI assistant. Generate a professional, customer-friendly claim explanation.

VEHICLE: {vehicle_info['year']} {vehicle_info['make']} {vehicle_info['model']}

DAMAGE ASSESSMENT:
{chr(10).join(parts_summary)}

TOTAL ESTIMATED COST: ${cost_output['total_cost_range'][0]}-${cost_output['total_cost_range'][1]}

LOCATION: ZIP {cost_output['zip_code']} ({labor_note})

Write a professional 2-3 sentence explanation for the customer that:
1. Clearly describes what damage was found
2. Explains the repair approach (repair vs replace)
3. Mentions the cost estimate
4. Is reassuring and professional in tone
5. Avoids technical jargon

Return ONLY the explanation text. No JSON, no extra formatting, no preamble."""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": 0.3}
            )
            
            return response.text.strip()
            
        except Exception as e:
            print(f"Warning: Gemini API error: {e}")
            parts_list = ", ".join([p['part'] for p in cost_output['damaged_parts']])
            return (
                f"Your {vehicle_info['year']} {vehicle_info['make']} {vehicle_info['model']} "
                f"has sustained damage to the following parts: {parts_list}. "
                f"Based on our assessment, the estimated repair cost is "
                f"${cost_output['total_cost_range'][0]}-${cost_output['total_cost_range'][1]}."
            )
    
    def _validate_output(self, output):
        """JSON validation"""
        
        required_fields = [
            'damaged_parts',
            'total_cost_range',
            'explanation',
            'confidence_score',
            'stp_eligible',
            'stp_reasoning'
        ]
        
        for field in required_fields:
            if field not in output:
                raise ValueError(f"Missing required field: {field}")
        
        assert isinstance(output['confidence_score'], (int, float))
        assert isinstance(output['stp_eligible'], bool)
        assert isinstance(output['explanation'], str)
        
        return True