"""
LLM Integration - Sprint 3
Gemini 2.5 Flash with all 4 requirements
"""

from google import genai
import json
import os
from dotenv import load_dotenv

load_dotenv()

MODEL = 'gemini-2.5-flash'

class LLMClient:
    def __init__(self):
        """Initialize Gemini client"""

        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")

        self.client = genai.Client(api_key=api_key)
    
    def process_claim(self, cv_output, cost_output, vehicle_info):
        """Main LLM processing function"""
        
        # Calculate confidence
        confidence = self._calculate_confidence(cv_output, cost_output)

        # Determine STP eligibility
        total_cost = sum(cost_output['total_cost_range']) / 2
        severity_levels = [p['severity'] for p in cost_output['damaged_parts']]
        stp_decision = self._decide_stp(total_cost, confidence, severity_levels)
        
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
            confidence < 0.40
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
            'override_allowed': True,
            'model_version': '1.0.0',
        }
        
        # Validate structure
        self._validate_output(output)
        
        return output
    
    def _calculate_confidence(self, cv_output, cost_output):
        """
        Confidence = max Mask R-CNN part confidence across all detections.
        Uses max rather than mean because the strongest single detection is the
        most meaningful signal — averaging across many weak detections penalises
        images with multiple marginal detections even when at least one is solid.
        """
        cv_confidences = [d.get('confidence', 0.5) for d in cv_output['damaged_parts']]
        return round(max(cv_confidences) if cv_confidences else 0.5, 2)
    
    def _decide_stp(self, total_cost, confidence, severity_levels):
        """STP recommendation"""

        cost_ok       = total_cost < 1500
        confidence_ok = confidence > 0.60

        # Severity is intentionally excluded: the cost gate already bounds financial
        # risk. Blocking on severity alone would reject a $400 severe scratch — not useful.
        stp_eligible = cost_ok and confidence_ok

        if stp_eligible:
            reasoning = (
                f"Claim eligible for auto-approval: "
                f"cost ${total_cost:.0f} under $1,500 threshold, "
                f"{confidence:.0%} confidence meets requirement."
            )
        else:
            reasons = []
            if not cost_ok:
                reasons.append(f"cost ${total_cost:.0f} exceeds $1,500")
            if not confidence_ok:
                reasons.append(f"{confidence:.0%} confidence below 60%")
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
                f"{p['part'].lower()}: {p['damage_type'].lower()} ({p['severity']}) - "
                f"${p['cost_range'][0]}-${p['cost_range'][1]} to {p['action']}"
            )
        
        labor_rates = cost_output.get('labor_rates', {})
        labor_note  = f"body: ${labor_rates.get('body', 58)}/hr" if labor_rates else "national average rates"
        state = cost_output.get('state', '')
        location_note = f"state of {state} ({labor_note})" if state else f"national average rates ({labor_note})"

        prompt = f"""You are an auto insurance AI assistant. Generate a concise, professional claim explanation.

DAMAGE ASSESSMENT:
{chr(10).join(parts_summary)}

TOTAL ESTIMATED COST: ${cost_output['total_cost_range'][0]}-${cost_output['total_cost_range'][1]}

LOCATION: {location_note}

Rules:
- Refer to "the vehicle" only — never mention a year, make, or model
- Copy damage types and part names word-for-word from the DAMAGE ASSESSMENT above — do NOT substitute, paraphrase, or default to "dent"
- All part names and damage types are already lowercase — keep them lowercase mid-sentence
- Write exactly 2 sentences: one describing the damage and repair approach, one stating the cost estimate
- End after stating the cost — do not add next steps, reassurances, or follow-up offers

Return ONLY the 2-sentence explanation. No JSON, no extra formatting, no preamble."""

        try:
            response = self.client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config={"temperature": 0.3},
            )

            return response.text.strip()
            
        except Exception as e:
            print(f"Warning: Gemini API error: {e}")
            parts_list = ", ".join([p['part'].lower() for p in cost_output['damaged_parts']])
            return (
                f"This vehicle has sustained damage to the following parts: {parts_list}. "
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