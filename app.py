from flask import Flask, request, jsonify
import os
import re
import logging
import time
from xai_grok_sdk.client import Client
from xai_grok_sdk.chat import user

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

XAI_API_KEY = os.getenv("XAI_API_KEY")
if not XAI_API_KEY:
    logger.error("XAI_API_KEY is not set")
    raise ValueError("XAI_API_KEY is not set")

client = Client(api_key=XAI_API_KEY)

template = """ 
Context: {context} 
 
Historical Data (for reference, use only for general trends): 
{historical_data} 
 
Box Information: {box_info} 
 
You are an expert in evaluating Goodiebox welcome boxes for their ability to attract new 
members in Denmark. Predict the daily intake (new members per day) at a CAC of 17.5 
EUR. A regression model predicts {predicted_intake}. Use this as the primary basis, apply 
minimal adjustments based on Box Information and historical trends, and return the final 
predicted daily intake as a whole number. Prioritize higher intakes for boxes with premium 
products and high retail value, adding up to +5% for boxes with 2 or more premium products. 
 
**Step 1: Start with Predicted Intake** 
- Use {predicted_intake}. 
 
**Step 2: Apply Minimal Adjustments** 
- Adjust based on retail value, premium products, ratings, free gift value. 
- Use historical data for trends (e.g., high retail value increases intake). 
- Max ±5% change: 
  - +0.5% per 10 EUR retail value above 100 EUR (max +2%). 
  - +0.5% per premium product above 3 (max +1.5%). 
  - +0.25% per 10 EUR free gift value if rating >4.0 (max +1.5%). 
  - +5% if 2 or more premium products. 
 
**Step 3: Clamp** 
- Ensure 1–90 members/day. 
 
**Step 4: Round** 
- Round to nearest whole number. 
 
Return only the numerical value (e.g., 10). 
"""

def call_xai_api(prompt_text, model_name="grok-4-0709"):
    try:
        logger.info(f"Calling xAI API with prompt: {prompt_text[:50]}...")
        chat = client.chat.create(model=model_name, temperature=0.7)
        chat.append(user(prompt_text))
        response = chat.sample()
        return response.content.strip()
    except Exception as e:
        logger.error(f"xAI API error: {str(e)}")
        raise

def predict_box_intake(context, box_info, predicted_intake, historical_data):
    try:
        logger.info(f"Processing prediction with predicted_intake: {predicted_intake}")
        prompt_text = template.format(
            context=context or "",
            historical_data=historical_data or "",
            box_info=box_info or "",
            predicted_intake=predicted_intake or 0
        )
        for attempt in range(3):
            try:
                result = call_xai_api(prompt_text)
                match = re.search(r'^\d+$', result)
                if match:
                    intake = round(float(match.group()))
                    intake = max(1, min(90, intake))
                    logger.info(f"Predicted intake: {intake}")
                    return intake
                logger.warning(f"Invalid format: {result}")
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed: {str(e)}")
                if attempt < 2:
                    time.sleep(2)
        raise ValueError("No valid intake value after retries")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise

@app.route('/predict_box_score', methods=['POST'])
def box_score():
    try:
        data = request.get_json() or {}
        logger.info(f"Received request data: {data}")
        required = ['box_info', 'context', 'predicted_intake']
        missing = set(required) - set(data.keys())
        if missing:
            logger.error(f"Missing fields: {missing}")
            return jsonify({'error': f"Missing fields: {missing}"}), 400
        if not isinstance(data['predicted_intake'], (int, float)):
            logger.error("Invalid predicted_intake type")
            return jsonify({'error': "predicted_intake must be a number"}), 400
        intake = predict_box_intake(
            data['context'],
            data['box_info'],
            float(data['predicted_intake']),
            data.get('historical_data', '')
        )
        return jsonify({'predicted_intake': intake}), 200
    except ValueError as e:
        logger.error(f"Value error: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    logger.info("Health check")
    return jsonify({'status': 'healthy'}), 200

@app.route('/test_model', methods=['GET'])
def test_model():
    try:
        logger.info("Testing xAI model")
        result = call_xai_api("Please provide a single number: 42")
        match = re.search(r'^\d+$', result)
        if match and int(match.group()) == 42:
            logger.info("Test successful")
            return jsonify({'status': 'success', 'response': result}), 200
        return jsonify({'status': 'error', 'error': 'Invalid test response'}), 400
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask app...")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
