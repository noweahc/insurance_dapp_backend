from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask_cors import CORS
import os
import logging

app = Flask(__name__)
CORS(app)  # 모든 도메인에 대해 CORS 허용

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hugging Face에서 텍스트 생성 모델 불러오기 (여기서는 gpt-neo-2.7B 사용)
try:
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
    logger.info("Tokenizer and model loaded successfully.")
except Exception as e:
    logger.error("Error loading tokenizer or model: %s", e)
    raise e

# 보험 계약서 생성 함수
def generate_insurance_contract(customer_data):
    prompt = f"Generate an insurance contract based on the following customer data: {customer_data}"

    try:
        # 입력 텍스트를 토큰화하고, 모델을 통해 텍스트 생성
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(inputs['input_ids'], max_length=300, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
        
        # 생성된 텍스트 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        logger.error("Error generating contract: %s", e)
        return "Error generating contract."

# Flask API 엔드포인트
@app.route('/generate-contract', methods=['POST'])
def generate_contract():
    try:
        data = request.get_json()  # 프론트엔드에서 보내는 JSON 데이터를 받음
        if not data or 'customer_data' not in data:
            return jsonify({'error': 'Invalid input data'}), 400

        customer_data = data.get('customer_data')
        logger.info("Received customer data: %s", customer_data)
        
        # 보험 계약서 생성
        contract = generate_insurance_contract(customer_data)
        logger.info("Generated contract: %s", contract)
        
        return jsonify({'contract': contract}), 200
    except Exception as e:
        logger.error("Error in /generate-contract endpoint: %s", e)
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
