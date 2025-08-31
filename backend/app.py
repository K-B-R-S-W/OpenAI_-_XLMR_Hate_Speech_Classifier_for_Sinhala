"""
XLM-RoBERTa Backend Service for Sinhala/Singlish Hate Speech Detection
Secondary/Fallback service for the AI-Backend system
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import csv
import glob
import logging
from datetime import datetime

# Delay heavy imports/model load until first use
from inference_xlm_roberta import HateSpeechDetectorXLM
import config_xlm_roberta as config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Lazy detector (initialize on first request)
_detector = None


def get_detector():
    global _detector
    if _detector is None:
        print("Initializing XLM-RoBERTa hate speech detector (lazy init)...")
        _detector = HateSpeechDetectorXLM()
    return _detector


def _latest_finetuned_snapshot():
    base = config.RETRAIN_OUTPUT_BASE
    if not os.path.exists(base):
        return None
    snaps = sorted(glob.glob(os.path.join(base, "*")))
    return snaps[-1] if snaps else None


@app.route('/')
def health_check():
    """Health check endpoint for backend service"""
    return jsonify({
        'service': 'roberta-backend',
        'status': 'healthy',
        'model_loaded': _detector is not None,
        'version': '1.0.0'
    })

@app.route('/admin/retrain', methods=['POST'])
def admin_retrain():
    """Trigger fine-tune on feedback CSV and hot-swap to latest snapshot."""
    try:
        # Fine-tune on feedback CSV
        from retrain_on_feedback import fine_tune_on_feedback
        epochs = int((request.json or {}).get('epochs', 2))
        lr = float((request.json or {}).get('lr', 2e-5))
        bs = int((request.json or {}).get('batch_size', 8))
        fine_tune_on_feedback(num_epochs=epochs, lr=lr, batch_size=bs)

        # Load latest snapshot
        latest = _latest_finetuned_snapshot()
        if latest:
            global _detector
            print(f"Reloading detector to snapshot: {latest}")
            _detector = HateSpeechDetectorXLM(model_path=latest)
            return jsonify({'ok': True, 'model_path': latest})
        else:
            return jsonify({'ok': False, 'error': 'No snapshots found after retrain'}), 500
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/detect', methods=['POST'])
def detect_hate_speech():
    """API endpoint for hate speech detection"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'error': 'No text provided'
            }), 400
        
        detector = get_detector()
        # Detect hate speech using XLM-RoBERTa
        result = detector.detect_hate_speech(text)
        
        # Format result for frontend
        formatted_result = {
            'sentence': result['sentence'],
            'sentence_label': result['sentence_label'],
            'confidence': round(float(result['confidence']), 3),
            'hate_words': result['highlighted_hate_words'],
            'words': result['words'],
            'word_predictions': result['word_predictions'],
            'word_probabilities': [
                [round(float(p[0]), 3), round(float(p[1]), 3)] if hasattr(p, '__len__') and len(p) > 1 else round(float(p), 3)
                for p in result['word_probabilities']
            ]
        }
        
        return jsonify(formatted_result)
        
    except Exception as e:
        print(f"Error in detection: {e}")
        return jsonify({
            'error': f'Detection failed: {str(e)}'
        }), 500


def _append_feedback(row: list):
    header = [
        'timestamp','input_text','model_label','user_thinks_label','user_words','client_ip','user_agent'
    ]
    is_new = not os.path.exists(config.FEEDBACK_CSV)
    os.makedirs(config.FEEDBACK_DIR, exist_ok=True)
    with open(config.FEEDBACK_CSV, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(header)
        writer.writerow(row)

@app.route('/feedback/hate_missed', methods=['POST'])
def feedback_hate_missed():
    """User thinks NOT_HATE output is actually HATE; collect words they think are hate."""
    try:
        data = request.get_json() or {}
        input_text = (data.get('text') or '').strip()
        user_words = (data.get('words') or '').strip()  # comma-separated words
        model_label = 'NOT_HATE'
        user_thinks_label = 'HATE'
        if not input_text or not user_words:
            return jsonify({'error': 'text and words are required'}), 400
        row = [
            datetime.utcnow().isoformat(),
            input_text,
            model_label,
            user_thinks_label,
            user_words,
            request.remote_addr,
            request.headers.get('User-Agent','')
        ]
        _append_feedback(row)
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feedback/false_positive', methods=['POST'])
def feedback_false_positive():
    """User thinks HATE output is actually NOT_HATE; collect words that are not hate."""
    try:
        data = request.get_json() or {}
        input_text = (data.get('text') or '').strip()
        user_words = (data.get('words') or '').strip()  # comma-separated words
        model_label = 'HATE'
        user_thinks_label = 'NOT_HATE'
        if not input_text or not user_words:
            return jsonify({'error': 'text and words are required'}), 400
        row = [
            datetime.utcnow().isoformat(),
            input_text,
            model_label,
            user_thinks_label,
            user_words,
            request.remote_addr,
            request.headers.get('User-Agent','')
        ]
        _append_feedback(row)
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_detect', methods=['POST'])
def batch_detect():
    """API endpoint for batch hate speech detection"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts or not isinstance(texts, list):
            return jsonify({
                'error': 'No texts provided or invalid format'
            }), 400
        
        detector = get_detector()
        results = []
        for text in texts:
            if text.strip():
                result = detector.detect_hate_speech(text.strip())
                results.append({
                    'text': result['sentence'],
                    'label': result['sentence_label'],
                    'confidence': round(float(result['confidence']), 3),
                    'hate_words': result['highlighted_hate_words']
                })
        
        return jsonify({
            'results': results,
            'total_processed': len(results)
        })
        
    except Exception as e:
        print(f"Error in batch detection: {e}")
        return jsonify({
            'error': f'Batch detection failed: {str(e)}'
        }), 500

@app.route('/stats')
def get_stats():
    """Get system statistics"""
    try:
        d = get_detector()
        model_info = {
            'model_name': config.MODEL_NAME,
            'max_length': config.MAX_LENGTH,
            'num_labels': config.NUM_LABELS,
            'device': str(d.device)
        }
        dict_info = {
            'total_words': len(d.hate_dict),
            'sample_words': list(d.hate_dict)[:10] if d.hate_dict else []
        }
        return jsonify({'model': model_info, 'dictionary': dict_info})
    except Exception as e:
        return jsonify({'error': f'Failed to get stats: {str(e)}'}), 500

@app.route('/health')
def backend_health():
    """Detailed health check for backend service"""
    d = _detector
    return jsonify({
        'service': 'roberta-backend',
        'status': 'healthy',
        'model_loaded': d is not None and d.model is not None,
        'tokenizer_loaded': d is not None and d.tokenizer is not None,
        'device': str(d.device) if d else 'unknown'
    })

if __name__ == '__main__':
    print("🤖 Starting XLM-RoBERTa Backend Service...")
    print("🔧 Secondary/Fallback service for AI-Backend")
    print(f"🌐 Running on: http://localhost:{config.ROBERTA_BACKEND_PORT if hasattr(config, 'ROBERTA_BACKEND_PORT') else 8002}")
    print("=" * 50)
    
    app.run(
        debug=True, 
        host='0.0.0.0', 
        port=getattr(config, 'ROBERTA_BACKEND_PORT', 8002)
    )
