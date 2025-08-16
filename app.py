from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)

CORS(app)

model = YOLO("./best.pt")

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return {'error': 'No image uploaded'}, 400

    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    results = model(img)
    result_img = results[0].plot()

    _, buffer = cv2.imencode('.jpg', result_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    dets = []
    names = results[0].names
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        dets.append({
            "label": names[cls_id],
            "confidence": float(box.conf[0]),
            "bbox": [float(x) for x in box.xyxy[0].tolist()]
        })

    return jsonify({
        "detections": dets,
        "image_base64": img_base64
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)