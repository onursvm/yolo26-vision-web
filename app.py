import base64
import time
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
from ultralytics import YOLO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'yolo26-vision-secret'
app.config['TEMPLATES_AUTO_RELOAD'] = True
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# YOLO26 Modelini Yükle
print("[INFO] YOLO26 modeli yükleniyor...")
model = YOLO("yolo26n.pt")
print("[INFO] YOLO26 modeli hazır!")

# 80 COCO Sınıfının Türkçe Çevirileri
COCO_TR = {
    'person': 'İnsan',
    'bicycle': 'Bisiklet',
    'car': 'Araba',
    'motorcycle': 'Motosiklet',
    'airplane': 'Uçak',
    'bus': 'Otobüs',
    'train': 'Tren',
    'truck': 'Kamyon',
    'boat': 'Tekne',
    'traffic light': 'Trafik Işığı',
    'fire hydrant': 'Yangın Musluğu',
    'stop sign': 'Dur İşareti',
    'parking meter': 'Parkmetre',
    'bench': 'Bank',
    'bird': 'Kuş',
    'cat': 'Kedi',
    'dog': 'Köpek',
    'horse': 'At',
    'sheep': 'Koyun',
    'cow': 'İnek',
    'elephant': 'Fil',
    'bear': 'Ayı',
    'zebra': 'Zebra',
    'giraffe': 'Zürafa',
    'backpack': 'Sırt Çantası',
    'umbrella': 'Şemsiye',
    'handbag': 'El Çantası',
    'tie': 'Kravat',
    'suitcase': 'Bavul',
    'frisbee': 'Frizbi',
    'skis': 'Kayak',
    'snowboard': 'Snowboard',
    'sports ball': 'Spor Topu',
    'kite': 'Uçurtma',
    'baseball bat': 'Beyzbol Sopası',
    'baseball glove': 'Beyzbol Eldiveni',
    'skateboard': 'Kaykay',
    'surfboard': 'Sörf Tahtası',
    'tennis racket': 'Tenis Raketi',
    'bottle': 'Şişe',
    'wine glass': 'Kadeh',
    'cup': 'Fincan / Kupa',
    'fork': 'Çatal',
    'knife': 'Bıçak',
    'spoon': 'Kaşık',
    'bowl': 'Kase',
    'banana': 'Muz',
    'apple': 'Elma',
    'sandwich': 'Sandviç',
    'orange': 'Portakal',
    'broccoli': 'Brokoli',
    'carrot': 'Havuç',
    'hot dog': 'Sosisli',
    'pizza': 'Pizza',
    'donut': 'Donut',
    'cake': 'Kek / Pasta',
    'chair': 'Sandalye',
    'couch': 'Koltuk',
    'potted plant': 'Saksı Bitkisi',
    'bed': 'Yatak',
    'dining table': 'Yemek Masası',
    'toilet': 'Tuvalet',
    'tv': 'Televizyon',
    'laptop': 'Dizüstü Bilgisayar',
    'mouse': 'Fare (Mouse)',
    'remote': 'Kumanda',
    'keyboard': 'Klavye',
    'cell phone': 'Cep Telefonu',
    'microwave': 'Mikrodalga',
    'oven': 'Fırın',
    'toaster': 'Tost Makinesi',
    'sink': 'Lavabo',
    'refrigerator': 'Buzdolabı',
    'book': 'Kitap',
    'clock': 'Saat',
    'vase': 'Vazo',
    'scissors': 'Makas',
    'teddy bear': 'Oyuncak Ayı',
    'hair drier': 'Saç Kurutma',
    'toothbrush': 'Diş Fırçası'
}


def process_frame(frame, conf_threshold=0.40):
    """Kare üzerinde YOLO26 nesne algılama çalıştırır ve sonuçları döndürür."""
    start_time = time.time()
    results = model(frame, conf=conf_threshold, verbose=False)
    infer_time = (time.time() - start_time) * 1000

    detections = []
    h, w = frame.shape[:2]

    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                class_name_tr = COCO_TR.get(class_name, class_name)

                detections.append({
                    'class_id': class_id,
                    'name': class_name,
                    'name_tr': class_name_tr,
                    'confidence': round(confidence, 3),
                    'box': [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    'box_normalized': [
                        round(x1 / w, 4),
                        round(y1 / h, 4),
                        round(x2 / w, 4),
                        round(y2 / h, 4)
                    ]
                })

    return {
        'detections': detections,
        'count': len(detections),
        'inference_time_ms': round(infer_time, 1),
        'frame_size': {'width': w, 'height': h}
    }


@app.route('/')
def index():
    """Ana web sayfası."""
    return render_template('index.html')


@app.route('/api/detect', methods=['POST'])
def api_detect():
    """Webcam'den gelen kareyi alıp YOLO26 ile algılayıp JSON döndürür."""
    data = request.get_json(silent=True)
    if not data or 'image' not in data:
        return jsonify({'error': 'Resim verisi eksik'}), 400

    try:
        # Base64 string'den görüntüye çevir
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Görüntü çözülemedi'}), 400

        conf = float(data.get('confidence', 0.40))
        result = process_frame(frame, conf_threshold=conf)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# WebSocket Desteği
@socketio.on('detect_frame')
def handle_detect_frame(data):
    """WebSocket üzerinden hızlı kare algılama."""
    try:
        img_data = data.get('image', '')
        if ',' in img_data:
            img_data = img_data.split(',')[1]

        image_bytes = base64.b64decode(img_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is not None:
            conf = float(data.get('confidence', 0.40))
            result = process_frame(frame, conf_threshold=conf)
            emit('detection_result', result)
    except Exception as e:
        emit('detection_error', {'error': str(e)})


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 7860))
    print("\n" + "=" * 60)
    print("  🚀 YOLO26 Web Nesne Algılama Sunucusu Başlatılıyor...")
    print(f"  🌐 Yerel Adres : http://localhost:{port}")
    print(f"  📱 Ağ Adresi   : http://0.0.0.0:{port} (Aynı WiFi'daki cihazlar)")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
