"""
YOLO26 Gerçek Zamanlı Nesne Algılama - Webcam & Telefon Kamera Uygulaması
==========================================================================
Bu uygulama webcam veya telefon kamerası üzerinden canlı görüntü alır ve
YOLO26 modeli ile görüntüdeki nesneleri tespit edip ekrana yazdırır.

Kullanım:
    python main.py              → Bilgisayar webcam'ini kullanır
    python main.py --phone      → Telefon kamerasını kullanır (IP Webcam)
    python main.py --ip IP_ADRESI  → Özel IP adresi belirler

Telefon Kamerası Kurulumu (Samsung / Android):
    1. Google Play Store'dan "IP Webcam" uygulamasını indirin
       (Geliştirici: Pavel Khlebovich)
    2. Uygulamayı açın ve en altta "Start server" butonuna basın
    3. Ekranda gösterilen IP adresini not edin (örn: 192.168.1.105:8080)
    4. Bu scripti çalıştırın: python main.py --phone
    5. IP adresini girin veya: python main.py --ip 192.168.1.105:8080

Kontroller:
    - 'q' tuşu: Uygulamayı kapatır
    - 's' tuşu: Mevcut kareyi ekran görüntüsü olarak kaydeder
"""


import argparse
import cv2
import sys
import time
from ultralytics import YOLO


def get_camera_source(args):
    """Kamera kaynağını belirler (webcam veya telefon kamerası)."""

    # --ip parametresi verildiyse doğrudan kullan
    if args.ip:
        ip = args.ip
        if not ip.startswith("http"):
            ip = f"http://{ip}"
        url = f"{ip}/video"
        print(f"[INFO] Telefon kamerasına bağlanılıyor: {url}")
        return url

    # --phone parametresi verildiyse IP adresi iste
    if args.phone:
        print("=" * 60)
        print("  📱 TELEFON KAMERASI MODU")
        print("=" * 60)
        print()
        print("  Adımlar:")
        print("  1. Telefonunuza 'IP Webcam' uygulamasını kurun")
        print("  2. Telefon ve bilgisayar AYNI WiFi ağında olmalı")
        print("  3. IP Webcam uygulamasında 'Start server' yapın")
        print("  4. Ekrandaki IP adresini aşağıya girin")
        print()
        ip_input = input("  IP adresini girin (örn: 192.168.1.105:8080): ").strip()

        if not ip_input:
            print("[HATA] IP adresi boş olamaz!")
            sys.exit(1)

        if not ip_input.startswith("http"):
            ip_input = f"http://{ip_input}"

        url = f"{ip_input}/video"
        print(f"\n[INFO] Telefon kamerasına bağlanılıyor: {url}")
        return url

    # Varsayılan: bilgisayar webcam'i
    print("[INFO] Bilgisayar webcam'i kullanılıyor...")
    return 0


def main():
    # ── Argüman Ayrıştırıcı ───────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="YOLO26 ile Gerçek Zamanlı Nesne Algılama"
    )
    parser.add_argument(
        "--phone",
        action="store_true",
        help="Telefon kamerasını kullan (IP Webcam uygulaması gerekli)",
    )
    parser.add_argument(
        "--ip",
        type=str,
        default=None,
        help="Telefon IP adresi (örn: 192.168.1.105:8080)",
    )
    args = parser.parse_args()

    # ── YOLO26 Modelini Yükle ──────────────────────────────────────────
    # İlk çalıştırmada model ağırlıkları otomatik olarak indirilir (~6MB)
    print("[INFO] YOLO26 modeli yükleniyor...")
    model = YOLO("yolo26n.pt")  # yolo26n = nano (en hızlı), yolo26s/m/l/x de kullanılabilir
    print("[INFO] Model başarıyla yüklendi!")

    # ── Kamera Bağlantısı ──────────────────────────────────────────────
    source = get_camera_source(args)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        if isinstance(source, str):
            print(f"[HATA] Telefon kamerasına bağlanılamadı!")
            print(f"       Kontrol edin:")
            print(f"       - IP Webcam uygulaması çalışıyor mu?")
            print(f"       - Telefon ve PC aynı WiFi ağında mı?")
            print(f"       - IP adresi doğru mu? ({source})")
        else:
            print("[HATA] Webcam açılamadı! Lütfen kameranın bağlı olduğundan emin olun.")
        sys.exit(1)

    # Webcam modunda çözünürlüğü ayarla
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    camera_type = "Telefon Kamerası" if isinstance(source, str) else "Webcam"
    print(f"[INFO] {camera_type} başarıyla açıldı!")
    print("[INFO] Çıkmak için 'q' tuşuna basın | Ekran görüntüsü için 's' tuşuna basın")
    print("-" * 60)

    # FPS hesaplama değişkenleri
    prev_time = time.time()
    fps = 0

    # ── Ana Döngü ──────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[HATA] Kameradan görüntü alınamadı!")
            break

        # FPS hesapla
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        # ── YOLO26 ile Nesne Algılama ──────────────────────────────────
        results = model(frame, verbose=False)

        # Algılanan nesneleri işle
        detected_objects = []

        for result in results:
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # Sınırlayıcı kutu koordinatları
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Güven skoru ve sınıf adı
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]

                    # Güven skoru %40'ın üzerindeyse göster
                    if confidence > 0.40:
                        detected_objects.append(
                            f"{class_name}: %{confidence * 100:.1f}"
                        )

                        # Sınırlayıcı kutu çiz
                        color = _get_color(class_id)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # Etiket arka planı (kutunun İÇİNDE, nesnenin üzerinde)
                        label = f"{class_name} {confidence:.2f}"
                        label_size, _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                        )
                        # Etiket kutunun içinde üst kısımda
                        label_y = y1 + label_size[1] + 8
                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x1 + label_size[0] + 8, y1 + label_size[1] + 16),
                            color,
                            -1,
                        )
                        cv2.putText(
                            frame,
                            label,
                            (x1 + 4, label_y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                        )

        # Konsola algılanan nesneleri yazdır
        if detected_objects:
            print(
                f"[FPS: {fps:.1f}] Algılanan nesneler: {', '.join(detected_objects)}"
            )

        # FPS bilgisini görüntüye ekle
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Toplam algılanan nesne sayısını göster
        cv2.putText(
            frame,
            f"Nesne: {len(detected_objects)}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        # Kamera kaynağını göster
        cv2.putText(
            frame,
            camera_type,
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        # ── Görüntüyü Göster ──────────────────────────────────────────
        cv2.imshow("YOLO26 Nesne Algilama", frame)

        # Klavye kontrolleri
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\n[INFO] Uygulama kapatılıyor...")
            break
        elif key == ord("s"):
            filename = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[INFO] Ekran görüntüsü kaydedildi: {filename}")

    # ── Temizlik ───────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Uygulama başarıyla kapatıldı.")


def _get_color(class_id: int) -> tuple:
    """Her sınıf için tutarlı ve farklı bir renk döndürür."""
    colors = [
        (0, 255, 0),    # Yeşil
        (255, 0, 0),    # Mavi
        (0, 0, 255),    # Kırmızı
        (255, 255, 0),  # Cyan
        (0, 255, 255),  # Sarı
        (255, 0, 255),  # Magenta
        (128, 255, 0),  # Lime
        (255, 128, 0),  # Açık mavi
        (0, 128, 255),  # Turuncu
        (255, 0, 128),  # Pembe
    ]
    return colors[class_id % len(colors)]


if __name__ == "__main__":
    main()
