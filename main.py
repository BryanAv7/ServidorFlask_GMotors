from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ExifTags
import io
from fast_plate_ocr import LicensePlateRecognizer

# ======================================================
# CONFIG
# ======================================================
MODELO_PATH = r"C:/Users/USUARIO/Desktop/prototipoFlask/models/best.pt"
CONF_THRESHOLD = 0.25

# ======================================================
# FLASK
# ======================================================
app = Flask(__name__)

print("Cargando YOLO...")
model = YOLO(MODELO_PATH)
model.to("cpu")
print("✓ YOLO cargado")

print("Cargando OCR...")
ocr_model = LicensePlateRecognizer('cct-xs-v1-global-model')
print("✓ OCR cargado")


# ======================================================
# LECTURA IMAGEN (EXIF)
# ======================================================
def leer_imagen_con_exif(file):
    image = Image.open(io.BytesIO(file.read()))

    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = image._getexif()
        if exif is not None:
            orient = exif.get(orientation)

            if orient == 3:
                image = image.rotate(180, expand=True)
            elif orient == 6:
                image = image.rotate(270, expand=True)
            elif orient == 8:
                image = image.rotate(90, expand=True)
    except:
        pass

    image = image.convert("RGB")
    np_img = np.array(image)
    return np_img[:, :, ::-1]


# ======================================================
# OBB
# ======================================================
def ordenar_puntos_obb(pts):
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    pts = pts[np.argsort(angles)]
    idx = np.argmin(pts[:, 0] + pts[:, 1])
    return np.roll(pts, -idx, axis=0)


def extraer_placa(imagen, bbox_obb):
    pts = ordenar_puntos_obb(np.array(bbox_obb).reshape(4, 2).astype(np.float32))

    w = int(max(np.linalg.norm(pts[0]-pts[1]), np.linalg.norm(pts[2]-pts[3])))
    h = int(max(np.linalg.norm(pts[1]-pts[2]), np.linalg.norm(pts[3]-pts[0])))

    w = max(w, 80)
    h = max(h, 30)

    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)

    return cv2.warpPerspective(imagen, M, (w,h))


# ======================================================
# OCR
# ======================================================
def leer_placa(placa_img):
    try:
        resultado = ocr_model.run(placa_img)

        if not resultado:
            return ""

        return resultado[0].plate

    except Exception as e:
        print("Error OCR:", e)
        return ""


# ======================================================
# ENDPOINT
# ======================================================
@app.route("/api/ocr/placa", methods=["POST"])
def detectar_placa():

    if 'image' not in request.files:
        return jsonify({"error": "Imagen no enviada"}), 400

    file = request.files['image']
    imagen = leer_imagen_con_exif(file)

    if imagen is None:
        return jsonify({"placa": ""})

    h, w = imagen.shape[:2]
    if max(h, w) > 1600:
        scale = 1600 / max(h, w)
        imagen = cv2.resize(imagen, (int(w*scale), int(h*scale)))

    resultados = model.predict(imagen, conf=CONF_THRESHOLD, imgsz=640, verbose=False)

    for r in resultados:
        if r.obb is None:
            continue

        for bbox in r.obb.xyxyxyxy:
            placa_img = extraer_placa(imagen, bbox.cpu().numpy())
            texto = leer_placa(placa_img)

            if texto:
                return jsonify({"placa": texto})

    return jsonify({"placa": ""})


# ======================================================
# RUN
# ======================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)