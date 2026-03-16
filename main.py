from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import re
import requests
import tempfile
import os
from typing import Tuple
from PIL import Image, ExifTags
import io

# ======================================================
# CONFIGURACIÓN
# ======================================================
MODELO_PATH = r"C:/Users/USUARIO\Desktop/prototipoFlask/models/best.pt"
CONF_THRESHOLD = 0.25

OCR_API_KEY = "K8715073538895"
OCR_API_URL = "https://api.ocr.space/parse/image"

# ======================================================
# FLASK
# ======================================================
app = Flask(__name__)

print("Cargando YOLO...")
model = YOLO(MODELO_PATH)
model.to("cpu")
print("✓ YOLO cargado")

# ======================================================
# LECTURA DE IMAGEN CON EXIF (CÁMARA)
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
# FUNCIONES OCR Y PROCESAMIENTO DE TEXTO
# ======================================================

def ocr_space_imagen(imagen: np.ndarray, engine: int = 2) -> Tuple[str, float]:
    try:
        if len(imagen.shape) == 2:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)

        _, img_encoded = cv2.imencode('.jpg', imagen, [cv2.IMWRITE_JPEG_QUALITY, 95])

        payload = {
            'apikey': OCR_API_KEY,
            'language': 'eng',
            'OCREngine': engine,
            'filetype': 'JPG',
            'scale': True,
            'detectOrientation': True,
            'isTable': False,
        }

        files = {'file': ('placa.jpg', img_encoded.tobytes(), 'image/jpeg')}
        response = requests.post(OCR_API_URL, files=files, data=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            if not result.get('IsErroredOnProcessing'):
                parsed = result.get('ParsedResults', [])
                if parsed:
                    texto = parsed[0].get('ParsedText', '').strip()
                    return texto, 0.9 if texto else 0.0

        return "", 0.0
    except Exception:
        return "", 0.0


def filtrar_palabras_no_deseadas(texto: str) -> str:
    if not texto:
        return ""

    palabras_prohibidas = [
        'ECUADOR', 'ECUDOR', 'ECUADR', 'ECUADO',
        'ANT', 'DAYTONA', 'MOTORCYCLE', 'MOTO',
        'REPUBLIC', 'REPUBLICA', 'DEL',
        'PLACA', 'PLATE', 'LICENSE', 'PROVISIONAL',
    ]

    texto = texto.upper()
    for palabra in palabras_prohibidas:
        texto = texto.replace(palabra, '')

    return texto.strip()

# funcion para extraer candidatos a placa del texto OCR
def extraer_candidatos_placa(texto: str) -> list:
    if not texto:
        return []

    texto = filtrar_palabras_no_deseadas(texto)
    texto = texto.replace('\n', ' ').replace('\r', ' ')

    candidatos = []
    palabras = texto.split()

    for palabra in palabras:
        palabra = ''.join(c for c in palabra if c.isalnum()).upper()
        if 5 <= len(palabra) <= 7:
            if any(c.isalpha() for c in palabra) and any(c.isdigit() for c in palabra):
                candidatos.append(palabra)

    texto_continuo = ''.join(c for c in texto if c.isalnum()).upper()

    patrones = [
        r'[A-Z]{2}\d{3}[A-Z]',
        r'[A-Z]{2}\d{4}',
        r'[A-Z]{3}\d{3}',
    ]

    for patron in patrones:
        for m in re.finditer(patron, texto_continuo):
            candidatos.append(m.group())

    return list(dict.fromkeys(candidatos))


def formatear_placa(texto: str) -> str:
    return texto


def es_placa_valida(texto: str) -> bool:
    patrones = [
        r'^[A-Z]{2}\d{3}[A-Z]$',
        r'^[A-Z]{2}\d{4}$',
        r'^[A-Z]{3}\d{3}$',
    ]
    return any(re.match(p, texto) for p in patrones)

# funcion para limpiar el texto de la placa
def limpiar_texto_placa(texto: str) -> str:
    if not texto:
        return ""

    candidatos = extraer_candidatos_placa(texto)

    for c in candidatos:
        if es_placa_valida(c):
            return formatear_placa(c)

    correcciones = {'O': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'B': '8'}
    for c in candidatos:
        corregido = ""
        for i, ch in enumerate(c):
            if 2 <= i <= 5 and ch in correcciones:
                corregido += correcciones[ch]
            else:
                corregido += ch
        if es_placa_valida(corregido):
            return formatear_placa(corregido)

    return ""

# funcion para ordenar los puntos de la caja OBB
def ordenar_puntos_obb(pts: np.ndarray) -> np.ndarray:
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    pts = pts[np.argsort(angles)]
    idx = np.argmin(pts[:, 0] + pts[:, 1])
    return np.roll(pts, -idx, axis=0)

# función para extraer la placa usando la caja OBB
def extraer_placa(imagen: np.ndarray, bbox_obb: np.ndarray) -> np.ndarray:
    pts = ordenar_puntos_obb(np.array(bbox_obb).reshape(4, 2).astype(np.float32))

    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[1] - pts[2]), np.linalg.norm(pts[3] - pts[0])))

    w = max(w, 50)
    h = max(h, 20)

    dst = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)

    return cv2.warpPerspective(imagen, M, (w, h), flags=cv2.INTER_CUBIC)


def leer_placa(placa_img: np.ndarray) -> str:
    h, w = placa_img.shape[:2]

    if w < 300:
        scale = 400 / w
        placa_img = cv2.resize(placa_img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

    texto, _ = ocr_space_imagen(placa_img, engine=2)
    texto_limpio = limpiar_texto_placa(texto)
    if texto_limpio:
        return texto_limpio

    texto, _ = ocr_space_imagen(placa_img, engine=1)
    return limpiar_texto_placa(texto)


# ======================================================
# ENDPOINT FLASK
# ======================================================
@app.route("/api/ocr/placa", methods=["POST"])
def detectar_placa():
    if 'image' not in request.files:
        return jsonify({"error": "Imagen no enviada"}), 400

    file = request.files['image']
    print("Nombre:", file.filename)
    print("Content-Type:", file.content_type)

    imagen = leer_imagen_con_exif(file)

    if imagen is None:
        return jsonify({"placa": ""})

    # Normalizar tamaño si viene la img muy grande
    h, w = imagen.shape[:2]
    if max(h, w) > 1600:
        scale = 1600 / max(h, w)
        imagen = cv2.resize(imagen, (int(w * scale), int(h * scale)))

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
