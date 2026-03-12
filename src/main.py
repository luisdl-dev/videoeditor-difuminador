# video_blur_editor.py - Versión con eliminación progresiva robusta y blur persistente
# =============================================================================
# NOTA TÉCNICA (OpenCV / Trackers / Desfase de frames)
#
# Hallazgo (nov-2025):
# - Apareció un “desfase” de detecciones y errores con CSRT/KCF después de
#   reinstalar OpenCV. Sin tocar el código, volvió a funcionar al estabilizar
#   la instalación del paquete.
#
# Causas más probables:
# 1) Mezcla de paquetes OpenCV:
#    - opencv-python (básico, SIN trackers), opencv-python-headless (sin GUI/codec),
#      y opencv-contrib-python (incluye trackers y cv2.legacy).
# 2) Cambio de backend de VideoCapture (FFmpeg/DirectShow/MediaFoundation):
#    - Puede provocar off-by-one al usar set(POS_FRAMES)+grab()+retrieve().
# 3) Diferencias de resolución/rotación entre el video actual y los JSON guardados.
#
# Recomendación mínima (mantener estable):
# - Usar SOLO contrib y fijar versión:
#     pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
#     pip install --no-cache-dir opencv-contrib-python==4.8.1.78
#   (Alternativa muy estable: ==4.7.0.72)
# - Evitar instalar a la vez opencv-python y opencv-contrib-python.
# - Evitar la variante -headless en este proyecto.
#
# (Opcional, para blindaje futuro)
# A) Autochequeo al iniciar:
#     import cv2
#     print(f"[OPENCV] v={cv2.__version__} | legacy={hasattr(cv2,'legacy')} "
#           f"| CSRT={hasattr(getattr(cv2,'legacy',cv2),'TrackerCSRT_create')}")
#
# B) Lectura exacta de frames si algún backend vuelve a desfasar:
#     def _read_frame_exact(cap, idx):
#         idx = max(0, int(idx))
#         if idx > 0:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)
#             cap.read()
#         else:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#         return cap.read()
#
# C) Guardar metadatos en los JSON (fps, W, H, versión de OpenCV) para detectar
#    y compensar automáticamente escalas/orientación al cargar.
#
# Resumen:
# - El problema se resolvió al tener la rueda correcta de OpenCV.
# - Para no repetirlo: anclar opencv-contrib-python, evitar mezclas y (si se
#   desea) usar _read_frame_exact + metadatos para robustez total.
# =============================================================================

# ================================
# Versión 960 - 2026-02-12
# Se ajustó el reescalado a letterbox correcto para 960x960.
# Ejemplo: 1920x1080 ahora se escala proporcionalmente a 960x540
# y se agrega padding (114) arriba y abajo hasta completar 960x960,
# evitando cualquier deformación de la imagen.
# Se corrigió la reversión de coordenadas (remoción de padding + división por r).
# No fue necesario modificar otros métodos del pipeline.
# ================================


import sys
import cv2
print(f"[OPENCV] v={cv2.__version__} | legacy={hasattr(cv2, 'legacy')} | CSRT={hasattr(getattr(cv2, 'legacy', cv2), 'TrackerCSRT_create')}")
import numpy as np
import os
import json
import math

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog,
    QHBoxLayout, QProgressBar, QLineEdit, QMessageBox, QSizePolicy, 
    QCheckBox, QDesktopWidget, QDialog, QGraphicsOpacityEffect, QSlider, QHBoxLayout, 
    QTimeEdit
)

#<--PARA EFECTOS
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont, QRadialGradient, QBrush, QPainterPath, QFontMetrics
from PyQt5.QtCore import QPropertyAnimation, QTimer, Qt, QThread, pyqtSignal, QPoint, QRectF, QRect, QTime

from datetime import datetime

import torch
import torch_directml
from ultralytics import YOLO
import time  # ← ¡Agrega esta línea!

try:
    from ultralytics.utils import ops  # v8 reciente
except Exception:
    from ultralytics.yolo.utils import ops  # fallback en builds anteriores

# Inicializar el dispositivo DML (GPU AMD)
#DML = torch_directml.device()

# Cargar modelo YOLO una sola vez y moverlo a GPU AMD

# model = YOLO("best.pt")
# model.model.to(DML).float().eval()
# print("[IA] Modelo YOLO cargado en GPU DirectML:", DML)
# # 🔹 PRIMERA PASADA "dummy" para inicializar DirectML (evita cuelgue posterior)
# dummy = torch.zeros((1, 3, 640, 640), dtype=torch.float32).to(DML)
# print("[IA] Inicializando DirectML con pasada de prueba...")
# with torch.no_grad():
#     _ = model.model(dummy)
# print("[IA] DirectML inicializado correctamente.")

# === Helper compatible con estructura de YOLO.predict() ===
class DummyBox:
    def __init__(self, xyxy, conf):
        self.xyxy = [torch.tensor(xyxy, dtype=torch.float32)]
        self.conf = [torch.tensor(conf, dtype=torch.float32)]

class DummyResult:
    def __init__(self, boxes):
        self.boxes = boxes

# arriba del archivo (una vez):
import math
try:
    from ultralytics.utils import ops  # v8 reciente
except Exception:
    from ultralytics.yolo.utils import ops  # fallback en builds anteriores

def inferir_result_dml(frames_bgr, conf=0.5, iou_thres=0.45, max_det=300):
    """Inferencia en GPU AMD (DirectML) para una lista de frames. Devuelve lista de listas de rects."""
    import cv2, torch
    from ultralytics.utils import ops
    global DML

    resultados_batch = []

    try:
        for frame_bgr in frames_bgr:
            if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
                resultados_batch.append([])
                continue

            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h0, w0 = img_rgb.shape[:2]

            # Redimensionar a 640x640
            #resized = cv2.resize(img_rgb, (960, 960))
            # --- Letterbox manual a 960 ---
            

            target = 960
            r = min(target / w0, target / h0)
            new_w, new_h = round(w0 * r), round(h0 * r)

            resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            pad_w = target - new_w
            pad_h = target - new_h
            pad_left = pad_w // 2
            pad_top = pad_h // 2

            resized = cv2.copyMakeBorder(
                resized,
                pad_top, pad_h - pad_top,
                pad_left, pad_w - pad_left,
                cv2.BORDER_CONSTANT,
                value=(114, 114, 114)  # mismo padding que YOLO
            )

            t = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            t = t.to(DML)

            # Inferencia DirectML
            with torch.no_grad():
                preds = model.model(t)

            if isinstance(preds, (list, tuple)):
                preds = preds[0]

            preds = preds.detach().cpu()
            dets = ops.non_max_suppression(preds, conf_thres=conf, iou_thres=iou_thres, max_det=max_det)

            rects = []
            if len(dets) and dets[0] is not None and len(dets[0]) > 0:
                det = dets[0]
                for *xyxy, conf, cls in det.tolist():
                    x1, y1, x2, y2 = xyxy

                    # Quitar padding
                    x1 -= pad_left
                    x2 -= pad_left
                    y1 -= pad_top
                    y2 -= pad_top

                    # Reescalar al tamaño original
                    x1 /= r
                    x2 /= r
                    y1 /= r
                    y2 /= r

                    # Convertir a int
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    # Clampear a límites
                    x1 = max(0, min(w0 - 1, x1))
                    x2 = max(0, min(w0 - 1, x2))
                    y1 = max(0, min(h0 - 1, y1))
                    y2 = max(0, min(h0 - 1, y2))

                    w = x2 - x1
                    h = y2 - y1

                    if w > 0 and h > 0:
                        rects.append((x1, y1, w, h))


            resultados_batch.append(rects)

    except Exception as e:
        import traceback
        print(f"[DML][Error batch] {e}")
        traceback.print_exc()
        # si hay error global, llenar vacíos
        resultados_batch = [[] for _ in frames_bgr]

    return resultados_batch



def inferir_result_dml_antiguo(frame_bgr, conf_min=0.5, iou_thres=0.45, max_det=300):
    """
    Inferencia DirectML nativa (GPU AMD) con prints de diagnóstico.
    Permite confirmar en consola si realmente se está usando la GPU.
    """
    import math, time
    from ultralytics.utils import ops

    t0 = time.time()

    try:
        # --- Verificar dispositivo ---
        print("\n[IA][DML] === Diagnóstico de dispositivo ===")
        print(f"[IA][DML] torch_directml.device_count(): {torch_directml.device_count()}")
        print(f"[IA][DML] Nombre GPU: {torch_directml.device_name(0)}")
        print(f"[IA][DML] Modelo cargado en: {next(model.model.parameters()).device}")

        # --- Preprocesamiento ---
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h0, w0 = img_rgb.shape[:2]
        stride = 32
        new_h = int(math.ceil(h0 / stride) * stride)
        new_w = int(math.ceil(w0 / stride) * stride)
        if new_h != h0 or new_w != w0:
            img_rgb = cv2.copyMakeBorder(
                img_rgb, 0, new_h - h0, 0, new_w - w0,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

        # --- Tensor en GPU ---
        t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        t = t.to(DML).float()
        print(f"[IA][DML] Tensor shape: {tuple(t.shape)}, dtype: {t.dtype}, device: {t.device}")

        # --- Inferencia ---
        with torch.no_grad():
            preds = model.model(t)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        # --- Postprocesamiento ---
        preds = preds.detach().cpu()
        dets = ops.non_max_suppression(preds, conf_thres=conf_min, iou_thres=iou_thres, max_det=max_det)

        dummy_boxes = []
        if len(dets) and dets[0] is not None and len(dets[0]):
            det = dets[0]
            for *xyxy, conf, cls in det.tolist():
                x1, y1, x2, y2 = map(int, xyxy)
                dummy_boxes.append(DummyBox([x1, y1, x2, y2], conf))
            print(f"[IA][DML] Detecciones: {len(dummy_boxes)}")
        else:
            print("[IA][DML] Sin detecciones en este frame.")

        print(f"[IA][DML] Tiempo total de inferencia: {time.time()-t0:.2f}s\n")
        return [DummyResult(dummy_boxes)]

    except Exception as e:
        print("[IA][ERROR][DML]", e)
        import traceback
        traceback.print_exc()
        return [DummyResult([])]


LOG_FILE = "log.txt"
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"[{datetime.now()}] Iniciando aplicación...\n")

def log(msg):
    # Desactivar log si app_instance aún no existe o DEBUG_MODE está desactivado
    if "app_instance" in globals():
        try:
            if hasattr(app_instance, "DEBUG_MODE") and not app_instance.DEBUG_MODE:
                return
        except Exception:
            return
    else:
        return
        
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] {msg}\n")

def log_visual_gui(msg):
    #Desactivar log si app_instance aún no existe o DEBUG_MODE está desactivado
    if "app_instance" in globals():
        try:
            if hasattr(app_instance, "DEBUG_MODE") and not app_instance.DEBUG_MODE:
                return
        except Exception:
            return
    else:
        return
        
    with open("log_visual_gui.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

IA_CLICK_LOG_FILE = "log_ia_click_debug.txt"
with open(IA_CLICK_LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"[{datetime.now()}] Log de clic derecho sobre blur IA\n")

def log_ia_click(msg):
    # Desactivar log si app_instance aún no existe o DEBUG_MODE está desactivado
    if "app_instance" in globals():
        try:
            if hasattr(app_instance, "DEBUG_MODE") and not app_instance.DEBUG_MODE:
                return
        except Exception:
            return
    else:
        return
        
    with open(IA_CLICK_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] {msg}\n")
        
IA_POST_LOG_FILE = "log_ia_post.txt"
_ia_post_log_initialized = False  # Indicador de borrado único
def log_ia_post(msg):
     #Desactivar log si app_instance aún no existe o DEBUG_MODE está desactivado
    if "app_instance" in globals():
        try:
            if hasattr(app_instance, "DEBUG_MODE") and not app_instance.DEBUG_MODE:
                return
        except Exception:
            return
    else:
        return
        
    global _ia_post_log_initialized
    # Borrar el archivo al primer uso
    if not _ia_post_log_initialized:
        with open(IA_POST_LOG_FILE, "w", encoding="utf-8") as f:
            f.write("")  # Vaciar contenido
        _ia_post_log_initialized = True

    # Agregar nueva línea
    with open(IA_POST_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] {msg}\n")

def rects_similares(r1, r2, tolerancia=100):#antiguo
    x1c = (r1[0] + r1[2] / 2)  # Centro x del rectángulo 1
    y1c = (r1[1] + r1[3] / 2)  # Centro y del rectángulo 1
    x2c = (r2[0] + r2[2] / 2)  # Centro x del rectángulo 2
    y2c = (r2[1] + r2[3] / 2)  # Centro y del rectángulo 2
    return abs(x1c - x2c) <= tolerancia and abs(y1c - y2c) <= tolerancia

def log_salto_frames(mensaje):
    # Desactivar log si app_instance aún no existe o DEBUG_MODE está desactivado
    if "app_instance" in globals():
        try:
            if hasattr(app_instance, "DEBUG_MODE") and not app_instance.DEBUG_MODE:
                return
        except Exception:
            return
    else:
        return
        
    with open("debuglog_salto_frames.txt", "a") as f:
        f.write(mensaje + "\n")    

def resetear_log_salto_frames():
    # Desactivar log si app_instance aún no existe o DEBUG_MODE está desactivado
    if "app_instance" in globals():
        try:
            if hasattr(app_instance, "DEBUG_MODE") and not app_instance.DEBUG_MODE:
                return
        except Exception:
            return
    else:
        return
        
    with open("debuglog_salto_frames.txt", "w") as f:
        f.write("=== Nuevo Log de Sesión ===\n")        

def rect_to_key(rect, precision=10):
        # Redondea las coordenadas para crear una clave tipo hash
        return tuple((round(v / precision) * precision) for v in rect)

def aplicar_blur_circular(frame, rect, kernel=51):
    if len(rect) >= 4:
        x, y, w, h = rect[:4]
    #else:
        #continue
    roi = frame[y:y+h, x:x+w]
    if roi.size > 0:
        
        k = kernel
        # Asegura que sea impar y mínimo 3
        if k % 2 == 0:
            k += 1
        k = max(3, k)
        blurred = cv2.GaussianBlur(roi, (k, k), 0)
        #blurred = cv2.GaussianBlur(roi, (51, 51), 0)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (w // 2, h // 2), min(w, h) // 2, 255, -1)
        mask_3ch = cv2.merge([mask]*3)
        roi_result = np.where(mask_3ch == 255, blurred, roi)
        frame[y:y+h, x:x+w] = roi_result
        
def aplicar_blur_capsula(frame, rect, escala=1.2, kernel=51):
    import numpy as np
    if len(rect) >= 4:
        x, y, w, h = rect[:4]
    else:
        return  # ←  usar return, ya que no hay bucle

    # Expande opcionalmente el rectángulo
    exp_x = int(w * (escala - 1) / 2)
    exp_y = int(h * (escala - 1) / 2)

    x1 = max(0, x - exp_x)
    y1 = max(0, y - exp_y)
    x2 = min(frame.shape[1], x + w + exp_x)
    y2 = min(frame.shape[0], y + h + exp_y)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return

    # Asegura que el kernel sea impar y al menos 3
    k = max(3, kernel if kernel % 2 == 1 else kernel + 1)

    blurred = cv2.GaussianBlur(roi, (k, k), 0)

    mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
    center = (roi.shape[1] // 2, roi.shape[0] // 2)
    axes = (roi.shape[1] // 2, roi.shape[0] // 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    mask_3ch = cv2.merge([mask]*3)
    roi_result = np.where(mask_3ch == 255, blurred, roi)

    frame[y1:y2, x1:x2] = roi_result


 #IOU
def calcular_iou(r1, r2):
        """
        Calcula la Intersección sobre Unión (IoU) entre dos rectángulos.
        r1, r2: Tuplas/listas de la forma (x, y, w, h) o (x, y, w, h, conf).
        Retorna: Valor de IoU (0.0 a 1.0).
        """
        x1, y1, w1, h1 = r1[:4]
        x2, y2, w2, h2 = r2[:4]
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0
    #RECS
def rects_similares_iou(r1, r2, umbral_iou=0.5):
        """
        Determina si dos rectángulos son similares basándose en IoU.
        r1, r2: Tuplas/listas de la forma (x, y, w, h) o (x, y, w, h, conf).
        umbral_iou: Umbral mínimo de IoU para considerar los rectángulos similares.
        Retorna: True si IoU >= umbral_iou, False de lo contrario.
        """
        return calcular_iou(r1, r2) >= umbral_iou

def _bbox_center_wh(r):
    x, y, w, h = r[:4]
    return (x + w/2.0, y + h/2.0, w, h)

def _match_rects_greedy(rectsA, rectsB, iou_thr=0.60, size_tol=0.15, shift_px=20):
    """Devuelve (num_matches, nuevos_en_B, coverageA) usando emparejamiento simple por IoU."""
    usedB = set()
    matches = 0
    for ra in rectsA:
        best_iou, best_j = 0.0, -1
        for j, rb in enumerate(rectsB):
            if j in usedB:
                continue
            iou = calcular_iou(ra, rb)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0:
            xa, ya, wa, ha = ra[:4]
            xb, yb, wb, hb = rectsB[best_j][:4]
            ca = _bbox_center_wh(ra)
            cb = _bbox_center_wh(rectsB[best_j])
            # checks extra: tamaño y desplazamiento
            size_ok = abs(wb - wa)/max(1.0, wa) <= size_tol and abs(hb - ha)/max(1.0, ha) <= size_tol
            shift_ok = (abs(cb[0] - ca[0]) <= shift_px) and (abs(cb[1] - ca[1]) <= shift_px)
            if best_iou >= iou_thr and size_ok and shift_ok:
                matches += 1
                usedB.add(best_j)
    coverageA = matches / max(1, len(rectsA))
    nuevosB = len(rectsB) - len(usedB)  # rostros “nuevos” en B sin emparejar
    return matches, nuevosB, coverageA


# Llama esta función al inicio de tu programa/principal/clase
resetear_log_salto_frames()


class MinuteProcessor(QThread):
    minuto_procesado = pyqtSignal(int, dict)
    progreso_actualizado = pyqtSignal(int)

    def __init__(self, video_path, minuto, fps, conf=0.5):
        super().__init__()
        self.video_path = video_path
        self.minuto = minuto
        self.fps = fps
        self.conf = conf

    def run(self):
        log(f"[HILO] Procesando minuto {self.minuto}...")
        cap = cv2.VideoCapture(self.video_path)
        inicio = int(self.minuto * 10 * self.fps)
        fin = int((self.minuto + 1) * 10 * self.fps)
        #inicio = int(self.minuto * 60 * self.fps)
        #fin = int((self.minuto + 1) * 60 * self.fps)
        total = fin - inicio
        cap.set(cv2.CAP_PROP_POS_FRAMES, inicio)

        resultado = {}
        for i, idx in enumerate(range(inicio, fin)):
            ret, frame = cap.read()
            if not ret:
                break
            import torch
            import torch_directml

            dml = torch_directml.device()

            # Convertir el frame a tensor en GPU AMD
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float().to(dml) / 255.0

            # Mover modelo a GPU AMD
            model.model.to(DML).float().eval()

            # Ejecutar inferencia directamente sobre la GPU AMD
            with torch.no_grad():
                preds = model.model(img)

            # Decodificar resultados al formato YOLO
            rects = inferir_result_dml(frame, conf_min=self.conf)
            
            resultado[idx] = rects
            progreso = int((i / total) * 100)
            self.progreso_actualizado.emit(progreso)

        cap.release()
        log(f"[HILO] Minuto {self.minuto} procesado.")
        self.minuto_procesado.emit(self.minuto, resultado)

#IA Este hilo es el que realmente hace el trabajo frame por frame:
class IAFrameProcessor(QThread):
    progreso_actualizado = pyqtSignal(int)
    procesamiento_terminado = pyqtSignal(dict)

    def __init__(self, video_path, inicio_frame, fin_frame, fps, conf=0.5):
        super().__init__()
        self.video_path = video_path
        self.inicio_frame = int(inicio_frame)
        self.fin_frame = int(fin_frame)
        self.fps = int(fps)
        self.conf = conf

    def run_backup(self):
        import math, time, gc

        t_inicio = time.time()
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.inicio_frame)

        # --- dentro de IAFrameProcessor.run(...) [versión GPU] ---
        
        #WINDOW = 10
        WINDOW = 32
        frame_idx = self.inicio_frame
        resultado = {}

        def _store(idx, rects):
            if rects:
                resultado[idx] = rects

        total_frames = max(0, self.fin_frame - self.inicio_frame)
        total_windows = max(1, math.ceil(total_frames / WINDOW))
        windows_done = 0

        print("[IA][GPU] Iniciando procesamiento DirectML (sin fallback CPU).")

        try:
            while frame_idx < self.fin_frame:
                buffer = []
                first_idx = frame_idx
                last_idx = min(self.fin_frame - 1, frame_idx + WINDOW)
                current_idx = first_idx

                while current_idx <= last_idx:
                    ret, frm = cap.read()
                    if not ret or frm is None:
                        break
                    buffer.append((current_idx, frm))
                    current_idx += 1

                if not buffer:
                    break

                # === INFERENCIA DIRECTML ===
                batch_np = [f for (_, f) in buffer]
                dets_batch = inferir_result_dml(batch_np, conf=self.conf)  # tu función DML obligatoria

                # === REGISTRO ===
                for (idx, _), dets in zip(buffer, dets_batch):
                    rects = []
                    for d in dets:
                        if len(d) >= 4:
                            x, y, w, h = map(int, d[:4])
                            if w > 0 and h > 0:
                                if len(d) >= 5:
                                    rects.append((x, y, w, h, float(d[4])))
                                else:
                                    rects.append((x, y, w, h))
                    _store(idx, rects)

                windows_done += 1
                progreso = int((windows_done / total_windows) * 100)
                self.progreso_actualizado.emit(progreso)

                frame_idx = last_idx + 1

        finally:
            cap.release()
            print("[IA][GPU] Procesamiento DirectML finalizado.")
            self.procesamiento_terminado.emit(resultado)

    def run(self):
        import threading, math, time, gc, torch, torch_directml, cv2
        from ultralytics import YOLO
        from ultralytics.utils import ops

        print("[IA] Hilo actual:", threading.current_thread().name)
        t_inicio = time.time()

        # 🔹 Inicializar DirectML y modelo dentro del hilo (esto evita cuelgues)
        try:
            dml = torch_directml.device()
            model = YOLO("../models/best.pt")
            model.model.to(dml).float().eval()

            # Warm-up (previene el freeze en la primera inferencia)
            dummy = torch.zeros((1, 3, 960, 960), dtype=torch.float32).to(dml)
            with torch.no_grad():
                _ = model.model(dummy)
            print("[IA][GPU] Modelo inicializado correctamente en hilo secundario.")
        except Exception as e:
            print("[IA][ERROR] Fallo al inicializar DirectML:", e)
            self.procesamiento_terminado.emit({})
            return

        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.inicio_frame)

        WINDOW = 10
        frame_idx = self.inicio_frame
        resultado = {}

        def _store(idx, rects):
            if rects:
                resultado[idx] = rects

        total_frames = max(0, self.fin_frame - self.inicio_frame)
        total_windows = max(1, math.ceil(total_frames / WINDOW))
        windows_done = 0

        print("[IA][GPU] Iniciando procesamiento DirectML con detección de movimiento.")

        try:
            while frame_idx < self.fin_frame:
                buffer = []
                first_idx = frame_idx
                last_idx = min(self.fin_frame - 1, frame_idx + WINDOW)
                current_idx = first_idx

                while current_idx <= last_idx:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
                    ret, frm = cap.read()
                    if not ret or frm is None:
                        break
                    buffer.append((current_idx, frm))
                    current_idx += 1

                if not buffer:
                    break

                A_idx, A_frame = buffer[0]
                B_idx, B_frame = buffer[-1]

                # === inferir extremos A y B en GPU ===
                dets_A = self._inferir_directml(model, dml, A_frame, self.conf)
                _store(A_idx, dets_A)

                if B_idx != A_idx:
                    dets_B = self._inferir_directml(model, dml, B_frame, self.conf)
                    _store(B_idx, dets_B)
                else:
                    dets_B = dets_A

                densificar = False
                if dets_A and dets_B:
                    matches, nuevosB, coverageA = _match_rects_greedy(
                        dets_A, dets_B, iou_thr=0.60, size_tol=0.15, shift_px=20
                    )
                    if not (coverageA >= 0.70 and nuevosB == 0):
                        densificar = True
                elif dets_A or dets_B:
                    densificar = True

                if densificar and len(buffer) > 2:
                    for mid_idx, mid_frame in buffer[1:-1]:
                        dets_mid = self._inferir_directml(model, dml, mid_frame, self.conf)
                        _store(mid_idx, dets_mid)
                        del dets_mid, mid_frame

                windows_done += 1
                progreso = int((windows_done / total_windows) * 100)
                self.progreso_actualizado.emit(min(progreso, 99))

                del A_frame, B_frame, buffer
                gc.collect()
                frame_idx = last_idx + 1

        except Exception as e:
            import traceback
            print("[IA][ERROR] Excepción durante run():", e)
            traceback.print_exc()

        finally:
            cap.release()
            self.progreso_actualizado.emit(100)
            print("[IA][GPU] Procesamiento DirectML finalizado.")
            self.procesamiento_terminado.emit(resultado)


    # 🔸 Nueva función auxiliar dentro de la clase IAFrameProcessor:
    def _inferir_directml(self, model, dml, frame_bgr, conf=0.5, iou_thres=0.45, max_det=300):
        import cv2, torch
        from ultralytics.utils import ops

        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h0, w0 = img_rgb.shape[:2]

        # === Letterbox a 960 (sin deformar) ===
        target = 960
        r = min(target / w0, target / h0)

        new_w = round(w0 * r)
        new_h = round(h0 * r)

        resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = target - new_w
        pad_h = target - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2

        resized = cv2.copyMakeBorder(
            resized,
            pad_top, pad_h - pad_top,
            pad_left, pad_w - pad_left,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        )

        # === Tensor ===
        t = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        t = t.to(dml)

        # === Inferencia ===
        with torch.no_grad():
            preds = model.model(t)

        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        preds = preds.detach().cpu()
        dets = ops.non_max_suppression(
            preds,
            conf_thres=conf,
            iou_thres=iou_thres,
            max_det=max_det
        )

        rects = []

        if len(dets) and dets[0] is not None and len(dets[0]) > 0:
            det = dets[0]

            for *xyxy, conf_score, cls in det.tolist():
                x1, y1, x2, y2 = xyxy

                # Quitar padding
                x1 -= pad_left
                x2 -= pad_left
                y1 -= pad_top
                y2 -= pad_top

                # Reescalar al tamaño original
                x1 /= r
                x2 /= r
                y1 /= r
                y2 /= r

                # Convertir a int
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # Clamp a límites
                x1 = max(0, min(w0 - 1, x1))
                x2 = max(0, min(w0 - 1, x2))
                y1 = max(0, min(h0 - 1, y1))
                y2 = max(0, min(h0 - 1, y2))

                w = x2 - x1
                h = y2 - y1

                if w > 0 and h > 0:
                    rects.append((x1, y1, w, h))

        return rects

    
class WarmupThread(QThread):
    def run_backup(self):
        import torch, torch_directml
        from ultralytics import YOLO

        try:
            dml = torch_directml.device()
            model = YOLO("../models/best.pt")
            model.model.to(dml).float().eval()

            # Pasada de calentamiento
            dummy = torch.zeros((1, 3, 960, 960), dtype=torch.float32).to(dml)
            with torch.no_grad():
                _ = model.model(dummy)

            print("[IA][Warmup] GPU DirectML inicializada correctamente.")
        except Exception as e:
            print("[IA][Warmup][ERROR]", e)

    def run(self):
        import torch, torch_directml, gc
        from ultralytics import YOLO

        try:
            dml = torch_directml.device()

            model = YOLO("../models/best.pt")
            model.model.to(dml).float().eval()

            # Warmup consistente con 960
            dummy = torch.zeros((1, 3, 960, 960), dtype=torch.float32).to(dml)

            with torch.no_grad():
                _ = model.model(dummy)

            print("[IA][Warmup] GPU DirectML inicializada correctamente.")

            # Liberar explícitamente
            del dummy
            del model
            gc.collect()

        except Exception as e:
            print("[IA][Warmup][ERROR]", e)        




class VideoReviewer(QWidget):
    
    def actualizar_tiempo_desde_slider(self):
        if self.cap:
            frame = self.slider.value()
            tiempo_actual = self.formato_tiempo(int(frame / self.fps))
            tiempo_total = self.formato_tiempo(self.total_frames / self.fps)
            self.time_label.setText(f"{tiempo_actual} / {tiempo_total}")

    def recibir_resultado_ia(self, resultado):
        for frame_idx, rects in resultado.items():
            if frame_idx not in self.blur_ia_por_frame:
                self.blur_ia_por_frame[frame_idx] = []
            self.blur_ia_por_frame[frame_idx].extend(rects)

        self.ia_procesando = False
        self.slider.setEnabled(True)
        self.play_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.ia_estado_label.setText("IA: Detenida")

        self.ia_ultimo_frame = self.ia_fin_frame  # Guardar hasta qué frame se procesó
            
    def finalizar_procesamiento_ia(self, resultado):
        for idx, rects in resultado.items():
            if idx not in self.blur_ia_por_frame:
                self.blur_ia_por_frame[idx] = []
            self.blur_ia_por_frame[idx].extend(rects)

        self.progress_bar.setVisible(False)
        self.ia_estado_label.setText("IA: Detenida")
        self.ia_procesando = False
        self.slider.setEnabled(True)
        self.play_button.setEnabled(True)

        # Actualizar frame actual si está dentro del rango procesado
        if self.ia_inicio_frame <= self.frame_idx <= self.ia_fin_frame:
            #self.show_frame()
            self.show_frame(forzado=True)
            log(f"[DEBUG] self.show_frame() desde el metodo def finalizar_procesamiento_ia(self, resultado):")
    
    #IA---
    def iniciar_procesamiento_ia(self):
        self.pause_video()  # ← Detiene la reproducción si estaba activa
        self.btn_procesar_ia.setEnabled(False)
        

        if not self.cap or self.ia_procesando:
            return

        # Leer duración personalizada desde QLineEdit
        try:
            duracion_seg = int(self.ia_duracion_input.text())
            if duracion_seg <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Duración inválida", "La duración debe ser un número entero positivo.")
            return

        # Bloquear controles mientras se procesa
        self.ia_procesando = True
        self.slider.setEnabled(False)
        self.play_button.setEnabled(False)
        self.ia_estado_label.setText("IA: Procesando...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # indeterminado
        QApplication.processEvents()      # refrescar GUI
        self.progress_bar.setValue(0)

        # Leer posición actual del slider
        frame_actual = self.slider.value()
        self.ia_inicio_frame = frame_actual
        self.ia_fin_frame = min(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, frame_actual + int(duracion_seg * self.fps))

        # Resetear bandera de slider
        self.slider_tocado = False

         # ⏱ Inicia tiempo total de procesamiento
        self._t_inicio_ia = time.time()
        log("[IA] ▶ Iniciado procesamiento manual IA")

        
        # Iniciar procesamiento
        self.procesar_ia_en_rango(self.ia_inicio_frame, self.ia_fin_frame)

    def iniciar_procesamiento_ia_backup(self):
        self.pause_video()  # ← Detiene la reproducción si estaba activa
        self.btn_procesar_ia.setEnabled(False)
        

        if not self.cap or self.ia_procesando:
            return

        # Leer duración personalizada desde QLineEdit
        try:
            duracion_seg = int(self.ia_duracion_input.text())
            if duracion_seg <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Duración inválida", "La duración debe ser un número entero positivo.")
            return

        # Bloquear controles mientras se procesa
        self.ia_procesando = True
        self.slider.setEnabled(False)
        self.play_button.setEnabled(False)
        self.ia_estado_label.setText("IA: Procesando...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Leer posición actual del slider
        frame_actual = self.slider.value()
        self.ia_inicio_frame = frame_actual
        self.ia_fin_frame = min(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, frame_actual + int(duracion_seg * self.fps))

        # Resetear bandera de slider
        self.slider_tocado = False

         # ⏱ Inicia tiempo total de procesamiento
        self._t_inicio_ia = time.time()
        log("[IA] ▶ Iniciado procesamiento manual IA")

        # Iniciar procesamiento
        self.procesar_ia_en_rango(self.ia_inicio_frame, self.ia_fin_frame)

    def procesar_ia_en_rango(self, inicio_frame, fin_frame):
        
        try:
            conf_valor = float(self.qlineedit_conf.text())
        except:
            conf_valor = 0.5  # fallback si falla
            
        self.slider_tocado = False
        log(f"[IA] FPS enviado al hilo de procesamiento: {self.fps}")
        self.ia_thread = IAFrameProcessor(self.video_path, inicio_frame, fin_frame, self.fps, conf=conf_valor)
        self.ia_thread.progreso_actualizado.connect(self.progress_bar.setValue)
        self.progress_bar.setRange(0, 100)
        self.ia_thread.procesamiento_terminado.connect(self.procesamiento_ia_completado)
        self.ia_thread.start()
        self.btn_procesar_ia.setEnabled(True)
        self.ia_estado_label.setText("IA: Detenida")

    def procesamiento_ia_completado(self, resultado):
        for frame_idx, rects in resultado.items():
            self.blur_ia_por_frame[frame_idx] = rects

        self.ia_procesando = False
        self.progress_bar.setVisible(False)
        self.slider.setEnabled(True)
        self.play_button.setEnabled(True)
        self.ia_estado_label.setText("IA: Detenida")
        log(f"[IA] Resultado aplicado. Frames: {list(resultado.keys())}")

        # ⏱ Log de duración total
        t_fin = time.time()
        duracion = t_fin - getattr(self, "_t_inicio_ia", t_fin)
        log(f"[IA] ⏱ Tiempo total de procesamiento: {duracion:.2f} segundos")

        
        #self.generar_blurs_clonados_si_continua_v3_2(resultado)

        t0 = time.perf_counter()
        self.generar_blurs_clonados_si_continua_v3_2(resultado)
        t1 = time.perf_counter()
        print(f"[CLONES] Tiempo de clonación v3_2: {t1 - t0:.2f} segundos")

        # Activar clonación si corresponde
        #comentado temporalmente --> self.generar_blurs_clonados_si_continua(resultado)

        # Mostrar el frame actualizado (importante para que se vean los nuevos blurs)
        self.show_frame(forzado=True)

    def generar_blurs_clonados_si_continua_v3_1(self, resultado: dict):
        # Limpiar registros anteriores
        self.rects_clonados_tmp = {}
        log_ia_post("[CLONES_V3_1] Iniciando clonación con merge diferido...")

        MAX_CLONES = 5
        TOLERANCIA_V2 = 50
        frames_originales = sorted(resultado.keys())

        # ====== PASADA 1: CLONAR HACIA ATRÁS ======
        for frame_idx in frames_originales:
            rects = self.blur_ia_por_frame.get(frame_idx, [])
            if not rects:
                continue

            log_ia_post(f"[CLONES_V3_1] Frame {frame_idx} tiene {len(rects)} detecciones:")
            for i, r in enumerate(rects):
                log_ia_post(f"   → #{i} {r}")

            for rect in rects:
                if len(rect) < 4:
                    continue

                conf = rect[4] if len(rect) >= 5 else 0.5

                for offset in range(1, MAX_CLONES + 1):
                    f_prev = frame_idx - offset
                    if f_prev < 0:
                        break

                    rects_prev = self.blur_ia_por_frame.get(f_prev, [])
                    rects_clonados_actuales = self.rects_clonados_tmp.get(f_prev, [])

                    if any(rects_similares(r, rect, TOLERANCIA_V2)
                        for r in rects_prev + rects_clonados_actuales):
                        log_ia_post(f"[CLONES_V3_1] ← Frame {f_prev} ya tiene detección similar, se omite clon hacia atrás.")
                        break

                    clon = (*rect[:4], conf)
                    self.rects_clonados_tmp.setdefault(f_prev, []).append(clon)
                    log_ia_post(f"[CLONES_V3_1] ← Clonado (pendiente) en frame {f_prev} desde {frame_idx}: {clon}")

        # ====== PASADA 2: CLONAR HACIA ADELANTE ======
        for frame_idx in frames_originales:
            rects = self.blur_ia_por_frame.get(frame_idx, [])
            if not rects:
                continue

            for rect in rects:
                if len(rect) < 4:
                    continue

                conf = rect[4] if len(rect) >= 5 else 0.5

                for offset in range(1, MAX_CLONES + 1):
                    f_next = frame_idx + offset
                    if f_next >= self.total_frames:
                        break

                    rects_next = self.blur_ia_por_frame.get(f_next, [])
                    rects_clonados_actuales = self.rects_clonados_tmp.get(f_next, [])

                    if any(rects_similares(r, rect, TOLERANCIA_V2)
                        for r in rects_next + rects_clonados_actuales):
                        log_ia_post(f"[CLONES_V3_1] → Frame {f_next} ya tiene detección similar, se omite clon hacia adelante.")
                        break

                    clon = (*rect[:4], conf)
                    self.rects_clonados_tmp.setdefault(f_next, []).append(clon)
                    log_ia_post(f"[CLONES_V3_1] → Clonado (pendiente) en frame {f_next} desde {frame_idx}: {clon}")

        # ====== MERGE FINAL: APLICAR CLONES A blur_ia_por_frame ======
        for f, clones in self.rects_clonados_tmp.items():
            self.blur_ia_por_frame.setdefault(f, []).extend(clones)
            log_ia_post(f"[CLONES_V3_1] ✅ Merge final: {len(clones)} clones añadidos a frame {f}")

        log_ia_post("[CLONES_V3_1] Finalizado proceso de clonación con merge.")

    def generar_blurs_clonados_si_continua_v3_2(self, resultado: dict):
        self.rects_clonados_tmp = {}
        log_ia_post("[CLONES_V3] Iniciando clonación optimizada (atrás + adelante)...")

        MAX_CLONES = 5
        TOLERANCIA_V2 = 80
        frames_originales = sorted(resultado.keys())

        for frame_idx in frames_originales:
            rects = self.blur_ia_por_frame.get(frame_idx, [])
            if not rects:
                continue

            log_ia_post(f"[CLONES_V3] Frame {frame_idx} tiene {len(rects)} detecciones:")
            for i, r in enumerate(rects):
                log_ia_post(f"   → #{i} {r}")

            for rect in rects:
                if len(rect) < 4:
                    continue

                conf = rect[4] if len(rect) >= 5 else 0.5

                # CLONAR HACIA ATRÁS
                for offset in range(1, MAX_CLONES + 1):
                    f_prev = frame_idx - offset
                    if f_prev < 0:
                        break

                    rects_prev = self.blur_ia_por_frame.get(f_prev, [])
                    rects_clonados = self.rects_clonados_tmp.get(f_prev, [])

                    if any(rects_similares(r, rect, TOLERANCIA_V2) for r in rects_prev + rects_clonados):
                        log_ia_post(f"[CLONES_V3] ← Frame {f_prev} ya tiene detección similar, se omite clon hacia atrás.")
                        break

                    clon = (*rect[:4], conf)
                    self.rects_clonados_tmp.setdefault(f_prev, []).append(clon)
                    log_ia_post(f"[CLONES_V3] ← Clonado (pendiente) en frame {f_prev} desde {frame_idx}: {clon}")

                # CLONAR HACIA ADELANTE
                for offset in range(1, MAX_CLONES + 1):
                    f_next = frame_idx + offset
                    if f_next >= self.total_frames:
                        break

                    rects_next = self.blur_ia_por_frame.get(f_next, [])
                    rects_clonados = self.rects_clonados_tmp.get(f_next, [])

                    if any(rects_similares(r, rect, TOLERANCIA_V2) for r in rects_next + rects_clonados):
                        log_ia_post(f"[CLONES_V3] → Frame {f_next} ya tiene detección similar, se omite clon hacia adelante.")
                        break

                    clon = (*rect[:4], conf)
                    self.rects_clonados_tmp.setdefault(f_next, []).append(clon)
                    log_ia_post(f"[CLONES_V3] → Clonado (pendiente) en frame {f_next} desde {frame_idx}: {clon}")

        # MERGE FINAL
        for f, clones in self.rects_clonados_tmp.items():
            self.blur_ia_por_frame.setdefault(f, []).extend(clones)
            log_ia_post(f"[CLONES_V3] ✅ Merge final: {len(clones)} clones añadidos a frame {f}")

        # Solo conservar los clones temporales si DEBUG_MODE está activado
        if not self.DEBUG_MODE:
            self.rects_clonados_tmp.clear()
            log_ia_post("[CLONES_V3] 🔍 DEBUG_MODE desactivado, rects_clonados_tmp ha sido limpiado.")
        else:
            log_ia_post("[CLONES_V3] 🔬 DEBUG_MODE activado, se conservan los clones temporales.")
            # tener en cuenta que los clones generan conflicto en showframe y en la eliminacion
            #ya que habria 2 registros iguales en la GUI blur_ia_por_frame y rects_clonados_tmp

        log_ia_post("[CLONES_V3] Finalizado proceso de clonación.")


    def generar_blurs_clonados_si_continua_v2(self, resultado: dict):
        
        #limpiar registro anterior
        self.rects_clonados_tmp = {}

        log_ia_post("[CLONES_V2] Iniciando clonación simple hacia atrás y adelante...")
        
        

        MAX_CLONES = 5
        TOLERANCIA_V2 = 50  # píxeles de diferencia máxima para considerar rects similares
        self.rects_clonados_tmp = {}
        #total_frames = sorted(self.blur_ia_por_frame.keys())
        # Solo procesar frames originales, sin incluir los clones que se generarán
        frames_originales = sorted(resultado.keys())

        for frame_idx in frames_originales:
            rects = self.blur_ia_por_frame.get(frame_idx, [])
            if not rects:
                continue

            log_ia_post(f"[CLONES_V2] Frame {frame_idx} tiene {len(rects)} detecciones:")
            for i, r in enumerate(rects):
                log_ia_post(f"   → #{i} {r}")

            for rect in rects:
                if len(rect) < 4:
                    continue

                conf = rect[4] if len(rect) >= 5 else 0.5

                # ========== CLONAR HACIA ATRÁS ==========
                for offset in range(1, MAX_CLONES + 1):
                    f_prev = frame_idx - offset
                    if f_prev < 0:
                        break

                    rects_prev = self.blur_ia_por_frame.get(f_prev, [])
                    clones_prev = self.rects_clonados_tmp.get(f_prev, [])

                   
                    #temportalmente comentado
                    if any(rects_similares(r, rect, TOLERANCIA_V2) for r in rects_prev + clones_prev):
                        log_ia_post(f"[CLONES_V2] ← Frame {f_prev} ya tiene detección similar, se omite clon hacia atrás.")
                        #self.generar_blurs_clonados_si_continua_v2_1(frame_idx, rect, "atras")  # ← agregar aquí
                        break


                    clon = (*rect[:4], conf)
                    self.blur_ia_por_frame.setdefault(f_prev, []).append(clon)
                    self.rects_clonados_tmp.setdefault(f_prev, []).append(clon)
                    log_ia_post(f"[CLONES_V2] ← Clonado en frame {f_prev} desde {frame_idx}: {clon}")

                # ========== CLONAR HACIA ADELANTE ==========
                for offset in range(1, MAX_CLONES + 1):
                    f_next = frame_idx + offset
                    if f_next >= self.total_frames:
                        break

                    rects_next = self.blur_ia_por_frame.get(f_next, [])
                    clones_next = self.rects_clonados_tmp.get(f_next, [])

                    
                    if any(rects_similares(r, rect, TOLERANCIA_V2) for r in rects_next + clones_next):
                        log_ia_post(f"[CLONES_V2] → Frame {f_next} ya tiene detección similar, se omite clon hacia adelante.")
                        #self.generar_blurs_clonados_si_continua_v2_1(frame_idx, rect, "adelante")  # ← agregar aquí
                        break


                    clon = (*rect[:4], conf)
                    self.blur_ia_por_frame.setdefault(f_next, []).append(clon)
                    self.rects_clonados_tmp.setdefault(f_next, []).append(clon)
                    log_ia_post(f"[CLONES_V2] → Clonado en frame {f_next} desde {frame_idx}: {clon}")

        log_ia_post("[CLONES_V2] Finalizado proceso de clonación.")
    
    def generar_blurs_clonados_si_continua_v2_1(self, frame_idx: int, rect: tuple, direccion: str):
        MAX_CLONES = 3
        TOLERANCIA = 100  # píxeles de diferencia máxima para considerar rects similares

        def rects_similares_por_centro(r1, r2):
            x1c = (r1[0] + r1[2]) / 2
            y1c = (r1[1] + r1[3]) / 2
            x2c = (r2[0] + r2[2]) / 2
            y2c = (r2[1] + r2[3]) / 2
            return abs(x1c - x2c) <= TOLERANCIA and abs(y1c - y2c) <= TOLERANCIA

        # → lógica completa para "adelante" y "atras" según corresponda...
        
    #---IA

    def slider_released(self):
        log("[DEBUG] slider_released() llamado. Frame idx slider: {}".format(self.slider.value()))
        self.slider_tocado = False
        self.frame_idx = self.slider.value()
        self.origen_show_frame = 'slider'

        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
            self.cap.grab()
            ret, frame = self.cap.retrieve()
            if ret:
                self.current_frame = frame.copy()
            else:
                log(f"[ERROR] No se pudo leer el frame {self.frame_idx} desde slider_released")

        log(f"[DEBUG] Llamando show_frame(forzado=True) desde slider_released en frame {self.frame_idx} | origen={self.origen_show_frame}")
        self.show_frame(forzado=True)
        self.setFocus()  # <-- Recupera los atajos de teclado después de usar el slider
    
    def generar_log_show_frame(self, frame_idx, rects_ia, eliminados_ia):
        # Desactivar log si app_instance aún no existe o DEBUG_MODE está desactivado
        # Verificar si app_instance existe y tiene DEBUG_MODE activo
        if not globals().get("app_instance") or not getattr(app_instance, "DEBUG_MODE", False):
            return
        
        ruta = "show_frame_debug.txt"
        with open(ruta, "a", encoding="utf-8") as f:
            f.write(f"\n[FRAME {frame_idx}] IA DETECTIONS\n")
            for i, rect in enumerate(rects_ia):
                estado = "IGNORADO" if any(rects_similares(rect, r) for r in eliminados_ia) else "DIBUJADO"
                f.write(f"   #{i:02d} {rect} -> {estado}\n")
            f.write(f"  Eliminados IA: {len(eliminados_ia)}\n")
            for r in eliminados_ia:
                f.write(f"     Eliminado: {r}\n")

    def registrar_append_blur_eliminado(self, frame_idx, rect, origen="desconocido"):
        eliminados_actuales = self.blur_eliminado_ia.setdefault(frame_idx, [])
        for existente in eliminados_actuales:
            if rects_similares(rect, existente):
                log_ia_click(f"  [DUPLICADO DETECTADO] Rectángulo {rect} ya registrado en frame {frame_idx} (origen: {origen})")
                return False
        eliminados_actuales.append(rect)
        log_ia_click(f"  [AGREGADO] Rectángulo agregado a blur_eliminado_ia en frame {frame_idx}: {rect} (origen: {origen})")
        return True

    def exportar_log_frame(self):
        frame = self.frame_idx
        ia_rects = self.blur_ia_por_frame.get(frame, [])
        manual_rects = self.blur_manual_por_frame.get(frame, [])
        #eliminados = self.blur_eliminado_ia.get(frame, [])
        eliminados = [
            r for r, _, frame_origen in self.coords_para_eliminar
            if frame >= frame_origen
        ]


        with open(f"frame_{frame}_log.txt", "w", encoding="utf-8") as f:
            f.write(f"--- LOG DE FRAME {frame} ---\n\n")
            f.write(f"[BLUR IA] ({len(ia_rects)}):\n")
            for r in ia_rects:
                f.write(f"  {r}\n")
            f.write(f"\n[BLUR MANUAL] ({len(manual_rects)}):\n")
            for r in manual_rects:
                f.write(f"  {r}\n")
            f.write(f"\n[BLUR ELIMINADOS IA] ({len(eliminados)}):\n")
            for r in eliminados:
                f.write(f"  {r}\n")
            f.write(f"\n[COORDS_PARA_ELIMINAR] ({len(self.coords_para_eliminar)} en total):\n")
            #for rect, fallos in self.coords_para_eliminar:
            for rect, fallos, frame_origen in self.coords_para_eliminar:
                f.write(f"  Rect: {rect} | Fallos: {fallos}\n")
        log(f"[EXPORT] Log de frame {frame} exportado como frame_{frame}_log.txt")
        
    def remover_foco(self):
            self.sender().clearFocus()
            self.setFocus()
            
    def __init__(self):
        
        #self.blur_escala = 1.2 
        #self.slider_tocado = False
        #self.blur_kernel = 31  # entre 15 y 51 recomendado; menor = más suave
        
        super().__init__()

        # En __init__ de VideoReviewer:
        self.warmup_thread = WarmupThread()
        self.warmup_thread.start()
        self._step_en_progreso = False
        self.modo_redimension_fantasma = False
        self.blur_fantasma_redimensionandose = None
        self._resize_anchor = None   # "tl", "se", etc. (por ahora usaremos "se")
        self._resize_start = None    # (vx, vy, x, y, w, h) al iniciar

        self._id_blur_fantasma = 1
        self.blur_fantasma_moviendose = None  # Dict temporal del blur fantasma que se está moviendo
        self.modo_mover_blur_fantasma = False  # Flag para saber si estamos arrastrando


        self.bitrate_original_kbps = 5000

        self._ultimo_frame_timestamp = None

        self.origen_show_frame = "desconocido"
        
        # Ajustar el tamaño de la ventana según la pantalla
        screen_rect = QDesktopWidget().availableGeometry()
        screen_w = screen_rect.width()
        screen_h = screen_rect.height()

        # Márgenes de seguridad para barra de tareas y bordes
        margen_w = 10
        margen_h = 40

        # Tamaño deseado
        deseado_w = 1280
        deseado_h = 720

        # Tamaño real sin sobrepasar pantalla
        final_w = min(deseado_w, screen_w - margen_w)
        final_h = min(deseado_h, screen_h - margen_h)

        # Evitar que se exija más de lo que la pantalla permite
        self.resize(final_w, final_h)
        self.setMinimumSize(min(400, final_w), min(300, final_h))

        #IR A TIEMPO
        self.ir_tiempo_button = QPushButton("🔍 IR")
        self.ir_tiempo_button.clicked.connect(self.mostrar_selector_tiempo)
        #self.selector_tiempo = QTimeEdit()
        #self.selector_tiempo.setDisplayFormat("HH:mm:ss")
        #self.selector_tiempo.setTime(QTime(0, 0, 0))
        #self.selector_tiempo.setVisible(False)
        #self.boton_ir = QPushButton("IR")
        #self.boton_ir.setVisible(False)
        #self.boton_ir.clicked.connect(self.ir_a_tiempo_seleccionado)

        


        self.fps = 30.0 
        self.rects_clonados_tmp = {}
        self.actualizar_button = QPushButton("ACTUALIZAR VISTA")
        self.actualizar_button.clicked.connect(lambda: self.show_frame(forzado=True))
        
        
        self.debug_checkbox = QCheckBox("DEBUG")
        self.debug_checkbox.setChecked(False)  # Por defecto activado si deseas
        
        
        self.velocidad_reproduccion = 1.0  # x1 por defecto
        self.setFocusPolicy(Qt.StrongFocus)
        
        self.blurs_en_construccion = []
        self.blur_en_construccion = None
        #self.setFocusPolicy(Qt.StrongFocus) # Esto garantiza que keyPressEvent funcione aunque se hagan clics en otros widgets como botones o el slider.
        self.blur_escala = 1.1 
        self.slider_tocado = False
        self.blur_kernel = 41  # debe ser inpar entre 15 y 51 recomendado; menor = más suave #27 o 30 para camaras poco lejanas  42 a mas para cara mas borrosa o camara muy cerca    
        
        self._tecla_izquierda_presionada = False
        self._tecla_derecha_presionada = False
        self._timer_navegacion_teclado = QTimer()
        self._timer_navegacion_teclado.setInterval(80)
        self._timer_navegacion_teclado.timeout.connect(self._navegar_por_tecla_mantenida)
        
        
        self.setWindowTitle(f"Dev. Luis De La Torre P.")
        #self.resize(1000, 800)


        
        self.image_label = VideoLabel(self)
        self.image_label.setText("Carga un video")
        #self.image_label.setFixedSize(800, 600)
        self.image_label.setMinimumSize(400, 300)  # opcional: tamaño mínimo
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setAlignment(Qt.AlignCenter)
        #self.image_label.mousePressEvent = self.handle_mouse_click
        #self.image_label.mouseReleaseEvent = self.handle_mouse_release
        self.image_label.mouseMoveEvent = self.handle_mouse_move



        #-->FLASH VISUAL EN PANTALLA
        self.flash_label = QLabel("", self.image_label)
        self.flash_label.setStyleSheet("""
            color: white;
            font-size: 20px;
            font-weight: bold;
            background-color: transparent;  /* background-color: rgba(0, 0, 0, 100) barra color negro más transparente */
            border-radius: 6px;
            padding: 4px;
        """)

        self.flash_label.setAlignment(Qt.AlignCenter)
        self.flash_label.setVisible(False)
        
        self.flash_foto_label = QLabel("", self.image_label)
        self.flash_foto_label.setStyleSheet("""
            color: white;
            font-size: 16px;
            font-weight: bold;
            background-color: transparent;
            border-radius: 6px;
            padding: 4px;
        """)
        self.flash_foto_label.setAlignment(Qt.AlignCenter)
        self.flash_foto_label.setVisible(False)
        #<--FLASH VISUAL EN PANTALLA
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.save_progress_bar = QProgressBar()
        self.save_progress_bar.setVisible(False)
        self.save_progress_bar.setMaximum(0)  # modo indeterminado
        
        
        self.clones_progress_bar = QProgressBar(self)
        self.clones_progress_bar.setGeometry(50, 100, 400, 20)  # Ajusta la posición y tamaño
        self.clones_progress_bar.setVisible(False)
        

        
        # Barra de progreso para exportación
        self.export_progress_bar = QProgressBar()
        self.export_progress_bar.setVisible(False)  # Oculta por defecto
        self.export_progress_bar.setValue(0)
       
        self.frame_label = QLabel("Frame: 0 | Velocidad: x1.00")
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setStyleSheet("font-weight: bold;")

        self.load_button = QPushButton("Cargar Video")
        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pausa")
        self.prev_frame_button = QPushButton("⏮ Frame anterior")
        self.next_frame_button = QPushButton("⏭ Siguiente frame")
        
        # Para el botón "Frame Anterior"
        #self.prev_frame_button.pressed.connect(lambda: self._iniciar_navegacion_continua(-1))
        self.prev_frame_button.pressed.connect(self._manejar_boton_retroceso)
        self.prev_frame_button.released.connect(self._detener_navegacion_continua)

        # Para el botón "Siguiente Frame"
        #self.next_frame_button.pressed.connect(lambda: self._iniciar_navegacion_continua(1))
        self.next_frame_button.pressed.connect(self._manejar_boton_avance)
        self.next_frame_button.released.connect(self._detener_navegacion_continua)


        self.export_button = QPushButton("EXPORTAR VIDEO")
        self.export_button.clicked.connect(self.exportar_avance)
        
        self.foto_button = QPushButton("📸 FOTO")
        self.foto_button.clicked.connect(self.capturar_foto)
        
        self.guardar_button = QPushButton("GUARDAR AVANCE")
        self.guardar_button.clicked.connect(self.guardar_avance)
        
        self.cargar_button = QPushButton("CARGAR AVANCE")
        self.cargar_button.clicked.connect(self.cargar_avance)
        
        self.export_frame_log_button = QPushButton("Exportar Log del Frame")
        self.export_frame_log_button.clicked.connect(self.exportar_log_frame)
        self.export_frame_log_button.setVisible(False)

        self.load_button.clicked.connect(self.load_video)
        self.play_button.clicked.connect(self.play_video)
        self.pause_button.clicked.connect(self.pause_video)
        #self.prev_frame_button.clicked.connect(self.prev_frame)
        #self.next_frame_button.clicked.connect(self.next_frame_manual)
        #self.export_button.clicked.connect(self.exportar_avance)

        # SLIDER
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setSingleStep(1)
        self.slider.setEnabled(False)
        self.slider.sliderReleased.connect(self.slider_released)
        self.slider.valueChanged.connect(self.actualizar_tiempo_desde_slider)
        
        self.time_label = QLabel("00:00 / 00:00")  # Tiempo actual / total
        #self.frame_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("font-weight: bold;")
        

        
        # BLOQUE DE IA
        self.ia_duracion_label = QLabel("Duración a procesar (seg):")
        self.ia_duracion_input = QLineEdit()
        self.ia_duracion_input.setFixedWidth(40)
        self.ia_duracion_input.setText("20")  # Valor por defecto

        self.ia_umbral_label = QLabel("Auto-extensión si faltan (seg):")
        self.ia_umbral_input = QLineEdit()
        self.ia_umbral_input.setFixedWidth(40)
        self.ia_umbral_input.setText("5")  # Valor por defecto
        
        # Crear QTimeEdit para exportación por rango
        self.export_inicio_time = QTimeEdit()
        self.export_fin_time = QTimeEdit()
        
        self.frame_nav_timer = QTimer()
        self.frame_nav_timer.setInterval(50)  # 50 ms = 20 fps aprox.
        self.frame_nav_timer.timeout.connect(self._frame_nav_step)
        self._frame_nav_direction = 0  # -1 = anterior, +1 = siguiente

        # Configuración visual
        self.export_inicio_time.setDisplayFormat("HH:mm:ss")
        self.export_fin_time.setDisplayFormat("HH:mm:ss")
        self.export_inicio_time.setTime(QTime(0, 0, 0))
        self.export_fin_time.setTime(QTime(0, 0, 10))

        

        self.ia_estado_label = QLabel("IA: Detenida")

        self.btn_procesar_ia = QPushButton("PROCESAR IA")
        self.btn_procesar_ia.clicked.connect(self.iniciar_procesamiento_ia)
        self.conf_label = QLabel("Confianza mínima (conf):")
        self.conf_input = QLineEdit()
        self.conf_input.setText("0.5")  # Valor por defecto
        
        self.auto_procesamiento_checkbox = QCheckBox("AUTO PROCESAMIENTO")
        self.auto_procesamiento_checkbox.setChecked(False)  # Desactivado por defecto
        #layout_principal.addWidget(self.auto_procesamiento_checkbox)  # O donde tengas tu layout
        self.en_pantalla_completa = False #atajo de teclado con F11 para pantalla completa
        self.auto_procesamiento_checkbox.stateChanged.connect(self.actualizar_estado_procesamiento)

        

        # Sub-layouts para agrupar label + input
        duracion_layout = QHBoxLayout()
        duracion_layout.addWidget(self.ia_duracion_label)
        duracion_layout.addWidget(self.ia_duracion_input)

        umbral_layout = QHBoxLayout()
        umbral_layout.addWidget(self.ia_umbral_label)
        umbral_layout.addWidget(self.ia_umbral_input)
        
        umbral_layout.addWidget(QLabel("Inicio exportación:"))
        umbral_layout.addWidget(self.export_inicio_time)
        umbral_layout.addWidget(QLabel("Fin exportación:"))
        umbral_layout.addWidget(self.export_fin_time)

        # Layout principal IA
        ia_layout = QHBoxLayout()
        ia_layout.addLayout(duracion_layout)
        ia_layout.addLayout(umbral_layout)
        ia_layout.addWidget(self.btn_procesar_ia)
        ia_layout.addWidget(self.ia_estado_label)
        ia_layout.addWidget(self.auto_procesamiento_checkbox)  # O donde tengas tu layout
        ia_layout.addWidget(self.conf_label)
        ia_layout.addWidget(self.conf_input) 
        
        #ia_layout = QHBoxLayout()
        ia_layout.addWidget(self.debug_checkbox)
        # Agrega este layout al layout principal de la GUI

        # Botones principales
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.load_button)
        btn_layout.addWidget(self.play_button)
        btn_layout.addWidget(self.pause_button)
        btn_layout.addWidget(self.prev_frame_button)
        btn_layout.addWidget(self.next_frame_button)
        btn_layout.addWidget(self.export_button)
        btn_layout.addWidget(self.export_frame_log_button)
        btn_layout.addWidget(self.guardar_button)
        btn_layout.addWidget(self.cargar_button)
        btn_layout.addWidget(self.foto_button)
        btn_layout.addWidget(self.actualizar_button)
        btn_layout.addWidget(self.ir_tiempo_button)
        
        
        # Layout del reproductor
        #image_wrapper = QWidget()
        #image_layout = QVBoxLayout(image_wrapper)
        #image_layout.addStretch()
        #image_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        #image_layout.addStretch()
        #image_wrapper.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
       
        # Layout del reproductor
        image_layout = QVBoxLayout()
        image_layout.addWidget(self.image_label)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(0)

        image_wrapper = QWidget()
        image_wrapper.setLayout(image_layout)
        image_wrapper.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        
        # Layout principal
        main_layout = QVBoxLayout()
        main_layout.addWidget(image_wrapper)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.clones_progress_bar)  # si estás usando layouts
        main_layout.addWidget(self.export_progress_bar)
        main_layout.addWidget(self.frame_label)
        main_layout.addWidget(self.slider)
        main_layout.addWidget(self.time_label, alignment=Qt.AlignCenter)
        main_layout.addLayout(ia_layout)
        main_layout.addLayout(btn_layout)
        #btn ir
        #main_layout.addWidget(self.selector_tiempo, alignment=Qt.AlignCenter)
        #main_layout.addWidget(self.boton_ir, alignment=Qt.AlignCenter)
        
        self.setLayout(main_layout)
        main_layout.addWidget(self.save_progress_bar)

        # Variables internas
        self.cap = None
        self.video_path = ""
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        log(f"[DEBUG] Timer activo: {self.timer.isActive()}")
        self.frame_idx = 0
        self.fps = 30
        
        #self.timer.stop()# Evita que avance por sí solo
        
        self.blur_ia_por_frame = {}
        self.blur_manual_por_frame = {}
        self.blur_eliminado_ia = {}
        self.minutos_procesados = set()
        self.procesando = False

        # Estado del procesamiento IA
        self.ia_procesando = False
        self.ia_inicio_frame = 0
        self.ia_fin_frame = 0

        self.coords_para_eliminar = []  # (rect, fallos)
        self.drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        
        
        
        with open("show_frame_debug.txt", "w") as f:
            f.write("")  # limpiar contenido anterior
        
        # Conectar todos los botones
        for btn in [self.load_button, self.play_button, self.pause_button,
                    self.prev_frame_button, self.next_frame_button,
                    self.export_button, self.export_frame_log_button,
                    self.guardar_button, self.cargar_button,
                    self.btn_procesar_ia]:
            btn.clicked.connect(self.remover_foco)
            
        self.setFocus()  # <- Esto asegura que se capturen las teclas

    def mostrar_selector_tiempo(self):
        if not self.cap:
            QMessageBox.warning(self, "Sin video", "Primero debes cargar un video.")
            return
        dialog = TiempoDialog(self, self.fps, self.total_frames)
        dialog.exec_()
        self.setFocus()  # Recuperar atajos

    def ir_a_tiempo_seleccionado(self):
        qtime = self.selector_tiempo.time()
        frame = self.qtime_a_frame(qtime)
        if 0 <= frame < self.total_frames:
            self.pause_video()
            self.frame_idx = frame
            self.slider.setValue(frame)
            self.show_frame(forzado=True)
        else:
            QMessageBox.warning(self, "Tiempo inválido", "El tiempo ingresado está fuera del rango del video.")    

    def forzar_actualizacion_vista(self):
        log(f"[ACTUALIZAR_VISTA] Botón presionado en frame {self.frame_idx}")
        log_ia_click(f"[ACTUALIZAR_VISTA] Forzando show_frame en frame {self.frame_idx}")
        
        # Información detallada del estado de los blurs
        rects_ia = self.blur_ia_por_frame.get(self.frame_idx, [])
        rects_manual = self.blur_manual_por_frame.get(self.frame_idx, [])
        eliminados_ia = self.blur_eliminado_ia.get(self.frame_idx, [])
        coords_propagados = [
            r for r, _, frame_origen in self.coords_para_eliminar if self.frame_idx >= frame_origen
        ]

        log_ia_click(f"[ACTUALIZAR_VISTA] Rects IA en frame {self.frame_idx}: {rects_ia}")
        log_ia_click(f"[ACTUALIZAR_VISTA] Rects MANUAL en frame {self.frame_idx}: {rects_manual}")
        log_ia_click(f"[ACTUALIZAR_VISTA] Eliminados IA explícitos en frame {self.frame_idx}: {eliminados_ia}")
        log_ia_click(f"[ACTUALIZAR_VISTA] Eliminaciones propagadas aplicables: {coords_propagados}")
        
        self.origen_show_frame = 'actualizar_vista'

        self.show_frame(forzado=True)
   
    def actualizar_estado_procesamiento(self):
        if self.auto_procesamiento_checkbox.isChecked():
            self.ia_estado_label.setText("IA: Auto habilitado")
        else:
            self.ia_estado_label.setText("IA: Solo manual")
        
    #-->METODO DE CAPTURAR FOTO
    def capturar_foto_backup20251204(self):
        if not hasattr(self, 'current_frame') or self.current_frame is None:
            QMessageBox.warning(self, "Sin video", "No hay frame cargado para capturar.")
            return

        # Crear carpeta si no existe
        os.makedirs("FOTOS", exist_ok=True)

        # Generar nombre único por frame y timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"FOTOS/frame_{self.frame_idx}_{timestamp}.jpg"

        # Guardar la imagen original (sin overlays)
        cv2.imwrite(filename, self.current_frame)
        #log(f"[FOTO] Captura guardada: {filename}")
        self.mostrar_flash_foto()
        #QMessageBox.information(self, "Captura guardada", f"Imagen guardada en:\n{filename}")

    def capturar_foto(self):
        """
        Captura el frame actual con blurs aplicados.
        La primera vez pregunta al usuario en qué resolución quiere las fotos.
        """

        # 0️⃣ Solo permitir si el video está pausado
        if self.timer.isActive():
            self.image_label.setFocus()
            return

        # 1️⃣ Preguntar resolución solo una vez
        if not hasattr(self, "resolucion_foto_elegida"):
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Question)
            msg.setWindowTitle("Resolución de captura")
            msg.setText("¿En qué resolución quieres guardar las fotos?")

            btn_640 = msg.addButton("640×360 (16:9)", QMessageBox.AcceptRole)
            btn_960 = msg.addButton("960×540 (16:9)", QMessageBox.AcceptRole)
            btn_orig = msg.addButton("Original (máxima calidad)", QMessageBox.AcceptRole)
            btn_cancel = msg.addButton("Cancelar", QMessageBox.RejectRole)

            msg.exec_()

            if msg.clickedButton() == btn_640:
                self.resolucion_foto_elegida = (640, 360)
                self.resolucion_nombre = "640x360"
            elif msg.clickedButton() == btn_960:
                self.resolucion_foto_elegida = (960, 540)
                self.resolucion_nombre = "960x540"
            elif msg.clickedButton() == btn_orig:
                self.resolucion_foto_elegida = None
                self.resolucion_nombre = "original"
            else:
                return

        # 2️⃣ Obtener frame visible
        pixmap = self.image_label.pixmap()
        if not pixmap or pixmap.isNull():
            QMessageBox.warning(self, "Error", "No hay imagen visible para capturar.")
            return

        qimg = pixmap.toImage()
        ptr = qimg.constBits()
        ptr.setsize(qimg.byteCount())

        arr = np.array(ptr).reshape(qimg.height(), qimg.width(), 4)
        frame_con_blurs = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

        # 3️⃣ Aplicar resolución
        if self.resolucion_foto_elegida is not None:
            target_w, target_h = self.resolucion_foto_elegida
            frame_final = cv2.resize(
                frame_con_blurs,
                (target_w, target_h),
                interpolation=cv2.INTER_AREA
            )
            subcarpeta = f"FOTOS/{self.resolucion_nombre}"
        else:
            frame_final = frame_con_blurs.copy()
            subcarpeta = "FOTOS/original"

        # 4️⃣ Construir nombre profesional
        if hasattr(self, "nombre_video_base") and self.video_path:
            nombre_video = "".join(
                c if c.isalnum() or c in " _-()" else "_"
                for c in self.nombre_video_base
            )
        else:
            nombre_video = "video_sin_nombre"

        timestamp = datetime.now().strftime("%H%M%S_%f")[:-4]
        frame_str = f"{self.frame_idx:08d}"
        nombre_archivo = f"{nombre_video}_f{frame_str}_{timestamp}.jpg"

        os.makedirs(subcarpeta, exist_ok=True)
        ruta_final = os.path.join(subcarpeta, nombre_archivo)

        # 5️⃣ Guardar UNA sola vez
        exito = cv2.imwrite(
            ruta_final,
            frame_final,
            [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        )

        # 6️⃣ Verificar guardado
        if exito:
            log(f"[FOTO] Guardada → {ruta_final}")
            self.mostrar_flash_foto()
            self.image_label.setFocus()
        else:
            log(f"[FOTO] ERROR al guardar → {ruta_final}")
            QMessageBox.critical(self, "Error", "No se pudo guardar la imagen.")


    
    def mostrar_flash_foto_backup20251204(self):
        """Muestra un pequeño flash blanco o mensaje visual al tomar una foto."""
        if not hasattr(self, "flash_foto_label"):
            self.flash_foto_label = QLabel(self.image_label)
            # self.flash_foto_label.setStyleSheet(
            #     "background-color: white; border-radius: 10px;"
            # )
            self.flash_foto_label.setStyleSheet("color: yellow; font-size: 22px; font-weight: bold;")
            self.flash_foto_label.setText("📸 Foto guardada")
            self.flash_foto_label.setGeometry(0, 0, self.image_label.width(), self.image_label.height())
            self.flash_foto_label.setAlignment(Qt.AlignCenter)
            self.flash_foto_label.setVisible(False)

        self.flash_foto_label.raise_()
        self.flash_foto_label.setVisible(True)

        # Efecto de opacidad (fade-out)
        efecto = QGraphicsOpacityEffect()
        self.flash_foto_label.setGraphicsEffect(efecto)
        anim = QPropertyAnimation(efecto, b"opacity")
        anim.setDuration(600)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.finished.connect(lambda: self.flash_foto_label.setVisible(False))
        anim.start() 
    
    def mostrar_flash_foto(self):
        """
        Muestra un efecto de 'flash' blanco breve en toda la pantalla al tomar una foto.
        Muy visual, profesional y no molesta.
        """
        # Crear un widget overlay transparente que cubra toda la ventana
        self.flash_overlay = QWidget(self)
        self.flash_overlay.setStyleSheet("background-color: white;")
        self.flash_overlay.setWindowFlags(Qt.FramelessWindowHint | Qt.Widget)
        self.flash_overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.flash_overlay.resize(self.size())
        self.flash_overlay.move(0, 0)
        self.flash_overlay.show()

        # Animación de opacidad: aparece → desaparece rápido
        self.flash_opacity = QGraphicsOpacityEffect(self.flash_overlay)
        self.flash_overlay.setGraphicsEffect(self.flash_opacity)

        self.anim_flash = QPropertyAnimation(self.flash_opacity, b"opacity")
        self.anim_flash.setDuration(600)        # 600 ms total
        self.anim_flash.setStartValue(0.0)
        self.anim_flash.setKeyValueAt(0.15, 0.7)  # pico de brillo
        self.anim_flash.setEndValue(0.0)

        # Al terminar, eliminar el overlay
        self.anim_flash.finished.connect(self.flash_overlay.deleteLater)

        self.anim_flash.start()

    #METODO DE FUNCIONA DE PANTALLA COMPLETA CON F11 AMPLIA, ESC O F11 RESTABLECE
    def toggle_pantalla_completa(self):
        if not self.en_pantalla_completa:
            self.old_geometry = self.geometry()
            self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
            self.showFullScreen()
            self.show()
            self.en_pantalla_completa = True
        else:
            self.setWindowFlags(Qt.Window)
            self.showNormal()
            self.setGeometry(self.old_geometry)
            self.show()
            self.en_pantalla_completa = False

        self.centrar_flash_label()  # <- Reposiciona si estaba visible
        # Redibuja el frame si el video está pausado
        if not self.timer.isActive():
            self.show_frame(forzado=True)
   
    #-->METODOS DEL FLASH VISUAL
    def mostrar_flash_velocidad(self):
        texto = f"Velocidad: x{self.velocidad_reproduccion:.2f}"
        self.flash_label.setText(texto)

        # Centrar en el QLabel de video
        w = self.image_label.width()
        h = self.image_label.height()
        self.flash_label.resize(w, 40)
        self.flash_label.move(0, h // 2 - 20)
        self.flash_label.raise_()
        self.flash_label.setVisible(True)

        # Crear efecto de opacidad
        self.flash_opacity = QGraphicsOpacityEffect()
        self.flash_label.setGraphicsEffect(self.flash_opacity)
        self.flash_opacity.setOpacity(1.0)

        # Crear animación
        self.flash_animacion = QPropertyAnimation(self.flash_opacity, b"opacity")
        self.flash_animacion.setDuration(1000)
        self.flash_animacion.setStartValue(1.0)
        self.flash_animacion.setEndValue(0.0)

        # Conectar para ocultar al terminar
        self.flash_animacion.finished.connect(lambda: self.flash_label.setVisible(False))
        self.flash_animacion.finished.connect(lambda: self.flash_label.setGraphicsEffect(None))
        self.flash_animacion.start()


        # Al terminar, ocultar y limpiar
    def ocultar_foto_flash(self):
        self.flash_foto_label.setVisible(False)
        self.flash_foto_label.setGraphicsEffect(None)
        self.flash_foto_animacion = None
        self.flash_foto_opacity = None

    def ocultar_flash(self):
        self.flash_label.setVisible(False)
        self.flash_label.setGraphicsEffect(None)
        self.flash_opacity = None
        self.flash_animacion = None
        self.flash_foto_label.setVisible(True)

        # Aplicar efecto de opacidad
        self.flash_foto_opacity = QGraphicsOpacityEffect()
        self.flash_foto_label.setGraphicsEffect(self.flash_foto_opacity)
        self.flash_foto_opacity.setOpacity(1.0)

        # Animación
        self.flash_foto_animacion = QPropertyAnimation(self.flash_foto_opacity, b"opacity")
        self.flash_foto_animacion.setDuration(1000)
        self.flash_foto_animacion.setStartValue(1.0)
        self.flash_foto_animacion.setEndValue(0.0)

        # Asegurarse de mantener referencia viva
        self.flash_foto_animacion.finished.connect(lambda: self.flash_foto_label.setVisible(False))
        self.flash_foto_animacion.finished.connect(lambda: self.flash_foto_label.setGraphicsEffect(None))

        self.flash_foto_animacion.start()

    def ocultar_foto_flash(self):
        self.flash_foto_label.setVisible(False)
        self.flash_foto_label.setGraphicsEffect(None)
        self.flash_foto_animacion = None
        self.flash_foto_opacity = None

        self.flash_foto_animacion.finished.connect(self.ocultar_foto_flash)
        self.flash_foto_animacion.start()              
    
    def ocultar_flash(self):# Al terminar, ocultar y limpiar
        self.flash_label.setVisible(False)
        self.flash_label.setGraphicsEffect(None)
        self.flash_opacity = None
        self.flash_animacion = None

        self.flash_animacion.finished.connect(self.ocultar_flash)
        self.flash_animacion.start()
        
    def centrar_flash_label(self):
        if not self.flash_label.isVisible():
            return

        w = self.image_label.width()
        h = self.image_label.height()
        self.flash_label.resize(w, 40)
        self.flash_label.move(0, h // 2 - 20)
        self.flash_label.raise_()
    #<--METODOS DEL FLASH VISUAL

    def _manejar_boton_retroceso(self):
        if self.timer.isActive():
            #nueva_vel = max(0.25, self.velocidad_reproduccion - 0.5)
            # Solo permitir velocidades x8 → x4 → x2 → x1
            if self.velocidad_reproduccion > 4.0:
                nueva_vel = 4.0
            elif self.velocidad_reproduccion > 2.0:
                nueva_vel = 2.0
            elif self.velocidad_reproduccion > 1.0:
                nueva_vel = 1.0
            else:
                nueva_vel = 1.0  # mínimo permitido

            if nueva_vel != self.velocidad_reproduccion:
                self.velocidad_reproduccion = nueva_vel
                log(f"[VELOCIDAD] Disminuyendo velocidad: x{self.velocidad_reproduccion:.2f}")
                intervalo = int(1000 / (self.fps * self.velocidad_reproduccion))
                self.timer.setInterval(intervalo)  # ← en lugar de reiniciar todo con play_video()
                self.frame_label.setText(f"Frame: {self.frame_idx}   |   Velocidad: x{self.velocidad_reproduccion:.2f}")
                self.mostrar_flash_velocidad()
        else:
            self._iniciar_navegacion_continua(-1)

    #avance boton #velocidad
    def _manejar_boton_avance(self):
        if self.timer.isActive():
            # Solo permitir velocidades x2, x4, x8
            if self.velocidad_reproduccion < 2.0:
                nueva_vel = 2.0
            elif self.velocidad_reproduccion < 4.0:
                nueva_vel = 4.0
            elif self.velocidad_reproduccion < 8.0:
                nueva_vel = 8.0
            elif self.velocidad_reproduccion < 16.0:
                nueva_vel = 16.0    
            else:
                nueva_vel = 16.0  # ya está en el máximo

            if nueva_vel != self.velocidad_reproduccion:
                self.velocidad_reproduccion = nueva_vel
                log(f"[VELOCIDAD] Aumentando velocidad: x{self.velocidad_reproduccion:.2f}")
                intervalo = int(1000 / (self.fps * self.velocidad_reproduccion))
                self.timer.setInterval(intervalo)
                self.frame_label.setText(f"Frame: {self.frame_idx}   |   Velocidad: x{self.velocidad_reproduccion:.2f}")
                self.mostrar_flash_velocidad()
            else:
                self._iniciar_navegacion_continua(1)

    #key
    def keyPressEvent(self, event):
        tecla = event.key()
        ahora = time.time()

        # Control antispam básico: bloquear si pasan < 0.1s entre eventos
        if not hasattr(self, "_ultimo_evento_key"):
            self._ultimo_evento_key = 0
        if ahora - self._ultimo_evento_key < 0.1:
            event.ignore()
            return
        self._ultimo_evento_key = ahora

        if tecla == Qt.Key_F11:
            self.toggle_pantalla_completa()
            event.accept()
            return

        if tecla == Qt.Key_Escape and self.en_pantalla_completa:
            self.toggle_pantalla_completa()
            event.accept()
            return

        if tecla == Qt.Key_Space:
            origen = "tecla_space_toggle_play"
            log(f"[KEY] SPACE presionado → toggle_play() | origen: {origen}")
            self.origen_show_frame = origen
            self.toggle_play()
            event.accept()
            return

        if tecla == Qt.Key_Right:
            if self.timer.isActive():
                origen = "tecla_right_modo_reproduccion"
                log(f"[KEY] RIGHT presionado mientras reproducía | origen: {origen}")
                self.origen_show_frame = origen
                self._manejar_boton_avance()
            else:
                origen = "tecla_right_manual"
                log(f"[KEY] RIGHT presionado en pausa | origen: {origen}")
                self.origen_show_frame = origen
                self._iniciar_navegacion_continua(1)
            event.accept()
            return

        if tecla == Qt.Key_Left:
            if self.timer.isActive():
                origen = "tecla_left_modo_reproduccion"
                log(f"[KEY] LEFT presionado mientras reproducía | origen: {origen}")
                self.origen_show_frame = origen
                self._manejar_boton_retroceso()
            else:
                origen = "tecla_left_manual"
                log(f"[KEY] LEFT presionado en pausa | origen: {origen}")
                self.origen_show_frame = origen
                self._iniciar_navegacion_continua(-1)
            event.accept()
            return

        # ---- NUEVO: Navegación con teclas ↑ ↓ para saltos fijos (solo en pausa) ----
        if not self.timer.isActive():  # solo si está pausado
            salto_frames = int(4 * self.fps)

            if tecla == Qt.Key_Down:
                self.frame_idx = min(self.frame_idx + salto_frames, self.total_frames - 1)
                origen = "tecla_down_salto_4s"
                log(f"[KEY] DOWN presionado → salto +{salto_frames} frames | origen: {origen}")
                self.origen_show_frame = origen
                self.show_frame(forzado=True)
                event.accept()
                return

            if tecla == Qt.Key_Up:
                self.frame_idx = max(self.frame_idx - salto_frames, 0)
                origen = "tecla_up_salto_4s"
                log(f"[KEY] UP presionado → retroceso -{salto_frames} frames | origen: {origen}")
                self.origen_show_frame = origen
                self.show_frame(forzado=True)
                event.accept()
                return

        if tecla == Qt.Key_Alt:
            self.capturar_foto()
            event.accept()
            return

        super().keyPressEvent(event)


    def keyReleaseEvent(self, event):
        if event.key() in [Qt.Key_Right, Qt.Key_Left]:
            log(f"[KEY_RELEASE] {event.text()} liberada")
            self._detener_navegacion_continua()
            event.accept()
        else:
            super().keyReleaseEvent(event)
            
    def toggle_play(self):
        if self.timer.isActive() or self.frame_nav_timer.isActive():
            log("[TOGGLE_PLAY] Se detiene reproducción")
            self.pause_video()
        else:
            log("[TOGGLE_PLAY] Se inicia reproducción")
            self.play_video()      

    def _navegar_por_tecla_mantenida(self):
        if self._tecla_izquierda_presionada:
            log("[KEYBOARD] Tecla ← mantenida, retrocediendo frame")
            self._cambiar_frame(-1)
        elif self._tecla_derecha_presionada:
            log("[KEYBOARD] Tecla → mantenida, avanzando frame")
            self._cambiar_frame(+1)
    
    def _frame_nav_step(self):
        if self._step_en_progreso:
            return  # evita encimar pasos
        
        if self._frame_nav_direction == -1:
            self.ir_frame_anterior()
        elif self._frame_nav_direction == 1:
            self.ir_frame_siguiente()
            
    def ir_frame_anterior(self):
        if self.cap and self.frame_idx > 0:
            self.frame_idx -= 1
            self.show_frame(forzado=True)

            # 👈 Recorta el final del tracking al frame actual cuando vas hacia atrás en pausa
            if getattr(self, "blurs_en_construccion", None):
                for blur in self.blurs_en_construccion:
                    if blur.get("tracking_activado") and "frame_ultimo_B" in blur:
                        if self.frame_idx < int(blur["frame_ultimo_B"]):
                            blur["frame_ultimo_B"] = int(self.frame_idx)

                    # 👇 recorte in-place de tracker_hist:
                    if blur.get("tracker_hist"):
                        hist = blur["tracker_hist"]
                        if hist and any(f > self.frame_idx for f, _ in hist):
                            blur["tracker_hist"] = [(f, r) for (f, r) in hist if f <= self.frame_idx]


            # Paso “estático” del tracking al retroceder (no update hacia atrás)
            self._aplicar_tracking_en_frame_estatico(self.frame_idx)

            self.image_label.update()
  
            
    def _cambiar_frame(self, delta):
        nuevo_frame = self.frame_idx + delta
        if 0 <= nuevo_frame < self.total_frames:
            self.frame_idx = nuevo_frame
            self.show_frame(forzado=True)
            
    def ir_frame_siguiente(self):
        if self._step_en_progreso:
            return
        self._step_en_progreso = True
        try:
            if self.cap and self.frame_idx < self.total_frames - 1:
                self.frame_idx += 1
                self.show_frame(forzado=True)

                # Avanza SIEMPRE el tracker hacia adelante en pausa
                if any(b.get("tracking_activado") for b in getattr(self, "blurs_en_construccion", [])):
                    self.actualizar_trackers_blurs_en_construccion(self.frame_idx, self.current_frame)

                self.image_label.update()
        finally:
            self._step_en_progreso = False
            
    def _iniciar_navegacion_continua(self, direccion):
        if self.timer.isActive():
            log("[TECLA] Navegación ignorada: video está reproduciéndose.")
            return

        self._frame_nav_direction = direccion
        self._frame_nav_step()  # primer avance inmediato

        if not self.frame_nav_timer.isActive():
            self.frame_nav_timer.start()

    def _detener_navegacion_continua(self):
        self.frame_nav_timer.stop()
        self._frame_nav_direction = 0
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        
        new_size = self.image_label.size()
        delta_w = abs(new_size.width() - getattr(self, "_last_resize_w", 0))
        delta_h = abs(new_size.height() - getattr(self, "_last_resize_h", 0))

        # Solo redibuja si el cambio fue mayor a 10 píxeles
        if delta_w > 10 or delta_h > 10:
            if hasattr(self, 'current_frame') and self.current_frame is not None:
                self.show_frame(forzado=True)
            self._last_resize_w = new_size.width()
            self._last_resize_h = new_size.height()
            
    def aplicar_eliminacion_progresiva(self):
        nuevas_coords = []
        for rect_objetivo, fallos, frame_origen in self.coords_para_eliminar:
            eliminado = False

            # IA 
            frame_ia = self.blur_ia_por_frame.get(self.frame_idx, [])
            for rect in frame_ia:
                iou = calcular_iou(rect, rect_objetivo)  # Calcular IoU para depuración
                if rects_similares_iou(rect, rect_objetivo, umbral_iou=0.1):  # 🔄 Umbral reducido a 0.3
                    if not any(rects_similares_iou(rect, r, umbral_iou=0.1) for r, _, _ in self.coords_para_eliminar):
                        self.coords_para_eliminar.append((rect, 0, self.frame_idx))
                        log(f"[PROPAGACIÓN IA] Eliminación registrada para {rect} desde frame {self.frame_idx} (IoU: {iou:.3f})")
                    eliminado = True
                else:
                    log(f"[PROPAGACIÓN IA] No similar en frame {self.frame_idx}: {rect} con {rect_objetivo} (IoU: {iou:.3f})")

            # Manual
            frame_manual = self.blur_manual_por_frame.get(self.frame_idx, [])
            nueva_manual = []
            for r in frame_manual:
                iou = calcular_iou(r, rect_objetivo)  # Calcular IoU para depuración
                if not rects_similares_iou(r, rect_objetivo, umbral_iou=0.3):  # 🔄 Umbral reducido a 0.3
                    nueva_manual.append(r)
                else:
                    log(f"[PROPAGACIÓN MANUAL] Eliminado rect manual {r} en frame {self.frame_idx} (IoU: {iou:.3f})")
                    eliminado = True

            if len(nueva_manual) < len(frame_manual):
                self.blur_manual_por_frame[self.frame_idx] = nueva_manual

            if eliminado:
                nuevas_coords.append((rect_objetivo, 0, frame_origen))
            elif fallos + 1 < 120:  # 🔄 Límite aumentado a 120
                nuevas_coords.append((rect_objetivo, fallos + 1, frame_origen))
            else:
                log(f"[PROGRESIVA] Eliminación detenida para {rect_objetivo} en frame {self.frame_idx} tras {fallos + 1} fallos")

        self.coords_para_eliminar = nuevas_coords

    def load_video(self):
        # Al comienzo de load_video (una sola vez)
        open("log_visual_gui.txt", "w", encoding="utf-8").close()

        filtro_videos = "Videos (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm)"
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar video", "", filtro_videos)
        if path:
            log(f"[LOAD] Iniciando carga de video: {path}")
            self.cap = cv2.VideoCapture(path)

            # Limpiar el log visual anterior
            with open("log_visual_gui.txt", "w", encoding="utf-8") as f:
                f.write("[INICIO DE NUEVA SESIÓN VISUAL]\n")

            self.video_path = path

            self.bitrate_original_kbps = self._detectar_bitrate_original(path)#nueva linea para capturar los bitrates originales
            log(f"[LOAD] Bitrate original detectado: {self.bitrate_original_kbps} kbps")

            self.nombre_video_base = os.path.splitext(os.path.basename(path))[0]
            log(f"[LOAD] Nombre base del video cargado: {self.nombre_video_base}")

            # Detectar FPS real
            fps_detectado = self.cap.get(cv2.CAP_PROP_FPS)
            log(f"[FPS DETECTADO] {fps_detectado:.6f}")

            # Establecer FPS real o fallback
            if fps_detectado > 1.0:
                self.fps = fps_detectado
                log(f"[FPS USADO] Se usará FPS detectado del video: {self.fps:.3f}")
            else:
                self.fps = 30.0  # Valor por defecto si la detección falla
                log("[FPS FALLBACK] FPS no válido detectado. Se usará 30.0 por defecto")


            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            log(f"[LOAD] Total de frames: {self.total_frames}")
            self.frame_idx = 0

            # SLIDER
            self.slider.setMaximum(self.total_frames - 1)
            self.slider.setEnabled(True)

            # Mostrar primer frame sin procesar IA
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                log("[DEBUG] Llamando show_frame(forzado=True) desde load_video")
                self.show_frame(forzado=True)
            else:
                log("[LOAD] Error al leer el primer frame")

        log(f"[DEBUG] Timer activo tras cargar video: {self.timer.isActive()}")
        self.timer.stop()
        log(f"[DEBUG] Timer detenido manualmente en load_video")
        log(f"[DEBUG] Timer activo tras detenerlo: {self.timer.isActive()}")

    def _detectar_bitrate_original(self, ruta):
        try:
            import subprocess
            result = subprocess.run([
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=bit_rate",
                "-of", "default=nokey=1:noprint_wrappers=1",
                ruta
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            bitrate_bps = int(result.stdout.strip())
            return bitrate_bps // 1000  # Convertir a kbps
        except Exception as e:
            log(f"[BITRATE] No se pudo detectar el bitrate original: {e}")
            return 5000  # Valor por defecto si falla

    def generar_log_visual_debug(self, frame_idx):
        # Desactivar log si app_instance aún no existe o DEBUG_MODE está desactivado
        if "app_instance" in globals():
            try:
                if hasattr(app_instance, "DEBUG_MODE") and not app_instance.DEBUG_MODE:
                    return
            except Exception:
                return
        ruta = "log_visual_gui.txt"
        rects_ia = self.blur_ia_por_frame.get(frame_idx, [])
        rects_manual = self.blur_manual_por_frame.get(frame_idx, [])
        eliminados_ia_expl = self.blur_eliminado_ia.get(frame_idx, [])
        coords_prop = [r for r, _, origen in self.coords_para_eliminar if frame_idx >= origen]

        with open(ruta, "a", encoding="utf-8") as f:
            f.write(f"\n[FRAME {frame_idx}] ESTADO VISUAL\n")

            for i, rect in enumerate(rects_ia):
                estado = "IGNORADO"
                if not any(rects_similares(rect, r) for r in eliminados_ia_expl) and \
                   not any(rects_similares(rect, r) for r in coords_prop):
                    estado = "VISIBLE"
                f.write(f"  [IA #{i}] {rect} → {estado}\n")

            for i, rect in enumerate(rects_manual):
                f.write(f"  [MANUAL #{i}] {rect}\n")

            f.write(f"  [ELIMINADOS IA EXPLÍCITOS]: {len(eliminados_ia_expl)}\n")
            for r in eliminados_ia_expl:
                f.write(f"     {r}\n")

            f.write(f"  [PROPAGACIÓN ELIMINACIÓN]: {len(coords_prop)}\n")
            for r in coords_prop:
                f.write(f"     {r}\n")

    def show_frame(self, forzado=False):
        log(f"[SHOW_FRAME] ← llamado desde: {getattr(self, 'origen_show_frame', 'desconocido')} | forzado={forzado} | frame_idx={self.frame_idx}")
        log_ia_click(f"[SHOW_FRAME] Renderizando frame {self.frame_idx}")

        if not self.cap:
            return

        if forzado:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
            self.cap.grab()
            ret, frame = self.cap.retrieve()
        else:
            frame = self.current_frame.copy() if self.current_frame is not None else None
            ret = frame is not None

        if not ret:
            log(f"[SHOW_FRAME] ️ No se pudo leer el frame {self.frame_idx}")
            return

        self.current_frame = frame

        rects_ia = self.blur_ia_por_frame.get(self.frame_idx, [])
        rects_manual = self.blur_manual_por_frame.get(self.frame_idx, [])
        eliminados_ia = self.blur_eliminado_ia.get(self.frame_idx, [])

        log_visual_gui(f"[VISUAL {self.frame_idx}] blur_ia={len(rects_ia)}, blur_manual={len(rects_manual)}, eliminados_ia={len(eliminados_ia)}")

        for i, rect in enumerate(rects_ia):
            omitido = any(rect[:4] == r[:4] for r in eliminados_ia)
            estado = "OCULTO" if omitido else "VISIBLE"
            log_visual_gui(f"[VISUAL {self.frame_idx}] IA #{i} {rect} → {estado}")


        for i, rect in enumerate(rects_manual):
            log_visual_gui(f"[VISUAL {self.frame_idx}] MANUAL #{i} {rect} → VISIBLE")

        self.generar_log_show_frame(self.frame_idx, rects_ia, eliminados_ia)

        hay_blur_ia = any(
            not any(rects_similares_iou(rect, r, umbral_iou=0.5) for r in eliminados_ia)
            for rect in rects_ia
        )
        hay_blur_manual = len(rects_manual) > 0
        aplicar_blur = hay_blur_ia or hay_blur_manual

        if aplicar_blur:
            frame_to_show = frame.copy()

            # === Procesar BLUR IA ===
            for rect in rects_ia:
                if any(rect[:4] == r[:4] for r in eliminados_ia):
                    log_ia_click(f"[SHOW_FRAME] OMITIENDO blur IA en frame {self.frame_idx}: {rect}")
                    continue


                if len(rect) >= 4:
                    x, y, w, h = rect[:4]
                else:
                    continue

                aplicar_blur_capsula(frame_to_show, rect, escala=self.blur_escala, kernel=self.blur_kernel)

                exp_x = int(w * (self.blur_escala - 1) / 2)
                exp_y = int(h * (self.blur_escala - 1) / 2)
                x1 = max(0, x - exp_x)
                y1 = max(0, y - exp_y)
                x2 = min(frame_to_show.shape[1], x + w + exp_x)
                y2 = min(frame_to_show.shape[0], y + h + exp_y)

                roi_w = x2 - x1
                roi_h = y2 - y1

                es_clon = any(rects_similares_iou(rect, r_clonado, umbral_iou=0.5)
                            for r_clonado in self.rects_clonados_tmp.get(self.frame_idx, []))

                color_borde = (0, 165, 255) if es_clon else (0, 0, 255)
                cv2.ellipse(frame_to_show,
                            (x1 + roi_w // 2, y1 + roi_h // 2),
                            (roi_w // 2, roi_h // 2),
                            0, 0, 360,
                            color_borde,
                            2)
                log_ia_click(f"[VISUAL] Dibujando blur IA en GUI en frame {self.frame_idx}: {rect}")

            # === Procesar BLUR MANUAL ===
            for rect in rects_manual:
                if len(rect) >= 4:
                    x, y, w, h = rect[:4]
                else:
                    continue

                aplicar_blur_capsula(frame_to_show, rect, escala=self.blur_escala, kernel=self.blur_kernel)

                exp_x = int(w * (self.blur_escala - 1) / 2)
                exp_y = int(h * (self.blur_escala - 1) / 2)
                x1 = max(0, x - exp_x)
                y1 = max(0, y - exp_y)
                x2 = min(frame_to_show.shape[1], x + w + exp_x)
                y2 = min(frame_to_show.shape[0], y + h + exp_y)

                roi_w = x2 - x1
                roi_h = y2 - y1

                cv2.ellipse(frame_to_show,
                            (x1 + roi_w // 2, y1 + roi_h // 2),
                            (roi_w // 2, roi_h // 2),
                            0, 0, 360,
                            (0, 255, 0),  # verde
                            2)
                log_ia_click(f"[VISUAL] Dibujando blur MANUAL confirmado en GUI en frame {self.frame_idx}: {rect}")
        else:
            frame_to_show = frame.copy()

        # === Mostrar BLURS FANTASMA (en construcción) ===
        # if hasattr(self, "blurs_en_construccion"):
        #     for blur_temp in self.blurs_en_construccion:
        #         if self.frame_idx >= blur_temp["frame_ultimo_B"]:
        #             rect = blur_temp["rect_B"]
        #             if len(rect) >= 4:
        #                 x, y, w, h = rect[:4]
        #             else:
        #                 continue
        #             aplicar_blur_capsula(frame_to_show, rect, escala=self.blur_escala, kernel=self.blur_kernel)

        #             exp_x = int(w * (self.blur_escala - 1) / 2)
        #             exp_y = int(h * (self.blur_escala - 1) / 2)
        #             x1 = max(0, x - exp_x)
        #             y1 = max(0, y - exp_y)
        #             x2 = min(frame_to_show.shape[1], x + w + exp_x)
        #             y2 = min(frame_to_show.shape[0], y + h + exp_y)
        #             roi_w = x2 - x1
        #             roi_h = y2 - y1

        #             cv2.ellipse(frame_to_show,
        #                         (x1 + roi_w // 2, y1 + roi_h // 2),
        #                         (roi_w // 2, roi_h // 2),
        #                         0, 0, 360,
        #                         (0, 255, 255),  # amarillo
        #                         2)
        #             #print(f"[FANTASMA] Mostrando blur fantasma desde frame {blur_temp['frame_ultimo_B']} hasta {self.frame_idx}: {rect}")

        # === Mostrar frame final ===
        rgb_image = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        label_width = self.image_label.parent().width()
        label_height = self.image_label.parent().height()
        scaled_pixmap = pixmap.scaled(label_width, label_height, Qt.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)
        self.flash_label.raise_()

        self.frame_label.setText(f"Frame: {self.frame_idx}   |   Velocidad: x{self.velocidad_reproduccion:.2f}")
        self.slider.blockSignals(True)
        self.slider.setValue(self.frame_idx)
        self.slider.blockSignals(False)

        tiempo_actual = self.formato_tiempo(int(self.frame_idx / self.fps))
        tiempo_total = self.formato_tiempo(self.total_frames / self.fps)
        self.time_label.setText(f"{tiempo_actual} / {tiempo_total}")

    def punto_dentro_de_rect(punto, rect):
        x, y = punto
        rx, ry, rw, rh = rect[:4]
        return rx <= x <= rx + rw and ry <= y <= ry + rh
    
    def registrar_eliminacion(self, frame, rect, fuente):
        if frame not in self.blur_eliminado_ia:
            self.blur_eliminado_ia[frame] = []
        if not any(rects_similares(rect, r) for r in self.blur_eliminado_ia[frame]):
            self.blur_eliminado_ia[frame].append(rect)
            log_ia_click(f"[REGISTRO] {fuente} agregó eliminación en frame {frame}: {rect}")
        else:
            log_ia_click(f"[REGISTRO] {fuente} omitido (ya estaba) en frame {frame}: {rect}")
    
    #clic
    def handle_mouse_click(self, event):
        if not hasattr(self, "current_frame") or self.current_frame is None:
            return
        
        # Solo permitir acciones si el video está pausado o detenido
        if self.timer.isActive():
            log_salto_frames("[ERROR] handle_mouse_click Se ignoró clic porque el video está reproduciéndose (timer activo).")
            return
            
        # Log: Estado del timer y datos iniciales del evento
        log_salto_frames(f"[MOUSE_CLICK] handle_mouse_click Timer activo: {self.timer.isActive()}, frame_idx: {self.frame_idx}, botón: {event.button()}")

        # Chequeo: Si no hay frame cargado, ignora el evento
        if not hasattr(self, 'current_frame') or self.current_frame is None:
            log_salto_frames("[MOUSE_CLICK] handle_mouse_click Ignorado porque no hay video cargado.")
            return

        # Solo manejamos el clic izquierdo aquí
        
        if event.button() == Qt.LeftButton:
            log_salto_frames(f"[MOUSE_CLICK] handle_mouse_click Clic izquierdo detectado en frame {self.frame_idx}")

            self.pause_video()
            
            # 1) Etiquetas (❌, ⚓, 🔍…)
            if self.click_en_etiqueta(event.pos().x(), event.pos().y()):
                return

            # 2) Asa SE (usar cajas GUI guardadas por paintEvent)
            if self._click_en_handle(event.pos().x(), event.pos().y()):
                return
            
            # Obtiene dimensiones del label (área visible en la GUI)
            label_width = self.image_label.width()
            label_height = self.image_label.height()
            frame_height, frame_width = self.current_frame.shape[:2]

            # Calcula escala y offset para mapear el clic del QLabel al frame de video
            scale = min(label_width / frame_width, label_height / frame_height)
            new_w = int(frame_width * scale)
            new_h = int(frame_height * scale)
            offset_x = (label_width - new_w) // 2
            offset_y = (label_height - new_h) // 2

            # Ajusta la posición del mouse por el offset
            click_x = event.pos().x() - offset_x
            click_y = event.pos().y() - offset_y

            # Verifica si el clic está dentro del área de video, si no, ignora
            if not (0 <= click_x <= new_w and 0 <= click_y <= new_h):
                log_salto_frames(f"[MOUSE_CLICK] handle_mouse_click Clic izquierdo fuera de área de video en pos: {event.pos()} | offset=({offset_x},{offset_y}) | new_w={new_w} | new_h={new_h}")
                event.accept()
                return

            # Convierte a coordenadas de video
            x = int(click_x / scale)
            y = int(click_y / scale)
            log_salto_frames(f"[MOUSE_CLICK] handle_mouse_click Coordenadas en video: ({x}, {y}), escala usada: {scale:.3f}")

            
            # Primero, verificar si tocó algún blur fantasma para mover 
            for blur in self.blurs_en_construccion:
                # 🚫 Si el blur está en tracking, impedir interacción antes de su frame de inicio
                if blur.get("tracking_activado") and self.frame_idx < blur.get("frame_inicio_tracking", 0):
                    print(f"[TRACKING] Ignorado clic: frame {self.frame_idx} antes del inicio de tracking ({blur.get('frame_inicio_tracking')}).")
                    continue

                for rect in [blur["rect_A"], blur["rect_B"]]:
                    bx, by, bw, bh = rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self.modo_mover_blur_fantasma = True
                        self.blur_fantasma_moviendose = blur
                        self.offset_mover_blur = (x - bx, y - by)
                        print(f"[FANTASMA] Iniciando movimiento de blur fantasma ID {blur['ID']} en frame {self.frame_idx}: {rect}")
                        return  # ← Salimos para evitar que entre en modo dibujo


            # 🟢 Solo si NO tocó blur fantasma y clic fue válido, activa dibujo
            self.drawing = True
            self.start_point = event.pos()
            self.end_point = event.pos()
            
            # Inicializa la lista si no existe aún
            if not hasattr(self, "blurs_en_construccion") or self.blurs_en_construccion is None:
                self.blurs_en_construccion = []
            
            # Acepta el evento para que no siga propagándose
            event.accept()
        elif event.button() == Qt.RightButton:
            # 🖱️ Clic derecho (prioridad: FANTASMA → MANUAL → IA) con returns tempranos
            print(f"[MOUSE_CLICK] handle_mouse_click Clic derecho detectado en frame {self.frame_idx}")

            # --- Mapeo coords GUI → frame ---
            label_width = self.image_label.width()
            label_height = self.image_label.height()
            frame_height, frame_width = self.current_frame.shape[:2]

            scale = min(label_width / frame_width, label_height / frame_height)
            new_w = int(frame_width * scale)
            new_h = int(frame_height * scale)
            offset_x = (label_width - new_w) // 2
            offset_y = (label_height - new_h) // 2

            click_x_gui = event.pos().x()
            click_y_gui = event.pos().y()
            click_x = click_x_gui - offset_x
            click_y = click_y_gui - offset_y

            if not (0 <= click_x <= new_w and 0 <= click_y <= new_h):
                print(f"[MOUSE_CLICK] Clic derecho fuera de área de video en pos GUI=({click_x_gui},{click_y_gui})")
                event.accept()
                return

            x = int(click_x / scale)
            y = int(click_y / scale)
            print(f"[MOUSE_CLICK] Coordenadas de video: ({x},{y}) | scale={scale:.3f}")

            # ========== 1) PRIORIDAD: FANTASMA ==========
            def _hit_fantasma(xv, yv):
                if not hasattr(self, "blurs_en_construccion") or not self.blurs_en_construccion:
                    return None
                for blur in self.blurs_en_construccion:
                    for rect in (blur.get("rect_B"), blur.get("rect_A")):
                        if not rect:
                            continue
                        bx, by, bw, bh = rect
                        if bx <= xv <= bx + bw and by <= yv <= by + bh:
                            return blur
                return None

            blur_fantasma = _hit_fantasma(x, y)
            if blur_fantasma is not None:
                print(f"[PRIORIDAD] Fantasma tocado en frame {self.frame_idx}. Ejecutando SOLO su llamada y return.")
                # 🔹 Llamada propia del fantasma (mantengo tu firma original)
                #    Si tu método crea/gestiona los manuales o confirma el fantasma, lo hará aquí.
                try:
                    self.interpolar_rect_fantasma(x, y)
                except Exception as e:
                    print(f"[PRIORIDAD][FANTASMA] Error en interpolar_rect_fantasma: {e}")
                self.show_frame(forzado=True)
                event.accept()
                return  # ⛔️ Nada más (no eliminar MANUAL ni IA)

            # ========== 2) MANUAL (solo si NO hubo fantasma) ==========
            manuales = self.blur_manual_por_frame.get(self.frame_idx, [])
            if manuales:
                to_keep = []
                eliminados_manual = []
                for rect in manuales:
                    rx, ry, rw, rh = rect[:4]
                    if rx <= x <= rx + rw and ry <= y <= y + rh:
                        eliminados_manual.append(rect)
                    else:
                        to_keep.append(rect)

                if eliminados_manual:
                    # 1) Quitar del frame actual
                    self.blur_manual_por_frame[self.frame_idx] = to_keep

                    print(f"[PRIORIDAD] Eliminado(s) {len(eliminados_manual)} rect(s) MANUAL en frame {self.frame_idx}: {eliminados_manual}")

                    # 2) ➜ PROPAGACIÓN HACIA ADELANTE (manual)
                    #    Usa tu método existente; si ya gestiona tolerancias/TTL, no toques nada más.
                    for r in eliminados_manual:
                        try:
                            self.eliminar_rects_frame_sgtes(r, self.frame_idx, tipo="manual")
                        except Exception as e:
                            print(f"[MANUAL][PROPAGACION] Error propagando eliminación desde frame {self.frame_idx} para {r}: {e}")

                    self.show_frame(forzado=True)
                    event.accept()
                    return  # ⛔️ Nada más (no tocar IA)


            # ========== 3) IA (solo si NO hubo fantasma ni manual) ==========
            rects_ia = self.blur_ia_por_frame.get(self.frame_idx, [])
            if rects_ia:
                to_keep_ia = []
                eliminados_ia = []
                for rect in rects_ia:
                    ix, iy, iw, ih = rect[:4]
                    if ix <= x <= ix + iw and iy <= y <= y + ih:
                        eliminados_ia.append(rect)
                    else:
                        to_keep_ia.append(rect)

                if eliminados_ia:
                    # 1) Quitar del frame actual
                    self.blur_ia_por_frame[self.frame_idx] = to_keep_ia

                    # 2) Registrar eliminaciones IA en el frame actual (evitar duplicados exactos)
                    lista_reg = self.blur_eliminado_ia.setdefault(self.frame_idx, [])
                    for r in eliminados_ia:
                        if r not in lista_reg:
                            lista_reg.append(r)

                    print(f"[PRIORIDAD] Eliminado(s) {len(eliminados_ia)} rect(s) IA en frame {self.frame_idx}: {eliminados_ia}")

                    # 3) ➜ PROPAGACIÓN HACIA ADELANTE (clave para falsos positivos)
                    #    Usa tu método existente; si ya gestiona tolerancias/TTL, no toques nada más.
                    for r in eliminados_ia:
                        try:
                            self.eliminar_rects_frame_sgtes(r, self.frame_idx, tipo="ia")
                        except Exception as e:
                            print(f"[IA][PROPAGACION] Error propagando eliminación desde frame {self.frame_idx} para {r}: {e}")

                    self.show_frame(forzado=True)
                    event.accept()
                    return  # ⛔️ Fin IA (no tocar nada más)


            # ========== 4) Nada debajo del clic ==========
            print(f"[MOUSE_CLICK] Clic derecho sin objetivo en frame {self.frame_idx}")
            self.show_frame(forzado=False)
            event.accept()
            return

    def handle_mouse_release(self, event):
        if not hasattr(self, "current_frame") or self.current_frame is None:
            return

        if self.timer.isActive():
            log_salto_frames("[ERROR] handle_mouse_release Se ignoró mouseReleaseEvent porque el video está reproduciéndose (timer activo).")
            return

        log_salto_frames(f"[MOUSE_RELEASE] mouseReleaseEvent con botón {event.button()} en frame {self.frame_idx}")

        if self.modo_mover_blur_fantasma:
            # ✅ Actualizar la lista principal con la posición final del blur
            if self.blur_fantasma_moviendose:
                for blur in self.blurs_en_construccion:
                    if blur["ID"] == self.blur_fantasma_moviendose["ID"]:
                        blur["rect_B"] = self.blur_fantasma_moviendose["rect_B"]
                        blur["frame_ultimo_B"] = self.blur_fantasma_moviendose["frame_ultimo_B"]
                        print(f"[FANTASMA] Actualizado blur ID {blur['ID']} con nueva posición final: {blur['rect_B']}")
                        break

            self.modo_mover_blur_fantasma = False
            self.blur_fantasma_moviendose = None
            self.image_label.update()  # usar update para evitar show_frame forzado
            return

        if self.modo_redimension_fantasma:
            self.modo_redimension_fantasma = False
            self.blur_fantasma_redimensionandose = None
            self._resize_start = None
            self._resize_anchor = None
            self.image_label.update()
            return



        if self.drawing and event.button() == Qt.LeftButton:
            self.drawing = False
            self.procesar_rect_fantasma(self.start_point, self.end_point)

    def handle_mouse_move(self, event):
        if not hasattr(self, "current_frame") or self.current_frame is None:
            return
        
        #print("[DEBUG] modo_mover_blur_fantasma:", self.modo_mover_blur_fantasma)
        #print("[DEBUG] blur_fantasma_moviendose:", self.blur_fantasma_moviendose)
        # Solo permitir acciones si el video está pausado o detenido
        if self.timer.isActive():
            log_salto_frames("[ERROR] handle_mouse_move Se ignoró mouseMoveEvent porque el video está reproduciéndose (timer activo).")
            return

        # Si estamos moviendo un blur fantasma
        if self.modo_mover_blur_fantasma and self.blur_fantasma_moviendose:
            label_width = self.image_label.width()
            label_height = self.image_label.height()
            frame_height, frame_width = self.current_frame.shape[:2]
            scale = min(label_width / frame_width, label_height / frame_height)
            if scale <= 0:
                return  # seguridad

            new_w = int(frame_width * scale)
            new_h = int(frame_height * scale)
            offset_x = (label_width - new_w) // 2
            offset_y = (label_height - new_h) // 2

            # 1) Mouse en coords GUI relativas al área de video
            mx_gui = event.pos().x() - offset_x
            my_gui = event.pos().y() - offset_y

            # (opcional) Limitar el cursor a la zona visible del video para evitar “arrastres” raros
            mx_gui = max(0, min(mx_gui, new_w - 1))
            my_gui = max(0, min(my_gui, new_h - 1))

            # 2) GUI → video
            vx = mx_gui / scale
            vy = my_gui / scale

            # 3) Aplica el offset (offset está en coords de video)
            fx = vx - self.offset_mover_blur[0]
            fy = vy - self.offset_mover_blur[1]

            # 4) Clamp al frame (NECESITAMOS w,h antes del clamp)
            w, h = self.blur_fantasma_moviendose["rect_B"][2:]
            fx, fy, w, h = self._clamp_rect_a_frame(int(round(fx)), int(round(fy)), int(w), int(h))

            # 5) Actualiza
            self.blur_fantasma_moviendose["rect_B"] = (fx, fy, w, h)
            self.blur_fantasma_moviendose["frame_ultimo_B"] = self.frame_idx
            self.image_label.update()
            return

        # Si estamos redimensionando el blur fantasma
        if self.modo_redimension_fantasma and self.blur_fantasma_redimensionandose:
            label_w = self.image_label.width()
            label_h = self.image_label.height()
            frame_h, frame_w = self.current_frame.shape[:2]
            scale = min(label_w / frame_w, label_h / frame_h)
            if scale <= 0:
                return
            new_w = int(frame_w * scale)
            new_h = int(frame_h * scale)
            offset_x = (label_w - new_w) // 2
            offset_y = (label_h - new_h) // 2

            gx = event.pos().x()
            gy = event.pos().y()
            vx = int((gx - offset_x) / scale)
            vy = int((gy - offset_y) / scale)

            # Estado inicial
            sx, sy, x0, y0, w0, h0 = self._resize_start

            # Asa SE: x,y anclados; cambian w,h con el mouse
            w = max(4, vx - x0)
            h = max(4, vy - y0)

            # Shift = mantener proporción original
            if QApplication.keyboardModifiers() & Qt.ShiftModifier:
                aspect = w0 / h0 if h0 > 0 else 1.0
                if w / h > aspect:
                    w = int(h * aspect)
                else:
                    h = int(w / aspect)

            # Clamp
            x, y = x0, y0
            x, y, w, h = self._clamp_rect_a_frame(x, y, w, h)

            self.blur_fantasma_redimensionandose["rect_B"] = (x, y, w, h)
            self.blur_fantasma_redimensionandose["frame_ultimo_B"] = self.frame_idx
            self.image_label.update()
            return



        # Si está en modo "dibujo", actualiza el endpoint y refresca el frame para mostrar el rectángulo en tiempo real
        if self.drawing:
            self.end_point = event.pos()
            #log_salto_frames(f"[MOUSE_MOVE] handle_mouse_move mouseMoveEvent con mouse en {event.pos()} en frame {self.frame_idx}")
            #self.show_frame(forzado=True)
            #log(f"[DEBUG] self.show_frame(forzado=False) desde el metodo def handle_mouse_move(self, event):")
            
            self.image_label.update()
            # Llama a update() para que Qt ejecute paintEvent() en el próximo ciclo (dibuja el rect azul).
            # Esto redibuja el rectángulo azul en tiempo real sin recargar el frame.
            # No se repite automáticamente; deja de redibujar cuando se suelta el mouse.
    
    def _clamp_rect_a_frame(self, x, y, w, h):#helper para mantener en el frame de la GUI
        H, W = self.current_frame.shape[:2]
        x = max(0, min(x, W - w))
        y = max(0, min(y, H - h))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))
        return x, y, w, h

    def click_en_etiqueta(self, x, y):
        if not getattr(self, "blurs_en_construccion", None):
            return False

        # Recorre al revés por si hay solapes
        for blur in reversed(self.blurs_en_construccion):
            # ⛔ Sin interacción antes del inicio A
            if self.frame_idx < int(blur.get("frame_inicio_A", 0)):
                continue

            # --- ❌ cerrar ---
            box_x = blur.get("etiqueta_x")
            if box_x:
                tx, ty, tw, th = box_x
                if tx <= x <= tx + tw and ty <= y <= ty + th:
                    print(f"[FANTASMA] Eliminado blur fantasma ID {blur.get('ID')}")
                    self.blurs_en_construccion.remove(blur)
                    self.image_label.update()
                    return True

            # --- ⚓ confirmar & continuar ---
            box_a = blur.get("etiqueta_anchor")
            if box_a:
                ax, ay, aw, ah = box_a
                if ax <= x <= ax + aw and ay <= y <= ay + ah:
                    if blur.get("rect_A") != blur.get("rect_B"):
                        print("[ANCHOR] Clic en ancla detectado, confirmando & continuando…")
                        self.confirmar_interpolacion_y_continuar(blur)
                    else:
                        print("[ANCHOR] A==B; arrastra primero para crear tramo.")
                    return True

            # --- 🔍 activar tracking ---
            box_m = blur.get("etiqueta_mov")  # se registra en paintEvent al dibujar el icono
            if box_m:
                mx, my, mw, mh = box_m
                if mx <= x <= mx + mw and my <= y <= my + mh:
                    blur["tracking_activado"] = True
                    blur["frame_inicio_tracking"] = self.frame_idx  # útil para bloquear interacción atrás
                    # Limpia/siembra estado
                    for k in ("tracker", "tracker_hist", "ema_wh"):
                        blur.pop(k, None)
                    x0, y0, w0, h0 = map(int, blur["rect_B"])
                    blur["tracker_hist"] = [(self.frame_idx, (x0, y0, w0, h0))]

                    print(f"[TRACKING] 🔍 Activado en blur ID={blur.get('ID')} desde frame {self.frame_idx}.")
                    self.image_label.update()
                    return True

        return False

    def _click_en_handle(self, gx, gy):
        if not getattr(self, "blurs_en_construccion", None):
            return False

        for blur in reversed(self.blurs_en_construccion):
            if self.frame_idx < int(blur.get("frame_inicio_A", 0)):
                continue  # ⛔ no resize antes del inicio

        # Recorre al revés por si hay solapes
        for blur in reversed(self.blurs_en_construccion):
            box = blur.get("handle_se")
            if box:
                hx, hy, hw, hh = box
                if hx <= gx <= hx + hw and hy <= gy <= hy + hh:
                    # Init resize mode
                    self.modo_redimension_fantasma = True
                    self.blur_fantasma_redimensionandose = blur

                    # Guarda estado inicial en coords de video
                    frame_h, frame_w = self.current_frame.shape[:2]
                    label_w, label_h = self.image_label.width(), self.image_label.height()
                    scale = min(label_w / frame_w, label_h / frame_h)
                    new_w, new_h = int(frame_w * scale), int(frame_h * scale)
                    offset_x = (label_w - new_w) // 2
                    offset_y = (label_h - new_h) // 2

                    # pos GUI → video
                    vx = int((gx - offset_x) / scale)
                    vy = int((gy - offset_y) / scale)

                    x, y, w, h = blur["rect_B"]
                    self._resize_start = (vx, vy, x, y, w, h)
                    self._resize_anchor = "se"
                    return True

        return False

    def procesar_rect_fantasma(self, start_point: QPoint, end_point: QPoint):
        # 🧮 Obtener dimensiones del QLabel (donde se muestra el video)
        label_width = self.image_label.width()
        label_height = self.image_label.height()

        # 📹 Obtener dimensiones reales del frame del video
        frame_height, frame_width = self.current_frame.shape[:2]

        # 🔍 Calcular escala de ajuste para preservar aspecto del video
        scale = min(label_width / frame_width, label_height / frame_height)

        # 🧭 Calcular tamaño del video escalado
        new_w = int(frame_width * scale)
        new_h = int(frame_height * scale)

        # ↕️↔️ Calcular márgenes negros (barras negras) si existen
        offset_x = (label_width - new_w) // 2
        offset_y = (label_height - new_h) // 2

        # 🎯 Calcular las coordenadas del rectángulo (en la GUI)
        x1 = min(start_point.x(), end_point.x()) - offset_x
        y1 = min(start_point.y(), end_point.y()) - offset_y
        x2 = max(start_point.x(), end_point.x()) - offset_x
        y2 = max(start_point.y(), end_point.y()) - offset_y

        # 🚫 Validación: si el rectángulo quedó fuera del área visible del video
        if not (0 <= x1 <= new_w and 0 <= x2 <= new_w and 0 <= y1 <= new_h and 0 <= y2 <= new_h):
            log_salto_frames(f"[RELEASE FUERA DE VIDEO] QRect fuera de límites: ({x1},{y1})-({x2},{y2})")
            return

        # 🎯 Convertir coordenadas desde GUI (escaladas) a coordenadas reales del video
        fx1 = int(x1 / scale)
        fy1 = int(y1 / scale)
        fx2 = int(x2 / scale)
        fy2 = int(y2 / scale)
        w, h = fx2 - fx1, fy2 - fy1

        if w > 0 and h > 0:
            nuevo_rect = (fx1, fy1, w, h)

            # 🔄 Inicializar lista si no existe
            if not hasattr(self, "blurs_en_construccion"):
                self.blurs_en_construccion = []

            # 📌 Crear nuevo blur fantasma
            if not hasattr(self, "_id_blur_fantasma"):
                self._id_blur_fantasma = 1

            nuevo_blur = {
                "ID": self._id_blur_fantasma,
                "rect_A": nuevo_rect,
                "rect_B": nuevo_rect,
                "frame_inicio_A": self.frame_idx,
                "frame_ultimo_B": self.frame_idx,
                "confirmado": False
            }
            self.blurs_en_construccion.append(nuevo_blur)
            print(f"[DEBUG] Blur fantasma creado: {nuevo_blur}")

            self._id_blur_fantasma += 1

            self.show_frame(forzado=True)

        else:
            log_ia_click("[INFO] Se ignoró intento de crear blur manual sin área válida.")

            # ⚠️ Si el área del rect es inválida (0 o negativa), se ignora
            log_ia_click("[INFO] Se ignoró intento de crear blur manual sin área válida.")

    def interpolar_rect_fantasma(self, x: int, y: int):
        if not self.blurs_en_construccion:
            return

        for blur_temp in list(self.blurs_en_construccion):
            fx, fy, fw, fh = blur_temp["rect_B"]
            if not (fx <= x <= fx + fw and fy <= y <= fy + fh):
                continue

            # 🔍 Si este blur está trackeando o tiene historial, confirmar SOLO hasta el frame actual
            if blur_temp.get("tracking_activado") or blur_temp.get("tracker_hist"):
                # 1) Asegura que rect_B represente el frame actual
                self._aplicar_tracking_en_frame_estatico(self.frame_idx)

                # 2) Recorta el “final lógico”
                blur_temp["frame_ultimo_B"] = int(self.frame_idx)

                # 3) Recorta el historial en sitio (defensivo)
                if blur_temp.get("tracker_hist"):
                    blur_temp["tracker_hist"] = [(f, r) for (f, r) in blur_temp["tracker_hist"] if f <= self.frame_idx]

                print(f"[TRACKING] Recorte en clic derecho: fin={self.frame_idx}, len(hist)={len(blur_temp.get('tracker_hist') or [])}")

                # 4) Confirmación recortada
                self._confirmar_tracker_hasta_frame_actual(blur_temp)
                return


            # === (lo que ya tenías) lógica A→B manual ===
            frame_a = int(blur_temp.get("frame_inicio_A", self.frame_idx))
            frame_b = int(self.frame_idx)  # confirmar al frame actual
            rect_a  = tuple(map(int, blur_temp["rect_A"]))
            rect_b  = tuple(map(int, blur_temp["rect_B"]))

            if frame_b < frame_a:
                print("[INTERPOLACIÓN CANCELADA] frame_ultimo_B < frame_inicio_A")
                return

            if frame_b == frame_a:
                rect_single = rect_b if rect_b != rect_a else rect_a
                self.blur_manual_por_frame.setdefault(frame_a, []).append(rect_single)
                print(f"[CONFIRMADO] Blur manual único en frame {frame_a}: {rect_single}")
            else:
                x1, y1, w1, h1 = rect_a
                x2, y2, w2, h2 = rect_b
                denom = frame_b - frame_a
                for f in range(frame_a, frame_b + 1):
                    t = (f - frame_a) / denom
                    xi = int(round(x1 + (x2 - x1) * t))
                    yi = int(round(y1 + (y2 - y1) * t))
                    wi = int(round(w1 + (w2 - w1) * t))
                    hi = int(round(h1 + (h2 - h1) * t))
                    self.blur_manual_por_frame.setdefault(f, []).append((xi, yi, wi, hi))
                print(f"[INTERPOLADO] Blur manual del frame {frame_a} al {frame_b}")

            # En el flujo manual sí eliminas el fantasma (mantengo tu comportamiento)
            self.blurs_en_construccion.remove(blur_temp)
            self.show_frame(forzado=True)
            break

    def interpolar_rect_fantasma_antiguo(self, x: int, y: int):
        if not self.blurs_en_construccion:
            return

        for blur_temp in list(self.blurs_en_construccion):
            rect_b = blur_temp["rect_B"]  # detección sobre posición actual
            fx, fy, fw, fh = rect_b
            if not (fx <= x <= fx + fw and fy <= y <= fy + fh):
                continue

            frame_a = int(blur_temp.get("frame_inicio_A", self.frame_idx))
            frame_b = int(self.frame_idx)  # confirmar al frame actual
            rect_a  = tuple(map(int, blur_temp["rect_A"]))
            rect_b  = tuple(map(int, blur_temp["rect_B"]))

            if frame_b < frame_a:
                print("[INTERPOLACIÓN CANCELADA] frame_ultimo_B < frame_inicio_A")
                return

            # === CASO 1: MISMO FRAME → crear solo 1 rect ===
            if frame_b == frame_a:
                # Preferimos volcar la posición actual (B) si difiere
                rect_single = rect_b if rect_b != rect_a else rect_a
                self.blur_manual_por_frame.setdefault(frame_a, []).append(rect_single)
                print(f"[CONFIRMADO] Blur manual único en frame {frame_a}: {rect_single}")

            else:
                # === CASO 2: INTERPOLACIÓN A→B EN RANGO [frame_a, frame_b] ===
                x1, y1, w1, h1 = rect_a
                x2, y2, w2, h2 = rect_b
                denom = frame_b - frame_a  # > 0 garantizado aquí

                for f in range(frame_a, frame_b + 1):
                    t = (f - frame_a) / denom
                    xi = int(round(x1 + (x2 - x1) * t))
                    yi = int(round(y1 + (y2 - y1) * t))
                    wi = int(round(w1 + (w2 - w1) * t))
                    hi = int(round(h1 + (h2 - h1) * t))
                    self.blur_manual_por_frame.setdefault(f, []).append((xi, yi, wi, hi))

                print(f"[INTERPOLADO] Blur manual del frame {frame_a} al {frame_b}")

            # ✅ eliminar el fantasma y refrescar
            self.blurs_en_construccion.remove(blur_temp)
            self.show_frame(forzado=True)
            break

    def confirmar_interpolacion_y_continuar(self, blur):
        """
        Vuelca la interpolación de A→B a blur_manual_por_frame y
        reinicia el mismo blur fantasma posicionándolo en B (A=B),
        para poder seguir creando otro tramo rápidamente.
        """
        rect_a = blur.get("rect_A")
        rect_b = blur.get("rect_B")
        frame_a = int(blur.get("frame_inicio_A", self.frame_idx))
        frame_b = int(blur.get("frame_ultimo_B", self.frame_idx))

        # 🔍 Si hay tracking/historial, confirmar RECORTADO hasta el frame actual y salir
        if blur.get("tracking_activado") or blur.get("tracker_hist"):
            # Garantiza que rect_B corresponda al frame actual
            self._aplicar_tracking_en_frame_estatico(self.frame_idx)

            # Recorta final lógico e historial (defensivo)
            blur["frame_ultimo_B"] = int(self.frame_idx)
            if blur.get("tracker_hist"):
                blur["tracker_hist"] = [(f, r) for (f, r) in blur["tracker_hist"] if f <= self.frame_idx]

            # Vuelca solo hasta el frame actual y deja el blur listo en A=B
            self._confirmar_tracker_hasta_frame_actual(blur)
            return

        if not rect_a or not rect_b:
            print("[ANCHOR] Falta rect_A o rect_B.")
            return

        if frame_b < frame_a:
            print("[ANCHOR] frame_ultimo_B < frame_inicio_A; no se confirma.")
            return

        # 1) Crear manuales con interpolación (o sin si A==B)
        x1, y1, w1, h1 = rect_a
        x2, y2, w2, h2 = rect_b
        if (x1, y1, w1, h1) == (x2, y2, w2, h2):
            for f in range(frame_a, frame_b + 1):
                self.blur_manual_por_frame.setdefault(f, []).append((x1, y1, w1, h1))
        else:
            denom = max(1, frame_b - frame_a)
            for f in range(frame_a, frame_b + 1):
                t = (f - frame_a) / denom
                xi = int(x1 + (x2 - x1) * t)
                yi = int(y1 + (y2 - y1) * t)
                wi = int(w1 + (w2 - w1) * t)
                hi = int(h1 + (h2 - h1) * t)
                self.blur_manual_por_frame.setdefault(f, []).append((xi, yi, wi, hi))

        print(f"[ANCHOR] ✅ Interpolación volcada: {frame_a} → {frame_b} (ID={blur.get('ID')})")

        # 2) Reset del blur fantasma para continuar en la última posición
        blur["rect_A"] = rect_b
        blur["frame_inicio_A"] = frame_b
        blur["confirmado"] = False

        # Limpia estado de tracking (nuevo tramo será manual o reactivas 🔍)
        for k in ("tracker", "tracker_hist", "ema_wh"):
            blur.pop(k, None)
        blur["tracking_activado"] = False

        # 3) Refrescar
        self.show_frame(forzado=True)
        self.image_label.update()

    def _confirmar_tracker_hasta_frame_actual(self, blur):
        """
        Convierte el historial del tracker en blurs manuales SOLO hasta self.frame_idx.
        No usa pasos 'hacia adelante' que ya existan en tracker_hist > frame actual.
        Tras confirmar, deja el blur listo para continuar desde aquí (A=B en frame actual).
        """
        hist = list(blur.get("tracker_hist") or [])
        if not hist:
            print("[TRACKING] No hay historial para confirmar (recortado).")
            return

        frame_to = int(self.frame_idx)

        # 1) Recortar el historial al frame actual
        hist = [(f, r) for (f, r) in hist if f <= frame_to]
        if not hist:
            print("[TRACKING] Historial queda vacío al recortar al frame actual; nada que volcar.")
            return

        # 2) Asegurar A al inicio si corresponde
        fA = int(blur.get("frame_inicio_A", hist[0][0]))
        rA = tuple(map(int, (blur.get("rect_A") or hist[0][1])[:4]))
        if fA < hist[0][0]:
            hist = [(fA, rA)] + hist

        # 3) Asegurar punto final EXACTO en frame actual
        rB_actual = tuple(map(int, (blur.get("rect_B") or hist[-1][1])[:4]))
        if hist[-1][0] < frame_to:
            hist.append((frame_to, rB_actual))

        # 4) Interpolar tramo a tramo y volcar a blur_manual_por_frame
        H = W = None
        if hasattr(self, "current_frame") and self.current_frame is not None:
            H, W = self.current_frame.shape[:2]

        def clamp_rect(x, y, w, h):
            if W is None or H is None:
                return int(x), int(y), int(w), int(h)
            x = max(0, min(int(x), W - 1))
            y = max(0, min(int(y), H - 1))
            w = max(1, min(int(w), W - x))
            h = max(1, min(int(h), H - y))
            return x, y, w, h

        creados = 0
        for i in range(len(hist) - 1):
            f1, r1 = hist[i]
            f2, r2 = hist[i + 1]
            x1, y1, w1, h1 = map(int, r1[:4])
            x2, y2, w2, h2 = map(int, r2[:4])

            dt = f2 - f1
            if dt <= 0:
                continue

            for f in range(f1, f2):
                t = (f - f1) / dt
                xi = x1 + (x2 - x1) * t
                yi = y1 + (y2 - y1) * t
                wi = w1 + (w2 - w1) * t
                hi = h1 + (h2 - h1) * t
                rect_i = clamp_rect(xi, yi, wi, hi)

                lst = self.blur_manual_por_frame.setdefault(f, [])
                # evita duplicados razonables
                if not any(calcular_iou(rect_i, r) >= 0.6 for r in lst):
                    lst.append(rect_i)
                    creados += 1

        # último punto (frame_to)
        f_last, r_last = hist[-1]
        rect_last = clamp_rect(*map(int, r_last[:4]))
        lst = self.blur_manual_por_frame.setdefault(f_last, [])
        if not any(calcular_iou(rect_last, r) >= 0.6 for r in lst):
            lst.append(rect_last)
            creados += 1

        print(f"[TRACKING] ✅ Confirmado RECORTADO hasta frame {frame_to}: {creados} rects manuales.")

        # 5) Dejar el blur listo para continuar desde aquí (A=B en frame actual)
        blur["rect_A"] = rect_last
        blur["frame_inicio_A"] = frame_to
        blur["confirmado"] = False

        # Apagar el tracking para permitir nuevo tramo (puedes reactivar con 🔍 cuando quieras)
        for k in ("tracker", "tracker_hist", "ema_wh"):
            blur.pop(k, None)
        blur["tracking_activado"] = False

        self.show_frame(forzado=True)
        self.image_label.update()

    def actualizar_trackers(self, frame_idx, frame_actual):
        for blur in self.blurs_en_construccion:
            if blur.get("tracking_activado") and "tracker" in blur:
                success, bbox = blur["tracker"].update(frame_actual)
                if success:
                    blur["rect_B"] = tuple(map(int, bbox))
                    blur["frame_ultimo_B"] = frame_idx

    def actualizar_trackers_blurs_en_construccion(self, frame_idx, frame_actual):
        """
        Actualiza el rect_B de todos los blur fantasmas con seguimiento activado.
        Guarda la trayectoria en blur['tracker_hist'].
        """

        # Guardas simple: si no hay frame, no intentes trackear
        if frame_actual is None or not hasattr(self, "blurs_en_construccion"):
            return

        H, W = frame_actual.shape[:2]

        for blur in list(self.blurs_en_construccion):
            if not blur.get("tracking_activado"):
                continue

            # Inicialización del tracker si hace falta
            if "tracker" not in blur:
                try:
                    tracker = cv2.TrackerCSRT_create()
                    x, y, w, h = map(int, blur["rect_B"])

                    # Validación y clamp a bordes del frame
                    if w <= 0 or h <= 0:
                        print(f"[TRACKING] ❌ Bbox inválido (w/h ≤ 0): {blur['rect_B']}")
                        continue
                    x = max(0, min(x, W - 1))
                    y = max(0, min(y, H - 1))
                    w = max(1, min(w, W - x))
                    h = max(1, min(h, H - y))

                    print(f"[TRACKING] Init tracker rect_B={(x,y,w,h)} | frame shape={frame_actual.shape}")
                    tracker.init(frame_actual, (x, y, w, h))
                    blur["tracker"] = tracker

                    # Sembrar historial si no existía
                    hist = blur.setdefault("tracker_hist", [])
                    if not hist or hist[-1][0] != frame_idx:
                        hist.append((frame_idx, (x, y, w, h)))

                    print(f"[TRACKING] ✅ Tracker inicializado (ID={blur.get('ID')}) en frame {frame_idx}")

                except Exception as e:
                    print(f"[TRACKING] ❌ Error al inicializar tracker: {e}")
                    continue

            # Actualización
            tracker = blur["tracker"]
            success, bbox = tracker.update(frame_actual)  # (x,y,w,h) floats
            if success:
                x, y, w, h = map(int, bbox)

                # Clamp a bordes (evita cortes y errores de dibujo)
                x = max(0, min(x, W - 1))
                y = max(0, min(y, H - 1))
                w = max(1, min(int(w), W - x))
                h = max(1, min(int(h), H - y))

                # (Opcional) Suavizado leve de w/h para quitar jitter
                if "ema_wh" in blur:
                    pw, ph = blur["ema_wh"]
                    alpha = 0.4
                    w = int(pw * (1 - alpha) + w * alpha)
                    h = int(ph * (1 - alpha) + h * alpha)
                blur["ema_wh"] = (w, h)

                blur["rect_B"] = (x, y, w, h)
                blur["frame_ultimo_B"] = frame_idx

                # Guardar trayectoria por frame (evita duplicar por mismo frame)
                hist = blur.setdefault("tracker_hist", [])
                if not hist or hist[-1][0] != frame_idx:
                    # Evita también duplicar el mismo rect exacto
                    if not hist or hist[-1][1] != (x, y, w, h):
                        hist.append((frame_idx, (x, y, w, h)))

                # print(f"[TRACKING] ✔ Update (ID={blur.get('ID')}) → rect_B={blur['rect_B']}")
            else:
                # No matamos el tracker al primer fallo; lo dejamos por si se recupera siguiente frame
                # Si quieres auto-apagar tras N fallos, añade un contador en el blur.
                # print(f"[TRACKING] ⚠ Fallo update en frame {frame_idx} (ID={blur.get('ID')})")
                pass

    def actualizar_trackers_blurs_en_construccion_backup(self, frame_idx, frame_actual):
            """
            Actualiza el rect_B de todos los blur fantasmas con seguimiento activado.
            Guarda la trayectoria en blur['tracker_hist'].
            """
            # Guardas simple: si no hay frame, no intentes trackear
            if frame_actual is None or not hasattr(self, "blurs_en_construccion"):
                return

            H, W = frame_actual.shape[:2]

            for blur in list(self.blurs_en_construccion):
                if not blur.get("tracking_activado"):
                    continue

                # Inicialización del tracker si hace falta
                try:
                    # Compatibilidad entre versiones de OpenCV
                    if hasattr(cv2, "TrackerCSRT_create"):
                        tracker = cv2.TrackerCSRT_create()
                    else:
                        tracker = cv2.legacy.TrackerCSRT_create()

                    x, y, w, h = map(int, blur["rect_B"])
                    if w <= 0 or h <= 0:
                        print(f"[TRACKING] ❌ Bbox inválido (w/h ≤ 0): {blur['rect_B']}")
                        continue

                    x = max(0, min(x, W - 1))
                    y = max(0, min(y, H - 1))
                    w = max(1, min(w, W - x))
                    h = max(1, min(h, H - y))

                    print(f"[TRACKING] Init tracker rect_B={(x,y,w,h)} | frame shape={frame_actual.shape}")
                    tracker.init(frame_actual, (x, y, w, h))
                    blur["tracker"] = tracker

                    hist = blur.setdefault("tracker_hist", [])
                    if not hist or hist[-1][0] != frame_idx:
                        hist.append((frame_idx, (x, y, w, h)))

                    print(f"[TRACKING] ✅ Tracker inicializado (ID={blur.get('ID')}) en frame {frame_idx}")

                except Exception as e:
                    print(f"[TRACKING] ❌ Error al inicializar tracker: {e}")
                    continue



                # Actualización
                tracker = blur["tracker"]
                success, bbox = tracker.update(frame_actual)  # (x,y,w,h) floats
                if success:
                    x, y, w, h = map(int, bbox)

                    # Clamp a bordes (evita cortes y errores de dibujo)
                    x = max(0, min(x, W - 1))
                    y = max(0, min(y, H - 1))
                    w = max(1, min(int(w), W - x))
                    h = max(1, min(int(h), H - y))

                    # (Opcional) Suavizado leve de w/h para quitar jitter
                    if "ema_wh" in blur:
                        pw, ph = blur["ema_wh"]
                        alpha = 0.4
                        w = int(pw * (1 - alpha) + w * alpha)
                        h = int(ph * (1 - alpha) + h * alpha)
                    blur["ema_wh"] = (w, h)

                    blur["rect_B"] = (x, y, w, h)
                    blur["frame_ultimo_B"] = frame_idx

                    # Guardar trayectoria por frame (evita duplicar por mismo frame)
                    hist = blur.setdefault("tracker_hist", [])
                    if not hist or hist[-1][0] != frame_idx:
                        # Evita también duplicar el mismo rect exacto
                        if not hist or hist[-1][1] != (x, y, w, h):
                            hist.append((frame_idx, (x, y, w, h)))

                    # print(f"[TRACKING] ✔ Update (ID={blur.get('ID')}) → rect_B={blur['rect_B']}")
                else:
                    # No matamos el tracker al primer fallo; lo dejamos por si se recupera siguiente frame
                    # Si quieres auto-apagar tras N fallos, añade un contador en el blur.
                    # print(f"[TRACKING] ⚠ Fallo update en frame {frame_idx} (ID={blur.get('ID')})")
                    pass


    def confirmar_tracker_en_blur(self, blur):
        """
        Convierte la trayectoria del tracker en blurs manuales con interpolación
        por frame (sin huecos). Limpia el blur fantasma al final.
        """
        hist = blur.get("tracker_hist", [])
        if not hist:
            print(f"[TRACKING] No hay historial para confirmar (ID={blur.get('ID')})")
            return

        # Orden por frame
        hist = sorted(hist, key=lambda t: t[0])

        # Extender con A (inicio) y B (final) si es necesario
        fA = blur.get("frame_inicio_A", hist[0][0])
        rA = blur.get("rect_A", hist[0][1])
        if fA < hist[0][0]:
            hist = [(int(fA), tuple(map(int, rA[:4])))] + hist

        fB = blur.get("frame_ultimo_B", hist[-1][0])
        rB = blur.get("rect_B", hist[-1][1])
        if fB > hist[-1][0]:
            hist = hist + [(int(fB), tuple(map(int, rB[:4])))]

        # Dimensiones para clamp
        H = W = None
        if hasattr(self, "current_frame") and self.current_frame is not None:
            H, W = self.current_frame.shape[:2]

        def clamp_rect(x, y, w, h):
            if W is None or H is None:
                return int(x), int(y), int(w), int(h)
            x = max(0, min(int(x), W - 1))
            y = max(0, min(int(y), H - 1))
            w = max(1, min(int(w), W - x))
            h = max(1, min(int(h), H - y))
            return x, y, w, h

        creados = 0

        # Interpolar tramo a tramo
        for i in range(len(hist) - 1):
            f1, r1 = hist[i]
            f2, r2 = hist[i + 1]
            x1, y1, w1, h1 = map(int, r1[:4])
            x2, y2, w2, h2 = map(int, r2[:4])

            dt = f2 - f1
            if dt <= 0:
                continue

            # Frames intermedios (incluye f1, excluye f2 aquí)
            for f in range(f1, f2):
                t = (f - f1) / dt
                xi = x1 + (x2 - x1) * t
                yi = y1 + (y2 - y1) * t
                wi = w1 + (w2 - w1) * t
                hi = h1 + (h2 - h1) * t
                rect_i = clamp_rect(xi, yi, wi, hi)

                lst = self.blur_manual_por_frame.setdefault(f, [])
                # Evita duplicados por IoU (bastante permisivo)
                if not any(rects_similares_iou(rect_i, r, umbral_iou=0.6) for r in lst):
                    lst.append(rect_i)
                    creados += 1

        # Asegura el último frame (f2 y r2 del último tramo)
        f_last, r_last = hist[-1]
        rect_last = clamp_rect(*map(int, r_last[:4]))
        lst = self.blur_manual_por_frame.setdefault(f_last, [])
        if not any(rects_similares_iou(rect_last, r, umbral_iou=0.6) for r in lst):
            lst.append(rect_last)
            creados += 1

        # Limpieza del blur fantasma
        if "tracker" in blur:
            del blur["tracker"]
        blur.pop("tracker_hist", None)
        blur.pop("ema_wh", None)
        blur["tracking_activado"] = False
        self.blurs_en_construccion = [b for b in self.blurs_en_construccion if b.get("ID") != blur.get("ID")]

        print(f"[TRACKING] ✅ Confirmado con interpolación: {creados} rects manuales creados (ID={blur.get('ID')})")
        self.show_frame(forzado=True)

    def _aplicar_tracking_en_frame_estatico(self, frame_idx):
        """
        Al navegar en pausa, mueve rect_B al último bbox registrado en tracker_hist
        cuyo frame <= frame_idx. No toca el objeto tracker (no hay 'update' hacia atrás).
        """
        if not getattr(self, "blurs_en_construccion", None):
            return

        

        for blur in self.blurs_en_construccion:
            if frame_idx < int(blur.get("frame_inicio_A", 0)):
                continue  # ⛔ no fijar rect_B antes del inicio
            if not blur.get("tracking_activado"):
                continue
            hist = blur.get("tracker_hist") or []
            if not hist:
                continue

            # Busca el último keyframe <= frame_idx
            candidato = None
            for f, r in hist:
                if f <= frame_idx:
                    candidato = (f, r)
                else:
                    break  # hist debe estar ordenado ascendente

            if candidato:
                _, rect = candidato
                x, y, w, h = map(int, rect[:4])
                # Clamp por seguridad
                H, W = self.current_frame.shape[:2]
                x = max(0, min(x, W - 1))
                y = max(0, min(y, H - 1))
                w = max(1, min(w, W - x))
                h = max(1, min(h, H - y))
                blur["rect_B"] = (x, y, w, h)
                blur["frame_ultimo_B"] = frame_idx


    def eliminar_rects_frame_actual(self, punto_clic: tuple, frame_idx: int, tolerancia=50) -> list:
        """
        Elimina rects IA similares al punto clic en el frame actual.
        Los remueve físicamente de blur_ia_por_frame y los copia a blur_eliminado_ia.
        """
        x_click, y_click = punto_clic
        rects_actuales = self.blur_ia_por_frame.get(frame_idx, [])
        nuevos_rects = []
        eliminados = []

        for rect in rects_actuales:
            rx, ry, rw, rh = rect[:4]
            if rx <= x_click <= rx + rw and ry <= y_click <= ry + rh:
                self.blur_eliminado_ia.setdefault(frame_idx, []).append(rect)
                eliminados.append(rect)
                log_ia_click(f"[RECT_ACTUAL] Eliminado en frame {frame_idx}: {rect}")
            else:
                nuevos_rects.append(rect)

        self.blur_ia_por_frame[frame_idx] = nuevos_rects
        return eliminados

    #IMPORTANTE: por defecto max_frames 3 y duracion_max int(self.fps * 5 tolerancia 50)
    def eliminar_rects_frame_sgtes(self, rect_base: tuple, frame_inicio: int, max_frames=200, tolerancia=100, tipo="ia"):
        """
        Propaga la eliminación desde el frame siguiente hacia adelante.
        - tipo="ia": elimina de blur_ia_por_frame y además registra en blur_eliminado_ia (como hoy).
        - tipo="manual": elimina de blur_manual_por_frame (no registra en blur_eliminado_ia).
        Los parámetros por defecto replican tu comportamiento actual para IA.
        """
        duracion_max = int(self.fps * 60)  # máx. 5 s de detecciones válidas
        contador_exitos = 0
        frame_actual = frame_inicio + 1
        fallos = 0

        while frame_actual < self.total_frames and fallos < max_frames:
            if tipo == "ia":
                lista = self.blur_ia_por_frame.get(frame_actual, [])
            else:
                lista = self.blur_manual_por_frame.get(frame_actual, [])

            nuevos_rects = []
            encontrado = False

            for rect in lista:
                if not encontrado and rects_similares(rect_base, rect, tolerancia=tolerancia):
                    # Eliminar
                    if tipo == "ia":
                        # registrar en eliminados IA + remover físicamente
                        self.blur_eliminado_ia.setdefault(frame_actual, []).append(rect)
                        log_ia_click(f"[PROPAGACIÓN IA] Eliminado en frame {frame_actual}: {rect}")
                    else:
                        log_ia_click(f"[PROPAGACIÓN MANUAL] Eliminado en frame {frame_actual}: {rect}")

                    rect_base = rect  # seguir propagando desde el último hallado
                    encontrado = True
                else:
                    nuevos_rects.append(rect)

            # Escribir de vuelta la lista correspondiente
            if tipo == "ia":
                self.blur_ia_por_frame[frame_actual] = nuevos_rects
            else:
                self.blur_manual_por_frame[frame_actual] = nuevos_rects

            if encontrado:
                fallos = 0
                contador_exitos += 1
                if contador_exitos >= duracion_max:
                    break
            else:
                fallos += 1

            frame_actual += 1


    def eliminar_rects_frame_actual_mixto(self, punto_clic: tuple, frame_idx: int):
        """
        Intenta eliminar primero rects MANUALES que contengan el clic.
        Si elimina alguno manual, NO elimina IA en este clic (prioridad manual).
        Si no hay manuales en el punto, elimina IA usando la lógica actual.
        Retorna: dict {"manual": [rects_eliminados_manual], "ia": [rects_eliminados_ia]}
        """
        x_click, y_click = punto_clic
        eliminados = {"manual": [], "ia": []}

        # 1) PRIORIDAD: MANUAL
        manuales = self.blur_manual_por_frame.get(frame_idx, [])
        if manuales:
            nueva_lista = []
            for rect in manuales:
                rx, ry, rw, rh = rect[:4]
                if rx <= x_click <= rx + rw and ry <= y_click <= ry + rh:
                    eliminados["manual"].append(rect)
                    log_ia_click(f"[MANUAL] Eliminado en frame {frame_idx}: {rect}")
                else:
                    nueva_lista.append(rect)
            # Actualiza la lista manual del frame actual
            if len(nueva_lista) != len(manuales):
                self.blur_manual_por_frame[frame_idx] = nueva_lista

        # 2) Si NO se eliminó ningún manual, intenta IA con la lógica existente
        if not eliminados["manual"]:
            eliminados_ia_actual = self.eliminar_rects_frame_actual(punto_clic, frame_idx)
            if eliminados_ia_actual:
                eliminados["ia"].extend(eliminados_ia_actual)

        return eliminados


    #ELIMINACION IA
    def procesar_prop_ia(self, rect, frame_idx, rects_ia, set_ia, set_eliminados):
        """
        Intenta eliminar un blur IA en el frame actual si es similar al rect original.
        Registra la eliminación directamente en blur_eliminado_ia si aplica.
        """
        for r in rects_ia:
            # Verifica que no se haya eliminado ya (por clave)
            if rect_to_key(r) not in set_eliminados:
                if rects_similares_iou(rect, r, umbral_iou=0.1):
                    self.blur_eliminado_ia.setdefault(frame_idx, []).append(r)  # ← ✅ r real del frame actual
                    self.registrar_eliminacion(frame_idx, r, fuente="PROPAGACIÓN_IA")
                    log_ia_click(f"[PROPAGACIÓN_IA] Eliminado {r} en frame {frame_idx} por similitud con {rect}")
                    self.coords_para_eliminar = [c for c in self.coords_para_eliminar if not rects_similares_iou(c[0], rect, umbral_iou=0.1)]
                    log_ia_click(f"[CLEANUP] coords_para_eliminar limpiado en procesar_prop_ia para {rect}")
                    return True
        return False

    def aplicar_eliminacion_inmediata_ia(self, rect_base, frame_inicio, umbral_iou=0.1, max_fallos=60, duracion_segundos=5):
        fps = self.fps if self.fps > 0 else 30
        frame_actual = frame_inicio + 1
        fallos = 0
        total_frames = self.total_frames
        limite = min(frame_inicio + int(duracion_segundos * fps), total_frames)

        while frame_actual <= limite and fallos < max_fallos:
            rects_actuales = self.blur_ia_por_frame.get(frame_actual, [])
            encontrado = False

            for r in rects_actuales:
                if rects_similares_iou(rect_base, r, umbral_iou=umbral_iou):
                    ya_registrado = any(rects_similares_iou(r, r_exist, umbral_iou=umbral_iou)
                                        for r_exist in self.blur_eliminado_ia.get(frame_actual, []))
                    if not ya_registrado:
                        # Buscar rect exacto dentro de blur_ia_por_frame[frame_actual]
                        rects_orig = self.blur_ia_por_frame.get(frame_actual, [])
                        rect_elegido = None
                        for orig in rects_orig:
                            if rects_similares_iou(rect_base, orig, umbral_iou=umbral_iou):
                                rect_elegido = orig
                                break

                        if rect_elegido:
                            self.blur_eliminado_ia.setdefault(frame_actual, []).append(rect_elegido)
                            log_ia_click(f"[PERMANENTE] Eliminado inmediato en frame {frame_actual} por similitud con {rect_elegido}")
                            # ✅ Remover físicamente de blur_ia_por_frame
                            if frame_actual in self.blur_ia_por_frame:
                                self.blur_ia_por_frame[frame_actual] = [
                                    r for r in self.blur_ia_por_frame[frame_actual]
                                    if not rects_similares_iou(r, rect_elegido, umbral_iou)
                                ]
                                log_ia_click(f"[REMOVIDO] Blur eliminado físicamente de blur_ia_por_frame[{frame_actual}]: {rect_elegido}")

                        else:
                            log_ia_click(f"[❌ ERROR] No se encontró rect exacto para registrar en frame {frame_actual}")


            if encontrado:
                fallos = 0
            else:
                fallos += 1

            frame_actual += 1
        # Limpieza de registros temporales tras completar eliminación progresiva
        self.coords_para_eliminar = [c for c in self.coords_para_eliminar if not rects_similares_iou(c[0], rect_base, umbral_iou)]
        log_ia_click(f"[CLEANUP] coords_para_eliminar limpiado para {rect_base}")

        # === LOG DE VERIFICACIÓN ===
        for frame in range(frame_inicio, frame_actual):
            ia = self.blur_ia_por_frame.get(frame, [])
            elim = self.blur_eliminado_ia.get(frame, [])
            for r in elim:
                if r not in ia:
                    log_ia_click(f"[⚠️ DISCREPANCIA] Frame {frame}: rect eliminado {r} NO está en blur_ia_por_frame")
                else:
                    log_ia_click(f"[✔️ COINCIDE] Frame {frame}: rect eliminado {r} está en blur_ia_por_frame")

    def procesar_prop_manual(self, rect, frame_idx, rects_manual, set_manual):
        key = rect_to_key(rect)
        if key in set_manual:
            nueva_lista = [r for r in rects_manual if not rects_similares(r, rect, tolerancia=10)]
            self.blur_manual_por_frame[frame_idx] = nueva_lista
            log_ia_click(f"[PROPAGACIÓN_MANUAL] Eliminado {rect} en frame {frame_idx}")
            self.coords_para_eliminar = [c for c in self.coords_para_eliminar if not rects_similares(c[0], rect, tolerancia=10)]
            log_ia_click(f"[CLEANUP] coords_para_eliminar limpiado en procesar_prop_manual para {rect}")
            return True
        return False
    
    #next
    def next_frame(self):
        t_inicio = datetime.now()
        log(f"[CALL] next_frame llamado desde frame {self.frame_idx}")

        if self._ultimo_frame_timestamp:
            delta_ms = (t_inicio - self._ultimo_frame_timestamp).total_seconds() * 1000
        else:
            delta_ms = 0
        self._ultimo_frame_timestamp = t_inicio

        if not self.cap or self.procesando:
            return

        log(f"[TRACK] frame_idx incrementado a {self.frame_idx}")

        if (
            self.ia_procesando is False
            and self.ia_fin_frame > 0
            and not self.slider_tocado
            and self.auto_procesamiento_checkbox.isChecked()
        ):
            try:
                segundos_umbral = int(self.ia_umbral_input.text())
            except ValueError:
                segundos_umbral = 5

            umbral = self.ia_fin_frame - int(segundos_umbral * self.fps)
            if self.frame_idx >= umbral and self.ia_fin_frame < self.total_frames:
                try:
                    duracion = int(self.ia_duracion_input.text())
                except ValueError:
                    duracion = 20

                nuevo_inicio = self.ia_fin_frame
                nuevo_fin = min(nuevo_inicio + duracion * self.fps, self.total_frames)

                log(f"[AUTO] Extendiendo procesamiento IA hasta frame {nuevo_fin} desde {nuevo_inicio}")
                self.pause_video()
                self.ia_procesando = True
                self.ia_inicio_frame = nuevo_inicio
                self.ia_fin_frame = nuevo_fin
                self.slider.setEnabled(False)
                self.play_button.setEnabled(False)
                self.progress_bar.setValue(0)
                self.progress_bar.setVisible(True)
                self.ia_estado_label.setText("IA: Procesando (auto)")
                self.procesar_ia_en_rango(nuevo_inicio, nuevo_fin)

        self.origen_show_frame = "next_frame_reproduccion" if self.timer.isActive() else "next_frame_manual"
        log(f"[DEBUG] next_frame → show_frame(forzado=True) | origen={self.origen_show_frame}")

        if self.cap:
            if not self.timer.isActive():
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
                self.cap.grab()
                ret, frame = self.cap.retrieve()
                if ret and frame is not None:
                    self.current_frame = frame.copy()
                else:
                    log(f"[ERROR] No se pudo recuperar el frame {self.frame_idx} en next_frame()")
            else:
                salto = int(self.velocidad_reproduccion)
                if salto == 1:
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.current_frame = frame.copy()
                        self.frame_idx += 1
                    else:
                        log(f"[ERROR] No se pudo leer el siguiente frame en reproducción (frame_idx={self.frame_idx})")
                elif salto > 1:
                    ret, frame = None, None
                    for i in range(salto):
                        ret, frame = self.cap.read()
                        if not ret or frame is None:
                            log(f"[ERROR] Fallo al leer frame saltado desde frame_idx={self.frame_idx} (i={i})")
                            break
                        self.current_frame = frame.copy()
                        self.frame_idx += 1

        self.show_frame(forzado=False)

        # 🚀 Seguimiento automático de blurs con tracking_activado
        self.actualizar_trackers_blurs_en_construccion(self.frame_idx, self.current_frame)
        


        t_fin = datetime.now()
        delta = (t_fin - t_inicio).total_seconds()
        log(f"[FRAME TIME REAL] {delta:.3f} s")

    def formato_tiempo(self, segundos):
        horas = int(segundos) // 3600
        minutos = (int(segundos) % 3600) // 60
        segs = int(segundos) % 60
        return f"{horas:02d}:{minutos:02d}:{segs:02d}"
        
    def next_frame_manual(self):
        if self.cap:
            self.pause_video()  # ← Agregado
            self.frame_idx += 1
            self.show_frame(forzado=True)

    def prev_frame(self):
        if not self.cap or self.frame_idx <= 1:
            return
            
        self.pause_video()
        self.frame_idx -= 1
        log(f"[DEBUG] Llamando show_frame(forzado=True) desde prev_frame en frame {self.frame_idx}")
        self.show_frame(forzado=True)
        
    def minuto_listo(self, minuto, datos):
        #self.blur_ia_por_frame.update(datos)
        #self.minutos_procesados.add(minuto)
        #self.procesando = False
        #self.progress_bar.setVisible(False)
        #log(f"[GUI] Minuto {minuto} cargado")
        #self.show_frame()
        #self.show_frame(forzado=True)
        #log(f"[DEBUG] self.show_frame() desde el metodo def minuto_listo(self, minuto, datos):")
        #self.play_video()
        pass

    def play_video(self):
        log(f"[PLAY] Iniciando reproducción desde frame {self.frame_idx} | origen_show_frame = {self.origen_show_frame}")

        if self.cap:
            # Solo cargar el frame si no hay uno válido actualmente
            if not hasattr(self, "current_frame") or self.current_frame is None:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
                self.cap.grab()
                ret, frame = self.cap.retrieve()
                if ret:
                    self.current_frame = frame.copy()
                    self.show_frame(forzado=True)
                else:
                    log(f"[PLAY] No se pudo recuperar el frame {self.frame_idx} al iniciar reproducción")
            else:
                # Redibuja sin lectura forzada si ya hay frame cargado
                self.show_frame(forzado=False)

            self._salto_doble_realizado = False  # ← al salir de pausa, habilitamos 1 salto de nuevo

            self.timer.stop()  # Detener antes de reiniciar
            intervalo = int(1000 / (self.fps * self.velocidad_reproduccion))
            self.timer.start(intervalo)

            # Actualiza botones de velocidad
            self.prev_frame_button.setText("🔽 Vel. -")
            self.next_frame_button.setText("🔼 Vel. +")

            # ⛔ Desactivar eventos del mouse mientras se reproduce
            self.image_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            print("[PLAY] Eventos de mouse desactivados durante la reproducción")
                                
    def pause_video(self):
        log_salto_frames(f"[PAUSE] Botón pause presionado. frame_idx={self.frame_idx}, timer activo: {self.timer.isActive()}")
        self.timer.stop()
        self.frame_nav_timer.stop()

        self.image_label.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        print("[PAUSE] Eventos de mouse reactivados")

        # Solo forzar lectura si no hay current_frame válido
        if not hasattr(self, "current_frame") or self.current_frame is None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
            self.cap.grab()
            ret, frame = self.cap.retrieve()
            if ret:
                self.current_frame = frame.copy()
            self.show_frame(forzado=True)
        else:
            self.show_frame(forzado=False)

        # Restaurar botones
        self.prev_frame_button.setText("⏮ Frame anterior")
        self.next_frame_button.setText("⏭ Siguiente frame")

        # Ocultar el flash si está activo
        if self.flash_label.isVisible():
            self.flash_label.setVisible(False)
            self.flash_label.setGraphicsEffect(None)
            self.flash_animacion = None
            self.flash_opacity = None

        # Restablecer velocidad
        self.velocidad_reproduccion = 1.0

        # Actualizar visualmente el label del frame
        self.frame_label.setText(f"Frame: {self.frame_idx}   |   Velocidad: x{self.velocidad_reproduccion:.2f}")

    def aplicar_blur_circular(frame, rect):
        if len(rect) >= 4:
            x, y, w, h = rect[:4]
        #else:
        #    continue
        roi = frame[y:y+h, x:x+w]
        if roi.size > 0:
            if k % 2 == 0:
                k += 1
            k = max(3, k)
            blurred = cv2.GaussianBlur(roi, (k, k), 0)
            #blurred = cv2.GaussianBlur(roi, (51, 51), 0)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (w // 2, h // 2), min(w, h) // 2, 255, -1)
            mask_3ch = cv2.merge([mask]*3)
            roi_result = np.where(mask_3ch == 255, blurred, roi)
            frame[y:y+h, x:x+w] = roi_result

    #EXPORTAR
    def exportar_avance(self):
        if not self.cap:
            log("[EXPORT] No hay video cargado.")
            return

        log("[EXPORT] Iniciando exportación de avance...")

        msg = QMessageBox(self)
        msg.setWindowTitle("Exportar video")
        msg.setText("Selecciona el modo de exportación:")
        normal_btn = msg.addButton("Normal", QMessageBox.AcceptRole)
        comprimido_btn = msg.addButton("Comprimido", QMessageBox.AcceptRole)
        cancelar_btn = msg.addButton("Cancelar", QMessageBox.RejectRole)  # 🔹 nuevo
        msg.setDefaultButton(cancelar_btn)

        msg.exec()

        if msg.clickedButton() == normal_btn:
            self.modo_exportacion = "normal"
        elif msg.clickedButton() == comprimido_btn:
            self.modo_exportacion = "comprimido"
        else:
            log("[EXPORT] Exportación cancelada por el usuario.")
            return

        inicio = self.qtime_a_frame(self.export_inicio_time.time())
        fin = self.qtime_a_frame(self.export_fin_time.time())

        # 🔹 Diálogo para elegir carpeta/archivo de salida
        default_name = f"avance_blur_{datetime.now().strftime('%H%M%S')}.mp4"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar video exportado",
            default_name,
            "Archivos de video (*.mp4)"
        )

        if not file_path:
            log("[EXPORT] Exportación cancelada por el usuario (sin ruta).")
            return

        # Iniciar progreso
        self.export_progress_bar.setVisible(True)
        self.export_progress_bar.setMaximum(fin - inicio)
        self.export_progress_bar.setValue(0)

        # Pasar la ruta elegida al hilo de exportación
        self.export_thread = ExportThread(self, inicio, fin, file_path, self.bitrate_original_kbps)
        self.export_thread.progreso.connect(self.export_progress_bar.setValue)
        self.export_thread.terminado.connect(self.exportar_finalizado)
        self.export_thread.start()


    def exportar_finalizado(self, nombre_archivo, total_frames):
        self.export_progress_bar.setVisible(False)
        log(f"[EXPORT] Total de frames exportados: {total_frames}")
        log(f"[EXPORT] Archivo generado: {nombre_archivo}")
      
    def guardar_avance(self):  # GUARDA LOS REGISTROS DE EDICION
        if not self.video_path:
            QMessageBox.warning(self, "Guardar avance", "Primero debes cargar un video.")
            return

        log("[GUARDAR] Exportando avance de edición a JSON...")

        # Mostrar barra de progreso
        self.save_progress_bar.setVisible(True)
        QApplication.processEvents()

        # Crear carpeta si no existe
        carpeta = "REGISTROS_DE_EDICION"
        os.makedirs(carpeta, exist_ok=True)

        # Obtener el rango de frames usados en los registros
        todos_los_frames = set(self.blur_ia_por_frame.keys()) | set(self.blur_manual_por_frame.keys())
        if todos_los_frames:
            inicio = min(todos_los_frames)
            fin = max(todos_los_frames)
            rango = f"_{inicio}-{fin}"
        else:
            rango = "_0-0"

        # Preparar nombre
        nombre_video = os.path.splitext(os.path.basename(self.video_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Formato de hora de video
        def a_hora_formateada(frame_idx, fps):
            total_seg = int(frame_idx / fps)
            h = total_seg // 3600
            m = (total_seg % 3600) // 60
            s = total_seg % 60
            return f"{h}h-{m}m-{s}s"

        if todos_los_frames:
            t_inicio = a_hora_formateada(inicio, self.fps)
            t_fin = a_hora_formateada(fin, self.fps)
            hora_video = f"_{t_inicio}__{t_fin}_"
        else:
            hora_video = "_0h-0m-0s__0h-0m-0s_"

        nombre_archivo = f"avance_{nombre_video}_{timestamp}{hora_video}{rango}.json"
        ruta_completa = os.path.join(carpeta, nombre_archivo)


        # --- Normalizar blur_ia_por_frame: asegurar 5 elementos por rectángulo ---
        errores_detectados = 0
        for frame, rects in self.blur_ia_por_frame.items():
            nuevos_rects = []
            for r in rects:
                if not isinstance(r, (list, tuple)) or not all(isinstance(v, (int, float)) for v in r):
                    log(f"[GUARDAR] Rectángulo inválido en frame {frame}: {r} (tipo incorrecto)")
                    errores_detectados += 1
                    continue
                if len(r) == 4:
                    log(f"[GUARDAR] Rectángulo incompleto (4 elementos) en frame {frame}: {r} → se completará con conf=0.5")
                    nuevos_rects.append((*r, 0.5))
                elif len(r) == 5:
                    nuevos_rects.append(r)
                else:
                    log(f"[GUARDAR] Rectángulo con longitud inesperada ({len(r)}) en frame {frame}: {r}")
                    errores_detectados += 1
            self.blur_ia_por_frame[frame] = nuevos_rects

        if errores_detectados > 0:
            log(f"[GUARDAR] Total de rectángulos descartados o corregidos: {errores_detectados}")

        # Validación opcional para blur_manual_por_frame
        for frame, rects in self.blur_manual_por_frame.items():
            for r in rects:
                if not isinstance(r, (list, tuple)) or len(r) != 4:
                    log(f"[GUARDAR] Rectángulo manual inválido en frame {frame}: {r}")

        # Preparar datos (solo estructuras válidas)
        datos = {
            "blur_ia_por_frame": self.blur_ia_por_frame,
            "blur_manual_por_frame": self.blur_manual_por_frame,
            "blur_eliminado_ia": self.blur_eliminado_ia
        }

        try:
            with open(ruta_completa, "w") as f:
                json.dump(datos, f, indent=2)
            log(f"[GUARDAR] Avance guardado en: {ruta_completa}")
            QMessageBox.information(self, "Avance guardado", f"Se guardó correctamente:\n{nombre_archivo}")

        except Exception as e:
            log(f"[GUARDAR] Error al guardar avance: {e}")
            QMessageBox.critical(self, "Error", f"No se pudo guardar el avance:\n{e}")

        self.save_progress_bar.setVisible(False)

        log(f"[GUARDAR] blur_ia: {sum(len(v) for v in self.blur_ia_por_frame.values())}")
        log(f"[GUARDAR] blur_manual: {sum(len(v) for v in self.blur_manual_por_frame.values())}")
        log(f"[GUARDAR] blur_eliminado_ia: {sum(len(v) for v in self.blur_eliminado_ia.values())}")

    def guardar_avance_backup(self):  #04/08/2025
        if not self.video_path:
            QMessageBox.warning(self, "Guardar avance", "Primero debes cargar un video.")
            return

        log("[GUARDAR] Exportando avance de edición a JSON...")

        # Mostrar barra de progreso
        self.save_progress_bar.setVisible(True)
        QApplication.processEvents()

        # Crear carpeta si no existe
        carpeta = "REGISTROS_DE_EDICION"
        os.makedirs(carpeta, exist_ok=True)

        # Obtener el rango de frames usados en los registros
        todos_los_frames = set(self.blur_ia_por_frame.keys()) | set(self.blur_manual_por_frame.keys())
        if todos_los_frames:
            inicio = min(todos_los_frames)
            fin = max(todos_los_frames)
            rango = f"_{inicio}-{fin}"
        else:
            rango = "_0-0"

        # Preparar nombre
        nombre_video = os.path.splitext(os.path.basename(self.video_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"avance_{nombre_video}_{timestamp}{rango}.json"
        ruta_completa = os.path.join(carpeta, nombre_archivo)

        # --- Normalizar blur_ia_por_frame: asegurar 5 elementos por rectángulo ---
        errores_detectados = 0
        for frame, rects in self.blur_ia_por_frame.items():
            nuevos_rects = []
            for r in rects:
                if not isinstance(r, (list, tuple)) or not all(isinstance(v, (int, float)) for v in r):
                    log(f"[GUARDAR] Rectángulo inválido en frame {frame}: {r} (tipo incorrecto)")
                    errores_detectados += 1
                    continue
                if len(r) == 4:
                    log(f"[GUARDAR] Rectángulo incompleto (4 elementos) en frame {frame}: {r} → se completará con conf=0.5")
                    nuevos_rects.append((*r, 0.5))
                elif len(r) == 5:
                    nuevos_rects.append(r)
                else:
                    log(f"[GUARDAR] Rectángulo con longitud inesperada ({len(r)}) en frame {frame}: {r}")
                    errores_detectados += 1
            self.blur_ia_por_frame[frame] = nuevos_rects

        if errores_detectados > 0:
            log(f"[GUARDAR] Total de rectángulos descartados o corregidos: {errores_detectados}")

        # Validación opcional para blur_manual_por_frame
        for frame, rects in self.blur_manual_por_frame.items():
            for r in rects:
                if not isinstance(r, (list, tuple)) or len(r) != 4:
                    log(f"[GUARDAR] Rectángulo manual inválido en frame {frame}: {r}")

        # Preparar datos (solo estructuras válidas)
        datos = {
            "blur_ia_por_frame": self.blur_ia_por_frame,
            "blur_manual_por_frame": self.blur_manual_por_frame,
            "blur_eliminado_ia": self.blur_eliminado_ia
        }

        try:
            with open(ruta_completa, "w") as f:
                json.dump(datos, f, indent=2)
            log(f"[GUARDAR] Avance guardado en: {ruta_completa}")
            QMessageBox.information(self, "Avance guardado", f"Se guardó correctamente:\n{nombre_archivo}")

        except Exception as e:
            log(f"[GUARDAR] Error al guardar avance: {e}")
            QMessageBox.critical(self, "Error", f"No se pudo guardar el avance:\n{e}")

        self.save_progress_bar.setVisible(False)

        log(f"[GUARDAR] blur_ia: {sum(len(v) for v in self.blur_ia_por_frame.values())}")
        log(f"[GUARDAR] blur_manual: {sum(len(v) for v in self.blur_manual_por_frame.values())}")
        log(f"[GUARDAR] blur_eliminado_ia: {sum(len(v) for v in self.blur_eliminado_ia.values())}")

    def cargar_avance(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Cargar avance de edición", "", "Archivos JSON (*.json)")
        if not ruta:
            return

        try:
            with open(ruta, "r") as f:
                datos = json.load(f)

            log(f"[DEBUG] Tipo de datos JSON cargado: {type(datos)}")

            # Asegurar estructuras
            if not isinstance(self.blur_ia_por_frame, dict):
                self.blur_ia_por_frame = {}
            if not isinstance(self.blur_manual_por_frame, dict):
                self.blur_manual_por_frame = {}
            if not isinstance(self.blur_eliminado_ia, dict):
                self.blur_eliminado_ia = {}
            if not hasattr(self, "coords_para_eliminar") or not isinstance(self.coords_para_eliminar, list):
                self.coords_para_eliminar = []  # ← sigue existiendo pero no se carga desde JSON

            # --- CASO 1: JSON COMPLETO ---
            if isinstance(datos, dict):
                # Fusionar blur_ia_por_frame
                blur_ia = datos.get("blur_ia_por_frame", {})
                if isinstance(blur_ia, dict):
                    for frame, rects in blur_ia.items():
                        frame = int(frame)
                        if frame not in self.blur_ia_por_frame:
                            self.blur_ia_por_frame[frame] = rects
                        else:
                            self.blur_ia_por_frame[frame].extend(
                                [r for r in rects if r not in self.blur_ia_por_frame[frame]]
                            )
                else:
                    log(f"[CARGAR] Advertencia: 'blur_ia_por_frame' no es un dict: {type(blur_ia)}")

                # Normalizar los blur IA
                for frame, rects in self.blur_ia_por_frame.items():
                    nuevos_rects = []
                    for r in rects:
                        if len(r) == 4:
                            nuevos_rects.append((*r, 0.5))
                        else:
                            nuevos_rects.append(r)
                    self.blur_ia_por_frame[frame] = nuevos_rects

                # Validación adicional: longitud inesperada
                for frame, rects in self.blur_ia_por_frame.items():
                    for r in rects:
                        if not self.es_rect_valido(r):
                            log(f"[CARGAR] Rectángulo IA inválido en frame {frame}: {r}")

                # Fusionar blur_eliminado_ia
                blur_eliminado_ia = datos.get("blur_eliminado_ia", {})
                if isinstance(blur_eliminado_ia, dict):
                    for frame_str, rects in blur_eliminado_ia.items():
                        frame = int(frame_str)
                        if frame not in self.blur_eliminado_ia:
                            self.blur_eliminado_ia[frame] = rects
                        else:
                            self.blur_eliminado_ia[frame].extend(
                                [r for r in rects if r not in self.blur_eliminado_ia[frame]]
                            )
                    # Después de fusionar blur_ia_por_frame y blur_eliminado_ia
                    for frame, eliminados in self.blur_eliminado_ia.items():
                        if frame in self.blur_ia_por_frame:
                            originales = self.blur_ia_por_frame[frame]
                            self.blur_ia_por_frame[frame] = [r for r in originales if r not in eliminados]
                            log(f"[CARGAR] Frame {frame}: {len(originales) - len(self.blur_ia_por_frame[frame])} rects IA eliminados al cargar JSON.")
                else:
                    log(f"[CARGAR] Advertencia: 'blur_eliminado_ia' no es un dict: {type(blur_eliminado_ia)}")

                # ⚠️ coords_para_eliminar será ignorado si viene en el JSON
                if "coords_para_eliminar" in datos:
                    log(f"[CARGAR] ⚠️ Advertencia: 'coords_para_eliminar' fue ignorado (estructura temporal)")

                # Fusionar blur_manual_por_frame
                blur_manual = datos.get("blur_manual_por_frame", {})
                if isinstance(blur_manual, dict):
                    for frame_str, rects in blur_manual.items():
                        frame = int(frame_str)
                        if frame not in self.blur_manual_por_frame:
                            self.blur_manual_por_frame[frame] = rects
                        else:
                            self.blur_manual_por_frame[frame].extend(
                                [r for r in rects if r not in self.blur_manual_por_frame[frame]]
                            )
                else:
                    log(f"[CARGAR] Advertencia: 'blur_manual_por_frame' no es un dict: {type(blur_manual)}")

            else:
                raise ValueError("Formato de JSON no reconocido.")

            log(f"[CARGAR] Avance fusionado correctamente desde: {ruta}")
            QMessageBox.information(self, "Avance cargado", f"El avance se fusionó correctamente:\n{os.path.basename(ruta)}")

            # Pausa breve para permitir que se asienten los datos y forzar la lectura del frame correcto
            QTimer.singleShot(500, lambda: self.show_frame(forzado=True))

        except Exception as e:
            log(f"[CARGAR] Error al cargar avance: {e}")
            QMessageBox.critical(self, "Error", f"No se pudo cargar el avance:\n{e}")

    def calcular_escala_y_offset(self):
        """
        Devuelve una tupla con:
        (scale, new_w, new_h, offset_x, offset_y)
        """
        if not hasattr(self, "current_frame") or self.current_frame is None:
            return 1.0, 0, 0, 0, 0  # valor seguro

        frame_h, frame_w = self.current_frame.shape[:2]
        label_w = self.image_label.width()
        label_h = self.image_label.height()
        scale = min(label_w / frame_w, label_h / frame_h)
        new_w = int(frame_w * scale)
        new_h = int(frame_h * scale)
        offset_x = (label_w - new_w) // 2
        offset_y = (label_h - new_h) // 2
        return scale, new_w, new_h, offset_x, offset_y

    
    @property
    def DEBUG_MODE(self):
        return self.debug_checkbox.isChecked()
    
    @staticmethod
    def es_rect_valido(r):
        return (
            isinstance(r, (list, tuple)) and
            len(r) in (4, 5) and
            all(isinstance(v, (int, float)) for v in r)
        )     
        
    def tiempo_a_frame(self, texto):
        try:
            partes = list(map(int, texto.strip().split(":")))
            if len(partes) != 3:
                return None
            horas, minutos, segundos = partes
            total_segundos = horas * 3600 + minutos * 60 + segundos
            return int(total_segundos * self.fps)
        except Exception as e:
            log(f"[ERROR] Formato de tiempo inválido: '{texto}' → {e}")
            return None
                
    def procesar_minuto(self, minuto):
        self.procesando = True
        self.pause_video()
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.thread = MinuteProcessor(self.video_path, minuto, self.fps)
        self.thread.minuto_procesado.connect(self.minuto_listo)
        self.thread.progreso_actualizado.connect(self.progress_bar.setValue)
        self.thread.start()   
        
    def qtime_a_frame_antiguo(self, qtime):
        segundos = qtime.hour() * 3600 + qtime.minute() * 60 + qtime.second()
        return int(segundos * self.fps)

    def qtime_a_frame(self, qtime):
        total_ms = (
            qtime.hour() * 3600000 +
            qtime.minute() * 60000 +
            qtime.second() * 1000 +
            qtime.msec()
        )
        return round((total_ms / 1000) * self.fps)                  
              
class FrameRangeProcessor(QThread):
    resultado_listo = pyqtSignal(dict)
    progreso_actualizado = pyqtSignal(int)

    def __init__(self, video_path, inicio, fin, fps, conf=0.5):
        super().__init__()
        self.video_path = video_path
        self.inicio = inicio
        self.fin = fin
        self.fps = fps
        self.conf = conf

    def run(self):
        log(f"[IA] Procesando frames del {self.inicio} al {self.fin}")
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.inicio)
        total = self.fin - self.inicio
        resultado = {}

        for i, idx in enumerate(range(self.inicio, self.fin)):
            ret, frame = cap.read()
            if not ret:
                break
            
            rects = inferir_result_dml(frame, conf_min=self.conf)
            resultado[idx] = rects
            progreso = int((i / total) * 100)
            self.progreso_actualizado.emit(progreso)

        cap.release()
        log(f"[IA] Procesamiento de {self.inicio} a {self.fin} completado.")
        self.resultado_listo.emit(resultado)

#EXPORT
class ExportThread(QThread):
    progreso = pyqtSignal(int)  # Señal para actualizar la barra
    terminado = pyqtSignal(str, int)  # Señal cuando termina (nombre archivo, frames)

    def __init__(self, app_ref, inicio, fin, nombre_salida, bitrate_kbps):
        super().__init__()
        self.app = app_ref
        self.inicio = inicio
        self.fin = fin
        self.nombre_salida = nombre_salida
        self.bitrate_kbps = bitrate_kbps
        
    def run(self):
        from ffmpeg_gpu_saver import FFmpegGPUSaver
        t_inicio = datetime.now()
        log_ia_click(f"[EXPORT] ▶ Inicio exportación desde frame {self.inicio} hasta {self.fin}")
        
        self.app.cap.set(cv2.CAP_PROP_POS_FRAMES, self.inicio)
        self.app.cap.grab()
        ret, frame = self.app.cap.retrieve()
        if not ret:
            log_ia_click(f"[EXPORT] Error: No se pudo leer el frame inicial {self.inicio}")
            return
        
        target_width = int(self.app.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        target_height = int(self.app.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.app.fps if self.app.fps > 0 else 30

        if target_width == 0 or target_height == 0:
            log_ia_click(f"[EXPORT] Error: Dimensiones del video inválidas (ancho={target_width}, alto={target_height})")
            return

        if self.app.modo_exportacion == "normal":
            saver = cv2.VideoWriter(self.nombre_salida,
                                    cv2.VideoWriter_fourcc(*'avc1'),
                                    fps,
                                    (target_width, target_height))
            usar_ffmpeg = False
        else:
            saver = FFmpegGPUSaver(
                output_path=self.nombre_salida,
                width=target_width,
                height=target_height,
                fps=fps,
                bitrate_kbps=self.bitrate_kbps
            )
            usar_ffmpeg = True

        for idx in range(self.inicio, self.fin):
            if idx == self.inicio or (idx - self.inicio) % 30 == 0:
                self.app.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                self.app.cap.grab()
                ret, frame = self.app.cap.retrieve()
                if not ret:
                    print(f"[EXPORT] Error: No se pudo leer el frame {idx} con set()")
                    break
            else:
                ret, frame = self.app.cap.read()
                if not ret:
                    print(f"[EXPORT] Error: No se pudo leer el frame {idx} con read()")
                    break

            rects_ia = self.app.blur_ia_por_frame.get(idx, [])
            rects_manual = self.app.blur_manual_por_frame.get(idx, [])

            rects_exportar = [
                r for r in rects_ia + rects_manual
                if self.app.es_rect_valido(r)
            ]

            for r in rects_exportar:
                x, y, w, h = r[:4]
                aplicar_blur_capsula(frame, (x, y, w, h), self.app.blur_escala, kernel=self.app.blur_kernel)

            # frame = cv2.resize(frame, (target_width, target_height))  # ← desactivado por optimización

            if usar_ffmpeg:
                saver.write_frame(frame)
            else:
                saver.write(frame)

            self.progreso.emit(idx - self.inicio + 1)

        if usar_ffmpeg:
            saver.close()
        else:
            saver.release()

        t_fin = datetime.now()
        duracion = (t_fin - t_inicio).total_seconds()
        fps_calc = (self.fin - self.inicio + 1) / duracion if duracion > 0 else 0
        log_ia_click(f"[EXPORT] ⏱️ Exportación finalizada en {duracion:.2f} segundos ({fps_calc:.2f} fps promedio)")
        log_ia_click(f"[EXPORT] Modo usado: {self.app.modo_exportacion.upper()}")

        self.terminado.emit(self.nombre_salida, self.fin - self.inicio + 1)

        

    def _recomprimir_con_bitrate(self, input_path):
        import subprocess
        import os

        #nombre_salida_comprimido = f"comprimido_{os.path.basename(input_path)}"
        base = self.app.nombre_video_base
        nuevo_nombre = f"{base} (difuminado).mp4"
        output_path = os.path.join(os.path.dirname(input_path), nuevo_nombre)

        #GPU
        try:
            log_ia_click(f"[EXPORT] Recompresión iniciada: bitrate={self.bitrate_kbps} kbps")
            time.sleep(0.5)
            subprocess.run([
            "ffmpeg",
            "-i", input_path,
            "-c:v", "hevc_amf",
            "-usage", "transcoding",
            "-quality", "quality",
            "-b:v", f"{self.bitrate_kbps}k",  # Control por bitrate
            "-g", "25",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            "-y",
            output_path
        ], check=True)





        #CPU
        # try:
        #     log_ia_click(f"[EXPORT] Recompresión iniciada: bitrate={self.bitrate_kbps} kbps")
        #     time.sleep(0.5)
        #     subprocess.run([
        #     "ffmpeg",
        #     "-i", input_path,
        #     "-c:v", "libx265",
        #     "-preset", "medium",
        #     "-crf", "28",
        #     "-x265-params", "keyint=25:min-keyint=25:no-scenecut=1",
        #     "-c:a", "aac",
        #     "-b:a", "128k",
        #     "-movflags", "+faststart",
        #     "-y",
        #     output_path
        # ])

            log_ia_click(f"[EXPORT] ✅ Recompresión finalizada: {output_path}")

            # ✅ Eliminar el original solo si se creó correctamente el comprimido
            if os.path.exists(output_path):
                try:
                    os.remove(input_path)
                    log_ia_click(f"[EXPORT] 🧹 Archivo original eliminado: {input_path}")
                except Exception as e:
                    log_ia_click(f"[EXPORT] ⚠️ No se pudo eliminar el original: {e}")

            return output_path

        except Exception as e:
            log_ia_click(f"[EXPORT] ⚠️ Error al recomprimir: {e}")
            return input_path

    def _revisarVideo(self, video_path=None):
        import subprocess
        import json
        import os
        import traceback

        if video_path is None:
            video_path = self.nombre_salida

        print(f"\n[REVISAR] 🧪 Iniciando revisión técnica del video: {video_path}")

        if not os.path.exists(video_path):
            print("[REVISAR] ❌ El archivo no existe.")
            return

        try:
            # 1. Obtener info general y del stream de video
            cmd_info = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "format=duration,bit_rate",
                "-show_entries", "stream=codec_name,profile,r_frame_rate,avg_frame_rate,nb_frames",
                "-of", "json", video_path
            ]
            result_info = subprocess.run(cmd_info, capture_output=True, text=True, check=True)
            info = json.loads(result_info.stdout)

            # 2. Buscar todos los frames y sus tipos (I/P/B)
            cmd_frames = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "frame=pts_time,pict_type",
                "-of", "json",
                "-read_intervals", "%+60",  # Escanear el primer minuto
                video_path
            ]
            result_frames = subprocess.run(cmd_frames, capture_output=True, text=True, check=True)
            frames_data = json.loads(result_frames.stdout)

            frames = frames_data.get("frames", [])
            tipos = [f["pict_type"] for f in frames if "pict_type" in f]

            total = len(tipos)
            count_I = tipos.count("I")
            count_P = tipos.count("P")
            count_B = tipos.count("B")

            duracion = float(info["format"].get("duration", 0))
            fps_texto = info["streams"][0].get("avg_frame_rate", "0/1")
            fps = round(eval(fps_texto), 2) if "/" in fps_texto else float(fps_texto)

            print(f"[REVISAR] Duración: {duracion:.2f} segundos")
            print(f"[REVISAR] FPS promedio: {fps}")
            print(f"[REVISAR] Total de frames analizados: {total}")
            print(f"[REVISAR] I-frames: {count_I} | P-frames: {count_P} | B-frames: {count_B}")

            if count_I == 0:
                print("[REVISAR] ❌ No se encontraron I-frames en el primer minuto.")
            elif count_I < 10:
                print("[REVISAR] ⚠️ Muy pocos I-frames. Puede causar freeze en reproducción rápida.")
            else:
                print("[REVISAR] ✅ Cantidad de I-frames adecuada para navegación rápida.")

            # Detectar si el GOP es regular
            if total >= 10 and count_I >= 2:
                distancias = []
                last_i = None
                for idx, tipo in enumerate(tipos):
                    if tipo == "I":
                        if last_i is not None:
                            distancias.append(idx - last_i)
                        last_i = idx
                if distancias:
                    promedio = round(sum(distancias) / len(distancias), 2)
                    print(f"[REVISAR] Estimación de GOP (promedio entre I-frames): {promedio} frames")
                    if promedio > 50:
                        print("[REVISAR] ⚠️ GOP muy largo. Riesgo de freeze en saltos grandes.")
                    else:
                        print("[REVISAR] ✅ GOP razonable.")
            else:
                print("[REVISAR] ⚠️ No se pudo estimar GOP correctamente.")

        except Exception as e:
            print(f"[REVISAR] ❌ Error durante la revisión: {e}")
            print(traceback.format_exc())

class VideoLabel(QLabel):
    def __init__(self, reviewer):
        super().__init__()
        self.reviewer = reviewer  # para acceder a datos como start_point
        self.setMouseTracking(True)
        self.rects_fantasma_visibles = []  # lista de dicts con clave 'rect'

    def paintEvent(self, event):
        super().paintEvent(event)

        if not self.reviewer or not hasattr(self.reviewer, "current_frame") or self.reviewer.current_frame is None:
            return

        painter = QPainter(self)

        if self.reviewer.drawing:
            painter.setPen(QPen(QColor(0, 0, 255), 2, Qt.SolidLine))
            start_x, start_y, end_x, end_y = self.kwarg_calcular_rect_gui_visible(
                p1=self.reviewer.start_point, p2=self.reviewer.end_point
            )
            width = end_x - start_x
            height = end_y - start_y
            painter.drawEllipse(start_x, start_y, width, height)

        
        self.dibujar_interpolacion_fantasma(painter)
        self.dibujar_blurs_fantasma(painter)
        painter.end()

    def dibujar_blurs_fantasma(self, painter):
        if not self.reviewer or not hasattr(self.reviewer, "blurs_en_construccion"):
            return

        for blur in self.reviewer.blurs_en_construccion:
            # ⛔ No mostrar/etiquetar antes del inicio
            if self.reviewer.frame_idx < int(blur.get("frame_inicio_A", 0)):
                continue


        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)

        # Opacidad/tono del “blur” fantasma (ajustables)
        alpha_centro = 90   # 0–255 → más alto = más “oscurecido”
        color_base   = QColor(20, 20, 20)  # tono gris oscuro (puedes variar)

        for blur in self.reviewer.blurs_en_construccion:
            x, y, w, h = blur["rect_B"]
            sx, sy, ex, ey = self.kwarg_calcular_rect_gui_visible(x=x, y=y, w=w, h=h)
            width  = ex - sx
            height = ey - sy

            # ===== RELLENO SUAVE (fake blur con degradado radial) =====
            cx = sx + width  / 2.0
            cy = sy + height / 2.0
            radius = max(width, height) / 2.0

            grad = QRadialGradient(cx, cy, radius)
            grad.setColorAt(0.0, QColor(color_base.red(), color_base.green(), color_base.blue(), alpha_centro))
            grad.setColorAt(0.6, QColor(color_base.red(), color_base.green(), color_base.blue(), int(alpha_centro*0.6)))
            grad.setColorAt(1.0, QColor(color_base.red(), color_base.green(), color_base.blue(), 0))

            painter.setBrush(QBrush(grad))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(sx, sy, width, height)  # ← pinta el “difuminado” dentro

            # ===== CONTORNO AMARILLO =====
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(255, 255, 0), 2, Qt.SolidLine))
            painter.drawEllipse(sx, sy, width, height)

            # ======== BARRA DE ETIQUETAS FLOTANTE (🔍 ⚓ ❌) ========
            label_w, label_h = self.width(), self.height()

            # 1) Fuente y métricas
            font = QFont()
            font.setPointSize(11)  # un poquito más grande para acertar fácil
            painter.setFont(font)
            fm = QFontMetrics(font)

            # 2) Orden y estilos
            items = [
                ("etiqueta_mov",    "🔍", QColor(0, 150, 255)),   # tracking
                ("etiqueta_anchor", "⚓", QColor(255, 215, 0)),   # ancla
                ("etiqueta_x",      "❌", QColor(255, 0, 0)),     # cerrar
            ]

            # Si quieres desactivar ⚓ cuando A==B (sin tramo), descomenta:
            # if blur.get("rect_A") == blur.get("rect_B"):
            #     items = [i for i in items if i[0] != "etiqueta_anchor"]

            # 3) Medidas de cada ítem
            spacing = 8
            min_hit = 22  # tamaño mínimo de hitbox
            texts = [t for _, t, _ in items]
            widths = [fm.horizontalAdvance(t) for t in texts]
            th = fm.height()
            total_w = sum(widths) + spacing * (len(items) - 1)

            # 4) Posición base (centrada sobre el rect), con offset y “flip”
            cx = sx + width / 2
            offset = 8
            baseline_y_above = sy - offset              # baseline del texto si va arriba
            baseline_y_below = sy + height + offset + th  # baseline si va abajo (sumamos alto)

            # ¿Cabe arriba? (parte superior del fondo >= 0)
            bar_top_above = baseline_y_above - th - 3
            use_above = bar_top_above >= 0

            baseline_y = int(baseline_y_above if use_above else baseline_y_below)

            # 5) X inicial centrado y clamped a los bordes del label
            start_x = int(cx - total_w / 2)
            start_x = max(6, min(start_x, label_w - total_w - 6))

            # 6) Fondo semitransparente de toda la barrita (mejora legibilidad)
            bg_margin_x = 6
            bg_margin_y = 3
            bg_x = start_x - bg_margin_x
            bg_y = baseline_y - th - bg_margin_y
            bg_w = total_w + 2 * bg_margin_x
            bg_h = th + 2 * bg_margin_y

            painter.save()
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 0, 0, 110))
            painter.drawRoundedRect(bg_x, bg_y, bg_w, bg_h, 6, 6)
            painter.restore()

            # 7) Dibujo de textos y registro de hitboxes amplias
            x_cursor = start_x
            for (key, text, color), tw in zip(items, widths):
                painter.setPen(QPen(color))
                painter.drawText(x_cursor, baseline_y, text)

                # Caja clickeable (ampliada): centramos el texto dentro del hitbox
                hit_w = max(min_hit, tw)
                hit_h = max(min_hit, th)
                hit_x = x_cursor + (tw - hit_w) // 2
                hit_y = baseline_y - th + (th - hit_h) // 2

                # Clamp de seguridad del hitbox a los bordes del label
                hit_x = max(0, min(hit_x, label_w - hit_w))
                hit_y = max(0, min(hit_y, label_h - hit_h))

                blur[key] = (int(hit_x), int(hit_y), int(hit_w), int(hit_h))

                x_cursor += tw + spacing
            # ======== FIN BARRA DE ETIQUETAS ========

            # ===== Asa de redimensión (SE) – siempre por encima =====
            handle_size = 12  # píxeles GUI (ajusta si quieres)
            # posición base: esquina inferior-derecha del rect visible
            hx = int(sx + width  - handle_size // 2)
            hy = int(sy + height - handle_size // 2)

            # si prefieres que quede ligeramente fuera del borde:
            # hx += 4; hy += 4

            # clamp a los límites del label (para que no se “pierda” fuera)
            label_w, label_h = self.width(), self.height()
            hx = max(0, min(hx, label_w - handle_size))
            hy = max(0, min(hy, label_h - handle_size))

            painter.save()
            painter.setPen(QPen(QColor(255, 255, 0), 1))
            painter.setBrush(QColor(255, 255, 0, 220))  # amarillito sólido
            painter.drawRect(hx, hy, handle_size, handle_size)
            painter.restore()

            # registra hitbox para clicks
            blur["handle_se"] = (hx, hy, handle_size, handle_size)





        painter.restore()

    def dibujar_interpolacion_fantasma(self, painter):
        if not self.reviewer or not hasattr(self.reviewer, "blurs_en_construccion"):
            return

        scale, _, _, offset_x, offset_y = self.reviewer.calcular_escala_y_offset()
        frame_actual = self.reviewer.frame_idx

        # Usaremos este pen para todo el trazo magenta punteado
        pen_magenta = QPen(QColor(255, 0, 255), 1, Qt.DashLine)

        for blur in self.reviewer.blurs_en_construccion:
            inicio = int(blur.get("frame_inicio_A", 0))
            if frame_actual < inicio:
                continue  # ⛔ nada antes del inicio

            # ===== 1) Construir clip con “agujero” (la elipse amarilla del fantasma) =====
            # Tomamos rect_B del blur actual (posición visible del fantasma)
            try:
                xB, yB, wB, hB = blur["rect_B"]
            except Exception:
                continue

            sx, sy, ex, ey = self.kwarg_calcular_rect_gui_visible(x=xB, y=yB, w=wB, h=hB)
            widthB  = ex - sx
            heightB = ey - sy
            if widthB <= 0 or heightB <= 0:
                continue

            pad = 2  # acolchado para evitar que asome el trazo bajo el borde amarillo
            hole_rect = QRectF(sx + pad, sy + pad,
                            max(0, widthB  - 2*pad),
                            max(0, heightB - 2*pad))

            full_path = QPainterPath()
            full_path.addRect(QRectF(self.rect()))  # todo el label, como QRectF

            hole_path = QPainterPath()
            hole_path.addEllipse(hole_rect)         # “agujero” con forma de elipse

            clip_path = full_path.subtracted(hole_path)

            # ===== 2) Activar clip y dibujar SOLO fuera del fantasma =====
            painter.save()
            painter.setClipPath(clip_path)
            painter.setPen(pen_magenta)

            hist = blur.get("tracker_hist") or []
            if hist:
                # Solo hasta frame_actual y nunca antes de 'inicio'
                for f, rect in hist:
                    if f < inicio or f > frame_actual:
                        continue
                    x, y, w, h = rect[:4]
                    gx = int(x * scale + offset_x)
                    gy = int(y * scale + offset_y)
                    gw = int(w * scale)
                    gh = int(h * scale)
                    painter.drawEllipse(gx, gy, gw, gh)
            else:
                # Fallback A→B visual, recortado desde 'inicio'
                a = blur.get("rect_A"); b = blur.get("rect_B")
                if not a or not b:
                    painter.restore()
                    continue
                fa = int(blur.get("frame_inicio_A", inicio))
                fb = int(blur.get("frame_ultimo_B", frame_actual))
                if a != b and fa < fb and frame_actual >= inicio:
                    x1, y1, w1, h1 = a
                    x2, y2, w2, h2 = b
                    f_ini = max(fa + 1, inicio)
                    f_fin = min(fb, frame_actual)
                    denom = max(1, (fb - fa))
                    for f in range(f_ini, f_fin + 1):
                        t = (f - fa) / denom
                        x = int(x1 + (x2 - x1) * t)
                        y = int(y1 + (y2 - y1) * t)
                        w = int(w1 + (w2 - w1) * t)
                        h = int(h1 + (h2 - h1) * t)
                        gx = int(x * scale + offset_x)
                        gy = int(y * scale + offset_y)
                        gw = int(w * scale)
                        gh = int(h * scale)
                        painter.drawEllipse(gx, gy, gw, gh)

            painter.restore()  # 🔚 restaurar clip antes de pasar al siguiente blur


    def dibujar_interpolacion_fantasma_antiguo(self, painter):
        if not self.reviewer or not hasattr(self.reviewer, "blurs_en_construccion"):
            return

        # Pen magenta punteado (lo aplicamos dentro del loop por claridad)
        # painter.setPen(QPen(QColor(255, 0, 255), 1, Qt.DashLine))

        for blur in self.reviewer.blurs_en_construccion:
            a = blur["rect_A"]
            b = blur["rect_B"]
            frame_a = blur["frame_inicio_A"]
            frame_b = blur["frame_ultimo_B"]

            # Si no hay tramo, no hay interpolación
            if a == b or frame_a >= frame_b:
                continue

            # 1) Calcula el rect de la cápsula amarilla (rect_B) en coords GUI
            xB, yB, wB, hB = b
            sx, sy, ex, ey = self.kwarg_calcular_rect_gui_visible(x=xB, y=yB, w=wB, h=hB)
            widthB  = ex - sx
            heightB = ey - sy
            if widthB <= 0 or heightB <= 0:
                continue

            # 2) Crea un clip con "agujero" elíptico donde NO queremos dibujar magenta
            #    (truco: todo el QLabel menos la elipse amarilla)
            pad = 2  # acolchado para evitar que asome el trazo al borde interior
            hole_rect = QRectF(sx + pad, sy + pad, max(0, widthB - 2*pad), max(0, heightB - 2*pad))

            full_path = QPainterPath()
            full_path.addRect(QRectF(self.rect()))  # ✅ convierte a QRectF

            hole_path = QPainterPath()
            hole_path.addEllipse(hole_rect)         # “agujero” con forma de cápsula/elipse

            clip_path = full_path.subtracted(hole_path)

            painter.save()
            painter.setClipPath(clip_path)
            painter.setPen(QPen(QColor(255, 0, 255), 1, Qt.DashLine))  # magenta punteado

            # 3) Dibuja SOLO fuera del agujero
            x1, y1, w1, h1 = a
            x2, y2, w2, h2 = b
            scale, _, _, offset_x, offset_y = self.reviewer.calcular_escala_y_offset()

            for f in range(frame_a + 1, frame_b):
                t = (f - frame_a) / (frame_b - frame_a)
                x = int(x1 + (x2 - x1) * t)
                y = int(y1 + (y2 - y1) * t)
                w = int(w1 + (w2 - w1) * t)
                h = int(h1 + (h2 - h1) * t)

                gx = int(x * scale + offset_x)
                gy = int(y * scale + offset_y)
                gw = int(w * scale)
                gh = int(h * scale)

                painter.drawEllipse(gx, gy, gw, gh)

            painter.restore()


    def kwarg_calcular_rect_gui_visible(self, **kwargs):
        """
        Retorna (start_x, start_y, end_x, end_y) visibles en la GUI.
        Acepta:
        - p1, p2: QPoint para selección manual
        - x, y, w, h: coordenadas absolutas del video
        """
        scale, _, _, offset_x, offset_y = self.reviewer.calcular_escala_y_offset()

        if "p1" in kwargs and "p2" in kwargs:
            p1 = kwargs["p1"]
            p2 = kwargs["p2"]
            start_x = max(offset_x, min(p1.x(), p2.x()))
            start_y = max(offset_y, min(p1.y(), p2.y()))
            end_x = min(offset_x + self.width(), max(p1.x(), p2.x()))
            end_y = min(offset_y + self.height(), max(p1.y(), p2.y()))
            return start_x, start_y, end_x, end_y

        elif all(k in kwargs for k in ("x", "y", "w", "h")):
            x = kwargs["x"]
            y = kwargs["y"]
            w = kwargs["w"]
            h = kwargs["h"]
            gx = int(x * scale + offset_x)
            gy = int(y * scale + offset_y)
            gw = int(w * scale)
            gh = int(h * scale)
            return gx, gy, gx + gw, gy + gh

        raise ValueError("Parámetros inválidos para calcular rect GUI")

    # def mouseMoveEvent(self, event):       
    #     if self.reviewer.drawing:
    #         print(f"[VideoLabel] mouseMoveEvent con drawing=True en pos {event.pos()}")
    #         log("[DEBUG] mouseMoveEvent activado mientras dibujas blur manual")
    #         self.reviewer.end_point = event.pos()
    #         #self.update()
    #     else:
    #         print(f"[VideoLabel] mouseMoveEvent con drawing=False en pos {event.pos()}")
          
    def mousePressEvent(self, event):
        mouse_x = event.pos().x()
        mouse_y = event.pos().y()

        # 1. Clic en 🔍 (etiqueta_mov) → activar seguimiento
        for blur in self.reviewer.blurs_en_construccion:
            if "etiqueta_mov" in blur:
                ex, ey, ew, eh = blur["etiqueta_mov"]
                if ex <= mouse_x <= ex + ew and ey <= mouse_y <= ey + eh:
                    blur["tracking_activado"] = True
                    blur["tracker_hist"] = [(self.reviewer.frame_idx, tuple(blur["rect_B"]))]
                    print(f"[BLUR_FANTASMA] Seguimiento activado para blur ID={blur.get('ID')}")
                    self.update()
                    return

        # 2. Clic en ❌ (etiqueta_x) → eliminar blur
        for blur in self.reviewer.blurs_en_construccion:
            if "etiqueta_x" in blur:
                ex, ey, ew, eh = blur["etiqueta_x"]
                if ex <= mouse_x <= ex + ew and ey <= mouse_y <= ey + eh:
                    self.reviewer.blurs_en_construccion.remove(blur)
                    print(f"[BLUR_FANTASMA] Eliminado con clic en ❌: {blur}")
                    self.update()
                    return

        # ✅ clic derecho: confirmar trayecto del tracker → crear manuales 1:1
        if event.button() == Qt.RightButton:
            for blur in list(self.reviewer.blurs_en_construccion):
                if not blur.get("tracking_activado"):
                    continue

                x, y, w, h = blur["rect_B"]
                sx, sy, ex, ey = self.kwarg_calcular_rect_gui_visible(x=x, y=y, w=w, h=h)
                if sx <= mouse_x <= ex and sy <= mouse_y <= ey:
                    self.reviewer.confirmar_tracker_en_blur(blur)
                    return

        # 3. Si el video está reproduciendo, se ignora el clic
        if hasattr(self.reviewer, "timer") and self.reviewer.timer.isActive():
            print("[VideoLabel] mousePressEvent Click ignorado: video en reproducción.")
            return

        # 4. Pasar control al reviewer
        self.reviewer.handle_mouse_click(event)
        event.accept()
        super().mousePressEvent(event)


    def mouseReleaseEvent(self, event):
        if hasattr(self.reviewer, 'timer'):
            log(f"[DEBUG] mouseReleaseEvent - Timer activo: {self.reviewer.timer.isActive()}")
        if hasattr(self.reviewer, 'handle_mouse_release'):
            self.reviewer.handle_mouse_release(event)
        event.accept()  # ← CLAVE: evitar propagación


class TiempoDialog(QDialog):
    def __init__(self, parent, fps, total_frames):
        super().__init__(parent)
        self.setWindowTitle("Ir a tiempo específico")
        self.setWindowModality(Qt.ApplicationModal)
        self.fps = fps
        self.total_frames = total_frames

        self.setStyleSheet("background-color: #2c2c2c; color: white;")
        self.setFixedSize(250, 100)

        self.time_edit = QTimeEdit()
        self.time_edit.setDisplayFormat("HH:mm:ss")
        self.time_edit.setTime(QTime(0, 0, 0))

        self.ir_button = QPushButton("IR")
        self.ir_button.clicked.connect(self.ir_a_tiempo)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Selecciona tiempo:"))
        layout.addWidget(self.time_edit)
        layout.addWidget(self.ir_button)
        self.setLayout(layout)

    def ir_a_tiempo(self):
        qtime = self.time_edit.time()
        total_ms = (
            qtime.hour() * 3600000 +
            qtime.minute() * 60000 +
            qtime.second() * 1000 +
            qtime.msec()
        )
        frame = round((total_ms / 1000) * self.fps)
        if 0 <= frame < self.total_frames:
            self.parent().pause_video()
            self.parent().frame_idx = frame
            self.parent().slider.setValue(frame)
            self.parent().show_frame(forzado=True)
            self.close()
            self.parent().setFocus()
        else:
            QMessageBox.warning(self, "Tiempo inválido", "El tiempo está fuera del rango del video.")

        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    #-->ESTILO DARK - OSCURO DEL TEMA
    dark_stylesheet = """
    QWidget {
        background-color: #121212;
        color: #FFFFFF;
        font-family: Segoe UI, Arial;
        font-size: 12px;
    }

    QPushButton {
        background-color: #1E1E1E;
        color: #FFFFFF;
        border: 1px solid #333;
        padding: 6px;
        border-radius: 4px;
    }

    QPushButton:hover {
        background-color: #2C2C2C;
    }

    QPushButton:pressed {
        background-color: #444444;
    }

    QLineEdit, QTimeEdit, QSlider, QProgressBar {
        background-color: #1E1E1E;
        color: #FFFFFF;
        border: 1px solid #333;
    }

    QLabel {
        color: #FFFFFF;
    }

    QSlider::groove:horizontal {
        border: 1px solid #444;
        height: 6px;
        background: #2C2C2C;
    }

    QSlider::handle:horizontal {
        background: #888;
        width: 12px;
        margin: -6px 0;
        border-radius: 6px;
    }

    QProgressBar {
        border: 1px solid #444;
        text-align: center;
        background: #1E1E1E;
    }

    QProgressBar::chunk {
        background-color: #0078d7;
        width: 20px;
    }
    """
    app.setStyleSheet(dark_stylesheet)
    #<--ESTILO DARK - OSCURO DEL TEMA
    viewer = VideoReviewer()
    app_instance = viewer
    viewer.show()

    # ✅ Limpieza DirectML/IA al cerrar (hook global)
    def _cleanup_on_exit():
        # 1) Apagar hilo IA si existe
        try:
            ia = getattr(viewer, "ia_thread", None)
            if ia is not None:
                try: ia.progreso_actualizado.disconnect()
                except: pass
                try: ia.procesamiento_terminado.disconnect()
                except: pass
                try: ia.requestInterruption()
                except: pass
                ia.quit(); ia.wait(3000)
                if ia.isRunning():
                    ia.terminate(); ia.wait(1000)
        except: pass

        # 2) Soltar VideoCapture si existe
        try:
            cap = getattr(viewer, "cap", None)
            if cap: cap.release()
        except: pass

        # 3) Liberar DirectML / PyTorch
        try:
            import sys, gc, torch
            try: torch.cuda.empty_cache()
            except: pass
            try:
                import torch_directml
                if "torch_directml" in sys.modules:
                    del sys.modules["torch_directml"]
            except: pass
            try: del globals()["model"]
            except: pass
            try: del globals()["DML"]
            except: pass
            gc.collect()
            print("[DML] Limpiado al salir.")
        except: pass

    app.aboutToQuit.connect(_cleanup_on_exit)
    
    sys.exit(app.exec_())
    
    
