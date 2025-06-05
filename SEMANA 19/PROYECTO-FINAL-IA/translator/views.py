from django.shortcuts import render
from django.http import JsonResponse
import cv2, numpy as np
import logging

logger = logging.getLogger(__name__)

def home(request):
    return render(request, 'index.html')

def predict_frame(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)
    up = request.FILES.get("frame")
    if not up:
        return JsonResponse({"error": "no file"}, status=400)
    img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JsonResponse({"error": "invalid image"}, status=400)
    # Aquí tu función que devuelve la letra:
    from .ml.hand_utils import predict_from_frame
    letter, conf = predict_from_frame(img)  # debe devolver, p.ej., 'A', 'B', ...
    return JsonResponse({"letter": letter, "conf": round(conf,2)})

