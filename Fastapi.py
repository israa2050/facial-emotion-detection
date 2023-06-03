import io
import json
from PIL import Image
import torchvision
# import python-multipart
# from segmentation import get_image_from_bytes
# from starlette.responses import Response ,FileResponse
from fastapi import FastAPI, Response
from fastapi import File, FastAPI
import torch
import cv2
import yaml
import tqdm
import ultralytics
import numpy as np
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse, StreamingResponse
import tensorflow as tf
import tensorflow_addons as tfa

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt')
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
# load weights into new model
emotion_model= tf.keras.models.load_model("./emotionl_last.h5",custom_objects={'my_custom_loss': tfa.metrics.F1Score})
print("Loaded model from disk")
app = FastAPI()

@app.post("/object-to-img")
async def detect_food_return_base64_img():
    cap = cv2.VideoCapture(0)
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        print(ret)
        if not ret:
            break
        # Inference
        results = model(frame)

        crops = results.crop(save=True, save_dir='runs/detect/exp')  # specify save dir
        detections = results.pandas().xyxy[0].to_dict(orient='records')
        for detection in detections:
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            gray_img = cv2.cvtColor(crops[detections.index(detection)]["im"], cv2.COLOR_RGB2GRAY)
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray_img, (48, 48)), -1), 0)
            # predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x1+5, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        _, img_encoded = cv2.imencode(".jpg", frame)

        img_bytes = img_encoded.tobytes()
        response = Response(content=img_bytes, media_type="image/jpeg")
        return response