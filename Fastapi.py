from fastapi import FastAPI, Response
from fastapi import File, FastAPI
import torch
import cv2
import numpy as np
from fastapi import File
import tensorflow as tf
import tensorflow_addons as tfa

# Load the face detection model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt')
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
# load emotion classification model
emotion_model= tf.keras.models.load_model("./emotionl_last.h5",custom_objects={'my_custom_loss': tfa.metrics.F1Score})
print("Loaded model from disk")
app = FastAPI()

# Define a function to capture an image from the webcam
def capture_image():
    cap = cv2.VideoCapture(0) # 0 indicates the default camera (usually the built-in webcam)
    ret, frame = cap.read() # Read a frame from the camera
    frame = cv2.resize(frame, (1280, 720))
    cap.release() # Release the camera
    return frame

@app.post("/image")
async def detect_face_emotion_img(file:bytes = File(None)):
    if file is not None:
        input_image = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
    else:
        input_image = capture_image()


    # detecting the faces
    results = model(input_image)
    #Getting the cropped faces detected in the photo
    crops = results.crop(save=True, save_dir='runs/detect/exp')  # specify save dir
    detections = results.pandas().xyxy[0].to_dict(orient='records')
    for detection in detections:
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        #drawing bounding boxes around the faces
        cv2.rectangle(input_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #processing croped face to be reday as input of emotin classification model
        gray_img = cv2.cvtColor(crops[detections.index(detection)]["im"], cv2.COLOR_RGB2GRAY)
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray_img, (48, 48)), -1), 0)
        # predict the emotion
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        #putting the emotion detected of the face on the image
        cv2.putText(input_image, emotion_dict[maxindex], (x1+5, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    _, img_encoded = cv2.imencode(".jpg", input_image)

    img_bytes = img_encoded.tobytes()
    response = Response(content=img_bytes, media_type="image/jpeg")
    #return the hole image with faces detected and the emotions
    return response