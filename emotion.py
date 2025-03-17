import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 모델 로드
model = load_model('FER_model.h5')  # 파일명 맞춤
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 이미지 로드
img = cv2.imread('test.jpg', 0)  # 흑백
faces = face_cascade.detectMultiScale(img, 1.1, 4)

# 감정 예측
emotions = ['Angry', 'Happy', 'Sad', 'Neutral', 'Surprise', 'Disgust', 'Fear']
for (x, y, w, h) in faces:
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    face = np.expand_dims(face, axis=0) / 255.0
    prediction = model.predict(face)
    emotion = emotions[np.argmax(prediction)]
    print(f"감정: {emotion}")