import os

from PIL import Image
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.shortcuts import render
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

from ml.modelTF.loadTrainData import dataset
from ml.modelTF.optimizeModel import loaded_model
from ml.models import UploadedImage


def index(request):
    return HttpResponse("Hello world")


def upload_images(request):
    if request.method == 'POST':
        results = []
        for file_name, file_content in request.FILES.items():
            pil_image = Image.open(file_content)
            pil_image = pil_image.resize((176, 208))
            img_array = np.array(pil_image)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array.astype('float32') / 255.0

            prediction = loaded_model.predict(img_array)

            predicted_class = np.argmax(prediction)
            class_label = dataset[predicted_class]

            results.append({'file_name': file_name, 'predicted_class': class_label})


        return JsonResponse({'status': 'success', 'results': results})
    else:
        return JsonResponse({'status': 'error', 'message': 'Метод не поддерживается'})

def check_video(request):
    if request.method == 'POST':
        results = []
        for file_name, file_content in request.FILES.items():
            cap = cv2.VideoCapture(file_content)

            if not cap.isOpened():
                print("Error: Couldn't open the video file.")
                exit()

            num_frame = 0
            image_1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
            image_2 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
            while True:
                ret, frame = cap.read()

                if(num_frame==0):
                    image_1 = frame
                if(num_frame<5):
                    num_frame += 1
                    continue
                elif (num_frame == 5):
                    image_2 = frame
                    detect_contrast_change(image_1, image_2, 50)
                    num_frame = 0
                frame_resized = cv2.resize(frame, (176, 208))

                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img_array = image.img_to_array(frame_rgb)
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0

                prediction = loaded_model.predict(img_array)
                predicted_class = np.argmax(prediction)
                class_label = dataset[predicted_class]

                results.append({'file_name': file_name, 'predicted_class': class_label})

        return JsonResponse({'status': 'success', 'results': results})
    else:
        return JsonResponse({'status': 'error', 'message': 'Метод не поддерживается'})


def detect_contrast_change(image1, image2, threshold):
    # Вычисляем разницу между двумя изображениями
    diff = cv2.absdiff(image1, image2)

    # Применяем пороговую обработку
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Находим контуры на пороговом изображении
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Проверяем наличие резких изменений контрастности
    contrast_change_detected = len(contours) > 0
    if contrast_change_detected:
      result = "Обнаружено резкий переход контраста"
      return JsonResponse({'status': 'success', 'results': result})
    else:
      result = "Изменение контраста в пределах нормы"
      return JsonResponse({'status': 'success', 'results': result})
    #return contrast_change_detected, diff, thresholded