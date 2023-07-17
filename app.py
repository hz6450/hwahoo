from flask import Flask, render_template, request
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch
import cv2
import numpy as np

app = Flask(__name__)

# 모델과 변환기 초기화
model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
transforms = ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier')

def predict(im):
    labels = {0: "0-2", 1: "3-9", 2: "10-19", 3: "20-29", 4: "30-39", 5: "40-49", 6: "50-59", 7: "60-69", 8: "70세 이상"}

    # 이미지 변환 후 모델 통과
    inputs = transforms(im, return_tensors='pt')
    output = model(**inputs)

    # 예측된 클래스 확률
    proba = output.logits.softmax(1)

    # 예측된 클래스
    preds = proba.argmax(1)
    values, indices = torch.topk(proba, k=1)

    return {labels[indices.item()]: v.item() for i, v in zip(indices.numpy()[0], values.detach().numpy()[0])}

@app.route('/face', methods=['GET'])
def face():
    return render_template('face.html')




@app.route('/face', methods=['POST'])
def process_age_prediction():
    if 'image' in request.files:
        # 이미지 파일이 제공된 경우
        image = request.files['image']
        img = Image.open(image)
        result = predict(img)
        return render_template('face.html', age=result)  # age만 템플릿으로 전달
    else:
        # 이미지 파일이 제공되지 않았을 경우 카메라 접근 시도
        capture = cv2.VideoCapture(0)
        ret, frame = capture.read()
        capture.release()

        if ret:
            # 카메라 캡처 성공
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = predict(img)
            return render_template('face.html', age=result)  # age만 템플릿으로 전달
        else:
            # 카메라 캡처 실패
            return render_template('face.html', error='카메라 캡처에 실패했습니다.')



@app.route('/')
def logo():
    return render_template('logo.html')

@app.route('/ocr')
def ocr():
    return render_template('ocr.html')

@app.route('/index')
def index():
    return render_template('index.html')

posts = [
    {"title": "3D 모델링", "content": "3D 모델링 만들어 주실 분 구합니다.\n간단한 모양입니다.", "image": "/static/images/image1.jpg"},
    {"title": "웹 페이지 개발", "content": "기능은 다 만들었는데 이 기능과 사이트 분위기에 맞는\n디자인으로 만들어 주실 분 구합니다.", "image": "/static/images/image2.jpg"},
    {"title": "영어 과외", "content": "토익 800점 이상 선생님을 원합니다.\n기간은 상담을 통해 정하고 싶습니다.", "image": "/static/images/image3.jpg"},
    {"title": "영어 논문 작성", "content": "제가 한글로 작성한 논문을 영어로 변경해 주실 분!!\n페이는 상담으로 얘기해보고 싶습니다.", "image": "/static/images/image4.jpg"},
    {"title": "요리 교실", "content": "주말마다 취미로 요리를 배우고 싶습니다.\n현재 3명 정도 같이 하기로 했습니다.", "image": "/static/images/image5.jpg"}
]

@app.route('/main')
def main():
    return render_template('main.html',posts=posts)


if __name__ == '__main__':
    app.run(debug=True)
