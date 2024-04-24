# Inference for ONNX model

import cv2
import tensorflow as tf
import onnxruntime as ort
import numpy as np
import random

# TensorFlow 1.x의 v2 호환성을 활성화
tf.compat.v1.disable_v2_behavior()

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

# ONNX 모델 로딩 및 세션 설정
w = "yolov7-tiny.onnx"
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession(w, providers=providers)

# 클래스 이름 및 색상 설정
names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
         'hair drier', 'toothbrush']
colors = {name: [random.randint(0, 255) for _ in range(3)] for name in names}

# 웹캠 설정
webcam = cv2.VideoCapture(0)
webcam.set(3, 640)
webcam.set(4, 480)
ratio_factor = 0.9

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

while webcam.isOpened():
    status, img = webcam.read()

    height, width = img.shape[:2]
    img = cv2.resize(img, (1280, 720))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 이미지 전처리
    image = img.copy()
    image, ratio, dwdh = letterbox(image, auto=False)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)
    im = image.astype(np.float32) / 255

    # ONNX 모델 추론
    outname = [i.name for i in session.get_outputs()]
    inname = [i.name for i in session.get_inputs()]
    inp = {inname[0]: im}

    outputs = session.run(outname, inp)[0]

    ori_images = [img.copy()]

    # 검출된 객체 시각화
    for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):
        image = ori_images[int(batch_id)]
        box = np.array([x0, y0, x1, y1])
        box -= np.array(dwdh * 2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        score = round(float(score), 3)
        name = names[cls_id]
        color = colors[name]
        name += ' ' + str(score)
        cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), color, 2)
        cv2.putText(image, name, (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [255, 255, 255], thickness=2)

    # 화면에 출력
    text = "LEEYUMIN_2314209"
    cv2.putText(image, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("results", image)

    # 객체 검출 정보 출력
    print('[INFO] Detected Objects:')
    for (batch_id, x0, y0, x1, y1, cls_id, score) in outputs:
        cls_name = names[int(cls_id)]
        print(f"  - {cls_name}: {score:.3f}")
        print("LEEYUMIN_2314209")

    # 종료 조건
    if cv2.waitKey(10) & 0xFF == ord('q'):
        print("LEEYUMIN_2314209")
        break

# 웹캠 해제 및 창 닫기
webcam.release()
cv2.destroyAllWindows()
