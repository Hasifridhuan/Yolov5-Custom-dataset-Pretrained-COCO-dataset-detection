import cv2
import numpy as np

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4

colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]


def onnx_model(weight_path):
    is_cuda = True
    net = cv2.dnn.readNet(weight_path)

    if is_cuda:
        print("Attempt to use CUDA")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        print("Running on CPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    return net


def onnx_prediction(CV_FRAME, MODEL, CLASS_PATH):
    row, col, _ = CV_FRAME.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = CV_FRAME

    blob = cv2.dnn.blobFromImage(result, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    MODEL.setInput(blob)
    preds = MODEL.forward()
    print(type(preds))

    class_ids = []
    confidences = []
    boxes = []

    rows = preds[0].shape[0]

    image_width, image_height, _ = result.shape

    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    for r in range(rows):
        row = preds[0][r]
        confidence = row[4]
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if classes_scores[class_id] > SCORE_THRESHOLD:
                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD)

    # result_class_ids = []
    # result_confidences = []
    # result_boxes = []
    list_col = []

    for i in indexes:
        list = [boxes[i][0], boxes[i][1], (boxes[i][0] + boxes[i][2]), (boxes[i][1] + boxes[i][3]), confidences[i],
                class_ids[i]]
        list_col.append(list)

        # result_confidences.append(confidences[i])
        # result_class_ids.append(class_ids[i])
        # result_boxes.append(boxes[i])

    ndarray = np.array(list_col)

    with open(CLASS_PATH, "r") as f:
        class_list_name = [(cname.strip()).title() for cname in f.readlines()]

    return ndarray, class_list_name
