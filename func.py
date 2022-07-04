import cv2
import matplotlib as mpl
import numpy as np


def combine_pred_boxes(LIST,
                       LIST2):  # list 1 is prediction and list 2 is labelling extract from last array in detection
    i = 0
    label = []

    while i < len(LIST):
        for idx, val in enumerate(LIST[i]):
            label.append(LIST2[i][int(val[-1])])

        i += 1

    if len(LIST) > 1:
        combined = np.append(LIST[0], LIST[1], axis=0)
        # print(type(combined))
    else:
        combined = np.array(LIST[0])
        # print(type(combined))
    return combined, label


def isRectangleOverlap(R1, R2):
    if (R1[0] > R2[2]) or (R1[2] < R2[0]) or (R1[3] < R2[1]) or (R1[1] > R2[3]):
        return False
    else:
        return True


def resizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def draw_detection(
        img,
        boxes,
        class_names,
        # drawing configs
        font=cv2.FONT_HERSHEY_DUPLEX,
        font_scale=1,
        box_thickness=10,
        border=10,
        text_color=(255, 255, 255),
        text_weight=1,
        clr=(255, 0, 0)

):
    '''
    Draw the bounding boxes on the image
    '''
    # generate some colors for different classes
    num_classes = len(class_names)  # number of classes
    colors = [mpl.colors.hsv_to_rgb((i / num_classes, 1, 1)) * 255 for i in range(num_classes)]

    result_pistol = []
    result_person = []
    Truelst_person = []
    True_lst = []  # pistol

    detection = []
    wkr = 1

    # draw the detections
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box[:4].astype(int)
        score = box[-2]
        cluster = int(box[-1])
        label = class_names[idx]

        if label == 'Pistol':
            if score > 0.5:
                result_pistol.append(([x1, y1, x2, y2], score, label))

        if label == 'Person':
            person = {}
            person["Person"] = [x1, y1, x2, y2]
            detection.append(person)
            result_person.append((wkr, [x1, y1, x2, y2], score, label))
            Truelst_person.append(Truelst_person)
            wkr = wkr + 1

    textg = "Pistol"
    textp = "Person"
    Type = []
    Confidence = []

    for id, ap, p, clas in result_person:
        x1, y1, x2, y2 = ap
        for ag, g, cls in result_pistol:
            xg, yg, xg2, yg2 = ag
            intercept = isRectangleOverlap([x1, y1, x2, y2], [xg, yg, xg2, yg2])
            if intercept:
                Type.append(textg)
                Confidence.append(p)
                detection[id - 1]["Weapon"] = Type
                detection[id - 1]["Conf"] = Confidence

                True_lst.append(True_lst)

    if len(True_lst) >= 1 and len(Truelst_person) >= 1:
        print("False")
        for id, ap, p, clas in result_person:
            x1, y1, x2, y2 = ap
            img = cv2.rectangle(img, (x1, y1), (x2, y2), clr, box_thickness)
            text = f'{clas} {p * 100:.0f}%'
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)

            # background rectangle for the text(person)
            tb_x1 = x1 - box_thickness // 2
            tb_y1 = y1 - box_thickness // 2 - th - 2 * border
            tb_x2 = x1 + tw + 2 * border
            tb_y2 = y1

            img = cv2.rectangle(img, (tb_x1, tb_y1), (tb_x2, tb_y2), clr, 1)
            img = cv2.putText(img, text, (x1 + border, y1 - border), font, font_scale, text_color, text_weight,
                              cv2.LINE_AA)

        for ag, g, cls in result_pistol:
            print(cls)
            xg, yg, xg2, yg2 = ag

            img = cv2.rectangle(img, (xg, yg), (xg2, yg2), (0, 0, 255), box_thickness)

            # text: <object class> (<confidence score in percent>%)

            textg = f'{cls} {g * 100:.0f}%'

            # get width (tw) and height (th) of the text

            (tw2, th2), _ = cv2.getTextSize(textg, font, font_scale, 1)

            # background rectangle for the text(pistol)
            tbg_x1 = xg - box_thickness // 2
            tbg_y1 = yg - box_thickness // 2 - th2 - 2 * border
            tbg_x2 = xg + tw2 + 2 * border
            tbg_y2 = yg

            # draw the background rectangle

            img = cv2.rectangle(img, (tbg_x1, tbg_y1), (tbg_x2, tbg_y2), (0, 0, 255), -1)

            # put the text

            img = cv2.putText(img, textg, (xg + border, yg - border), font, font_scale, text_color, text_weight,
                              cv2.LINE_AA)

    elif len(Truelst_person) >= 1:
        # print("True")
        for id, ap, p, clas in result_person:
            x1, y1, x2, y2 = ap
            img = cv2.rectangle(img, (x1, y1), (x2, y2), clr, box_thickness)

            text = f'{clas} {p * 100:.0f}%'

            (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)

            tb_x1 = x1 - box_thickness // 2
            tb_y1 = y1 - box_thickness // 2 - th - 2 * border
            tb_x2 = x1 + tw + 2 * border
            tb_y2 = y1

            img = cv2.rectangle(img, (tb_x1, tb_y1), (tb_x2, tb_y2), clr, -1)
            img = cv2.putText(img, text, (x1 + border, y1 - border), font, font_scale, text_color, text_weight,
                              cv2.LINE_AA)

    # print(detection)

    return img




