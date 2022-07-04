import cv2
import os
from src.onnx.detect import onnx_model, onnx_prediction
from src.utils.func import resizeWithAspectRatio, draw_detection, combine_pred_boxes


def run_detection(img):  # data/folder/filename.JPG
    """Get the file"""

    cap = cv2.VideoCapture(img)

    # Step 1 Creating Folder
    # if not os.path.isdir('temporary_files1'):
    #     os.mkdir('temporary_files1')

    # print( f"{path_parent}/{file_name}")

    """Download file from S3"""
    # client.download_file(os.environ['BUCKET'], f"{path_parent}/{file_name}", f'temporary_files1/{file_name}')

    # path = 'temporary_files1/{}'.format(file_name)

    img_list = ['.jpg', '.jpeg', '.png']
    video_list = ['.mp4']

    """If image file"""
    # if suffix.lower() in img_list:
    #     """Copy EXIF data using PIL/Pillow"""
    #     ImgWithEXIF = Image.open(path)
    #     cv_image = cv2.cvtColor(np.array(ImgWithEXIF), cv2.COLOR_RGB2BGR)
    #
    #     """Initiate Keras model and perform prediction"""
    #     #h5_model = keras_model('models/keras/pictor-ppe-v302-a1-yolo-v3-weights.h5')
    #     #boxes, class_names = keras_prediction(cv_image, h5_model)
    #
    #     """Initiate Onnx model and perform prediction"""
    #     # boot_model = onnx_model('models/onnx/boot.onnx')
    #     # boxes2, class_names2 = onnx_prediction(cv_image, boot_model, 'models/onnx/boot.txt')
    #     #excavation_model = onnx_model('models/onnx/excavation.onnx')
    #     stair_model = onnx_model('models/onnx/stairsv2.onnx')
    #     #boxes2, class_names2 = onnx_prediction(cv_image, excavation_model, 'models/onnx/excavation.txt')
    #     boxes3, class_names3 = onnx_prediction(cv_image, stair_model, 'models/onnx/stair.txt')
    #     print(boxes3, class_names3)
    #
    #     # x1, y1, x2, y2 = boxes3
    #     # cls = 'Stairs'
    #
    #     #combined, combined_class_names = combine_pred_boxes([boxes, boxes3], [['Hat', 'Vest', 'Worker'], ['Stairs']])
    #
    #     # if boxes3.size > 0:
    #     #     combine_box, combined_class_names = combine_pred_boxes([boxes3],
    #     #                                                            [['Stairs']])
    #     #draw_detection(cv_image, boxes3, cls, source_file=f"{path_parent}/{file_name}")
    #
    #     """Draw the boxes"""
    #     #draw_detection(cv_image, combined, combined_class_names)
    #     #draw_detection(cv_image, boxes2, class_names2)
    #     draw_detection(cv_image, boxes3, class_names3)
    #
    #     """Show result - must to turn-off in production mode"""
    #     resize = resizeWithAspectRatio(cv_image, width=640)  # Resize by width OR
    #     # resize = ResizeWithAspectRatio(image, height=1280) # Resize by height
    #     cv2.imshow('resize', resize)
    #     cv2.waitKey(0)  # waits until a key is pressed
    #     cv2.destroyAllWindows()  # destroys the window showing image
    #
    #     """Save the annotated image"""
    #     # Convert image from BGR to RGB
    #     RGB_IMG = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    #
    #     # Convert OpenCV image onto PIL Image
    #     OpenCVImageAsPIL = Image.fromarray(RGB_IMG)
    #
    #     # Encode newly-created image into memory as JPEG along with EXIF from other image
    #     out_path = 'temporary_files1/{}_AI{}'.format(file_name_no_suffix, suffix)
    #     OpenCVImageAsPIL.save(out_path, format='JPEG', exif=ImgWithEXIF.info['exif'])
    #
    #     # Upload Prediction to S3 Bucket
    #     client.upload_file(out_path, os.environ['BUCKET'],  f"{path_parent}/{file_name_no_suffix}_AI_{suffix}")
    #
    #     return f"{path_parent}/{file_name_no_suffix}_AI_{suffix}"

    # elif suffix.lower() in video_list:

    # cap = cv2.VideoCapture("rtmp://red5.vertikaliti.com:1935/live/ESSCOMM-INPUT", cv2.CAP_FFMPEG)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    # Check if video opened successfully

    if not cap.isOpened():
        print("Error opening video stream or file.")

        """Config video output"""

        # Default resolutions of the frame are obtained.The default resolutions are system dependent.

        # We convert the resolutions from float to integer.

    # frame_width = int(cap.get(3))

    # frame_height = int(cap.get(4))

    # Define the codec and create VideoWriter object.The output is stored in 'output.avi/mp4' file.

    # Define the fps to be equal to 30. Also frame size is passed. (*'MP4V') ('M', 'J', 'P', 'G')

    # out = cv2.VideoWriter("output",cv2.VideoWriter_fourcc(*'MP4V'),30, (frame_width, frame_height))

    """Initiate Onnx model and perform prediction"""
    person_model = onnx_model('models/onnx/yolov5x.onnx')
    pistol_model = onnx_model('models/onnx/pistol.onnx')

    count = 0

    frame2process = 0

    # Read until video is completed

    while cap.isOpened():

        # Capture frame-by-frame

        ret, frame = cap.read()

        if ret:

            if count == frame2process:
                # boxes, class_names = keras_prediction(frame, h5_model)
                boxes, class_names = onnx_prediction(frame, person_model, 'models/onnx/classes.txt')
                boxes2, class_names2 = onnx_prediction(frame, pistol_model, 'models/onnx/pistol.txt')

                frame2process += 5

            if boxes.size > 0 and boxes2.size > 0:
                combine_box, combined_class_names = combine_pred_boxes([boxes, boxes2],
                                                                       [['Person', 'bicycle',
                                                                         'car',
                                                                         'motorcycle',
                                                                         'airplane',
                                                                         'bus',
                                                                         'train',
                                                                         'truck',
                                                                         'boat',
                                                                         'traffic light',
                                                                         'fire hydrant',
                                                                         'stop sign',
                                                                         'parking meter',
                                                                         'bench',
                                                                         'bird',
                                                                         'cat',
                                                                         'dog',
                                                                         'horse',
                                                                         'sheep',
                                                                         'cow',
                                                                         'elephant',
                                                                         'bear',
                                                                         'zebra',
                                                                         'giraffe',
                                                                         'backpack',
                                                                         'umbrella',
                                                                         'handbag'
                                                                         'tie',
                                                                         'suitcase',
                                                                         'frisbee',
                                                                         'skis',
                                                                         'snowboard',
                                                                         'sports ball',
                                                                         'kite',
                                                                         'baseball bat',
                                                                         'baseball glove',
                                                                         'skateboard',
                                                                         'surfboard',
                                                                         'tennis racket',
                                                                         'bottle',
                                                                         'wine glass',
                                                                         'cup',
                                                                         'fork',
                                                                         'knife',
                                                                         'spoon',
                                                                         'bowl',
                                                                         'banana',
                                                                         'apple',
                                                                         'sandwich',
                                                                         'orange',
                                                                         'broccoli',
                                                                         'carrot',
                                                                         'hot dog',
                                                                         'pizza',
                                                                         'donut',
                                                                         'cake',
                                                                         'chair',
                                                                         'couch',
                                                                         'potted plant',
                                                                         'bed',
                                                                         'dining table',
                                                                         'toilet',
                                                                         'tv',
                                                                         'laptop',
                                                                         'mouse',
                                                                         'remote',
                                                                         'keyboard',
                                                                         'cell phone',
                                                                         'microwave',
                                                                         'oven',
                                                                         'toaster',
                                                                         'sink',
                                                                         'refrigerator',
                                                                         'book',
                                                                         'clock',
                                                                         'vase',
                                                                         'scissors',
                                                                         'teddy bear',
                                                                         'hair drier',
                                                                         'toothbrush'], ['Pistol']])

                draw_detection(frame, combine_box, combined_class_names)
                print("Box", combine_box)


            elif boxes.size > 0:
                combine_box, combined_class_names = combine_pred_boxes([boxes],
                                                                       [['Person', 'bicycle',
                                                                         'car',
                                                                         'motorcycle',
                                                                         'airplane',
                                                                         'bus',
                                                                         'train',
                                                                         'truck',
                                                                         'boat',
                                                                         'traffic light',
                                                                         'fire hydrant',
                                                                         'stop sign',
                                                                         'parking meter',
                                                                         'bench',
                                                                         'bird',
                                                                         'cat',
                                                                         'dog',
                                                                         'horse',
                                                                         'sheep',
                                                                         'cow',
                                                                         'elephant',
                                                                         'bear',
                                                                         'zebra',
                                                                         'giraffe',
                                                                         'backpack',
                                                                         'umbrella',
                                                                         'handbag'
                                                                         'tie',
                                                                         'suitcase',
                                                                         'frisbee',
                                                                         'skis',
                                                                         'snowboard',
                                                                         'sports ball',
                                                                         'kite',
                                                                         'baseball bat',
                                                                         'baseball glove',
                                                                         'skateboard',
                                                                         'surfboard',
                                                                         'tennis racket',
                                                                         'bottle',
                                                                         'wine glass',
                                                                         'cup',
                                                                         'fork',
                                                                         'knife',
                                                                         'spoon',
                                                                         'bowl',
                                                                         'banana',
                                                                         'apple',
                                                                         'sandwich',
                                                                         'orange',
                                                                         'broccoli',
                                                                         'carrot',
                                                                         'hot dog',
                                                                         'pizza',
                                                                         'donut',
                                                                         'cake',
                                                                         'chair',
                                                                         'couch',
                                                                         'potted plant',
                                                                         'bed',
                                                                         'dining table',
                                                                         'toilet',
                                                                         'tv',
                                                                         'laptop',
                                                                         'mouse',
                                                                         'remote',
                                                                         'keyboard',
                                                                         'cell phone',
                                                                         'microwave',
                                                                         'oven',
                                                                         'toaster',
                                                                         'sink',
                                                                         'refrigerator',
                                                                         'book',
                                                                         'clock',
                                                                         'vase',
                                                                         'scissors',
                                                                         'teddy bear',
                                                                         'hair drier',
                                                                         'toothbrush']])
                draw_detection(frame, combine_box, combined_class_names)
                print("Box", combine_box)

            elif boxes2.size > 0:
                combine_box, combined_class_names = combine_pred_boxes([boxes2],
                                                                       [['Pistol']])
                draw_detection(frame, combine_box, combined_class_names)
                print("Box", combine_box)

            """Draw the boxes"""
            # draw_detection(frame, boxes, class_names)
            # draw_detection(frame, boxes2, class_names2)

            # Write the frame into the file 'output.avi/mp4'

            # out.write(frame)

            # counting the frame

            count += 1

            """Show result - must to turn-off in production mode"""

            resize = resizeWithAspectRatio(frame, width=1280)

            cv2.imshow('Video', resize)

            # Press Q on keyboard to  exit

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break


        # Break the loop

        else:

            break

        # When everything done, release the video capture object

    cap.release()

    # out.release()

    # Closes all the frames

    cv2.destroyAllWindows()


if __name__ == '__main__':
    # run_detection('dosh/khairul_images/excavation/DJI_20220210150727_0164_Z.JPG')

    # run_detection('dosh/khairul_images/stair/Test/DJI_20220225111405_0411_Z.JPG')
    # run_detection('testdronos.mp4')
    run_detection('testdronos.mp4')

    # run_detection('D:/Intern/DOSH Khairul/images/DJI_20220210151723_0314_Z.JPG')
    # run_detection('data/PPE/6. Site Ara Damansara/DJI_20211101144342_0067_Z.JPG')
    # run_detection('data/PPE/6. Site Ara Damansara/DJI_20211101144653_0113_Z.MP4')
