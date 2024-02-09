/v4l2-ctl --list-devices

# before edit the code infernce 


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow as tf
import cv2
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np


# Function to calculate buffer size and expand bounding box coordinates
def expand_box_with_buffer(box, buffer_percentage, image_shape):
    y_min, x_min, y_max, x_max = box
    height_buffer = int((y_max - y_min) * buffer_percentage)
    width_buffer = int((x_max - x_min) * buffer_percentage)
    
    y_min_expanded = max(0, y_min - height_buffer)
    y_max_expanded = min(image_shape[0], y_max + height_buffer)
    x_min_expanded = max(0, x_min - width_buffer)
    x_max_expanded = min(image_shape[1], x_max + width_buffer)
    
    return y_min_expanded, x_min_expanded, y_max_expanded, x_max_expanded

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = '/home/cognitica-i7-13thgen/NPS/Tensorflow-model-inference/fine_tuned_model/content/fine_tuned_model/saved_model'

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = '/home/cognitica-i7-13thgen/NPS/Tensorflow-model-inference/ppe_label_map.pbtxt'

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = 0.5

# LOAD THE MODEL
print('Loading model...', end='')
start_time = time.time()

# LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
detect_fn = tf.saved_model.load(PATH_TO_MODEL_DIR)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# LOAD LABEL MAP DATA FOR PLOTTING
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


# Function to process each frame and save annotated video
def process_video(output_path, input_path=0):
    # Open the video file
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if (cap.isOpened() == False): 
        print("Unable to read camera feed")
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    # fps
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (0,100)
    fontScale              = 1
    fontColor              = (255,255,255)
    thickness              = 1
    lineType               = 2




    # used to record the time when we processed last frame 
    prev_frame_time = 0
    
    # used to record the time at which we processed current frame 
    new_frame_time = 0

    MIN_CONF_THRESH_inner = 0.3

    while cap.isOpened():
        ret, image_np = cap.read()
        print(np.shape(image_np))
        if not ret:
            break
        

        new_frame_time = time.time()
        # i +=1

        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time 
        # Convert to RGB (if needed) and expand dimensions to create a batch of size 1
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image_rgb, axis=0)

        image_not = image_np
        # Run object detection
        detections = detect_fn(image_expanded)


        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        # Draw bounding boxes and labels on the image
        for i in range(num_detections):
            box = detections['detection_boxes'][i]
            class_id = detections['detection_classes'][i]
            score = detections['detection_scores'][i]

            if score >= MIN_CONF_THRESH:
                y_min, x_min, y_max, x_max = box
                y_min = int(y_min * frame_height)
                x_min = int(x_min * frame_width)
                y_max = int(y_max * frame_height)
                x_max = int(x_max * frame_width)
                '''
                # Draw bounding box and label on the image
                # cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # label = f'{category_index[class_id]["name"]} {int(score * 100)}%'
                # cv2.putText(image_np, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    '''
                # person identification
                if category_index[class_id]["name"]=='Person':
                    roi_person = image_not[y_min:y_max, x_min:x_max]
                    # print(y_min,y_max, x_min,x_max)

                    
                    try:
                        cv2.imshow('Person alone',cv2.resize(roi_person,(640, 240)))
                        # top half
                        top_half = roi_person[:frame_height//3, :].copy() 
                        image_expanded_top = np.expand_dims(cv2.resize(top_half,(480,640)), axis=0)
                        
                        frame_width,frame_height,ch = np.shape(top_half)
                        top_detections = detect_fn(image_expanded_top)
                        top_num_detections = int(top_detections.pop('num_detections'))
                        top_detections = {key: value[0, :top_num_detections].numpy() for key, value in top_detections.items()}
                        top_detections['num_detections'] = top_num_detections
                        top_detections['detection_classes'] = top_detections['detection_classes'].astype(np.int64)

                        # Draw bounding boxes and labels on the image
                        for i in range(top_num_detections):
                            box = top_detections['detection_boxes'][i]
                            class_id = top_detections['detection_classes'][i]
                            score = top_detections['detection_scores'][i]

                            if score >= MIN_CONF_THRESH:
                                y_min, x_min, y_max, x_max = box
                                y_min = int(y_min * frame_height)
                                x_min = int(x_min * frame_width)
                                y_max = int(y_max * frame_height)
                                x_max = int(x_max * frame_width)

                                # Draw bounding box and label on the image
                                cv2.rectangle(top_half, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                                label = f'{category_index[class_id]["name"]} {int(score * 100)}%'
                                cv2.putText(top_half, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)                   
                                cv2.imshow('top_half',top_half)



                        # bottom half
                        bottom_half = roi_person[frame_height//2:, :]
                        cv2.imshow('bottom_half',cv2.resize(bottom_half,(640, 240)))
                    except Exception as Error:
                        print(Error)


        # Ensure top_half and bottom_half have the same number of rows as the full frame
        # top_half = cv2.resize(top_half, (frame_width, frame_height//2))
        # bottom_half = cv2.resize(bottom_half, (frame_width, frame_height//2))

        
        
        
        cv2.putText(image_np,f'FPS : {fps:.2f}', 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)
	
        # Save annotated frame to the output video
        cv2.imshow('frame',image_np)
        out.write(image_np)

        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# PROVIDE PATH TO VIDEO FILE
# VIDEO_PATH = '/home/cognitica-i7-13thgen/NPS/temp_camera_tf/30FPS_longoutpy.avi'


# Process the video and save annotated version
output_video_path = 'annotated_video.mp4'
process_video(output_video_path)
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# ver2 : 4 frames 
'''
frames is segregating the frames for the person is good with the threshhold of 0.2
Class -oops used 
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow as tf
import cv2
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np


class ObjectDetectionProcessor:
    def __init__(self, model_dir, labels_path, min_conf_thresh=0.5):
        self.MIN_CONF_THRESH = min_conf_thresh
        self.category_index = label_map_util.create_category_index_from_labelmap(labels_path, use_display_name=True)
        self.detect_fn = tf.saved_model.load(model_dir)

    def expand_box_with_buffer(self, box, buffer_percentage, image_shape):
        y_min, x_min, y_max, x_max = box
        height_buffer = int((y_max - y_min) * buffer_percentage)
        width_buffer = int((x_max - x_min) * buffer_percentage)
        
        y_min_expanded = max(0, y_min - height_buffer)
        y_max_expanded = min(image_shape[0], y_max + height_buffer)
        x_min_expanded = max(0, x_min - width_buffer)
        x_max_expanded = min(image_shape[1], x_max + width_buffer)
        
        return y_min_expanded, x_min_expanded, y_max_expanded, x_max_expanded

    def process_frame(self, image_np):
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image_rgb, axis=0)
        detections = self.detect_fn(image_expanded)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        for i in range(num_detections):
            box = detections['detection_boxes'][i]
            class_id = detections['detection_classes'][i]
            score = detections['detection_scores'][i]

            if score >= .2 and self.category_index[class_id]["name"] == 'Person':
                y_min, x_min, y_max, x_max = box
                y_min = int(y_min * image_np.shape[0])
                x_min = int(x_min * image_np.shape[1])
                y_max = int(y_max * image_np.shape[0])
                x_max = int(x_max * image_np.shape[1])
                
                roi_person = image_np[y_min:y_max, x_min:x_max]

                # Additional processing for top and bottom halves
                self.process_person_alone(roi_person)
                self.process_top_half(roi_person)
                self.process_bottom_half(roi_person)

    def process_person_alone(self, roi_person):
        cv2.imshow('Person Alone',cv2.resize(roi_person, (640, 240)))

    def process_top_half(self, roi_person):
        top_half = roi_person[:roi_person.shape[0]//3, :].copy()
        # frame_width, frame_height, _ = np.shape(top_half)
        # image_expanded_top = np.expand_dims(top_half, axis=0)

        # top_detections = self.detect_fn(image_expanded_top)
        # top_num_detections = int(top_detections.pop('num_detections'))
        # top_detections = {key: value[0, :top_num_detections].numpy() for key, value in top_detections.items()}
        # top_detections['num_detections'] = top_num_detections
        # top_detections['detection_classes'] = top_detections['detection_classes'].astype(np.int64)

        # for i in range(top_num_detections):
        #     box = top_detections['detection_boxes'][i]
        #     class_id = top_detections['detection_classes'][i]
        #     score = top_detections['detection_scores'][i]

        #     if score >= self.MIN_CONF_THRESH:
        #         y_min, x_min, y_max, x_max = box
        #         y_min = int(y_min * frame_height)
        #         x_min = int(x_min * frame_width)
        #         y_max = int(y_max * frame_height)
        #         x_max = int(x_max * frame_width)

        #         cv2.rectangle(top_half, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        #         label = f'{self.category_index[class_id]["name"]} {int(score * 100)}%'
        #         cv2.putText(top_half, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)                   
        cv2.imshow('Top Half',  cv2.resize(top_half, (640, 240)))

    def process_bottom_half(self, roi_person):
        bottom_half = roi_person[roi_person.shape[0]//2:, :]
        cv2.imshow('Bottom Half', cv2.resize(bottom_half, (640, 240)))

    def process_video(self, output_path, input_path=0):
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            print("Unable to read video feed")
            return

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(5))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner_of_text = (0, 100)
        font_scale = 1
        font_color = (255, 255, 255)
        thickness = 1
        line_type = 2

        prev_frame_time = 0
        new_frame_time = 0

        while cap.isOpened():
            ret, image_np = cap.read()
            if not ret:
                break

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time) 
            prev_frame_time = new_frame_time

            self.process_frame(image_np)

            cv2.putText(image_np, f'FPS : {fps:.2f}', bottom_left_corner_of_text, font, font_scale, font_color, thickness, line_type)
            cv2.imshow('Frame', image_np)
            out.write(image_np)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # PROVIDE PATH TO MODEL DIRECTORY
    model_dir = '/home/cognitica-i7-13thgen/NPS/Tensorflow-model-inference/fine_tuned_model/content/fine_tuned_model/saved_model'

    # PROVIDE PATH TO LABEL MAP
    labels_path = '/home/cognitica-i7-13thgen/NPS/Tensorflow-model-inference/ppe_label_map.pbtxt'

    # Create an instance of the ObjectDetectionProcessor class
    processor = ObjectDetectionProcessor(model_dir, labels_path, min_conf_thresh=0.5)

    # Provide the output video path
    output_video_path = 'annotated_video.mp4'

    # Process the video and save the annotated version
    processor.process_video(output_video_path)
# ------------------------------------------------------------------------------------------------------------------------------------------------------
    
# VER : 3 expanded video 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow as tf
import cv2
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np


class ObjectDetectionProcessor:
    def __init__(self, model_dir, labels_path, min_conf_thresh=0.5):
        self.MIN_CONF_THRESH = min_conf_thresh
        self.category_index = label_map_util.create_category_index_from_labelmap(labels_path, use_display_name=True)
        self.detect_fn = tf.saved_model.load(model_dir)

    def expand_box_with_buffer(self, box, buffer_percentage, image_shape):
        y_min, x_min, y_max, x_max = box
        y_min = int(y_min * image_shape.shape[0])
        x_min = int(x_min * image_shape.shape[1])
        y_max = int(y_max * image_shape.shape[0])
        x_max = int(x_max * image_shape.shape[1])
        height_buffer = int((y_max - y_min) * buffer_percentage)
        width_buffer = int((x_max - x_min) * buffer_percentage)

        print(f"Original Box: {y_min}, {x_min}, {y_max}, {x_max}")
        print(f"Height Buffer: {height_buffer}, Width Buffer: {width_buffer}")

        y_min_expanded = max(0, y_min - height_buffer)
        y_max_expanded = min(image_shape.shape[0], y_max + height_buffer)
        x_min_expanded = max(0, x_min - width_buffer)
        x_max_expanded = min(image_shape.shape[1], x_max + width_buffer)

        print(f"Expanded Box: {y_min_expanded}, {x_min_expanded}, {y_max_expanded}, {x_max_expanded}")

        return y_min_expanded, x_min_expanded, y_max_expanded, x_max_expanded


    def process_frame(self, image_np):
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image_rgb, axis=0)
        detections = self.detect_fn(image_expanded)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        for i in range(num_detections):
            # time.sleep(.02)
            box = detections['detection_boxes'][i]
            class_id = detections['detection_classes'][i]
            score = detections['detection_scores'][i]

            if score >= .4 and self.category_index[class_id]["name"] == 'Person':
                y_min, x_min, y_max, x_max = box
                y_min = int(y_min * image_np.shape[0])
                x_min = int(x_min * image_np.shape[1])
                y_max = int(y_max * image_np.shape[0])
                x_max = int(x_max * image_np.shape[1])
                # print('Normal box : ',y_min, x_min, y_max, x_max )
                roi_person = image_np[y_min:y_max, x_min:x_max]
                cv2.imshow('Person Alone',roi_person)

                # box = [0, 0, 1, 1]  # Default box, assuming roi_person covers the entire bounding box
                buffer_percentage = .3  # You can adjust the buffer percentage as needed
                expanded_box = self.expand_box_with_buffer(box, buffer_percentage, image_np)
                
                # Extract the expanded region from the image
                y_min, x_min, y_max, x_max = expanded_box
                # y_min = int(y_min * image_np.shape[0])
                # x_min = int(x_min * image_np.shape[1])
                # y_max = int(y_max * image_np.shape[0])
                # x_max = int(x_max * image_np.shape[1])
                # print('Expanded box : ',y_min, x_min, y_max, x_max )

                roi_person_expanded = image_np[y_min:y_max, x_min:x_max]

                # Display the expanded region
                cv2.imshow('Person Alone (Expanded)', roi_person_expanded)
                # roi_person = image_np[y_min:y_max, x_min:x_max]

                # Additional processing for top and bottom halves
                # self.process_person_alone(roi_person)
                # self.process_top_half(roi_person)
                # self.process_bottom_half(roi_person)

    def process_person_alone(self, roi_person):
        cv2.imshow('Person Alone',cv2.resize(roi_person, (640, 240)))

    def process_top_half(self, roi_person):
        top_half = roi_person[:roi_person.shape[0]//3, :].copy()                   
        cv2.imshow('Top Half',  cv2.resize(top_half, (640, 240)))

    def process_bottom_half(self, roi_person):
        bottom_half = roi_person[roi_person.shape[0]//2:, :]
        cv2.imshow('Bottom Half', cv2.resize(bottom_half, (640, 240)))

    def process_video(self, output_path, input_path=0):
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            print("Unable to read video feed")
            return

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(5))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner_of_text = (0, 100)
        font_scale = 1
        font_color = (255, 255, 255)
        thickness = 1
        line_type = 2

        prev_frame_time = 0
        new_frame_time = 0

        while cap.isOpened():
            ret, image_np = cap.read()
            if not ret:
                break

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time) 
            prev_frame_time = new_frame_time

            self.process_frame(image_np)

            cv2.putText(image_np, f'FPS : {fps:.2f}', bottom_left_corner_of_text, font, font_scale, font_color, thickness, line_type)
            cv2.imshow('Frame', image_np)
            out.write(image_np)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # PROVIDE PATH TO MODEL DIRECTORY
    model_dir = '/home/cognitica-i7-13thgen/NPS/Tensorflow-model-inference/fine_tuned_model/content/fine_tuned_model/saved_model'

    # PROVIDE PATH TO LABEL MAP
    labels_path = '/home/cognitica-i7-13thgen/NPS/Tensorflow-model-inference/ppe_label_map.pbtxt'

    # Create an instance of the ObjectDetectionProcessor class
    processor = ObjectDetectionProcessor(model_dir, labels_path, min_conf_thresh=0.5)

    # Provide the output video path
    output_video_path = 'annotated_video.mp4'

    # Process the video and save the annotated version
    processor.process_video(output_video_path)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
version : 4
'''
model frame is looking good working fine it will have below setup to to seg the human from whole image and 
add some padding to give model inference
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow as tf
import cv2
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np


class ObjectDetectionProcessor:
    def __init__(self, model_dir, labels_path, min_conf_thresh=0.5):
        self.MIN_CONF_THRESH = min_conf_thresh
        self.category_index = label_map_util.create_category_index_from_labelmap(labels_path, use_display_name=True)
        self.detect_fn = tf.saved_model.load(model_dir)

    def expand_box_with_buffer(self, box, buffer_percentage, image_shape):
        y_min, x_min, y_max, x_max = box
        y_min = int(y_min * image_shape.shape[0])
        x_min = int(x_min * image_shape.shape[1])
        y_max = int(y_max * image_shape.shape[0])
        x_max = int(x_max * image_shape.shape[1])
        height_buffer = int((y_max - y_min) * buffer_percentage)
        width_buffer = int((x_max - x_min) * buffer_percentage)

        print(f"Original Box: {y_min}, {x_min}, {y_max}, {x_max}")
        print(f"Height Buffer: {height_buffer}, Width Buffer: {width_buffer}")

        y_min_expanded = max(0, y_min - height_buffer)
        y_max_expanded = min(image_shape.shape[0], y_max + height_buffer)
        x_min_expanded = max(0, x_min - width_buffer)
        x_max_expanded = min(image_shape.shape[1], x_max + width_buffer)

        print(f"Expanded Box: {y_min_expanded}, {x_min_expanded}, {y_max_expanded}, {x_max_expanded}")

        return y_min_expanded, x_min_expanded, y_max_expanded, x_max_expanded


    def process_frame(self, image_np):
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image_rgb, axis=0)
        detections = self.detect_fn(image_expanded)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        for i in range(num_detections):
            # time.sleep(.02)
            box = detections['detection_boxes'][i]
            class_id = detections['detection_classes'][i]
            score = detections['detection_scores'][i]

            if score >= .4 and self.category_index[class_id]["name"] == 'Person':
                y_min, x_min, y_max, x_max = box
                y_min = int(y_min * image_np.shape[0])
                x_min = int(x_min * image_np.shape[1])
                y_max = int(y_max * image_np.shape[0])
                x_max = int(x_max * image_np.shape[1])
                # print('Normal box : ',y_min, x_min, y_max, x_max )
                roi_person = image_np[y_min:y_max, x_min:x_max]
                cv2.imshow('Person Alone',roi_person)

                # box = [0, 0, 1, 1]  # Default box, assuming roi_person covers the entire bounding box
                buffer_percentage = .3  # You can adjust the buffer percentage as needed
                expanded_box = self.expand_box_with_buffer(box, buffer_percentage, image_np)
                
                # Extract the expanded region from the image
                y_min, x_min, y_max, x_max = expanded_box
                roi_person_expanded = image_np[y_min:y_max, x_min:x_max]

                # Display the expanded region
                cv2.imshow('Person Alone (Expanded)', roi_person_expanded)
                
                # Additional processing for top and bottom halves
                # self.process_person_alone(roi_person)
                self.process_top_half(roi_person_expanded)
                self.process_bottom_half(roi_person_expanded)

    def process_person_alone(self, roi_person):
        cv2.imshow('Person Alone',cv2.resize(roi_person, (640, 240)))

    def process_top_half(self, roi_person):
        top_half = roi_person[roi_person.shape[0]//6:roi_person.shape[0]//2, :].copy()                   
        cv2.imshow('Top Half',  cv2.resize(top_half, (640, 240)))

    def process_bottom_half(self, roi_person):
        bottom_half = roi_person[roi_person.shape[0]//2:, :]
        cv2.imshow('Bottom Half', cv2.resize(bottom_half, (640, 240)))

    def process_video(self, output_path, input_path=0):
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            print("Unable to read video feed")
            return

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(5))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner_of_text = (0, 100)
        font_scale = 1
        font_color = (255, 255, 255)
        thickness = 1
        line_type = 2

        prev_frame_time = 0
        new_frame_time = 0

        while cap.isOpened():
            ret, image_np = cap.read()
            if not ret:
                break

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time) 
            prev_frame_time = new_frame_time

            self.process_frame(image_np)

            cv2.putText(image_np, f'FPS : {fps:.2f}', bottom_left_corner_of_text, font, font_scale, font_color, thickness, line_type)
            cv2.imshow('Frame', image_np)
            out.write(image_np)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # PROVIDE PATH TO MODEL DIRECTORY
    model_dir = '/home/cognitica-i7-13thgen/NPS/Tensorflow-model-inference/fine_tuned_model/content/fine_tuned_model/saved_model'

    # PROVIDE PATH TO LABEL MAP
    labels_path = '/home/cognitica-i7-13thgen/NPS/Tensorflow-model-inference/ppe_label_map.pbtxt'

    # Create an instance of the ObjectDetectionProcessor class
    processor = ObjectDetectionProcessor(model_dir, labels_path, min_conf_thresh=0.5)

    # Provide the output video path
    output_video_path = 'annotated_video.mp4'

    # Process the video and save the annotated version
    processor.process_video(output_video_path)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------
# vers : 5 POC is working
'''
model input : whole image - > tophalf and detection bottom half detection
POC is working
'''


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow as tf
import cv2
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np


class ObjectDetectionProcessor:
    def __init__(self, model_dir, labels_path, min_conf_thresh=0.5):
        self.MIN_CONF_THRESH = min_conf_thresh
        self.category_index = label_map_util.create_category_index_from_labelmap(labels_path, use_display_name=True)
        self.detect_fn = tf.saved_model.load(model_dir)

    def expand_box_with_buffer(self, box, buffer_percentage, image_shape):
        y_min, x_min, y_max, x_max = box
        y_min = int(y_min * image_shape.shape[0])
        x_min = int(x_min * image_shape.shape[1])
        y_max = int(y_max * image_shape.shape[0])
        x_max = int(x_max * image_shape.shape[1])
        height_buffer = int((y_max - y_min) * buffer_percentage)
        width_buffer = int((x_max - x_min) * buffer_percentage)

        print(f"Original Box: {y_min}, {x_min}, {y_max}, {x_max}")
        print(f"Height Buffer: {height_buffer}, Width Buffer: {width_buffer}")

        y_min_expanded = max(0, y_min - height_buffer)
        y_max_expanded = min(image_shape.shape[0], y_max + height_buffer)
        x_min_expanded = max(0, x_min - width_buffer)
        x_max_expanded = min(image_shape.shape[1], x_max + width_buffer)

        print(f"Expanded Box: {y_min_expanded}, {x_min_expanded}, {y_max_expanded}, {x_max_expanded}")

        return y_min_expanded, x_min_expanded, y_max_expanded, x_max_expanded


    def process_frame(self, image_np):
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image_rgb, axis=0)
        detections = self.detect_fn(image_expanded)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        for i in range(num_detections):
            # time.sleep(.02)
            box = detections['detection_boxes'][i]
            class_id = detections['detection_classes'][i]
            score = detections['detection_scores'][i]

            if score >= .4 and self.category_index[class_id]["name"] == 'Person':
                y_min, x_min, y_max, x_max = box
                y_min = int(y_min * image_np.shape[0])
                x_min = int(x_min * image_np.shape[1])
                y_max = int(y_max * image_np.shape[0])
                x_max = int(x_max * image_np.shape[1])
                # roi_person = image_np[y_min:y_max, x_min:x_max]

                buffer_percentage = .3  # You can adjust the buffer percentage as needed
                expanded_box = self.expand_box_with_buffer(box, buffer_percentage, image_np)
                
                # Extract the expanded region from the image
                y_min, x_min, y_max, x_max = expanded_box
                roi_person_expanded = image_np[y_min:y_max, x_min:x_max]

                # Display the expanded region
                # cv2.imshow('Person Alone (Expanded)', roi_person_expanded)
                
                # Additional processing for top and bottom halves
                try : 
                    self.process_top_half(roi_person_expanded)
                    self.process_bottom_half(roi_person_expanded)
                except Exception as error:
                    print(error)

    def process_half_frame(self, image_np):
            # image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            image_expanded = np.expand_dims(image_np, axis=0)
            detections = self.detect_fn(image_expanded)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
            detections['num_detections'] = num_detections
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            for i in range(num_detections):
                box = detections['detection_boxes'][i]
                class_id = detections['detection_classes'][i]
                score = detections['detection_scores'][i]

                if score >= .2 and self.category_index[class_id]["name"] != 'Person':
                    y_min, x_min, y_max, x_max = box
                    y_min = int(y_min * image_np.shape[0])
                    x_min = int(x_min * image_np.shape[1])
                    y_max = int(y_max * image_np.shape[0])
                    x_max = int(x_max * image_np.shape[1])
                    # Draw bounding box and label on the image
                    cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    label = f'{self.category_index[class_id]["name"]} {int(score * 100)}%'
                    cv2.putText(image_np, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    # cv2.imshow('Top Half',  cv2.resize(image_np, (640, 240)))
                    return image_np
    
    def process_bottom_half_frame(self, image_np):
            # image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            image_expanded = np.expand_dims(image_np, axis=0)
            detections = self.detect_fn(image_expanded)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
            detections['num_detections'] = num_detections
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            for i in range(num_detections):
                box = detections['detection_boxes'][i]
                class_id = detections['detection_classes'][i]
                score = detections['detection_scores'][i]

                if score >= .2 and self.category_index[class_id]["name"] != 'Person':
                    y_min, x_min, y_max, x_max = box
                    y_min = int(y_min * image_np.shape[0])
                    x_min = int(x_min * image_np.shape[1])
                    y_max = int(y_max * image_np.shape[0])
                    x_max = int(x_max * image_np.shape[1])
                    # Draw bounding box and label on the image
                    cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    label = f'{self.category_index[class_id]["name"]} {int(score * 100)}%'
                    cv2.putText(image_np, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    # cv2.imshow('Top Half',  cv2.resize(image_np, (640, 240)))
                    return image_np

    def process_person_alone(self, roi_person):
        cv2.imshow('Person Alone',cv2.resize(roi_person, (640, 240)))

    def process_top_half(self, roi_person):
        top_half = roi_person[roi_person.shape[0]//6:roi_person.shape[0]//2, :].copy() 
        top_half = self.process_half_frame(top_half)                  
        cv2.imshow('Top Half',  cv2.resize(top_half, (640, 240)))

    def process_bottom_half(self, roi_person):
        bottom_half = roi_person[roi_person.shape[0]//2:, :]
        bottom_half = self.process_bottom_half_frame(bottom_half)                  
        cv2.imshow('Bottom Half', cv2.resize(bottom_half, (640, 240)))

    def process_video(self, output_path, input_path=0):
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            print("Unable to read video feed")
            return

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(5))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner_of_text = (0, 100)
        font_scale = 1
        font_color = (255, 255, 255)
        thickness = 1
        line_type = 2

        prev_frame_time = 0
        new_frame_time = 0

        while cap.isOpened():
            ret, image_np = cap.read()
            if not ret:
                break

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time) 
            prev_frame_time = new_frame_time

            self.process_frame(image_np)

            cv2.putText(image_np, f'FPS : {fps:.2f}', bottom_left_corner_of_text, font, font_scale, font_color, thickness, line_type)
            cv2.imshow('Frame', image_np)
            out.write(image_np)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # PROVIDE PATH TO MODEL DIRECTORY
    model_dir = '/home/cognitica-i7-13thgen/NPS/Tensorflow-model-inference/fine_tuned_model/content/fine_tuned_model/saved_model'

    # PROVIDE PATH TO LABEL MAP
    labels_path = '/home/cognitica-i7-13thgen/NPS/Tensorflow-model-inference/ppe_label_map.pbtxt'

    # Create an instance of the ObjectDetectionProcessor class
    processor = ObjectDetectionProcessor(model_dir, labels_path, min_conf_thresh=0.5)

    # Provide the output video path
    output_video_path = 'annotated_video.mp4'

    # Process the video and save the annotated version
    processor.process_video(output_video_path)
# ------------------------------------------------------------------------------------------------------------------------------------------------