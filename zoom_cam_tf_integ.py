import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
import tensorflow as tf
import cv2
import time
from object_detection.utils import label_map_util
import numpy as np
from threading import Thread

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

        y_min_expanded = max(0, y_min - height_buffer)
        y_max_expanded = min(image_shape.shape[0], y_max + height_buffer)
        x_min_expanded = max(0, x_min - width_buffer)
        x_max_expanded = min(image_shape.shape[1], x_max + width_buffer)

        return y_min_expanded, x_min_expanded, y_max_expanded, x_max_expanded

    def process_detection(self, frame, expanded_boxes):
        for expanded_box in expanded_boxes:
            y_min, x_min, y_max, x_max = expanded_box
            roi_person_expanded = frame[y_min:y_max, x_min:x_max].copy()
            
            # Additional processing for top and bottom halves
            # Thread(target=self.process_top_half, args=(roi_person_expanded ,)).start()
            Thread(target=self.process_bottom_half, args=(roi_person_expanded ,)).start()

    def process_bottom_half(self, roi_person):
        print('process_bottom_half')
        bottom_half = roi_person[roi_person.shape[0]//2:, :].copy() 
        bottom_half = self.process_bottom_half_frame(bottom_half)                  
        cv2.imshow('Bottom Half', bottom_half)

    def process_bottom_half_frame(self, image_np):
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

    def process_top_half(self, roi_person):
        top_half = roi_person[roi_person.shape[0]//6:roi_person.shape[0]//2, :].copy() 
        top_half = self.process_half_frame(top_half)                  
        cv2.imshow('Top Half',  cv2.resize(top_half, (640, 240))) 

    def process_half_frame(self, image_np):
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

    def process_frame(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image_rgb, axis=0)
        detections = self.detect_fn(image_expanded)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        expanded_boxes = []
        for i in range(num_detections):
            box = detections['detection_boxes'][i]
            class_id = detections['detection_classes'][i]
            score = detections['detection_scores'][i]

            if score >= self.MIN_CONF_THRESH and self.category_index[class_id]["name"] == 'Person':
                expanded_box = self.expand_box_with_buffer(box, 0.3, frame)

                expanded_boxes.append(expanded_box)

        if expanded_boxes:
            # print('expanded_boxes : ',expanded_boxes)
            # try:
            Thread(target=self.process_detection, args=(frame, expanded_boxes)).start()
            # self.process_detection(frame, expanded_boxes)
            # except Exception as Error:
                # print(Error)

        return frame

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
            ret, frame = cap.read()
            if not ret:
                break

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time) 
            prev_frame_time = new_frame_time

            processed_frame = self.process_frame(frame)

            cv2.putText(processed_frame, f'FPS : {fps:.2f}', bottom_left_corner_of_text, font, font_scale, font_color, thickness, line_type)
            cv2.imshow('Frame', processed_frame)
            out.write(processed_frame)

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
