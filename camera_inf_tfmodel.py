import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow as tf
import cv2
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np

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


    while cap.isOpened():
        ret, image_np = cap.read()
        if not ret:
            break
        

        new_frame_time = time.time()
        # i +=1

        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time 
        # Convert to RGB (if needed) and expand dimensions to create a batch of size 1
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image_rgb, axis=0)

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

                # Draw bounding box and label on the image
                cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                label = f'{category_index[class_id]["name"]} {int(score * 100)}%'
                cv2.putText(image_np, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


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