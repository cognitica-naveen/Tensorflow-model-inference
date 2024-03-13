import cv2


cap = cv2.VideoCapture(3)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# fps = 30
fps = int(cap.get(5))

nrml = cv2.VideoWriter('/home/cognitica-i7-13thgen/NPS/Tensorflow-model-inference/cam_output/hd.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    cv2.imshow('frame',frame)
    nrml.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

nrml.release()
cap.release()
cv2.destroyAllWindows()
print('working')


# import cv2

# frame0 = cv2.VideoCapture(1)
# frame1 = cv2.VideoCapture(3)

# frame_width = int(frame0.get(3))
# frame_height = int(frame0.get(4))
# fps = 30

# frame_width1 = int(frame1.get(3))
# frame_height1 = int(frame1.get(4))

# forklift_cam = cv2.VideoWriter('/home/cognitica-i7-13thgen/NPS/Tensorflow-model-inference/cam_output/forklift_cam.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
# step_camera = cv2.VideoWriter('/home/cognitica-i7-13thgen/NPS/Tensorflow-model-inference/cam_output/step_camera.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width1, frame_height1))

# while 1:

#    ret0, img0 = frame0.read()
#    ret1, img00 = frame1.read()
#    img00 = cv2.rotate(img00, cv2.ROTATE_180)

   
#    if (frame0):
#        cv2.imshow('forklift_cam',img0)
#    if (frame1):
#        cv2.imshow('step_camera',img00)

#    forklift_cam.write(img00)
#    step_camera.write(img0)

#    if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# frame0.release()
# forklift_cam.release()
# frame1.release()
# step_camera.release()
# cv2.destroyAllWindows()