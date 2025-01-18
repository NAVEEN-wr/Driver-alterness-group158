#Enter this command in terminal and run 
#pip install mediapipe opencv-python
#pip install playsound

import cv2
import mediapipe as mp
import time
import numpy as np
import winsound  


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)


EYE_CLOSED_THRESHOLD = 0.3  
EYE_CLOSED_DURATION = 10  


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def calculate_eye_aspect_ratio(landmarks, eye_points, width, height):
    """Calculate the Eye Aspect Ratio (EAR)."""
    eye = np.array([(landmarks[i].x * width, landmarks[i].y * height) for i in eye_points])
    hor_line = np.linalg.norm(eye[0] - eye[3])
    ver_line_1 = np.linalg.norm(eye[1] - eye[5])
    ver_line_2 = np.linalg.norm(eye[2] - eye[4])
    return (ver_line_1 + ver_line_2) / (2 * hor_line)

# Initialize variables
cap = cv2.VideoCaptur
closed_eyes_start_time = None  
drowsy_alert = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    
    mask_feed = frame.copy()

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * width), int(landmark.y * height)
                cv2.circle(mask_feed, (x, y), 1, (0, 255, 0), -1)
            
            
            left_ear = calculate_eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE, width, height)
            right_ear = calculate_eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE, width, height)
            avg_ear = (left_ear + right_ear) / 2
            
            
            if avg_ear < EYE_CLOSED_THRESHOLD:
                if closed_eyes_start_time is None:
                    closed_eyes_start_time = time.time()
                elif time.time() - closed_eyes_start_time >= EYE_CLOSED_DURATION:
                    drowsy_alert = True
            else:
                closed_eyes_start_time = None
                drowsy_alert = False

    
    if drowsy_alert:
        cv2.putText(frame, "ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.putText(mask_feed, "ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        winsound.Beep(1000, 500)  

    
    cv2.imshow("Masked Feed", mask_feed)
    cv2.imshow("Driver Monitor", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
