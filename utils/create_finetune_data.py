import cv2
import dlib
import numpy as np
from collections import deque
import time
import os

CLASS_NAMES = [
    "ACCESS", "ACCUSED", "ACTUALLY", "AFTERNOON", "AGAIN", "AGREE", "ALREADY", "AMONG", "ASKING", "ATTACKS", 
    "BECOME", "BEING", "BETWEEN", "BRITISH", "BUILD", "CHANGE", "CHARGE", "CHIEF", "CHINA", "CLAIMS", 
    "CLEAR", "COUNCIL", "COUPLE", "DESCRIBED", "DIFFICULT", "EVERY", "EVERYTHING", "EXACTLY", "EXAMPLE", "EXPECTED", 
    "FAMILIES", "FAMILY", "FOCUS", "FOOTBALL", "FORCE", "GIVEN", "GREAT", "GROWTH", "HAPPENING", "HOMES", 
    "INSIDE", "INVESTMENT", "ISLAMIC", "JUSTICE", "LABOUR", "LARGE", "LEAST", "LEAVE", "MARKET", "MIDDLE", 
    "MILLIONS", "NUMBERS", "OPPOSITION", "ORDER", "PATIENTS", "PERIOD", "PLACE", "PLANS", "POLICY", "POLITICS", 
    "POSSIBLE", "POWER", "PRESIDENT", "PROBABLY", "PROBLEMS", "PROTECT", "QUESTION", "QUESTIONS", "REMEMBER", "RETURN", 
    "RIGHT", "RUNNING", "SCHOOL", "SCHOOLS", "SECRETARY", "SEEMS", "SERIOUS", "SERVICES", "SOCIETY", "SPECIAL", 
    "SPEECH", "SPENT", "STATE", "STILL", "SUPPORT", "TAKING", "TERMS", "THINK", "TOWARDS", "UNION", 
    "USING", "WAITING", "WANTS", "WEAPONS", "WELFARE", "WESTMINSTER", "WORKING", "WORST", "WOULD", "WRONG"
]

DLIB_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
FINETUNE_SAVE_DIR = "finetune_data" 

TARGET_RESOLUTION = (96, 96)
MOUTH_START_IDX = 48
MOUTH_END_IDX = 67

STATE_WAITING = 0
STATE_RECORDING = 1
STATE_COOLDOWN = 2 

TOTAL_FRAMES_FOR_MODEL = 29
FRAMES_TO_CAPTURE_PRE_TRIGGER = 9
FRAMES_TO_CAPTURE_POST_TRIGGER = 20

PRE_BUFFER_SIZE = 10 

SAVE_COOLDOWN_S = 2.0 
MOUTH_OPEN_THRESHOLD = 6.0 

def get_word_counts(save_dir, class_list):
    counts = {word: 0 for word in class_list}
    if not os.path.exists(save_dir):
        return counts
    
    for word in class_list:
        word_dir = os.path.join(save_dir, word)
        if os.path.isdir(word_dir):
            try:
                clips = [d for d in os.listdir(word_dir) if os.path.isdir(os.path.join(word_dir, d))]
                counts[word] = len(clips)
            except OSError:
                pass 
    return counts

def get_next_word(counts):
    min_count = min(counts.values())
    min_words = [word for word, count in counts.items() if count == min_count]
    return min_words[0]

def get_mouth_roi_from_landmarks(landmarks, frame_shape_hw):
    mouth_points = []
    for i in range(MOUTH_START_IDX, MOUTH_END_IDX + 1):
        mouth_points.append((landmarks.part(i).x, landmarks.part(i).y))
    
    if not mouth_points: return None
    mouth_points = np.array(mouth_points)
    
    x_min, y_min = np.min(mouth_points, axis=0)
    x_max, y_max = np.max(mouth_points, axis=0)
    
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    height = y_max - y_min
    width = x_max - x_min
    
    max_dim = max(height, width)
    padded_dim = int(max_dim * 1.60) 
    
    x1 = center_x - (padded_dim // 2)
    x2 = center_x + (padded_dim // 2)
    y1 = center_y - (padded_dim // 2)
    y2 = center_y + (padded_dim // 2)
    
    frame_h, frame_w = frame_shape_hw
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_w, x2)
    y2 = min(frame_h, y2)
    
    return (x1, y1, x2, y2)

def main():
    print("Loading dlib...")
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(DLIB_PREDICTOR_PATH)
    except RuntimeError as e:
        print(f"Error loading dlib predictor '{DLIB_PREDICTOR_PATH}': {e}")
        return
    print("Dlib loaded successfully.")

    print("Scanning existing data to balance recordings...")
    word_counts = get_word_counts(FINETUNE_SAVE_DIR, CLASS_NAMES)
    current_word = get_next_word(word_counts)
    print(f"Starting with least-recorded word: {current_word} (Count: {word_counts[current_word]})")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    current_state = STATE_WAITING
    pre_buffer = deque(maxlen=PRE_BUFFER_SIZE)
    recording_buffer = [] 
    frames_to_collect = 0
    last_save_time = 0
    lip_dist = 0.0
    status_text = "READY"
    
    while True:
        ret, frame = cap.read()
        if not ret: break
            
        frame = cv2.flip(frame, 1)
        frame_copy = frame.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = detector(frame_gray, 0)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        if len(faces) > 0:
            face = faces[0]
            landmarks = predictor(frame_gray, face)
            roi = get_mouth_roi_from_landmarks(landmarks, frame.shape[:2])
            
            top_lip = landmarks.part(62)
            bottom_lip = landmarks.part(66)
            lip_dist = bottom_lip.y - top_lip.y
            
            if roi:
                x1, y1, x2, y2 = roi
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                cropped_lip = frame[y1:y2, x1:x2]
                resized_lip = cv2.resize(cropped_lip, TARGET_RESOLUTION, interpolation=cv2.INTER_AREA)
                
                if current_state == STATE_WAITING:
                    status_text = "READY"
                    pre_buffer.append(resized_lip) 
                    
                    if lip_dist > MOUTH_OPEN_THRESHOLD:
                        if len(pre_buffer) < FRAMES_TO_CAPTURE_PRE_TRIGGER:
                            print(f"Triggered, but pre-buffer only has {len(pre_buffer)} frames. Waiting...")
                            status_text = "Wait..."
                        else:
                            print(f"TRIGGERED! Starting recording for: {current_word}")
                            current_state = STATE_RECORDING
                            recording_buffer = list(pre_buffer)[-FRAMES_TO_CAPTURE_PRE_TRIGGER:] 
                            frames_to_collect = FRAMES_TO_CAPTURE_POST_TRIGGER
                            status_text = "REC..."
                    
                elif current_state == STATE_RECORDING:
                    recording_buffer.append(resized_lip)
                    frames_to_collect -= 1
                    status_text = f"REC...{frames_to_collect}"
                    
                    if frames_to_collect == 0:
                        print("...Recording complete. Saving clip.")
                        
                        if len(recording_buffer) == TOTAL_FRAMES_FOR_MODEL:
                            word_dir = os.path.join(FINETUNE_SAVE_DIR, current_word)
                            os.makedirs(word_dir, exist_ok=True)
                            
                            clip_num = 1
                            while os.path.exists(os.path.join(word_dir, f"{clip_num:03d}")):
                                clip_num += 1
                            
                            clip_dir = os.path.join(word_dir, f"{clip_num:03d}")
                            os.makedirs(clip_dir)
                            
                            for i, img_frame in enumerate(recording_buffer):
                                save_path = os.path.join(clip_dir, f"frame_{i:02d}.png")
                                cv2.imwrite(save_path, img_frame)
                            
                            print(f"Saved clip to {clip_dir}")
                            status_text = f"SAVED! ({clip_num:03d})"
                            
                            word_counts[current_word] += 1
                            current_word = get_next_word(word_counts)
                            print(f"New word to record: {current_word} (Count: {word_counts[current_word]})")
                        else:
                            print(f"Error: Final clip has {len(recording_buffer)} frames, expected {TOTAL_FRAMES_FOR_MODEL}.")
                            status_text = "Record Error"

                        recording_buffer.clear()
                        current_state = STATE_COOLDOWN
                        last_save_time = time.time()
                        
                elif current_state == STATE_COOLDOWN:
                    if time.time() - last_save_time > SAVE_COOLDOWN_S:
                        current_state = STATE_WAITING
                        pre_buffer.clear()
                        
            else:
                if current_state == STATE_WAITING: pre_buffer.clear()
                
        else:
            if current_state == STATE_WAITING: pre_buffer.clear()
            status_text = "No face"


        cv2.rectangle(frame_copy, (0, 0), (frame_copy.shape[1], 120), (0, 0, 0), -1)
        
        cv2.putText(
            frame_copy, 
            f"Word: {current_word} (Count: {word_counts[current_word]})", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, (255, 255, 255), 2
        )
        
        state_text = ["WAITING", "RECORDING", "COOLDOWN"][current_state]
        cv2.putText(
            frame_copy, 
            f"Status: {status_text} | Lip Dist: {lip_dist:.1f} (Trig: {MOUTH_OPEN_THRESHOLD})",
            (10, 65), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, (0, 255, 255), 1
        )
        
        cv2.putText(
            frame_copy, 
            "Speak the word to auto-record | Q=Quit", 
            (10, 100), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, (255, 255, 0), 1
        )
        
        cv2.imshow("Finetune Data Collector (Press 'q' to quit)", frame_copy)
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    os.makedirs(FINETUNE_SAVE_DIR, exist_ok=True)
    main()