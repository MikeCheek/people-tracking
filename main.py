import cv2
import db
import numpy as np
from core.camera import WebcamStream
from core.face import FaceEngine
from collections import deque, Counter
from deepface import DeepFace

# --- CONFIG ---
SMOOTHING_WINDOW = 20  # Number of frames to remember for smoothing

# This dictionary will store: {person_id: {'age': deque, 'gender': deque}}
history = {}
def get_smoothed_attributes(person_id, raw_age, raw_gender, raw_emotion):
    if person_id not in history:
        history[person_id] = {
            'age': deque(maxlen=SMOOTHING_WINDOW),
            'gender': deque(maxlen=SMOOTHING_WINDOW),
            'emotion': deque(maxlen=SMOOTHING_WINDOW)
        }
    
    history[person_id]['age'].append(raw_age)
    history[person_id]['gender'].append(raw_gender)
    if raw_emotion:
        history[person_id]['emotion'].append(raw_emotion)
    
    smooth_age = int(np.mean(history[person_id]['age']))
    smooth_gender = Counter(history[person_id]['gender']).most_common(1)[0][0]
    
    # Get the most frequent emotion in the window
    if history[person_id]['emotion']:
        smooth_emo = Counter(history[person_id]['emotion']).most_common(1)[0][0]
    else:
        smooth_emo = "Neutral"
        
    return smooth_age, smooth_gender, smooth_emo

def main():
    db.init_db()
    engine = FaceEngine(model_name='buffalo_s')
    video_stream = WebcamStream(src=0).start()
    
    known_faces_cache = db.get_known_faces()
    engine.update_search_index(known_faces_cache)
    
    frame_count = 0
    
    while True:
        frame = video_stream.read()
        if frame is None: continue
        frame_count += 1
        if frame_count % 5 != 0:
            continue
        
        # Scaling for performance
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        faces = engine.get_face_features(small_frame)

        for face in faces:
            bbox = (face['bbox'] * 2).astype(int)
            new_vec = face['embedding']
            
            current_emotion = None
            # 1. Identity Search
            person_id, confidence = engine.search_face(new_vec)
            
            try:
                analysis = DeepFace.analyze(img_path=small_frame, actions=['emotion'], enforce_detection=False)
                current_emotion = analysis[0]['dominant_emotion']
            except Exception as e:
                print("DeepFace emotion analysis error:", e)
                current_emotion = "Neutral"
            
            if person_id:
                name = db.get_person_name(person_id)
            else:
                person_id = db.create_new_person(new_vec)
                name = "New User"
                
                x1, y1, x2, y2 = np.clip(bbox, 0, [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                face_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                img_path = f"captures/person_{person_id}.jpg"
                cv2.imwrite(img_path, face_crop)
                db.update_thumbnail_path(person_id, img_path)
                
                # 3. REFRESH CACHE: A new face exists, update the search index
                known_faces_cache = db.get_known_faces()
                engine.update_search_index(db.get_known_faces())

            # 2. SMOOTHING LOGIC (The Fix)
            age, gender, emotion = get_smoothed_attributes(person_id, face['age'], face['gender'], current_emotion)
            
            # 3. UI Styling
            color = (255, 100, 200) if gender == 0 else (255, 200, 0) # Pink vs Cyan
            
            emo_colors = {
                "happy": (0, 255, 255),    # Yellow
                "sad": (255, 0, 0),       # Blue
                "angry": (0, 0, 255),      # Red
                "surprise": (0, 165, 255), # Orange
                "neutral": (255, 255, 255) # White
            }
            color = emo_colors.get(emotion.lower(), (255, 200, 0))
            
            # Map Age to Stage
            if age < 13: stage = "Child"
            elif age < 20: stage = "Teen"
            elif age < 30: stage = "Young Adult"
            elif age < 60: stage = "Adult"
            else: stage = "Senior"

            # 4. Draw Smooth UI
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Overlay Header
            overlay = frame.copy()
            cv2.rectangle(overlay, (bbox[0], bbox[1] - 60), (bbox[2], bbox[1]), color, -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            
            cv2.putText(frame, f"{name} | {stage}", (bbox[0] + 5, bbox[1] - 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Age: ~{age} | {'Male' if gender == 1 else 'Female'}", (bbox[0] + 5, bbox[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Mood: {emotion.upper()}", (bbox[0], bbox[3] + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        cv2.imshow('Smooth Face Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()