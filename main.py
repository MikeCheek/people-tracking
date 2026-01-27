import cv2
from core.ui import draw_cyberpunk_hud, draw_dense_mesh
import db
import numpy as np
from core.camera import WebcamStream
from core.face import FaceEngine
from collections import deque, Counter
from deepface import DeepFace

# --- CONFIG ---
SMOOTHING_WINDOW = 10  # Number of frames to remember for smoothing

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
    approved_ids = []

    while True:
        frame = video_stream.read()
        if frame is None: continue
        
        output_frame = frame.copy() # We work on a copy to keep the original clean
        
        # 1. Update Permissions from DB
        if frame_count % 30 == 0:
            approved_ids = db.get_approved_ids()
            privacy_active = db.get_setting("enable_privacy_cloak") == "True"
            hud_active = db.get_setting("enable_hud") == "True"
            show_landmarks = db.get_setting("show_landmarks") == "True"

        frame_count += 1
        # 2. Get Faces
        faces = engine.get_face_features(output_frame)

        for face in faces:
            bbox = face['bbox'].astype(int)
            x1, y1, x2, y2 = np.clip(bbox, 0, [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            
            person_id, confidence = engine.search_face(face['embedding'])
            
            # --- IDENTITY & EMOTION ---
            current_emotion = "Neutral"
            if frame_count % 10 == 0:
                try:
                    face_crop = frame[y1:y2, x1:x2]
                    analysis = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False, silent=True)
                    current_emotion = analysis[0]['dominant_emotion']
                except: pass

            if not person_id:
                person_id = db.create_new_person(face['embedding'])
                cv2.imwrite(f"captures/person_{person_id}.jpg", frame[y1:y2, x1:x2])
                engine.update_search_index(db.get_known_faces())
                db.update_thumbnail_path(person_id, f"captures/person_{person_id}.jpg")
            
            name = db.get_person_name(person_id)
            age, gender, emotion = get_smoothed_attributes(person_id, face['age'], face['gender'], current_emotion)

            # --- PRIVACY LOGIC: LOCALIZED BLUR ---
            if person_id in approved_ids or not privacy_active:
                # AUTHORIZED: Draw the Cyberpunk HUD
                emo_colors = {"happy": (0, 255, 255), "sad": (255, 0, 0), "angry": (0, 0, 255), "surprise": (0, 165, 255), "neutral": (255, 255, 255)}
                color = emo_colors.get(emotion.lower(), (255, 200, 0))
                if hud_active:
                    draw_cyberpunk_hud(output_frame, face, name, age, emotion, color)
                if show_landmarks:
                    draw_dense_mesh(output_frame, face, color, alpha=0.5)
            else:
                # UNAUTHORIZED: Blur ONLY the face region
                face_roi = output_frame[y1:y2, x1:x2]
                # Apply a heavy blur to the localized crop
                blurred_face = cv2.GaussianBlur(face_roi, (51, 51), 30)
                # Put it back into the main frame
                output_frame[y1:y2, x1:x2] = blurred_face
                
                # Visual indicator that it's blocked
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.putText(output_frame, "UNAUTHORIZED", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        cv2.imshow('Selective Privacy Shield', output_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_stream.stop()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()