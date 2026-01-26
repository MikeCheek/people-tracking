import cv2
from db import create_new_person, get_known_faces, get_person_name, update_thumbnail_path
from face import FaceEngine

if __name__ == "__main__":
  engine = FaceEngine(model_name='buffalo_s')
  video_capture = cv2.VideoCapture(0)

  print("Starting InsightFace System...")
  
  frame_count = 0

  while True:
      ret, frame = video_capture.read()
      if not ret: break
      
      if frame_count % 3 == 0:
        faces = engine.get_face_features(frame)
    
      frame_count += 1
      
      known_faces = get_known_faces()

      for face in faces:
          bbox = face['bbox'].astype(int)
          new_vec = face['embedding']
          
          person_id = None
          highest_sim = 0
          threshold = 0.45  # 0.4 - 0.6 is the sweet spot for ArcFace
          name = "Unknown"

          for pid, db_vec in known_faces:
              sim = engine.compute_similarity(new_vec, db_vec)
              if sim > threshold and sim > highest_sim:
                  highest_sim = sim
                  person_id = pid
          
          if person_id:
              name = get_person_name(person_id)
          else:
              person_id = create_new_person(new_vec)
              
              x1, y1, x2, y2 = bbox
              y1, y2 = max(0, y1), min(frame.shape[0], y2)
              x1, x2 = max(0, x1), min(frame.shape[1], x2)
              face_crop = frame[y1:y2, x1:x2]
              
              img_path = f"captures/person_{person_id}.jpg"
              cv2.imwrite(img_path, face_crop)
              
              update_thumbnail_path(person_id, img_path)

          cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
          label = f"{name} {person_id} ({highest_sim:.2f})" if person_id else name
          cv2.putText(frame, label, (bbox[0], bbox[1] - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

      cv2.imshow('InsightFace System', frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  video_capture.release()
  cv2.destroyAllWindows()