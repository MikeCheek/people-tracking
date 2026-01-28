import time
import cv2
import numpy as np

def draw_skeleton(frame, kps, color):
    # 1. Create a transparent overlay (same size as frame)
    overlay = frame.copy()
    
    # Define a soft alpha (0.0 = invisible, 1.0 = solid)
    # We'll use a very light touch for that "less invasive" feel
    alpha = 0.3 

    # 2. Define the refined connections
    # Using eye-to-eye and feature-to-nose paths
    connections = [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4)]
    
    # 3. Draw the thin "Neural" lines on the overlay
    for p1, p2 in connections:
        pt1, pt2 = tuple(kps[p1].astype(int)), tuple(kps[p2].astype(int))
        
        # Subtle glow (slightly thicker, drawn first)
        cv2.line(overlay, pt1, pt2, color, 2, cv2.LINE_AA)
        # Sharp data line (thin 1px)
        cv2.line(overlay, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA)

    # 4. Draw the Minimalist Nodes
    for i, kp in enumerate(kps):
        pt = tuple(kp.astype(int))
        
        # Micro-circle for the landmark
        cv2.circle(overlay, pt, 2, color, -1, cv2.LINE_AA)
        
        # Add a tiny "bracket" look to the eyes only
        if i in [0, 1]:
            d = 4
            cv2.line(overlay, (pt[0]-d, pt[1]-d), (pt[0]-d, pt[1]+d), color, 1, cv2.LINE_AA)
            cv2.line(overlay, (pt[0]+d, pt[1]-d), (pt[0]+d, pt[1]+d), color, 1, cv2.LINE_AA)

    # 5. Blend the overlay back into the original frame
    # frame = (1 - alpha) * frame + alpha * overlay
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
def draw_dense_mesh(frame, face, color, alpha=0.2):
    landmarks = face['landmark_2d_106'].astype(int)
    overlay = frame.copy()

    # Draw small dots for all 106 points
    for pt in landmarks:
        cv2.circle(overlay, tuple(pt), 1, color, -1, cv2.LINE_AA)
    
    # Draw a line specifically for the jawline (points 0 to 32 usually)
    jaw_pts = landmarks[0:33]
    cv2.polylines(overlay, [jaw_pts], False, color, 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_cyberpunk_hud(frame, face, name, age, gender, emotion, color, resize=1.0):
    bbox = (face['bbox'] * resize).astype(int)
    kps = (face['kps'] * resize).astype(int) # 5 Keypoints
    
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    center_x, center_y = x1 + w // 2, y1 + h // 2
    
    # --- 1. HUD CORNER BRACKETS (Instead of a full box) ---
    length = int(w * 0.2)
    thickness = 2
    # Top Left
    cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)
    # Top Right
    cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)
    # Bottom Left
    cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)
    # Bottom Right
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)

    # --- 2. THE SCANNING CIRCLE (Animated) ---
    # Rotates based on system time
    angle = int(time.time() * 100) % 360
    radius = int(max(w, h) * 0.6)
    cv2.ellipse(frame, (center_x, center_y), (radius, radius), 0, angle, angle + 90, color, 1)
    cv2.ellipse(frame, (center_x, center_y), (radius, radius), 0, angle + 180, angle + 270, color, 1)

    # --- 3. LANDMARK CONNECTORS (The "Digital Skeleton") ---
    # Connect eyes to nose to mouth
    # draw_skeleton(frame, kps, color)

    # --- 4. THE DATA BLOCK (Glassmorphism Sidebar) ---
    sidebar_x = x2 + 10
    # Draw vertical line from the box to the data
    cv2.line(frame, (x2, y1 + 20), (sidebar_x, y1 + 20), color, 1)
    
    # Background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (sidebar_x, y1), (sidebar_x + 150, y1 + 80), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"ID: {name.upper()}", (sidebar_x + 5, y1 + 10), font, 0.4, color, 1)
    cv2.putText(frame, f"AGE: {age}", (sidebar_x + 5, y1 + 30), font, 0.4, color, 1)
    cv2.putText(frame, f"MOOD: {emotion.upper()}", (sidebar_x + 5, y1 + 50), font, 0.4, color, 1)
    cv2.putText(frame, f"GENDER: {gender.upper()}", (sidebar_x + 5, y1 + 70), font, 0.4, color, 1)

    # --- 5. HEAD POSE INDICATOR (Simple Yaw) ---
    # If the nose is further from the left eye than the right, they are looking right
    dist_l = np.linalg.norm(kps[2] - kps[0])
    dist_r = np.linalg.norm(kps[2] - kps[1])
    yaw_ratio = dist_l / (dist_r + 1e-6)
    
    bar_w = 40
    cv2.rectangle(frame, (center_x - bar_w//2, y2 + 10), (center_x + bar_w//2, y2 + 15), (50,50,50), -1)
    indicator_pos = int((yaw_ratio - 1) * 20) # Simple offset
    cv2.circle(frame, (center_x + indicator_pos, y2 + 12), 3, color, -1)