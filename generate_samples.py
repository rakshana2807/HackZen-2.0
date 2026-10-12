"""Script to generate realistic test road-scene images and a short demonstration video.

Creates:
- samples/road_day_normal.jpg
- samples/road_night_dark.jpg
- samples/road_foggy_simulated.jpg
- samples/demo_road_clip.mp4
"""

import os
import cv2
import numpy as np
from vision.enhancement import apply_simulated_fog


def draw_road_scene(width: int = 640, height: int = 360, nighttime: bool = False) -> np.ndarray:
    """Generate a clean synthetic road scene with road, lane markings, horizon, and vehicle silhouette."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # 1. Sky & Horizon
    if nighttime:
        img[: int(height * 0.45)] = (15, 12, 20)  # Very dark night sky
    else:
        # Daylight gradient
        for y in range(int(height * 0.45)):
            r = int(180 - (y / height) * 80)
            g = int(210 - (y / height) * 60)
            b = int(240 - (y / height) * 40)
            img[y, :] = (b, g, r)

    # 2. Road surface (asphalt)
    road_color = (35, 35, 38) if nighttime else (65, 68, 72)
    pts = np.array([
        [int(width * 0.38), int(height * 0.45)],
        [int(width * 0.62), int(height * 0.45)],
        [int(width * 0.95), height],
        [int(width * 0.05), height]
    ], np.int32)
    cv2.fillPoly(img, [pts], road_color)

    # 3. Roadside terrain/grass
    grass_color = (20, 30, 20) if nighttime else (45, 100, 50)
    pts_left = np.array([
        [0, int(height * 0.45)],
        [int(width * 0.38), int(height * 0.45)],
        [int(width * 0.05), height],
        [0, height]
    ], np.int32)
    pts_right = np.array([
        [int(width * 0.62), int(height * 0.45)],
        [width, int(height * 0.45)],
        [width, height],
        [int(width * 0.95), height]
    ], np.int32)
    cv2.fillPoly(img, [pts_left], grass_color)
    cv2.fillPoly(img, [pts_right], grass_color)

    # 4. Lane markings
    lane_color = (120, 120, 120) if nighttime else (220, 220, 220)
    for y_step in range(int(height * 0.48), height, 25):
        scale = (y_step - height * 0.45) / (height * 0.55)
        lw = max(2, int(4 * scale))
        lh = max(6, int(15 * scale))
        x_center = int(width * 0.5)
        cv2.rectangle(img, (x_center - lw, y_step), (x_center + lw, min(height, y_step + lh)), lane_color, -1)

    # 5. Vehicle shape (lead car in middle lane)
    car_x = int(width * 0.45)
    car_y = int(height * 0.55)
    car_w = 90
    car_h = 55

    car_body_color = (40, 20, 80) if nighttime else (30, 30, 180)  # Red car
    cv2.rectangle(img, (car_x, car_y), (car_x + car_w, car_y + car_h), car_body_color, -1)
    # Windshield
    cv2.rectangle(img, (car_x + 10, car_y + 8), (car_x + car_w - 10, car_y + 25), (20, 20, 25), -1)
    # Taillights
    tail_color = (0, 0, 255) if nighttime else (30, 30, 200)
    cv2.circle(img, (car_x + 12, car_y + car_h - 12), 7, tail_color, -1)
    cv2.circle(img, (car_x + car_w - 12, car_y + car_h - 12), 7, tail_color, -1)
    # License plate
    cv2.rectangle(img, (car_x + 30, car_y + car_h - 14), (car_x + car_w - 30, car_y + car_h - 4), (200, 200, 200), -1)

    # 6. If night, add subtle ambient headlights glow
    if nighttime:
        overlay = img.copy()
        cv2.circle(overlay, (car_x + 12, car_y + car_h - 12), 22, (0, 0, 180), -1)
        cv2.circle(overlay, (car_x + car_w - 12, car_y + car_h - 12), 22, (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

    return img


def generate_all_samples():
    samples_dir = os.path.join(os.path.dirname(__file__), "samples")
    os.makedirs(samples_dir, exist_ok=True)

    # 1. Day Road
    day_img = draw_road_scene(640, 360, nighttime=False)
    cv2.imwrite(os.path.join(samples_dir, "road_day_normal.jpg"), day_img)

    # 2. Night Road
    night_img = draw_road_scene(640, 360, nighttime=True)
    cv2.imwrite(os.path.join(samples_dir, "road_night_dark.jpg"), night_img)

    # 3. Simulated Fog
    foggy_img = apply_simulated_fog(day_img, fog_density=0.75, fog_brightness=210)
    cv2.imwrite(os.path.join(samples_dir, "road_foggy_simulated.jpg"), foggy_img)

    # 4. Short video demo clip (30 frames, 10 fps)
    video_path = os.path.join(samples_dir, "demo_road_clip.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (640, 360))

    base = draw_road_scene(640, 360, nighttime=False)
    for f in range(30):
        frame = base.copy()
        # Add slight movement to car
        shift = int(f * 1.5)
        # Shift road markings
        cv2.putText(
            frame,
            f"FRAME {f+1:02d} / 30 [TEST CLIP]",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 240, 240),
            2,
            cv2.LINE_AA
        )
        out.write(frame)

    out.release()
    print("Successfully generated sample road scenes and demo video clip in samples/")


if __name__ == "__main__":
    generate_all_samples()
