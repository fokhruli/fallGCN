import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import requests
import os

def download_model():
    """Download the pose landmarker model if it doesn't exist."""
    model_path = 'pose_landmarker.task'
    if not os.path.exists(model_path):
        print("Downloading pose landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
        response = requests.get(url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
        print("Model downloaded successfully!")
    return model_path

def process_video(video_path, output_file):
    """Process a single video file and extract pose landmarks."""
    
    # Download model and create detector
    model_path = download_model()
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True,
        running_mode=vision.RunningMode.VIDEO)
    detector = vision.PoseLandmarker.create_from_options(options)

    try:
        # Open the video file
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)
        print(f"fps of video: {fps}")
        frame_interval_ms = int(1000.0 / fps)

        # Clear/create the output file
        with open(output_file, "w") as file:
            file.write(f"Video: {os.path.basename(video_path)}\nFPS: {fps}\n\n")

        # Process frames
        frame_count = 0
        timestamp_ms = 0

        while True:
            ret, frame = video.read()
            if not ret:
                break

            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Detect landmarks
            detection_result = detector.detect_for_video(mp_image, timestamp_ms)

            # Save landmarks if detected
            if detection_result.pose_landmarks:
                with open(output_file, "a") as file:
                    file.write(f"Frame {frame_count} (timestamp: {timestamp_ms}ms):\n")
                    for idx, landmark in enumerate(detection_result.pose_landmarks[0]):
                        file.write(f"Landmark {idx}: x={landmark.x:.6f}, y={landmark.y:.6f}, z={landmark.z:.6f}\n")
                    file.write("\n")

            frame_count += 1
            timestamp_ms += frame_interval_ms

        print(f"Processed {frame_count} frames")

    except Exception as e:
        print(f"Error processing video: {str(e)}")
    finally:
        if 'video' in locals():
            video.release()
        if 'detector' in locals():
            detector.close()

# Example usage
if __name__ == "__main__":
    video_path = "dataset/chute01/cam1.avi"  # Replace with your video path
    output_file = "chute01_cam1.txt"          # Replace with desired output path
    process_video(video_path, output_file)