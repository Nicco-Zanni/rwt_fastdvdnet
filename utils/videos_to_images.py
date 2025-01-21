import os
import cv2

# Path to the directory containing MP4 videos
input_dir = "data/DAVIS/DAVIS_train/val/soft/mp4_td_avc"
output_dir = "data/DAVIS/DAVIS_train/val/soft/JPEG_td_avc"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get all MP4 files in the input directory
video_files = [f for f in os.listdir(input_dir) if f.endswith((".mp4", ".mkv"))]

# Process each video file
for video_file in video_files:
    video_path = os.path.join(input_dir, video_file)
    video_name = os.path.splitext(video_file)[0]  # Get the video name without the extension
    
    # Create a specific directory for the video's frames
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_file}")
        continue
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:  # Break when the video ends
            break
        
        # Save the frame as an image
        frame_filename = os.path.join(video_output_dir, f"{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        frame_count += 1
    
    cap.release()
    print(f"Processed {frame_count} frames for video: {video_file}")

print("All videos processed.")
