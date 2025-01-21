import argparse
import cv2
import os


def compress_videos(input_videos_path, output_videos_path, frame_rate):

    os.makedirs(output_videos_path, exist_ok=True)

    videos = [f for f in os.listdir(input_videos_path) if f.endswith((".mp4", ".mkv"))]
    for video in videos:
        input_video_path = os.path.join(input_videos_path, video)
        output_video_path = os.path.join(output_videos_path, video)

        cap = cv2.VideoCapture(input_video_path)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

        while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

        cap.release()
        out.release()

        print(f"Video saved as {output_video_path}")
        
    print("All videos processed.")


def compress_sequences(input_path, output_path, frame_rate):

    os.makedirs(output_path, exist_ok=True)

    image_sequences = [seq for seq in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, seq))]

    if len(image_sequences) == 0:
        print("No image sequences found in the specified directory.")
        return

    for seq in image_sequences:
        seq_folder = os.path.join(input_path, seq)
        output_video = os.path.join(output_path, f"{seq}.mp4")

        

        images = sorted([img for img in os.listdir(seq_folder) if img.endswith(('.jpg', '.png', '.tiff'))])
        if not images:
            print("No TIFF images found in the specified directory.")
            return

        first_image = cv2.imread(os.path.join(seq_folder, images[0]))
        height, width = first_image.shape[:2]

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Codec for MP4
        video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

        for image_name in images:
            image_path = os.path.join(seq_folder, image_name)
            frame = cv2.imread(image_path)
            video_writer.write(frame)

        video_writer.release()
        print(f"Video saved as {output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the denoiser")

    parser.add_argument("--input_path", "-i", type=str, help="Image video path")
    parser.add_argument("--output_path", "-o", type=str, help="Output directory or video path")
    parser.add_argument("--frame_rate", "-f", type=int, default=25, help="Frame rate of the output video")
    parser.add_argument("--image_sequences", "-im", action="store_true", help="Use image sequences as inputs")

    args = parser.parse_args()

    if args.image_sequences:
        compress_sequences(args.input_path, args.output_path, args.frame_rate)
    else:
        compress_videos(args.input_path, args.output_path, args.frame_rate)