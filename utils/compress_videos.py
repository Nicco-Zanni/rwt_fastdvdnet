
import cv2

def compress_videos(input_videos_path, output_videos_path, frame_rate):

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the denoiser")

    parser.add_argument("--input_path", "-i", type=str, help="Image video path")
    parser.add_argument("--output_path", "-o", type=str, help="Output directory or video path")
    parser.add_argument("--frame_rate", "-f", type=int, default=25, help="Frame rate of the output video")

    args = parser.parse_args()

    compress_videos(args.input_path, args.output_path, args.frame_rate)