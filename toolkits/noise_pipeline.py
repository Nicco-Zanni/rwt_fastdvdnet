import cv2
import numpy as np
import os
import argparse
from random import randint

def add_gaussian_noise(image, mean=0, sigma=25, random=False):
    """
    Add Gaussian noise to an image.
    
    Parameters:
        image (numpy.ndarray): The input image.
        mean (float): Mean of the Gaussian noise.
        sigma (float): Standard deviation of the Gaussian noise.
        
    Returns:
        numpy.ndarray: The noisy image.
    """
    if random:
        mean = np.random.uniform(0, 255)
        sigma = np.random.uniform(0, 50)
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), noise)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_impulsive_noise(image, prob=0.02):
    """
    Add impulsive noise to an image.
    
    Parameters:
        image (numpy.ndarray): The input image.
        prob (float): Probability of adding noise.
        
    Returns:
        numpy.ndarray: The noisy image.
    """
    noisy_image = np.copy(image)
    height, width, channels = image.shape

    total_pixels = height * width
    num_salt = int(total_pixels * prob / 2)
    num_pepper = int(total_pixels * prob / 2)

    for _ in range(num_salt):
        y, x = np.random.randint(0, height-1), np.random.randint(0, width-1)
        noisy_image[y, x] = [255, 255, 255]

    for _ in range(num_pepper):
        y, x = np.random.randint(0, height-1), np.random.randint(0, width-1)
        noisy_image[y, x] = [0, 0, 0]

    return noisy_image

def add_poisson_noise(image, gain=1):
    """
    Add Poisson noise to an image.
    
    Parameters:
        image (numpy.ndarray): Input image (uint8 format, range 0-255).
    
    Returns:
        numpy.ndarray: Noisy image (uint8 format, range 0-255).
    """
    # Ensure the input image is in float32 format and scale it to [0, 1]
    image_float = image.astype(np.float32) / 255.0
    # Apply Poisson noise
    noisy_image = np.random.poisson(image_float * gain * 255) / (255.0 * gain)
    # Clip the values and convert back to uint8
    noisy_image = np.clip(noisy_image * 255, 0, 255)
    return noisy_image.astype(np.uint8)
    

def add_noise_image_sequence(input_folder, output_folder, noise_type='gaussian', random=False, soft=False):
    images = sorted([img for img in os.listdir(input_folder) if img.endswith(('.jpg', '.png', '.tiff'))])
    if not images:
        print("No images found in the specified directory.")
        return

    os.makedirs(output_folder, exist_ok=True)

    for image_name in images:

        if soft:
            if random:
                sigma = randint(0, 5)
                gain = randint(3, 6)
            else:
                sigma = 2
                gain = 4
        else:
            sigma = 25
            gain = 1

        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path)

        if noise_type == 'gaussian':
            noisy_image = add_gaussian_noise(image, sigma=sigma)
        elif noise_type == 'poisson':
            noisy_image = add_poisson_noise(image, gain=gain)
        elif noise_type == 'mixed':
            noisy_image = add_gaussian_noise(image, sigma=sigma)
            noisy_image = add_poisson_noise(noisy_image, gain=gain)
        elif noise_type == 'impulsive':
            noisy_image = add_impulsive_noise(image)
        else:
            raise ValueError("Unsupported noise type.")

        output_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_path, noisy_image)
        print(f"Saved: {output_path}")


def add_noise_to_video(input_video_path, output_video_path, noise_type='gaussian', frame_rate=25, random=False, soft=False):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if soft:
            if random:
                sigma = randint(0, 5)
                gain = randint(2, 6)
            else:
                sigma = 2
                gain = 4
        else:
            sigma = 25
            gain = 1

        if noise_type == 'gaussian':

            noisy_frame = add_gaussian_noise(frame, sigma=sigma)
        elif noise_type == 'poisson':
            noisy_frame = add_poisson_noise(frame, gain=gain)
        elif noise_type == 'mixed':
            noisy_frame = add_gaussian_noise(frame, sigma=sigma)
            noisy_frame = add_poisson_noise(noisy_frame, gain=gain)
        elif noise_type == 'impulsive':
            noisy_frame = add_impulsive_noise(frame)
        else:
            raise ValueError("Unsupported noise type. Use 'gaussian'.")

        out.write(noisy_frame)

    cap.release()
    out.release()
    print(f"Video saved as {output_video_path}")


def save_original_video(input_video_path, output_video_path, frame_rate=25):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the denoiser")

    parser.add_argument("--input_path", "-i", type=str, help="Image sequence directory or video path")
    parser.add_argument("--output_path", "-o", type=str, help="Output directory or video path")
    parser.add_argument("--frame_rate", "-f", type=int, default=25, help="Frame rate of the output video")
    parser.add_argument("--data_type", "-dt", type=str, default="video", choices=["video", "images"], help="Use image sequences or videos as input")
    parser.add_argument("--noise_type", "-n", type=str, default="mixed", choices=["gaussian", "poisson", "impulsive", "mixed"], help="Type of noise to add")
    parser.add_argument("--soft", "-s", type=bool, default=True, help="Use soft noise")
    parser.add_argument("--save_compressed_originals","-so", action="store_true", help="Save compressed original video")
    parser.add_argument("--random", "-r", type=bool, default=True, help="Apply random intensity")

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    if args.data_type == "images":
        sequences = [seq for seq in os.listdir(args.input_path) if os.path.isdir(os.path.join(args.input_path, seq))]
        for seq in sequences:
            input_folder = os.path.join(args.input_path, seq)
            output_folder = os.path.join(args.output_path, seq)
            add_noise_image_sequence(input_folder, output_folder, noise_type=args.noise_type, soft=args.soft, random=args.random)
    elif args.data_type == "video":
        videos = [video for video in os.listdir(args.input_path) if video.endswith(('.mp4', '.avi', '.mkv'))]
        for video in videos:
            input_video_path = os.path.join(args.input_path, video)
            output_video_path = os.path.join(args.output_path, video)
            add_noise_to_video(input_video_path, output_video_path, noise_type=args.noise_type, frame_rate=args.frame_rate, soft=args.soft, random=args.random)
    else:
        raise ValueError("Unsupported data type. Use 'video' or 'images'.")
