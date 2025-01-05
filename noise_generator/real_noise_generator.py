import torch
import torch.nn.functional as F
import cv2
import numpy as np
from noise_generator import smartphone_noise_generator

# Define noise generation functions with random intensities
def gaussian_noise(images, intensity_range=(5, 25)):
    """Add Gaussian noise with random intensity to a batch of images."""
    intensity = np.random.uniform(*intensity_range)
    noise = torch.randn_like(images) * intensity
    noisy_images = images + noise
    return torch.clamp(noisy_images, 0, 255)

def salt_and_pepper_noise(images, batch=True, prob_range=(0.01, 0.05)):
    """Add salt-and-pepper noise with random probability to a batch of images."""
    prob = np.random.uniform(*prob_range)
    noisy_images = images.clone()
    if batch:
        batch_size, num_frames, channels, height, width = images.shape
        total_pixels = height * width
        num_salt = int(total_pixels * prob / 2)
        num_pepper = int(total_pixels * prob / 2)

        for i in range(batch_size):
            for j in range(num_frames):
                # Add salt
                salt_coords = torch.randint(0, height, (num_salt,)), torch.randint(0, width, (num_salt,))
                noisy_images[i, j, :, salt_coords[0], salt_coords[1]] = 255

                # Add pepper
                pepper_coords = torch.randint(0, height, (num_pepper,)), torch.randint(0, width, (num_pepper,))
                noisy_images[i, j, :, pepper_coords[0], pepper_coords[1]] = 0
    
    else:
        num_frames, channels, height, width = images.shape
        total_pixels = height * width
        num_salt = int(total_pixels * prob / 2)
        num_pepper = int(total_pixels * prob / 2)

        for i in range(num_frames):
            # Add salt
            salt_coords = torch.randint(0, height, (num_salt,)), torch.randint(0, width, (num_salt,))
            noisy_images[i, :, salt_coords[0], salt_coords[1]] = 255

            # Add pepper
            pepper_coords = torch.randint(0, height, (num_pepper,)), torch.randint(0, width, (num_pepper,))
            noisy_images[i, :, pepper_coords[0], pepper_coords[1]] = 0

    return noisy_images

def poisson_noise(images, scale_range=(30, 100)):
    """Add Poisson noise with random scaling factor to a batch of images."""
    scale = np.random.uniform(*scale_range)
    noisy_images = torch.poisson(images / 255.0 * scale) / scale * 255.0
    return torch.clamp(noisy_images, 0, 255)

def speckle_noise(images, intensity_range=(0.05, 0.2)):
    """Add speckle noise with random intensity to a batch of images."""
    intensity = np.random.uniform(*intensity_range)
    noise = torch.randn_like(images) * intensity
    noisy_images = images + images * noise
    return torch.clamp(noisy_images, 0, 255)

def heteroscedastic_noise(images, a_range=(0.001, 0.02), b_range=(0.0005, 0.01)):
    """Add heteroscedastic noise with random coefficients to a batch of images."""
    a = np.random.uniform(*a_range)
    b = np.random.uniform(*b_range)
    epsilon = torch.randn_like(images)
    noise = torch.sqrt(a * images + b) * epsilon
    noisy_images = images + noise
    return torch.clamp(noisy_images, 0, 255)

def jpeg_compression(images, quality_range=(10, 50)):
    """Simulate JPEG compression artifacts with random quality for a batch of images."""
    quality = int(np.random.uniform(*quality_range))
    noisy_images = []
    for img in images:
        image_np = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', image_np, encode_param)
        decoded_img = cv2.imdecode(encimg, cv2.IMREAD_UNCHANGED)
        decoded_tensor = torch.tensor(decoded_img, device=img.device).permute(2, 0, 1)
        noisy_images.append(decoded_tensor)
    return torch.stack(noisy_images)

def apply_random_noise(img, probabilities, batch=True, noise_gen_folder='./noise_generator'):
    """Apply random noise to a batch of images based on the given probabilities."""
    assert sum(probabilities.values()) <= 1.0, "Probabilities must sum to 1 or less."

    # Normalize probabilities for clarity
    normalized_probs = {k: v / sum(probabilities.values()) for k, v in probabilities.items()}

    # Randomly choose a noise type for each image in the batch
    noise_type = np.random.choice(list(normalized_probs.keys()), p=list(normalized_probs.values()))
    noisy_images = torch.zeros_like(img)

    if noise_type == 'gaussian':
        noisy_images = gaussian_noise(img)
    elif noise_type == 'salt_and_pepper':
        noisy_images = salt_and_pepper_noise(img, batch)
    elif noise_type == 'poisson':
        noisy_images = poisson_noise(img)
    elif noise_type == 'speckle':
        noisy_images = speckle_noise(img)
    elif noise_type == 'heteroscedastic':
        noisy_images = heteroscedastic_noise(img)
    elif noise_type == 'jpeg':
        noisy_images = jpeg_compression(img)
    elif noise_type == 'smartphone':
        if batch:
            noisy_images = smartphone_noise_generator.generate_train_noisy_tensor(img, noise_gen_folder, device=img.device)
        else:
            noisy_images = smartphone_noise_generator.generate_val_noisy_tensor(img, noise_gen_folder, device=img.device)
            
    return noisy_images, noise_type


# Example usage
if __name__ == "__main__":
    # Load an example batch of images (pixel values in range [0, 255])
    image = cv2.imread("example.jpg")
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # No normalization
    batch = image.unsqueeze(0).repeat(8, 1, 1, 1)  # Create a batch of 8 identical images

    # Define probabilities for each noise type
    noise_probabilities = {
        'gaussian': 0.2,
        'salt_and_pepper': 0.1,
        'poisson': 0.1,
        'speckle': 0.1,
        'heteroscedastic': 0.1,
        'smartphone': 0.4,
        'jpeg': 0.0
    }

    # Apply random noise
    noisy_batch, noise_type = apply_random_noise(batch, noise_probabilities)

    # Save and visualize one of the noisy images
    noisy_image_np = noisy_batch[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    cv2.imwrite("noisy_image.jpg", noisy_image_np)
    cv2.imshow("Noisy Image", noisy_image_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
