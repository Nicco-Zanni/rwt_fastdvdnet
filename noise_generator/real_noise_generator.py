import torch
import torch.nn.functional as F
import cv2
import numpy as np
from noise_generator import smartphone_noise_generator

# Define noise generation functions
def gaussian_noise(images, intensity=0.05):
    """Add Gaussian noise to a batch of images."""
    noise = torch.randn_like(images) * intensity
    noisy_images = images + noise
    return torch.clamp(noisy_images, 0, 1)

def salt_and_pepper_noise(images, batch=True, prob=0.02):
    """Add salt-and-pepper noise to a batch of images."""
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
                noisy_images[i, j, :, salt_coords[0], salt_coords[1]] = 1

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
            noisy_images[i, :, salt_coords[0], salt_coords[1]] = 1

            # Add pepper
            pepper_coords = torch.randint(0, height, (num_pepper,)), torch.randint(0, width, (num_pepper,))
            noisy_images[i, :, pepper_coords[0], pepper_coords[1]] = 0

    return noisy_images

def poisson_noise(images):
    """Add Poisson noise to a batch of images."""
    noisy_images = torch.poisson(images * 255) / 255
    return torch.clamp(noisy_images, 0, 1)

def speckle_noise(images, intensity=0.1):
    """Add speckle noise to a batch of images."""
    noise = torch.randn_like(images) * intensity
    noisy_images = images + images * noise
    return torch.clamp(noisy_images, 0, 1)

def heteroscedastic_noise(images, a=0.01, b=0.005):
    """Add heteroscedastic noise to a batch of images."""
    epsilon = torch.randn_like(images)
    noise = torch.sqrt(a * images + b) * epsilon
    noisy_images = images + noise
    return torch.clamp(noisy_images, 0, 1)

def jpeg_compression(images, quality=20):
    """Simulate JPEG compression artifacts for a batch of images."""
    noisy_images = []
    for img in images:
        image_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', image_np, encode_param)
        decoded_img = cv2.imdecode(encimg, cv2.IMREAD_UNCHANGED) / 255.0
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
    # Load an example batch of images (normalized to range [0, 1])
    image = cv2.imread("example.jpg")
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
    batch = image.unsqueeze(0).repeat(8, 1, 1, 1)  # Create a batch of 8 identical images

    # Define probabilities for each noise type
    noise_probabilities = {
        'gaussian': 0.2,
        'salt_and_pepper': 0.1,
        'poisson': 0.1,
        'speckle': 0.1,
        'heteroscedastic': 0.1,
        'smartphone': 0.4,
        'jpeg': 0.
    }

    # Apply random noise
    noisy_batch = apply_random_noise(batch, noise_probabilities)

    # Save and visualize one of the noisy images
    noisy_image_np = (noisy_batch[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite("noisy_image.jpg", noisy_image_np)
    cv2.imshow("Noisy Image", noisy_image_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
