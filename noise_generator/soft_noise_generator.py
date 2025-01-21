import cv2
import numpy as np
import os
import argparse
import torch

def add_gaussian_noise(image, mean=0, sigma=25, device="cuda"):
    """
    Add Gaussian noise to an image.
    
    Parameters:
        image (torch.tensor): The input image.
        mean (float): Mean of the Gaussian noise.
        sigma (float): Standard deviation of the Gaussian noise.
        
    Returns:
        torch.tensor: The noisy image.
    """
    noise = (torch.randn_like(image) * sigma).to(device)
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0, 255)


def add_poisson_noise(image, gain=1, device="cuda"):
    """
    Add Poisson noise to an image.
    
    Parameters:
        image (torch.tensor): Input image (uint8 format, range 0-255).
    
    Returns:
        torch.tensor: Noisy image (uint8 format, range 0-255).
    """
    # Ensure the input image is in float32 format and scale it to [0, 1]
    image = image / 255.0
    # Apply Poisson noise
    noisy_image = (torch.poisson(image * gain * 255) / (255.0 * gain)).to(device)

    return torch.clamp(noisy_image * 255, 0, 255)


def add_soft_noise(img_tensor, gain=4, sigma=2, device="cuda"):
    img_tensor = add_gaussian_noise(img_tensor, sigma=sigma, device=device)
    img_tensor = add_poisson_noise(img_tensor, gain=gain, device=device)
    return img_tensor