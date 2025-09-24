import cv2
import numpy as np
import torch
import torch.nn.functional as F


def descriptor(image: np.ndarray) -> torch.Tensor:
    """
    Extract a feature descriptor from an image for comparison purposes.
    
    This function converts the image to grayscale, applies Gaussian blur for noise reduction,
    computes gradients, and creates a normalized feature vector that can be used for
    image comparison tasks.
    
    Args:
        image: Input image as np.ndarray (BGR format)
        
    Returns:
        torch.Tensor: Normalized feature descriptor vector
    """
    # Convert to grayscale if the image is colored
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Resize to standard size for consistent descriptor length
    resized = cv2.resize(blurred, (64, 64))
    
    # Compute gradients
    grad_x = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude and direction
    magnitude = cv2.magnitude(grad_x, grad_y)
    
    # Create histogram of oriented gradients (simplified)
    # Divide image into 8x8 blocks and compute histogram for each
    block_size = 8
    n_bins = 9
    descriptor_features = []
    
    for i in range(0, 64, block_size):
        for j in range(0, 64, block_size):
            block_mag = magnitude[i:i+block_size, j:j+block_size]
            block_grad_x = grad_x[i:i+block_size, j:j+block_size]
            block_grad_y = grad_y[i:i+block_size, j:j+block_size]
            
            # Compute angles
            angles = cv2.phase(block_grad_x, block_grad_y, angleInDegrees=True)
            
            # Create histogram
            hist, _ = torch.histogram(
                torch.from_numpy(angles.flatten()).float(),
                bins=n_bins,
                range=(0, 180),
                weight=torch.from_numpy(block_mag.flatten()).float()
            )
            
            descriptor_features.append(hist)
    
    # Concatenate all block histograms
    feature_vector = torch.cat(descriptor_features)
    
    # Add global features
    # Mean and standard deviation of pixel intensities
    global_mean = torch.tensor([resized.mean()], dtype=torch.float32)
    global_std = torch.tensor([resized.std()], dtype=torch.float32)
    
    # Edge density
    edges = cv2.Canny(resized, 50, 150)
    edge_density = torch.tensor([edges.sum() / (64 * 64)], dtype=torch.float32)
    
    # Combine all features
    final_descriptor = torch.cat([feature_vector, global_mean, global_std, edge_density])
    
    # Normalize the descriptor
    descriptor_norm = F.normalize(final_descriptor.unsqueeze(0), p=2, dim=1).squeeze(0)
    
    return descriptor_norm

