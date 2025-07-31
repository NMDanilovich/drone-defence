from typing import Iterable

import cv2
import torch
import torch.nn.functional as F


class BBox(object):
    def __init__(self, x:int, y:int, w:int, h:int):
        x, y, w, h = tuple(map(int, (x, y, w, h)))

        self._xywh = x, y, w, h
        self._xyxy = x - w // 2,    y - h // 2,    x + w // 2,    y + h // 2

    def __getitem__(self, index):
        return self._xywh[index]

    def __contains__(self, temp):
        
        if isinstance(temp, BBox):
            x1, y1, x2, y2 = temp.xyxy
            pt1_condition = (self._xyxy[0] <= x1 <= self._xyxy[2]) and (self._xyxy[1] <= y1 <= self._xyxy[3])
            pt2_condition = (self._xyxy[0] <= x2 <= self._xyxy[2]) and (self._xyxy[1] <= y2 <= self._xyxy[3])

            return pt1_condition and pt2_condition

        elif isinstance(temp, Iterable) and 2 <= len(temp) <= 4:
            ptx, pty, *other = temp
            x_condition = self._xyxy[0] <= ptx <= self._xyxy[2]
            y_condition = self._xyxy[1] <= pty <= self._xyxy[3]

            return x_condition and y_condition
        
        else:
            raise ValueError("Point should by the two coordinates (X and Y).")
        
    def __repr__(self):
        return f"ROI(x={self._xywh[0]} y={self._xywh[1]} w={self._xywh[2]} h={self._xywh[3]})"

    @property
    def xyxy(self):
        return self._xyxy
    
    @property
    def xywh(self):
        return self._xywh


def descriptor(image: cv2.Mat) -> torch.Tensor:
    """
    Extract a feature descriptor from an image for comparison purposes.
    
    This function converts the image to grayscale, applies Gaussian blur for noise reduction,
    computes gradients, and creates a normalized feature vector that can be used for
    image comparison tasks.
    
    Args:
        image: Input image as cv2.Mat (BGR format)
        
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
