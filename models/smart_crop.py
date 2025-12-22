"""
Smart Crop using Saliency Detection
Automatically detects the most important region (plant) in the image
and crops around it for better feature extraction.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SaliencyCropper:
    """
    Crops images intelligently using saliency detection.
    Focuses on the most "interesting" part of the image (usually the plant).
    """

    def __init__(
        self,
        target_size: int = 384,
        saliency_threshold: float = 0.3,  # Lower threshold (was 0.5)
        min_region_ratio: float = 0.01,  # Much lower minimum (was 0.3 = 30%)
        use_fine_grained: bool = False  # Use Spectral Residual by default
    ):
        """
        Args:
            target_size: Final crop size (e.g., 384x384)
            saliency_threshold: Threshold for saliency map (0-1)
            min_region_ratio: Minimum size of salient region relative to image (1% default)
            use_fine_grained: Use fine-grained saliency (usually worse for plants)
        """
        self.target_size = target_size
        self.saliency_threshold = saliency_threshold
        self.min_region_ratio = min_region_ratio
        self.use_fine_grained = use_fine_grained

        # Initialize saliency detector
        try:
            if use_fine_grained:
                # Fine Grained Saliency - often highlights too much
                self.saliency = cv2.saliency.StaticSaliencyFineGrained_create()
            else:
                # Spectral Residual - better focus on salient objects
                self.saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        except Exception as e:
            logger.warning(f"Failed to create saliency detector: {e}. Using center crop fallback.")
            self.saliency = None

    def get_saliency_bbox(self, image_np: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Compute saliency map and return bounding box of salient region.

        Args:
            image_np: Image as numpy array (H, W, 3) in BGR format

        Returns:
            (x, y, w, h) bounding box or None if detection fails
        """
        if self.saliency is None:
            return None

        try:
            # Compute saliency map
            success, saliency_map = self.saliency.computeSaliency(image_np)

            if not success:
                return None

            # Normalize saliency map to 0-255
            saliency_map = (saliency_map * 255).astype(np.uint8)

            # Threshold to binary mask
            _, binary_mask = cv2.threshold(
                saliency_map,
                int(self.saliency_threshold * 255),
                255,
                cv2.THRESH_BINARY
            )

            # Find contours
            contours, _ = cv2.findContours(
                binary_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                return None

            # Get largest contour (main salient region)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Check if region is large enough
            image_area = image_np.shape[0] * image_np.shape[1]
            region_area = w * h
            if region_area < (self.min_region_ratio * image_area):
                return None

            return (x, y, w, h)

        except Exception as e:
            logger.debug(f"Saliency detection failed: {e}")
            return None

    def smart_crop(self, image: Image.Image) -> Image.Image:
        """
        Crop image intelligently using saliency detection.
        Falls back to center crop if saliency detection fails.

        Args:
            image: PIL Image

        Returns:
            Cropped PIL Image of size (target_size, target_size)
        """
        # Convert PIL to OpenCV format (BGR)
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = image_np.shape[:2]

        # Try to get saliency-based bounding box
        bbox = self.get_saliency_bbox(image_np)

        if bbox is not None:
            x, y, bw, bh = bbox

            # Strategy: Create a crop that focuses on the salient region
            # but maintains proper zoom level

            # Calculate center of salient region
            cx = x + bw // 2
            cy = y + bh // 2

            # Determine crop size based on salient region + padding
            # Add 40% padding (not 20%) to include context
            size = int(max(bw, bh) * 1.4)

            # But ensure crop size makes sense relative to target_size
            # We want to resize, not just crop the exact size
            # Minimum crop size should be ~1.5x target_size to allow for detail
            min_crop_size = int(self.target_size * 1.5)

            # Maximum should be ~3x to avoid taking too much background
            max_crop_size = int(self.target_size * 3)

            size = max(min_crop_size, min(size, max_crop_size))

            # Calculate square crop coordinates centered on salient region
            x1 = max(0, cx - size // 2)
            y1 = max(0, cy - size // 2)
            x2 = min(w, x1 + size)
            y2 = min(h, y1 + size)

            # Adjust if crop goes out of bounds
            if x2 - x1 < size:
                x1 = max(0, x2 - size)
            if y2 - y1 < size:
                y1 = max(0, y2 - size)

            # Crop
            cropped_np = image_np[y1:y2, x1:x2]

            # Convert back to PIL
            cropped_pil = Image.fromarray(cv2.cvtColor(cropped_np, cv2.COLOR_BGR2RGB))

            # Resize to target size
            return cropped_pil.resize((self.target_size, self.target_size), Image.LANCZOS)

        else:
            # Fallback to center crop
            return self.center_crop(image)

    def center_crop(self, image: Image.Image) -> Image.Image:
        """
        Fallback center crop when saliency detection fails.

        First resizes maintaining aspect ratio so the smaller dimension equals target_size,
        then crops the center to get a square of target_size.

        Args:
            image: PIL Image

        Returns:
            Center-cropped PIL Image of size (target_size, target_size)
        """
        w, h = image.size

        # Resize so the smaller dimension equals target_size
        # This ensures we don't upscale small images too much
        if w < h:
            new_w = self.target_size
            new_h = int(h * (self.target_size / w))
        else:
            new_h = self.target_size
            new_w = int(w * (self.target_size / h))

        # Resize image
        image = image.resize((new_w, new_h), Image.LANCZOS)

        # Now crop the center to target_size x target_size
        left = (new_w - self.target_size) // 2
        top = (new_h - self.target_size) // 2
        right = left + self.target_size
        bottom = top + self.target_size

        return image.crop((left, top, right, bottom))


class SmartCropTransform:
    """
    Transform wrapper for use with PyTorch datasets.
    """

    def __init__(self, target_size: int = 384, **kwargs):
        self.cropper = SaliencyCropper(target_size=target_size, **kwargs)

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply smart crop to image."""
        return self.cropper.smart_crop(image)


# Convenience function
def smart_crop_image(
    image: Image.Image,
    target_size: int = 384,
    **kwargs
) -> Image.Image:
    """
    Convenience function to smart crop a single image.

    Args:
        image: PIL Image
        target_size: Target size for crop
        **kwargs: Additional args for SaliencyCropper

    Returns:
        Cropped PIL Image
    """
    cropper = SaliencyCropper(target_size=target_size, **kwargs)
    return cropper.smart_crop(image)
