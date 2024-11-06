import cv2
import numpy as np
from PIL import Image
from pathlib import Path

def enhance_ir_image(input_path, output_path, scale_factor=2, denoise_strength=10):
    """
    Enhance and upscale an IR image while preserving important thermal details.
    
    Parameters:
    input_path (str): Path to input IR image
    output_path (str): Path to save enhanced image
    scale_factor (int): How much to upscale the image (2 = 2x, 3 = 3x, etc.)
    denoise_strength (int): Strength of denoising (higher = more smoothing)
    """
    # Read the image in grayscale mode
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Image at path '{input_path}' could not be read.")
    
    # Apply denoising to reduce noise while preserving edges
    img_denoised = cv2.fastNlMeansDenoising(img, None, denoise_strength)
    
    # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_denoised)
    
    # Upscale the image using Lanczos interpolation
    height, width = img_enhanced.shape
    new_height, new_width = height * scale_factor, width * scale_factor
    
    # Convert to PIL Image for high-quality resizing
    pil_img = Image.fromarray(img_enhanced)
    pil_img_upscaled = pil_img.resize((new_width, new_height), Image.LANCZOS)
    
    # Convert back to numpy array
    img_upscaled = np.array(pil_img_upscaled)
    
    # Apply sharpening kernel to enhance details
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    img_sharpened = cv2.filter2D(img_upscaled, -1, kernel)
    
    # Save the enhanced image
    cv2.imwrite(output_path, img_sharpened)
    
    return img_sharpened

# Example usage
if __name__ == "__main__":
    input_path = "ir_image.jpg"  # Replace with your input image path
    output_path = "enhanced_ir_image.jpg"  # Replace with desired output path
    
    # Enhance the image with 2x upscaling
    enhanced_img = enhance_ir_image(
        input_path=input_path,
        output_path=output_path,
        scale_factor=2,
        denoise_strength=10
    )
    
    print(f"Enhanced image saved to: {output_path}")
