import cv2
import numpy as np

def enhance_image(input_path, output_path):
    # Load the image
    img = cv2.imread(input_path)

    # Apply a thermal imaging effect
    thermal_img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

    # Increase the image resolution
    upscaled_img = cv2.resize(thermal_img, (800, 600), interpolation=cv2.INTER_CUBIC)

    # Save the enhanced image
    cv2.imwrite(output_path, upscaled_img)

    return upscaled_img

# Example usage
input_image_path = 'ir_image.jpg'
output_image_path = 'enhanced_image2.jpg'
enhanced_image = enhance_image(input_image_path, output_image_path)

# Display the enhanced image
cv2.imshow('Enhanced Image', enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()