import cv2
import numpy as np
from pathlib import Path

def create_temperature_scale(height, width=60, min_temp=20, max_temp=40):
    """
    Create a temperature scale bar with colors and temperature values.
    
    Parameters:
    height (int): Height of the scale bar
    width (int): Width of the scale bar
    min_temp (float): Minimum temperature in Celsius
    max_temp (float): Maximum temperature in Celsius
    
    Returns:
    numpy.ndarray: Image array containing the temperature scale
    """
    # Create gradient
    scale = np.linspace(0, 255, height, dtype=np.uint8)
    scale = np.tile(scale.reshape(-1, 1), (1, width))
    
    # Apply the same colormap as the main image
    colored_scale = cv2.applyColorMap(scale, cv2.COLORMAP_JET)
    
    # Add white background for text
    text_overlay = np.ones((height, 40, 3), dtype=np.uint8) * 255
    
    # Add temperature labels
    num_labels = 6
    for i in range(num_labels):
        y_pos = int(i * (height - 20) / (num_labels - 1))
        temp = max_temp - (i * (max_temp - min_temp) / (num_labels - 1))
        cv2.putText(text_overlay, f'{temp:0.1f}°C', 
                    (2, y_pos + 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4, (0, 0, 0), 1)
    
    # Combine scale and text
    return np.hstack((colored_scale, text_overlay))

def enhance_image(input_path, output_path, target_size=(800, 600), 
                 min_temp=20, max_temp=40, colormap=cv2.COLORMAP_JET):
    """
    Enhance an IR image with thermal effects and temperature scale.
    
    Parameters:
    input_path (str): Path to input image
    output_path (str): Path to save enhanced image
    target_size (tuple): Desired output resolution (width, height)
    min_temp (float): Minimum temperature in Celsius
    max_temp (float): Maximum temperature in Celsius
    colormap (int): OpenCV colormap to use for thermal effect
    
    Returns:
    numpy.ndarray: Enhanced image array with temperature scale
    """
    try:
        # Check if input file exists
        if not Path(input_path).is_file():
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        # Load the image
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Failed to load image: {input_path}")
            
        # Convert to grayscale for better thermal mapping
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply noise reduction
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(denoised)
        
        # Apply thermal imaging effect
        thermal_img = cv2.applyColorMap(contrast_enhanced, colormap)
        
        # Apply sharpening
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        sharpened = cv2.filter2D(thermal_img, -1, kernel)
        
        # Resize the image
        upscaled_img = cv2.resize(sharpened, target_size, interpolation=cv2.INTER_CUBIC)
        
        # Create and add temperature scale
        scale_height = target_size[1]
        temp_scale = create_temperature_scale(scale_height, min_temp=min_temp, max_temp=max_temp)
        
        # Add the scale to the right of the image
        final_img = np.hstack((upscaled_img, temp_scale))
        
        # Add temperature estimation legend
        legend_text = [
            "Temperature Scale (Estimated)",
            f"Range: {min_temp}°C - {max_temp}°C",
            "Note: Values are approximate"
        ]
        
        y_offset = 30
        for text in legend_text:
            cv2.putText(final_img, text, 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
            y_offset += 20
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the enhanced image
        success = cv2.imwrite(str(output_path), final_img)
        if not success:
            raise IOError(f"Failed to save image to: {output_path}")
            
        return final_img
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise

def display_image(image, window_name="Enhanced Thermal Image", wait_time=0):
    """
    Display an image in a window with proper cleanup.
    
    Parameters:
    image (numpy.ndarray): Image to display
    window_name (str): Name of the window
    wait_time (int): Time to wait in milliseconds (0 = wait for key press)
    """
    try:
        cv2.imshow(window_name, image)
        cv2.waitKey(wait_time)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error displaying image: {str(e)}")

def estimate_temperature_at_point(image, x, y, min_temp=20, max_temp=40):
    """
    Estimate temperature at a specific point in the thermal image.
    
    Parameters:
    image (numpy.ndarray): Thermal image
    x, y (int): Coordinates of the point
    min_temp, max_temp (float): Temperature range in Celsius
    
    Returns:
    float: Estimated temperature at the point
    """
    try:
        # Get the blue channel value (used in JET colormap)
        pixel_value = image[y, x, 0]
        # Map the pixel value to temperature range
        temp = min_temp + (pixel_value / 255.0) * (max_temp - min_temp)
        return temp
    except IndexError:
        print("Point coordinates out of image bounds")
        return None

if __name__ == "__main__":
    # Example usage
    input_image_path = 'ir_image.jpg'
    output_image_path = 'enhanced_thermal_image.jpg'
    
    try:
        # Process the image with temperature scale
        enhanced_image = enhance_image(
            input_image_path,
            output_image_path,
            target_size=(800, 600),
            min_temp=20,  # Minimum temperature in Celsius
            max_temp=40,  # Maximum temperature in Celsius
            colormap=cv2.COLORMAP_JET
        )
        
        # Display the result
        display_image(enhanced_image)
        
        # Example of temperature estimation at a point
        # (Click on image to get coordinates for your actual use)
        center_x, center_y = 400, 300  # Center point of image
        temp = estimate_temperature_at_point(enhanced_image, center_x, center_y)
        if temp is not None:
            print(f"Estimated temperature at center point: {temp:.1f}°C")
        
    except Exception as e:
        print(f"Application error: {str(e)}")