import cv2
import numpy as np
from pathlib import Path

class ThermalColorMap:
    """
    Define temperature ranges and their corresponding colors
    """
    def __init__(self, min_temp=20, max_temp=40):
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.range = max_temp - min_temp
        
        # Define color ranges and their temperatures
        self.color_ranges = [
            {
                'color': (0, 0, 255),    # Red (BGR format)
                'color_name': 'Red',
                'temp_range': (min_temp + 0.8 * self.range, max_temp),
                'description': 'Very Hot'
            },
            {
                'color': (0, 127, 255),   # Orange
                'color_name': 'Orange',
                'temp_range': (min_temp + 0.6 * self.range, min_temp + 0.8 * self.range),
                'description': 'Hot'
            },
            {
                'color': (0, 255, 255),   # Yellow
                'color_name': 'Yellow',
                'temp_range': (min_temp + 0.4 * self.range, min_temp + 0.6 * self.range),
                'description': 'Warm'
            },
            {
                'color': (0, 255, 0),     # Green
                'color_name': 'Green',
                'temp_range': (min_temp + 0.2 * self.range, min_temp + 0.4 * self.range),
                'description': 'Cool'
            },
            {
                'color': (255, 0, 0),     # Blue
                'color_name': 'Blue',
                'temp_range': (min_temp, min_temp + 0.2 * self.range),
                'description': 'Cold'
            }
        ]

def create_temperature_legend(height, width, color_map):
    """
    Create a detailed temperature legend showing color ranges.
    """
    # Create black background
    legend = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add title
    cv2.putText(legend, "Temperature Range", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (255, 255, 255), 2)
    
    # Add color boxes with temperature ranges
    start_y = 60
    box_height = 40
    margin = 10
    
    for idx, range_info in enumerate(color_map.color_ranges):
        # Draw color box
        y = start_y + idx * (box_height + margin)
        cv2.rectangle(legend, 
                     (10, y),
                     (40, y + box_height),
                     range_info['color'], -1)
        
        # Add white border
        cv2.rectangle(legend,
                     (10, y),
                     (40, y + box_height),
                     (255, 255, 255), 1)
        
        # Add temperature range text
        temp_range_text = f"{range_info['temp_range'][0]:.1f}°C - {range_info['temp_range'][1]:.1f}°C"
        cv2.putText(legend,
                   f"{range_info['color_name']}: {temp_range_text}",
                   (50, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 1)
        
        # Add description
        cv2.putText(legend,
                   f"({range_info['description']})",
                   (50, y + box_height - 5),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.4, (200, 200, 200), 1)
    
    return legend

def enhance_image(input_path, output_path, target_size=(800, 600), 
                 min_temp=20, max_temp=40, colormap=cv2.COLORMAP_JET):
    """
    Enhance an IR image with thermal effects and detailed temperature legend.
    """
    try:
        # Initialize color map
        color_map = ThermalColorMap(min_temp, max_temp)
        
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
        
        # Create temperature legend
        legend_width = 300  # Width of the legend panel
        legend = create_temperature_legend(target_size[1], legend_width, color_map)
        
        # Add the legend to the right of the image
        final_img = np.hstack((upscaled_img, legend))
        
        # Add timestamp and temperature range info
        info_height = 40
        info_img = np.zeros((info_height, final_img.shape[1], 3), dtype=np.uint8)
        
        # Add temperature range info
        cv2.putText(info_img,
                   f"Temperature Range: {min_temp}°C - {max_temp}°C",
                   (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 1)
        
        # Add timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(info_img,
                   f"Processed: {timestamp}",
                   (final_img.shape[1] - 300, 25),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 1)
        
        # Combine final image with info bar
        final_img = np.vstack((final_img, info_img))
        
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

def get_temperature_description(temp, color_map):
    """
    Get the description for a given temperature.
    """
    for range_info in color_map.color_ranges:
        if range_info['temp_range'][0] <= temp <= range_info['temp_range'][1]:
            return range_info['description']
    return "Unknown"

def estimate_temperature_at_point(image, x, y, min_temp=20, max_temp=40):
    """
    Estimate temperature at a specific point in the thermal image.
    """
    try:
        # Get the blue channel value (used in JET colormap)
        pixel_value = image[y, x, 0]
        # Map the pixel value to temperature range
        temp = min_temp + (pixel_value / 255.0) * (max_temp - min_temp)
        # Get description
        color_map = ThermalColorMap(min_temp, max_temp)
        description = get_temperature_description(temp, color_map)
        return temp, description
    except IndexError:
        print("Point coordinates out of image bounds")
        return None, None

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
            min_temp=20,
            max_temp=40
        )
        
        # Example of getting temperature at a point
        center_x, center_y = 400, 300
        temp, desc = estimate_temperature_at_point(enhanced_image, center_x, center_y)
        if temp is not None:
            print(f"Temperature at point: {temp:.1f}°C ({desc})")
        
        # Display the result
        cv2.imshow('Enhanced Thermal Image', enhanced_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Application error: {str(e)}")