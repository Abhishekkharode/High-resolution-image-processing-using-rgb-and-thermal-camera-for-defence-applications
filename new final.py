import cv2
import numpy as np
import os

def enhance_ir_image(image_path, scale_factor=2):
    print(f"Attempting to process image: {image_path}")
   
    if not os.path.isfile(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return None

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
   
    if img is None:
        print(f"Error: Unable to read the image file '{image_path}'. It may be corrupted or in an unsupported format.")
        return None

    print(f"Image successfully loaded. Shape: {img.shape}, Dtype: {img.dtype}")
   
    # Check if the image is single-channel or has identical channels
    if len(img.shape) == 2 or (len(img.shape) == 3 and np.all(img[:,:,0] == img[:,:,1]) and np.all(img[:,:,0] == img[:,:,2])):
        print("Processing as single-channel image...")
        # For single-channel or identical multi-channel images, use only one channel
        if len(img.shape) == 3:
            img = img[:,:,0]
       
        ir_upscaled = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
       
        # Ensure the image is in 8-bit format
        if ir_upscaled.dtype != np.uint8:
            ir_upscaled = cv2.normalize(ir_upscaled, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
       
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        ir_enhanced = clahe.apply(ir_upscaled)
       
        # Apply colormap
        result = cv2.applyColorMap(ir_enhanced, cv2.COLORMAP_JET)
    else:
        print("Processing multi-channel image...")
        # For true multi-channel images, proceed with the original logic
        ir_channel = img[:,:,0]
        rgb_channels = img[:,:,1:]
       
        ir_upscaled = cv2.resize(ir_channel, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        rgb_upscaled = cv2.resize(rgb_channels, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
       
        ir_enhanced = cv2.detailEnhance(ir_upscaled, sigma_s=10, sigma_r=0.15)
        ir_colored = cv2.applyColorMap(ir_enhanced, cv2.COLORMAP_JET)
        result = cv2.addWeighted(ir_colored, 0.7, cv2.cvtColor(rgb_upscaled, cv2.COLOR_BGR2RGB), 0.3, 0)
   
    print("Image processing completed.")
    return result

# Debugging information
print(f"Current working directory: {os.getcwd()}")
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Script directory: {script_dir}")

# List all files in the script directory
print("Files in the script directory:")
for file in os.listdir(script_dir):
    print(f"  {file}")

# Use the file name provided by the user
image_filename = "ir_image.jpg"
input_image = os.path.join(script_dir, image_filename)

print(f"Full path to image: {input_image}")

if not os.path.isfile(input_image):
    print(f"Error: The file '{input_image}' does not exist.")
    print("Please make sure you entered the correct filename and that the image is in the specified directory.")
else:
    try:
        output_image = enhance_ir_image(input_image)

        if output_image is not None:
            output_path = os.path.join(script_dir, 'enhanced_ir_image.jpg')
            cv2.imwrite(output_path, output_image)
            print(f"Enhanced image saved as: {output_path}")

            # Display the result (optional)
            cv2.imshow('Enhanced IR Image', output_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Image processing failed. Please check the error messages above.")
    except Exception as e:
        print(f"An error occurred during image processing: {str(e)}")

print("Script execution completed.")