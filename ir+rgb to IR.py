import cv2
import numpy as np
import os

def merge_ir_rgb_and_convert(ir_path, rgb_path, output_path):
    # Read the IR and RGB images
    ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    rgb_img = cv2.imread(rgb_path)

    # Check if images are loaded properly
    if ir_img is None or rgb_img is None:
        raise ValueError("Failed to load one or both images. Please check the file paths.")

    # Ensure both images have the same dimensions
    if ir_img.shape[:2] != rgb_img.shape[:2]:
        # Resize IR image to match RGB image
        ir_img = cv2.resize(ir_img, (rgb_img.shape[1], rgb_img.shape[0]))

    # Normalize the IR image to 0-255 range
    ir_normalized = cv2.normalize(ir_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Create a colormap for the IR image
    ir_colormap = cv2.applyColorMap(ir_normalized, cv2.COLORMAP_JET)

    # Merge the IR and RGB images
    merged = cv2.addWeighted(rgb_img, 0.7, ir_colormap, 0.3, 0)

    # Convert merged image back to grayscale (simulating IR)
    merged_gray = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)

    # Apply a thermal-like colormap to the grayscale image
    final_ir = cv2.applyColorMap(merged_gray, cv2.COLORMAP_INFERNO)

    # Save the final IR-like image
    cv2.imwrite(output_path, final_ir)

    return final_ir

def get_valid_path(prompt):
    while True:
        path = input(prompt)
        if os.path.exists(path):
            return path
        else:
            print(f"File not found: {path}")
            print("Please enter a valid file path or type 'exit' to quit.")
        if path.lower() == 'exit':
            exit()

try:
    # Prompt user for input file paths
    ir_image_path = get_valid_path("Enter the path to the IR image: ")
    rgb_image_path = get_valid_path("Enter the path to the RGB image: ")
    
    # Prompt user for output file path
    output_image_path = input("Enter the path for the output image (including filename): ")

    # Merge the images and convert back to IR-like image
    final_ir_image = merge_ir_rgb_and_convert(ir_image_path, rgb_image_path, output_image_path)

    print(f"Successfully created and saved the final IR-like image to {output_image_path}")

    # Display the final IR-like image
    cv2.imshow('Final IR-like Image', final_ir_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"An error occurred: {str(e)}")
    