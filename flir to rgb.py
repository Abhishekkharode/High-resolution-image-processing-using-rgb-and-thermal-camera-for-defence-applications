import cv2
import numpy as np
import os

def merge_ir_rgb(ir_path, rgb_path):
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
    ir_normalized = cv2.normalize(ir_img, None, 0, 200, cv2.NORM_MINMAX).astype(np.uint8)

    # Create a colormap for the IR image
    ir_colormap = cv2.applyColorMap(ir_normalized, cv2.COLORMAP_JET)

    # Merge the IR and RGB images
    merged = cv2.addWeighted(rgb_img, 0.7, ir_colormap, 0.3, 0)

    return merged, ir_normalized

def create_heatmap(ir_img, threshold=200):
    # Create a blank RGB image
    heatmap = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2RGB)

    # Create a mask for high-temperature areas
    mask = ir_img > threshold

    # Apply the JET colormap only to high-temperature areas
    heatmap[mask] = cv2.applyColorMap(ir_img, cv2.COLORMAP_JET)[mask]

    return heatmap

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
    
    # Prompt user for output file paths
    merged_output_path = input("Enter the path for the merged output image: ")
    heatmap_output_path = input("Enter the path for the heatmap output image: ")

    # Merge the images
    merged_image, ir_normalized = merge_ir_rgb(ir_image_path, rgb_image_path)

    # Create heatmap
    heatmap_image = create_heatmap(ir_normalized)

    # Save the merged and heatmap images
    cv2.imwrite(merged_output_path, merged_image)
    cv2.imwrite(heatmap_output_path, heatmap_image)

    print(f"Successfully created and saved the merged image to {merged_output_path}")
    print(f"Successfully created and saved the heatmap image to {heatmap_output_path}")

    # Display the merged and heatmap images
    cv2.imshow('Merged IR and RGB Image', merged_image)
    cv2.imshow('Heatmap of High Temperature Areas', heatmap_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"An error occurred: {str(e)}")