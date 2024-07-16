import cv2
import os
import argparse
import skimage.io
import numpy as np

def convert_to_color(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for filename in os.listdir(input_path):
        if filename.endswith(".png"):
            bw_image_path = os.path.join(input_path, filename)
            color_image_path = os.path.join(output_path, filename)

            # Read the black and white disparity image
            bw_image = skimage.io.imread(bw_image_path).astype(np.uint16)
            
            # Apply color map
            color_image = cv2.applyColorMap(cv2.convertScaleAbs(bw_image, alpha=0.01), cv2.COLORMAP_JET)
            
            # Save the color image
            cv2.imwrite(color_image_path, color_image)
            print(f"Converted {filename} to color and saved at {color_image_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert BW Disparity Images to Color')
    parser.add_argument('--input_path', required=True, help='Path to input directory containing BW disparity images')
    parser.add_argument('--output_path', required=True, help='Path to output directory to save color disparity images')
    
    args = parser.parse_args()
    convert_to_color(args.input_path, args.output_path)
