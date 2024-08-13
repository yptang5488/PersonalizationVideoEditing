import os
from PIL import Image
import numpy as np

def create_white_mask_images(directory, count, size):
    # Step 1: Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Step 2: Generate and save the images
    for i in range(count):
        # Create a NumPy array filled with 255 (white in 'L' mode)
        white_mask = np.ones(size, dtype=np.uint8) * 255
        
        # Convert the NumPy array to a PIL image
        mask_image = Image.fromarray(white_mask, 'L')
        
        # Create the filename with leading zeros
        filename = os.path.join(directory, f'{i:06}.png')
        
        # Save the image
        mask_image.save(filename)
        print(f'Saved {filename}')
        
def create_black_mask_images(directory, count, size):
    # Step 1: Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Step 2: Generate and save the images
    for i in range(count):
        # Create a NumPy array filled with 0 (black in 'L' mode)
        black_mask = np.zeros(size, dtype=np.uint8)
        
        # Convert the NumPy array to a PIL image
        mask_image = Image.fromarray(black_mask, 'L')
        
        # Create the filename with leading zeros
        filename = os.path.join(directory, f'{i:06}.png')
        
        # Save the image
        mask_image.save(filename)
        print(f'Saved {filename}')

# Parameters
output_directory = './data/mask/fox_turn_head_black/'
number_of_images = 38
image_size = (512, 512)

# Create and save the white mask images
create_black_mask_images(output_directory, number_of_images, image_size)
