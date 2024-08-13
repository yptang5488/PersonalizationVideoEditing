import cv2
import os
import numpy as np
import PIL.Image as Image
import re

def pil_to_cv2(pil_image):
    """Convert PIL Image to OpenCV format"""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def get_files_from_dir(input_folder):
    # Get all file names from the input folder
    frames = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    frames.sort(key=extract_number)  # Sort the frames in alphabetical order
    print(frames)
    return frames

def pil_images_to_video(pil_images, output_path, output_name, fps=6):
    '''
    video generatatior by using pil images
    '''
    
    # Get the size of the images (assuming all images are the same size)
    frame_size = pil_images[0].size  # (width, height)
    frame_width, frame_height = frame_size
    
    output_path = os.path.join(output_path, output_name)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi files or 'mp4v' for .mp4 files
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Write each frame to the video
    for pil_image in pil_images:
        frame = pil_to_cv2(pil_image)
        video_writer.write(frame)

    # Release the VideoWriter object
    video_writer.release()
    print('Video saved successfully!')
    
def pil_images_to_gif(pil_images, output_path, output_name, fps=6):
    '''
    gif generatatior by using pil images
    '''
    # Define the frames per second (fps)
    duration = int(1000 / fps)  # Duration of each frame in milliseconds

    # Save as GIF
    output_path = os.path.join(output_path, output_name)
    pil_images[0].save(
        output_path,
        save_all=True,
        append_images=pil_images[1:],  # Add the rest of the images
        duration=duration,             # Set the duration between frames
        loop=0                         # 0 means loop forever, set to 1 to loop once
    )
    print('GIF saved successfully!')

def frames_from_folder_to_gif(input_folder, output_gif, fps=6):
    frames = get_files_from_dir(input_folder)
    
    duration = int(1000 // fps)  # Duration of each frame in milliseconds
    
    # Open the first image to get its size
    first_frame = Image.open(os.path.join(input_folder, frames[0]))
    frames_to_save = [Image.open(os.path.join(input_folder, frame)) for frame in frames]
    
    # Save frames as GIF
    first_frame.save(
        output_gif,
        save_all=True,
        append_images=frames_to_save[1:],
        duration=duration,
        loop=0
    )
    print('GIF saved successfully!')

# Function to extract the numerical part of the filename
def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

def frames_from_folder_to_video(input_folder, output_video, fps):
    frames = get_files_from_dir(input_folder)
    
    frame = cv2.imread(os.path.join(input_folder, frames[0]))
    height, width, layers = frame.shape

    # Initialize VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs if needed
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for frame_name in frames:
        frame = cv2.imread(os.path.join(input_folder, frame_name))
        video.write(frame)

    # cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    #### setting
    frames_folder = '/home/cgvsl/P76111131/RAVE/data/mp4_frames/fox_turn_head/27frames'
    output_name = 'fox_turn_head_27'
    fps = 11 # Set your desired fps
    ####
    
    output_video_path = f'{frames_folder}/{output_name}_fps{fps}.mp4'
    output_gif_path = f'{frames_folder}/{output_name}_fps{fps}.gif'

    frames_from_folder_to_video(frames_folder, output_video_path, fps)
    frames_from_folder_to_gif(frames_folder, output_gif_path, fps)