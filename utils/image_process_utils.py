from PIL import Image
import torch
import numpy as np
import cv2 as cv
import os

def load_pil_img(load_path):
    img = Image.open(load_path).convert('RGB')
    return img

def load_img_as_torch_batch(load_path):
    pil_img = load_pil_img(load_path)
    return pil_img_to_torch_tensor(pil_img).unsqueeze(0)

def pil_img_to_torch_tensor_grayscale(img_pil):
    '''
    Takes a PIL image and returns a torch tensor of shape (1, 1, H, W) with values in [0, 1]
    '''
    print(f'check!!!! {np.array(img_pil).shape}')
    if img_pil.mode != 'L':
        img_pil = img_pil.convert('L')
    return torch.tensor(np.array(img_pil).transpose(0, 1)/255, dtype=torch.float).unsqueeze(0).unsqueeze(0)

def pil_img_to_torch_tensor(img_pil):
    '''
    Takes a PIL image and returns a torch tensor of shape (1, 3, H, W) with values in [0, 1]
    '''
    
    return torch.tensor(np.array(img_pil).transpose(2, 0, 1)/255, dtype=torch.float).unsqueeze(0)

def torch_to_pil_img(img_torch):
    '''
    Takes a torch tensor of shape (1, 3, H, W) with values in [0, 1] and returns a PIL image
    '''
    return Image.fromarray((img_torch.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)*255).astype('uint8'))

def torch_to_pil_img_batch(img_torch):
    '''
    Takes a torch tensor of shape (1, 3, H, W) with values in [0, 1] and returns a PIL image
    '''
    return [torch_to_pil_img(img_torch[i]) for i in range(img_torch.shape[0])]

# for pil video saving
def pil_to_cv(pil_image):
    return cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)

def pil_to_cv_gray(pil_img):
    return cv.cvtColor(cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR), cv.COLOR_RGB2GRAY)

def np_to_pil(np_img):
    return Image.fromarray((np_img/255).astype(np.float32).transpose(2,0,1), 'RGB')

def cv_to_pil(np_img):
    return Image.fromarray((np_img/255).astype(np.float32), 'RGB')

# copied from diffusers.image_processor.blur
from PIL import ImageFilter
def pil_blur(pil_image, blur_factor=4):
    """
    Applies Gaussian blur to an image.
    """
    image = pil_image.filter(ImageFilter.GaussianBlur(blur_factor))

    return image

def create_grid_from_numpy(np_img, grid_size=[2,2]):
    ## deal with both np array shape dim = 3 and 4
    if np_img.ndim == 3:
        num_images, h, w = np_img.shape
        c = 1
    elif np_img.ndim == 4:
        num_images, h, w, c = np_img.shape
    else:
        raise ValueError("np_img must be a 3D or 4D array")
    
    w_grid = w * grid_size[1]
    h_grid = h * grid_size[0]
    
    if c == 1:
        grid = np.zeros((h_grid, w_grid))
    elif c == 3:
        grid = np.zeros((h_grid, w_grid, c))
        
    img_idx = 0
    
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if img_idx >= num_images:
                break
            if c == 1:
                grid[i*h:(i+1)*h, j*w:(j+1)*w] = np_img[img_idx]
            elif c == 3:
                grid[i*h:(i+1)*h, j*w:(j+1)*w, :] = np_img[img_idx]
            img_idx += 1

    return grid




def pil_images_to_video(pil_images, output_path, fps=11):
    '''
    video generatatior by using pil images
    '''
    
    # Get the size of the images (assuming all images are the same size)
    frame_size = pil_images[0].size  # (width, height)
    frame_width, frame_height = frame_size

    # Define the codec and create a VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi files or 'mp4v' for .mp4 files
    video_writer = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Write each frame to the video
    for pil_image in pil_images:
        frame = pil_to_cv(pil_image)
        video_writer.write(frame)

    # Release the VideoWriter object
    video_writer.release()
    print('Video saved successfully!')