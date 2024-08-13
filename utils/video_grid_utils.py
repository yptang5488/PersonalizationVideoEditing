import os
import cv2 as cv
import numpy as np
import torch
import imageio
import glob
from PIL import ImageDraw

from torchvision.utils import make_grid
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_pil_image

ori_w, ori_h = 0, 0

def add_black_border(pil_image, border_size=10, border_color=(0, 0, 0)):
    """
    Add border to the image.
    
    Args:
        pil_image (PIL.Image): The image to which the border will be added.
        border_size (int): Size of the border to add.
        border_color (tuple): Color of the border.
        
    Returns:
        PIL.Image: Image with border added.
    """
    w, h = pil_image.size
    new_w = w + 2 * border_size
    new_h = h + 2 * border_size
    new_image = Image.new('RGB', (new_w, new_h), border_color)
    new_image.paste(pil_image, (border_size, border_size))
    return new_image

def add_border_with_corner_colors(pil_image, border_size=10):
    """
    Add border to the image using the edge color of the image and the corner colors for the corners.
    
    Args:
        pil_image (PIL.Image): The image to which the border will be added.
        border_size (int): Size of the border to add.
        
    Returns:
        PIL.Image: Image with border added.
    """
    w, h = pil_image.size
    new_w = w + 2 * border_size
    new_h = h + 2 * border_size

    # Get edge colors
    top_color = pil_image.crop((0, 0, w, 1)).resize((new_w, border_size))
    bottom_color = pil_image.crop((0, h-1, w, h)).resize((new_w, border_size))
    left_color = pil_image.crop((0, 0, 1, h)).resize((border_size, new_h))
    right_color = pil_image.crop((w-1, 0, w, h)).resize((border_size, new_h))

    # Create new image with expanded size
    new_image = Image.new('RGB', (new_w, new_h))
    
    # Paste the original image into the center
    new_image.paste(pil_image, (border_size, border_size))

    # Paste the edge colors
    new_image.paste(top_color, (0, 0))
    new_image.paste(bottom_color, (0, new_h - border_size))
    new_image.paste(left_color, (0, 0))
    new_image.paste(right_color, (new_w - border_size, 0))

    # Get corner colors
    top_left_color = pil_image.getpixel((0, 0))
    top_right_color = pil_image.getpixel((w-1, 0))
    bottom_left_color = pil_image.getpixel((0, h-1))
    bottom_right_color = pil_image.getpixel((w-1, h-1))

    # Fill the corners with the respective corner colors
    for x in range(border_size):
        for y in range(border_size):
            new_image.putpixel((x, y), top_left_color)
            new_image.putpixel((new_w - border_size + x, y), top_right_color)
            new_image.putpixel((x, new_h - border_size + y), bottom_left_color)
            new_image.putpixel((new_w - border_size + x, new_h - border_size + y), bottom_right_color)

    return new_image

def remove_border_and_resize(pil_image, border_size=10):
    # 獲取圖片尺寸
    w, h = pil_image.size
    global ori_w, ori_h
    
    # 創建一個去掉邊界的圖片
    cropped_image = pil_image.crop((border_size, border_size, w - border_size, h - border_size))
    resized_image = cropped_image.resize((ori_w, ori_h))
    return resized_image


def prepare_video_to_grid(path, grid_count, grid_size, pad, grid_save_path, border_size=-1, border_type='black'):
    """
    get video and make them as frame grids

    Args:
        path (str): video path
        grid_count (int): config.sample_size, the number of grids to be generated 
                            (-1 for the full video)
        grid_size (int): the size of each grid (e.g. grid_size x grid_size)
        pad (int): the padding of the video (if 1, use the same video)
        grid_save_path (str): path to save frame grids
        
    Returns:
        grids: list of frame grids
    """
    video = cv.VideoCapture(path)
    video_fps = video.get(cv.CAP_PROP_FPS)
    if grid_count == -1:
        frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    else:
        frame_count = min(grid_count * pad * grid_size**2, int(video.get(cv.CAP_PROP_FRAME_COUNT)))

    transform = transforms.Compose([
        transforms.ConvertImageDtype(dtype=torch.float),
    ])
    success = True

    max_grid_area = 512*512* grid_size**2
    grids = []
    frames = []
    grid_index = 0 # for saving grid

    total_grid = grid_size**2
    global ori_w, ori_h
    
    for idx in range(frame_count):
        success,image = video.read()
        
        # get frame size for final resizing
        if idx == 0:
            print(image.shape)
            ori_h, ori_w, _ = image.shape
            
        assert success, 'Video read failed'
        if idx % pad == 0:
            rgb_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            
            if border_size != -1:
                pil_img = Image.fromarray(rgb_img)
                
                ## YP : add border to the image
                if border_type == 'black':
                    pil_img = add_black_border(pil_img, border_size)
                elif border_type == 'edge':
                    pil_img = add_border_with_corner_colors(pil_img, border_size)
                    
                # Convert to tensor
                rgb_img = np.array(pil_img)
            
            ## original version
            rgb_img = np.transpose(rgb_img, (2, 0, 1))
            
            frames.append(transform(torch.from_numpy(rgb_img)))
            
            if len(frames) == total_grid:
                grid = make_grid(frames, nrow=grid_size, padding=0)
                pil_image = (to_pil_image(grid))
                w,h = pil_image.size
                a = float(np.sqrt((w*h/max_grid_area)))
                w1 = int((w//a)//(grid_size*8))*grid_size*8
                h1 = int((h//a)//(grid_size*8))*grid_size*8
                pil_image= pil_image.resize((w1, h1))
                grids.append(pil_image)
                
                print(f'[vgu] frame resize : w,h = ({w}, {h}) ; w1,h1 = ({w1}, {h1})')
                
                # Save the grid image
                save_file = os.path.join(grid_save_path, f'grid_{grid_index}.png')
                pil_image.save(save_file)
                grid_index += 1

                frames = []

    return grids #, video_fps # list of frames

def prepare_video_to_frames(path, grid_count, grid_size, pad, format='gif'):
    video = cv.VideoCapture(path)
    
    if grid_count == -1:
        frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        
    else:
        frame_count = min(grid_count * pad * grid_size**2, int(video.get(cv.CAP_PROP_FRAME_COUNT)))
        
    frame_idx = 0
    frames = []
    frames_grid = []

    dir_path = os.path.dirname(path)
    video_name = path.split('/')[-1].split('.')[0]
    os.makedirs(os.path.join(dir_path, 'frames/', video_name), exist_ok=True)
    os.makedirs(os.path.join(dir_path, 'video/', video_name), exist_ok=True)

    for idx in range(frame_count):
        success,image = video.read()
        assert success, 'Video read failed'
        if idx % pad == 0:
            frames.append(image)

    for frame in frames[:(len(frames)//(grid_size**2)*(grid_size**2))]:
        frames_grid.append(frame)
        cv.imwrite(os.path.join(dir_path, 'frames/', video_name, f'{str(frame_idx).zfill(5)}.png'), frame)
        frame_idx += 1


    if format == 'gif':
        with imageio.get_writer(os.path.join(dir_path, 'video/', f'{video_name}_fc{frame_idx}_pad{pad}_grid{grid_size}.gif'), mode='I') as writer:
            for frame in frames_grid:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                writer.append_data(frame)
    elif format == 'mp4':
        image_files = sorted(glob.glob(os.path.join(dir_path, 'frames/', video_name, '*.png')))
        images = [imageio.imread(image_file) for image_file in image_files]
        save_file_path = os.path.join(dir_path, 'video/', f'{video_name}_fc{frame_idx}_pad{pad}_grid{grid_size}.mp4')
        imageio.mimsave(save_file_path, images, fps=20)

    return frame_idx # number of frames


import re
from PIL import Image

def numerical_sort(value):
    parts = re.split(r'(\d+)', value)
    return [int(part) if part.isdigit() else part for part in parts]

def prepare_frames_to_grid(path, grid_count, grid_size, pad, grid_save_path, border_size=-1, border_type='black'):
    """
    make frames into frame grids

    Args:
        path (str): frames path
        grid_count (int): config.sample_size, the number of grids to be generated 
                            (-1 for the full video)
        grid_size (int): the size of each grid (e.g. grid_size x grid_size)
        pad (int): the padding of the video (if 1, use the same video)
        grid_save_path (str): path to save frame grids
        
    Returns:
        grids: list of frame grids
    """
    # Get list of image files in the directory
    image_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]
                         , key=numerical_sort)
    
    if grid_count == -1:
        frame_count = len(image_files)
    else:
        frame_count = min(grid_count * pad * grid_size**2, len(image_files))

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    max_grid_area = 512*512* grid_size**2
    grids = []
    frames = []
    grid_index = 0 # for saving grid

    total_grid = grid_size**2
    for idx in range(frame_count):
        if idx % pad == 0:
            image = Image.open(image_files[idx])
            rgb_img = image.convert('RGB')
            
            if border_size != -1:
                ## YP : add border to the image
                if border_type == 'black':
                    rgb_img = add_black_border(rgb_img, border_size)
                elif border_type == 'edge':
                    rgb_img = add_border_with_corner_colors(rgb_img, border_size)
            
            rgb_img = np.array(rgb_img)
            # print(f'[read control] rgb_img.shape={rgb_img.shape}') #[h,w,3]
            
            # rgb_img = np.transpose(rgb_img, (2, 0, 1))
            frames.append(transform(rgb_img))
            
            if len(frames) == total_grid:
                grid = make_grid(frames, nrow=grid_size, padding=0)
                pil_image = (to_pil_image(grid))
                w,h = pil_image.size
                a = float(np.sqrt((w*h/max_grid_area)))
                w1 = int((w//a)//(grid_size*8))*grid_size*8
                h1 = int((h//a)//(grid_size*8))*grid_size*8
                pil_image= pil_image.resize((w1, h1))
                grids.append(pil_image)
                
                # Save the grid image
                save_file = os.path.join(grid_save_path, f'grid_{grid_index}.png')
                pil_image.save(save_file)
                grid_index += 1

                frames = []

    return grids # list of frames

import utils.image_process_utils as ipu
def prepare_frames_to_grid_gray(path, grid_count, grid_size, pad, grid_save_path, mask_blur_factor=-1, border_size=-1):
    """
    make frames into frame grids

    Args:
        path (str): frames path
        grid_count (int): config.sample_size, the number of grids to be generated 
                            (-1 for the full video)
        grid_size (int): the size of each grid (e.g. grid_size x grid_size)
        pad (int): the padding of the video (if 1, use the same video)
        grid_save_path (str): path to save frame grids
        
    Returns:
        grids: list of frame grids
    """
    # Get list of image files in the directory
    image_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]
                         , key=numerical_sort)
    
    if grid_count == -1:
        frame_count = len(image_files)
    else:
        frame_count = min(grid_count * pad * grid_size**2, len(image_files))

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    max_grid_area = 512*512* grid_size**2
    grids = []
    frames = []
    grid_index = 0 # for saving grid

    total_grid = grid_size**2
    for idx in range(frame_count):
        if idx % pad == 0:
            image = Image.open(image_files[idx])
            gray_img = image.convert('L')
            
            gray_img = np.array(gray_img)
            
            # mask blur
            if mask_blur_factor != -1:
                kernel = np.ones((30, 30), np.uint8) # 30
                dilation = cv.dilate(gray_img, kernel, iterations=1)
                
                # Apply Gaussian blur using OpenCV
                gray_img = cv.GaussianBlur(dilation, (15, 15), 0) # 15
                
                
                # # do gaussian filter first
                # tmp_gray_img = ipu.pil_blur(gray_img, mask_blur_factor)
                
                # tmp_gray_np = np.array(tmp_gray_img) / 255.0
                # original_gray_np = np.array(gray_img) / 255.0
                
                # first_gray_np = np.where(tmp_gray_np == 0, 0, original_gray_np)
                # first_gray_img = Image.fromarray((first_gray_np * 255).astype(np.uint8))
                
                # # do gaussian filter twice
                # gray_img = ipu.pil_blur(first_gray_img, mask_blur_factor)
                
            if border_size != -1:
                ## YP : Add border to the image
                # Convert numpy array to PIL image
                gray_img = Image.fromarray(gray_img)
                gray_img = add_black_border(gray_img, border_size)
            
            # gray_img = np.transpose(gray_img, (2, 0, 1))
            frames.append(transform(gray_img))
            
            if len(frames) == total_grid:
                grid = make_grid(frames, nrow=grid_size, padding=0)
                pil_image = (to_pil_image(grid))
                w,h = pil_image.size
                a = float(np.sqrt((w*h/max_grid_area)))
                w1 = int((w//a)//(grid_size*8))*grid_size*8
                h1 = int((h//a)//(grid_size*8))*grid_size*8
                pil_image= pil_image.resize((w1, h1))
                grids.append(pil_image)
                
                print(f'[vgu] gray resize : w,h = ({w}, {h}) ; w1,h1 = ({w1}, {h1})')
                
                # Save the grid image
                save_file = os.path.join(grid_save_path, f'grid_{grid_index}.png')
                pil_image.save(save_file)
                grid_index += 1

                frames = []

    return grids # list of frames