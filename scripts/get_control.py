import torch
import argparse
import os
import itertools
import sys
import yaml
import datetime
sys.path.append(os.getcwd())
from pipelines.sd_controlnet_rave import RAVE
from pipelines.sd_multicontrolnet_rave import RAVE_MultiControlNet

import utils.constants as const
import utils.video_grid_utils as vgu
import utils.image_process_utils as ipu
import utils.preprocesser_utils as pu

import warnings
import cv2
# import PIL
from PIL import Image

warnings.filterwarnings("ignore")

import numpy as np


def init_device():
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    return device

def init_paths(input_ns):
    input_ns.video_path = f'{const.MP4_PATH}/{input_ns.video_name}.mp4'
    
    if '-' in input_ns.preprocess_name:
        input_ns.hf_cn_path = [const.PREPROCESSOR_DICT[i] for i in input_ns.preprocess_name.split('-')]
    else:
        input_ns.hf_cn_path = const.PREPROCESSOR_DICT[input_ns.preprocess_name]
    input_ns.hf_path = "runwayml/stable-diffusion-v1-5"
    
    input_ns.root_save_path = 'generated/controlframes'
    input_ns.control_path = (
        f'{root_save_path}/{input_ns.video_name}/'
        f'{input_ns.preprocess_name}'
    )
    
    os.makedirs(control_path, exist_ok=True)
    return input_ns

# Function to check variable type
def check_variable_type(variable):
    if isinstance(variable, np.ndarray):
        return "NumPy array"
    elif isinstance(variable, Image.Image):
        return "PIL image"
    elif isinstance(variable, torch.Tensor):
        return "PyTorch tensor"
    else:
        return "Unknown type"

def preprocess_control_frame(image_pil, preprocess_name):

    # list_of_image_pils = fu.pil_grid_to_frames(image_pil, grid_size=self.grid) # List[C, W, H] -> len = num_frames
    # list_of_pils = [pu.pixel_perfect_process(np.array(frame_pil, dtype='uint8'), preprocess_name) for frame_pil in image_pil_list]
    control_map = pu.pixel_perfect_process(np.array(image_pil, dtype='uint8'), preprocess_name)
    
    print(f'[output type check] check_variable_type(detected_map) = {check_variable_type(control_map)}')
    print(f'[output type check] list_of_pils[0].shape = {control_map.shape}')
    
    # 將 NumPy 數組轉換為 PIL 圖像
    control_img = Image.fromarray(control_map)
    
    # control_images = np.array(list_of_pils)
    # control_img = ipu.create_grid_from_numpy(control_images, grid_size=self.grid)
    # control_img = PIL.Image.fromarray(control_img).convert("L")

    return control_img
    
def process_image_batch(image_pil_list):
    width, height = image_pil_list[0].size
    
    image_torch_list = []
    control_pil_list = []
    control_torch_list = []
    
    print(f'[INFO] video frames number', len(image_pil_list))
    for i, image_pil in enumerate(image_pil_list):
        print(f'{i}th frame : [{width},{height}]', end='\n')
        # width, height = image_pil.size

        control_pil = self.preprocess_control_grid(image_pil)
        control_pil_list.append(control_pil)
        
        ## for video frames
        image_torch_list.append(ipu.pil_img_to_torch_tensor(image_pil))
        
    # control_torch_list = [self.prepare_control_image(control_pil, width, height) for control_pil in control_pil_list]

    # print(f'start to save control.pt')
    control_torch = torch.cat(control_torch_list, dim=0).to(self.device) # [grid_count, 3, h*grid_size, w*grid_size]
    img_torch = torch.cat(image_torch_list, dim=0).to(self.device) # [grid_count, 3, h*grid_size, w*grid_size]

    ## save img and control .pt ##
    # print(f"[INFO] Run image and control patch then save!")
    torch.save(control_torch, os.path.join(self.controls_path, 'control.pt'))
    print(f"[INFO] save {os.path.join(self.controls_path, 'control.pt')} successfully!")
    torch.save(img_torch, os.path.join(self.controls_path, 'img.pt'))
    print(f"[INFO] save {os.path.join(self.controls_path, 'img.pt')} successfully!")
    
    print(f'img_torch.shape = {img_torch.shape}')
    print(f'control_torch.shape = {control_torch.shape}')
    return img_torch, control_torch, mask_torch


def save_image_with_numbering(image, base_path, number):
    # 格式化文件名，使用六位數字，前綴0
    filename = f"{base_path}/{number:06d}.jpg"
    image.save(filename, format='JPEG')

def process_control_list(image_pil_list, save_path, preprocessor):
    width, height = image_pil_list[0].size
    
    control_pil_list = []
    
    print(f'[INFO] video frames number', len(image_pil_list))
    if len(os.listdir(save_path)) == len(image_pil_list):
        print('[INFO] there are already control frames in the directory !')
        return control_pil_list
        
    for i, image_pil in enumerate(image_pil_list):
        print(f'{i}th frame : [{width},{height}]', end='\n')
        # width, height = image_pil.size

        control_pil = preprocess_control_frame(image_pil, preprocessor)
        save_image_with_numbering(control_pil, save_path, i)
        
        control_pil_list.append(control_pil)
        
        widthc, heightc = control_pil.size
        print(f'{i}th frame : [{widthc},{heightc}]', end='\n')
        
    return control_pil_list

def prepare_video_to_frames(path, format='gif'):
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

def read_video_frames(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    frames = []
    while True:
        success, frame = video.read()
        if not success:
            break
        
        # Convert the frame from BGR (OpenCV format) to RGB (PIL format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert the frame to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        frames.append(pil_image)
    
    # # if not cap.isOpened():
    # #     print("Error: Could not open video.")
    # #     return []
    
    # frames = []
    # while True:
    #     # Read a frame from the video
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
        
    #     # Convert the frame from BGR (OpenCV format) to RGB (PIL format)
    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    #     # Convert the frame to PIL Image
    #     pil_image = Image.fromarray(frame_rgb)
        
    #     frames.append(pil_image)
    
    # Release the video capture object
    video.release()
    
    return frames

def read_gif_frames(gif_path):
    # Open the GIF file
    gif = Image.open(gif_path)
    
    frames = []
    try:
        while True:
            # Append the current frame
            frames.append(gif.copy())
            # Move to the next frame
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    
    return frames

def read_frames(file_path):
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension in ['.mp4', '.avi', '.mov']:
        return read_video_frames(file_path)
    elif file_extension in ['.gif']:
        return read_gif_frames(file_path)
    else:
        print("Unsupported file format.")
        return []
    

'''
preprocessors_dict = {
    'lineart_realistic': lineart,
    'lineart_coarse': lineart_coarse,
    'lineart_standard': lineart_standard,
    'lineart_anime': lineart_anime,
    'lineart_anime_denoise': lineart_anime_denoise,
    'softedge_hed': hed,
    'softedge_hedsafe': hed_safe,
    'softedge_pidinet': pidinet,
    'softedge_pidsafe': pidinet_safe,
    'canny': canny,
    'depth_leres': leres,
    'depth_leres++': lerespp,
    'depth_midas': midas,
    'depth_zoe': zoe_depth,
}
'''
if __name__ == '__main__':
    ### filled in
    video_name = 'fox_turn_head_27_fps11'
    proprocessor = 'canny'
    ###
    
    save_path = f'generated/controlframes/{video_name}/{proprocessor}'
    os.makedirs(save_path, exist_ok=True)
    ctrl_video_save_path = f'generated/controlframes/{video_name}/0'
    os.makedirs(ctrl_video_save_path, exist_ok=True)
    ctrl_video_save_name = f'{video_name}_{proprocessor}'
    
    video_base_path = 'data/mp4_videos'
    video_path = f'{video_base_path}/{video_name}.mp4'
    frames = read_frames(video_path)
    print(f'frames length = {len(frames)}')
    
    control_pil_list = process_control_list(frames, save_path, proprocessor)
    
    if len(control_pil_list) != 0:
        video_fps = 11
        control_pil_list[0].save(f"{ctrl_video_save_path}/{ctrl_video_save_name}.gif",
                        save_all=True, 
                        append_images=control_pil_list[1:], 
                        optimize=False,
                        duration=int(1000/video_fps),
                        loop=0)
        video_save_path = f"{ctrl_video_save_path}/{ctrl_video_save_name}.mp4"
        ipu.pil_images_to_video(control_pil_list, video_save_path, video_fps)
    
    
    
    
    
    
    
    
    # config_path = sys.argv[1]
    # # config_path = '/home/cgvsl/P76111131/RAVE/configs/run-multicontrolnet-turnhead.yaml'
    # input_dict_list = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    # list_vals = []
    # list_keys = []
    # for key in input_dict_list.keys():
    #     if type(input_dict_list[key]) is list:
    #         list_vals.append(input_dict_list[key])
    #         list_keys.append(key)

    # input_dict_list_temp = {k:v for k,v in input_dict_list.items() if k not in list_keys}        
    # for item in list(itertools.product(*list_vals)):
    #     input_dict_list_temp.update({list_keys[i]:item[i] for i in range(len(list_keys))})

    #     input_ns = argparse.Namespace(**input_dict_list_temp)
    #     run(input_ns)
    