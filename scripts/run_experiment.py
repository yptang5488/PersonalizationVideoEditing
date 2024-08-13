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

import warnings

warnings.filterwarnings("ignore")

import numpy as np


def init_device():
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    return device

def init_paths(input_ns):
    if input_ns.save_folder == None or input_ns.save_folder == '':
        input_ns.save_folder = input_ns.video_name.replace('.mp4', '').replace('.gif', '')
    else:
        input_ns.save_folder += f"/{input_ns.video_name.replace('.mp4', '').replace('.gif', '')}"
    save_dir = f'{const.OUTPUT_PATH}/{input_ns.save_folder}'
    os.makedirs(save_dir, exist_ok=True)
    save_idx = max([int(x[-5:]) for x in os.listdir(save_dir)])+1 if os.listdir(save_dir) != [] else 0
    input_ns.save_path = f'{save_dir}/{input_ns.positive_prompts}-{str(save_idx).zfill(5)}'
    

    input_ns.video_path = f'{const.MP4_PATH}/{input_ns.video_name}.mp4'
    
    if '-' in input_ns.preprocess_name:
        input_ns.hf_cn_path = [const.PREPROCESSOR_DICT[i] for i in input_ns.preprocess_name.split('-')]
    else:
        input_ns.hf_cn_path = const.PREPROCESSOR_DICT[input_ns.preprocess_name]
    input_ns.hf_path = "runwayml/stable-diffusion-v1-5"
    
    # original (for golden retrieval puppy without border)
    # input_ns.inverse_path = f'{const.GENERATED_DATA_PATH}/inverses/{input_ns.video_name}/{input_ns.preprocess_name}_{input_ns.model_id}_{input_ns.grid_size}x{input_ns.grid_size}_{input_ns.pad}'
    # input_ns.control_path = f'{const.GENERATED_DATA_PATH}/controls/{input_ns.video_name}/{input_ns.preprocess_name}_{input_ns.grid_size}x{input_ns.grid_size}_{input_ns.pad}'
    
    ### border case inversion (for golden retrieval puppy with border)
    # input_ns.inverse_path = '/home/cgvsl/P76111131/RAVE/generated/data/inverses/dog_wind_30f/softedge_hed_None_3x3_1'

    # # original
    if input_ns.border_size == -1: # no use border
        input_ns.inverse_path = (
            f'{const.GENERATED_DATA_PATH_new}/inverses/{input_ns.video_name}/'
            f'{input_ns.preprocess_name}_{input_ns.model_id}_{input_ns.grid_size}x{input_ns.grid_size}_B{input_ns.border_size}/'
            f'lora{input_ns.lora_name}_inversion{input_ns.num_inversion_step}'
        )
    else: # use different border
        input_ns.inverse_path = (
            f'{const.GENERATED_DATA_PATH_new}/inverses/{input_ns.video_name}/'
            f'{input_ns.preprocess_name}_{input_ns.model_id}_{input_ns.grid_size}x{input_ns.grid_size}_B{input_ns.border_type}{input_ns.border_size}/'
            f'lora{input_ns.lora_name}_inversion{input_ns.num_inversion_step}'
        )
        
    input_ns.control_path = (
        f'{const.GENERATED_DATA_PATH_new}/controls/{input_ns.video_name}/'
        f'{input_ns.preprocess_name}_{input_ns.grid_size}x{input_ns.grid_size}_B{input_ns.border_size}'
    )
    
    os.makedirs(input_ns.control_path, exist_ok=True)
    os.makedirs(input_ns.inverse_path, exist_ok=True)
    
    os.makedirs(input_ns.save_path, exist_ok=True)
    
    #### original ####
    input_ns.original_base_path = f'{input_ns.save_path}/original'
    # for image grid saving
    input_ns.origrid_save_path = f'{input_ns.original_base_path}/origrids'
    os.makedirs(input_ns.origrid_save_path, exist_ok=True)
    
    #### control ####
    input_ns.ctrl_base_path = f'{input_ns.save_path}/control'
    # for control frames saving
    input_ns.ctrlframes_save_path = f'{input_ns.ctrl_base_path}/controlframes'
    os.makedirs(input_ns.ctrlframes_save_path, exist_ok=True)

    #### result ####
    input_ns.result_base_path = f'{input_ns.save_path}/result'
    # for result frames saving
    input_ns.resultframes_save_path = f'{input_ns.result_base_path}/resultframes'
    os.makedirs(input_ns.resultframes_save_path, exist_ok=True)
    # for result grids saving
    input_ns.resultgrid_save_path = f'{input_ns.result_base_path}/resultgirds'
    os.makedirs(input_ns.resultgrid_save_path, exist_ok=True)
    
    return input_ns

def set_condition_pil_list(input_ns):
    if input_ns.control_image_path_1 != 'None':
        # for control grid saving
        input_ns.ctrlgrid_save_path = f'{input_ns.ctrl_base_path}/controlgrids'
        os.makedirs(input_ns.ctrlgrid_save_path, exist_ok=True)
        
        input_ns.ctrlimage_pil_list_1 = vgu.prepare_frames_to_grid(
                                            input_ns.control_image_path_1,
                                            input_ns.sample_size,
                                            input_ns.grid_size,
                                            input_ns.pad,
                                            input_ns.ctrlgrid_save_path,
                                            input_ns.border_size,
                                            input_ns.border_type,
                                        )
    else:
        input_ns.ctrlimage_pil_list_1 = []
        
    if input_ns.control_image_path_2 != 'None':
        # for control grid saving
        input_ns.ctrlgrid_save_path = f'{input_ns.ctrl_base_path}/controlgrids'
        os.makedirs(input_ns.ctrlgrid_save_path, exist_ok=True)
        
        input_ns.ctrlimage_pil_list_2 = vgu.prepare_frames_to_grid(
                                            input_ns.control_image_path_2,
                                            input_ns.sample_size,
                                            input_ns.grid_size,
                                            input_ns.pad,
                                            input_ns.ctrlgrid_save_path,
                                            input_ns.border_size,
                                            input_ns.border_type,
                                        )
    else:
        input_ns.ctrlimage_pil_list_2 = []
        
    print(f'[INFO] len(input_ns.ctrlimage_pil_list_1) = {len(input_ns.ctrlimage_pil_list_1)}')
    print(f'[INFO] len(input_ns.ctrlimage_pil_list_2) = {len(input_ns.ctrlimage_pil_list_2)}')
    
    if input_ns.enable_mask and input_ns.mask_image_path != 'None':
        # for mask grid saving
        input_ns.maskgrid_save_path = f'{input_ns.ctrl_base_path}/maskgrids'
        os.makedirs(input_ns.maskgrid_save_path, exist_ok=True)
        
        input_ns.mask_pil_list = vgu.prepare_frames_to_grid_gray(
                                    input_ns.mask_image_path,
                                    input_ns.sample_size,
                                    input_ns.grid_size,
                                    input_ns.pad,
                                    input_ns.maskgrid_save_path,
                                    input_ns.mask_blur_factor,
                                    input_ns.border_size,
                                )
    else:
        input_ns.mask_pil_list = []
    
    
def run(input_ns):
    # TODO : check if it's correct
    optional_config = ['model_id', 'lora_path', 'control_image_path_1', 'control_image_path_2', 'mask_image_path', 'fps']
    print(f'input_ns.control_image_path_1 = {input_ns.control_image_path_1}') #### check control_image_path_1 and 2 是啥？？？？？？
    print(f'input_ns.control_image_path_2 = {input_ns.control_image_path_2}')
    
    for op_config in optional_config:
        if op_config not in list(input_ns.__dict__.keys()):
            setattr(input_ns, op_config, "None")
        if getattr(input_ns, op_config) == '':
            setattr(input_ns, op_config, "None")
            
            
    device = init_device()
    input_ns = init_paths(input_ns)

    ## read video and put them into pil grid
    input_ns.image_pil_list = vgu.prepare_video_to_grid(
        input_ns.video_path,
        input_ns.sample_size,
        input_ns.grid_size,
        input_ns.pad,
        input_ns.origrid_save_path,
        input_ns.border_size,
        input_ns.border_type,
    )
    input_ns.sample_size = len(input_ns.image_pil_list)
    print(f'Frame count: {len(input_ns.image_pil_list)}')
    
    ## if there are control images and mask
    set_condition_pil_list(input_ns)
        
    controlnet_class = RAVE_MultiControlNet if '-' in str(input_ns.controlnet_conditioning_scale) else RAVE
    if '-' in str(input_ns.controlnet_conditioning_scale):
        print(f'[INFO] controlnet_class = RAVE_MultiControlNet')
    else:
        print(f'[INFO] controlnet_class = RAVE')
    
    CN = controlnet_class(device)
    CN.init_models(input_ns.hf_cn_path, input_ns.hf_path, input_ns.preprocess_name, input_ns.model_id, input_ns.lora_path, input_ns.lora_scale)
    
    input_dict = vars(input_ns)
    keys_to_exclude = ['image_pil_list', 'ctrlimage_pil_list_1', 'ctrlimage_pil_list_2', 'mask_pil_list']
    yaml_dict = {k:v for k,v in input_dict.items() if k != 'image_pil_list' and k not in keys_to_exclude}

    # fps = 8

    video_fps = 11 # default fps
    if isinstance(input_ns.fps, int):
        video_fps = input_ns.fps
    
    start_time = datetime.datetime.now()
    # save_name = f"{'-'.join(input_ns.positive_prompts.split())}_cstart-{input_ns.controlnet_guidance_start}_gs-{input_ns.guidance_scale}_pre-{'-'.join((input_ns.preprocess_name.replace('-','+').split('_')))}_cscale-{input_ns.controlnet_conditioning_scale}_grid-{input_ns.grid_size}_pad-{input_ns.pad}_model-{input_ns.model_id.split('/')[-1]}"
    save_name = f"{'-'.join(input_ns.positive_prompts.split())}"
    if '-' in str(input_ns.controlnet_conditioning_scale):
        res_vid, control_vid_1, control_vid_2 = CN(input_dict)
        control_vid_1[0].save(f"{input_ns.save_path}/control1_{save_name}.gif",
                              save_all=True,
                              append_images=control_vid_1[1:],
                              optimize=False,
                              duration=int(1000/video_fps),
                              loop=0)
        control_vid_2[0].save(f"{input_ns.save_path}/control2_{save_name}.gif", 
                              save_all=True,
                              append_images=control_vid_2[1:],
                              optimize=False,
                              duration=int(1000/video_fps),
                              loop=0)
        
        # saveing frames and video
        for i, f in enumerate(control_vid_1):
            f.save(f"{input_ns.ctrlframes_save_path}/ctrl1_{i}.jpg")
        for i, f in enumerate(control_vid_2):
            f.save(f"{input_ns.ctrlframes_save_path}/ctrl2_{i}.jpg")
    else: 
        res_vid, control_vid = CN(input_dict)
        control_vid[0].save(f"{input_ns.save_path}/control_{save_name}.gif",
                            save_all=True,
                            append_images=control_vid[1:],
                            optimize=False,
                            duration=int(1000/video_fps),
                            loop=0)
        
        # saveing frames and video
        for i, f in enumerate(control_vid):
            f.save(f"{input_ns.ctrlframes_save_path}/{i}.jpg")
    end_time = datetime.datetime.now()
    
    ## saving result frames and video ##
    for i, f in enumerate(res_vid):
        ## YP : remove border ##
        if input_ns.border_size != -1:
            res_vid[i] = vgu.remove_border_and_resize(f, input_ns.border_size)
        res_vid[i].save(f"{input_ns.resultframes_save_path}/{i}.jpg")
        
    res_vid[0].save(f"{input_ns.save_path}/{save_name}.gif",
                    save_all=True, 
                    append_images=res_vid[1:], 
                    optimize=False,
                    duration=int(1000/video_fps),
                    loop=0)
    video_save_path = f"{input_ns.save_path}/{save_name}.mp4"
    ipu.pil_images_to_video(res_vid, video_save_path, video_fps)
    
    # TODO : saving original frame to video (alignment fps)
    # DONE : (post-process) saving result grids
    vgu.prepare_frames_to_grid(input_ns.resultframes_save_path,
                               input_ns.sample_size,
                               input_ns.grid_size,
                               input_ns.pad,
                               input_ns.resultgrid_save_path,
                               -1
                            )

    ## saving yaml ##
    yaml_dict['total_time'] = (end_time - start_time).total_seconds()
    yaml_dict['total_number_of_frames'] = len(res_vid)
    yaml_dict['sec_per_frame'] = yaml_dict['total_time']/yaml_dict['total_number_of_frames']
    with open(f'{input_ns.save_path}/config.yaml', 'w') as yaml_file:
        yaml.dump(yaml_dict, yaml_file)
        

if __name__ == '__main__':
    config_path = sys.argv[1]
    # config_path = '/home/cgvsl/P76111131/RAVE/configs/run-turnhead.yaml'
    input_dict_list = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    list_vals = []
    list_keys = []
    for key in input_dict_list.keys():
        if type(input_dict_list[key]) is list:
            list_vals.append(input_dict_list[key])
            list_keys.append(key)

    input_dict_list_temp = {k:v for k,v in input_dict_list.items() if k not in list_keys}        
    for item in list(itertools.product(*list_vals)):
        input_dict_list_temp.update({list_keys[i]:item[i] for i in range(len(list_keys))})

        input_ns = argparse.Namespace(**input_dict_list_temp)
        run(input_ns)
    