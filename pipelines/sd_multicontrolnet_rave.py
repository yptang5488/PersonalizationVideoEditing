import random
import os
import PIL
import torch
import warnings

warnings.filterwarnings("ignore")

from transformers import set_seed
from tqdm import tqdm
from transformers import logging
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, DDIMScheduler

import torch.nn as nn
import numpy as np

import utils.constants as const
import utils.feature_utils as fu
import utils.preprocesser_utils as pu
import utils.image_process_utils as ipu


logging.set_verbosity_error()

def set_seed_lib(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    set_seed(seed)

@torch.no_grad()
class RAVE_MultiControlNet(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device
        self.dtype = torch.float

    @torch.no_grad()
    def __init_pipe(self, hf_cn_path, hf_path, lora_path, lora_scale):
        controlnet_1 = ControlNetModel.from_pretrained(hf_cn_path[0], torch_dtype=self.dtype).to(self.device, self.dtype)
        controlnet_2 = ControlNetModel.from_pretrained(hf_cn_path[1], torch_dtype=self.dtype).to(self.device, self.dtype)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(hf_path, controlnet=[controlnet_1, controlnet_2], torch_dtype=self.dtype).to(self.device, self.dtype) 
        
        ### lora loading
        if lora_path != 'None':
            pipe.load_lora_weights(lora_path)
        
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()
        return pipe
        
    @torch.no_grad()
    def init_models(self, hf_cn_path, hf_path, preprocess_name, model_id=None, lora_path=None, lora_scale=0.0):
        print(f'lora_path check : {lora_path}')
        if lora_path == 'None':
            print(f'[INFO] no LoRA using')
        else:
            print(f'[INFO] LoRA using; path={lora_path}; scale={lora_scale}')
            
        if model_id is None or model_id == "None":
            pipe = self.__init_pipe(hf_cn_path, hf_path, lora_path, lora_scale)  
        else:
            pipe = self.__init_pipe(hf_cn_path, model_id, lora_path, lora_scale)  
        self.preprocess_name_1, self.preprocess_name_2 = preprocess_name.split('-')
        
        
        self._prepare_control_image = pipe.prepare_control_image
        self.run_safety_checker = pipe.run_safety_checker
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder

        self.vae = pipe.vae
        self.unet = pipe.unet    

        self.controlnet = pipe.controlnet
        self.scheduler_config = pipe.scheduler.config        
        
        del pipe
                

    
    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        cond_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        cond_embeddings = self.text_encoder(cond_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        return cond_embeddings, uncond_embeddings

    @torch.no_grad()
    def prepare_control_image(self, control_pil, width, height):

        control_image = self._prepare_control_image(
            image=control_pil,
            width=width,
            height=height,
            device=self.device,
            dtype=self.controlnet.dtype,
            batch_size=1,
            num_images_per_prompt=1
        )

        return control_image
    
    @torch.no_grad()
    # modified from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint.StableDiffusionInpaintPipeline.prepare_mask_latents
    def prepare_mask(
        self,
        mask_pil,
        height,
        width,
    ):
        # print(f'[prepare_mask] check!!!! {np.array(mask_pil).shape}')
        mask = ipu.pil_img_to_torch_tensor_grayscale(mask_pil)
        
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=self.device, dtype=self.dtype)

        return mask
    
    @torch.no_grad()
    def pred_controlnet_sampling(self, current_sampling_percent, latent_model_input, t, text_embeddings, control_image):
        if (current_sampling_percent < self.controlnet_guidance_start or current_sampling_percent > self.controlnet_guidance_end):
            down_block_res_samples = None
            mid_block_res_sample = None
        else:
            
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                t,
                conditioning_scale=self.controlnet_conditioning_scale,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=control_image,
                return_dict=False,
            )
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings,                    
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample)['sample']
        return noise_pred
    
    
    @torch.no_grad()
    def denoising_step(self, latents, control_image_1, control_image_2, text_embeddings, t, guidance_scale, current_sampling_percent):
        ## do classifier_free_guidance here
        latent_model_input = torch.cat([latents] * 2)
        control_image_1 = torch.cat([control_image_1] * 2)
        control_image_2 = torch.cat([control_image_2] * 2)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        
        
        noise_pred = self.pred_controlnet_sampling(current_sampling_percent, latent_model_input, t, text_embeddings, [control_image_1, control_image_2])

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        return latents

    @torch.no_grad() ## YP
    def denoising_step_with_mask(self, latents, control_image_1, control_image_2, mask, init_latents_pre, text_embeddings, t, i, guidance_scale, current_sampling_percent):
        ## do classifier_free_guidance here
        latent_model_input = torch.cat([latents] * 2)
        control_image_1 = torch.cat([control_image_1] * 2)
        control_image_2 = torch.cat([control_image_2] * 2)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        noise_pred = self.pred_controlnet_sampling(current_sampling_percent, latent_model_input, t, text_embeddings, [control_image_1, control_image_2])

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        if t > self.scheduler.timesteps[-1]:
            noise_timestep = self.scheduler.timesteps[i+1]
            
            tmp_batch_size = init_latents_pre.shape[0]
            batch_noise = self.noise.repeat(tmp_batch_size, 1,1,1)
            # print(f'#### init_latents_pre.shape={init_latents_pre.shape}')
            # print(f'#### self.noise.shape={self.noise.shape}')
            init_latents_noise = self.scheduler.add_noise(init_latents_pre, batch_noise, torch.tensor([noise_timestep])) # LAST UPDATE : add same noise!

            latents = (1 - mask) * init_latents_noise + mask * latents ## mask blending ##
        
        return latents

    @torch.no_grad()
    def preprocess_control_grid_single(self, image_pil, preprocess_name):
        """get preprocess control image and make them into grid
        
        Args:
            image_pil (PIL.Image): 

        Returns:
            control_img_1, control_img_2: tuple of contrl image
        """
        list_of_image_pils = fu.pil_grid_to_frames(image_pil, grid_size=self.grid) # List[C, W, H] -> len = num_frames
        list_of_pils = [pu.pixel_perfect_process(np.array(frame_pil, dtype='uint8'), preprocess_name) for frame_pil in list_of_image_pils]
        control_images = np.array(list_of_pils)
        control_img = ipu.create_grid_from_numpy(control_images, grid_size=self.grid)
        control_img = PIL.Image.fromarray(control_img).convert("L")

        return control_img


    @torch.no_grad()
    def preprocess_control_grid(self, image_pil):
        """get preprocess control image and make them into grid
        
        Args:
            image_pil (PIL.Image): 

        Returns:
            control_img_1, control_img_2: tuple of contrl image
        """
        list_of_image_pils = fu.pil_grid_to_frames(image_pil, grid_size=self.grid) # List[C, W, H] -> len = num_frames
        list_of_pils_1, list_of_pils_2 = [], []
        for frame_pil in list_of_image_pils:
            frame_pil_1 = pu.pixel_perfect_process(np.array(frame_pil, dtype='uint8'), self.preprocess_name_1)
            frame_pil_2 = pu.pixel_perfect_process(np.array(frame_pil, dtype='uint8'), self.preprocess_name_2)
            list_of_pils_1.append(frame_pil_1)
            list_of_pils_2.append(frame_pil_2)
        control_images_1 = np.array(list_of_pils_1) # control_images_1.shape = [f, h, w]
        control_images_2 = np.array(list_of_pils_2)
        
        control_img_1 = ipu.create_grid_from_numpy(control_images_1, grid_size=self.grid) 
        control_img_1 = PIL.Image.fromarray(control_img_1).convert("L")
        control_img_1.save('control1_output.jpg')
        
        control_img_2 = ipu.create_grid_from_numpy(control_images_2, grid_size=self.grid)
        control_img_2 = PIL.Image.fromarray(control_img_2).convert("L") # pil image to array
        control_img_1.save('control2_output.jpg')
        return control_img_1, control_img_2
    
    @torch.no_grad()
    def shuffle_latents(self, latents, control_image_1, control_image_2, indices):
        rand_i = torch.randperm(self.total_frame_number).tolist()
        # latents, _ = fu.prepare_key_grid_latents(latents, self.grid, self.grid, rand_i)
        # control_image, _ = fu.prepare_key_grid_latents(control_image, self.grid, self.grid, rand_i)
        
        latents_l, controls_l_1, controls_l_2, randx = [], [], [], []
        for j in range(self.sample_size):
            rand_indices = rand_i[j*self.grid_frame_number:(j+1)*self.grid_frame_number]

            latents_keyframe, _ = fu.prepare_key_grid_latents(latents, self.grid, self.grid, rand_indices)
            control_keyframe_1, _ = fu.prepare_key_grid_latents(control_image_1, self.grid, self.grid, rand_indices)
            control_keyframe_2, _ = fu.prepare_key_grid_latents(control_image_2, self.grid, self.grid, rand_indices)
            
            latents_l.append(latents_keyframe)
            controls_l_1.append(control_keyframe_1)
            controls_l_2.append(control_keyframe_2)
            randx.extend(rand_indices)
        rand_i = randx.copy()
        latents = torch.cat(latents_l, dim=0)
        control_image_1 = torch.cat(controls_l_1, dim=0)
        control_image_2 = torch.cat(controls_l_2, dim=0)
        indices = [indices[i] for i in rand_i]
        return latents, indices, control_image_1, control_image_2
    
    @torch.no_grad() ## YP
    def shuffle_latents_with_mask(self, latents, control_image_1, control_image_2, mask, init_latents_pre, indices):
        rand_i = torch.randperm(self.total_frame_number).tolist()
        
        latents_l, controls_l_1, controls_l_2, masks_l, init_latents_l, randx = [], [], [], [], [], []
        for j in range(self.sample_size):
            rand_indices = rand_i[j*self.grid_frame_number:(j+1)*self.grid_frame_number]

            latents_keyframe, _ = fu.prepare_key_grid_latents(latents, self.grid, self.grid, rand_indices)
            control_keyframe_1, _ = fu.prepare_key_grid_latents(control_image_1, self.grid, self.grid, rand_indices)
            control_keyframe_2, _ = fu.prepare_key_grid_latents(control_image_2, self.grid, self.grid, rand_indices)
            mask_keyframe, _ = fu.prepare_key_grid_latents(mask, self.grid, self.grid, rand_indices)
            init_latents_keyframe, _ = fu.prepare_key_grid_latents(init_latents_pre, self.grid, self.grid, rand_indices)
            
            latents_l.append(latents_keyframe)
            controls_l_1.append(control_keyframe_1)
            controls_l_2.append(control_keyframe_2)
            masks_l.append(mask_keyframe)
            init_latents_l.append(init_latents_keyframe)
            randx.extend(rand_indices)
            
        rand_i = randx.copy()
        latents = torch.cat(latents_l, dim=0)
        control_image_1 = torch.cat(controls_l_1, dim=0)
        control_image_2 = torch.cat(controls_l_2, dim=0)
        mask = torch.cat(masks_l, dim=0)
        init_latents_pre = torch.cat(init_latents_l, dim=0)
        indices = [indices[i] for i in rand_i]
        return latents, indices, control_image_1, control_image_2, mask, init_latents_pre
    
    @torch.no_grad()
    def batch_denoise(self, latents, control_image_1, control_image_2, indices, t, guidance_scale, current_sampling_percent):
        latents_l, controls_l_1, controls_l_2 = [], [], []
        
        # split by batch_size ; [(batch_size, :), (batch_size, :), ..., (remainder, :)]
        control_split_1 = control_image_1.split(self.batch_size, dim=0)
        control_split_2 = control_image_2.split(self.batch_size, dim=0)
        latents_split = latents.split(self.batch_size, dim=0)
        
        for idx in range(len(control_split_1)):
            txt_embed = torch.cat([self.uncond_embeddings] * len(latents_split[idx]) + [self.cond_embeddings] * len(latents_split[idx])) 

            # control_split : [batch_size, 4, h//8*grid_size, h//8*grid_size]
            # latents_split : [batch_size, 3, h*grid_size, w*grid_size]
            latents = self.denoising_step(latents_split[idx], control_split_1[idx], control_split_2[idx], txt_embed, t, guidance_scale, current_sampling_percent)
            
            latents_l.append(latents)
            controls_l_1.append(control_split_1[idx])
            controls_l_2.append(control_split_2[idx])
            
        latents = torch.cat(latents_l, dim=0)
        controls_1 = torch.cat(controls_l_1, dim=0)
        controls_2 = torch.cat(controls_l_2, dim=0)
        return latents, indices, controls_1, controls_2
    
    @torch.no_grad() ## YP
    def batch_denoise_with_mask(self, latents, control_image_1, control_image_2, mask, init_latents_pre, indices, t, i, guidance_scale, current_sampling_percent):
        latents_l, controls_l_1, controls_l_2, masks_l, init_latents_l = [], [], [], [], []

        # split by batch_size ; [(batch_size, :), (batch_size, :), ..., (remainder, :)]
        control_split_1 = control_image_1.split(self.batch_size, dim=0)
        control_split_2 = control_image_2.split(self.batch_size, dim=0)
        latents_split = latents.split(self.batch_size, dim=0)
        masks_split = mask.split(self.batch_size, dim=0)
        init_latents_split = init_latents_pre.split(self.batch_size, dim=0)
        
        for idx in range(len(control_split_1)):
            txt_embed = torch.cat([self.uncond_embeddings] * len(latents_split[idx]) + [self.cond_embeddings] * len(latents_split[idx])) 

            # control_split : [batch_size, 4, h//8*grid_size, h//8*grid_size]
            # latents_split : [batch_size, 3, h*grid_size, w*grid_size]
            # masks_split: [batch_size, 1, h*grid_size, w*grid_size] (needed)
            # init_latents : [batch_size, 3, h*grid_size, w*grid_size] (needed)
            latents = self.denoising_step_with_mask(latents_split[idx], control_split_1[idx], control_split_2[idx], masks_split[idx], init_latents_split[idx], txt_embed, t, i, guidance_scale, current_sampling_percent)
            
            latents_l.append(latents)
            controls_l_1.append(control_split_1[idx])
            controls_l_2.append(control_split_2[idx])
            masks_l.append(masks_split[idx])
            init_latents_l.append(init_latents_split[idx])
            
        latents = torch.cat(latents_l, dim=0)
        controls_1 = torch.cat(controls_l_1, dim=0)
        controls_2 = torch.cat(controls_l_2, dim=0)
        masks = torch.cat(masks_l, dim=0)
        init_latents_pre = torch.cat(init_latents_l, dim=0)
        return latents, indices, controls_1, controls_2, masks, init_latents_pre
    
    @torch.no_grad()
    def reverse_diffusion(self, latents=None, control_image_1=None, control_image_2=None, guidance_scale=7.5, indices=None):
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        with torch.autocast('cuda'):

            for i, t in tqdm(enumerate(self.scheduler.timesteps), desc='reverse_diffusion'):
                indices = list(indices)
                current_sampling_percent = i / len(self.scheduler.timesteps)

                if self.is_shuffle:
                    latents, indices, control_image_1, control_image_2 = self.shuffle_latents(latents, control_image_1, control_image_2, indices)
                    print(f'timestep={t} ; indices={indices}')
                    
                if self.cond_step_start < current_sampling_percent:
                    latents, indices, control_image_1, control_image_2 = self.batch_denoise(latents, control_image_1, control_image_2, indices, t, guidance_scale, current_sampling_percent)
                else:
                    latents, indices, control_image_1, control_image_2 = self.batch_denoise(latents, control_image_1, control_image_2, indices, t, 0.0, current_sampling_percent)

        return latents, indices, control_image_1, control_image_2
    
    @torch.no_grad() ## YP
    def reverse_diffusion_with_mask(self, latents=None, control_image_1=None, control_image_2=None, mask=None, init_latents_pre=None, guidance_scale=7.5, indices=None):
        # print(f'>>>>>>> [reverse_diffusion_with_mask] latents.shape={latents.shape} ; init_latents_pre.shape={init_latents_pre.shape}')
        
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        with torch.autocast('cuda'):

            for i, t in tqdm(enumerate(self.scheduler.timesteps), desc='reverse_diffusion'):
                indices = list(indices)
                current_sampling_percent = i / len(self.scheduler.timesteps)

                ## with mask!
                if self.is_shuffle:
                    latents, indices, control_image_1, control_image_2, mask, init_latents_pre = self.shuffle_latents_with_mask(latents, control_image_1, control_image_2, mask, init_latents_pre, indices)
                    print(f'timestep={t} ; indices={indices}')
                    
                if self.cond_step_start < current_sampling_percent:
                    latents, indices, control_image_1, control_image_2, mask, init_latents_pre = self.batch_denoise_with_mask(latents, control_image_1, control_image_2, mask, init_latents_pre, indices, t, i, guidance_scale, current_sampling_percent)
                else:
                    latents, indices, control_image_1, control_image_2, mask, init_latents_pre = self.batch_denoise_with_mask(latents, control_image_1, control_image_2, mask, init_latents_pre, indices, t, i, 0.0, current_sampling_percent)

        return latents, indices, control_image_1, control_image_2, mask, init_latents_pre
    
    
    @torch.no_grad()
    def encode_imgs(self, img_torch):
        latents_l = []
        splits = img_torch.split(self.batch_size_vae, dim=0)
        for split in splits:
            image = 2 * split - 1
            posterior = self.vae.encode(image).latent_dist
            latents = posterior.mean * self.vae.config.scaling_factor
            latents_l.append(latents)
        

        return torch.cat(latents_l, dim=0)

    @torch.no_grad()
    def decode_latents(self, latents):
        image_l = []
        splits = latents.split(self.batch_size_vae, dim=0)
        for split in splits:
            image = self.vae.decode(split / self.vae.config.scaling_factor, return_dict=False)[0]
            image = (image / 2 + 0.5).clamp(0, 1)
            image_l.append(image)
        return torch.cat(image_l, dim=0)


    @torch.no_grad()
    def controlnet_pred(self, latent_model_input, t, text_embed_input, controlnet_cond):
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embed_input,
            controlnet_cond=controlnet_cond,
            conditioning_scale=self.controlnet_conditioning_scale,
            return_dict=False,
        )
        
        # apply the denoising network
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embed_input,
            cross_attention_kwargs={},
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]
        return noise_pred

    @torch.no_grad()
    def ddim_inversion(self, latents, control_batch_1, control_batch_2, indices):
        # init_latents : [total_frame_number, 4, h//8, w//8]
        # control_batch : [total_frame_number, 3, h, w]
        
        ## load inversion .pt ##
        k = None # index of inversion frame already exists
        els = os.listdir(self.inverse_path) 
        els = [el for el in els if el.endswith('.pt')]
        
        # Load initial latent from .pt files if available
        for k,inv_path in enumerate(sorted(els, key=lambda x: int(x.split('.')[0]))):
            latents[k] = torch.load(os.path.join(self.inverse_path, inv_path)).to(device=self.device)

        # Initialize DDIM scheduler
        self.inverse_scheduler = DDIMScheduler.from_config(self.scheduler_config)
        self.inverse_scheduler.set_timesteps(self.num_inversion_step, device=self.device)
        self.timesteps = reversed(self.inverse_scheduler.timesteps)

        # there already exists the inversion latents of all images
        if k == (latents.shape[0]-1):
            return latents, indices, control_batch_1, control_batch_2
        
        # Prepare conditional embeddings for inversion process
        inv_cond = torch.cat([self.inv_uncond_embeddings] * 1 + [self.inv_cond_embeddings] * 1)[1].unsqueeze(0)
        
        # Iterate over timesteps to do inversion
        for i, t in enumerate(tqdm(self.timesteps)):
            
            alpha_prod_t = self.inverse_scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (self.inverse_scheduler.alphas_cumprod[self.timesteps[i - 1]] if i > 0 else self.inverse_scheduler.final_alpha_cumprod)
            
            # Split latents and control_batch into single frame for processing
            latents_l = []
            latents_split = latents.split(self.batch_size, dim=0)
            control_batch_split_1 = control_batch_1.split(self.batch_size, dim=0)
            control_batch_split_2 = control_batch_2.split(self.batch_size, dim=0)
            
            # Process each batch using DDIM inversion step
            for idx in range(len(latents_split)):
                cond_batch = inv_cond.repeat(latents_split[idx].shape[0], 1, 1)
                latents = self.ddim_step(latents_split[idx], t, cond_batch, alpha_prod_t, alpha_prod_t_prev, control_batch_split_1[idx], control_batch_split_2[idx])
                latents_l.append(latents)
            latents = torch.cat(latents_l, dim=0)
            
        ## save inversion .pt ##
        for k,i in enumerate(latents):
            torch.save(i.detach().cpu(), f'{self.inverse_path}/{str(k).zfill(5)}.pt')
        print(f"[INFO] save inversion in {self.inverse_path} successfully!")
        
        # init_latents : [total_frame_number, 4, h//8, w//8]
        # control_batch : [total_frame_number, 3, h, w]
        return latents, indices, control_batch_1, control_batch_2
    
   
    def ddim_step(self, latent_frames, t, cond_batch, alpha_prod_t, alpha_prod_t_prev, control_batch_1, control_batch_2):
        mu = alpha_prod_t ** 0.5
        mu_prev = alpha_prod_t_prev ** 0.5
        sigma = (1 - alpha_prod_t) ** 0.5
        sigma_prev = (1 - alpha_prod_t_prev) ** 0.5
        if self.give_control_inversion:
            eps = self.controlnet_pred(latent_frames, t, text_embed_input=cond_batch, controlnet_cond=[control_batch_1, control_batch_2])
        else:
            eps = self.unet(latent_frames, t, encoder_hidden_states=cond_batch, return_dict=False)[0]
        pred_x0 = (latent_frames - sigma_prev * eps) / mu_prev
        latent_frames = mu * pred_x0 + sigma * eps
        return latent_frames
    

    def process_image_batch(self, image_pil_list, ctrlimage_pil_list_1, ctrlimage_pil_list_2, mask_pil_list):
        width, height = image_pil_list[0].size
        
        if len(ctrlimage_pil_list_1) != 0 and len(ctrlimage_pil_list_2) != 0: ## 1. 從檔案讀入(兩個一起)
            print(f'[INFO] read img and control from pre-generated directory')
            image_torch_list = []
            control_torch_list_1, control_torch_list_2 = [], []
            
            for i, image_pil in enumerate(image_pil_list):
                
                control_image_1 = self.prepare_control_image(ctrlimage_pil_list_1[i], width, height)
                control_torch_list_1.append(control_image_1)
                
                control_image_2 = self.prepare_control_image(ctrlimage_pil_list_2[i], width, height)
                control_torch_list_2.append(control_image_2)
                
                ## for video frames
                image_torch_list.append(ipu.pil_img_to_torch_tensor(image_pil))
                
            control_torch_1 = torch.cat(control_torch_list_1, dim=0).to(self.device) # [grid_count, 3, h*grid_size, w*grid_size]
            control_torch_2 = torch.cat(control_torch_list_2, dim=0).to(self.device) # [grid_count, 3, h*grid_size, w*grid_size]
            img_torch = torch.cat(image_torch_list, dim=0).to(self.device) # [grid_count, 3, h*grid_size, w*grid_size]
        elif len(os.listdir(self.controls_path)) > 0: ## 2. 從 .pt 讀取
            control_torch_1 = torch.load(os.path.join(self.controls_path, 'control_1.pt')).to(self.device)
            control_torch_2 = torch.load(os.path.join(self.controls_path, 'control_2.pt')).to(self.device)
            img_torch = torch.load(os.path.join(self.controls_path, 'img.pt')).to(self.device)
        else: ## 3. 經過 preprocessor 處理，再存成 .pt
            image_torch_list = []
            control_torch_list_1, control_torch_list_2 = [], []
            
            print(f'[INFO] video frames number', len(image_pil_list))
            for i, image_pil in enumerate(image_pil_list):
                print(f'{i}th frame : [{width},{height}]', end='\n')
                # control_pil = PIL.Image.fromarray(pu.pixel_perfect_process(np.array(image_pil, dtype='uint8'), self.preprocess_name))
                
                ## for control frames 1
                # if len(ctrlimage_pil_list_1) == 0:
                
                control_pil = self.preprocess_control_grid_single(image_pil, self.preprocess_name_1)
                control_image_1 = self.prepare_control_image(control_pil, width, height)
                
                # else:
                #     control_image_1 = self.prepare_control_image(ctrlimage_pil_list_1[i], width, height)
                    
                ## for control frames 2
                # if len(ctrlimage_pil_list_2) == 0:
                
                control_pil = self.preprocess_control_grid_single(image_pil, self.preprocess_name_2)
                control_image_2 = self.prepare_control_image(control_pil, width, height)
                
                # else:
                #     control_image_2 = self.prepare_control_image(ctrlimage_pil_list_2[i], width, height)
                
                control_torch_list_1.append(control_image_1)
                control_torch_list_2.append(control_image_2)
                
                ## for video frames
                image_torch_list.append(ipu.pil_img_to_torch_tensor(image_pil))
                
            control_torch_1 = torch.cat(control_torch_list_1, dim=0).to(self.device)
            control_torch_2 = torch.cat(control_torch_list_2, dim=0).to(self.device)
            img_torch = torch.cat(image_torch_list, dim=0).to(self.device)
            
            ## save img and control .pt ##
            torch.save(control_torch_1, os.path.join(self.controls_path, 'control_1.pt'))
            torch.save(control_torch_2, os.path.join(self.controls_path, 'control_2.pt'))
            torch.save(img_torch, os.path.join(self.controls_path, 'img.pt'))
            
        ## for mask
        mask_torch_list = []
        mask_torch = None
        if self.enable_mask:
            for i, mask_pil in enumerate(mask_pil_list):
                mask_torch_list.append(self.prepare_mask(mask_pil, height, width))
                
            mask_torch = torch.cat(mask_torch_list, dim=0).to(self.device)
            # print(f'mask_torch.shape = {mask_torch.shape}')    
        
        return img_torch, control_torch_1, control_torch_2, mask_torch
        
    def order_grids(self, list_of_pils, indices):
        k = []
        for i in range(len(list_of_pils)):
            k.extend(fu.pil_grid_to_frames(list_of_pils[i], self.grid))
            
        frames = [k[indices.index(i)] for i in np.arange(len(indices))]    
        return frames


    @torch.no_grad()
    def __preprocess_inversion_input(self, init_latents, control_batch_1, control_batch_2):
        list_of_flattens = [fu.flatten_grid(el.unsqueeze(0), self.grid) for el in init_latents]
        init_latents = torch.cat(list_of_flattens, dim=-1)
        init_latents = torch.cat(torch.chunk(init_latents, self.total_frame_number, dim=-1), dim=0)
        
        control_batch_flattens_1 = [fu.flatten_grid(el.unsqueeze(0), self.grid) for el in control_batch_1]
        control_batch_1 = torch.cat(control_batch_flattens_1, dim=-1)
        control_batch_1 = torch.cat(torch.chunk(control_batch_1, self.total_frame_number, dim=-1), dim=0)
        
        control_batch_flattens_2 = [fu.flatten_grid(el.unsqueeze(0), self.grid) for el in control_batch_2]
        control_batch_2 = torch.cat(control_batch_flattens_2, dim=-1)
        control_batch_2 = torch.cat(torch.chunk(control_batch_2, self.total_frame_number, dim=-1), dim=0)
        
        return init_latents, control_batch_1, control_batch_2
    
    
    @torch.no_grad()
    def __postprocess_inversion_input(self, latents_inverted, control_batch_1, control_batch_2):
            latents_inverted = torch.cat([fu.unflatten_grid(torch.cat([a for a in latents_inverted[i*self.grid_frame_number:(i+1)*self.grid_frame_number]], dim=-1).unsqueeze(0), self.grid) for i in range(self.sample_size)] , dim=0)
            control_batch_1 = torch.cat([fu.unflatten_grid(torch.cat([a for a in control_batch_1[i*self.grid_frame_number:(i+1)*self.grid_frame_number]], dim=-1).unsqueeze(0), self.grid) for i in range(self.sample_size)] , dim=0)
            control_batch_2 = torch.cat([fu.unflatten_grid(torch.cat([a for a in control_batch_2[i*self.grid_frame_number:(i+1)*self.grid_frame_number]], dim=-1).unsqueeze(0), self.grid) for i in range(self.sample_size)] , dim=0)
            return latents_inverted, control_batch_1, control_batch_2
        
    
    
    
    @torch.no_grad()
    def __call__(self, input_dict):
        set_seed_lib(input_dict['seed'])
        self.seed = input_dict['seed']
        
        self.grid_size = input_dict['grid_size']
        self.sample_size = input_dict['sample_size']
        
        self.grid_frame_number = self.grid_size * self.grid_size
        self.total_frame_number = (self.grid_frame_number) * self.sample_size
        self.grid = [self.grid_size, self.grid_size]
        
        self.cond_step_start = input_dict['cond_step_start']
        
        self.controlnet_guidance_start = input_dict['controlnet_guidance_start']
        self.controlnet_guidance_end = input_dict['controlnet_guidance_end']
        self.controlnet_conditioning_scale = [float(x) for x in input_dict['controlnet_conditioning_scale'].split('-')]
        
        self.positive_prompts = input_dict['positive_prompts']
        self.negative_prompts = input_dict['negative_prompts']
        self.inversion_prompt = input_dict['inversion_prompt']
        
        self.batch_size = input_dict['batch_size']
        self.batch_size_vae = input_dict['batch_size_vae']

        self.num_inference_steps = input_dict['num_inference_steps']
        self.num_inversion_step = input_dict['num_inversion_step']
        self.inverse_path = input_dict['inverse_path']
        self.controls_path = input_dict['control_path']
        
        self.is_ddim_inversion = input_dict['is_ddim_inversion']

        self.is_shuffle = input_dict['is_shuffle']
        self.give_control_inversion = input_dict['give_control_inversion']

        self.guidance_scale = input_dict['guidance_scale']
        
        self.lora_scale = input_dict['lora_scale']
        self.enable_mask = input_dict['enable_mask']
        
        indices = list(np.arange(self.total_frame_number))
        
        width, height = input_dict['image_pil_list'][0].size
        self.vae_scale_factor = 8 # TODO: set by vae setting
        
        img_batch, control_batch_1, control_batch_2, mask_batch = self.process_image_batch(input_dict['image_pil_list'], input_dict['ctrlimage_pil_list_1'], input_dict['ctrlimage_pil_list_2'], input_dict['mask_pil_list'])
        init_latents_pre = self.encode_imgs(img_batch)
        
        self.scheduler = DDIMScheduler.from_config(self.scheduler_config)
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        self.inv_cond_embeddings, self.inv_uncond_embeddings = self.get_text_embeds(self.inversion_prompt, "")
        if self.is_ddim_inversion:
            init_latents, control_batch_1, control_batch_2 = self.__preprocess_inversion_input(init_latents_pre, control_batch_1, control_batch_2)
            latents_inverted, indices, control_batch_1, control_batch_2 = self.ddim_inversion(init_latents, control_batch_1, control_batch_2, indices)
            latents_inverted, control_batch_1, control_batch_2 = self.__postprocess_inversion_input(latents_inverted, control_batch_1, control_batch_2)
        else:
            ## original ##
            # init_latents_pre = torch.cat([init_latents_pre], dim=0) 
            # noise = torch.randn_like(init_latents_pre)
            # latents_inverted = self.scheduler.add_noise(init_latents_pre, noise, self.scheduler.timesteps[:1])
            
            ## YP ##
            noise_shape = (1, init_latents_pre.shape[1], height//self.vae_scale_factor, width//self.vae_scale_factor)
            set_seed_lib(self.seed)
            noise = torch.randn(noise_shape).to(self.device)
            tmp_batch_size = init_latents_pre.shape[0]
            grid_noise = noise.repeat(tmp_batch_size, 1,1,1)
            latents_inverted = self.scheduler.add_noise(init_latents_pre, grid_noise, self.scheduler.timesteps[:1])
            latents_inverted = torch.cat([latents_inverted], dim=0)

        if self.enable_mask:
            noise_shape = (1, init_latents_pre.shape[1], height//self.vae_scale_factor, width//self.vae_scale_factor)
            # init_latents_pre = torch.cat([init_latents_pre], dim=0)
            set_seed_lib(self.seed)
            self.noise = torch.randn(noise_shape).to(self.device)


        self.cond_embeddings, self.uncond_embeddings = self.get_text_embeds(self.positive_prompts, self.negative_prompts)
        
        # latents_inverted : [grid_count, 4, h//8*grid_size, w//8*grid_size]
        # control_batch : [grid_count, 3, h*grid_size, w*grid_size]
        # (if exist) mask_batch : [grid_count, 1, h//8*grid_size, w//8*grid_size]
        # init_latents_pre : [grid_count, 4, h//8*grid_size, w//8*grid_size]
        
        ## setting lora_scale to unet
        self.unet.fuse_lora(lora_scale=self.lora_scale)
        
        if not self.enable_mask:
            print('[INFO] not enable mask')
            latents_denoised, indices, controls_1, controls_2 = self.reverse_diffusion(latents_inverted, control_batch_1, control_batch_2, self.guidance_scale, indices=indices)
        else:
            # print(f'mask_batch.shape={mask_batch.shape}')
            # print(f'init_latents_pre.shape={init_latents_pre.shape}')
            print('[INFO] enable mask')
            latents_denoised, indices, controls_1, controls_2, mask, init_latents_pre = self.reverse_diffusion_with_mask(latents_inverted, control_batch_1, control_batch_2, mask_batch, init_latents_pre, self.guidance_scale, indices=indices)
    
    
        image_torch = self.decode_latents(latents_denoised)
        ordered_img_frames = self.order_grids(ipu.torch_to_pil_img_batch(image_torch), indices)
        ordered_control_frames_1 = self.order_grids(ipu.torch_to_pil_img_batch(controls_1), indices)
        ordered_control_frames_2 = self.order_grids(ipu.torch_to_pil_img_batch(controls_2), indices)
        return ordered_img_frames, ordered_control_frames_1, ordered_control_frames_2
    
