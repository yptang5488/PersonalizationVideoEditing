from rembg import remove
import os
from PIL import Image

def get_masked_image(frame, mask):
    # Convert the mask to 'L' mode
    mask = mask.convert("L")
    black_image = Image.new("RGB", frame.size, color="black")

    # Composite the original image and the black image using the binary mask
    result_image = Image.composite(frame, black_image, mask)
    
    return result_image

def read_frames_from_folder(data_path):
    frames_path = [os.path.join(data_path, f) for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    frames_path.sort()  # Sort the frames in alphabetical order

    frames = [Image.open(frame_path) for frame_path in frames_path]
    # frame = cv2.imread(os.path.join(input_folder, frames[0]))
    print(f'len(frames) = {len(frames)} ; frame.size = {frames[0].size}')
    return frames, frames_path

def get_frames_and_masks(data_path, output_path_mask):
    if os.path.isdir(data_path):
        frames, frames_path = read_frames_from_folder(data_path)
        
        for i, f in enumerate(frames):
            rm_f = remove(f)
            target_mask = rm_f.convert("RGB").point(lambda x: 0 if x < 1 else 255).convert('L').convert('RGB')
            
            # save mask
            output_file_path = os.path.join(output_path, 'mask')
            os.makedirs(output_file_path, exist_ok=True)
            output_file_name = os.path.basename(frames_path[i]).split('.')[0] + '_mask.png'
            print(output_file_name)
            target_mask.save(os.path.join(output_file_path, output_file_name))
            
            # save masked_image
            masked_img = get_masked_image(f, target_mask)
            output_file_path = os.path.join(output_path, 'masked_image')
            os.makedirs(output_file_path, exist_ok=True)
            output_file_name = os.path.basename(frames_path[i]).split('.')[0] + '_masked.jpg'
            print(output_file_name)
            masked_img.save(os.path.join(output_file_path, output_file_name))
            
            
    elif os.path.isfile(data_path):
        frames = [Image.open(data_path)]
        print(f'len(frames) = {len(frames)} ; frame.size = {frames[0].size}')

        rm_f = remove(frames[0])
        target_mask = rm_f.convert("RGB").point(lambda x: 0 if x < 1 else 255).convert('L').convert('RGB')
        
        # save mask
        output_file_name = os.path.basename(frames_path[i]).split('.')[0] + '_mask'
        print(output_file_name)
        target_mask.save(os.path.join(output_file_path, output_file_name))
        
        # save masked_image
        masked_img = get_masked_image(frames[0], target_mask)
        output_file_name = os.path.basename(frames_path[i]).split('.')[0] + '_masked.jpg'
        print(output_file_name)
        masked_img.save(os.path.join(output_file_path, output_file_name))

def get_SAM_masked_images(data_path, SAM_mask_path, output_path):
    if os.path.isdir(data_path):
        frames, frames_path = read_frames_from_folder(data_path)
        masks, _ = read_frames_from_folder(SAM_mask_path)
        
        # save masked_image
        for i in range(len(frames)):
            masked_img = get_masked_image(frames[i], masks[i])
            output_file_path = os.path.join(output_path, 'SAM_masked_image')
            os.makedirs(output_file_path, exist_ok=True)
            output_file_name = os.path.basename(frames_path[i]).split('.')[0] + '_masked.jpg'
            print(output_file_name)
            masked_img.save(os.path.join(output_file_path, output_file_name))
    else:
        frame = Image.open(data_path)
        mask = Image.open(SAM_mask_path)
        
        masked_img = get_masked_image(frame, mask)
        os.makedirs(output_path, exist_ok=True)
        output_file_name = os.path.basename(data_path).split('.')[0] + '_masked.jpg'
        print(output_file_name)
        masked_img.save(os.path.join(output_path, output_file_name))

# data_path = '/home/cgvsl/P76111131/MotionDirector/ddim-inversion/DogRun_higher_res/zeroscope_v2_576w/dog_run_2sec/steps_500/nframes_16/frames'
# output_path = '/home/cgvsl/P76111131/MotionDirector/ddim-inversion/DogRun_higher_res/zeroscope_v2_576w/dog_run_2sec/steps_500/nframes_16'
# get_frames_and_masks(data_path, output_path)

data_path = '/home/cgvsl/P76111131/PersonalizedVideo/__assets__/input/white_fluffy_dog.jpg'
output_path = '/home/cgvsl/P76111131/PersonalizedVideo/__assets__/input/masked_image'
SAM_mask_path = '/home/cgvsl/P76111131/PersonalizedVideo/__assets__/input/mask/white_fluffy_dog_mask.png'
get_SAM_masked_images(data_path, SAM_mask_path, output_path)