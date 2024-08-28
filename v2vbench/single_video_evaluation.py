import logging
import sys

from v2vbench import EvaluatorWrapper

logging.basicConfig(
    level=logging.INFO,  # Change it to logging.DEBUG if you want to troubleshoot
    format="%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)d)",
    handlers=[logging.StreamHandler(sys.stdout),]
)

evaluation = EvaluatorWrapper(metrics='all')  # 'all' for all metrics
# print(EvaluatorWrapper.all_metrics)  # list all available metrics

results = evaluation.evaluate(
    edit_video='/home/cgvsl/P76111131/RAVE/results/07-26-2024/dogwindow_goldenretrieverpuppy/testcontrol/softedge_pidinet1.0_withoutmask_withborder/dog_wind_30f/A golden retriever puppy sticking its head out of a car window moving through a landscape filled with tall trees-00000/A-golden-retriever-puppy-sticking-its-head-out-of-a-car-window-moving-through-a-landscape-filled-with-tall-trees.gif',
    reference_video='/home/cgvsl/P76111131/RAVE/data/mp4_frames/dog_wind/27frames',
    edit_prompt='A golden retriever puppy sticking its head out of a car window moving through a landscape filled with tall trees',
    output_dir='/home/cgvsl/P76111131/awesome-diffusion-v2v/result/dogwind_goldenretrieverpuppy_2',
)

results = evaluation.evaluate(
    edit_video='/home/cgvsl/P76111131/awesome-diffusion-v2v/myvideo/edit_video_mp4/dog_wind/0.mp4',
    reference_video='/home/cgvsl/P76111131/awesome-diffusion-v2v/myvideo/dogwind_27_fps11.mp4',
    edit_prompt='A golden retriever puppy sticking its head out of a car window moving through a landscape filled with tall trees',
    output_dir='/home/cgvsl/P76111131/awesome-diffusion-v2v/result/dogwind_mp4',
)

# results = evaluation.evaluate(
#     edit_video='/home/cgvsl/P76111131/RAVE/results/07-26-2024/foxturn_wolf_text/depth_zoe1.0_withborder_inversion150_inference100_neg/fox_turn_head_27_fps11/A wolf standing in a serene wooded clearing with tall trees, 4k, high quality-00005/A-wolf-standing-in-a-serene-wooded-clearing-with-tall-trees,-4k,-high-quality.gif',
#     reference_video='/home/cgvsl/P76111131/RAVE/data/mp4_frames/fox_turn_head/27frames',
#     edit_prompt='A wolf standing in a serene wooded clearing with tall trees',
#     output_dir='/home/cgvsl/P76111131/awesome-diffusion-v2v/result/foxturn_wolftextonly',
# )