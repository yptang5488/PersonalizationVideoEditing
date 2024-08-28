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
    edit_video='/home/cgvsl/P76111131/awesome-diffusion-v2v/myvideo_0820/edit_video_mp4',
    reference_video='/home/cgvsl/P76111131/awesome-diffusion-v2v/myvideo_0820/reference_video_mp4',
    index_file='/home/cgvsl/P76111131/awesome-diffusion-v2v/data_index_0820.yaml',
    output_dir='/home/cgvsl/P76111131/awesome-diffusion-v2v/result/0820_total',
    # it is recommended to cache the flow of source videos for motion_alignment to avoid redundant computation
    # evaluator_kwargs={'motion_alignment': {'cache_flow': True}},
)