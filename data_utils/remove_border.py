import os
from PIL import Image

def remove_border(pil_image, border_size=10):
    # 獲取圖片尺寸
    w, h = pil_image.size
    
    # 創建一個去掉邊界的圖片
    return pil_image.crop((border_size, border_size, w - border_size, h - border_size))

def process_and_copy_frames(input_folder, output_folder, border_size=10):
    # 檢查輸入資料夾是否存在
    if not os.path.exists(input_folder):
        print(f"input directory {input_folder} is not exist")
        return

    # 如果輸出資料夾不存在，則創建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 讀取輸入資料夾中的所有檔案
    for filename in os.listdir(input_folder):
        input_file_path = os.path.join(input_folder, filename)
        
        # 檢查是否為文件（而不是子資料夾）
        if os.path.isfile(input_file_path):
            try:
                # 打開圖片
                with Image.open(input_file_path) as img:
                    # 去掉邊界
                    processed_img = remove_border(img, border_size)
                    
                    # 構建輸出文件路徑
                    output_file_path = os.path.join(output_folder, filename)
                    
                    # 保存處理過的圖片
                    processed_img.save(output_file_path)
                    print(f"deal with {filename} and output to {output_folder}")
            except Exception as e:
                print(f"deal with {filename} and get error: {e}")

    print("[INFO] all done !")

# 使用範例
input_folder = "/home/cgvsl/P76111131/RAVE/results/07-01-2024/dogwind_goldenretrieverpuppy/*step100_border_rebuildbyoldeinverse/dog_wind_30f/A golden retriever puppy sticking its head out of a car window moving through a landscape filled with tall trees, autumn-00000/result/resultframes"  # 輸入資料夾的路徑
output_folder = "/home/cgvsl/P76111131/RAVE/results/07-01-2024/dogwind_goldenretrieverpuppy/*step100_border_rebuildbyoldeinverse/dog_wind_30f/A golden retriever puppy sticking its head out of a car window moving through a landscape filled with tall trees, autumn-00000/result/removebroderframes"  # 輸出資料夾的路徑
border_size = 8  # 邊界大小

process_and_copy_frames(input_folder, output_folder, border_size)
