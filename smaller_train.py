import os
import shutil

# 原目录路径
source_dir = "/mnt/workspace/a_pku_spike/video_datasets/hmdb51/annotations/data/hmdb51/train_s"

# 目标目录路径
target_dir = "/mnt/workspace/a_pku_spike/video_datasets/hmdb51/annotations/data/hmdb51/test"

# 要处理的子文件夹名称
categories = ["eat", "drink", "run"]

# 遍历每个子文件夹
for category in categories:
    # 获取源文件夹路径
    source_category_dir = os.path.join(source_dir, category)
    
    # 创建目标文件夹路径
    target_category_dir = os.path.join(target_dir, category)
    os.makedirs(target_category_dir, exist_ok=True)

    # 获取所有文件并按文件名排序
    files = sorted(os.listdir(source_category_dir))
    
    # 只取前100个文件
    files_to_copy = files[-15:]
    
    for file_name in files_to_copy:
        # 获取源文件和目标文件的完整路径
        source_file = os.path.join(source_category_dir, file_name)
        target_file = os.path.join(target_category_dir, file_name)
        
        # 如果是文件夹，就递归复制文件夹内容
        if os.path.isdir(source_file):
            shutil.copytree(source_file, target_file)
        else:
            shutil.copy(source_file, target_file)

print("文件提取完成！")
