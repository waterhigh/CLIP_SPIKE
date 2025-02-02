import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from clip.model import CLIP
from clip.simple_tokenizer import SimpleTokenizer
from random import sample
from adabelief_pytorch import AdaBelief

# torch.cuda.empty_cache()

# 使用get_spike_matrix函数读取 .dat 文件的脉冲数据
def get_spike_matrix(filename, spike_height, spike_width, flipud=False, with_head=False):
    """
    从 .dat 文件读取脉冲数据并返回脉冲矩阵。
    """
    # 读取 .dat 文件数据
    with open(filename, 'rb') as file:
        video_seq = file.read()

    # 将读取的字节数据转换为 numpy 数组
    video_seq = np.frombuffer(video_seq, 'b')
    video_seq = np.array(video_seq).astype(np.byte)

    # 计算每帧的大小
    img_size = spike_height * spike_width
    img_num = len(video_seq) // (img_size // 8)  # 计算帧数

    # 创建一个空的矩阵来存储所有帧数据
    SpikeMatrix = np.zeros([img_num, spike_height, spike_width], np.byte)

    pix_id = np.arange(0, spike_height * spike_width)  # 计算每个像素的 ID
    pix_id = np.reshape(pix_id, (spike_height, spike_width))
    comparator = np.left_shift(1, np.mod(pix_id, 8))  # 每个像素的位掩码
    byte_id = pix_id // 8  # 每个像素所在的字节索引

    # 遍历每一帧
    for img_id in np.arange(img_num):
        id_start = img_id * img_size // 8  # 每帧的起始字节位置
        id_end = id_start + img_size // 8  # 每帧的结束字节位置
        cur_info = video_seq[id_start:id_end]  # 获取当前帧的字节数据
        data = cur_info[byte_id]  # 获取对应的字节
        result = np.bitwise_and(data, comparator)  # 进行位运算，提取脉冲数据

        if flipud:
            SpikeMatrix[img_id, :, :] = np.flipud((result == comparator))  # 如果需要上下翻转
        else:
            SpikeMatrix[img_id, :, :] = (result == comparator)  # 不需要翻转

    return SpikeMatrix

# 从 JSON 文件加载配置
def load_config(json_file):
    """
    从 JSON 文件加载配置信息。
    """
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config

# 定义数据集类
# 定义数据集类
# class SpikingVideoDataset(Dataset):
#     def __init__(self, json_file, spike_h, spike_w, device, num_timesteps=1000):
#         """
#         从 JSON 文件读取文件夹路径和标签，并加载脉冲数据。
#         """
#         self.device = device
#         self.num_timesteps = num_timesteps

#         # 从 JSON 文件加载数据
#         config = load_config(json_file)
#         self.dat_folder_paths = list(config.keys())  # 获取文件夹路径
#         self.labels = list(config.values())  # 获取对应的标签

#         self.frames_list = []

#         # 读取所有文件夹中的 .dat 文件并转换为视频帧
#         for folder_path in self.dat_folder_paths:
#             dat_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dat')])
#             video_frames = []

#             for dat_file in dat_files:
#                 frames = get_spike_matrix(dat_file, spike_h, spike_w)  # 调用 get_spike_matrix 获取脉冲数据
#                 video_frames.append(frames)

#             # 将多个文件的帧堆叠成一个完整的视频
#             video_frames = np.concatenate(video_frames, axis=0)
#             self.frames_list.append(video_frames)

#     def __len__(self):
#         return len(self.frames_list)

#     def __getitem__(self, idx):
#         frames = self.frames_list[idx]  # 获取视频帧
        
#         # 如果时间步数大于指定的时间步数，按固定间隔选择时间步
#         num_frames = frames.shape[0]
#         if num_frames > self.num_timesteps:
#             step_size = (num_frames - 1) // (self.num_timesteps - 1)  # 计算步长，确保包含第一个时间步
#             selected_frames_idx = [i * step_size for i in range(self.num_timesteps)]  # 获取固定间隔的时间步
#             frames = frames[selected_frames_idx]

#         # 获取视频对应的文本标签
#         caption = self.labels[idx % len(self.labels)]  # 获取标签

#         # 转换为 tensor
#         frames = torch.tensor(frames).float().to(self.device)

#         return frames, caption

class SpikingVideoDataset(Dataset):
    def __init__(self, json_file, spike_h, spike_w, device, step_size=6, num_timesteps=50):
        """
        从 JSON 文件读取文件夹路径和标签，并加载脉冲数据。
        
        Parameters:
            json_file (str): JSON 配置文件路径
            spike_h (int): 图像高度
            spike_w (int): 图像宽度
            device (str): 设备 ('cuda' 或 'cpu')
            step_size (int): 从每个文件夹中选择 .dat 文件的步长（每隔多少个文件选择一个）
            num_timesteps (int): 每个视频的时间步数（每个 `.dat` 文件包含 50 个时间步）
        """
        self.device = device
        self.num_timesteps = num_timesteps  # 每个文件中包含 50 个时间步
        self.step_size = step_size  # 用户手动设置的步长

        # 从 JSON 文件加载数据
        config = load_config(json_file)
        self.dat_folder_paths = list(config.keys())  # 获取文件夹路径
        self.labels = list(config.values())  # 获取对应的标签

        self.frames_list = []

        # 读取所有文件夹中的 .dat 文件并转换为视频帧
        for folder_path in self.dat_folder_paths:
            dat_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dat')])

            # 使用用户提供的 step_size
            selected_dat_files = dat_files[::self.step_size]  # 按步长选择文件

            video_frames = []

            for dat_file in selected_dat_files:
                frames = get_spike_matrix(dat_file, spike_h, spike_w)  # 获取脉冲数据

                # 由于每个文件已经包含 50 个时间步，这里无需再选择时间步
                video_frames.append(frames)

            # 将多个文件的帧堆叠成一个完整的视频
            video_frames = np.concatenate(video_frames, axis=0)
            self.frames_list.append(video_frames)

    def __len__(self):
        return len(self.frames_list)

    def __getitem__(self, idx):
        frames = self.frames_list[idx]  # 获取视频帧

        # 如果时间步数大于指定的时间步数，按固定间隔选择时间步
        num_frames = frames.shape[0]
        if num_frames > self.num_timesteps:
            step_size = (num_frames - 1) // (self.num_timesteps - 1)  # 计算步长，确保包含第一个时间步
            selected_frames_idx = [i * step_size for i in range(self.num_timesteps)]  # 获取固定间隔的时间步
            frames = frames[selected_frames_idx]

        # 获取视频对应的文本标签
        caption = self.labels[idx % len(self.labels)]  # 获取标签

        # 转换为 tensor
        frames = torch.tensor(frames).float().to(self.device)

        return frames, caption




# 设备选择
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "/mnt/workspace/2_1_Clip-spikeCV/Clip-video-spike/models/2_1.pth"
json_config_path = "/mnt/workspace/2_1_Clip-spikeCV/Clip-video-spike/hmdb51_30.json"  # JSON 配置文件路径
spike_h = 240  # 图像高度
spike_w = 320  # 图像宽度

# 初始化模型
model = CLIP(
    embed_dim=256,
    image_resolution=(spike_h, spike_w),  # 视频帧大小
    vision_layers=(3, 4, 6, 3),  # 例如：每个阶段的残差块数
    vision_width=256,  # 可以根据需要调整宽度
    vision_patch_size=None,  # ResNet不需要这个参数
    context_length=77,
    vocab_size=49408,
    transformer_width=256,
    transformer_heads=8,
    transformer_layers=8
).to(DEVICE)


# 加载模型权重
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f"Loaded model weights from {MODEL_PATH}")
else:
    print(f"No pre-trained model found at {MODEL_PATH}, starting from scratch.")

# 初始化 Tokenizer
tokenizer = SimpleTokenizer()

# 定义优化器
optimizer = AdaBelief(model.parameters(), lr=1e-4, weight_decay=1e-4)

# 创建数据集和数据加载器
dataset = SpikingVideoDataset(json_config_path, spike_h, spike_w, DEVICE, step_size=6, num_timesteps=50)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 训练参数
TOTAL_EPOCHS = 100

for epoch in range(TOTAL_EPOCHS):
    ITER_BATCH_COUNT = len(dataloader)
    total_correct_i = 0
    total_correct_t = 0
    total_samples = 0
    
    for iteration, (spike_matrices, captions) in enumerate(dataloader):
        # 将数据转移到设备上
        spike_matrices = spike_matrices.to(DEVICE)  # (batch_size, timesteps, height, width)

        # 使用 Tokenizer 对文本进行编码
        tokenized_texts = []
        for caption in captions:
            tokenized = tokenizer.encode(caption)
            tokenized = tokenized[:77]  # 截断到最大长度
            tokenized += [0] * (77 - len(tokenized))  # 填充到最大长度
            tokenized_texts.append(tokenized)
        tokenized_texts = torch.tensor(tokenized_texts).to(DEVICE)

        # 调用模型的 forward 方法获取 logits
        logits_per_image, logits_per_text = model(spike_matrices, tokenized_texts)

        # 创建目标
        targets = torch.arange(len(spike_matrices)).to(DEVICE)

        # 计算对比损失
        loss_i = torch.nn.functional.cross_entropy(logits_per_image, targets)
        loss_t = torch.nn.functional.cross_entropy(logits_per_text, targets)
        loss = (loss_i + loss_t) / 2

        # 计算准确率
        pred_i = logits_per_image.argmax(dim=1)
        pred_t = logits_per_text.argmax(dim=1)
        correct_i = (pred_i == targets).sum().item()
        correct_t = (pred_t == targets).sum().item()

        total_correct_i += correct_i
        total_correct_t += correct_t
        total_samples += len(targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 打印训练进度
        if iteration % 10 == 0:
            batch_acc_i = correct_i / len(targets) * 100
            batch_acc_t = correct_t / len(targets) * 100
            accum_acc_i = total_correct_i / total_samples * 100
            accum_acc_t = total_correct_t / total_samples * 100

            print(f"Epoch {epoch+1}, Iteration {iteration}/{ITER_BATCH_COUNT}, "
                  f"Loss: {loss.item():.4f} | "
                  f"Batch Acc: I->T {batch_acc_i:.2f}% T->I {batch_acc_t:.2f}% | "
                  f"Accum Acc: I->T {accum_acc_i:.2f}% T->I {accum_acc_t:.2f}%")

    # 定期保存模型
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH} at Epoch {epoch+1}")

print("Training completed.")


