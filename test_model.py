import torch
from torch.utils.data import Dataset, DataLoader
from clip.model import CLIP
from clip.simple_tokenizer import SimpleTokenizer
import os
import json
import numpy as np

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

# from train import SpikingVideoDataset  # 这里你应该导入你的数据集类

# 定义数据集类
class SpikingVideoDataset(Dataset):
    def __init__(self, json_file, spike_h, spike_w, device, step_size=6, num_timesteps=50):
        self.device = device
        self.num_timesteps = num_timesteps  # 每个文件中包含 50 个时间步
        self.step_size = step_size

        config = load_config(json_file)
        self.dat_folder_paths = list(config.keys())  # 获取文件夹路径
        self.labels = list(config.values())  # 获取对应的标签

        self.frames_list = []

        for folder_path in self.dat_folder_paths:
            dat_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dat')])
            selected_dat_files = dat_files[::self.step_size]

            video_frames = []
            for dat_file in selected_dat_files:
                frames = get_spike_matrix(dat_file, spike_h, spike_w)
                video_frames.append(frames)

            video_frames = np.concatenate(video_frames, axis=0)
            self.frames_list.append(video_frames)

    def __len__(self):
        return len(self.frames_list)

    def __getitem__(self, idx):
        frames = self.frames_list[idx]

        num_frames = frames.shape[0]
        if num_frames > self.num_timesteps:
            step_size = (num_frames - 1) // (self.num_timesteps - 1)
            selected_frames_idx = [i * step_size for i in range(self.num_timesteps)]
            frames = frames[selected_frames_idx]
        elif num_frames < self.num_timesteps:
            # 填充帧数不足的情况，填充为零
            padding = self.num_timesteps - num_frames
            frames = np.pad(frames, ((0, padding), (0, 0), (0, 0)), mode='constant', constant_values=0)

        caption = self.labels[idx % len(self.labels)]  # 获取标签

        frames = torch.tensor(frames).float().to(self.device)

        return frames, caption




print("hello!")
#  配置和路径
MODEL_PATH = "/mnt/workspace/2_1_Clip-spikeCV/Clip-video-spike/models/2_2_500.pth"  # 已训练的模型路径
TEST_JSON_PATH = "/mnt/workspace/2_1_Clip-spikeCV/Clip-video-spike/hmdb51_30.json"  # 测试数据集的路径
spike_h = 240  # 图像高度
spike_w = 320  # 图像宽度
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
context_length = 77  # 文本的最大长度
vocab_size = 49408  # 词汇大小
transformer_width = 256  # transformer的宽度
transformer_heads = 8  # transformer的头数
transformer_layers = 8  # transformer的层数
batch_size = 2  # 测试时的batch size

# 加载模型
model = CLIP(
    embed_dim=256,
    image_resolution=(spike_h, spike_w),
    vision_layers=(3, 4, 6, 3),
    vision_width=256,
    vision_patch_size=None,
    context_length=context_length,
    vocab_size=vocab_size,
    transformer_width=transformer_width,
    transformer_heads=transformer_heads,
    transformer_layers=transformer_layers
).to(DEVICE)

# 加载训练好的模型权重
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f"Loaded model weights from {MODEL_PATH}")
else:
    print(f"No pre-trained model found at {MODEL_PATH}")
    exit(1)

# 设置模型为评估模式
model.eval()

# 加载测试数据集
dataset = SpikingVideoDataset(TEST_JSON_PATH, spike_h, spike_w, DEVICE, num_timesteps=500)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 定义Tokenizer
tokenizer = SimpleTokenizer()

# 计算测试性能
correct = 0
total = 0

with torch.no_grad():  # 不计算梯度
# 在评估时打印出每个batch的预测和真实标签
    for iteration, (spike_matrices, captions) in enumerate(dataloader):
        spike_matrices = spike_matrices.to(DEVICE)
        # 对文本进行编码
        tokenized_texts = []
        for caption in captions:
            tokenized = tokenizer.encode(caption)
            tokenized = tokenized[:context_length]  # 截断
            tokenized += [0] * (context_length - len(tokenized))  # 填充
            tokenized_texts.append(tokenized)
        tokenized_texts = torch.tensor(tokenized_texts).to(DEVICE)

        # 模型推理
        logits_per_image, logits_per_text = model(spike_matrices, tokenized_texts)

        # 获取预测结果
        _, predicted = torch.max(logits_per_image, 1)  # 取出每个样本的最大预测值

        # 打印预测的类别和实际标签
        print(f"Predicted: {predicted}, True Labels: {captions}")

        # 计算准确率
        total += len(predicted)
        correct += (predicted == torch.arange(len(spike_matrices)).to(DEVICE)).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
