import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from clip.model import CLIP
from clip.simple_tokenizer import SimpleTokenizer
from random import sample

# 使用get_spike_matrix函数读取 .dat 文件的脉冲数据
def get_spike_matrix(filename, spike_height, spike_width, flipud=False, with_head=False):
    """
    从 .dat 文件读取脉冲数据并返回脉冲矩阵。
    """
    with open(filename, 'rb') as file:
        video_seq = file.read()

    video_seq = np.frombuffer(video_seq, 'b')
    video_seq = np.array(video_seq).astype(np.byte)

    img_size = spike_height * spike_width
    img_num = len(video_seq) // (img_size // 8)  # 计算帧数

    SpikeMatrix = np.zeros([img_num, spike_height, spike_width], np.byte)

    pix_id = np.arange(0, spike_height * spike_width)
    pix_id = np.reshape(pix_id, (spike_height, spike_width))
    comparator = np.left_shift(1, np.mod(pix_id, 8))
    byte_id = pix_id // 8

    for img_id in np.arange(img_num):
        id_start = img_id * img_size // 8
        id_end = id_start + img_size // 8
        cur_info = video_seq[id_start:id_end]
        data = cur_info[byte_id]
        result = np.bitwise_and(data, comparator)

        if flipud:
            SpikeMatrix[img_id, :, :] = np.flipud((result == comparator))
        else:
            SpikeMatrix[img_id, :, :] = (result == comparator)

    return SpikeMatrix

# 从 JSON 文件加载配置
def load_config(json_file):
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config

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


# 设备选择
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "/mnt/workspace/2_1_Clip-spikeCV/Clip-video-spike/models/2_2_500.pth"
json_config_path = "/mnt/workspace/2_1_Clip-spikeCV/Clip-video-spike/hmdb51_30.json"
spike_h = 240  # 图像高度
spike_w = 320  # 图像宽度

model = CLIP(
    embed_dim=256,
    image_resolution=(spike_h, spike_w),  # 视频帧大小
    vision_layers=(3, 4, 6, 3),  # 例如：每个阶段的残差块数
    vision_width=256,  # 可以根据需要调整宽度
    vision_patch_size=16,  # 假设这是图像块的大小，通常是16或32
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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 创建数据集和数据加载器
dataset = SpikingVideoDataset(json_config_path, spike_h, spike_w, DEVICE, step_size=3, num_timesteps=500)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 训练参数
TOTAL_EPOCHS = 100

for epoch in range(TOTAL_EPOCHS):
    ITER_BATCH_COUNT = len(dataloader)
    total_correct_i = 0
    total_correct_t = 0
    total_samples = 0

    for iteration, (spike_matrices, captions) in enumerate(dataloader):
        spike_matrices = spike_matrices.to(DEVICE)

        # 使用 Tokenizer 对文本进行编码
        tokenized_texts = []
        for caption in captions:
            tokenized = tokenizer.encode(caption)
            tokenized = tokenized[:77]
            tokenized += [0] * (77 - len(tokenized))
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
