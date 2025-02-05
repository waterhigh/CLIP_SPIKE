import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from clip.model import CLIP
from clip.simple_tokenizer import SimpleTokenizer
from random import sample
import matplotlib.pyplot as plt

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
    def __init__(self, json_file, spike_h, spike_w, device, step_size=6, num_timesteps=500):
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
MODEL_PATH = "/mnt/workspace/2_1_Clip-spikeCV/Clip-video-spike/models/2_5_2000.pth"
json_config_path = "/mnt/workspace/2_1_Clip-spikeCV/Clip-video-spike/UCF101_30.json"
spike_h = 240  # 图像高度
spike_w = 320  # 图像宽度

# 初始化模型
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
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

from torch.utils.data import random_split

# 创建数据集
dataset = SpikingVideoDataset(json_config_path, spike_h, spike_w, DEVICE, step_size=6, num_timesteps=2000)

# 设置训练集和验证集的比例（比如 80% 训练集，20% 验证集）
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# 随机拆分数据集
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

import matplotlib.pyplot as plt

# 创建列表来存储每个 epoch 的训练和验证准确率
train_acc_i_list = []
train_acc_t_list = []
val_acc_i_list = []
val_acc_t_list = []

# 训练参数
TOTAL_EPOCHS = 150

for epoch in range(TOTAL_EPOCHS):
    model.train()  # 设置模型为训练模式
    ITER_BATCH_COUNT = len(train_dataloader)
    total_correct_i = 0
    total_correct_t = 0
    total_samples = 0

    # 训练阶段
    for iteration, (spike_matrices, captions) in enumerate(train_dataloader):
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

    # 将训练准确率记录到列表中
    train_acc_i_list.append(accum_acc_i)
    train_acc_t_list.append(accum_acc_t)

    # 验证阶段
    model.eval()  # 设置模型为评估模式
    total_correct_val_i = 0
    total_correct_val_t = 0
    total_samples_val = 0

    with torch.no_grad():  # 在验证时不进行梯度计算
        for iteration, (spike_matrices, captions) in enumerate(val_dataloader):
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

            # 计算准确率
            pred_i = logits_per_image.argmax(dim=1)
            pred_t = logits_per_text.argmax(dim=1)
            correct_i = (pred_i == targets).sum().item()
            correct_t = (pred_t == targets).sum().item()

            total_correct_val_i += correct_i
            total_correct_val_t += correct_t
            total_samples_val += len(targets)

        # 计算验证集准确率
        val_acc_i = total_correct_val_i / total_samples_val * 100
        val_acc_t = total_correct_val_t / total_samples_val * 100
        print(f"Epoch {epoch+1} Validation Acc: I->T {val_acc_i:.2f}% T->I {val_acc_t:.2f}%")

        # 将验证准确率记录到列表中
        val_acc_i_list.append(val_acc_i)
        val_acc_t_list.append(val_acc_t)

    # 定期保存模型
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH} at Epoch {epoch+1}")

# 绘制训练集和验证集准确率的折线图
plt.figure(figsize=(10, 6))
plt.plot(range(1, TOTAL_EPOCHS + 1), train_acc_i_list, label="Train I->T Accuracy")
plt.plot(range(1, TOTAL_EPOCHS + 1), train_acc_t_list, label="Train T->I Accuracy")
plt.plot(range(1, TOTAL_EPOCHS + 1), val_acc_i_list, label="Val I->T Accuracy")
plt.plot(range(1, TOTAL_EPOCHS + 1), val_acc_t_list, label="Val T->I Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy Over Epochs")
plt.legend()

# 保存图像到模型文件夹
plt.savefig("/mnt/workspace/2_1_Clip-spikeCV/Clip-video-spike/models/accuracy_plot.png")
print("saved!")

print("Training completed.")
