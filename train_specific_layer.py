import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt

# ==============================================================================
# --- 1. 超参数与配置 ---
# ==============================================================================

# -- !! 请在此处设置您从上一步中选择的最佳层 !! --
TARGET_LAYER = 25 

# -- 数据与模型路径 --
DATA_DIR = "activation"
OUTPUT_DIR = "training_results" # 将模型和图片都保存在这里
MODEL_NAME = f"ebm_decoder_layer_{TARGET_LAYER}_new.pt"
PLOT_NAME = f"training_progress_layer_{TARGET_LAYER}_new.png" # 可视化图片的文件名

# -- 训练参数 --
LEARNING_RATE = 3e-4
BATCH_SIZE = 256
NUM_EPOCHS = 150
PATIENCE = 15
TRAIN_VAL_SPLIT = 0.8

# ==============================================================================
# --- 2. EBM 模型定义 ---
# ==============================================================================

class EBM(nn.Module):
    def __init__(self, input_dim=3584):
        super(EBM, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.net(x)

# ==============================================================================
# --- 3. 主训练函数 ---
# ==============================================================================

def train_final_ebm():
    """
    在指定层上训练EBM模型，保存最佳模型，并保存训练过程的可视化图表。
    """
    hidden_states_path = os.path.join(DATA_DIR, "hidden_states.pt")
    labels_path = os.path.join(DATA_DIR, "labels.pt")
    output_model_path = os.path.join(OUTPUT_DIR, MODEL_NAME)
    output_plot_path = os.path.join(OUTPUT_DIR, PLOT_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- 开始在 Decoder Layer {TARGET_LAYER} 上训练EBM ---")
    print(f"使用设备: {device}")

    try:
        hidden_states = torch.load(hidden_states_path, map_location=device)
        labels = torch.load(labels_path, map_location=device)
        print(f"成功加载数据。激活值形状: {hidden_states.shape}")
    except FileNotFoundError:
        print(f"错误: 激活文件未找到。请确保 '{DATA_DIR}' 文件夹存在。")
        return

    print(f"正在准备第 {TARGET_LAYER} 层的激活数据...")
    layer_hs = hidden_states[:, TARGET_LAYER, :].float()
    dataset = TensorDataset(layer_hs, labels.long())

    train_size = int(TRAIN_VAL_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = EBM(input_dim=layer_hs.shape[1]).to(device)
    
    if (labels == 1).sum() > 0:
        pos_weight = (labels == 0).sum().float() / (labels == 1).sum().float()
    else:
        pos_weight = 1.0
    class_weights = torch.tensor([1.0, pos_weight], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_val_loss = float('inf')
    patience_counter = 0
    
    # 新增：用于记录训练历史的列表
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

    print("开始训练...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss_total = 0
        
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]")
        for batch_hs, batch_labels in train_iterator:
            optimizer.zero_grad()
            outputs = model(batch_hs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()
            train_iterator.set_postfix(loss=loss.item())

        scheduler.step()
        avg_train_loss = train_loss_total / len(train_loader)

        model.eval()
        val_loss = 0
        all_preds_proba = []
        all_gts = []
        with torch.no_grad():
            for batch_hs, batch_labels in val_loader:
                outputs = model(batch_hs)
                val_loss += criterion(outputs, batch_labels).item()
                probabilities = torch.softmax(outputs, dim=1)[:, 1]
                all_preds_proba.extend(probabilities.cpu().numpy())
                all_gts.extend(batch_labels.cpu().numpy())
        
        val_loss /= len(val_loader) if len(val_loader) > 0 else 1
        val_auc = roc_auc_score(all_gts, all_preds_proba) if len(np.unique(all_gts)) > 1 else 0.5

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
        
        # 记录当前epoch的指标
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_model_path)
            print(f"  -> 验证损失降低，已保存最佳模型至: '{output_model_path}'")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"验证损失连续 {PATIENCE} 轮未改善，触发早停。")
                break
    
    print("\n--- 训练完成 ---")
    print(f"最终的最佳模型已保存在: '{output_model_path}'")

    # ==============================================================================
    # --- 新增功能：绘制并保存训练过程图表 ---
    # ==============================================================================
    epochs_ran = range(1, len(history['train_loss']) + 1)
    
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制损失曲线 (左Y轴)
    color = 'tab:red'
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', color=color, fontsize=12)
    ax1.plot(epochs_ran, history['train_loss'], color='red', linestyle='--', marker='.', label='Train Loss')
    ax1.plot(epochs_ran, history['val_loss'], color='darkred', linestyle='-', marker='.', label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 创建双Y轴绘制AUC
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('AUC', color=color, fontsize=12)
    ax2.plot(epochs_ran, history['val_auc'], color=color, linestyle='-', marker='x', label='Validation AUC')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    plt.title(f'Training & Validation Metrics for Decoder Layer {TARGET_LAYER}', fontsize=16)
    fig.tight_layout()
    
    plt.savefig(output_plot_path)
    print(f"训练过程图已保存至: '{output_plot_path}'")

# ==============================================================================
# --- 4. 脚本执行入口 ---
# ==============================================================================

if __name__ == "__main__":
    train_final_ebm()