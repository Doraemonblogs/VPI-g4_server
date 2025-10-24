import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. 定义EBM模型结构 ---
# 参考论文附录C，我们使用一个双层MLP，并加入ReLU和Dropout
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
            nn.Linear(256, 2) # 输出2个logit，对应成功/失败两个类别
        )

    def forward(self, x):
        return self.net(x)

# --- 2. 主执行函数 ---
def find_best_layer():
    """
    加载数据，遍历所有层进行训练和评估，并找出性能最佳的层。
    """
    # --- 设置路径和参数 ---
    hidden_states_path = "activation/hidden_states.pt"
    labels_path = "activation/labels.pt"
    output_image_path = "layer_performance.png"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # --- 加载数据 ---
    try:
        # 激活值张量形状应为: (样本数, 总层数, 隐藏层维度)
        # Qwen2.5-7B包含1个embedding层 + 28个decoder层 = 29层
        hidden_states = torch.load(hidden_states_path, map_location=device)
        labels = torch.load(labels_path, map_location=device)
        print(f"激活值张量形状: {hidden_states.shape}")
        print(f"标签张量形状: {labels.shape}")
    except FileNotFoundError:
        print(f"错误: 无法在路径 '{os.getcwd()}' 中找到激活值或标签文件。")
        print("请确认 'activation/hidden_states.pt' 和 'activation/labels.pt' 文件存在。")
        return

    # 我们将遍历所有28个decoder层进行评估 (索引从1到28)
    num_total_layers = hidden_states.shape[1]
    layer_indices_to_train = range(1, num_total_layers) 

    all_val_losses = []
    all_val_aucs = []

    # --- 遍历所有decoder层 ---
    for layer_idx in tqdm(layer_indices_to_train, desc="遍历所有层"):
        layer_hs = hidden_states[:, layer_idx, :].float()
        dataset = TensorDataset(layer_hs, labels.long())

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        model = EBM(input_dim=layer_hs.shape[1]).to(device)
        
        # 计算类别权重以处理数据不平衡
        if (labels == 1).sum() > 0:
            pos_weight = (labels == 0).sum().float() / (labels == 1).sum().float()
        else:
            pos_weight = 1.0
        
        class_weights = torch.tensor([1.0, pos_weight], device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

        best_val_loss = float('inf')
        best_val_auc_for_this_layer = 0.0
        patience_counter = 0

        # --- 训练与验证循环 ---
        for epoch in range(100): # 最多训练100轮
            model.train()
            for batch_hs, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_hs)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
            scheduler.step()

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
            if len(np.unique(all_gts)) > 1: # AUC至少需要两个类别
                 val_auc = roc_auc_score(all_gts, all_preds_proba)
            else:
                 val_auc = 0.5

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_auc_for_this_layer = val_auc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10: # 早停机制
                    break
        
        all_val_losses.append(best_val_loss)
        all_val_aucs.append(best_val_auc_for_this_layer)

    # --- 结果分析与可视化 ---
    decoder_layer_offset = 1 
    best_layer_by_loss_idx = np.argmin(all_val_losses)
    best_layer_by_auc_idx = np.argmax(all_val_aucs)
    
    best_layer_by_loss = best_layer_by_loss_idx + decoder_layer_offset
    best_layer_by_auc = best_layer_by_auc_idx + decoder_layer_offset

    print("\n--- 结果分析 ---")
    print(f"根据最低验证损失，最佳层是: Decoder Layer {best_layer_by_loss}")
    print(f"  - 验证损失: {all_val_losses[best_layer_by_loss_idx]:.4f}")
    print(f"  - 验证AUC: {all_val_aucs[best_layer_by_loss_idx]:.4f}")
    print("-" * 20)
    print(f"根据最高验证AUC，最佳层是: Decoder Layer {best_layer_by_auc}")
    print(f"  - 验证损失: {all_val_losses[best_layer_by_auc_idx]:.4f}")
    print(f"  - 验证AUC: {all_val_aucs[best_layer_by_auc_idx]:.4f}")
    print("-" * 20)
    print(f"论文中选择的标准是'最低验证损失'，因此推荐使用 Decoder Layer {best_layer_by_loss}。")

    # --- 绘图 ---
    fig, ax1 = plt.subplots(figsize=(12, 6))
    decoder_layer_numbers = list(layer_indices_to_train)
    
    color = 'tab:red'
    ax1.set_xlabel('Decoder Layer Index')
    ax1.set_ylabel('Best Validate Loss', color=color)
    ax1.plot(decoder_layer_numbers, all_val_losses, color=color, marker='o', label='Validate Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.plot(best_layer_by_loss, all_val_losses[best_layer_by_loss_idx], 'r*', markersize=15, label=f'Best Loss (Layer {best_layer_by_loss})')

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Best Validate AUC', color=color)
    ax2.plot(decoder_layer_numbers, all_val_aucs, color=color, marker='x', linestyle='--', label='Validate AUC')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.plot(best_layer_by_auc, all_val_aucs[best_layer_by_auc_idx], 'b*', markersize=15, label=f'Best AUC (Layer {best_layer_by_auc})')

    fig.tight_layout()
    plt.title('EBM Performance Across Decoder Layers')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    
    plt.savefig(output_image_path)
    print(f"\n性能可视化图已保存至: {output_image_path}")


if __name__ == "__main__":
    if os.path.exists("activation/hidden_states.pt") and os.path.exists("activation/labels.pt"):
        find_best_layer()
    else:
        print("错误：缺失激活值文件。请确认 'activation' 文件夹及其中文件存在。")