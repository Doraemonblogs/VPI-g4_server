import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, roc_auc_score
from pytorch_lightning import seed_everything
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 定义 EBM 模型 ---
class EBM(nn.Module):
    def __init__(self, input_dim, hidden_dim1=1024, hidden_dim2=256, output_dim=2):
        super(EBM, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim2, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# --- 2. 训练和验证函数 ---
def train_and_evaluate_layer(all_hs, all_labels, layer_idx, device):
    """
    为指定层的激活值训练和评估一个 EBM 分类器。
    """
    # --- 超参数设置 ---
    total_epochs = 120
    warmup_epochs = 5
    batch_size = 256
    lr = 3e-4
    min_lr = lr / 100
    patience = 20
    patience_counter = patience

    # --- 模型初始化 ---
    model = EBM(input_dim=all_hs.shape[-1]).to(device)
    print("\n" + "="*50)
    print(f"开始训练第 {layer_idx} 层...")
    print("="*50)

    # --- 数据准备 ---
    hs = all_hs[:, layer_idx, :].clone()
    labels = all_labels.clone()

    good_idx = labels != -1
    labels = labels[good_idx].long()
    hs = hs[good_idx].float()

    pos_nums = (labels == 1).sum()
    neg_nums = (labels == 0).sum()
    if neg_nums == 0 or pos_nums == 0:
        print("警告：数据中只存在一个类别，将不使用类别权重。")
        weight = None
    else:
        total = pos_nums + neg_nums
        weight = torch.tensor([total / (2 * neg_nums), total / (2 * pos_nums)], device=device)
        print(f"数据分布: 正样本数={pos_nums}, 负样本数={neg_nums}")

    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    dataset = TensorDataset(hs, labels)
    val_size = max(1, int(len(dataset) * 0.2)) if len(dataset) > 5 else 1
    train_size = len(dataset) - val_size

    if train_size == 0 or val_size == 0:
        print("错误：数据集太小，无法划分为训练集和验证集。")
        return None

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    effective_batch_size = min(batch_size, train_size)
    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True)
    print(f"数据集划分: 训练集大小={train_size}, 验证集大小={val_size}")

    @torch.no_grad()
    def validate(model, val_data):
        model.eval()
        if len(val_data) == 0: return 0.0, 0.5, float('inf')
        val_inputs, val_labels = val_data[:]
        val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
        outputs = model(val_inputs)
        loss = criterion(outputs, val_labels)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds_class = torch.argmax(outputs, dim=1)
        acc = accuracy_score(val_labels.cpu(), preds_class.cpu())
        auc = roc_auc_score(val_labels.cpu().numpy(), probs.cpu().numpy()) if len(np.unique(val_labels.cpu().numpy())) > 1 else 0.5
        return acc, auc, loss.item()

    # --- 训练循环 ---
    best_val_loss = float('inf')
    best_auc = 0
    best_acc = 0
    best_model_state = None

    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(total_epochs - warmup_epochs) * len(train_loader), eta_min=min_lr)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs * len(train_loader))
    seq_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs * len(train_loader)])

    for epoch in tqdm(range(total_epochs), desc=f"Layer {layer_idx} Training"):
        model.train()
        running_train_loss = 0.0
        for batch_hs, batch_labels in train_loader:
            batch_hs, batch_labels = batch_hs.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            output = model(batch_hs)
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()
            seq_scheduler.step()
            running_train_loss += loss.item() * batch_hs.size(0)
        
        epoch_train_loss = running_train_loss / len(train_dataset)
        acc, auc, val_loss = validate(model, val_dataset)

        if epoch % 10 == 0:
             print(f"Epoch {epoch} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f} | Val AUC: {auc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_auc = auc
            best_acc = acc
            best_model_state = model.state_dict()
            patience_counter = patience
        else:
            patience_counter -= 1
            if patience_counter == 0:
                print(f"在 Epoch {epoch} 触发早停！")
                break

    print("-" * 50)
    print(f"第 {layer_idx} 层训练完成。")
    print(f"最佳结果 -> Val Loss: {best_val_loss:.4f} | Val Acc: {best_acc:.4f} | Val AUC: {best_auc:.4f}")
    print("-" * 50)

    return {
        "layer_idx": layer_idx,
        "best_val_loss": best_val_loss,
        "best_acc": best_acc,
        "best_auc": best_auc,
        "best_model_state": best_model_state,
    }

# --- 3. 主执行逻辑 ---
if __name__ == "__main__":
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # --- 创建结果文件夹 ---
    output_dir = "ebm_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"结果将保存至: {output_dir}")

    hs_path = "activation/hidden_states.pt"
    labels_path = "activation/labels.pt"

    print("正在加载预处理的数据...")
    try:
        all_hs = torch.load(hs_path, map_location='cpu')
        all_labels = torch.load(labels_path, map_location='cpu')
        print("数据加载成功！")
    except FileNotFoundError:
        print(f"错误: 找不到数据文件。请确保 '{hs_path}' 和 '{labels_path}' 存在。")
        exit()

    # --- 完整的训练和绘图流程 ---
    num_layers = all_hs.shape[1]
    layer_wise_results = []
    for i in range(num_layers):
        result = train_and_evaluate_layer(all_hs, all_labels, layer_idx=i, device=device)
        if result:
            layer_wise_results.append(result)

    if layer_wise_results:
        best_layer_by_auc = sorted(layer_wise_results, key=lambda x: x['best_auc'], reverse=True)[0]

        layer_indices = [res['layer_idx'] for res in layer_wise_results]
        all_best_aucs = [res['best_auc'] for res in layer_wise_results]
        all_best_losses = [res['best_val_loss'] for res in layer_wise_results]

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(12, 7))
        plt.plot(layer_indices, all_best_aucs, marker='o', markersize=5, linestyle='-', color='teal', label='Validate AUC')
        plt.plot(layer_indices, all_best_losses, marker='o', markersize=5, linestyle='-', color='lightcoral', label='Validate Loss')
        best_layer_idx = best_layer_by_auc['layer_idx']
        plt.axvline(x=best_layer_idx, color='lightcoral', linestyle='--', linewidth=1.5)
        plt.text(num_layers, (max(all_best_aucs) + min(all_best_losses))/2, f'  Best EBM\n  Layer: {best_layer_idx}\n  AUC: {best_layer_by_auc["best_auc"]:.3f}', color='black', fontsize=10, verticalalignment='center')
        plt.title('EBM Performance Across Layers', fontsize=16)
        plt.xlabel('Layer Index', fontsize=12)
        plt.ylabel('Metric Value', fontsize=12)
        plt.xticks(range(num_layers))
        plt.legend(fontsize=12, loc='best')
        plt.grid(True)
        plt.subplots_adjust(right=0.85)
        
        # --- 将结果保存到指定文件夹 ---
        plot_save_path = os.path.join(output_dir, "layer_performance_single_axis.png")
        plt.savefig(plot_save_path, bbox_inches='tight')
        print(f"\n📈 性能分析图已保存至: {plot_save_path}")

        best_model_path = os.path.join(output_dir, f"best_ebm_layer_{best_layer_idx}.pth")
        torch.save(best_layer_by_auc['best_model_state'], best_model_path)
        print(f"✅ 已将 AUC 最佳的模型权重保存至: {best_model_path}")