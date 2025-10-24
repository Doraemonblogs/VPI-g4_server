import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, roc_auc_score
from pytorch_lightning import seed_everything
import os
from tqdm import tqdm
import numpy as np
import argparse
import matplotlib.pyplot as plt

"""
train_layer.py

这是一个用于训练指定层 EBM 模型的脚本，并可视化该层训练过程中的指标变化。
通过命令行参数指定要训练的层索引，并将训练好的模型和指标图表保存到结果文件夹。

如何运行:
python train_layer.py --layer 21
"""

# --- 1. 定义 EBM 模型 (与之前完全相同) ---
class EBM(nn.Module):
    def __init__(self, input_dim, hidden_dim1=1024, hidden_dim2=256, output_dim=2):
        super(EBM, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim2, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# --- 新增: 绘制和保存指标图表的函数 ---
def plot_and_save_metrics(metrics, layer_idx, output_dir):
    """
    根据记录的指标数据，绘制图表并保存。
    """
    epochs = range(len(metrics['train_loss']))
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # --- 上图：损失曲线 ---
    axs[0].plot(epochs, metrics['train_loss'], 'o-', label='Training Loss', color='lightcoral', markersize=4)
    axs[0].plot(epochs, metrics['val_loss'], 'o-', label='Validation Loss', color='teal', markersize=4)
    axs[0].set_ylabel('Loss', fontsize=12)
    axs[0].set_title(f'Layer {layer_idx}: Training & Validation Loss', fontsize=14)
    axs[0].legend(fontsize=10)
    axs[0].grid(True)
    
    # --- 下图：准确率和AUC曲线 ---
    axs[1].plot(epochs, metrics['val_acc'], 'o-', label='Validation Accuracy', color='skyblue', markersize=4)
    axs[1].plot(epochs, metrics['val_auc'], 'o-', label='Validation AUC', color='slateblue', markersize=4)
    axs[1].set_xlabel('Epochs', fontsize=12)
    axs[1].set_ylabel('Metric Value', fontsize=12)
    axs[1].set_title(f'Layer {layer_idx}: Validation Accuracy & AUC', fontsize=14)
    axs[1].legend(fontsize=10)
    axs[1].grid(True)

    plt.tight_layout()
    
    plot_save_path = os.path.join(output_dir, f"metrics_layer_{layer_idx}.png")
    plt.savefig(plot_save_path, bbox_inches='tight')
    print(f"📈 指标变化图已保存至: {plot_save_path}")
    plt.close() # 关闭图像，防止在Jupyter等环境中重复显示

# --- 2. 训练和验证函数 (修改版) ---
def train_and_evaluate_layer(all_hs, all_labels, layer_idx, device, output_dir):
    """
    为指定层的激活值训练和评估一个 EBM 分类器，并记录、绘制指标。
    """
    # --- 超参数设置 (与之前相同) ---
    total_epochs = 120
    warmup_epochs = 5
    batch_size = 256
    lr = 3e-4
    min_lr = lr / 100
    patience = 20
    patience_counter = patience

    # --- 模型初始化 (与之前相同) ---
    model = EBM(input_dim=all_hs.shape[-1]).to(device)
    print("\n" + "="*50)
    print(f"开始训练第 {layer_idx} 层...")
    print("="*50)

    # --- 数据准备 (与之前相同) ---
    hs = all_hs[:, layer_idx, :].clone()
    labels = all_labels.clone()
    good_idx = labels != -1
    labels = labels[good_idx].long()
    hs = hs[good_idx].float()

    pos_nums = (labels == 1).sum()
    neg_nums = (labels == 0).sum()
    if neg_nums == 0 or pos_nums == 0:
        weight = None
    else:
        total = pos_nums + neg_nums
        weight = torch.tensor([total / (2 * neg_nums), total / (2 * pos_nums)], device=device)

    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    dataset = TensorDataset(hs, labels)
    val_size = max(1, int(len(dataset) * 0.2)) if len(dataset) > 5 else 1
    train_size = len(dataset) - val_size

    if train_size == 0 or val_size == 0:
        return None

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    effective_batch_size = min(batch_size, train_size)
    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True)
    
    @torch.no_grad()
    def validate(model, val_data):
        model.eval()
        if len(val_data) == 0: return 0.0, 0.5, float('inf')
        val_inputs, val_labels = val_data[:]
        val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
        outputs = model(val_inputs)
        loss = criterion(outputs, val_labels)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        acc = accuracy_score(val_labels.cpu(), torch.argmax(outputs, dim=1).cpu())
        auc = roc_auc_score(val_labels.cpu().numpy(), probs.cpu().numpy()) if len(np.unique(val_labels.cpu().numpy())) > 1 else 0.5
        return acc, auc, loss.item()

    # --- 训练循环 ---
    best_val_loss = float('inf')
    best_model_state = None
    
    # --- 新增: 用于记录指标的列表 ---
    history = {
        'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_auc': []
    }

    # ... 学习率调度器设置 (与之前相同) ...
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
        
        # --- 新增: 记录当前 epoch 的指标 ---
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(acc)
        history['val_auc'].append(auc)

        if epoch % 10 == 0:
             print(f"Epoch {epoch} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f} | Val AUC: {auc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = patience
        else:
            patience_counter -= 1
            if patience_counter == 0:
                print(f"在 Epoch {epoch} 触发早停！")
                break
    
    print("-" * 50)
    print(f"第 {layer_idx} 层训练完成。")

    # --- 新增: 调用绘图函数 ---
    if history['train_loss']: # 确保训练至少进行了一个epoch
        plot_and_save_metrics(history, layer_idx, output_dir)

    return best_model_state

# --- 3. 主执行逻辑 (修改版) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为模型的特定层训练EBM分类器并可视化训练过程。")
    parser.add_argument("--layer", type=int, required=True, help="需要训练的模型层的索引 (例如: 23)")
    parser.add_argument("--output_dir", type=str, default="specific_layer", help="保存结果的文件夹路径")
    args = parser.parse_args()

    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"结果将保存至: {args.output_dir}")

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

    num_layers = all_hs.shape[1]
    if not (0 <= args.layer < num_layers):
        print(f"错误: 无效的层索引 {args.layer}。索引必须在 0 到 {num_layers - 1} 之间。")
        exit()

    # --- 执行训练 ---
    best_model_state = train_and_evaluate_layer(
        all_hs, all_labels, 
        layer_idx=args.layer, 
        device=device,
        output_dir=args.output_dir
    )

    # --- 保存结果 ---
    if best_model_state is not None:
        model_save_path = os.path.join(args.output_dir, f"ebm_layer_{args.layer}.pth")
        torch.save(best_model_state, model_save_path)
        print(f"✅ 已将第 {args.layer} 层的最佳模型权重保存至: {model_save_path}")
    else:
        print(f"第 {args.layer} 层的训练未能成功完成，未保存任何模型。")