import torch
import json
import os
from tqdm import tqdm
from modelscope import AutoModelForCausalLM, AutoTokenizer

def prepare_and_save_activations():
    """
    加载模型，处理数据，提取并保存所有层的激活值和标签。
    """
    # --- 1. 设置路径 ---
    local_model_path = "model/Qwen2.5-7B-Instruct"
    data_file_path = "output/email_data_exfiltration_labled.json"
    output_dir = "activation"
    hidden_states_path = os.path.join(output_dir, "hidden_states.pt")
    labels_path = os.path.join(output_dir, "labels.pt")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # --- 2. 加载模型和分词器 ---
    print(f"正在从本地路径加载模型: {local_model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    print("模型和分词器加载完成。")

    # --- 3. 加载数据集 ---
    print(f"正在加载数据集: {data_file_path}")
    with open(data_file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    print(f"数据集加载完成，共 {len(raw_data)} 条数据。")

    # --- 4. 循环处理数据并提取激活值 ---
    all_hidden_states = []
    all_labels = []

    print("开始处理数据并提取激活值...")
    # 使用tqdm显示进度条
    for item in tqdm(raw_data, desc="Processing data"):
        system_prompt = item.get("user_prompt", "")
        user_prompt = item.get("inject_prompt", "")
        success_label = item.get("success", False)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 应用聊天模板并进行分词
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # 执行前向传播，并要求输出所有隐藏状态
        with torch.no_grad():
            outputs = model(**model_inputs, output_hidden_states=True)

        # outputs.hidden_states 是一个元组，包含了 embedding 层和所有 decoder 层的输出
        # 长度为 num_layers + 1。我们只取 28 个 decoder 层的输出。
        # 每个元素的形状为 (batch_size, sequence_length, hidden_size)
        # 我们提取每个层中最后一个 token 的激活值
        layer_activations = []
        # 第0个是embedding层的输出，我们从第1个开始取，直到最后一个
        for layer_hs in outputs.hidden_states[1:]:
            # 提取最后一个 token 的激活值，形状为 (hidden_size)
            last_token_hs = layer_hs[0, -1, :].cpu()
            layer_activations.append(last_token_hs)
        
        # 将当前样本的所有层激活值堆叠起来，形状为 (num_layers, hidden_size)
        # 即 (28, 3584)
        stacked_activations = torch.stack(layer_activations)
        
        all_hidden_states.append(stacked_activations)
        # 将布尔值标签转换为整数 (1 for True, 0 for False)
        all_labels.append(1 if success_label else 0)

    # --- 5. 整合并保存结果 ---
    print("所有数据处理完毕，正在保存结果...")
    
    # 将所有样本的隐藏状态堆叠成一个大的张量
    # 最终形状为 (num_samples, num_layers, hidden_size)
    # 即 (1000, 28, 3584)
    final_hidden_states = torch.stack(all_hidden_states)
    
    # 将标签列表转换为张量
    final_labels = torch.tensor(all_labels, dtype=torch.long)

    # 保存到文件
    torch.save(final_hidden_states, hidden_states_path)
    torch.save(final_labels, labels_path)

    print("-" * 30)
    print("数据准备完成！")
    print(f"隐藏状态已保存至: {hidden_states_path}")
    print(f"保存的隐藏状态张量形状: {final_hidden_states.shape}")
    print(f"标签已保存至: {labels_path}")
    print(f"保存的标签张量形状: {final_labels.shape}")
    print("-" * 30)


if __name__ == "__main__":
    prepare_and_save_activations()