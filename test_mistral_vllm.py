import torch
from vllm import LLM, SamplingParams

# --- 1. 配置模型路径 ---
# 请确保此路径指向你的 Mistral-7B-Instruct-v0.2 模型文件夹
MODEL_PATH = "model/Mistral-7B-Instruct-v0.2"

def main():
    """
    一个独立的、最小化的脚本，用于测试 Mistral-7B-Instruct-v0.2 与 vLLM 的集成。
    """
    print("--- 正在测试 Mistral-7B-Instruct-v0.2 with vLLM ---")
    print(f"Python Torch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"当前CUDA设备: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # --- 2. 加载模型 ---
    try:
        print(f"\n正在从路径加载模型: {MODEL_PATH}...")
        # tensor_parallel_size=1 表示在单张GPU上运行
        # gpu_memory_utilization 可以根据你的显存情况调整
        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9
        )
        print("模型加载成功！")
    except Exception as e:
        print(f"\n[错误] 模型加载失败: {e}")
        print("请检查：")
        print(f"1. 路径 '{MODEL_PATH}' 是否正确。")
        print("2. 是否有足够的GPU显存。")
        return

    # --- 3. 准备聊天提示 (应用Mistral的专属模板) ---
    print("\n--- 准备聊天提示 ---")
    
    # Mistral的聊天模板需要通过tokenizer来正确应用
    tokenizer = llm.get_tokenizer()
    
    # 这是一个标准的对话格式
    messages = [
        {"role": "user", "content": "Hello! Can you tell me a short story about a robot who learns to paint?"}
    ]

    # apply_chat_template 会自动将上面的对话转换为 Mistral 需要的 "<s>[INST]...[/INST]" 格式
    try:
        # tokenize=False 表示我们想要得到字符串而不是token IDs
        # add_generation_prompt=True 表示在末尾添加助手的起始标记，让模型知道该开始生成了
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print("成功应用聊天模板！")
        print(f"应用模板后的输入:\n---\n{formatted_prompt}\n---")
    except Exception as e:
        print(f"\n[错误] 应用聊天模板失败: {e}")
        print("这通常意味着模型的 'tokenizer_config.json' 文件可能存在问题或缺失。")
        return

    # --- 4. 配置采样参数并生成文本 ---
    print("\n--- 开始生成响应 ---")
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=256  # 生成的最大长度
    )

    # vLLM的generate方法接收一个字符串列表
    outputs = llm.generate([formatted_prompt], sampling_params)

    # --- 5. 打印结果 ---
    print("\n--- 模型输出 ---")
    for output in outputs:
        # 'result.prompt' 显示了最终进入模型的完整文本
        # 'result.outputs[0].text' 是模型生成的文本
        generated_text = output.outputs[0].text
        print(f"原始提示: {output.prompt[:100]}...") # 只显示部分提示
        print("-" * 20)
        print(f"生成文本:\n{generated_text.strip()}")
        print("-" * 20)
        print(f"生成完毕原因: {output.outputs[0].finish_reason}")
    
    print("\n--- 测试完成 ---")

if __name__ == "__main__":
    main()