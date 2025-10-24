import torch
import json
from tqdm import tqdm
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForMaskedLM
from torch import nn
from torch.distributions.categorical import Categorical
import numpy as np
import os
import random
import math
# --- 新增引用 ---
from sentence_transformers import SentenceTransformer

# --- Part 1: EBM和评分模型定义 ---

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

    @torch.no_grad()
    def decision_function(self, x):
        device = next(self.parameters()).device
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(device)
        output = self.forward(x)
        # 返回正类的分数，分数越高代表越可能是攻击
        return output[:, 1].cpu().numpy()

class ActivationScoreModel:
    def __init__(self, act_llm_dir, ebm_path, device_map):
        print("加载代理模型 (用于激活值评分)...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            act_llm_dir, torch_dtype="auto", device_map=device_map
        )
        self.tokenizer = AutoTokenizer.from_pretrained(act_llm_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model_device = self.llm.device
        
        print(f"从 {ebm_path} 加载EBM模型...")
        ckpt = torch.load(ebm_path)
        self.classifier = EBM(ckpt["input_dim"], ckpt["hidden_dims"], ckpt["output_dim"]).to(model_device)
        self.classifier.load_state_dict(ckpt["ebm_ckpt"])
        self.classifier.eval()
        
        self.layer_idx = ckpt["best_layer_idx"]
        print(f"EBM加载成功。将使用第 {self.layer_idx} 层的激活值，并已移动到设备: {model_device}")

    def build_msg(self, system_prompt, user_prompts_batch):
        msgs_batch = []
        for user_prompt in user_prompts_batch:
            msgs_batch.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
        
        # 使用 apply_chat_template，它会自动处理指令格式
        text_batch = [self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) for msgs in msgs_batch]
        
        encoded_msgs_dict = self.tokenizer(
            text_batch, return_tensors="pt", padding=True
        ).to(self.llm.device)
        return encoded_msgs_dict

    @torch.no_grad()
    def get_last_token_activations(self, encoded_msgs_dict):
        outputs = self.llm(**encoded_msgs_dict, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        lengths = encoded_msgs_dict.attention_mask.sum(dim=1)
        target_layer_hs = hidden_states[self.layer_idx + 1].detach()
        last_token_activations = []
        for i, length in enumerate(lengths):
            last_token_activation = target_layer_hs[i, length - 1, :].cpu().float().numpy()
            last_token_activations.append(last_token_activation)
        return np.stack(last_token_activations)

    def __call__(self, system_prompt, str_batch):
        encoded_msgs_dict = self.build_msg(system_prompt, str_batch)
        activations = self.get_last_token_activations(encoded_msgs_dict)
        # EBM返回的是正类分数，我们希望最大化它
        scores = self.classifier.decision_function(activations)
        return scores, scores, None

# --- 新增: 语义相似度评分模型 (来自 expert_model.py) ---
class EmbScoreModel:
    def __init__(self, model_path, device):
        print(f"加载语义相似度模型: {model_path}...")
        self.model_emb = SentenceTransformer(model_path, device=device)
        self.device = device

    @torch.no_grad()
    def __call__(self, str_batch, str_batch_original):
        # 将文本解码为字符串列表
        str_batch_list = list(str_batch)
        str_batch_original_list = list(str_batch_original)

        emb_batch = self.model_emb.encode(str_batch_list, convert_to_tensor=True, device=self.device)
        emb_batch_original = self.model_emb.encode(str_batch_original_list, convert_to_tensor=True, device=self.device)
        
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        dist = cos(emb_batch, emb_batch_original).cpu().detach().numpy()
        # 我们希望相似度尽可能高，所以直接返回相似度分数
        return dist

# --- Part 2: MCMC采样器 ---

class TokenwiseSampler:
    def __init__(self, config, activation_model):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.activation_model = activation_model
        # --- 新增: 初始化 EmbScoreModel ---
        self.emb_model = EmbScoreModel(config['emb_model_path'], device=self.device)

        print("加载提议模型 (BERT)...")
        self.model_proposal = AutoModelForMaskedLM.from_pretrained(config['proposal_model_path']).to(self.device)
        self.model_proposal.eval()
        self.tokenizer_proposal = AutoTokenizer.from_pretrained(config['proposal_model_path'])
        
        self.mask_id = self.tokenizer_proposal.mask_token_id
        self.temperature = config['temperature']
        self.batch_original = None

    # --- 新增: MLM流畅性评分方法 (来自 block_sample.py) ---
    def energy_score_mlm(self, batch):
        seq_len = batch.shape[1] - 2
        posns = list(range(1, seq_len + 1))
        
        log_probs = torch.zeros(batch.shape[0], device=self.device)
        
        for posn in posns:
            old_wrd = batch[:, posn].clone()
            batch[:, posn] = self.mask_id
            
            with torch.no_grad():
                logits = self.model_proposal(batch).logits[:, posn, :]
            
            log_softmax_vals = torch.log_softmax(logits, dim=-1)
            
            # 收集每个样本在posn位置的原始token的log probability
            log_probs += log_softmax_vals[torch.arange(batch.shape[0]), old_wrd]
            
            batch[:, posn] = old_wrd
            
        # 返回平均log probability, 越高代表越流畅
        return (log_probs / seq_len).cpu().numpy()

    # --- 核心修改: 重构energy_score为多目标函数 ---
    def energy_score(self, system_prompt, batch):
        str_batch = self.tokenizer_proposal.batch_decode(batch, skip_special_tokens=True)
        str_batch_original = self.tokenizer_proposal.batch_decode(self.batch_original, skip_special_tokens=True)

        # 1. 激活值分数 (攻击性) - gamma
        act_raw, _, _ = self.activation_model(system_prompt, str_batch)
        
        # 2. MLM分数 (流畅性) - beta
        mlm_raw = self.energy_score_mlm(batch.clone()) # 使用clone防止被修改
        
        # 3. 语义相似度分数 (相似性) - theta
        emb_dist = self.emb_model(str_batch, str_batch_original)

        # 组合分数：我们希望最大化这个总分
        # act_raw: 越高越好
        # mlm_raw: 越高越好 (越接近0)
        # emb_dist: 越高越好 (越接近1)
        total_raw_score = (self.config['gamma'] * act_raw + 
                           self.config['beta'] * mlm_raw + 
                           self.config['theta'] * emb_dist)

        scores_info = {"act_raw": act_raw, "mlm_raw": mlm_raw, "emb_dist": emb_dist}
        return total_raw_score, total_raw_score, scores_info


    @torch.no_grad()
    def run(self, system_prompt, seed_prompt):
        seed_tokens = self.tokenizer_proposal.encode(seed_prompt, truncation=True, max_length=128)
        batch = torch.tensor([seed_tokens] * self.config['batch_size']).to(self.device)
        self.batch_original = batch.clone() # 保存原始batch
        
        seq_len = batch.shape[1] - 2 
        if seq_len <= 0:
            print(f"警告：种子提示 '{seed_prompt}' 过短，跳过优化。")
            return seed_prompt, -np.inf
        posns = list(range(1, seq_len + 1))

        best_samples = [seed_prompt] * self.config['batch_size']
        best_scores = np.full(self.config['batch_size'], -np.inf)
        
        pbar = tqdm(range(self.config['max_iter']), desc="MCMC Iterations")
        for iter_i in pbar:
            k = max(1, int(seq_len * self.config['token_portion']))
            curr_posns = random.sample(posns, k)
            
            old_raw, _, old_scores = self.energy_score(system_prompt, batch)

            old_wrd = batch[:, curr_posns].clone()
            batch[:, curr_posns] = self.mask_id
            
            output_logits = self.model_proposal(batch).logits[:, curr_posns, :] / self.temperature
            output_probs = torch.softmax(output_logits, dim=-1)
            
            d = Categorical(output_probs)
            new_wrd = d.sample()
            
            x_prob = torch.ones(self.config['batch_size'], device=self.device)
            x_bar_prob = torch.ones(self.config['batch_size'], device=self.device)
            for i in range(len(curr_posns)):
                for j in range(self.config['batch_size']):
                    x_prob[j] *= output_probs[j, i, old_wrd[j, i]]
                    x_bar_prob[j] *= output_probs[j, i, new_wrd[j, i]]
            
            batch[:, curr_posns] = new_wrd
            
            new_raw, _, new_scores = self.energy_score(system_prompt, batch)
            
            accept_prob = torch.min(
                torch.ones(1, device=self.device),
                torch.exp(torch.tensor(new_raw - old_raw, device=self.device, dtype=torch.float32)) * (x_prob / (x_bar_prob + 1e-9))
            )
            
            acc_mask = torch.bernoulli(accept_prob).bool()
            
            temp_batch = batch.clone()
            temp_batch[:, curr_posns][~acc_mask] = old_wrd[~acc_mask]
            batch = temp_batch

            current_scores = np.where(acc_mask.cpu().numpy(), new_raw, old_raw)
            for i in range(self.config['batch_size']):
                if current_scores[i] > best_scores[i]:
                    best_scores[i] = current_scores[i]
                    best_samples[i] = self.tokenizer_proposal.decode(batch[i], skip_special_tokens=True)
            
            # 更新进度条显示
            pbar.set_postfix({
                "score": f"{np.mean(current_scores):.2f}",
                "act": f"{np.mean(new_scores['act_raw']):.2f}",
                "mlm": f"{np.mean(new_scores['mlm_raw']):.2f}",
                "emb": f"{np.mean(new_scores['emb_dist']):.2f}",
                "accept": f"{acc_mask.float().mean().item():.2f}"
            })

        final_best_idx = np.argmax(best_scores)
        final_best_sample = best_samples[final_best_idx]
        final_best_score = best_scores[final_best_idx]

        return final_best_sample, final_best_score

if __name__ == "__main__":
    # --- Part 3: 配置和执行流程 ---
    CONFIG = {
        "surrogate_model_path": "model/Qwen2.5-7B-Instruct",
        "proposal_model_path": "model/bert-base-uncased",
        "ebm_model_path": "ebm_training_outputs/best_ebm_model.pt",
        # --- 新增模型路径 ---
        "emb_model_path": "./model/all-MiniLM-L6-v2", 
        
        "json_dataset_path": "raw_data.json",
        "output_path": "optimized_prompts.jsonl",
        
        # --- 核心超参数: 攻击性, 流畅性, 相似性的权重 ---
        "gamma": 0.8,  # EBM 激活值分数 (攻击性)
        "beta": 0.5,   # MLM 分数 (流畅性)
        "theta": 2.0,  # 语义相似度分数 (与原句的相似性)
        
        "max_iter": 10,             # 大幅增加迭代次数以适应多目标优化
        "batch_size": 20,
        "token_portion": 0.1,        # 每次修改10%的token
        "temperature": 0.7,          # 适当调整温度
    }

    print("\n--- 开始执行MCMC优化流程 ---")
    
    device_map_setting = "auto"
    
    activation_scorer = ActivationScoreModel(
        CONFIG['surrogate_model_path'], CONFIG['ebm_model_path'], device_map=device_map_setting
    )
    sampler = TokenwiseSampler(CONFIG, activation_scorer)
    
    with open(CONFIG['json_dataset_path'], 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    with open(CONFIG['output_path'], 'w', encoding='utf-8') as f_out:
        # 仅处理前5个样本作为示例
        for item in tqdm(dataset[:5], desc="正在优化提示词"):
            system_prompt = item['user_prompt']
            seed_prompt = item['inject_prompt']
            
            print(f"\n正在优化ID: {item['id']}...")
            print(f"原始注入提示: {seed_prompt}")
            
            optimized_prompt, final_score = sampler.run(system_prompt, seed_prompt)
            
            print(f"优化后提示: {optimized_prompt}")
            print(f"最终分数: {final_score:.4f}")
            
            result = {
                "id": item['id'], "original_prompt": seed_prompt,
                "optimized_prompt": optimized_prompt, "final_energy_score": float(final_score),
            }
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            
    print("\n" + "="*30)
    print("所有优化任务完成！")
    print(f"结果已保存至: {CONFIG['output_path']}")
    print("="*30)