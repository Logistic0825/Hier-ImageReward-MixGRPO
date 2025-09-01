"""
层次化奖励模型，集成到MixGRPO项目中
文件位置：fastvideo/models/reward_model/hierarchical_reward.py
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import ImageReward as reward
from PIL import Image
import tempfile
import os
from dataclasses import dataclass


@dataclass
class HierarchicalRewardConfig:
    """层次化奖励配置"""
    # 权重配置 - 不同时间步的权重分布
    start_weights: List[float] = None  # [CLIP, BLIP, Aesthetic] 早期权重
    end_weights: List[float] = None    # [CLIP, BLIP, Aesthetic] 后期权重
    transition_method: str = 'cosine'  # 权重过渡方法: linear, cosine, sigmoid, exponential
    grpo_update_interval: int = 5      # 每多少步进行一次GRPO更新
    
    def __post_init__(self):
        if self.start_weights is None:
            self.start_weights = [0.5, 0.5, 0.0]  # 早期关注全局结构和语义对齐
        if self.end_weights is None:
            self.end_weights = [0.2, 0.2, 0.6]    # 后期关注美学质量


class HierarchicalRewardModel:
    """层次化奖励模型"""
    
    def __init__(self, device: str = "cuda", config: HierarchicalRewardConfig = None):
        self.device = device
        self.config = config or HierarchicalRewardConfig()
        
        # 加载三个细粒度奖励模型
        print("Loading hierarchical reward models...")
        self.clip_score_rm = reward.load_score(name="CLIP", device=device)
        self.blip_score_rm = reward.load_score(name="BLIP", device=device)
        self.aes_score_rm = reward.load_score(name="Aesthetic", device=device)
        print("Hierarchical reward models loaded successfully.")
        
        # 存储历史奖励信息用于调试
        self.reward_history = []
    
    def smooth_weight_transition(self, t: float) -> List[float]:
        """
        设计平滑权重过渡函数
        Args:
            t: 当前时间步比例 (0.0 到 1.0)
        Returns:
            三个奖励模型的权重 [clip_weight, blip_weight, aesthetic_weight]
        """
        start_weights = np.array(self.config.start_weights)
        end_weights = np.array(self.config.end_weights)
        
        method = self.config.transition_method
        if method == 'linear':
            alpha = t
        elif method == 'cosine':
            alpha = (1 - np.cos(np.pi * t)) / 2
        elif method == 'sigmoid':
            alpha = 1 / (1 + np.exp(-10 * (t - 0.5)))
        elif method == 'exponential':
            alpha = t ** 2
        else:
            alpha = t
        
        weights = start_weights + alpha * (end_weights - start_weights)
        weights = weights / np.sum(weights)  # 归一化
        
        return weights.tolist()
    
    def compute_reward(self, prompt: str, image: Image.Image, 
                      current_step: int, total_steps: int) -> Dict[str, float]:
        """
        计算层次化奖励分数
        Args:
            prompt: 文本提示词
            image: 生成的图像
            current_step: 当前步数
            total_steps: 总步数
        Returns:
            包含各种奖励分数的字典
        """
        # 计算时间步比例
        t = current_step / total_steps
        weights = self.smooth_weight_transition(t)
        
        # 保存临时图像文件用于评估
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            image.save(tmp_file, format='JPEG')
            image_path = tmp_file.name
        
        try:
            # 计算三个细粒度分数
            clip_score = self.clip_score_rm.score(prompt, image_path)
            blip_score = self.blip_score_rm.score(prompt, image_path)
            aes_score = self.aes_score_rm.score(prompt, image_path)
            
            # 计算加权综合分数
            hierarchical_score = (weights[0] * clip_score +
                                  weights[1] * blip_score +
                                  weights[2] * aes_score)
            
            result = {
                'clip_score': float(clip_score),
                'blip_score': float(blip_score),
                'aesthetic_score': float(aes_score),
                'hierarchical_score': float(hierarchical_score),
                'weights': weights,
                'timestep_ratio': t,
                'current_step': current_step,
                'should_update_grpo': (current_step + 1) % self.config.grpo_update_interval == 0
            }
            
            # 记录历史用于分析
            self.reward_history.append(result.copy())
            
            return result
            
        finally:
            # 清理临时文件
            if os.path.exists(image_path):
                os.unlink(image_path)
    
    def get_reward_for_grpo(self, prompt: str, image: Image.Image, 
                           current_step: int, total_steps: int) -> float:
        """
        为GRPO训练提供单一奖励分数
        """
        reward_info = self.compute_reward(prompt, image, current_step, total_steps)
        return reward_info['hierarchical_score']
    
    def should_update_grpo(self, current_step: int) -> bool:
        """
        判断是否应该在当前步骤进行GRPO更新
        """
        return (current_step + 1) % self.config.grpo_update_interval == 0
    
    def get_reward_breakdown(self) -> Dict[str, List[float]]:
        """
        获取奖励分解历史，用于分析和可视化
        """
        if not self.reward_history:
            return {}
        
        breakdown = {
            'steps': [r['current_step'] for r in self.reward_history],
            'clip_scores': [r['clip_score'] for r in self.reward_history],
            'blip_scores': [r['blip_score'] for r in self.reward_history],
            'aesthetic_scores': [r['aesthetic_score'] for r in self.reward_history],
            'hierarchical_scores': [r['hierarchical_score'] for r in self.reward_history],
            'clip_weights': [r['weights'][0] for r in self.reward_history],
            'blip_weights': [r['weights'][1] for r in self.reward_history],
            'aesthetic_weights': [r['weights'][2] for r in self.reward_history],
        }
        return breakdown


class HierarchicalRewardCallback:
    """
    用于在生成过程中调用层次化奖励评估的回调函数
    集成到Flux pipeline的callback_on_step_end中
    """
    
    def __init__(self, reward_model: HierarchicalRewardModel, 
                 prompt: str, total_steps: int, height: int, width: int):
        self.reward_model = reward_model
        self.prompt = prompt
        self.total_steps = total_steps
        self.height = height
        self.width = width
        self.intermediate_rewards = []
    
    def __call__(self, pipe, step: int, timestep, callback_kwargs) -> Dict:
        """
        回调函数，在每个时间步后被调用
        """
        # 只在指定的步骤进行评估（例如每5步）
        if (step + 1) % self.reward_model.config.grpo_update_interval == 0:
            latents = callback_kwargs.get("latents")
            if latents is not None:
                try:
                    # 解码latents为图像
                    with torch.no_grad():
                        latents_unpacked = pipe._unpack_latents(
                            latents, self.height, self.width, pipe.vae_scale_factor
                        )
                        scaled_latents = (latents_unpacked / pipe.vae.config.scaling_factor) + \
                                       pipe.vae.config.shift_factor
                        image_tensor = pipe.vae.decode(
                            scaled_latents.to(pipe.vae.dtype), return_dict=False
                        )[0].cpu()
                        image = pipe.image_processor.postprocess(
                            image_tensor, output_type="pil"
                        )[0]
                    
                    # 计算层次化奖励
                    reward_info = self.reward_model.compute_reward(
                        self.prompt, image, step, self.total_steps
                    )
                    
                    self.intermediate_rewards.append(reward_info)
                    
                    print(f"Step {step+1}/{self.total_steps}: "
                          f"Hierarchical Score = {reward_info['hierarchical_score']:.4f} "
                          f"(CLIP:{reward_info['clip_score']:.3f}, "
                          f"BLIP:{reward_info['blip_score']:.3f}, "
                          f"Aesthetic:{reward_info['aesthetic_score']:.3f})")
                    
                except Exception as e:
                    print(f"Warning: Failed to compute reward at step {step}: {e}")
        
        return {}


def create_hierarchical_reward_function(device: str = "cuda", 
                                       config: HierarchicalRewardConfig = None):
    """
    创建层次化奖励函数，用于MixGRPO训练
    """
    reward_model = HierarchicalRewardModel(device=device, config=config)
    
    def reward_function(prompt: str, image: Image.Image, 
                       current_step: int = 0, total_steps: int = 50) -> float:
        """
        奖励函数接口，兼容MixGRPO的调用方式
        """
        return reward_model.get_reward_for_grpo(prompt, image, current_step, total_steps)
    
    return reward_function, reward_model


# 示例用法
if __name__ == "__main__":
    # 创建配置
    config = HierarchicalRewardConfig(
        start_weights=[0.5, 0.5, 0.0],  # 早期关注CLIP和BLIP
        end_weights=[0.2, 0.2, 0.6],    # 后期关注美学
        transition_method='cosine',
        grpo_update_interval=5
    )
    
    # 创建奖励模型
    reward_model = HierarchicalRewardModel(device="cuda", config=config)
    
    # 测试权重过渡
    print("权重过渡测试:")
    for step in [0, 10, 25, 40, 49]:
        t = step / 50
        weights = reward_model.smooth_weight_transition(t)
        print(f"Step {step}: weights = {weights}")