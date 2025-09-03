# Image-Reward: Copyied from https://github.com/THUDM/ImageReward
import os
from typing import Union, List, Callable
from PIL import Image

import torch
try:
    import ImageReward as RM
except:
    raise Warning("ImageReward is required to be installed (`pip install image-reward`) when using ImageReward for post-training.")


class ImageRewardModel(object):
    def __init__(self, model_name, device, http_proxy=None, https_proxy=None, med_config=None,
                 # 新增：细粒度分数的起始权重 [blip, clip, aesthetical]
                 start_weights: list = [0.5, 0.3, 0.2],
                 # 新增：细粒度分数的结束权重 [blip, clip, aesthetical]
                 end_weights: list = [0.2, 0.5, 0.3]):
        if http_proxy:
            os.environ["http_proxy"] = http_proxy
        if https_proxy:
            os.environ["https_proxy"] = https_proxy
        self.model_name = model_name if model_name else "ImageReward-v1.0"
        self.device = device
        self.med_config = med_config
        # 校验权重合法性
        assert len(start_weights) == 3 and len(end_weights) == 3, "权重列表必须包含3个元素"
        assert all(w >= 0 for w in start_weights) and all(w >= 0 for w in end_weights), "权重不能为负数"
        assert sum(start_weights) > 0 and sum(end_weights) > 0, "权重总和必须大于0"
        
        self.start_weights = start_weights
        self.end_weights = end_weights
        self.build_reward_model()
        
    def build_reward_model(self):
        self.model = RM.load(self.model_name, device=self.device, med_config=self.med_config)
        # 改：下面加上细粒度RM
        self.clip_rm = RM.load_score("CLIP", device=self.device)
        self.blip_rm = RM.load_score("BLIP", device=self.device)
        self.aesthtic_rm = RM.load_score("Aesthetic", device=self.device)
    
    def get_dynamic_weights(self, curr_step: int, total_steps: int, 
                           smooth_func: Callable = None) -> List[float]:
        """
        根据当前步数计算动态权重
        :param curr_step: 当前训练步数
        :param total_steps: 总训练步数
        :param smooth_func: 权重平滑函数，默认为线性过渡
        :return: 计算后的动态权重 [blip, clip, aesthetical]
        """
        # 确保步数在有效范围内
        progress = min(max(curr_step / total_steps, 0.0), 1.0)
        
        # 默认使用线性平滑
        if smooth_func is None:
            smooth_func = self._linear_smooth
        
        # 计算每个维度的动态权重
        dynamic_weights = []
        for s, e in zip(self.start_weights, self.end_weights):
            dynamic_weight = smooth_func(s, e, progress)
            dynamic_weights.append(dynamic_weight)
        
        # 归一化权重（确保总和为1）
        total = sum(dynamic_weights)
        return [w / total for w in dynamic_weights]

    @staticmethod
    def _linear_smooth(start: float, end: float, progress: float) -> float:
        """线性平滑：权重随进度线性过渡"""
        return start + (end - start) * progress

    @staticmethod
    def _exponential_smooth(start: float, end: float, progress: float, gamma: float = 2.0) -> float:
        """指数平滑：前期接近start，后期快速过渡到end"""
        return start + (end - start) * (progress ** gamma)

    @staticmethod
    def _logarithmic_smooth(start: float, end: float, progress: float, gamma: float = 0.5) -> float:
        """对数平滑：前期快速过渡，后期接近end"""
        if progress == 0:
            return start
        return start + (end - start) * (1 - (1 - progress) ** gamma)

    @torch.no_grad()
    def __call__(
            self,
            images,
            texts,
            # 新增传入时间 curr_step
            curr_step: int = 0,
            total_steps: int = 50,
            smooth_func: Callable = None
    ):
        if isinstance(texts, str):
            texts = [texts] * len(images)
        
        weights = self.get_dynamic_weights(curr_step, total_steps, smooth_func)
        blip_weight, clip_weight, aesthetic_weight = weights
        
        rewards = []
        # 生成唯一时间戳（精确到毫秒）
        import time
        import tempfile
        timestamp = time.time()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for idx, (image, text) in enumerate(zip(images, texts)):
                # 文件名格式：时间戳_索引.png（确保唯一）
                img_path = os.path.join(tmpdir, f"img_{timestamp}_{idx}.png")
                image.save(img_path, format="PNG")  # 保存PIL对象到临时路径
                
                # 后续评分逻辑不变，使用img_path作为图像输入
                _, reward_blip = self.blip_rm.inference_rank(text, [img_path])
                _, reward_clip = self.clip_rm.inference_rank(text, [img_path])
                _, reward_aes = self.aesthtic_rm.inference_rank(text, [img_path])
                
                combined_reward = (blip_weight * reward_blip[0] + 
                                clip_weight * reward_clip[0] + 
                                aesthetic_weight * reward_aes[0])
                
                rewards.append(combined_reward)
        # for image, text in zip(images, texts):
        #     # ranking, reward = self.model.inference_rank(text, [image])
        #     # 修改：使用3个细粒度的RM 计算reward，排序
        #     # BLIP更注重图像和文本的语义匹配度
        #     _, reward_blip = self.blip_rm.inference_rank(text, [image])
        #     # CLIP更注重图像和文本的细节对齐
        #     _, reward_clip = self.clip_rm.inference_rank(text, [image])
        #     # Aesthetic更注重图像的美学质量
        #     _, reward_aes = self.aesthtic_rm.inference_rank(text, [image])
            
        #     # 加权得到综合奖励分数
        #     combined_reward = (blip_weight * reward_blip[0] + 
        #                        clip_weight * reward_clip[0] + 
        #                        aesthetic_weight * reward_aes[0])
            
        #     rewards.append(combined_reward)
        return rewards
