<div align="center" style="font-family: charter;">

<h1>对原始代码修改点</h1>

1. scripts/finetune/finetune_flux_grpo_MixGRPO.sh

reward_model="multi_reward" # "hpsv2", "clip_score" "image_reward", "pick_score", "unified_reward", "hpsv2_clip_score", "multi_reward"

改成了

reward_model="image_reward" # "hpsv2", "clip_score" "image_reward", "pick_score", "unified_reward", "hpsv2_clip_score", "multi_reward"

因为我只做了ImageReward的细粒度分支RM

2. fastvideo/models/reward_model/image_reward.py

```python
# ranking, reward = self.model.inference_rank(text, [image])
# 修改：使用3个细粒度的RM 计算reward，排序
# BLIP更注重图像和文本的语义匹配度
_, reward_blip = self.blip_rm.inference_rank(text, [image])
# CLIP更注重图像和文本的细节对齐
_, reward_clip = self.clip_rm.inference_rank(text, [image])
# Aesthetic更注重图像的美学质量
_, reward_aes = self.aesthtic_rm.inference_rank(text, [image])

# 加权得到综合奖励分数
combined_reward = (blip_weight * reward_blip[0] + 
                    clip_weight * reward_clip[0] + 
                    aesthetic_weight * reward_aes[0])

```

要改scripts/finetune/finetune_flux_grpo_MixGRPO_Flash.sh的话可以同理


build_reward新增了对细粒度模型部署
```python
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
```

init加上了权重

```python
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
```


3. fastvideo/train_grpo_flux.py

## compute_reward 接收参数添加了 index, len(batch_indices) 也就是当前的batch index，和每次扩散的总batch，用来近似 curr_step 和 total_step_per_diffusion

```python
rewards, successes, rewards_dict, successes_dict = compute_reward(
    images, 
    prompts,
    reward_models,
    reward_weights,
    # 新增：当前去噪batch index，和batch总数，方便给他按照时间给不同权重
    curr_step=index,
    total_steps=len(batch_indices)
)
```


<h1>MixGRPO:</br>Unlocking Flow-based GRPO Efficiency with Mixed ODE-SDE</h1>


<a href="https://arxiv.org/abs/2507.21802" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-MixGRPO-red?logo=arxiv" height="20" /></a>
<a href="https://tulvgengenr.github.io/MixGRPO-Project-Page/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/💻_Project-MixGRPO-blue.svg" height="20" /></a>
<a href="https://huggingface.co/tulvgengenr/MixGRPO" target="_blank">
    <img alt="" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-MixGRPO-ffc107?color=ffc107&logoColor=white" height="20" /></a>

<div>
    <a href="https://scholar.google.com/citations?user=lQsMoJsAAAAJ&hl=en&oi=ao" target="_blank">Junzhe Li</a><sup>1,</sup><sup>2,</sup><sup>3</sup><sup>*</sup>,</span>
    <a href="https://scholar.google.com/citations?user=TSMchWcAAAAJ&hl=en&oi=ao" target="_blank">Yutao Cui</a><sup>1</sup><sup>*</sup>, </span>
    <a href="https://scholar.google.com/citations?hl=en&user=TaM4e4wAAAAJ" target="_blank">Tao Huang</a><sup>1</sup>,</span>
    <a href="" target="_blank">Yinping Ma</a><sup>3</sup>,</span>
    <a href="https://scholar.google.com/citations?hl=en&user=0ZZamLoAAAAJ&view_op=list_works&sortby=pubdate" target="_blank">Chun Fan</a><sup>3</sup>,</span>
    <a href="" target="_blank">Miles Yang</a><sup>1</sup>,</span>
    <a href="https://scholar.google.com/citations?user=igtXP_kAAAAJ&hl=en" target="_blank">Zhao Zhong</a><sup>1</sup></span>
</div>

<div>
    <sup>1</sup>Hunyuan, Tencent&emsp;
    </br>
    <sup>2</sup>School of Computer Science, Peking University&emsp;
    </br>
    <sup>3</sup>Computer Center, Peking University&emsp;
</div>


</div>      

## 📝 News
- [2025/7/30] We released the [model checkpoint](https://huggingface.co/tulvgengenr/MixGRPO) fine-tuned based on [FLUX.1 Dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) using the MixGRPO algorithm, with [HPSv2](https://github.com/tgxs002/HPSv2), [ImageReward](https://huggingface.co/zai-org/ImageReward), and [Pick Score](https://github.com/yuvalkirstain/PickScore) as multi-rewards !
- [2025/7/30] We released the [paper](https://arxiv.org/abs/2507.21802) and [code](https://github.com/Tencent-Hunyuan/MixGRPO) !

## ✅ TODO
- [ ] Take comparison with the [FlowGRPO](https://github.com/yifan123/flow_grpo) and update our technical report.

## 🚀 Quick Start

### Installation

#### 1. Environment setup
```bash
conda create -n MixGRPO python=3.12
conda activate MixGRPO
```

#### 2. Requirements installation
```bash
sudo yum install -y pdsh pssh mesa-libGL # centos
bash env_setup.sh
```
The environment dependency is basically the same as [DanceGRPO](https://github.com/XueZeyue/DanceGRPO).

### Models Preparation

#### 1. FLUX
Download the FLUX HuggingFace repository to `"./data/flux"`.
```bash
mkdir ./data/flux
huggingface-cli login
huggingface-cli download --resume-download  black-forest-labs/FLUX.1-dev --local-dir ./data/flux
```

#### 2. Reward Models

##### HPS-v2.1
Download the code of [HPSv2](https://github.com/tgxs002/HPSv2).
```bash
git clone https://github.com/tgxs002/HPSv2.git
```

Download the `"HPS_v2.1_compressed.pt"` and `"open_clip_model.safetensors"` to `"./hps_ckpt"`
```bash
mkdir hps_ckpt
huggingface-cli login
huggingface-cli download --resume-download xswu/HPSv2 HPS_v2.1_compressed.pt --local-dir ./hps_ckpt/
huggingface-cli download --resume-download laion/CLIP-ViT-H-14-laion2B-s32B-b79K open_clip_pytorch_model.bin --local-dir ./hps_ckpt/
```

##### Pick Score
Run the demo code to automatically download to `"~/.cache/huggingface"`:
```bash
python fastvideo/models/reward_model/pick_score.py \
    --device cuda \
    --http_proxy <Your HTTP_PROXY> \ # Default is None
    --https_proxy <Your HTTPS_PROXY>  # Default is None
```

##### ImageReward
Down the `"ImageReward.pt"` and `"med_config.json"` to `"./image_reward_ckpt"`
```bash
huggingface-cli login
huggingface-cli download --resume-download THUDM/ImageReward med_config.json --local-dir ./image_reward_ckpt/
huggingface-cli download --resume-download THUDM/ImageReward ImageReward.pt --local-dir ./image_reward_ckpt/
```

##### CLIP Score
Run the demo code to automatically download to `"~/.cache/huggingface"`:
```bash
python fastvideo/models/reward_model/clip_score.py \
    --device cuda \
    --http_proxy <Your HTTP_PROXY> \ # Default is None
    --https_proxy <Your HTTPS_PROXY>  # Default is None
```

### Preprocess Data
Adjust the `prompt_path` parameter in `"./scripts/preprocess/preprocess_flux_rl_embeddings.sh"` to obtain the embeddings of the prompt dataset.
```bash
bash scripts/preprocess/preprocess_flux_rl_embeddings.sh
```

### Run Training
The training dataset is the training prompts in [HPDv2](https://huggingface.co/datasets/ymhao/HPDv2), as shown in `"./data/prompts.txt"`

We use the `"pdsh"` command for multi-node training with `"torchrun"`. The default resource configuration consists of 4 nodes, each with 8 GPUs, totaling 32 GPUs.

First, set your multi-node IPs in the `data/hosts/hostfile`.

Then, run the following script to set the environment variable `INDEX_CUSTOM` on each node to 0, 1, 2, and 3, respectively.
```bash
bash scripts/preprocess/set_env_multinode.sh
```

Next, set the `wandb_key` to your Weights & Biases (WandB) key in `"./scripts/finetune/finetune_flux_grpo_FastGRPO.sh"`.

Finally, run the following training script:
```bash
bash scripts/finetune/finetune_flux_grpo_FastGRPO.sh
```

### Run Inference
The test dataset is also the test prompts in [HPDv2](https://huggingface.co/datasets/ymhao/HPDv2), as shown in `"./data/prompts_test.txt"`

First, you need to download the MixGRPO model weight `"diffusion_pytorch_model.safetensors"` to the `"./mix_grpo_ckpt"` directory.
```bash
mkdir mix_grpo_ckpt
huggingface-cli login
huggingface-cli download --resume-download tulvgengenr/MixGRPO diffusion_pytorch_model.safetensors --local-dir ./mix_grpo_ckpt/
```
Then, adjust the `Input parameters` in "scripts/inference/inference_flux.sh" (currently set to default) and then execute the single-node script.
```bash
bash scripts/inference/inference_flux.sh
```

### Run Evaluation
Set `prompt_file` to the path of the JSON file generated during inference in "scripts/evaluate/eval_reward.sh". Then run the following single-node script.
```bash
bash scripts/evaluate/eval_reward.sh
```


## 🤝 Acknowledgement

We are deeply grateful for the following GitHub repositories, as their valuable code and efforts have been incredibly helpful:

* [DanceGRPO](https://github.com/XueZeyue/DanceGRPO)
* [Flow-GRPO](https://github.com/yifan123/flow_grpo)
* [FastVideo](https://github.com/hao-ai-lab/FastVideo)
* [HPSv2](https://github.com/tgxs002/HPSv2)


## ✏️ Citation

### License
MixGRPO is licensed under the License Terms of MixGRPO. See `./License.txt` for more details.

### Bib
If you find MixGRPO useful for your research and applications, please cite using this BibTeX:
```
@misc{li2025mixgrpounlockingflowbasedgrpo,
      title={MixGRPO: Unlocking Flow-based GRPO Efficiency with Mixed ODE-SDE}, 
      author={Junzhe Li and Yutao Cui and Tao Huang and Yinping Ma and Chun Fan and Miles Yang and Zhao Zhong},
      year={2025},
      eprint={2507.21802},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2507.21802}, 
}
```
