<div align="center">
  <table style="border: none; background-color: transparent;">
    <tr style="border: none; background-color: transparent;">
      <td style="border: none; background-color: transparent; vertical-align: middle;">
        <img src="figs/wist.png" width="70" alt="WIST Logo"/>
      </td>
      <td style="border: none; background-color: transparent; vertical-align: middle;">
        <h1 style="margin: 0; border: none;">WIST: Web-Grounded Iterative Self-Play Tree <br> for Domain-Targeted Reasoning Improvement</h1>
      </td>
    </tr>
  </table>

  <p align="center">
    <a href="https://arxiv.org/abs/2603.22352"><img src="https://img.shields.io/badge/Paper-arXiv:2603.22352-red" alt="arXiv"></a>
    <a href="https://github.com/lfy-123/WIST/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  </p>

  <p align="center">
    <a href="README.md">English</a> | <a href="README_zh.md">简体中文</a>
  </p>
</div>

<p align="center">
  <img src="figs/diff.png" width="600" alt="与其他方法的对比图"/>
</p>

**WIST** (Web-Grounded Iterative Self-Play Tree) 是一个建立在 [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) 之上的领域定向推理能力提升框架。它通过整合以下四项机制，从根本上增强了语言模型的能力：

- 🌳 动态扩展的**领域知识树**
- 🌐 **开放网络内容的检索与清洗** 管道
- ⚔️ Challenger-Solver (挑战者-解答者) **自我博弈 (Self-Play)** 机制
- 📈 基于后验概率引导的**课程更新 (Curriculum updates)**

<p align="center">
  <img src="figs/agentic.png" width="1000" alt="WIST 整体框架"/>
</p>

## 💡 简介
近期在利用可验证奖励的强化学习（RLVR）方面取得的进展，为语言模型的自我进化提供了一条可行的路径。然而，现有方法面临一个关键的权衡：内部自驱动的自我博弈（Endogenous self-play）在迭代中容易发生信号漂移和退化，而依赖语料库的方法则严重受限于预先收集并标注好的昂贵数据环境。

为此，我们提出了 **WIST**。它直接从开放的互联网中进行学习，**完全无需任何预先整理或构建的领域语料库**。WIST 首先会进行领域树的渐进式扩展以探索知识，随后检索并清洗相关网页文档，构建出一个可控的轻量级训练环境。在此基础上，模型在可验证环境中进行 Challenger-Solver 自我博弈，博弈反馈的“可学习性”信号被用来更新树节点的后验概率分布，并以此作为自适应课程，指导下一步的探索偏好。

多项消融实验和在 4 个基础模型上的测试结果表明：WIST 一致且稳定地提升了基座模型能力，同时优于纯自我博弈基线和基于预构建语料库的自我博弈基线。例如，WIST 在 *Qwen3-4B-Base* 和 *OctoThinker-8B* 上分别取得了 **+9.8** 和 **+9.7** 的大幅综合得分跃升；展现出了极强的领域可引导性，如在医学上使 *Qwen3-8B-Base* 提升了 **+14.79**，在 PhyBench 上使 *Qwen3-4B-Base* 提升了 **+5.28**。

---

## 🎬 快速开始 / 复现指南

请按照以下步骤配置环境、处理数据、运行自我博弈训练，以及评估优化后的模型。

### 1. 环境配置

我们推荐使用专属的 Conda 环境。本项目所适用的核心依赖库均已列在 `requirements.txt` 中。

```bash
conda create -n wist python==3.12.0 -y
conda activate wist
pip install -r requirements.txt
pip install -e .
```

**核心依赖库：**
- `torch 2.7.1+cu126`
- `vllm 0.10.1`
- `ray 2.48.0`
- `deepspeed 0.16.8`
- `transformers 4.55.4`
- `flash_attn 2.8.3`
- `sentence_transformers 5.1.2`

**环境变量设置：**
在启动训练或 web worker 之前，你需要预设必要的环境变量。请将下列路径替换为你本地的实际地址：

```bash
export WIST_RUN_ROOT=/path/to/run_root
export MODEL_ROOT=/path/to/models
```

*(可选)* 如果你使用的是本地的 SearXNG 搜索引擎服务：
```bash
export SEARXNG_DIR=/path/to/searxng
export SEARXNG_PORT=8888
```

---

### 2. 搜索引擎与网页抓取服务初始化 (按需)

训练流程的启动需要后台抓取最新网页。您也可以使用自己提供的任何爬虫接口。

**步骤 2.1: 启动搜索引擎**
如果你使用本地的 [SearXNG](https://github.com/searxng/searxng)，请先启动它：
```bash
bash examples/scripts/web/worker_web_searxng.sh
```

**步骤 2.2: 启动抓取服务 (Web Worker)**
Web worker 将用来自动构建领域知识树并抓取网络上下文。保持此进程在后台持续运行。

用法：
```bash
bash examples/scripts/web/worker_web.sh <max_levels> <model_name> <task_name>
```

*各领域的启动示例:*
```bash
# 数学 (Mathematics)
bash examples/scripts/web/worker_web.sh 4 Qwen3-4B-Base wist_math_run

# 医学 (Medicine)
bash examples/scripts/web/worker_web.sh 4 Qwen3-4B-Base wist_medicine_run

# 物理 (Physics)
bash examples/scripts/web/worker_web.sh 4 Qwen3-4B-Base wist_physics_run
```

---

### 3. 运行 Self-Play 训练

我们提供了一个统一的训练启动脚本 `train_wist.sh`，可适用于任意目标领域。只需传入不同的参数，便能自动应用对应的训练配置。

**运行用法：**
```bash
bash examples/scripts/train_wist.sh <model_name> <max_levels> <gpu_count> <train_batch_size> <max_resample_attempts> <target_domain> <tree_window_size>
```

**各领域的训练命令示例：**

- **数学:**
  ```bash
  bash examples/scripts/train_wist.sh Qwen3-4B-Base 4 8 512 4 Mathematics 5
  ```
- **医学:**
  ```bash
  bash examples/scripts/train_wist.sh Qwen3-4B-Base 4 8 512 4 Medicine 5
  ```
- **物理:**
  ```bash
  bash examples/scripts/train_wist.sh Qwen3-4B-Base 4 8 512 4 Physics 5
  ```

---

### 4. 评估

训练结束后，可以使用以下统一工具评估模型的具体提升。

#### 通用评估套件
运行以下合并评估脚本，支持常见的数据集与任务：
```bash
bash examples/scripts/eval/run_math.sh /path/to/trained_model
```
支持的任务包括：`math`, `mmlu-pro`, `bbeh`, `supergpqa`, `gpqa_diamond`。

#### Maths 特化评估 (Re-check)
基于 LLM 裁判的数学重新交叉验证：
```bash
export WIST_RESULTS_RECHECK_BASE_URL=""  # OpenAI API base URL
export WIST_RESULTS_RECHECK_API_KEY=""  # OpenAI API key
bash examples/scripts/eval/run_math_recheck.sh /path/to/results_xxx.json
```

#### 医学和物理基准评估 (基于 OpenCompass)
医学和物理域的题库验证集成自 **[OpenCompass](https://github.com/open-compass/opencompass)**。在运行此类脚本之前，请在此处指定正确的系统路径：

```bash
export OPENCOMPASS_DIR=/path/to/opencompass

# 医学评估
bash examples/scripts/eval/run_opencompass_medicine.sh /path/to/trained_model

# 物理评估
bash examples/scripts/eval/run_opencompass_physics.sh /path/to/trained_model
```

---

## 📖 引用

如果你在研究中使用了或参考了 WIST 本代码库，请引用我们的论文：

```bibtex
@article{li2026wist,
  title={WIST: Web-Grounded Iterative Self-Play Tree for Domain-Targeted Reasoning Improvement},
  author={Li, Fangyuan and Li, Pengfei and Wang, Shijie and Gao, Junqi and Liu, Jianxing and Qi, Biqing and Li, Yuqiang},
  journal={arXiv preprint arXiv:2603.22352},
  year={2026}
}
```
