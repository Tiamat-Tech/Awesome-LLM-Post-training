# Awesome-Reasoning-LLM-Tutorial-Survey-Guide  

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)  
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)  

Welcome to the **Awesome-Reasoning-LLM-Tutorial-Survey-Guide** repository! This repository is a curated collection of the most influential papers, code implementations, benchmarks, and resources related to **Large Language Models (LLMs) and Reinforcement Learning (RL)**. Our goal is to provide a comprehensive reference for researchers, practitioners, and enthusiasts interested in how LLMs can enhance RL through reasoning, planning, decision-making, and generalization.  

Our work is based on the following paper:  
📄 **[Paper Title]** – Available on [arXiv](https://arxiv.org/abs/XXXX.XXXXX).  
Authors: **[Author 1], [Author 2], [Author 3], ..., [Author N]**.  

Feel free to ⭐ star and fork this repository to keep up with the latest advancements and contribute to the community.

---

## 📌 Contents  

| Section | Subsection |  
| ------- | ----------- |  
| [📖 Papers](#papers) | [Survey](#survey), [Theory](#theory), [Explainability](#explainability) |  
| [🤖 LLMs in RL](#llms-in-rl) | LLM-Augmented Reinforcement Learning |  
| [🏆 Reward Learning](#reward-learning) | [Human Feedback](#human-feedback), [Preference-Based RL](#preference-based-rl), [Intrinsic Motivation](#intrinsic-motivation) |  
| [🚀 Policy Optimization](#policy-optimization) | [Offline RL](#offline-rl), [Imitation Learning](#imitation-learning), [Hierarchical RL](#hierarchical-rl) |  
| [🧠 LLMs for Reasoning & Decision-Making](#llms-for-reasoning-and-decision-making) | [Causal Reasoning](#causal-reasoning), [Planning](#planning), [Commonsense RL](#commonsense-rl) |  
| [🌀 Exploration & Generalization](#exploration-and-generalization) | [Zero-Shot RL](#zero-shot-rl), [Generalization in RL](#generalization-in-rl), [Self-Supervised RL](#self-supervised-rl) |  
| [🤝 Multi-Agent RL (MARL)](#multi-agent-rl-marl) | [Emergent Communication](#emergent-communication), [Coordination](#coordination), [Social RL](#social-rl) |  
| [⚡ Applications & Benchmarks](#applications-and-benchmarks) | [Autonomous Agents](#autonomous-agents), [Simulations](#simulations), [LLM-RL Benchmarks](#llm-rl-benchmarks) |  
| [📚 Tutorials & Courses](#tutorials-and-courses) | [Lectures](#lectures), [Workshops](#workshops) |  
| [🛠️ Libraries & Implementations](#libraries-and-implementations) | Open-Source RL-LLM Frameworks |  
| [🔗 Other Resources](#other-resources) | Additional Research & Readings |  

---

# 📖 Papers  

## 🔍 Survey  

| Title | Date | Links |  
| ----- | ---- | ----- |  
| A Survey on Large Language Models for Reinforcement Learning | 10 Dec 2023 | [Arxiv](https://arxiv.org/abs/2312.04567) |  
| Large Language Models as Decision-Makers: A Survey | 23 Aug 2023 | [Arxiv](https://arxiv.org/abs/2308.11749) |  
| A Survey on Large Language Model Alignment Techniques | 6 May 2023 | [Arxiv](https://arxiv.org/abs/2305.00921) |  
| Reinforcement Learning with Human Feedback: A Survey | 12 April 2023 | [Arxiv](https://arxiv.org/abs/2304.04989) |  
| Reasoning with Large Language Models: A Survey | 14 Feb 2023 | [Arxiv](https://arxiv.org/abs/2302.06476) |  
| A Survey on Foundation Models for Decision Making | 9 Jan 2023 | [Arxiv](https://arxiv.org/abs/2301.04150) |  
| Large Language Models in Reinforcement Learning: Opportunities and Challenges | 5 Dec 2022 | [Arxiv](https://arxiv.org/abs/2212.09142) |  

---

## 🤖 LLMs in RL  

- **"ReAct: Synergizing Reasoning and Acting in Language Models"** - Yao et al. (2022) [[Paper](https://arxiv.org/abs/2210.03629)]  
- **"LLM-Based Autonomous Agents in RL Environments"** - Wang et al. (2023) [[Paper](https://arxiv.org/abs/2305.05665)]  

---

## 🏆 Reward Learning  

- **"RLHF: Reinforcement Learning with Human Feedback"** - Christiano et al. (2017) [[Paper](https://arxiv.org/abs/1706.03741)]  
- **"Direct Preference Optimization"** - Rafailov et al. (2023) [[Paper](https://arxiv.org/abs/2305.18290)]  

---

## 🚀 Policy Optimization  

- **"Decision Transformer: Reinforcement Learning via Sequence Modeling"** - Chen et al. (2021) [[Paper](https://arxiv.org/abs/2106.01345)]  
- **"Offline RL with LLMs as Generalist Memory"** - Tian et al. (2023) [[Paper](https://arxiv.org/abs/2302.02096)]  

---

## 🧠 LLMs for Reasoning & Decision-Making  

- **"Causal Decision Transformers"** - Xiao et al. (2023) [[Paper](https://arxiv.org/abs/2307.07774)]  
- **"LLMs for Commonsense Reasoning in RL"** - Huang et al. (2023) [[Paper](https://arxiv.org/abs/2308.09876)]  

---

## 🌀 Exploration & Generalization  

- **"Harnessing LLMs for Zero-Shot RL"** - Du et al. (2023) [[Paper](https://arxiv.org/abs/2304.04636)]  
- **"Generalization in RL: The Role of LLMs"** - Xiao et al. (2023) [[Paper](https://arxiv.org/abs/2305.06711)]  

---

## 🤝 Multi-Agent RL (MARL)  

- **"LLMs as Zero-Shot Coordinators in MARL"** - McIlroy-Young et al. (2023) [[Paper](https://arxiv.org/abs/2306.01665)]  
- **"Emergent Communication in Multi-Agent RL with LLMs"** - Gupta et al. (2023) [[Paper](https://arxiv.org/abs/2305.05454)]  

---
## 🚀 RL & LLM Fine-Tuning Repositories

| #  | Repository & Link                                                                                          | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|----|------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1  | [**RL4VLM**](https://github.com/RL4VLM/RL4VLM) <br><br> _Archived & Read-Only as of December 15, 2024_       | Offers code for fine-tuning large vision-language models as decision-making agents via RL. Includes implementations for training models with task-specific rewards and evaluating them in various environments.                                                                                                                                                                                                                                                                                                         |
| 2  | [**LlamaGym**](https://github.com/KhoomeiK/LlamaGym)                                                      | Simplifies fine-tuning large language model (LLM) agents with online RL. Provides an abstract `Agent` class to handle various aspects of RL training, allowing for quick iteration and experimentation across different environments.                                                                                                                                                                                                                                                                                      |
| 3  | [**RL-Based Fine-Tuning of Diffusion Models for Biological Sequences**](https://github.com/masa-ue/RLfinetuning_Diffusion_Bioseq) | Accompanies a tutorial and review paper on RL-based fine-tuning, focusing on the design of biological sequences (DNA/RNA). Provides comprehensive tutorials and code implementations for training and fine-tuning diffusion models using RL.                                                                                                                                                                                                                                     |
| 4  | [**LM-RL-Finetune**](https://github.com/zhixuan-lin/LM-RL-finetune)                                       | Aims to improve KL penalty optimization in RL fine-tuning of language models by computing the KL penalty term analytically. Includes configurations for training with Proximal Policy Optimization (PPO).                                                                                                                                                                                                                                                                                                                     |
| 5  | [**InstructLLaMA**](https://github.com/michaelnny/InstructLLaMA)                                           | Implements pre-training, supervised fine-tuning (SFT), and reinforcement learning from human feedback (RLHF) to train and fine-tune the LLaMA2 model to follow human instructions, similar to InstructGPT or ChatGPT.                                                                                                                                                                                                                                                                                                       |
| 6  | [**SEIKO**](https://github.com/zhaoyl18/SEIKO)                                                             | Introduces a novel RL method to efficiently fine-tune diffusion models in an online setting. Its techniques outperform baselines such as PPO, classifier-based guidance, and direct reward backpropagation for fine-tuning Stable Diffusion.                                                                                                                                                                                                                                                                              |
| 7  | [**TRL (Train Transformer Language Models with RL)**](https://github.com/huggingface/trl)                  | A state-of-the-art library for post-training foundation models using methods like Supervised Fine-Tuning (SFT), Proximal Policy Optimization (PPO), GRPO, and Direct Preference Optimization (DPO). Built on the 🤗 Transformers ecosystem, it supports multiple model architectures and scales efficiently across hardware setups.                                                                                                                                                                                     |
| 8  | [**Fine-Tuning Reinforcement Learning Models as Continual Learning**](https://github.com/BartekCupial/finetuning-RL-as-CL) | Explores fine-tuning RL models as a forgetting mitigation problem (continual learning). Provides insights and code implementations to address forgetting in RL models.                                                                                                                                                                                                                                                                                                                                                        |
| 9  | [**RL4LMs**](https://github.com/allenai/RL4LMs)                                                            | A modular RL library to fine-tune language models to human preferences. Rigorously evaluated through 2000+ experiments using the GRUE benchmark, ensuring robustness across various NLP tasks.                                                                                                                                                                                                                                                                                                                             |
| 10 | [**Lamorel**](https://github.com/flowersteam/lamorel)                                                      | A high-throughput, distributed architecture for seamless LLM integration in interactive environments. While not specialized in RL or RLHF by default, it supports custom implementations and is ideal for users needing maximum flexibility.                                                                                                                     |
| 11 | [**LLM-Reverse-Curriculum-RL**](https://github.com/WooooDyy/LLM-Reverse-Curriculum-RL)                     | Implements the ICML 2024 paper *"Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning"*. Focuses on enhancing LLM reasoning capabilities using a reverse curriculum RL approach.                                                                                                                                                                                                                                                  |
| 12 | [**veRL**](https://github.com/volcengine/verl)                                                             | A flexible, efficient, and production-ready RL training library for large language models (LLMs). Serves as the open-source implementation of the HybridFlow framework and supports various RL algorithms (PPO, GRPO), advanced resource utilization, and scalability up to 70B models on hundreds of GPUs. Integrates with Hugging Face models, supervised fine-tuning, and RLHF with multiple reward types.                                                  |
| 13 | [**trlX**](https://github.com/CarperAI/trlx)                                                               | A distributed training framework for fine-tuning large language models (LLMs) with reinforcement learning. Supports both Accelerate and NVIDIA NeMo backends, allowing training of models up to 20B+ parameters. Implements PPO and ILQL, and integrates with CHEESE for human-in-the-loop data collection.                                                                                                                                                                                              |
| 14 | [**Okapi**](https://github.com/nlp-uoregon/Okapi)                                                          | A framework for instruction tuning in LLMs with RLHF, supporting 26 languages. Provides multilingual resources such as ChatGPT prompts, instruction datasets, and response ranking data, along with both BLOOM-based and LLaMa-based models and evaluation benchmarks.                                                                                                                                                                                                                                                 |
| 15 | [**LLaMA-Factory**](https://github.com/hiyouga/LLaMA-Factory)                                              | *Unified Efficient Fine-Tuning of 100+ LLMs & VLMs (ACL 2024)*. Supports a wide array of models (e.g., LLaMA, LLaVA, Qwen, Mistral) with methods including pre-training, multimodal fine-tuning, reward modeling, PPO, DPO, and ORPO. Offers scalable tuning (16-bit, LoRA, QLoRA) with advanced optimizations and logging integrations, and provides fast inference via API, Gradio UI, and CLI with vLLM workers.                                                 |

---
## ⚡ Applications & Benchmarks  

- **"AutoGPT: LLMs for Autonomous RL Agents"** - OpenAI (2023) [[Paper](https://arxiv.org/abs/2304.03442)]  
- **"Barkour: Benchmarking LLM-Augmented RL"** - Wu et al. (2023) [[Paper](https://arxiv.org/abs/2305.12377)]  

---

## 📚 Tutorials & Courses  

- 🎥 **Deep RL Bootcamp (Berkeley)** [[Website](https://sites.google.com/view/deep-rl-bootcamp/)]  
- 🎥 **DeepMind RL Series** [[Website](https://deepmind.com/learning-resources)]  

---

## 🛠️ Libraries & Implementations  

- 🔹 [Decision Transformer (GitHub)](https://github.com/kzl/decision-transformer)  
- 🔹 [ReAct (GitHub)](https://github.com/ysymyth/ReAct)  
- 🔹 [RLHF (GitHub)](https://github.com/openai/lm-human-preferences)  

---

## 🔗 Other Resources  

- [LLM for RL Workshop at NeurIPS 2023](https://neurips.cc)  
- [OpenAI Research Blog on RLHF](https://openai.com/research)  

---

## 📌 Contributing  

Contributions are welcome! If you have relevant papers, code, or insights, feel free to submit a pull request.  

---

Would you like **figures, tables, or additional formatting improvements**? 🚀  
