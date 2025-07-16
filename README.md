# 🧠 Parameter-Efficient Fine-Tuning of LLaMA 3.2 (3B) on a Medical Chain-of-Thought Dataset

<img width="595" height="258" alt="image" src="https://github.com/user-attachments/assets/ce78e9e2-66fa-4f83-b4a9-2533c1eec867" />

[![Demo Video](https://img.shields.io/badge/Demo-YouTube-red)](https://www.youtube.com/watch?v=Hxl17HhdvNk&feature=youtu.be)
[![Hugging Face](https://img.shields.io/badge/HF_Model-PEFT_LLaMA3.2-blue)](https://huggingface.co/Hums003/PEFT_LlaMA_3.2_MCoT/tree/main)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue)](https://www.kaggle.com/code/humaimaanwar/peft-of-llama-3-2-3b-on-a-medical-cot/edit/run/236223559)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue)](https://www.kaggle.com/code/humaimaanwar/inference) 

---

## 📌 Project Overview

This repository demonstrates how to apply **Parameter-Efficient Fine-Tuning (PEFT)** using **LoRA** on the **LLaMA 3.2 (3B)** model for a domain-specific task: **Medical Chain-of-Thought (CoT) reasoning**.

By leveraging only **0.81% of the model parameters**, we fine-tuned a large model efficiently using **4-bit quantization** on limited hardware (single GPU ~7.5 GB VRAM).

---

## 🎯 Objectives

- Efficient fine-tuning of LLaMA 3.2 3B using LoRA adapters.
- Structure the training data using `<think>` and `<response>` tags for reasoning.
- Reduce memory usage using 4-bit quantization (`bitsandbytes`).
- Log training metrics using `wandb`.
- Host and share adapter using Hugging Face Hub.

---

## 🚀 Results Summary

| Metric               | Before FT     | After FT (Epoch 2) |
|----------------------|---------------|---------------------|
| Trainable Params     | -             | 24.3M (0.81%)       |
| Training Loss        | 2.0031        | 1.4339              |
| Validation Loss      | 1.6528        | 1.4577              |
| ROUGE-L Score        | -             | **1.0000**          |
| GPU Memory Use       | ~7.0 GB       | ~7.5 GB             |

---

## 🧰 Tools & Technologies

- **Model Training**: 🤗 Transformers, PEFT, Unsloth, bitsandbytes
- **Optimization**: LoRA, 4-bit quantization, AdamW
- **Tracking**: Weights & Biases (wandb)
- **Evaluation**: ROUGE-L (evaluate lib)
- **Deployment**: Hugging Face Hub, Kaggle
- **Dataset Source**: [`FreedomIntelligence/medical-cot`](https://huggingface.co/datasets/FreedomIntelligence/medical-cot)

---

## 🧾 Dataset Format

Each example in the dataset is structured as:

```html
<prompt>
<think>Chain-of-thought reasoning here...</think>
<response>Final medical answer here.</response>
</prompt>

🧪 Sample Inference
Input Prompt:

<prompt>
<think>Identify symptoms: cough, fever, night sweats → suspect TB. Confirm via sputum and chest X-ray.</think>
<response>Diagnosis: Pulmonary Tuberculosis</response>
</prompt>

Output:
<prompt>
<think>...</think>
<response>Diagnosis: Pulmonary Tuberculosis</response>
</prompt>
```

## 🛠️ Setup & Training (Summary)
🖥️ Environment

- Platform: Kaggle Notebook
- GPU: NVIDIA T4 (~16 GB RAM)
- Quantization: 4-bit via bitsandbytes

## 🧠 LoRA Adapter Settings

- Layers: q_proj, k_proj, v_proj, o_proj
- Rank: 16
- Alpha: 32

## 🏋️‍♂️ Training

- Epochs: 2
- Effective Batch Size: 8
- Optimizer: AdamW (lr=2e-5)

## 📦 Deployment & Hosting

- 🤗 Hugging Face Model: [PEFT_LLaMA_3.2_MCoT](https://huggingface.co/Hums003/PEFT_LlaMA_3.2_MCoT/tree/main)
- 📓 Kaggle Notebook: [Fine-Tuning](https://www.kaggle.com/code/humaimaanwar/peft-of-llama-3-2-3b-on-a-medical-cot/edit/run/236223559)
- 🧪 Inference Demo: [Kaggle Link](https://www.kaggle.com/code/humaimaanwar/inference)
- 🎥 Video Demo: [YouTube](https://www.youtube.com/watch?v=Hxl17HhdvNk&feature=youtu.be)

## 🧠 Final Thoughts

- ✅ Efficient LoRA-based fine-tuning on large models with low compute
- ✅ High quality medical reasoning and structured output
- ✅ Full pipeline from dataset → training → evaluation → deployment
- ✅ Easily adaptable to other domains with CoT-style data

## 📚 References

- LLaMA 3.2 PEFT Guide (Unsloth)
- PEFT Concepts – IBM Blog
- FreedomIntelligence/medical-cot Dataset
- ROUGE Metric - HuggingFace Evaluate

🔬 Built with ❤️ by Humaima Anwar as part of Arch Technologies Internship.
