# Lightweight LLM Fine-Tuning on Colab T4
Efficient LLM fine-tuning with MLsys techniques — all on a single Colab T4 GPU.

## 🚀 Introduction / Motivation

This project is a hands-on exploration of **efficient LLM fine-tuning** under limited computational resources. Instead of aiming for high performance or state-of-the-art results, the goal is to get **hands-on experience with ML systems techniques** like quantization, parameter-efficient fine-tuning (PEFT), and CPU swapping.

## 💻 Why Colab T4?

Google Colab with a T4 GPU offers an accessible platform for experimenting with LLMs:

* ✅ Easy to use and widely available
* ✅ Modest GPU RAM (≈15GB) and CPU RAM (≈12GB), which makes it necessary to plan carefully
* ✅ No setup required beyond a browser

Although the limited resources can be challenging, they also make it a great environment to practice system-level techniques such as:

* Quantization (to reduce memory usage)
* Parameter-efficient tuning like LoRA
* CPU offloading and activation checkpointing via DeepSpeed

Working within these constraints has helped me understand what's really happening during training — and how to make things work when memory is tight.

## 📦 What’s Inside?

This project currently includes one notebook focused on fine-tuning a quantized LLM on a math dataset using system-level techniques.

#### 📝 Notebook: `DeepSeek-R1-Distill-Qwen-1.5B-MATH-SFT.ipynb`
* Fine-tunes a 8-bit quantized version of [`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) using a small subset from the [MATH dataset](https://huggingface.co/datasets/hendrycks/competition_math).
* Applies we applied LoRA and ZeRO stage 3 for CPU swapping on model weights, optimizer states and activations.
* Libraries used: `transformers`, `deepspeed`, `bitsandbytes`, `peft`, `datasets`, `evaluate`.

More notebooks may be added in the future to explore RLHF, alternative tasks, or more advanced tuning strategies

## 🙏 Acknowledgments

This project builds on open-source tools and libraries, especially:

* [HuggingFace Transformers](https://github.com/huggingface/transformers)
* [DeepSpeed](https://github.com/microsoft/DeepSpeed)
* [PEFT (LoRA)](https://github.com/huggingface/peft)
