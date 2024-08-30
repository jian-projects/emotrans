
<h2 align="center"> <a href="https://aclanthology.org/2024.lrec-main.508/">EmoTrans: Emotional Transition-based Model for Emotion Recognition in Conversation</a></h2>
<h5 align="center"> If you appreciate our project, please consider giving us a star ⭐ on GitHub to stay updated with the latest developments.  </h2>

<h4 align="center">

🚀 Welcome to the repo of [**EmoTrans**](https://github.com/jian-projects/emotrans)!

EmoTrans addresses the unimodal ERC task by modeling emotion transition feature, accepted by COLING2024.

<!-- [![🤗Hugging Face](https://img.shields.io/badge/🤗Hugging_Face-Uni_MoE-yellow)](https://huggingface.co/Uni-MoE) -->
<!-- [![Project Page](https://img.shields.io/badge/Project_Page-Uni_MoE-blue)](https://uni-moe.github.io/) -->
<!-- [![Demo](https://img.shields.io/badge/Demo-Local-orange)](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/tree/master?tab=readme-ov-file#-demo-video)  -->
<!-- [![Paper](https://img.shields.io/badge/Paper-arxiv-yellow)](https://arxiv.org/abs/2405.11273) -->

[Zhongquan Jian](https://scholar.google.com/citations?user=C1PWVBUAAAAJ&hl=zh-CN), [Ante Wang](https://scholar.google.com/citations?user=xmwanZcAAAAJ&hl=zh-CN), [Jinsong Su](https://scholar.google.com/citations?user=w6qCk3sAAAAJ&hl=zh-CN), [Junfeng Yao](https://scholar.google.com/citations?hl=zh-CN&user=Szz3hSMAAAAJ), [Meihong Wang](https://dblp.uni-trier.de/pid/99/3203.html), [Qingqiang Wu](https://dblp.uni-trier.de/pid/130/0742.html)
</h4>

<!-- ## 🌟 Structure

The model architecture of Uni-MoE is shown below. Three training stages contain: 1) Utilize pairs from different modalities and languages to build connectors that map these elements to a unified language space, establishing a foundation for multimodal understanding; 2) Develop modality-specific experts using cross-modal data to ensure deep understanding, preparing for a cohesive multi-expert model; 3) Incorporate multiple trained experts into LLMs and refine the unified multimodal model using the LoRA technique on mixed multimodal data.

<div align=center><img src="https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/model.png" height="100%" width="75%"/></div> -->

## ⚡️ Install

The following instructions are for Linux installation.
We would like to recommend the requirements as follows.
* Python == 3.9.16
* CUDA Version >= 11.7

1. Clone this repository and navigate to the ccm folder
```bash
git git@github.com:jian-projects/emotrans.git
cd emotrans
```

2. Install Package
```Shell
conda create -n emotrans python==3.9.16
conda activate emotrans
pip install -r env.txt
```

## 🌈 How to train and inference

1. Specify the path of [RoBERTa-large](https://huggingface.co/FacebookAI/roberta-large) in global_var.py

2. run the script:
```bash
python run_emotrans.py
```

3. We re-write the code and thus lose the checkpoints, the performance can be easily reproduction with our provided code and hyper-parameters.

## Citation

If you find Uni-MoE useful for your research and applications, please cite using this BibTeX:
```bibtex
@inproceedings{2024.erc.emotrans,
  author={Zhongquan Jian and Ante Wang and Jinsong Su and Junfeng Yao and Meihong Wang and Qingqiang Wu},
  title={EmoTrans: Emotional Transition-based Model for Emotion Recognition in Conversation},
  booktitle={Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation},
  pages={5723--5733},
  year= {2024},
  doi={https://aclanthology.org/2024.lrec-main.508},
}
```