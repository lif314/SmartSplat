<p align="center">
  <h1 align="center">
    SmartSplat: Feature-Smart Gaussians for Scalable Compression of Ultra-High-Resolution Images
    <br>
    [AAAI 2026]
  </h1>
  <p align="center">
  <a href="https://github.com/lif314"><strong>Linfei Li</strong></a>
  ·
  <a href="https://scholar.google.com/citations?user=8VOk_S4AAAAJ&hl=en"><strong>Lin Zhang*</strong></a>
  ·
  <a href="https://scholar.google.com/citations?user=rrkp_usAAAAJ&hl=en"><strong>Zhong Wang</strong></a>
  ·
  <a href="https://scholar.google.com/citations?user=A0N_mS0AAAAJ&hl=en"><strong>Ying Shen</strong></a>
</p>

  <h3 align="center"><a href="https://smartsplat.github.io/SmartSplat-Website/">🌐Project page</a> 
  | <a href="https://arxiv.org/abs/2512.20377">📝Paper</a>
  </h3>
  <div align="center"></div>
</p>

<div align="center">
  <a href="">
    <img src="./assets/teaser.gif" alt="SmartSplat teaser" style="max-width: 100%; height: auto;">
  </a>
  <p style="margin-top: 8px; font-size: 14px; color: #555;">
   Raw Image info: 16320×10848, 189 MB -> 1.99 MB (.npz)
  </p>
</div>

</p>


<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#datasets">Datasets</a>
    </li>
    <li>
      <a href="#benchmarking">Benchmarking</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
  </ol>
</details>

## Installation

```bash
conda create -n smartsplat python==3.12
conda activate smartsplat

# install torch
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

pip install setuptools==78.0.1

pip install -r requirements.txt

cd submodules/fused-ssim
pip install -e .

# Used by GaussianImage/3DGS/SmartSplat
cd ../gsplat
pip install -e .

# Used by LIG
cd ../gsplat2d
pip install -e .
```

## Datasets
The datasets used in SmartSplat can be downloaded from the links provided below, including the tested subsets and the full versions hosted on Hugging Face or Baidu Netdisk.

| Dataset | Tested | Full |
| --- | --- | --- |
| DIV8K | [SmartSplat-DIV8K](https://huggingface.co/datasets/3David14/SmartSplat-DIV8K) | [Full DIV8K](https://huggingface.co/datasets/Iceclear/DIV8K_TrainingSet) |
| DIV16K | [SmartSplat-DIV16K](https://huggingface.co/datasets/3David14/SmartSplat-DIV16K) | [Full DIV16K](https://pan.baidu.com/s/1fpZyDjPC3JlxPWHVMeTISQ?pwd=agk5) |



## Benchmarking
This codebase integrates multiple GS-based image representation methods, including [GaussianImage](https://github.com/Xinjie-Q/GaussianImage), [3DGS](https://github.com/nerfstudio-project/gsplat/blob/main/examples/image_fitting.py), and [LIG](https://arxiv.org/abs/2502.09039). All our experiments were conducted on the A800 cluster. The corresponding run scripts are provided in the `slurm` folder.

- `LIG`
```bash
data_path="data"
base_log_path="logs"
current_cr=50
python train_lig_for_eval.py \
    -d $data_path \
    --data_name  DIV16K \
    --model_name LIG \
    --compression_ratio $current_cr \
    --log_dir $base_log_path \
    --iterations 50000 \
    --save_iter_img 10000 \
    --save_imgs
```

- `3DGS`
```bash
data_path="data"
base_log_path="logs"
current_cr=50
python train_all_for_eval.py \
    -d $data_path \
    --data_name DIV8K \
    --model_name 3DGS \
    --compression_ratio $current_cr \
    --log_dir $base_log_path \
    --iterations 50000 \
    --save_iter_img 10000
```

- `GaussianImage (RS)`
```bash
data_path="data"
base_log_path="logs"
current_cr=50
python train_all_for_eval.py \
    -d $data_path \
    --data_name  DIV16K \
    --model_name GaussianImage_RS \
    --compression_ratio $current_cr \
    --log_dir $base_log_path \
    --iterations 50000 \
    --save_iter_img 10000
```

- `GaussianImage (Cholesky)`
```bash
data_path="data"
base_log_path="logs"
current_cr=50
python train_all_for_eval.py \
    -d $data_path \
    --data_name  DIV16K \
    --model_name GaussianImage_Cholesky \
    --compression_ratio $current_cr \
    --log_dir $base_log_path \
    --iterations 50000 \
    --save_iter_img 10000
```

- `Image-GS`
> The `Image-GS` implementation used in our codebase is built upon `GaussianImage` and does not incorporate the `Top-K` strategy, resulting in suboptimal performance. For accurate reproduction, please refer to the [official implementation](https://github.com/NYU-ICL/image-gs).

```bash
data_path="data"
base_log_path="logs"
current_cr=50
python train_all_for_eval.py \
    -d $data_path \
    --data_name  DIV8K \
    --model_name ImageGS_RS \
    --compression_ratio $current_cr \
    --log_dir $base_log_path \
    --iterations 50000 \
    --save_iter_img 10000
```

- `SmartSplat`
```bash
data_path="data"
base_log_path="logs"
current_cr=3000
python train_all_for_eval_smart.py \
    -d $data_path \
    --data_name  DIV8K \
    --model_name SmartSplat \
    --compression_ratio $current_cr \
    --log_dir $base_log_path \
    --iterations 50000 \
    --save_iter_img 10000
```

## Acknowledgement
We thank the authors of the following repositories for their open-source code:


- [Gsplat](https://github.com/nerfstudio-project/gsplat)
- [GaussianImage](https://github.com/Xinjie-Q/GaussianImage)
- [ImageGS](https://github.com/NYU-ICL/image-gs)
- [LIG](https://arxiv.org/abs/2502.09039)
- [FusedSSIM](https://github.com/rahul-goel/fused-ssim)

## Citation

If you find our paper and code useful for your research, please use the following BibTeX entry.

```bibtex
@misc{li2025smartsplatfeaturesmartgaussiansscalable,
      title={SmartSplat: Feature-Smart Gaussians for Scalable Compression of Ultra-High-Resolution Images}, 
      author={Linfei Li and Lin Zhang and Zhong Wang and Ying Shen},
      year={2025},
      eprint={2512.20377},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.20377}, 
}
```