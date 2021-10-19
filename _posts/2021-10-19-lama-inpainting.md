---
title: "Paper Reading: Resolution-robust Large Mask Inpainting with Fourier Convolutions"
tags: ["paper reading", "inpainting", "fast fourier convolution"]
key: lama-inpainting
---

This paper[^lama] proposed a method for high-resolution images inpainting with large missing areas. The results look great! Three contributions were claimed but I find the **fast Fourier Convolutions (FFC)** to be most essential. 

<!--more-->

## Summary

**Goal**: Develop modern image inpainting system

**Challenges**:

- large missing areas
- complex geometric structures
- high-resolution images

**Reasons**:  lack of an effective receptive field in both the inpainting network and the loss function

**Solutions**:

- fast Fourier convolution
- high receptive field perceptual loss
- large training masks

## Methodology

![scheme](https://raw.githubusercontent.com/yuanpinz/blog/main/assets/images/posts/image-20211019164509053.png "The scheme of the proposed method for large-mask inpainting (LaMa)")

### Fast Fourier Convolutions[^ffc]

![image-20211019173512461](https://raw.githubusercontent.com/yuanpinz/blog/main/assets/images/posts/image-20211019173512461.png)

### Loss Functions

Final loss: $\mathcal{L}_{final}=\kappa L_{Adv}+\alpha \mathcal{L}_{HRFRL} + \beta \mathcal{L}_{DiscPL} + \gamma R_1$​​​​

- $L_{Adv}$: **Adversarial loss**

- $\mathcal{L}_{HRFRL}$: High receptive field perceptual loss

  $\mathcal{L}_{HRFRL}(x,\hat{x})=\mathcal{M}([\phi_{HRF}(x)-\phi_{HRF}(\hat{x})]^2)$​​ ,​

  where $\phi_{HRF}$ is a pre-trained network, $\mathcal{M}$ is the sequential two-stage mean operation (interlayer mean of intra-layer means).

- $\mathcal{L}_{DiscPL}$: **discriminator-based perceptual loss** (or feature matching loss)[^dpl]

- $R_1$: **Gradient penalty**​[^r1]

### Generation of Masks

![image-20211019170856702](https://raw.githubusercontent.com/yuanpinz/blog/main/assets/images/posts/image-20211019170856702.png)

## Experiments

### Ablation Study on Fast Fourier Convolutions

![image-20211019174309601](https://raw.githubusercontent.com/yuanpinz/blog/main/assets/images/posts/image-20211019174309601.png)

![image-20211019173624223](https://raw.githubusercontent.com/yuanpinz/blog/main/assets/images/posts/image-20211019173624223.png)

![image-20211019173637378](https://raw.githubusercontent.com/yuanpinz/blog/main/assets/images/posts/image-20211019173637378.png)

### Ablation Study on Losses

![image-20211019174435621](https://raw.githubusercontent.com/yuanpinz/blog/main/assets/images/posts/image-20211019174435621.png)

### Ablation Study on Masks

![image-20211019174446320](https://raw.githubusercontent.com/yuanpinz/blog/main/assets/images/posts/image-20211019174446320.png)



[^lama]: [Suvorov, Roman, et al. "Resolution-robust Large Mask Inpainting with Fourier Convolutions." *arXiv preprint arXiv:2109.07161* (2021).](https://arxiv.org/pdf/2109.07161.pdf)
[^ffc]: [Chi, Lu, Borui Jiang, and Yadong Mu. "Fast fourier convolution." *Advances in Neural Information Processing Systems* 33 (2020).](https://proceedings.neurips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf)

[^r1]: [Wang, Ting-Chun, et al. "High-resolution image synthesis and semantic manipulation with conditional gans." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2018.](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_High-Resolution_Image_Synthesis_CVPR_2018_paper.pdf)
[^dpl]: [Mescheder, Lars, Andreas Geiger, and Sebastian Nowozin. "Which training methods for GANs do actually converge?." *International conference on machine learning*. PMLR, 2018.](http://proceedings.mlr.press/v80/mescheder18a/mescheder18a.pdf)





![](https://raw.githubusercontent.com/yuanpinz/blog/main/assets/images/posts/image.png)

