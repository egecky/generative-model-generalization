# Generative Model Generalization: Split-Data GAN/VAE Course Project

This project compares how GANs and VAEs trained on separate parts of a dataset
map the same latent input to images. The question was inspired by a diffusion
model result presented by Kadkhodaie et al. at ICLR 2024.

The notebook loads two models of each architecture, feeds each pair the same
fixed latent vectors, and compares their outputs. It also explores latent
interpolations and nearest training images. The experiments use CelebA and
MNIST. The CelebA scripts create two disjoint 100,000-image subsets, leaving
2,599 images unused. The MNIST scripts use two disjoint 30,000-image subsets.
Image comparisons are qualitative, with cosine similarity computed in pixel
space.

## Example outputs

CelebA samples from GANs trained on the two separate splits, generated from
shared latent inputs:

| GAN, split 1 | GAN, split 2 |
| --- | --- |
| ![GAN trained on CelebA split 1](assets/celeba_gan_split1.png) | ![GAN trained on CelebA split 2](assets/celeba_gan_split2.png) |

Corresponding VAE samples:

| VAE, split 1 | VAE, split 2 |
| --- | --- |
| ![VAE trained on CelebA split 1](assets/celeba_vae_split1.png) | ![VAE trained on CelebA split 2](assets/celeba_vae_split2.png) |

These sample grids were saved from the project notebook.

## Repository contents

- `testbench.ipynb`: loads model checkpoints, generates shared-latent samples,
  visualizes interpolations, and compares generated images with training
  examples. The notebook includes saved cell outputs.
- `gan_s1_CelebA.py`, `gan_s2_CelebA.py`: GAN training scripts for the two
  CelebA splits.
- `vae_s1.py`, `vae_s2.py`: VAE training scripts for the two CelebA splits.
- `gan_s1_MNIST.py`, `gan_s2_MNIST.py`: GAN training scripts for MNIST.
- `assets/`: representative CelebA generated-sample grids for this README.

## Exploring the notebook

Install PyTorch and a matching Torchvision build for your platform, then the
additional Python packages used by the notebook:

```bash
python -m pip install numpy matplotlib pillow scikit-image tqdm pytorch-fid jupyter
```

Download the trained checkpoints into a `models/` directory beside
`testbench.ipynb` from the [project checkpoint folder](https://drive.google.com/drive/folders/1datMe52uMlDHtTKbOooFJP5esqUNZD5f?usp=share_link).
Then open the notebook:

```bash
jupyter lab testbench.ipynb
```

The notebook also needs the CelebA and MNIST datasets. Torchvision attempts to
download them automatically. If the CelebA mirror fails, the dataset files
listed in the notebook can be obtained from the
[CelebA dataset folder](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg).

## Training a model

Each standalone script starts training immediately and is configured for 200
epochs. Create a `models/` directory first. The scripts expect the datasets to
be available and save checkpoints there. CelebA runs can take substantial
compute, so check the device settings in each script before starting. To
inspect the saved results, use the notebook above.
