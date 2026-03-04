# AAE Encoder Training – Loss Functions & Weightings Reference

**Source:** `02b_aae_encoder_training.ipynb`

---

## Overview

The Adversarial Autoencoder (AAE) replaces the KL divergence term in the VAE with an adversarial objective. Training alternates between:

1. **Reconstruction phase** – encoder + decoder + classifier
2. **Adversarial phase** – discriminator vs encoder (generator)

---

## 1. Reconstruction Loss

**Equation:**
$$\mathcal{L}_{\text{recon}} = \frac{1}{N} \sum_{i=1}^{N} \| \mathbf{x}_i - \hat{\mathbf{x}}_i \|^2$$

Where:
- \(\mathbf{x}_i\) = input data (song feature vector)
- \(\hat{\mathbf{x}}_i\) = decoder output (reconstruction)

**Implementation:** Mean Squared Error (MSE), same as the VAE.

**Weighting:** 1.0 (no scaling)

---

## 2. Genre Classification Loss

**Equation:**
$$\mathcal{L}_{\text{cls}} = -\sum_{c} y_c \log(p_c)$$

Cross-entropy over genre classes \(c\), where:
- \(y_c\) = one-hot true genre
- \(p_c\) = softmax output of the genre classifier

**Implementation:** `F.cross_entropy(logits, batch_y)`

**Weighting:** \(\lambda_{\text{cls}} = 1.0\) (`cls_weight`)

---

## 3. Reconstruction-Phase Total Loss

Combined loss for the encoder, decoder, and classifier:

$$\mathcal{L}_{\text{recon-phase}} = \mathcal{L}_{\text{recon}} + \lambda_{\text{cls}} \cdot \mathcal{L}_{\text{cls}}$$

With \(\lambda_{\text{cls}} = 1.0\):

$$\mathcal{L}_{\text{recon-phase}} = \mathcal{L}_{\text{recon}} + 1.0 \cdot \mathcal{L}_{\text{cls}}$$

---

## 4. Discriminator Loss

Binary cross-entropy for real vs. fake latent vectors.

**Real samples** (from prior \(p(z)\)):
$$\mathcal{L}_{\text{real}} = -\mathbb{E}_{z \sim p(z)} \left[ \log D(z) \right]$$

**Fake samples** (from encoder \(q(z|x)\)):
$$\mathcal{L}_{\text{fake}} = -\mathbb{E}_{z \sim q(z|x)} \left[ \log(1 - D(z)) \right]$$

**Total discriminator loss:**
$$\mathcal{L}_D = \mathcal{L}_{\text{real}} + \mathcal{L}_{\text{fake}}$$

**Weighting:** 1.0 for each term (no additional scaling)

**Implementation:** Genre-conditioned prior is used: `real_z = prior.sample_conditional(batch_y)` (mixture-of-Gaussians per genre).

---

## 5. Generator (Encoder) Adversarial Loss

The encoder is trained to make the discriminator classify its latent vectors as “real”.

**Equation:**
$$\mathcal{L}_G = -\mathbb{E}_{z \sim q(z|x)} \left[ \log D(z) \right]$$

The encoder minimizes this so that \(D(z) \to 1\) for encoder outputs.

**Implementation:** `F.binary_cross_entropy(pred, torch.ones_like(pred))`

**Weighting:** 1.0

---

## Prior Parameters (Mixture-of-Gaussians)

**Prior distribution:**
$$p(z) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(z; \boldsymbol{\mu}_k, \sigma^2 \mathbf{I})$$

**Parameters:**
| Parameter | Value | Role |
|-----------|-------|------|
| \(K\) | `num_genres` (26) | Number of mixture components (one per genre) |
| \(\alpha\) | 1.5 | Controls cluster separation (mean spread) |
| \(\sigma\) | 1.0 | Controls overlap/smoothness |
| \(\pi_k\) | \(1/K\) | Uniform mixture weights |
| `learnable_means` | False | Means are fixed, not learned |

**Conditional sampling:** For genre \(y\), sample from \(\mathcal{N}(\boldsymbol{\mu}_y, \sigma^2 \mathbf{I})\).

---

## Summary Table

| Loss | Equation | Weight |
|------|----------|--------|
| Reconstruction | \(\frac{1}{N} \sum_i \| x_i - \hat{x}_i \|^2\) | 1.0 |
| Classification | \(-\sum_c y_c \log(p_c)\) | 1.0 (`cls_weight`) |
| Discriminator (real) | \(-\log D(z_{\text{real}})\) | 1.0 |
| Discriminator (fake) | \(-\log(1 - D(z_{\text{fake}}))\) | 1.0 |
| Generator (adversarial) | \(-\log D(z)\) | 1.0 |

---

## Training Order (per batch)

1. **Reconstruction phase:** Minimize \(\mathcal{L}_{\text{recon}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}}\)
2. **Discriminator phase:** Minimize \(\mathcal{L}_D\)
3. **Generator phase:** Minimize \(\mathcal{L}_G\) (encoder only)
