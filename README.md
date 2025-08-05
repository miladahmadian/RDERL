# RDERL: Reliable Deep Ensemble Reinforcement Learning-based Recommender System

This repository contains the implementation of the following paper:

> **RDERL: Reliable deep ensemble reinforcement learning-based recommender system**  
> *Knowledge-Based Systems*, 2023  
> [https://doi.org/10.1016/j.knosys.2023.110289](https://doi.org/10.1016/j.knosys.2023.110289)

---

## 📌 Overview

RDERL is a hybrid recommender system that:

- Learns user representations via **stacked denoising autoencoders** from:
  - User–Artist interactions
  - User–Tag associations
  - User–Trust relationships
- Computes **neighborhood-based collaborative filtering** using latent similarities
- Uses **Q-learning** to optimize the weights (`w1`, `w2`, `w3`) for combining similarity sources
- Applies a **reliability-aware ranking** to generate robust top-N recommendations

---

## 🛠️ Step-by-Step Usage

### 1. Prepare the Dataset

Place the following files in the `data/` folder:

- `user_taggedartists.csv` (from Last.fm dataset)
- `user_friends.csv`

Then run:
```bash
python src/preprocessing.py
```
This will generate:
- `train_artist.csv`, `test_artist.csv`
- `user_tag.csv`

---

### 2. Train Deep Autoencoders

```bash
python src/Stacked_Denoising_Autoencoder.py
```
This script learns 30-dimensional user embeddings based on:
- Artist preferences → `models/deep_user_artist.txt`
- Tag behavior → `models/deep_user_tag.txt`
- Trust network → `models/deep_user_trust.txt`

---

### 3. Optimize Fusion Weights via Reinforcement Learning

```bash
python src/Reinforcement_Learning.py
```
This step:
- Computes user similarities
- Optimizes `w1`, `w2`, `w3` using Q-learning
- Generates:
  - `outputs/initial_predictions.txt`
  - `outputs/neighbors_similarity.txt`

---

### 4. Generate Final Reliable Recommendations

```bash
python src/reliable_predictions.py
```
This script:
- Computes reliability factors (`fz`, `fv`)
- Refines predictions using reliability
- Evaluates using:
  - **Precision**
  - **Recall**
  - **F1-score**
  - **NDCG**

---

## 📈 Output Metrics

After execution, RDERL reports:

- Precision@N
- Recall@N
- F1-score
- NDCG

These reflect the ranking performance and recommendation quality.

---

## 📖 Citation

If you use this code, please cite:

```bibtex
@article{ahmadian2023reliable,
  title={A reliable deep ensemble reinforcement learning-based recommender system},
  author={Ahmadian, Milad and Ahmadi, Mehdi},
  journal={Knowledge-Based Systems},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.knosys.2023.111123}
}
```

---

## 📬 Contact

**Milad Ahmadian**  
Email: [miladahmadian@outlook.com](mailto:miladahmadian@outlook.com)
