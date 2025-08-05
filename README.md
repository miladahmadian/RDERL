# RDERL: Reliable deep ensemble reinforcement learning-based recommender system

This repository contains the implementation of the following paper:
> *RDERL: Reliable deep ensemble reinforcement learning-based recommender system*  
> Knowledge-Based Systems, 2023  
> doi: https://doi.org/10.1016/j.knosys.2023.110289


## Overview
RDERL is a hybrid recommendation framework that:
- Learns user representations via **stacked denoising autoencoders** using:
  - User–Artist interactions
  - User–Tag associations
  - User–Trust relationships
- Computes **neighborhood-based collaborative filtering** predictions using similarity between latent embeddings
- Applies **Reinforcement learning- Q-learning** to optimize weights (`w1`, `w2`, `w3`) for artist/tag/trust similarities
- Integrates a **reliability-aware weighting** mechanism for robust ranking and top-N recommendation

---

