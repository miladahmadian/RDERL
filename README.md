# RDERL: Reliable deep ensemble reinforcement learning-based recommender system

This repository contains the implementation of the following paper:
> *RDERL: Reliable deep ensemble reinforcement learning-based recommender system*  
> Knowledge-Based Systems, 2023  
> Doi: https://doi.org/10.1016/j.knosys.2023.110289


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

🛠️ **1. Prepare Dataset**

Place the following files in `data/`:

  - `user_taggedartists.csv` (from Last.fm dataset)
  - `user_friends.csv`
    
Run preprocessing:

bash
Copy
Edit
python src/preprocessing.py
This will generate:

train_artist.csv, test_artist.csv

user_tag.csv

Train/test sets for each user

🔧 3. Train Deep Autoencoders
bash
Copy
Edit
python src/Stacked_Denoising_Autoencoder.py
This script learns 30-dimensional latent vectors for each user from:

Artist interactions

Tag annotations

Trust links

Saved in:

models/deep_user_artist.txt

models/deep_user_tag.txt

models/deep_user_trust.txt

🧪 4. Optimize Similarity Fusion with Reinforcement Learning
bash
Copy
Edit
python src/Reinforcement_Learning.py
This step:

Calculates similarity matrices for artist, tag, trust

Optimizes weights w1, w2, w3 using Q-learning

Computes predicted scores and saves:

outputs/initial_predictions.txt

outputs/neighbors_similarity.txt

📊 5. Generate Final Reliable Predictions
bash
Copy
Edit
python src/reliable_predictions.py
This script:

Computes reliability scores (fz, fv)

Adjusts predicted scores accordingly

Ranks top-N items

Evaluates Precision, Recall, F1, NDCG

📈 Results & Metrics
After execution, the system outputs:

Precision@N

Recall@N

F1 Score

NDCG

These metrics reflect ranking accuracy and recommendation quality across users.

📖 Citation
If you use this code in your research, please cite:

perl
Copy
Edit
@article{ahmadian2023reliable,
  title={A reliable deep ensemble reinforcement learning-based recommender system},
  author={Ahmadian, Milad and Ahmadi, Mehdi},
  journal={Knowledge-Based Systems},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.knosys.2023.111123}
}
📬 Contact
For questions or collaborations:

Milad Ahmadian
Email: miladahmadian@outlook.com
