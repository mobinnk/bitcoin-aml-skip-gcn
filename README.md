#  Anti-Money Laundering in Bitcoin via Graph Convolutional Networks (Skip-GCN)

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-3C2179?style=for-the-badge&logo=pytorch&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Graph_Neural_Networks-FF6F00?style=for-the-badge)

A complete engineering reproduction and analytical validation of the paper **["Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics"](https://arxiv.org/abs/1908.02591)** by Weber et al. (MIT-IBM Watson AI Lab & Elliptic). 

This repository implements the **Skip-GCN** architecture from scratch using PyTorch and PyTorch Geometric (PyG), explicitly focusing on handling highly imbalanced temporal graph data to detect illicit Bitcoin transactions.

---

## 1. Project Overview & Paper Summary

Financial forensics in cryptocurrency is fundamentally a network topology problem. Traditional machine learning models (like Random Forests or MLPs) treat transactions as isolated events, ignoring the critical **payment flow** (the directional transfer of funds). 

This project tackles the **Binary Node Classification** problem of mapping pseudonymous Bitcoin transactions to either *Licit* (Class 0) or *Illicit* (Class 1) entities. By leveraging **Graph Convolutional Networks (GCNs)**, the model propagates features through the transaction graph, effectively aggregating 1-hop neighborhood statistics to identify anomalous subgraphs indicative of money laundering.

---

## 2. The Elliptic Dataset & Data Engineering

The model is trained on the **Elliptic Data Set**, the world's largest labeled transaction graph.
*   **Nodes**: 203,769 Bitcoin transactions.
*   **Edges**: 234,355 directed payment flows.
*   **Temporal Split**: 49 distinct time steps (graphs are completely disconnected across steps).
    *   **Train Set**: Time steps 1 to 34 (70%).
    *   **Test Set**: Time steps 35 to 49 (30%).
*   **Severe Class Imbalance**: ~2% Illicit, ~21% Licit, and **~77% Unknown** (Unlabeled).

###  Engineering Constraints Handled:
1.  **Masked Loss Computation**: The 77% *Unknown* nodes MUST participate in the message-passing mechanism ($\hat{A}X$) to provide structural context, but are strictly masked out via boolean tensors during the Cross-Entropy Loss computation to avoid skewing gradients.
2.  **Temporal Batching**: The dataset is processed using a `Full-Graph per Time Step` batching strategy, yielding 49 distinct PyG `Data` objects.
3.  **Normalization Check**: Features were strictly normalized via Z-Score using statistics derived **only** from the Train Set to prevent data leakage.

---

## 3. Model Architecture: Skip-GCN

The core of this repository is the mathematical and programmatic implementation of the **Skip-GCN** variant. Because over-smoothing is a known issue in GCNs applied to financial data, a linear skip connection directly projects the input features to the output space, ensuring original transactional signals are not lost.

**Mathematical Formulation:**
$$ \tilde{H}^{(2)} = \text{Softmax}\left( \hat{A} \cdot \text{ReLU}\left(\hat{A} X W^{(0)}\right) W^{(1)} + X W_e^{(1)} \right) $$

*   $X \in \mathbb{R}^{N_t \times 166}$: Node features (94 local + 72 aggregated).
*   $\hat{A} \in \mathbb{R}^{N_t \times N_t}$: Normalized sparse adjacency matrix.
*   **GCN Path**: $166 \xrightarrow{W^{(0)}} 100 \xrightarrow{W^{(1)}} 2$
*   **Skip Path**: $166 \xrightarrow{W_e^{(1)}} 2$
*   **Fusion**: Element-wise addition followed by LogSoftmax.

---

##  4. Results & Quantitative Evaluation

Due to the extreme 2% imbalance, accuracy is a misleading metric. The model is strictly evaluated on the **F1-Score of the Illicit class**. To force the model to prioritize illicit transaction detection, a Weighted Cross-Entropy Loss was applied `[0.3 for Licit, 0.7 for Illicit]`.

| Metric | Weber et al. (Reported) | Our Implementation |
| :--- | :---: | :---: |
| **Illicit Precision** | - | 0.6720 |
| **Illicit Recall** | - | 0.5065 |
| **Illicit F1-Score** | ~0.55 - 0.58 | **0.5774** |
| **Micro F1-Score** | ~0.94 - 0.95 | **0.9413** |

*The implementation successfully reproduces and matches the benchmark thresholds established by the MIT-IBM research team.*

---

##  5. Visual Analytics & Behavioral Insights

Understanding *how* the model behaves over time and in latent space is just as critical as the final metrics. The following evaluations demonstrate a deep phenomenological understanding of the network's behavior.

### A. Temporal Evaluation & The "Dark Market Shutdown"
In the original paper, the authors noted a sharp drop in illicit transaction detection around time step 43, correlating with the real-world shutdown of a major Darknet market. 

![Temporal Performance](assets/temporal_performance.png)
*Insight: By isolating the evaluation loop per time-step (35-49), our model exhibits this exact macro-economic phenomenon. The F1-Score and Recall take a noticeable hit at `t=43`, proving the GCN is highly sensitive to real-world topological shifts in the Bitcoin network.*

### B. Confusion Matrix & Thresholding
![Confusion Matrix](assets/confusion_matrix.png)
*Insight: The normalized confusion matrix shows the effectiveness of our `[0.3, 0.7]` loss weighting scheme. The model aggressively identifies the minority illicit class without severely compromising the True Negative rate of the licit class.*

### C. Precision-Recall & ROC Space Analysis
![PR and ROC Curves](assets/pr_roc_curves.png)
*Insight: While the ROC-AUC looks near perfect, the PR-AUC reveals the true difficulty of the task. Given a baseline of ~0.02 (random guessing), achieving a high PR-AUC proves the network's spatial message-passing is successfully capturing illicit neighborhood patterns.*

### D. Latent Space Representation (PCA on GCN Embeddings)
![PCA Embeddings](assets/pca_embeddings.png)
*Insight: By hooking into the hidden layer ($H^{(1)} \in \mathbb{R}^{100}$) and projecting it via PCA, we can visually confirm that the GCN is learning distinct topological manifolds. Even in 2D space, the illicit nodes (red) begin to form tighter, localized clusters distinct from the broader licit network (green).*

---

##  6. Implementation Stack
*   **Framework**: `PyTorch 2.x`, `PyTorch Geometric (PyG)`
*   **Data Processing**: `Pandas`, `NumPy`, `Scikit-Learn`
*   **Visualization**: `Matplotlib`, `Seaborn`
*   **Hardware**: Executed and profiled on NVIDIA T4/A100 (Google Colab).

