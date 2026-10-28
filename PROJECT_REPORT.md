# BAMAT207: Probability and Statistics
# Course Project Activity Report
## Title: Probabilistic SMS Spam Detection

**Course Code:** BAMAT207  
**Course Title:** Probability and Statistics  
**Topic:** Probability and Random Variables  
**Submission Date:** 27.10.2026  
**Student Name:** Aakash Viswanath  
**Registration Number:** 25BCE2460  

---

## Executive Summary
This project presents an empirical and mathematical implementation of a probabilistic SMS spam detection system. Using the UCI SMS Spam Collection benchmark dataset (5,574 messages), we explore the foundational relationships between probability axioms, conditional probability, Bayes' Theorem, Bernoulli random variables, and Bayesian machine learning. 

The developed pipeline performs automated data ingestion, cleaning, deduplication, exploratory prior probability estimation, verification of Bayes' theorem on targeted keywords (`free`, `prize`, `win`, `urgent`, `offer`), Bernoulli distribution parameter estimation ($E[Y] = 0.1263, \text{Var}(Y) = 0.1104$), and trains a Multinomial Naive Bayes classifier on stratified word-count feature representations. On an unseen test set of 1,034 messages, the model achieves **98.36% Accuracy**, **95.97% Precision**, **90.84% Recall**, and **93.33% F1-score**, successfully demonstrating how probabilistic principles form the core of intelligent textual classification systems.

---

## 1. Project Scenario & Objective
With the widespread proliferation of mobile communication, short message service (SMS) spam has become a critical vector for phishing, financial fraud, and unauthorized advertising. Detecting spam requires distinguishing genuine messages (ham) from unsolicited malicious messages (spam). 

The primary objectives of this project are:
1. Conduct formal data cleaning and analyze class proportions to establish baseline prior probabilities $P(\text{Spam})$ and $P(\text{Ham})$.
2. Apply conditional probability and verify Bayes' Theorem on prominent spam indicator keywords.
3. Formulate the spam classification target as a discrete Bernoulli random variable $Y \in \{0, 1\}$ and compare theoretical expectations and variances with empirical sample statistics.
4. Prepare stratified train/test partitions (80/20) and transform textual messages into numerical word-frequency vectors using `CountVectorizer`.
5. Train and evaluate a `MultinomialNB` classifier and analyze its confusion matrix and misclassification patterns.
6. Build a real-time probabilistic inference module that calculates the exact posterior probabilities $P(\text{Spam} \mid \text{Message})$ and $P(\text{Ham} \mid \text{Message})$ for any user-provided SMS.

---

## 2. Activity 1: Dataset Handling

### 2.1 Data Source and Ingestion
The dataset utilized is the **SMS Spam Collection** from the **UCI Machine Learning Repository** (Dataset ID: 228). The dataset was imported directly via UCI archive download and the `ucimlrepo` protocol.

```python
# Dataset Ingestion Summary
Raw Messages Loaded      : 5,572
Missing / Null Values    : 0 across all attributes
Duplicate Records Found  : 403 (7.23%)
Retained Clean Messages  : 5,169
```

### 2.2 Deduplication and Label Encoding
Duplicate messages in textual classification datasets artificially inflate word co-occurrences and distort probability estimates. Therefore, 403 duplicate messages were removed, yielding a deduplicated corpus of 5,169 unique SMS messages.

Labels were mapped into binary numerical indicators:
$$\text{Spam} \longrightarrow 1, \quad \text{Ham (Genuine)} \longrightarrow 0$$

---

## 3. Activity 2: Dataset Analysis & Prior Probabilities

### 3.1 Mathematical Formulation of Priors
Let the sample space $S$ consist of all $N = 5,169$ messages. Under the frequentist interpretation of probability:
$$P(\text{Spam}) = \frac{\text{Number of Spam messages}}{N_{\text{total}}} = \frac{653}{5169} \approx 0.12633 \quad (12.63\%)$$
$$P(\text{Ham}) = \frac{\text{Number of Ham messages}}{N_{\text{total}}} = \frac{4516}{5169} \approx 0.87367 \quad (87.37\%)$$

### 3.2 Verification of Axioms
Since any SMS message is either genuine or spam:
$$P(\text{Spam} \cup \text{Ham}) = P(\text{Spam}) + P(\text{Ham}) = 0.1263 + 0.8737 = 1.0000$$

### 3.3 Visual Distribution
A comprehensive class distribution chart was rendered to visualize the inherent class imbalance ($\approx 7:1$ genuine to spam ratio).

---

## 4. Activity 3: Word-Based Probability Analysis & Bayes' Theorem Verification

### 4.1 Keyword Selection
Five representative words commonly observed in mobile spam promotions were selected:
$$\mathcal{W} = \{\text{'free'}, \text{'prize'}, \text{'win'}, \text{'urgent'}, \text{'offer'}\}$$

### 4.2 Formulations
1. **Direct Empirical Conditional Probability:**
   $$P(\text{Spam} \mid W) = \frac{N(\text{Spam} \cap W)}{N(W)}$$
2. **Bayes' Theorem Formulation:**
   $$P(\text{Spam} \mid W) = \frac{P(W \mid \text{Spam}) \times P(\text{Spam})}{P(W)}$$
   where:
   - $P(W \mid \text{Spam}) = \frac{N(\text{Spam} \cap W)}{N_{\text{spam}}}$ (Likelihood)
   - $P(\text{Spam}) = \frac{N_{\text{spam}}}{N_{\text{total}}}$ (Prior)
   - $P(W) = \frac{N(W)}{N_{\text{total}}}$ (Marginal Evidence)

### 4.3 Computational Results & Verification
| Suspicious Word $W$ | Total Count $N(W)$ | Spam Count $N(S \cap W)$ | Ham Count $N(H \cap W)$ | Likelihood $P(W \mid \text{Spam})$ | Marginal Evidence $P(W)$ | Direct $P(\text{Spam} \mid W)$ | Bayes Derived $P(\text{Spam} \mid W)$ | Absolute Discrepancy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **free** | 203 | 148 | 55 | 0.2266 | 0.0393 | **0.7291** | **0.7291** | $0.0 \times 10^0$ |
| **prize** | 73 | 73 | 0 | 0.1118 | 0.0141 | **1.0000** | **1.0000** | $0.0 \times 10^0$ |
| **win** | 55 | 49 | 6 | 0.0750 | 0.0106 | **0.8909** | **0.8909** | $0.0 \times 10^0$ |
| **urgent** | 62 | 57 | 5 | 0.0873 | 0.0120 | **0.9194** | **0.9194** | $1.1 \times 10^{-16}$ |
| **offer** | 30 | 24 | 6 | 0.0368 | 0.0058 | **0.8000** | **0.8000** | $0.0 \times 10^0$ |

### 4.4 Analytical Insights
- **`prize`**: Possesses a conditional probability $P(\text{Spam} \mid \text{prize}) = 1.0000$. Across all 73 occurrences in the dataset, every single occurrence belonged to a spam message ($0$ genuine occurrences).
- **`urgent`**: Displays a $91.94\%$ posterior probability of being spam.
- **`free`**: While $72.91\%$ of occurrences are spam, $55$ legitimate messages also contained the word "free" (e.g., "Are you free this weekend?"). This highlights why single-word checking is insufficient and multi-feature joint probability modeling is essential.
- **Verification Verdict**: The numerical output from Bayes' Theorem matches the direct ratio to within $10^{-16}$, proving the exact validity of Bayes' law on empirical datasets.

---

## 5. Activity 4: Random Variable Analysis

### 5.1 Definition of Random Variable $Y$
Let $Y$ be an indicator random variable defined as:
$$Y = \begin{cases} 1, & \text{if incoming message is Spam} \\ 0, & \text{if incoming message is Genuine (Ham)} \end{cases}$$

Since $Y$ assumes values in the discrete binary set $\{0, 1\}$, $Y$ follows a **Bernoulli Distribution**:
$$Y \sim \text{Bernoulli}(p), \quad \text{where } p = P(Y = 1) = P(\text{Spam}) = 0.12633$$

### 5.2 Mathematical Derivations
1. **Probability Mass Function (PMF):**
   $$P(Y = y) = p^y (1 - p)^{1 - y}, \quad y \in \{0, 1\}$$
2. **Expectation (First Moment):**
   $$E[Y] = \sum_{y \in \{0, 1\}} y \cdot P(Y = y) = 0 \cdot (1 - p) + 1 \cdot p = p = 0.12633$$
3. **Second Moment:**
   $$E[Y^2] = 0^2 \cdot (1 - p) + 1^2 \cdot p = p$$
4. **Variance:**
   $$\text{Var}(Y) = E[Y^2] - (E[Y])^2 = p - p^2 = p(1 - p) = (0.12633)(0.87367) \approx 0.11037$$
5. **Standard Deviation:**
   $$\sigma(Y) = \sqrt{\text{Var}(Y)} = \sqrt{0.11037} \approx 0.33222$$

### 5.3 Comparison with Empirical Sample Statistics
| Metric | Theoretical Expression | Theoretical Value | Sample Statistic | Verification Status |
| :--- | :--- | :---: | :---: | :---: |
| Parameter $p$ | $p = N_{\text{spam}} / N_{\text{total}}$ | $0.12633$ | $0.12633$ | Exact Match |
| Expectation $E[Y]$ | $p$ | $0.12633$ | $0.12633$ | Exact Match |
| Variance $\text{Var}(Y)$ | $p(1 - p)$ | $0.11037$ | $0.11039$ | Unbiased Sample Variance ($N-1$) |
| Standard Deviation $\sigma(Y)$ | $\sqrt{p(1 - p)}$ | $0.33222$ | $0.33225$ | Exact Match |

---

## 6. Activity 5: Data Preparation

### 6.1 Stratified Train/Test Split
To prevent data leakage and guarantee that both partitions reflect the real-world spam prevalence, we applied **Stratified Sampling**:
- **Split Ratio:** $80\%$ Training, $20\%$ Testing
- **Training Set Size:** $4,135$ messages (Spam proportion: $12.62\%$)
- **Testing Set Size:** $1,034$ messages (Spam proportion: $12.67\%$)

### 6.2 Numerical Feature Extraction via CountVectorizer
`CountVectorizer(stop_words='english', lowercase=True)` converts raw unstructured text into numerical frequency vectors:
- **Vocabulary Size:** $7,414$ distinct token features
- **$X_{\text{train}}$ Shape:** $(4135, 7414)$ sparse matrix
- **$X_{\text{test}}$ Shape:** $(1034, 7414)$ sparse matrix

---

## 7. Activity 6: Model Building (Multinomial Naive Bayes)

### 7.1 Architecture & Suitability
The `MultinomialNB` model is chosen because SMS messages are represented as discrete word frequency counts. 

Under the Naive Bayes assumption, word occurrences are conditionally independent given the class:
$$P(\mathbf{x} \mid Y = c) = \prod_{j=1}^d P(w_j \mid Y = c)^{x_j}$$

### 7.2 Laplace Smoothing
To resolve the zero-frequency problem (when a word appears in the test set but was never observed in class $c$ in the training set), **Laplace Smoothing** ($\alpha = 1$) is employed:
$$\hat{\theta}_{cj} = \frac{N_{cj} + 1}{N_c + d}$$
where $N_{cj}$ is the total occurrences of word $j$ in class $c$, $N_c$ is the total word count in class $c$, and $d = 7,414$ is vocabulary dimension.

### 7.3 Posterior Decision Boundary
The decision rule optimizes the log posterior:
$$\hat{y} = \arg\max_{c \in \{0, 1\}} \left[ \log P(Y = c) + \sum_{j=1}^d x_j \log \hat{\theta}_{cj} \right]$$

---

## 8. Activity 7: Model Evaluation

### 8.1 Performance Metrics on Test Set ($N_{\text{test}} = 1,034$)
| Metric | Mathematical Expression | Value | Evaluation |
| :--- | :--- | :---: | :--- |
| **Accuracy** | $\frac{\text{TP} + \text{TN}}{\text{Total}}$ | **98.36%** | Exceptional overall diagnostic correctness |
| **Precision (Spam)** | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ | **95.97%** | Only 5 false alarms out of 124 flagged messages |
| **Recall (Spam)** | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ | **90.84%** | Successfully flags >90.8% of actual spam |
| **F1-Score (Spam)** | $\frac{2 \cdot P \cdot R}{P + R}$ | **93.33%** | Robust harmonic balance on imbalanced classes |

### 8.2 Confusion Matrix
$$
\begin{array}{|c|c|c|}
\hline
& \textbf{Predicted Ham (0)} & \textbf{Predicted Spam (1)} \\
\hline
\textbf{Actual Ham (0)} & \text{TN} = 898 & \text{FP} = 5 \\
\hline
\textbf{Actual Spam (1)} & \text{FN} = 12 & \text{TP} = 119 \\
\hline
\end{array}
$$

### 8.3 In-Depth Misclassification Analysis
Understanding the failure modes is essential for engineering real-world classification systems:

#### Type I Errors: False Positives ($\text{FP} = 5, 0.55\%$)
*Genuine messages incorrectly flagged as spam:*
1. *"Are you free now?can i call now?"* $\longrightarrow P(\text{Spam} \mid \text{Msg}) = 0.5250$  
   *Root Cause:* The word `free` has a high spam likelihood in the training corpus. When combined with a short message, the posterior tips marginally over $0.50$.
2. *"Unlimited texts. Limited minutes."* $\longrightarrow P(\text{Spam} \mid \text{Msg}) = 0.7305$  
   *Root Cause:* Vocabulary tokens like `unlimited`, `texts`, and `minutes` occur predominantly in commercial telecommunications spam.

#### Type II Errors: False Negatives ($\text{FN} = 12, 9.16\%$)
*Spam messages that bypassed detection:*
1. *"Missed call alert. These numbers called but left no message. 07008009200"* $\longrightarrow P(\text{Spam} \mid \text{Msg}) = 0.0087$  
   *Root Cause:* The message mimics official cellular operator service alerts and avoids promotional trigger keywords.
2. *"Hi its LUCY Hubby at meetins all day Fri & I will B alone at hotel U fancy cumin over? Pls..."* $\longrightarrow P(\text{Spam} \mid \text{Msg}) = 0.0102$  
   *Root Cause:* Employs heavy slang and conversational obfuscation designed specifically to evade keyword-based filters.

---

## 9. Activity 8: Real-Time Testing & Inference

The system provides a real-time prediction interface that applies the learned log-likelihood parameters to compute normalized posterior probabilities:

### Required Assignment Benchmark:
- **Input Text:** `"Congratulations! You have won a free prize. Claim now."`
- **Output:**
  - **Prediction:** `Spam`
  - **$P(\text{Spam} \mid \text{Message})$:** **$1.0000$ ($100.00\%$)**
  - **$P(\text{Ham} \mid \text{Message})$:** **$0.0000$ ($0.00\%$)**

### Additional Benchmark Evaluations:
| # | Test SMS Message | Predicted Class | $P(\text{Spam} \mid \text{Msg})$ | $P(\text{Ham} \mid \text{Msg})$ | Status |
| :-: | :--- | :---: | :---: | :---: | :---: |
| 1 | *"Congratulations! You have won a free prize. Claim now."* | **Spam** | **1.0000** | **0.0000** | Correct |
| 2 | *"Hey, are we still meeting for lunch at 12:30 today?"* | **Ham** | **0.0000** | **1.0000** | Correct |
| 3 | *"URGENT! Your bank account has been compromised. Verify your details immediately."* | **Spam** | **0.9999** | **0.0001** | Correct |
| 4 | *"Can you send me the lecture notes for BAMAT207 Probability and Statistics?"* | **Ham** | **0.0791** | **0.9209** | Correct |
| 5 | *"Exclusive offer: Win $1000 cash prize instantly! Text WIN to 8888 now."* | **Spam** | **1.0000** | **0.0000** | Correct |

---

## 10. Conclusion & Pedagogical Takeaways
Through this project, the core concepts of BAMAT207 were concretely realized:
1. **Mathematical Grounding:** We verified that Bayes' theorem accurately decomposes empirical conditional probabilities, proving the formula $P(A \mid B) = \frac{P(B \mid A)P(A)}{P(B)}$ in real-world textual contexts.
2. **Random Variable Realization:** Modeling spam classification as a Bernoulli trial provided exact analytical predictions for the mean $E[Y] = 0.1263$ and variance $\text{Var}(Y) = 0.1104$.
3. **Probabilistic AI in Practice:** We demonstrated how naive conditional independence and Laplace smoothing allow scalable classification across $7,414$ dimensions with over $98.36\%$ accuracy.
4. **Digital Portfolio:** The complete project is structured, documented, and published to GitHub with reproducible notebooks and scripts.
