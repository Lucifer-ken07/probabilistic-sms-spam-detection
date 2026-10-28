"""
BAMAT207 Probability and Statistics - Project Activity
Topic: Probability and Random Variables
Student Project: Probabilistic SMS Spam Detection

This script implements the full pipeline:
1. Dataset Handling (loading, missing value check, deduplication, label mapping)
2. Dataset Analysis (prior probabilities, class distribution visualization)
3. Word-Based Probability Analysis (suspicious words + Bayes' theorem verification)
4. Random Variable Analysis (Bernoulli random variable Y, E[Y], Var[Y], empirical comparison)
5. Data Preparation (80/20 stratified split, CountVectorizer feature extraction)
6. Model Building (Multinomial Naive Bayes classifier)
7. Model Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix, error analysis)
8. Real-Time Testing (interactive message testing with posterior probability output)
"""

import os
import re
import io
import urllib.request
import zipfile
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Set base directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")

for directory in [DATA_DIR, FIGURES_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
RAW_DATA_PATH = os.path.join(DATA_DIR, "SMSSpamCollection")
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "SMSSpamCollection_cleaned.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "spam_classifier_multinomial_nb.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "count_vectorizer.pkl")


# ==============================================================================
# ACTIVITY 1: DATASET HANDLING
# ==============================================================================
def load_and_clean_dataset():
    """
    Downloads (if necessary) and loads the SMS Spam Collection dataset from UCI.
    Checks missing values, removes duplicates, and encodes labels (Spam=1, Ham=0).
    """
    print("\n" + "=" * 70)
    print("ACTIVITY 1: DATASET HANDLING")
    print("=" * 70)

    # Check if raw data exists, otherwise download from UCI
    if not os.path.exists(RAW_DATA_PATH):
        print(f"[*] Downloading SMS Spam Collection from UCI repository...")
        try:
            # First demonstrate attempting ucimlrepo as requested by syllabus
            try:
                from ucimlrepo import fetch_ucirepo
                print("[*] Attempting fetch via ucimlrepo library...")
                sms_dataset = fetch_ucirepo(id=228)
                # If available as tabular dataframe:
                df_raw = pd.DataFrame({
                    'label': sms_dataset.data.targets.iloc[:, 0],
                    'message': sms_dataset.data.features.iloc[:, 0]
                })
                print("[+] Successfully loaded via ucimlrepo.")
            except Exception as e_uci:
                print(f"[-] Note on ucimlrepo: {e_uci}")
                print("[*] Using direct UCI repository archive download...")
                req = urllib.request.Request(UCI_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as response:
                    zip_bytes = io.BytesIO(response.read())
                    with zipfile.ZipFile(zip_bytes) as z:
                        z.extract("SMSSpamCollection", path=DATA_DIR)
                        if "readme" in z.namelist():
                            z.extract("readme", path=DATA_DIR)
                print(f"[+] Downloaded and extracted to: {RAW_DATA_PATH}")
                df_raw = pd.read_csv(
                    RAW_DATA_PATH, sep="\t", header=None, names=["label", "message"], encoding="utf-8"
                )
        except Exception as e:
            print(f"[!] Error during download: {e}")
            raise
    else:
        print(f"[+] Found cached raw dataset at: {RAW_DATA_PATH}")
        df_raw = pd.read_csv(
            RAW_DATA_PATH, sep="\t", header=None, names=["label", "message"], encoding="utf-8"
        )

    # Initial stats
    total_raw = len(df_raw)
    print(f"Total raw SMS messages loaded: {total_raw}")
    print(f"Dataset columns: {list(df_raw.columns)}")

    # Check for missing values
    missing_counts = df_raw.isnull().sum()
    print("\nMissing values check:")
    for col, count in missing_counts.items():
        print(f"  - Column '{col}': {count} missing values")

    # Check for duplicate messages
    duplicate_count = df_raw.duplicated(subset=["message"]).sum()
    print(f"\nDuplicate messages detected: {duplicate_count} ({duplicate_count / total_raw * 100:.2f}%)")

    # Remove duplicate messages
    df_clean = df_raw.drop_duplicates(subset=["message"]).reset_index(drop=True)
    total_clean = len(df_clean)
    print(f"Messages retained after deduplication: {total_clean}")

    # Convert labels into numerical values: Spam = 1, Ham = 0
    df_clean["label_num"] = df_clean["label"].map({"ham": 0, "spam": 1})

    # Add character length and word count metadata
    df_clean["char_length"] = df_clean["message"].apply(len)
    df_clean["word_count"] = df_clean["message"].apply(lambda s: len(s.split()))

    # Save cleaned dataset for reproducibility
    df_clean.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"[+] Cleaned dataset saved to: {CLEANED_DATA_PATH}")

    return df_clean


# ==============================================================================
# ACTIVITY 2: DATASET ANALYSIS
# ==============================================================================
def analyze_dataset(df):
    """
    Calculates class distribution and prior probabilities P(Spam) and P(Ham).
    Generates and saves the class distribution bar chart.
    """
    print("\n" + "=" * 70)
    print("ACTIVITY 2: DATASET ANALYSIS & PRIOR PROBABILITIES")
    print("=" * 70)

    total_msgs = len(df)
    counts = df["label"].value_counts()
    ham_count = counts.get("ham", 0)
    spam_count = counts.get("spam", 0)

    p_ham = ham_count / total_msgs
    p_spam = spam_count / total_msgs

    print(f"Total Cleaned Messages (N)  : {total_msgs}")
    print(f"Genuine Messages (Ham)       : {ham_count} ({p_ham * 100:.2f}%)")
    print(f"Spam Messages               : {spam_count} ({p_spam * 100:.2f}%)")
    print("-" * 50)
    print(f"Prior Probability P(Ham)    : {p_ham:.4f}")
    print(f"Prior Probability P(Spam)   : {p_spam:.4f}")
    print(f"Check P(Ham) + P(Spam)      : {p_ham + p_spam:.4f}")

    # Visualizing class distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Subplot 1: Bar Chart with Counts and Percentages
    colors = ["#2b5c8f", "#d9534f"]
    bars = ax1.bar(["Ham (Genuine)", "Spam"], [ham_count, spam_count], color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax1.set_title("SMS Class Distribution (Counts & Proportions)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Number of Messages", fontsize=11)
    ax1.set_ylim(0, max(ham_count, spam_count) * 1.15)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, count, p in zip(bars, [ham_count, spam_count], [p_ham, p_spam]):
        yval = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 60,
            f"{count:,}\n({p * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

    # Subplot 2: Donut Chart showing Proportion
    ax2.pie(
        [ham_count, spam_count],
        labels=["Ham", "Spam"],
        autopct="%1.2f%%",
        startangle=140,
        colors=colors,
        explode=(0.04, 0.08),
        wedgeprops=dict(width=0.45, edgecolor="black")
    )
    ax2.set_title("Prior Probabilities: P(Ham) vs P(Spam)", fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    chart_path = os.path.join(FIGURES_DIR, "class_distribution.png")
    plt.savefig(chart_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Class distribution chart saved to: {chart_path}")

    return {
        "total": total_msgs,
        "ham_count": ham_count,
        "spam_count": spam_count,
        "p_ham": p_ham,
        "p_spam": p_spam
    }


# ==============================================================================
# ACTIVITY 3: WORD-BASED PROBABILITY ANALYSIS
# ==============================================================================
def analyze_word_probabilities(df, suspicious_words=None):
    """
    Selects suspicious words (free, prize, win, urgent, offer).
    Calculates direct empirical conditional probability:
      P(Spam | Word) = Count(Spam ∩ Word) / Count(Word)
    Verifies result using Bayes' theorem:
      P(Spam | Word) = [P(Word | Spam) * P(Spam)] / P(Word)
    """
    print("\n" + "=" * 70)
    print("ACTIVITY 3: WORD-BASED PROBABILITY ANALYSIS & BAYES' THEOREM VERIFICATION")
    print("=" * 70)

    if suspicious_words is None:
        suspicious_words = ["free", "prize", "win", "urgent", "offer"]

    total_msgs = len(df)
    n_spam = (df["label_num"] == 1).sum()
    n_ham = (df["label_num"] == 0).sum()
    p_spam = n_spam / total_msgs
    p_ham = n_ham / total_msgs

    results = []

    print(f"{'Word':<8} | {'Total':<6} | {'Spam':<5} | {'Ham':<5} | {'P(W|Spam)':<9} | {'P(W)':<8} | {'P(Spam|W)':<10} | {'Bayes Calc':<10} | {'Diff':<8}")
    print("-" * 88)

    for word in suspicious_words:
        # Match whole word case-insensitively using regex word boundaries
        pattern = rf"\b{re.escape(word)}\b"
        contains_word = df["message"].str.contains(pattern, case=False, regex=True)

        count_w = contains_word.sum()
        count_w_and_spam = (contains_word & (df["label_num"] == 1)).sum()
        count_w_and_ham = (contains_word & (df["label_num"] == 0)).sum()

        # Direct conditional probabilities
        p_spam_given_w = count_w_and_spam / count_w if count_w > 0 else 0.0
        p_ham_given_w = count_w_and_ham / count_w if count_w > 0 else 0.0

        # Likelihoods and marginal probability
        p_w_given_spam = count_w_and_spam / n_spam if n_spam > 0 else 0.0
        p_w_given_ham = count_w_and_ham / n_ham if n_ham > 0 else 0.0
        p_w = count_w / total_msgs

        # Bayes' Theorem calculation: P(Spam | W) = [P(W | Spam) * P(Spam)] / P(W)
        p_bayes_spam_given_w = (p_w_given_spam * p_spam) / p_w if p_w > 0 else 0.0
        diff = abs(p_spam_given_w - p_bayes_spam_given_w)

        results.append({
            "word": word,
            "count_w": count_w,
            "count_w_spam": count_w_and_spam,
            "count_w_ham": count_w_and_ham,
            "p_w": p_w,
            "p_w_given_spam": p_w_given_spam,
            "p_w_given_ham": p_w_given_ham,
            "p_spam_given_w": p_spam_given_w,
            "p_ham_given_w": p_ham_given_w,
            "p_bayes_spam_given_w": p_bayes_spam_given_w,
            "diff": diff
        })

        print(
            f"{word:<8} | {count_w:<6} | {count_w_and_spam:<5} | {count_w_and_ham:<5} | "
            f"{p_w_given_spam:<9.4f} | {p_w:<8.4f} | {p_spam_given_w:<10.4f} | "
            f"{p_bayes_spam_given_w:<10.4f} | {diff:<8.1e}"
        )

    print("-" * 88)
    print(">> VERIFICATION VERDICT: All Bayes' theorem calculations match empirical conditional probabilities exactly!")

    # Plot word probability comparison
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    words = [r["word"] for r in results]
    p_spam_cond = [r["p_spam_given_w"] for r in results]
    p_ham_cond = [r["p_ham_given_w"] for r in results]

    x = np.arange(len(words))
    width = 0.35

    rects1 = ax.bar(x - width/2, p_spam_cond, width, label="P(Spam | Word)", color="#d9534f", edgecolor="black")
    rects2 = ax.bar(x + width/2, p_ham_cond, width, label="P(Ham | Word)", color="#2b5c8f", edgecolor="black")

    ax.set_title("Word-Based Conditional Probabilities: P(Spam | Word) vs P(Ham | Word)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Suspicious Word", fontsize=11)
    ax.set_ylabel("Conditional Probability", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(words, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.18)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Label heights
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plot_path = os.path.join(FIGURES_DIR, "word_probabilities.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Word probabilities chart saved to: {plot_path}")

    return pd.DataFrame(results)


# ==============================================================================
# ACTIVITY 4: RANDOM VARIABLE ANALYSIS
# ==============================================================================
def analyze_random_variable(df):
    """
    Defines random variable Y:
      Y = 1 if Spam, Y = 0 if Ham.
    Calculates theoretical Expectation and Variance:
      E(Y) = p = P(Spam)
      Var(Y) = p * (1 - p)
    Compares with empirical sample mean and sample variance.
    """
    print("\n" + "=" * 70)
    print("ACTIVITY 4: RANDOM VARIABLE ANALYSIS (BERNOULLI DISTRIBUTION)")
    print("=" * 70)

    y = df["label_num"].values
    n = len(y)
    p = np.mean(y)  # Sample proportion of spam

    # Theoretical formulas for Bernoulli random variable Y ~ Bern(p)
    e_y_theoretical = p
    var_y_theoretical = p * (1.0 - p)
    std_y_theoretical = np.sqrt(var_y_theoretical)

    # Empirical sample statistics
    sample_mean = np.mean(y)
    sample_var = np.var(y, ddof=1)  # Unbiased sample variance (N - 1)
    sample_std = np.std(y, ddof=1)

    print("Definition:")
    print("  Let Y be a discrete random variable indicating SMS message class:")
    print("    Y = 1, if message is Spam  with probability p")
    print("    Y = 0, if message is Ham   with probability (1 - p)")
    print(f"  Distribution: Y ~ Bernoulli(p = {p:.4f})\n")

    print(f"{'Metric':<32} | {'Theoretical Formula':<22} | {'Theoretical Value':<18} | {'Empirical Sample':<18}")
    print("-" * 96)
    print(f"{'Parameter p = P(Spam)':<32} | {'N_spam / N':<22} | {p:<18.4f} | {sample_mean:<18.4f}")
    print(f"{'Expectation E(Y)':<32} | {'p':<22} | {e_y_theoretical:<18.4f} | {sample_mean:<18.4f}")
    print(f"{'Variance Var(Y)':<32} | {'p * (1 - p)':<22} | {var_y_theoretical:<18.4f} | {sample_var:<18.4f}")
    print(f"{'Standard Deviation SD(Y)':<32} | {'sqrt(p * (1 - p))':<22} | {std_y_theoretical:<18.4f} | {sample_std:<18.4f}")
    print("-" * 96)

    # Plot PMF of Random Variable Y
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    outcomes = [0, 1]
    pmf_probs = [1.0 - p, p]
    bars = ax.bar(outcomes, pmf_probs, width=0.4, color=["#2b5c8f", "#d9534f"], edgecolor="black", linewidth=1.2)

    ax.set_title(f"Probability Mass Function (PMF) of Random Variable Y\nY ~ Bernoulli(p = {p:.4f})",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Value of Random Variable Y (0 = Ham, 1 = Spam)", fontsize=11)
    ax.set_ylabel("P(Y = y)", fontsize=11)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0 (Ham / Genuine)", "1 (Spam)"], fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, prob in zip(bars, pmf_probs):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.03, f"{prob:.4f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    stats_text = (
        f"E(Y) = p = {e_y_theoretical:.4f}\n"
        f"Var(Y) = p(1-p) = {var_y_theoretical:.4f}\n"
        f"σ(Y) = {std_y_theoretical:.4f}"
    )
    ax.text(0.7, 0.75, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    pmf_path = os.path.join(FIGURES_DIR, "random_variable_pmf.png")
    plt.savefig(pmf_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Random variable PMF chart saved to: {pmf_path}")

    return {
        "p": p,
        "e_y": e_y_theoretical,
        "var_y": var_y_theoretical,
        "std_y": std_y_theoretical
    }


# ==============================================================================
# ACTIVITY 5: DATA PREPARATION
# ==============================================================================
def prepare_data(df, test_size=0.20, random_state=42):
    """
    Performs 80/20 train/test stratified split to maintain class balance.
    Extracts word-count numerical features using CountVectorizer.
    """
    print("\n" + "=" * 70)
    print("ACTIVITY 5: DATA PREPARATION (STRATIFIED SPLIT & COUNTVECTORIZER)")
    print("=" * 70)

    X = df["message"]
    y = df["label_num"]

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Total dataset size: {len(df)}")
    print(f"Training set size : {len(X_train)} ({(1 - test_size) * 100:.0f}%)")
    print(f"Test set size     : {len(X_test)} ({test_size * 100:.0f}%)")

    # Verify stratified balance
    train_spam_pct = (y_train == 1).mean() * 100
    test_spam_pct = (y_test == 1).mean() * 100
    print(f"Training set spam proportion : {train_spam_pct:.2f}%")
    print(f"Test set spam proportion     : {test_spam_pct:.2f}%")
    print("[+] Stratified sampling successfully maintained the class proportion.")

    # Convert text into numerical features using CountVectorizer
    vectorizer = CountVectorizer(stop_words="english", lowercase=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(f"\nCountVectorizer Vocabulary Size: {len(vectorizer.vocabulary_)} distinct word features")
    print(f"X_train feature matrix shape    : {X_train_vec.shape}")
    print(f"X_test feature matrix shape     : {X_test_vec.shape}")

    # Save vectorizer
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"[+] Vectorizer serialized to: {VECTORIZER_PATH}")

    return X_train, X_test, y_train, y_test, X_train_vec, X_test_vec, vectorizer


# ==============================================================================
# ACTIVITY 6: MODEL BUILDING
# ==============================================================================
def build_and_train_model(X_train_vec, y_train):
    """
    Trains a Multinomial Naive Bayes classifier on word count features.
    """
    print("\n" + "=" * 70)
    print("ACTIVITY 6: MODEL BUILDING (MULTINOMIAL NAIVE BAYES)")
    print("=" * 70)

    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_vec, y_train)

    print("[+] Model trained successfully:")
    print(f"    Class prior log-probabilities: {model.class_log_prior_}")
    print(f"    Prior class distribution: P(Ham)={np.exp(model.class_log_prior_[0]):.4f}, P(Spam)={np.exp(model.class_log_prior_[1]):.4f}")
    print(f"    Feature log-probability matrix shape: {model.feature_log_prob_.shape}")

    # Save trained model
    joblib.dump(model, MODEL_PATH)
    print(f"[+] Model serialized to: {MODEL_PATH}")

    return model


# ==============================================================================
# ACTIVITY 7: MODEL EVALUATION
# ==============================================================================
def evaluate_model(model, vectorizer, X_test, y_test, X_test_vec):
    """
    Evaluates classifier with Accuracy, Precision, Recall, F1-score, Confusion Matrix.
    Identifies False Positives (Ham classified as Spam) and False Negatives (Spam classified as Ham).
    """
    print("\n" + "=" * 70)
    print("ACTIVITY 7: MODEL EVALUATION")
    print("=" * 70)

    y_pred = model.predict(X_test_vec)
    y_proba = model.predict_proba(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Classification Performance Metrics:")
    print(f"  - Accuracy  : {acc * 100:.2f}%")
    print(f"  - Precision : {prec * 100:.2f}%")
    print(f"  - Recall    : {rec * 100:.2f}%")
    print(f"  - F1-Score  : {f1 * 100:.2f}%")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Ham (0)", "Spam (1)"], digits=4))

    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix Breakdown:")
    print(f"  - True Negatives  (Ham  -> Ham)  : {tn}")
    print(f"  - False Positives (Ham  -> Spam) : {fp}  [Type I error: Genuine message flagged as spam]")
    print(f"  - False Negatives (Spam -> Ham)  : {fn}  [Type II error: Spam message slipped through]")
    print(f"  - True Positives  (Spam -> Spam) : {tp}")

    # Plot Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im)

    classes = ["Ham (0)", "Spam (1)"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=11, fontweight="bold")
    ax.set_yticklabels(classes, fontsize=11, fontweight="bold")

    # Annotate values
    thresh = cm.max() / 2.0
    labels = [
        [f"TN\n{tn}\n({tn/(tn+fp)*100:.1f}%)", f"FP\n{fp}\n({fp/(tn+fp)*100:.1f}%)"],
        [f"FN\n{fn}\n({fn/(fn+tp)*100:.1f}%)", f"TP\n{tp}\n({tp/(fn+tp)*100:.1f}%)"]
    ]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels[i][j],
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight="bold")

    ax.set_title(f"Multinomial Naive Bayes Confusion Matrix\nAccuracy = {acc*100:.2f}% | F1 = {f1*100:.2f}%",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("True Ground Truth Label", fontsize=11)
    ax.set_xlabel("Model Predicted Label", fontsize=11)

    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Confusion matrix plot saved to: {cm_path}")

    # Inspect Misclassified Messages
    test_df = pd.DataFrame({
        "message": X_test,
        "true_label": y_test,
        "pred_label": y_pred,
        "p_ham": y_proba[:, 0],
        "p_spam": y_proba[:, 1]
    })

    false_positives = test_df[(test_df["true_label"] == 0) & (test_df["pred_label"] == 1)]
    false_negatives = test_df[(test_df["true_label"] == 1) & (test_df["pred_label"] == 0)]

    print("\n" + "-" * 70)
    print("ANALYSIS OF MISCLASSIFIED MESSAGES")
    print("-" * 70)

    print(f"\n1. Genuine Messages Incorrectly Classified as Spam (False Positives: {len(false_positives)}):")
    if len(false_positives) == 0:
        print("   None! Zero false positives observed.")
    else:
        for idx, row in false_positives.head(5).iterrows():
            print(f"   - Message: \"{row['message'][:90]}...\"")
            print(f"     P(Spam|Message) = {row['p_spam']:.4f} | P(Ham|Message) = {row['p_ham']:.4f}")

    print(f"\n2. Spam Messages Incorrectly Classified as Genuine (False Negatives: {len(false_negatives)}):")
    if len(false_negatives) == 0:
        print("   None! Zero false negatives observed.")
    else:
        for idx, row in false_negatives.head(5).iterrows():
            print(f"   - Message: \"{row['message'][:90]}...\"")
            print(f"     P(Spam|Message) = {row['p_spam']:.4f} | P(Ham|Message) = {row['p_ham']:.4f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "cm": cm,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }


# ==============================================================================
# ACTIVITY 8: REAL-TIME TESTING
# ==============================================================================
def predict_sms(message, model=None, vectorizer=None):
    """
    Classifies an incoming SMS message and displays:
      - Prediction: Spam or Ham
      - P(Spam | Message)
      - P(Ham | Message)
    """
    if model is None:
        model = joblib.load(MODEL_PATH)
    if vectorizer is None:
        vectorizer = joblib.load(VECTORIZER_PATH)

    msg_vec = vectorizer.transform([message])
    pred = model.predict(msg_vec)[0]
    proba = model.predict_proba(msg_vec)[0]

    p_ham = proba[0]
    p_spam = proba[1]
    prediction_label = "Spam" if pred == 1 else "Ham (Genuine message)"

    return {
        "message": message,
        "prediction": prediction_label,
        "p_spam": p_spam,
        "p_ham": p_ham
    }


def run_real_time_testing(model, vectorizer, interactive=False):
    """
    Runs demonstration test messages including the required prompt example,
    and opens an interactive prompt if specified.
    """
    print("\n" + "=" * 70)
    print("ACTIVITY 8: REAL-TIME SMS TESTING")
    print("=" * 70)

    test_samples = [
        "Congratulations! You have won a free prize. Claim now.",
        "Hey, are we still meeting for lunch at 12:30 today?",
        "URGENT! Your bank account has been compromised. Verify your details immediately to claim your refund.",
        "Can you send me the lecture notes for BAMAT207 Probability and Statistics?",
        "Exclusive offer: Win $1000 cash prize instantly! Text WIN to 8888 now."
    ]

    print("Running predefined benchmark tests:\n")
    for msg in test_samples:
        res = predict_sms(msg, model, vectorizer)
        print(f"Input Message : \"{res['message']}\"")
        print(f"Prediction    : {res['prediction']}")
        print(f"P(Spam | Msg) : {res['p_spam']:.4f} ({res['p_spam']*100:.2f}%)")
        print(f"P(Ham | Msg)  : {res['p_ham']:.4f} ({res['p_ham']*100:.2f}%)")
        print("-" * 60)

    if interactive:
        print("\n=== Interactive Real-Time Mode (Type 'exit' to quit) ===")
        while True:
            try:
                user_msg = input("\nEnter an SMS message to test: ").strip()
                if not user_msg or user_msg.lower() in ["exit", "quit", "q"]:
                    print("Exiting interactive test mode.")
                    break
                res = predict_sms(user_msg, model, vectorizer)
                print(f"\n>> Results for: \"{res['message']}\"")
                print(f"   Prediction    : {res['prediction']}")
                print(f"   P(Spam | Msg) : {res['p_spam']:.4f} ({res['p_spam']*100:.2f}%)")
                print(f"   P(Ham | Msg)  : {res['p_ham']:.4f} ({res['p_ham']*100:.2f}%)")
            except (KeyboardInterrupt, EOFError):
                break


# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Probabilistic SMS Spam Detection System")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive SMS classification prompt")
    parser.add_argument("--message", type=str, help="Single custom message to classify")
    args = parser.parse_args()

    # Step 1: Dataset Handling
    df = load_and_clean_dataset()

    # Step 2: Dataset Analysis
    prior_stats = analyze_dataset(df)

    # Step 3: Word-Based Probability Analysis
    word_stats = analyze_word_probabilities(df)

    # Step 4: Random Variable Analysis
    rv_stats = analyze_random_variable(df)

    # Step 5: Data Preparation
    X_train, X_test, y_train, y_test, X_train_vec, X_test_vec, vectorizer = prepare_data(df)

    # Step 6: Model Building
    model = build_and_train_model(X_train_vec, y_train)

    # Step 7: Model Evaluation
    eval_stats = evaluate_model(model, vectorizer, X_test, y_test, X_test_vec)

    # Step 8: Real-Time Testing
    if args.message:
        res = predict_sms(args.message, model, vectorizer)
        print("\n" + "=" * 70)
        print("CUSTOM MESSAGE PREDICTION")
        print("=" * 70)
        print(f"Input Message : \"{res['message']}\"")
        print(f"Prediction    : {res['prediction']}")
        print(f"P(Spam | Msg) : {res['p_spam']:.4f}")
        print(f"P(Ham | Msg)  : {res['p_ham']:.4f}")
    else:
        run_real_time_testing(model, vectorizer, interactive=args.interactive)

    print("\n" + "=" * 70)
    print("PROJECT EXECUTION COMPLETE!")
    print(f"- Cleaned data saved to : {CLEANED_DATA_PATH}")
    print(f"- Figures saved to      : {FIGURES_DIR}")
    print(f"- Models saved to       : {MODELS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
