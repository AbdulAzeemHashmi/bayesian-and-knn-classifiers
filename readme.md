<div align="center">

<!-- Animated header banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:667eea,100:764ba2&height=200&section=header&text=Machine%20Learning%20Assignment%201&fontSize=36&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Bayesian%20Classification%20and%20KNN%20Classifiers&descAlignY=58&descSize=18" width="100%"/>

<!-- Badges -->
<br/>

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Only-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge&logo=checkmarx&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-Submitted-181717?style=for-the-badge&logo=github&logoColor=white)

</div>

---

## Student Information

| Field | Details |
|---|---|
| **Name** | Abdul Azeem Hashmi |
| **Roll Number** | 24I-2013 |
| **Section** | BAI-5A |
| **Course** | Machine Learning |
| **Due Date** | October 11, 2026 |
| **Total Marks** | 135 Base + 8 Bonus |

---

## What This Assignment Is About

This assignment has two main questions. Both questions require building machine learning classifiers completely from scratch, meaning without using any machine learning libraries except where specifically allowed.

**Question 1** asks you to build a Bayesian classifier that predicts what will happen to an animal at a shelter. The classifier uses a mix of category-based features and number-based features.

**Question 2** asks you to audit a Naive Bayes review system for a fictional company called CineTrust Analytics. You test the system, find its weaknesses, build a better version, and write a report.

---

## Repository Structure

```
bayesian-and-knn-classifiers/
|
|-- 24i-2013_AbdulAzeem_BAI-5A_Notebook1.ipynb     <- SUBMIT THIS (Question 1)
|-- 24i-2013_AbdulAzeem_BAI-5A_Question2.ipynb     <- SUBMIT THIS (Question 2)
|-- 24i-2013_AbdulAzeem_BAI-5A_Report.pdf          <- SUBMIT THIS (Report PDF)
|
|-- aac_intakes_outcomes.csv                        <- Dataset for Question 1
|
|-- dataset Question2/
|   |-- imdb1/
|   |   |-- pos/     <- 1000 positive IMDB movie reviews (.txt files)
|   |   |-- neg/     <- 1000 negative IMDB movie reviews (.txt files)
|   |-- english.stop <- List of 571 English stopwords
|
|-- readme.md                                       <- This file
|
|-- Notebook1_BayesianClassification.ipynb          <- Original template (Question 1)
|-- Notebook2_CineTrust_Review_Audit.ipynb          <- Original template (Question 2)
|-- i24-2013_Abdul-Azeem_1_BayesianClassification.ipynb  <- Alternate copy
|-- i24-2013_Abdul-Azeem_Report.pdf                 <- Alternate report copy
|-- Assignment1.pdf                                 <- Original assignment PDF
```

---

## File Details

<details>
<summary><strong>Click to expand - Notebook 1 (Question 1) Details</strong></summary>

### 24i-2013_AbdulAzeem_BAI-5A_Notebook1.ipynb

**Purpose:** Predict shelter animal outcomes using a mixed Bayesian classifier built purely with `numpy` and `math`.

**Dataset:** Austin Animal Center intakes and outcomes (`aac_intakes_outcomes.csv`)
- Total records: 79,672
- Train split: 63,737 records
- Test split: 15,935 records

**Four outcome classes:**

| Class | Description | Train Count |
|---|---|---|
| Adoption | Animal adopted by a person | 26,875 |
| Transfer | Animal moved to another shelter | 19,039 |
| Return to Owner | Animal returned to its owner | 11,833 |
| Other | Euthanasia, died, missing, etc. | 5,990 |

**Features used:**

| Feature | Type | How it is encoded |
|---|---|---|
| Animal Type | Categorical | Dog=1, Cat=0, other=0 |
| Sex Upon Intake | Categorical | Neutered/Spayed=2, Intact=1, Unknown=0 |
| Color | Categorical | First word of color string, uppercase |
| Breed | Categorical | Top-20 breeds from training set, rest = "Other" |
| Age in Days | Numeric (Gaussian) | Parsed from text like "2 years", z-score normalized |
| Time in Shelter | Numeric (Gaussian) | Calculated from intake and outcome dates, z-score normalized |

**Key settings (derived from roll number 24I-2013):**
- Random seed: `(242013 % 97) + 7 = 102`
- Laplace smoothing K: `0.5 + ((2+4+2+0+1+3) % 5) / 10 = 0.70`
- Test row for mandatory log: `r = 242013 % 97 = 95`

**Results:**

| Metric | Value |
|---|---|
| Overall Accuracy | 55.23% |
| Macro-F1 Score | 0.5629 |
| Adoption F1 | 0.5598 |
| Transfer F1 | 0.5401 |
| Return to Owner F1 | 0.5360 |
| Other F1 | 0.6155 |

**Libraries allowed:** `numpy`, `math`, `csv`, `os` only. No pandas, no scikit-learn.

</details>

<details>
<summary><strong>Click to expand - Notebook 2 (Question 2) Details</strong></summary>

### 24i-2013_AbdulAzeem_BAI-5A_Question2.ipynb

**Purpose:** Audit a Naive Bayes sentiment classifier for CineTrust Analytics using the IMDB movie review dataset.

**Dataset:** IMDB Movie Reviews
- Total reviews: 2,000 (1,000 positive, 1,000 negative)
- Average review length: 746.3 words
- Train split: 1,400 reviews
- Validation split: 300 reviews
- Test split: 300 reviews

**Parts covered:**

| Part | Topic | Marks |
|---|---|---|
| A | Bayes theorem derivation, manual calculation, log-space reasoning | 15 |
| B | Text preprocessing, MultinomialNB and BernoulliNB from scratch | 25 |
| C | Prior probability shift analysis (50% vs 68% positive prior) | 10 |
| D | Confidence calibration and overconfidence investigation | 15 |
| E | Stress testing with negation and sarcasm, negation tagging fix | 20 |
| F | KNN classifier from scratch with cosine distance | 20 |
| G | Scikit-learn sanity check and executive recommendation memo | 10 |
| Bonus | Bigram extension and temperature scaling calibration | 8 |

**Results summary:**

| Model | Test Accuracy | Macro-F1 | Speed |
|---|---|---|---|
| Multinomial Naive Bayes | 84.00% | 0.8400 | 25,000+ reviews/sec |
| Bernoulli Naive Bayes | 82.33% | 0.8233 | 25,000+ reviews/sec |
| KNN (k=7, cosine) | 79.33% | 0.7933 | ~480 reviews/sec |
| Scikit-learn MultinomialNB | 84.00% | 0.8400 | Matches from-scratch |

**Key findings:**
1. The model assigns over 99.9% confidence to 93.3% of all test reviews. This is overconfidence caused by multiplying many small probabilities together without calibration.
2. When the positive prior changes from 50% to 68%, 12 predictions flip without any change in review text.
3. Standard bag-of-words fails on 7 out of 8 negation and sarcasm sentences. The review "The movie was not good at all" gets classified as positive because of the word "good".
4. Adding negation tagging (converting "not good" to "NOT_good") fixes these errors.

**Libraries used:** `numpy`, `pandas`, `matplotlib`, `os`, `re`, `math`, `time`, `collections`. Scikit-learn is used only in Part G as a labeled sanity check.

</details>

<details>
<summary><strong>Click to expand - Dataset Details</strong></summary>

### aac_intakes_outcomes.csv

This file contains shelter intake and outcome records from the Austin Animal Center.

- **Size:** ~30 MB
- **Records:** 79,672
- **Key columns used:**

| Column | What it contains |
|---|---|
| animal_type | Type of animal (Dog, Cat, Bird, etc.) |
| sex_upon_intake | Sex and reproductive status at intake |
| breed | Breed description |
| color | Color description |
| age_upon_intake | Age as text (e.g. "2 years", "3 months") |
| intake_datetime | Date and time the animal entered the shelter |
| outcome_type | What happened to the animal (Adoption, Transfer, etc.) |
| outcome_datetime | Date and time the animal left the shelter |

### dataset Question2/

Contains the IMDB dataset for sentiment classification.

- `imdb1/pos/` - 1,000 text files, each containing one positive movie review
- `imdb1/neg/` - 1,000 text files, each containing one negative movie review
- `english.stop` - 571 common English stopwords to remove during preprocessing

</details>

---

## How to Run the Notebooks

**Requirements:**
```
Python 3.12+
numpy
pandas
matplotlib
scikit-learn
jupyter
```

**Install dependencies:**
```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

**Run Notebook 1:**
```bash
jupyter notebook 24i-2013_AbdulAzeem_BAI-5A_Notebook1.ipynb
```
Make sure `aac_intakes_outcomes.csv` is in the same folder before running.

**Run Notebook 2:**
```bash
jupyter notebook 24i-2013_AbdulAzeem_BAI-5A_Question2.ipynb
```
Make sure the `dataset Question2` folder is in the same directory before running.

---

## Submission Checklist

| Item | Status |
|---|---|
| Notebook 1 all cells executed and outputs visible | Done |
| Notebook 2 all cells executed and outputs visible | Done |
| Only numpy and math used in Notebook 1 | Done |
| Scikit-learn used only in Part G of Notebook 2 (labeled) | Done |
| Mandatory logs printed in Notebook 1 | Done |
| Correct file naming format (24i-2013_AbdulAzeem_BAI-5A) | Done |
| No em dashes or en dashes in any file | Done |
| Report PDF prepared | Done |
| All cells have markdown explanations | Done |

---

## Model Architecture Overview

```
QUESTION 1 - MIXED BAYESIAN CLASSIFIER
========================================

          Input Features
               |
     +---------+---------+
     |                   |
  Categorical          Numeric
  Features             Features
  (AnimalType,         (age_days,
   SexUponIntake,       time_in_shelter)
   Color, Breed)              |
     |                  Z-score Scaling
  Label Encoding              |
  + Laplace K=0.70     Gaussian NB fit
     |                  (mu, variance)
  Categorical NB              |
     |                        |
     +----------+-------------+
                |
         Log-Posterior
         (log P(c) + sum of log-likelihoods)
                |
         Log-Sum-Exp Normalization
                |
         argmax -> Predicted Class


QUESTION 2 - TEXT SENTIMENT CLASSIFIER
========================================

   Raw Review Text
         |
    Preprocessing
    (lowercase, remove numbers,
     remove stopwords, keep negation words)
         |
    Bag-of-Words Vectorization
    (term frequency matrix, 14,827 vocabulary)
         |
    +----+----+
    |         |
MultinomialNB  BernoulliNB
(word counts)  (word presence)
    |         |
    +----+----+
         |
    Evaluation: Accuracy, F1, Confusion Matrix
```

---

## Assignment Rules Followed

- This is an individual assignment. No group work.
- All core models are implemented from scratch without machine learning libraries.
- Scikit-learn is used only in Part G of Question 2 as a clearly labeled sanity check.
- All cells are executed and all outputs and plots are visible.
- Every number and result comes from code run in the notebook.
- Markdown explanations are added for every task.
- No plagiarism.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:667eea,100:764ba2&height=100&section=footer&animation=fadeIn" width="100%"/>

![Made with](https://img.shields.io/badge/Made%20with-NumPy%20and%20Math%20Only-013243?style=flat-square&logo=numpy)
![No sklearn](https://img.shields.io/badge/No%20sklearn%20in%20core%20models-F7931E?style=flat-square)
![Scratch](https://img.shields.io/badge/Built%20from-Scratch-blueviolet?style=flat-square)

</div>
