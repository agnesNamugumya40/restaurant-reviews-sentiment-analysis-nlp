# Natural Language Processing — Restaurant Review Sentiment Analysis

A natural language processing (NLP) project that predicts whether a restaurant review is **positive or negative** from its text. The reviews are cleaned, converted into numbers with a **Bag of Words** model, and classified with **Naive Bayes**.

## Dataset

**File:** `Restaurant_Reviews.tsv` (tab-separated)

- 1,000 restaurant reviews
- Balanced: 500 positive and 500 negative

| Column | Description |
|---|---|
| `Review` | The text of the review |
| `Liked` | Target variable — 1 (positive) or 0 (negative) |

The file is tab-separated rather than comma-separated because reviews often contain commas. It is read with `quoting = 3` so that double quotes inside reviews are ignored.

## Workflow

The notebook (`natural_language_processing.ipynb`) follows a standard NLP pipeline:

1. **Import libraries** — NumPy, Matplotlib, Pandas
2. **Load dataset** — reads `Restaurant_Reviews.tsv` with a tab delimiter
3. **Clean the texts** — for each review:
   - removes everything except letters
   - converts to lowercase
   - splits into individual words
   - removes stopwords (common words like "the", "is", "and") but **keeps "not"**, since it changes the meaning of a review
   - stems each word to its root with the Porter Stemmer (e.g. "loved" → "love")
4. **Bag of Words model** — `CountVectorizer` turns the cleaned reviews into a matrix of word counts, keeping the 1,500 most frequent words
5. **Train/test split** — 80% training, 20% test (`random_state=0`)
6. **Model training** — Gaussian Naive Bayes classifier
7. **Test set predictions** — predicted vs. actual labels side by side
8. **Evaluation** — confusion matrix and accuracy score

Example of cleaning:

| Original review | Cleaned |
|---|---|
| Wow... Loved this place. | wow love place |
| Crust is not good. | crust not good |
| Not tasty and the texture was just nasty. | not tasti textur nasti |

## Requirements

```
numpy
matplotlib
pandas
nltk
scikit-learn
```

Install with:

```bash
pip install numpy matplotlib pandas nltk scikit-learn
```

The notebook downloads the NLTK stopwords list automatically with `nltk.download('stopwords')`.

## Usage

Open the notebook in Jupyter or Google Colab and run all cells in order:

```bash
jupyter notebook natural_language_processing.ipynb
```

## Model Configuration

```python
CountVectorizer(max_features = 1500)   # vocabulary of the 1,500 most frequent words
GaussianNB()                           # Naive Bayes classifier
```

## Results

Evaluated on the 200 reviews in the test set:

**Confusion matrix**

| | Predicted negative | Predicted positive |
|---|---:|---:|
| **Actual negative** | 55 | 42 |
| **Actual positive** | 12 | 91 |

**Accuracy: 73%** (146 of 200 reviews classified correctly)

- The model is good at recognising positive reviews: 91 of 103 (about 88%) were identified correctly.
- It is weaker on negative reviews: 42 of 97 were wrongly predicted as positive, the main source of error.

## Key Concepts

- **Stopwords**: very common words that carry little meaning on their own and are removed to reduce noise.
- **Stemming**: reducing words to their root form so that "love", "loved" and "loving" are treated as the same word.
- **Bag of Words**: represents each review as a vector of word counts, ignoring word order.
- **Sparse matrix**: most entries in the Bag of Words matrix are 0, since each review uses only a few of the 1,500 words.
- **Naive Bayes**: a probabilistic classifier that assumes each word contributes independently to the probability of a review being positive or negative.
