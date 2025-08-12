##model:{https://huggingface.co/Thegame1161/tiny-bert-detect-fake-news}
# Fine-Tuning DistilBERT for Fake News Detection 📰

This repository contains an end-to-end project for fine-tuning a `distilbert-base-uncased` model to classify news article headlines as "Real" or "Fake". The entire workflow is built using the Hugging Face ecosystem, demonstrating advanced training techniques for building a fast and accurate text classifier.

The project proceeds from data loading and cleaning to model training, evaluation, and finally, saving the model for easy inference with the `pipeline` API.

## ✨ Key Features

  * **Efficient Model**: Utilizes `distilbert-base-uncased`, a smaller, faster, and lighter version of BERT, ideal for quick training and inference.
  * **Advanced Training Techniques**: Implements several modern training strategies to improve performance and efficiency:
      * **Mixed-Precision Training (`fp16`)**: Reduces memory usage and speeds up training on modern GPUs.
      * **Learning Rate Warmup**: Stabilizes training in the early stages for better convergence.
      * **Periodic Evaluation**: Monitors validation metrics during training epochs to save the best-performing model.
  * **End-to-End Workflow**: Covers every step from data loading and cleaning to tokenization, training, evaluation, and deployment.
  * **Hugging Face Integration**: Leverages the high-level `Trainer` API for a streamlined workflow and the `pipeline` function for simple, production-ready inference.

-----

## ⚙️ Project Workflow

The project follows a structured machine learning pipeline:

1.  **Environment Setup**: Installs all necessary libraries, including `transformers`, `datasets`, and `accelerate`.
2.  **Data Loading and Cleaning**: Loads a dataset of news articles from an Excel file using `pandas` and removes any rows with missing values to ensure data quality.
3.  **Exploratory Data Analysis (EDA)**:
      * Visualizes the class distribution to confirm the dataset is well-balanced between "Real" and "Fake" news.
      * Analyzes the token length of article titles and texts, ultimately deciding to **focus on titles only** for an efficient and lightweight model.
4.  **Data Splitting**: The data is split into training (70%), validation (10%), and testing (20%) sets. The split is stratified to maintain the same proportion of real and fake news in each set.
5.  **Tokenization**: A `distilbert-base-uncased` tokenizer is used to convert the news headlines into numerical IDs suitable for the model.
6.  **Model Training**:
      * `AutoModelForSequenceClassification` is loaded with a new classification head configured for our binary task.
      * `TrainingArguments` are configured with advanced settings like mixed-precision (`fp16`), learning rate warmup, and periodic evaluation to load the best model at the end.
      * The `Trainer` API is used to fine-tune the model on the training dataset.
7.  **Evaluation**: The model's final performance is measured on the unseen test set using a `classification_report` to assess precision, recall, and F1-score.
8.  **Saving and Inference**: The fine-tuned model is saved to disk and loaded into a `text-classification` `pipeline` for easy use on new, unseen headlines.

-----

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ and a CUDA-enabled GPU to take full advantage of the training optimizations.

### Installation

1.  Clone the repository to your local machine:

    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
    cd YOUR_REPOSITORY_NAME
    ```

2.  Install the required Python libraries:

    ```bash
    pip install -U pandas torch scikit-learn matplotlib seaborn openpyxl
    pip install -U transformers accelerate datasets
    ```

-----

## ▶️ How to Use the Trained Model

The fine-tuned model is saved in the `fake_news` directory. The easiest way to use it for prediction is with the Hugging Face `pipeline` function, which handles all the necessary preprocessing and post-processing steps.

```python
from transformers import pipeline

# Load the saved model into a text-classification pipeline
classifier = pipeline('text-classification', model='fake_news')

# Example of a real news headline
real_headline = "Researchers Publish Findings on Efficacy of New Alzheimer's Drug"

# Example of a potentially fake news headline
fake_headline = "You Won't Believe What This Celebrity Eats for Breakfast, Secret Revealed!"

# Get predictions
print(f"Headline: '{real_headline}'")
print("Prediction:", classifier(real_headline))

print(f"\nHeadline: '{fake_headline}'")
print("Prediction:", classifier(fake_headline))
```

### Expected Output

```
Headline: 'Researchers Publish Findings on Efficacy of New Alzheimer's Drug'
Prediction: [{'label': 'Real', 'score': 0.99...}]

Headline: 'You Won't Believe What This Celebrity Eats for Breakfast, Secret Revealed!'
Prediction: [{'label': 'Fake', 'score': 0.98...}]
```

-----

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
