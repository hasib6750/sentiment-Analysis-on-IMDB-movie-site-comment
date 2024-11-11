# IMDB Sentiment Analysis with Machine Learning Models

## Overview
This project is a comprehensive sentiment analysis pipeline built using Python. It leverages the IMDB movie review dataset to classify reviews as positive or negative. The analysis includes data cleaning, feature extraction, and training multiple machine learning models to evaluate performance. The project is ideal for those looking to understand the end-to-end process of text preprocessing, feature engineering, and model evaluation in a machine learning context.

## Key Features
- **Data Preprocessing**: Includes data exploration, handling missing values, and removal of duplicates.
- **Text Cleaning**: Utilizes regular expressions, stopword removal, and stemming for efficient text processing.
- **Feature Extraction**: Employs TF-IDF vectorization to convert textual data into numerical features.
- **Model Training**:
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - Logistic Regression
  - Support Vector Machine (SVM)
- **Evaluation**: Provides accuracy scores and detailed classification reports for each model.
- **Visualization**: Includes word cloud visualizations and a comparison of model accuracies using bar charts.

## Requirements
- Python (>=3.7)
- Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `wordcloud`, `scikit-learn`, `nltk`

## Installation
Clone the repository and install the necessary Python libraries using:
```bash
git clone https://github.com/hasib6750/sentiment-Analysis-on-IMDB-movie-site-comment.git
cd sentiment-Analysis-on-IMDB-movie-site-comment
pip install -r requirements.txt
```

## Usage
Run the notebook or Python script to execute data preprocessing, model training, and evaluation:
```bash
python sentiment_Analysis_on_IMDB_movie_site_comment.ipynb
```

## Results
The project outputs accuracy scores for each trained model and displays comparative bar charts to visualize performance. The Random Forest, Gradient Boosting, Logistic Regression, and SVM models are evaluated to determine the most effective approach for sentiment classification.

## Future Enhancements
- Hyperparameter tuning for model optimization.
- Implementation of deep learning models for improved performance.
- Integration with more robust NLP preprocessing techniques.

Feel free to contribute by opening an issue or submitting a pull request!
