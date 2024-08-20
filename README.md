# AI Chatbot

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)

## Overview

The AI Chatbot is a banking assistant application designed to handle a variety of customer queries related to credit cards, debit cards, accounts, etc. It uses machine learning models to understand and respond to user inputs. The chatbot employs both Support Vector Machine (SVM) and Random Forest Classifier models for predicting responses based on user queries.

## Features

- **Support Vector Machine (SVM)**: A machine learning model that classifies user queries using a linear kernel.
- **Random Forest Classifier**: An ensemble model that uses multiple decision trees to classify user queries.
- **TF-IDF Vectorization**: Converts text data into numerical vectors, capturing the importance of words in the dataset.
- **Oversampling and Undersampling**: Techniques to balance class distribution in the training dataset.
- **Interactive CLI Interface**: A command-line interface allowing users to interact with the chatbot.

## Libraries Used

- **scikit-learn**: Provides tools for machine learning including:
  - `TfidfVectorizer` for transforming text data into numerical features.
  - `train_test_split` for splitting the dataset into training and testing sets.
  - `SVC` (Support Vector Classifier) for building the SVM model.
  - `RandomForestClassifier` for building the Random Forest model.
  - `accuracy_score` for evaluating model performance.
- **imblearn**: Contains methods for dealing with imbalanced datasets:
  - `RandomOverSampler` for oversampling the minority class.
  - `RandomUnderSampler` for undersampling the majority class.
- **termcolor**: Provides support for colored terminal output.
- **time**: Used for adding delays and timestamps in console output.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/aaryaya/AI_Chatbot.git
   cd AI_Chatbot
   ```

2. **Create and Activate Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Chatbot:**
   ```bash
   python chatbot.py
   ```

2. **Interact with the Chatbot:**
   - After running the script, the chatbot will prompt you to enter your questions.
   - Type your query related to banking services and press Enter.
   - The chatbot will respond based on the predictions from both SVM and Random Forest models.
   - Type `quit`, `exit`, or `q` to exit the chatbot.

## Code Explanation

- **Class `BankingChatbot`**:
  - **`__init__`**: Initializes the chatbot and sets up the TF-IDF vectorizer and machine learning models.
  - **`train_ml_models`**: Trains the SVM and Random Forest models using the provided dataset. The dataset is split into training and testing sets, and the models are evaluated for accuracy.
  - **`predict_svm_response`**: Uses the SVM model to predict the response for a given query.
  - **`predict_rf_response`**: Uses the Random Forest model to predict the response for a given query.
  - **`respond`**: Combines predictions from both models and formats the response.

- **Synthetic Dataset**:
  - A synthetic dataset is generated and balanced using oversampling and undersampling techniques for training the models.

- **Interactive CLI**:
  - A command-line interface allows users to interact with the chatbot. User inputs are timestamped, and the responses are displayed with a delay for a natural conversational experience.

## Example Dataset

The example dataset used for training contains sample queries and responses related to:
- Credit card rewards redemption.
- Annual fee inquiries.
- Credit limit increases.
- Security features.
- Debit card activation.
- Withdrawal limits.
- Benefits and transaction alerts.
- Types of accounts
  
## Contribution

Contributions to this project are welcome! Please open an issue or submit a pull request if you have improvements or suggestions.

## License

This project is open to all and is released under a permissive license. You are free to use, modify, and distribute the work without copyright restrictions.

## Author
Aarya Shirbhate
