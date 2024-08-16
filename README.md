# AI_Chatbot

## Description

The AI Chatbot is a banking assistant designed to handle various customer queries related to credit and debit cards. It utilizes machine learning models to provide accurate and helpful responses to user inquiries.

### Features

- **Support Vector Machine (SVM)**: Uses a linear kernel for classifying user queries.
- **Random Forest Classifier**: A robust ensemble method for classifying user queries.
- **TF-IDF Vectorization**: Transforms text data into numerical vectors for model training.
- **Oversampling and Undersampling**: Addresses class imbalance in the dataset.

### Installation

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

### Usage

1. **Run the Chatbot:**
   ```bash
   python chatbot.py
   ```

2. **Interact with the Chatbot:**
   - You will be prompted to enter your questions.
   - Type `quit`, `exit`, or `q` to exit the chatbot.

### File Structure

- `chatbot.py`: Contains the implementation of the `BankingChatbot` class and the main script for running the chatbot.
- `requirements.txt`: Lists the Python packages required to run the chatbot.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `conncetion_info.txt : Connection information.

### Dependencies

- `scikit-learn`: For machine learning models and text vectorization.
- `imblearn`: For handling imbalanced datasets.
- `termcolor`: For colored terminal output.

### Example Dataset

The example dataset used in this project contains sample queries and responses related to banking services. It is used to train the machine learning models.

### Contribution

Contributions are welcome! Please open an issue or submit a pull request if you would like to contribute to this project.

### License

## Author
Aarya Shirbhate
