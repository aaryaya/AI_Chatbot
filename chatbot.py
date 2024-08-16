from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler
from termcolor import colored
import time


class BankingChatbot:
    def __init__(self):
        # Create machine learning models
        self.svm_model = None
        self.rf_model = None
        self.vectorizer = TfidfVectorizer()

    def train_ml_models(self, dataset):
        # Assuming the dataset is a list of tuples (query, response)
        queries, responses = zip(*dataset)

        # Vectorize the queries
        X = self.vectorizer.fit_transform(queries)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, responses, test_size=0.2, random_state=42)

        # Train Support Vector Machine (SVM) classifier
        svm_classifier = SVC(kernel='linear')
        svm_classifier.fit(X_train, y_train)
        svm_predictions = svm_classifier.predict(X_test)
        svm_accuracy = accuracy_score(y_test, svm_predictions)
        self.svm_model = svm_classifier

        #__________________________We have two different classifiers to produce _________________________________

        # Train Random Forest classifier
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train, y_train)
        rf_predictions = rf_classifier.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_predictions)
        self.rf_model = rf_classifier

    def predict_svm_response(self, query):
        query_vectorized = self.vectorizer.transform([query])
        return self.svm_model.predict(query_vectorized)[0]

    def predict_rf_response(self, query):
        query_vectorized = self.vectorizer.transform([query])
        return self.rf_model.predict(query_vectorized)[0]

    def respond(self, message):
        # Use both SVM and Random Forest models to predict the response
        svm_response = self.predict_svm_response(message)
        rf_response = self.predict_rf_response(message)
        return f'Responding - SVM: {svm_response}, Random Forest: {rf_response}'




####
# Generate a synthetic imbalanced dataset for illustration
X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=1000, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply oversampling to the training set
oversampler = RandomOverSampler(sampling_strategy='minority', random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Apply undersampling to the training set
undersampler = RandomUnderSampler(sampling_strategy='majority', random_state=42)
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)


# Example usage with a dataset of queries and responses
banking_chatbot = BankingChatbot()

# Define a dataset for machine learning
dataset = [
    ("Can I redeem my credit card rewards?",
     "Absolutely! You can redeem your credit card rewards through your online account or by contacting our rewards center."),
    ("What is the annual fee for my credit card?",
     "The annual fee for your credit card is $X.XX. Let me know if you have any questions regarding the fee structure."),
    ("How do I apply for a credit limit increase?",
     "You can apply for a credit limit increase by filling out the online form available in your account settings."),
    ("Tell me about the security features of my credit card.",
     "Your credit card comes with advanced security features, including fraud monitoring and zero-liability protection. Rest assured, your transactions are secure."),
    ("Can I add an authorized user to my credit card?",
     "Certainly! To add an authorized user, log in to your account and navigate to the 'Manage Authorized Users' section."),
    #debit card
    ("How can I activate my debit card?",
     "You can activate your debit card by calling the activation number provided with the card or by using our online banking services. Follow the instructions to complete the activation."),
    ("What is the daily withdrawal limit on my debit card?",
     "The daily withdrawal limit on your debit card is $X,XXX. Keep in mind that this limit may vary based on your account type and any additional settings you've configured."),
    ("Tell me about the benefits of my debit card.",
     "Your debit card comes with benefits such as convenient access to your funds, secure transactions, and the ability to make purchases online and in-store. Check our website for detailed information."),
    ("Can I set up transaction alerts for my debit card?",
     "Yes, you can set up transaction alerts for your debit card through our online banking platform. Log in to your account and navigate to the 'Alerts' or 'Notifications' section."),
    ("What should I do if my debit card is lost or stolen?",
     "If your debit card is lost or stolen, report it immediately by contacting our 24/7 customer service hotline. We'll assist you in securing your account and issuing a replacement card."),
    ("How can I change the PIN on my debit card?",
     "You can change the PIN on your debit card by visiting one of our ATMs or using our online banking services. Navigate to the 'Card Services' or 'Manage PIN' section for instructions."),
    ("Tell me about the foreign transaction fees on my debit card.",
     "There is a foreign transaction fee of X% on purchases made with your debit card in a currency other than U.S. dollars. Review our fee schedule for additional details."),
    ("Can I request a replacement for my damaged debit card?",
     "Certainly! If your debit card is damaged, contact our customer service, and we'll assist you in getting a new card. A replacement card will be sent to your registered address."),
    ("What is the daily spending limit on my debit card?",
     "The daily spending limit on your debit card is $X,XXX. This limit applies to both point-of-sale transactions and online purchases."),
    ("Tell me about the overdraft protection options for my debit card.",
     "We offer overdraft protection options to help you avoid declined transactions. You can link your debit card to a savings account or apply for overdraft protection services."),
    ("How can I dispute a transaction on my debit card?",
     "If you need to dispute a transaction, contact our customer service with details about the charge. We'll guide you through the dispute resolution process."),
    ("Can I customize the daily spending limit on my debit card?",
     "Yes, you can customize the daily spending limit on your debit card through our online banking services. Visit the 'Card Settings' or 'Spending Limits' section for adjustments."),
    ("Tell me about the benefits of the extended warranty on my debit card.",
     "The extended warranty on your debit card provides additional coverage on qualifying purchases. Check our terms and conditions for specific details."),
    ("How do I set up direct deposit for my debit card?",
     "To set up direct deposit for your debit card, provide your employer with our routing and account numbers. This can typically be done through your employer's HR or payroll department."),
    ("What steps should I take if I forget my debit card PIN?",
     "If you forget your debit card PIN, you can reset it by visiting one of our ATMs or using our online banking services. Choose the 'Forgot PIN' or 'Reset PIN' option."),
    ("Can I request a contactless debit card?",
     "Yes, you can request a contactless debit card through our customer service or by visiting a branch. Contactless cards offer a convenient and secure way to make transactions."),
    ("Tell me about the benefits of the rewards program linked to my debit card.",
     "The rewards program linked to your debit card offers benefits such as cashback, discounts, or loyalty points. Explore our rewards program details for information on available perks."),
    ("How can I track my debit card transactions?",
     "You can track your debit card transactions by logging in to our online banking platform and navigating to the 'Transactions' or 'Account Activity' section. A detailed history will be available."),
    ("What is the process for replacing an expired debit card?",
     "An expired debit card will automatically be replaced, and a new card will be sent to your registered address. You can also request a replacement through our customer service."),
    ("Are there any fees for using ATMs with my debit card?",
     "Fees for using ATMs with your debit card may apply, depending on the ATM's network and location. Review our fee schedule or use our ATM locator tool for fee-free options."),
    ("Can I request a debit card with a custom design?",
     "Yes, you can request a debit card with a custom design through our customer service or online banking services. Choose from our available design options or upload your own image."),
    ("Tell me about the benefits of the purchase protection on my debit card.",
     "The purchase protection on your debit card offers coverage against damage or theft for qualifying purchases. Review our terms and conditions for detailed information."),
    ("How can I set up e-statements for my debit card?",
     "To set up e-statements for your debit card, log in to our online banking platform and navigate to the 'Statements' or 'Account Preferences' section. Choose the electronic statement option."),
    ("What is the process for upgrading my debit card?",
     "To upgrade your debit card, contact our customer service or visit a branch to explore available options. Upgraded cards may offer additional features or benefits."),
    ("Tell me about the benefits of the travel insurance on my debit card.",
     "The travel insurance on your debit card provides coverage for [specific benefits]. Check our website for comprehensive details on the travel insurance policy."),
    ("How can I set a travel notification for my debit card?",
     "Set a travel notification for your debit card through our online banking platform. Visit the 'Travel' or 'Card Settings' section and provide your travel details to avoid any issues during your trip."),
    ("Can I request a debit card statement for the last six months?",
     "Certainly! You can request a debit card statement for the last six months by logging in to our online banking platform or contacting our customer service."),
    ("Tell me about the rewards redemption options for my debit card.",
     "You can redeem rewards from your debit card for cashback, gift cards, or other options. Explore the rewards catalog on our website for a list of available redemption choices."),
    ("What is the process for reporting unauthorized transactions on my debit card?",
     "If you notice unauthorized transactions on your debit card, contact our customer service immediately. We'll guide you through the process of reporting and resolving the issue."),
    ("Are there any restrictions on using my debit card abroad?",
     "Your debit card can be used internationally, but it's advisable to inform us before traveling to prevent any disruptions in card usage."),
    ("Can I request a debit card with a lower daily spending limit?",
     "Yes, you can request a debit card with a lower daily spending limit through our customer service or online banking services. Adjustments are subject to approval."),
    ("Tell me about the benefits of the cashback rewards program on my debit card.",
     "The cashback rewards program on your debit card allows you to earn cashback on qualifying purchases. Check our rewards program details for information on earning and redeeming cashback."),
    ("How do I update my contact information linked to my debit card?",
     "To update your contact information, log in to our online banking platform and navigate to the 'Profile' or 'Account Settings' section. Ensure your details are up to date for communication purposes."),
    ("What steps should I take if my debit card gets declined?",
     "If your debit card gets declined, verify your account balance and check for any transaction limits. If issues persist, contact our customer service for assistance."),
    ("Can I request an additional debit card for a family member?",
     "Yes, you can request an additional debit card for a family member through our customer service or online banking services. Follow the process for requesting an extra card."),
    ("Tell me about the benefits of the zero-liability protection on my debit card.",
     "The zero-liability protection on your debit card ensures you won't be held responsible for unauthorized transactions. Review our terms and conditions for details on this protection."),
    ("How can I apply for a new debit card?",
     "To apply for a new debit card, visit our website and follow the online application process. You'll receive the new card at your registered address upon approval."),
    ("What is the process for disputing an ATM transaction made with my debit card?",
     "If you need to dispute an ATM transaction, contact our customer service with details about the transaction. We'll initiate an investigation and guide you through the resolution process."),
    ("Tell me about the benefits of the overdraft protection service for my debit card.",
     "The overdraft protection service for your debit card helps prevent declined transactions by covering insufficient funds. Explore our terms and conditions for details on this service."),
    ("How can I set up account alerts for my debit card?",
     "Set up account alerts for your debit card through our online banking platform. Visit the 'Alerts' or 'Notifications' section and choose the alerts you wish to receive."),
    ("Can I request a replacement for my expired debit card?",
     "Certainly! If your debit card is expired, contact our customer service, and we'll guide you through the replacement process. A new card will be sent to your registered address."),
    ("What is the process for upgrading my debit card for contactless payments?",
     "To upgrade your debit card for contactless payments, contact our customer service or visit a branch to explore available options. Upgraded cards offer a convenient tap-and-go feature."),
    ("Tell me about the benefits of the contactless payment feature on my debit card.",
     "The contactless payment feature on your debit card provides a convenient and secure way to make quick transactions. Simply tap your card on the contactless reader to complete the payment."),
    ("How can I check the balance on my debit card?",
     "Check the balance on your debit card by logging in to our online banking platform or by using our automated phone service. Both options provide up-to-date information on your account."),
    ("What is the process for reporting a suspicious email or phishing attempt related to my debit card?",
     "If you receive a suspicious email or phishing attempt related to your debit card, forward it to our customer service or report it through our online banking platform. We'll investigate and take appropriate actions."),
    ("Tell me about the benefits of the rewards program linked to my debit card.",
     "The rewards program linked to your debit card offers various benefits, including discounts, cashback, or loyalty points. Explore our rewards program details for information on available perks."),
    ("How can I set up automatic payments for my utility bills using my debit card?",
     "To set up automatic payments for your utility bills, contact your utility providers and provide them with your debit card details. They will guide you through the process of enrollment."),
    ("What steps should I take if my debit card is damaged?",
     "If your debit card is damaged, contact our customer service, and we'll assist you in getting a replacement card. A new card will be sent to your registered address."),
    ("Can I dispute a recurring charge on my debit card?",
     "Yes, you can dispute a recurring charge on your debit card by contacting our customer service. Provide details about the charge, and we'll guide you through the dispute resolution process."),
    ("Tell me about the benefits of the purchase protection on my debit card.",
     "The purchase protection on your debit card offers coverage against damage or theft for qualifying purchases. Review our terms and conditions for specific details."),
    ("How can I set a spending limit for my debit card?",
     "You can set a spending limit for your debit card through our online banking platform. Visit the 'Card Settings' or 'Spending Controls' section and choose the desired limit."),
    ("What is the process for replacing a stolen debit card?",
     "If your debit card is stolen, report it immediately by contacting our 24/7 customer service hotline. We'll assist you in securing your account and issuing a replacement card."),
    ("Tell me about the benefits of the travel insurance on my debit card.",
     "The travel insurance on your debit card provides coverage for [specific benefits]. Check our website for comprehensive details on the travel insurance policy."),
    ("How can I set a travel notification for my debit card?",
     "Set a travel notification for your debit card through our online banking platform. Visit the 'Travel' or 'Card Settings' section and provide your travel details to avoid any issues during your trip."),
    ("Can I request a debit card statement for the last six months?",
     "Certainly! You can request a debit card statement for the last six months by logging in to our online banking platform or contacting our customer service."),

    # Credit Card Queries
    ("Can I redeem my credit card rewards?",
     "Absolutely! You can redeem your credit card rewards through your online account or by contacting our rewards center."),
    ("What is the annual fee for my credit card?",
     "The annual fee for your credit card is $X.XX. Let me know if you have any questions regarding the fee structure."),
    ("How do I apply for a credit limit increase?",
     "You can apply for a credit limit increase by filling out the online form available in your account settings."),
    ("Tell me about the security features of my credit card.",
     "Your credit card comes with advanced security features, including fraud monitoring and zero-liability protection. Rest assured, your transactions are secure."),
    ("What benefits come with my credit card?",
     "Your credit card offers benefits such as travel insurance, purchase protection, and extended warranty. Check our website for a detailed list of benefits."),
    ("How do I report a lost or stolen credit card?",
     "If your card is lost or stolen, please report it immediately by contacting our 24/7 customer service hotline."),
    ("What is the minimum payment due on my credit card?",
     "The minimum payment due on your card for this month is $X.XX. Make sure to pay at least this amount to avoid late fees."),
    ("Tell me about the cashback rewards on my credit card.",
     "Your credit card offers cashback rewards on eligible purchases. Check your rewards summary for details on how to redeem them."),
    ("Is there a foreign transaction fee on my credit card?",
     "Yes, there is a foreign transaction fee of X% on purchases made in a currency other than U.S. dollars."),
    ("How can I update my contact information on my credit card account?",
     "To update your contact information, log in to your account and go to the 'Profile' or 'Account Settings' section."),
    ("Can I transfer my credit card balance to another card?",
     "Yes, you can transfer your credit card balance to another card. Check your account for balance transfer options and terms."),
    ("Tell me about the rewards redemption options.",
     "You can redeem your rewards for cash back, travel, gift cards, and more. Explore the rewards catalog on our website for available options."),
    ("How can I set up account alerts for my credit card?",
     "To set up account alerts, log in to your account and navigate to the 'Alerts' or 'Notifications' section. Choose the alerts you'd like to receive."),
    ("What is the credit limit on my card?",
     "Your current credit limit is $X,XXX. If you're interested in an increase, you can apply through your online account."),
    ("Are there any restrictions on using my credit card abroad?",
     "Your credit card can be used internationally, but please notify us before traveling to prevent any potential issues with foreign transactions."),
    ("Can I customize the due date for my credit card payments?",
     "Yes, you can customize your credit card's due date through your online account settings. Choose a date that works best for you."),
    ("Tell me about the balance transfer fees.",
     "The balance transfer fee is X% of the transferred amount, with a minimum fee of $X.XX. Review the terms on our website for more details."),
    ("How can I check my credit card statement online?",
     "You can view your credit card statement by logging in to your account and navigating to the 'Statements' or 'Account Activity' section."),
    ("What is the maximum cash advance limit on my credit card?",
     "The maximum cash advance limit on your credit card is $X.XX. Keep in mind that cash advances may have additional fees and interest rates."),
    ("Tell me about the rewards expiration policy.",
     "Your credit card rewards expire after [expiration period]. Be sure to check your rewards summary and redeem them before the expiration date."),
    ("Can I request a replacement for my damaged credit card?",
     "Certainly! To request a replacement for your damaged credit card, contact our customer service, and we'll assist you in getting a new card."),
    ("How can I dispute a charge on my credit card?",
     "If you need to dispute a charge, you can initiate the process through your online account or by contacting our customer service. Provide details about the transaction."),
    ("Tell me about the benefits of upgrading my credit card.",
     "Upgrading your credit card may offer additional benefits such as higher rewards, travel perks, and exclusive offers. Check your eligibility and options online."),
    ("What credit bureau do you use for credit card applications?",
     "We typically use [Credit Bureau] for credit card applications. Your credit report from this bureau will be considered during the application process."),
    ("Can I set up a recurring payment for my credit card bills?",
     "Certainly! You can set up a recurring payment for your credit card bills through the 'AutoPay' option in your online account settings."),
    ("Tell me about the reward points earning structure.",
     "You earn [X] points for every [currency] spent on eligible purchases. Check the rewards program details for bonus categories and points redemption options."),
    ("How can I track my credit card spending?",
     "You can track your credit card spending by reviewing your online account statements, transaction history, and utilizing budgeting tools available on our website."),
    ("What steps should I take if I suspect fraudulent activity on my credit card?",
     "If you suspect fraudulent activity, contact our fraud department immediately. We'll guide you through the necessary steps to secure your account."),
    ("Can you block my credit card?",
     "Certainly, I'll go ahead and block your credit card. It will be effective immediately."),
    ("What are the new offers on credit cards?",
     "Here are the latest credit card offers: [List of offers]. Take advantage of these exclusive deals!"),
    ("What types of credit cards do you offer?",
     "We offer a range of credit cards, including rewards, cashback, and travel cards. Which category are you interested in?"),
    ("Give me my credit card bill for this month.",
     "Sure, your credit card bill for this month is $X.XX. Would you like a detailed breakdown of the transactions?"),
    ("How can I increase my credit limit?",
     "To increase your credit limit, you can request it through your online account or contact our customer service for assistance."),
    ("What is the interest rate on my credit card?",
     "The current interest rate on your credit card is X%. If you have any specific concerns, feel free to ask."),
    ("Can I set up automatic payments for my credit card bills?",
     "Absolutely, you can easily set up automatic payments through your online account. Let me guide you through the process if needed."),
    ("What is the grace period for my credit card payments?",
     "The grace period for your credit card payments is X days. During this period, no interest will be charged on your purchases."),
    ("How can I dispute a transaction on my credit card?",
     "If you need to dispute a transaction, please contact our customer service with details, and we'll assist you in resolving the issue."),
    ("Can I upgrade my credit card?",
     "Certainly! To explore upgrade options, log in to your account and check the 'Upgrade' section for available credit card upgrades."),
    ("How can I check my credit card balance?",
     "You can check your credit card balance by logging in to your online account or by contacting our automated phone service."),
    ("Tell me about the rewards program tiers.",
     "Our rewards program offers different tiers with varying benefits. Check the rewards program details on our website for information on each tier."),
    ("What should I do if my credit card is about to expire?",
     "If your credit card is about to expire, a new card will be automatically sent to your registered address. You can also request a replacement through customer service."),
    ("Are there any introductory APR offers on my credit card?",
     "Yes, there may be introductory APR offers. Check your account or our website for current promotions and their terms."),
    ("Can I use my credit card for online purchases?",
     "Absolutely! You can use your credit card for online purchases. Ensure that your billing information is up to date for a smooth transaction."),
    ("Tell me about the credit card application process.",
     "To apply for a credit card, visit our website and fill out the online application. You'll receive a response regarding your application status within [timeframe]."),
    ("What rewards are available for redemption this month?",
     "This month's available rewards include [list of rewards]. Check your rewards summary for detailed information on how to redeem them."),
    ("How can I enroll in paperless statements?",
     "To enroll in paperless statements, log in to your account, go to the 'Statements' section, and choose the paperless option. You'll receive statements via email."),
    ("Tell me about the credit limit decrease process.",
     "If you wish to decrease your credit limit, contact our customer service. Keep in mind that this may impact your credit utilization ratio."),
    ("What steps should I take if I find an error on my credit card statement?",
     "If you find an error on your statement, contact our customer service immediately. We'll investigate the issue and guide you through the resolution process."),
    ("Can I request a credit card PIN change?",
     "Yes, you can request a credit card PIN change through your online account or by contacting our customer service. Follow the instructions provided for security purposes."),
    ("Tell me about the benefits of the travel insurance offered on my credit card.",
     "The travel insurance on your credit card provides coverage for [specific benefits]. Review the travel insurance details on our website for comprehensive information."),
    ("How do I set up a travel notification for my credit card?",
     "To set up a travel notification, log in to your account and go to the 'Travel' or 'Account Settings' section. Provide your travel details to prevent any issues with transactions abroad."),
    ("Can I request a credit card statement for the last six months?",
     "Certainly! You can request a credit card statement for the last six months through your online account or by contacting our customer service."),
    ("Tell me about the credit card rewards expiration policy.",
     "Credit card rewards typically expire after [expiration period]. Make sure to review your rewards summary and redeem them before they expire."),
    ("What is the difference between a credit card and a debit card?",
     "A credit card allows you to borrow funds up to a set limit, while a debit card is linked to your bank account, and purchases are deducted directly from your account."),
    ("Can I have an additional credit card for my spouse?",
     "Yes, you can request an additional credit card for your spouse. Log in to your account and navigate to the 'Request Additional Card' section."),
    ("Tell me about the benefits of the extended warranty on my credit card.",
     "The extended warranty on your credit card offers additional coverage on eligible purchases. Check the terms and conditions on our website for specific details."),
    ("How often is my credit score updated in my account?",
     "Your credit score is typically updated monthly. You can view the latest score by logging in to your account and checking the 'Credit Score' or 'Account Overview' section."),
    ("Can I set spending limits on my credit card?",
     "Yes, you can set spending limits on your credit card for additional control. Log in to your account and navigate to the 'Card Settings' or 'Spending Controls' section."),
    ("How can I apply for a new credit card?",
     "To apply for a new credit card, visit our website and click on the 'Apply Now' button for the desired credit card. Follow the online application instructions."),
    ("What documents do I need to apply for a credit card?",
     "To complete your credit card application, you'll typically need proof of identity (e.g., driver's license), proof of income (e.g., pay stubs), and other relevant financial documents. Check the application page for specific requirements."),
    ("How long does the credit card application process take?",
     "The credit card application process usually takes [average time]. You'll receive a notification about your application status within [timeframe]. Feel free to check your application status online."),
    ("Tell me about the credit card activation process.",
     "Upon receiving your new credit card, you can activate it by visiting our website or calling the activation number provided on the card. Follow the prompts to complete the activation."),
    ("What is the credit limit on the new credit card I applied for?",
     "Your approved credit limit will be mentioned in the acceptance letter or email. Once you receive the card, you can also check the credit limit on your online account."),
    ("Tell me about the benefits of the credit card I just applied for.",
     "The credit card you applied for offers benefits such as [list of benefits]. Explore the credit card details on our website or in the provided materials for comprehensive information."),
    ("How can I track the status of my credit card application?",
     "To track the status of your credit card application, log in to your online account and navigate to the 'Application Status' or 'Check Application' section. You can also contact our customer service for updates."),
    # Debit card
    ("What is the balance on my debit card?",
     "Your current debit card balance is $X.XX."),
    ("Can I see the recent transactions on my debit card?",
     "Certainly! Here is the list of your recent debit card transactions: ..."),
    ("How do I activate my new debit card?",
     "To activate your new debit card, you can use it at any ATM with your PIN or call our customer service hotline."),
    ("Tell me about the benefits of my debit card.",
     "Your debit card provides convenient access to your funds, allows cash withdrawals, and can be used for online and in-store purchases."),
    ("Is there a daily withdrawal limit on my debit card?",
     "Yes, there is a daily withdrawal limit of $X on your debit card. Contact customer service if you need to adjust this limit."),
    ("Can I transfer money between my accounts using my debit card?",
     "Yes, you can transfer money between your linked accounts through online banking or our mobile app."),
    ("How do I report a lost or stolen debit card?",
     "If your debit card is lost or stolen, please report it immediately by contacting our 24/7 customer service hotline."),
    ("What is the process for disputing a transaction on my debit card?",
     "If you need to dispute a transaction, contact our customer service with details, and we'll guide you through the dispute process."),
    ("Tell me about the fees associated with using my debit card abroad.",
     "There may be foreign transaction fees and ATM withdrawal fees when using your debit card abroad. Check our fee schedule for details."),
    ("Can I set up alerts for transactions on my debit card?",
     "Certainly! To set up transaction alerts for your debit card, log in to your online banking and navigate to the 'Alerts' or 'Notifications' section."),
    ("What is the daily spending limit on my prepaid card?",
     "Your prepaid card has a daily spending limit of $X.XX. If needed, you can adjust this limit through your online account."),
    ("How can I reload funds onto my prepaid card?",
     "You can reload funds onto your prepaid card through various methods, including direct deposit, bank transfers, or at authorized reload locations."),
    ("Tell me about the expiration policy for my prepaid card.",
     "Your prepaid card is valid until [expiration date]. Be sure to check your card for the expiration date and renew it if necessary."),
    ("Is there a fee for checking the balance on my prepaid card?",
     "Checking your prepaid card balance online or through our mobile app is usually free. Refer to our fee schedule for any associated charges."),
    ("Can I use my prepaid card for online purchases?",
     "Absolutely! Your prepaid card can be used for online purchases wherever debit or credit cards are accepted."),
    ("How do I dispute a charge on my prepaid card?",
     "If you need to dispute a charge, contact our customer service with details, and we'll assist you in resolving the issue."),
    ("Tell me about the benefits of my prepaid card.",
     "Your prepaid card offers benefits such as budgeting control, no overdraft fees, and the ability to make secure transactions without a bank account."),
    ("What is the process for replacing a damaged prepaid card?",
     "To request a replacement for your damaged prepaid card, contact our customer service, and we'll assist you in getting a new card."),
    ("Can I use my forex card in multiple currencies?",
     "Yes, your forex card is designed for use in multiple currencies. It automatically converts the transaction amount to the local currency."),
    ("How can I check the balance on my forex card?",
     "You can check your forex card balance by logging in to your online account or by contacting our customer service."),
    ("Tell me about the fees associated with using my forex card abroad.",
     "There may be foreign transaction fees and ATM withdrawal fees when using your forex card abroad. Refer to our fee schedule for details."),
    ("Is there a limit on daily cash withdrawals with my forex card?",
     "Yes, there is a daily cash withdrawal limit on your forex card. Check your card documentation or contact customer service for specific details."),
    ("Can I reload additional funds onto my forex card while abroad?",
     "Yes, you can reload additional funds onto your forex card through your online account, but availability may vary by location."),
    ("What steps should I take if my forex card is lost or stolen?",
     "If your forex card is lost or stolen, please report it immediately by contacting our 24/7 customer service hotline."),
    ("Tell me about the benefits of using a forex card for international travel.",
     "Using a forex card for international travel provides advantages such as competitive exchange rates, security, and the convenience of carrying multiple currencies."),
    ("Can I set up travel alerts for my forex card?",
     "Certainly! To set up travel alerts for your forex card, log in to your online account and navigate to the 'Travel' or 'Account Settings' section."),
    ("What is the validity period of my forex card?",
     "Your forex card is typically valid for [validity period]. Be sure to check the expiration date on your card and renew it if necessary."),
    ("Tell me about the process of closing my forex card account.",
     "To close your forex card account, contact our customer service, and we'll assist you in completing the necessary steps."),
    ("Can I apply for a new debit card online?",
    "Yes, you can apply for a new debit card online by logging in to your account or visiting our website's 'Apply for a Debit Card' section."),
    ("What documents do I need to apply for a debit card?",
     "To complete your debit card application, you'll typically need proof of identity (e.g., driver's license or passport) and proof of address (e.g., utility bill or bank statement). Check the application page for specific requirements."),
    ("How long does the debit card application process take?",
     "The debit card application process usually takes [average time]. You'll receive your new debit card by mail within [timeframe]. Feel free to check your application status online."),
    ("Tell me about the activation process for my new debit card.",
     "Upon receiving your new debit card, you can activate it by using it at any ATM with your PIN or by calling our customer service hotline."),
    ("What is the daily spending limit on my debit card?",
     "Your debit card has a daily spending limit of $X.XX. Contact customer service if you need to adjust this limit."),
    ("Tell me about the benefits of the debit card I just applied for.",
     "The debit card you applied for offers benefits such as [list of benefits]. Explore the debit card details on our website or in the provided materials for comprehensive information."),
    ("How can I track the status of my debit card application?",
     "To track the status of your debit card application, log in to your online account and navigate to the 'Application Status' or 'Check Application' section. You can also contact our customer service for updates."),
    ("Can I request a replacement for my damaged debit card?",
     "Certainly! To request a replacement for your damaged debit card, contact our customer service, and we'll assist you in getting a new card."),
    ("Tell me about the fees associated with my debit card.",
     "Your debit card may have fees such as ATM withdrawal fees or overdraft fees. Review our fee schedule for details on debit card-related charges."),
    ("What is the process for upgrading my debit card?",
     "To explore debit card upgrade options, log in to your account and check the 'Upgrade' or 'Card Services' section for available debit card upgrades."),
    ("Can I customize the PIN for my debit card?",
     "Yes, you can customize the PIN for your debit card. Visit an ATM or contact our customer service to set up a new PIN."),
    ("Transfer 100 pounds to Aarya for groceries",
     "Transfer initiated to specified account."),
    ("Send 1000 pounds to Rishi",
     "Initiating a transfer to specified account."),
    ("Send forex transfer of 1000 GBP to INR to Pappa from my GBP account",
     "'Forex transfer initiated."),
    ("International transfer",
     "Initiating an international transfer."),
    ("Transfer 500 dollars to Emily for rent",
     "Transfer initiated..."),
    ("Send 200 euros to Maria",
     "Sending money...."),
    ("Can I apply for a credit card with joint account holders?",
     "Yes, you can apply for a credit card with joint account holders. During the application process, choose the option for a joint account and provide the required information for the additional cardholder."),
    ("What types of loans do you offer?",
     "We offer a variety of loans, including home loans, car loans, personal loans, and property loans. Each type of loan has its own set of features and eligibility criteria."),
     # Savings Account
    ("How to open a savings account?",
     "To open a savings account, visit our nearest branch with a valid ID proof, address proof, and a passport-sized photograph. You can also apply online through our website."),
    ("What documents are required for a savings account?",
     "To complete your savings account application, you'll typically need proof of identity (e.g., driver's license or passport) and proof of address (e.g., utility bill or bank statement). Check the application page for specific requirements."),
    ("How long does the savings account opening process take?",
     "The savings account opening process usually takes [average time]. You'll receive your account details and welcome kit by mail within [timeframe]. Feel free to check your application status online."),
    ("Tell me about the process of closing my savings account.",
     "To close your savings account, visit the branch where the account was opened, fill out the account closure form, and settle any outstanding transactions."),
    ("How to check savings account balance?",
     "Check your savings account balance through online banking, mobile banking apps, ATMs, or by visiting the nearest branch."),
    ("Can I block my savings account?",
     "If you need to block your savings account due to security concerns, contact our customer service immediately to report any unauthorized transactions."),
    ("How to check eligibility for a savings account?",
     "You can check your eligibility for a savings account by visiting our website or contacting our customer service. Eligibility criteria may vary."),
    ("Tell me about the benefits of the savings account I just opened.",
     "The savings account you opened offers benefits such as [list of benefits]. Explore the account details on our website or in the provided materials for comprehensive information."),
    ("Can I customize the PIN for my savings account?",
     "Yes, you can customize the PIN for your savings account. Visit an ATM or contact our customer service to set up a new PIN."),


    # Checking Account
    ("How to open a checking account?",
     "Opening a checking account is easy! Visit our branch with a valid ID proof and address proof. You can also apply online through our website."),
    ("What documents are required for a checking account?",
     "To open a checking account, you'll typically need proof of identity (e.g., driver's license or passport) and proof of address (e.g., utility bill or bank statement). Additional documents may be required based on the account type."),
    ("How long does the checking account opening process take?",
     "The checking account opening process usually takes [average time]. You'll receive your account details and checks by mail within [timeframe]. Check your application status online."),
    ("Tell me about the process of closing my checking account.",
     "To close your checking account, visit the branch where the account was opened, fill out the account closure form, and ensure all outstanding checks and transactions are settled."),
    ("How to check checking account balance?",
     "Check your checking account balance through online banking, mobile banking apps, ATMs, or by visiting the nearest branch."),
    ("Can I block my checking account?",
     "If you need to block your checking account due to security concerns, contact our customer service immediately to report any unauthorized transactions."),
    ("How to check eligibility for a checking account?",
     "You can check your eligibility for a checking account by visiting our website or contacting our customer service. Eligibility criteria may vary based on the type of checking account."),
    ("Tell me about the benefits of the checking account I just opened.",
     "The checking account you opened offers benefits such as [list of benefits]. Explore the account details on our website or in the provided materials for comprehensive information."),
    ("Can I customize the PIN for my checking account?",
     "Yes, you can customize the PIN for your checking account. Visit an ATM or contact our customer service to set up a new PIN."),

    # Mobile Banking
    ("How to register for mobile banking?",
     "Register for mobile banking by downloading our app and following the on-screen instructions. You'll need your account details and a valid phone number for verification."),
    ("How to reset mobile banking password?",
     "Reset your mobile banking password by selecting the 'Forgot Password' option on the login screen. Follow the prompts to verify your identity and set a new password."),
    ("How to check eligibility for mobile banking?",
     "To check your eligibility for mobile banking, ensure your account is active and contact our customer service if you encounter any issues."),
    ("Can I block my mobile banking access?",
     "If you suspect unauthorized access or need to block your mobile banking temporarily, contact our customer service immediately to initiate security measures."),
    ("Tell me about the features of your mobile banking app.",
     "Our mobile banking app offers features such as account balance checks, fund transfers, bill payments, and mobile check deposits. Explore the app for a seamless banking experience."),

    # Closing Accounts
    ("How to close my bank account?",
     "To close your bank account, visit the branch where the account is held, fill out the account closure form, and ensure all outstanding transactions are settled."),
    ("What is the process of closing a joint account?",
     "Closing a joint account requires both account holders to visit the branch together, fill out the joint account closure form, and settle any outstanding transactions."),
    ("How to close a fixed deposit account?",
     "To close your fixed deposit account, visit the branch, fill out the closure form, and collect the maturity amount. Closing it prematurely may incur penalties."),
    ("Can I reopen a closed bank account?",
     "Generally, reopening a closed bank account is not possible. You may need to open a new account if needed."),
    ("What happens to my credit card if I close my bank account?",
     "Closing your bank account doesn't automatically close your credit card. You can still use your credit card as usual, and any outstanding dues need to be settled separately."),

    # Miscellaneous
    ("How to apply for a credit card?",
     "Apply for a credit card online through our website or by visiting the nearest branch. Fill out the application form, provide necessary documents, and undergo the credit assessment process."),
    ("What documents are required for a credit card application?",
     "Credit card applications typically require proof of identity, address proof, income documents, and sometimes credit history reports."),
    ("How to report a lost or stolen credit card?",
     "If your credit card is lost or stolen, please report it immediately by contacting our 24/7 customer service hotline."),
    ("Can I request a replacement for my damaged debit card?",
     "Certainly! To request a replacement for your damaged debit card, contact our customer service, and we'll assist you in getting a new card."),
    ("Tell me about the fees associated with my debit card.",
     "Your debit card may have fees such as ATM withdrawal fees or overdraft fees. Review our fee schedule for details on debit card-related charges."),
    ("What is the process for upgrading my debit card?",
     "To explore debit card upgrade options, log in to your account and check the 'Upgrade' or 'Card Services' section for available debit card upgrades."),
    ("Can I customize the PIN for my debit card?",
     "Yes, you can customize the PIN for your debit card. Visit an ATM or contact our customer service to set up a new PIN."),

    # Current Account
    ("How to open a current account?",
     "Opening a current account is easy! Visit our branch with a valid ID proof, address proof, and business registration documents. You can also apply online through our website."),
    ("What documents are required for a current account?",
     "To open a current account, you'll typically need proof of identity (e.g., driver's license or passport), proof of address (e.g., utility bill or bank statement), and business registration documents. Additional documents may be required based on the business type."),
    ("How long does the current account opening process take?",
     "The current account opening process usually takes [average time]. You'll receive your account details and business checks by mail within [timeframe]. Check your application status online."),
    ("Tell me about the process of closing my current account.",
     "To close your current account, visit the branch where the account was opened, fill out the account closure form, and ensure all outstanding transactions are settled."),
    ("How to check current account balance?",
     "Check your current account balance through online banking, mobile banking apps, ATMs, or by visiting the nearest branch."),
    ("Can I block my current account?",
     "If you need to block your current account due to security concerns, contact our customer service immediately to report any unauthorized transactions."),
    ("How to check eligibility for a current account?",
     "You can check your eligibility for a current account by visiting our website or contacting our customer service. Eligibility criteria may vary based on the type of current account and business type."),
    ("Tell me about the benefits of the current account I just opened.",
     "The current account you opened offers benefits such as [list of benefits]. Explore the account details on our website or in the provided materials for comprehensive information."),
    ("Can I customize the PIN for my current account?",
     "Yes, you can customize the PIN for your current account. Visit an ATM or contact our customer service to set up a new PIN."),

    # Cheque
    ("How to request a cheque book?",
     "You can request a cheque book through online banking, mobile banking apps, ATMs, or by visiting the nearest branch. Alternatively, contact our customer service for assistance."),
    ("What is the process of stopping a cheque?",
     "To stop a cheque, log in to your online banking account, visit the nearest branch, or contact our customer service. Provide the cheque details and reason for stopping."),
    ("How long does it take to receive a new cheque book?",
     "You'll typically receive a new cheque book by mail within [timeframe] after placing a request. Check your request status online for updates."),
    ("Tell me about the fees associated with a bounced cheque.",
     "Fees for a bounced cheque may vary. Check our fee schedule or contact our customer service for details on cheque-related charges."),
    ("Can I customize the design of my cheque book?",
     "Customizing the design of your cheque book may be available. Check our website or contact our customer service for information on personalized cheque book options."),

    # Overdraft
    ("How to apply for an overdraft facility?",
     "Apply for an overdraft facility by visiting our branch, contacting our customer service, or applying online through our website. Provide necessary financial documents and details."),
    ("What documents are required for an overdraft application?",
     "Overdraft applications typically require proof of income, financial statements, and details of the purpose for which the overdraft is needed."),
    ("How is the interest calculated on an overdraft?",
     "Interest on overdraft is typically calculated on the utilized amount and charged monthly. The interest rate may vary based on market conditions."),
    ("Can I increase the overdraft limit on my account?",
     "You can apply for an increase in the overdraft limit by contacting our customer service or using the online request form available in your account."),
    ("Tell me about the benefits of an overdraft facility.",
     "The overdraft facility provides flexibility in managing short-term financial needs. Explore our website or contact our customer service for details on overdraft benefits."),

    # Certificate of Deposit
    ("How to open a Certificate of Deposit (CD) account?",
     "To open a Certificate of Deposit (CD) account, visit the branch or use our online banking platform. Provide your ID proof, address proof, and the amount you wish to deposit. Choose the tenure for your CD."),
    ("What documents are required for a Certificate of Deposit account?",
     "Opening a CD account typically requires ID proof, address proof, and the deposit amount. The CD tenure and interest payout options will be selected during account opening."),
    ("Can I withdraw funds from a Certificate of Deposit before maturity?",
     "Withdrawing funds from a CD before maturity may incur penalties. Contact the branch or customer service for information on early withdrawal penalties."),
    ("How to renew a Certificate of Deposit?",
     "Upon maturity, your CD will be automatically renewed for the same tenure. If you wish to make changes, contact the branch or use online banking to explore renewal options."),
    ("Tell me about the interest rates for Certificate of Deposit.",
     "Interest rates for CDs vary based on the tenure and prevailing market conditions. Check our website or contact our customer service for the current rates."),

    # Joint Account
    ("How to open a joint account?",
     "Opening a joint account is simple! Both account holders need to visit the branch with valid ID proofs and address proofs. You can also apply online through our website."),
    ("What documents are required for a joint account?",
     "To open a joint account, each account holder typically needs proof of identity (e.g., driver's license or passport) and proof of address (e.g., utility bill or bank statement). Additional documents may be required."),
    ("How long does the joint account opening process take?",
     "The joint account opening process usually takes [average time]. You'll both receive your account details and welcome kits by mail within [timeframe]. Check your application status online."),
    ("Tell me about the process of closing my joint account.",
     "To close your joint account, both account holders need to visit the branch together, fill out the joint account closure form, and settle any outstanding transactions."),
    ("How to check joint account balance?",
     "Check your joint account balance through online banking, mobile banking apps, ATMs, or by visiting the nearest branch."),
    ("Can I block my joint account?",
     "If you need to block your joint account due to security concerns, contact our customer service immediately to report any unauthorized transactions."),
    ("How to check eligibility for a joint account?",
     "You can check your eligibility for a joint account by visiting our website or contacting our customer service. Eligibility criteria may vary based on the type of joint account."),
    ("Tell me about the benefits of the joint account I just opened.",
     "The joint account you opened offers benefits such as [list of benefits]. Explore the account details on our website or in the provided materials for comprehensive information."),
    ("Can I customize the PIN for my joint account?",
     "Yes, you can customize the PIN for your joint account. Visit an ATM or contact our customer service to set up a new PIN."),


    # Fixed Deposit Account
    ("How to open a fixed deposit account?",
     "To open a fixed deposit account, visit the branch or use our online banking platform. Provide your ID proof, address proof, and the amount you wish to deposit. Choose the tenure and interest payout option."),
    ("What documents are required for a fixed deposit account?",
     "Opening a fixed deposit account typically requires ID proof, address proof, and the deposit amount. Additional documents may be needed for certain categories of customers."),
    ("How to close a fixed deposit account?",
     "To close your fixed deposit account, visit the branch, fill out the closure form, and collect the maturity amount. Closing it prematurely may incur penalties."),
    ("How to check fixed deposit account details?",
     "Check your fixed deposit details, including maturity date and interest earned, through online banking or by contacting our customer service."),
    ("Can I customize the tenure for my fixed deposit?",
     "The tenure for a fixed deposit is usually fixed at the time of opening. To make changes, you may need to prematurely close the existing deposit and open a new one with the desired tenure."),
    ("Tell me about the interest rates for fixed deposits.",
     "Interest rates for fixed deposits vary based on the tenure and prevailing market conditions. Check our website or contact our customer service for the current rates."),

    # Recurring Deposit Account
    ("How to open a Recurring Deposit (RD) account?",
     "To open a Recurring Deposit (RD) account, visit the branch or use our online banking platform. Provide your ID proof, address proof, and the initial deposit amount. Choose the tenure for your RD."),
    ("What documents are required for a Recurring Deposit account?",
     "Opening an RD account typically requires ID proof, address proof, and the initial deposit amount. The RD tenure and installment amount will be selected during account opening."),
    ("Can I increase the installment amount in a Recurring Deposit?",
     "Increasing the installment amount in an RD may be possible. Contact the branch or customer service to explore options and procedures."),
    ("How to close a Recurring Deposit account?",
     "To close your RD account, visit the branch, fill out the closure form, and collect the maturity amount. Closing it prematurely may incur penalties."),
    ("How to check Recurring Deposit account details?",
     "Check your RD details, including upcoming installments and maturity date, through online banking or by contacting our customer service."),
    ("Tell me about the interest rates for Recurring Deposit.",
     "Interest rates for RDs vary based on the tenure and prevailing market conditions. Check our website or contact our customer service for the current rates."),

    # Individual Retirement Account (IRA)
    ("How to contribute to my IRA account?",
     "Contribute to your IRA account through online banking, mobile apps, or by setting up automatic transfers. You can also make one-time contributions by visiting the branch."),
    ("What is the maximum annual contribution limit for an IRA?",
     "The maximum annual contribution limit for an IRA may vary based on your age and the type of IRA. Check our website or contact our customer service for the current limits."),
    ("Can I have multiple IRAs?",
    "Yes, you can have multiple IRAs. There are different types of IRAs, such as Traditional IRA, Roth IRA, and SEP IRA. Consult with a financial advisor to understand the best strategy for your retirement savings."),
    ("Tell me about the tax benefits of contributing to an IRA.",
    "Contributions to a Traditional IRA may be tax-deductible, and earnings grow tax-deferred. Roth IRA contributions are made with after-tax dollars, and qualified withdrawals are tax-free. Consult a tax advisor for personalized advice."),
    ("How to change the investment allocation in my IRA?",
    "You can change the investment allocation in your IRA through online banking, mobile apps, or by contacting our customer service. Review your investment strategy periodically to align with your retirement goals."),
    ("Can I transfer my IRA from another financial institution?",
    "Yes, you can transfer your IRA from another financial institution to our bank. Contact our customer service for assistance and guidance on the IRA transfer process."),
    ("Tell me about the penalties for early withdrawal from an IRA.",
    "Early withdrawal from an IRA before the age of 59½ may result in a 10% penalty in addition to income taxes. Certain exceptions apply, such as first-time home purchase or qualified education expenses."),
    ("What happens to my IRA in case of my demise?",
    "In the event of your demise, the beneficiary designated in your IRA will inherit the account. It's crucial to update your beneficiary information regularly to ensure your wishes are carried out."),
    ("How to convert my Traditional IRA to a Roth IRA?",
    "You can convert your Traditional IRA to a Roth IRA by contacting our customer service or using the online conversion tool. Be aware of the tax implications and consult with a financial advisor."),
    ("Tell me about the perks of having an IRA with senior citizen benefits.",
    "An IRA with senior citizen benefits may offer additional perks such as preferential interest rates, waived fees, and personalized financial advice. Explore our senior citizen banking offerings for comprehensive information."),

    # Senior Citizen Bank Account
    ("How to open a Senior Citizen Bank Account?",
    "Opening a Senior Citizen Bank Account is easy! Visit our nearest branch with a valid ID proof, address proof, and age verification document. You can also apply online through our website."),
    ("What documents are required for a Senior Citizen Bank Account?",
    "To open a Senior Citizen Bank Account, you'll typically need proof of identity (e.g., driver's license or passport), proof of address (e.g., utility bill or bank statement), and a document verifying your age (e.g., birth certificate or senior citizen card)."),
    ("How long does the Senior Citizen Bank Account opening process take?",
    "The Senior Citizen Bank Account opening process usually takes [average time]. You'll receive your account details and welcome kit by mail within [timeframe]. Feel free to check your application status online."),
    ("Tell me about the perks of having a Senior Citizen Bank Account.",
    "A Senior Citizen Bank Account may offer perks such as higher interest rates on deposits, special discounts on banking services, and priority customer service. Explore our senior citizen banking offerings for comprehensive information."),
    ("Can I have joint ownership of a Senior Citizen Bank Account?",
    "Yes, you can have joint ownership of a Senior Citizen Bank Account. Both account holders need to meet the senior citizen criteria. Visit the branch or apply online to open a joint senior citizen account."),
    ("How to check eligibility for a Senior Citizen Bank Account?",
    "You can check your eligibility for a Senior Citizen Bank Account by visiting our website or contacting our customer service. Eligibility criteria may include age requirements and additional documentation."),
    ("How to close a Senior Citizen Bank Account?",
    "To close your Senior Citizen Bank Account, visit the branch where the account was opened, fill out the account closure form, and settle any outstanding transactions. Ensure you provide the necessary identification."),
    ("How to block transactions on my Senior Citizen Bank Account?",
    "If you need to block transactions on your Senior Citizen Bank Account due to security concerns, contact our customer service immediately to report any unauthorized transactions."),
    ("How to request a replacement for a lost Senior Citizen Bank Account card?",
    "To request a replacement for a lost Senior Citizen Bank Account card, contact our customer service, and we'll assist you in getting a new card."),
    ("Tell me about the fees associated with a Senior Citizen Bank Account.",
    "Your Senior Citizen Bank Account may have fees such as ATM withdrawal fees or overdraft fees. Review our fee schedule for details on senior citizen account-related charges."),
    ("Is there a special interest rate for senior citizens on savings accounts?",
    "Yes, senior citizens enjoy a special higher interest rate on savings accounts. Check our current rates to see the benefits offered for your savings."),
    ("Do senior citizens get additional discounts on banking services?",
    "Absolutely! Senior citizens are eligible for additional discounts on various banking services, including transaction fees, locker rentals, and more. Explore our senior citizen banking benefits."),
    ("Are there exclusive perks for senior citizens on fixed deposits?",
    "Yes, senior citizens receive exclusive perks on fixed deposits, such as higher interest rates. Check our current rates and terms for senior citizen fixed deposit benefits."),
    ("Tell me about the special offers on loans for senior citizens.",
    "Senior citizens may enjoy special offers on loans, including lower interest rates and flexible repayment options. Contact our customer service or visit the branch for personalized loan offers."),
    ("Are there any fee waivers for senior citizens on debit cards?",
    "Certainly! Senior citizens often benefit from fee waivers on debit card transactions, annual fees, and other related charges. Review our fee schedule for details on senior citizen account privileges."),
    ("Do senior citizens receive priority customer service?",
    "Absolutely! Senior citizens receive priority customer service with dedicated helplines, faster response times, and personalized assistance. Experience banking with the care you deserve."),
    ("Are there special events or workshops for senior citizens account holders?",
    "Yes, we regularly organize special events, workshops, and social gatherings exclusively for our senior citizen account holders. Stay updated on our events calendar for exciting activities."),
    ("Tell me about the healthcare benefits for senior citizens with your bank.",
    "Senior citizens banking with us may enjoy healthcare benefits, including discounts on health insurance premiums, wellness programs, and partnerships with healthcare providers. Explore our health and wellness offerings."),
    ("Are there any travel benefits for senior citizens with your bank?",
    "Yes, senior citizens often enjoy travel benefits, including discounts on travel insurance, preferential forex rates, and exclusive travel packages. Check our travel-related offers for senior citizens."),
    ("Tell me about the exclusive lifestyle privileges for senior citizens.",
    "Our senior citizen account holders receive exclusive lifestyle privileges, including discounts on shopping, dining, entertainment, and more. Explore our partner network for exciting offers."),

    # Loans
    ("Tell me about the interest rates on home loans.",
     "The interest rates on our home loans vary depending on factors such as the loan amount, tenure, and your credit score. You can check our website or contact our loan department for specific details."),
    ("Can I apply for a car loan with your institution?",
     "Certainly! We provide car loans with competitive interest rates. You can apply for a car loan through our online application process or visit one of our branches for assistance."),
    ("What is the maximum loan amount for a personal loan?",
     "The maximum loan amount for a personal loan depends on your income, credit history, and other factors. You can check our website or contact our loan specialists to discuss your specific requirements."),
    ("Tell me about the eligibility criteria for property loans.",
     "To be eligible for a property loan, you need to meet certain criteria related to your income, creditworthiness, and the property's value. Contact our loan department for a detailed discussion on eligibility."),
    ("How long does the loan approval process take?",
     "The loan approval process duration varies based on the type of loan and the completeness of your application. Generally, it takes [average time]. You can track your application status through our online portal."),
    ("What is the minimum down payment required for a home loan?",
     "The minimum down payment for a home loan is typically a percentage of the property's value. The exact amount depends on various factors. You can find specific details on our website or by contacting our loan team."),
    ("Tell me about the repayment options for car loans.",
     "We offer flexible repayment options for car loans, including monthly installments and customized plans. You can choose the option that best suits your financial situation. Contact our loan department for further details."),
    ("Are there any prepayment penalties on personal loans?",
     "Our personal loans come with the flexibility to prepay without any penalties. You can make partial or full prepayments to reduce the loan tenure. Check your loan agreement for specific terms."),
    ("What documents are required for a property loan application?",
     "To apply for a property loan, you generally need documents such as income proof, property documents, identity proof, and more. Visit our website or contact our loan specialists for a complete list of required documents."),
    ("Tell me about the interest rates for loans on commercial properties.",
     "Interest rates for loans on commercial properties are determined based on various factors such as the loan amount, tenure, and the property's financial viability. Contact our loan department for specific details."),
    ("Can I get a loan for home renovations?",
     "Yes, we offer home renovation loans with attractive interest rates. You can apply for a home renovation loan to fund improvements, repairs, or expansions. Check our website or contact our loan team for more information."),
    ("What is the maximum loan tenure for personal loans?",
     "The maximum loan tenure for personal loans depends on the loan amount and your repayment capacity. Generally, personal loans have tenures ranging from [minimum tenure] to [maximum tenure]. Contact our loan department for specific details."),
    ("Tell me about the types of collateral accepted for secured loans.",
     "We accept various types of collateral for secured loans, including real estate, vehicles, and other valuable assets. The type of collateral accepted may vary based on the loan type. Contact our loan specialists for detailed information."),
    ("How can I check the status of my loan application?",
     "You can check the status of your loan application through our online portal. Additionally, our loan department is available to provide updates and assistance. Feel free to contact us for any queries regarding your application."),
    ("Can I apply for a loan online?",
     "Yes, you can conveniently apply for a loan online through our secure application portal. Simply visit our website, fill out the online application form, and submit the required documents. Our loan team will guide you through the process."),
    ("Tell me about the benefits of taking a loan with your institution.",
     "Taking a loan with us comes with benefits such as competitive interest rates, flexible repayment options, and personalized customer service. Explore our loan offerings on our website or contact our loan specialists for more details."),
    ("Are there any special offers or discounts on home loans?",
     "We periodically offer special promotions and discounts on home loans. Check our website or contact our loan department to inquire about any ongoing offers or exclusive deals for homebuyers."),
    ("How is the interest calculated on car loans?",
     "Interest on car loans is typically calculated using a fixed or floating rate, depending on the loan agreement. The interest is applied to the outstanding balance. You can find detailed information in your loan agreement or contact our loan team."),
    ("Tell me about the loan application processing fees.",
     "Loan application processing fees vary depending on the type of loan and the amount. You can find information on processing fees in our loan documentation. Feel free to contact our loan specialists for a breakdown of applicable fees."),
    ("Can I get a loan for educational purposes?",
     "Yes, we offer education loans to support your educational expenses. Whether it's tuition fees, accommodation, or other educational costs, our education loans come with competitive terms. Visit our website or contact our loan department for more details."),
    ("What is the maximum loan-to-value ratio for property loans?",
     "The maximum loan-to-value ratio for property loans depends on factors such as the type of property and the loan amount. You can find specific details in our loan documentation or by contacting our loan specialists."),
    ("Tell me about the process for refinancing a loan.",
     "The process for refinancing a loan involves assessing your current loan, applying for a new loan with better terms, and paying off the existing loan. Contact our loan department for a consultation on whether refinancing is suitable for you."),
    ("Can I get a loan if I have a low credit score?",
     "While a low credit score may affect your eligibility, we consider various factors during the loan approval process. You can discuss your specific situation with our loan specialists to explore available options."),
    ("What is the minimum loan amount for personal loans?",
     "The minimum loan amount for personal loans depends on the loan type and your eligibility. Generally, personal loans have a minimum amount, and you can find specific details on our website or by contacting our loan team."),
    ("Tell me about the insurance options available with home loans.",
     "We offer insurance options such as home insurance and mortgage protection insurance with our home loans. These options provide coverage for unforeseen events. Contact our loan department to discuss insurance offerings in detail."),
    ("Are there any pre-approved loan offers available?",
     "We periodically provide pre-approved loan offers to eligible customers. Check your account or contact our loan specialists to inquire about any pre-approved loan options available to you."),
    ("How can I calculate the EMI for a loan?",
     "You can use our online loan EMI calculator available on our website to estimate your monthly installment for a loan. Simply input the loan amount, tenure, and interest rate to get an instant EMI calculation."),
    ("Tell me about the foreclosure process for car loans.",
     "Foreclosure for car loans involves repaying the outstanding loan amount before the scheduled tenure ends. Contact our loan department for details on the foreclosure process, including any applicable fees or charges."),
    ("What is the loan disbursement timeline for personal loans?",
     "The loan disbursement timeline for personal loans depends on the completion of the documentation and verification process. Generally, personal loan disbursement occurs within [average time]. Contact our loan team for specific details."),
    ("Can I get a loan with a co-applicant?",
     "Yes, you can apply for a loan with a co-applicant, which can enhance your eligibility. The co-applicant's income and creditworthiness are considered during the loan approval process. Contact our loan specialists for more information."),
    ("Tell me about the terms and conditions for property loan prepayment.",
     "Property loan prepayment terms and conditions may vary. Some loans allow prepayment without penalties, while others may have specific conditions. Check your loan agreement or contact our loan department for detailed information."),
    ("What is the maximum loan tenure for car loans?",
     "The maximum loan tenure for car loans depends on factors such as the loan amount and the type of vehicle. Generally, car loan tenures range from [minimum tenure] to [maximum tenure]. Contact our loan specialists for specific details."),
    ("Are there any special offers for first-time homebuyers?",
     "Yes, we often have special offers and discounts for first-time homebuyers. Check our website or contact our loan department to inquire about any exclusive deals or promotions available for first-time homebuyers."),
    ("How can I update my contact information for loan-related communication?",
     "To update your contact information, log in to your online account or contact our customer service. It's important to keep your information up-to-date to ensure you receive timely loan-related communications."),
    ("Tell me about the process for loan assumption.",
     "Loan assumption involves a new borrower taking over the existing loan. The process includes assessing the new borrower's eligibility and obtaining approval from our loan department. Contact us for detailed information on loan assumption."),
    ("Can I get a loan for purchasing a second home?",
     "Yes, we offer loans for purchasing second homes. The eligibility criteria and terms may vary. Contact our loan specialists to discuss your specific requirements for a loan on a second home."),
    ("What is the process for getting a loan statement?",
     "You can easily obtain a loan statement by logging in to your online account or contacting our loan department. Loan statements provide details on your outstanding balance, payments made, and other relevant information."),
    ("Tell me about the benefits of opting for a fixed interest rate on loans.",
     "Opting for a fixed interest rate provides stability, as your EMI remains constant throughout the loan tenure. It shields you from interest rate fluctuations. Contact our loan specialists to discuss the benefits of fixed-rate loans in detail."),
    ("How can I avail of top-up loans on my existing home loan?",
     "Availing top-up loans on your existing home loan is a convenient way to access additional funds. You can apply for a top-up loan through our online portal or by contacting our loan department for assistance."),
    ("Can I prepay a part of my car loan?",
     "Yes, you can make partial prepayments on your car loan. This helps reduce the outstanding principal amount and may lead to interest savings. Check your loan agreement for details on partial prepayment terms."),
    ("Tell me about the benefits of taking a joint home loan.",
     "Taking a joint home loan with a co-applicant, such as a spouse, can enhance your eligibility and increase the loan amount. Additionally, both applicants share the responsibility for loan repayment. Contact our loan specialists for detailed information."),
    ("What is the loan-to-value ratio for car loans?",
     "The loan-to-value ratio for car loans depends on factors such as the type of vehicle and the loan amount. Generally, the ratio ranges from [minimum ratio] to [maximum ratio]. Contact our loan specialists for specific details."),
    ("Are there any special discounts for loyalty customers applying for loans?",
     "Yes, we offer special discounts and benefits for loyal customers applying for loans. Check your account or contact our loan department to inquire about any exclusive offers available to our loyal customers."),
    ("How can I request a loan amortization schedule?",
     "You can request a loan amortization schedule by contacting our loan department. The schedule provides a detailed breakdown of your EMI payments, interest, and outstanding balance over the loan tenure."),
    ("Tell me about the benefits of opting for a floating interest rate on loans.",
     "Opting for a floating interest rate allows you to benefit from interest rate fluctuations. It may result in lower EMIs during periods of decreasing interest rates. Contact our loan specialists to discuss the advantages of floating-rate loans."),
    ("Can I transfer my existing home loan to your institution for better terms?",
     "Yes, you can consider transferring your existing home loan to our institution through the loan balance transfer process. This may provide you with better terms and interest rates. Contact our loan specialists for assistance."),
    ("What is the process for loan foreclosure on personal loans?",
     "Loan foreclosure on personal loans involves repaying the entire outstanding amount before the scheduled tenure ends. There may be specific terms and conditions. Contact our loan department for detailed information on personal loan foreclosure."),
    ("Can I get a loan for purchasing commercial property?",
     "Yes, we offer loans for purchasing commercial properties. The eligibility criteria and terms may vary. Contact our loan specialists to discuss your specific requirements for a loan on commercial property."),
    ("Tell me about the benefits of taking a loan for debt consolidation.",
     "Taking a loan for debt consolidation allows you to combine multiple debts into a single loan with a potentially lower interest rate. It simplifies your finances and may reduce your overall interest payments. Contact our loan specialists for more information."),
    ("How can I change the tenure of my personal loan?",
     "You can request a change in the tenure of your personal loan by contacting our loan department. However, changes may be subject to approval and certain conditions. Reach out to us for assistance in modifying your loan tenure."),
    ("What is the process for getting a loan against property?",
     "The process for getting a loan against property involves assessing the property's value, your eligibility, and completing the required documentation. Contact our loan specialists to discuss the specific steps and requirements."),
    ("Can I get a loan if I am self-employed?",
     "Yes, we offer loans to self-employed individuals. The eligibility criteria may vary, and we consider factors such as income stability and business performance. Contact our loan specialists to discuss loan options for self-employed individuals."),
    ("Tell me about the benefits of taking a loan for home extension.",
     "Taking a loan for home extension provides funds for expanding or renovating your existing home. It can enhance your living space and property value. Contact our loan specialists to explore the benefits and terms of home extension loans."),
    ("What is the interest rate for a car loan?",
     "The interest rate for a car loan depends on various factors such as your credit score, the loan amount, and the loan tenure. Contact our loan department for personalized information on car loan interest rates."),
    ("Can I apply for a loan online?",
     "Yes, you can apply for a loan online through our official website. The online application process is secure and convenient. Visit our website to start your loan application."),
    ("Tell me about the repayment options for personal loans.",
     "Repayment options for personal loans include monthly installments. You can choose a tenure and EMI plan that suits your financial situation. Contact our loan specialists for assistance in selecting the right repayment option."),
    ("What documents are required for a home loan application?",
     "The documents required for a home loan application typically include proof of identity, address, income, and property documents. Contact our loan department for a detailed list of documents needed for your home loan application."),
    ("Is it possible to prepay a loan before the tenure ends?",
     "Yes, you can prepay a loan before the tenure ends. However, prepayment may be subject to certain terms and conditions, and there may be prepayment charges. Contact our loan specialists for information on loan prepayment."),
    ("Can I get a loan with a low credit score?",
     "Loan approval with a low credit score depends on various factors. We offer options for individuals with less-than-perfect credit. Contact our loan specialists to discuss available loan options based on your credit situation."),
    ("Tell me about the benefits of a fixed-rate mortgage.",
     "A fixed-rate mortgage offers the advantage of a constant interest rate throughout the loan tenure. It provides stability in monthly payments, making it easier to plan your finances. Contact our loan specialists to explore the benefits of a fixed-rate mortgage."),
    ("How does the loan approval process work?",
     "The loan approval process involves submitting an application, document verification, and credit assessment. Approval is based on factors such as income, credit history, and eligibility criteria. Contact our loan specialists for a step-by-step guide on the loan approval process."),
    ("Can I get a personal loan for a vacation?",
     "Yes, you can apply for a personal loan to fund your vacation. Personal loans offer flexibility in usage, and you can use the funds for various purposes, including travel. Contact our loan department to explore personal loan options for your vacation."),
    ("What is the interest rate for a personal loan?",
     "The interest rate for a personal loan varies based on factors like credit score and loan amount. Contact our loan department for personalized information on personal loan interest rates."),
    ("Can I get a loan for starting a small business?",
     "Yes, we offer loans for entrepreneurs looking to start or expand a small business. Contact our business loan specialists for details on eligibility and terms."),
    ("Tell me about the benefits of a home equity loan.",
     "A home equity loan allows you to borrow against the equity in your home. It's useful for major expenses like home renovations. Contact our loan specialists to explore the benefits of a home equity loan."),
    ("How long does it take to get approval for a loan?",
     "The time for loan approval varies based on the type of loan and documentation. Contact our loan department for an estimate of the approval timeline for your specific loan application."),
    ("Can I get a loan if I have no credit history?",
     "While having no credit history can be a factor, we offer options for individuals with limited credit history. Contact our loan specialists to discuss available loan options based on your financial situation."),
    ("Tell me about the benefits of a variable-rate mortgage.",
     "A variable-rate mortgage offers the advantage of potential interest rate decreases, which can result in lower monthly payments. Contact our loan specialists to explore the benefits of a variable-rate mortgage."),
    ("What is the maximum loan amount for a car loan?",
     "The maximum loan amount for a car loan depends on factors such as your income and the value of the car. Contact our loan department for information on the maximum loan amount you can qualify for."),
    ("Is there a penalty for repaying a loan early?",
     "Some loans may have prepayment penalties. Contact our loan specialists to understand the terms and conditions related to early repayment for your specific loan."),
    ("Can I get a loan if I'm a first-time homebuyer?",
     "Yes, we offer special programs for first-time homebuyers. Contact our mortgage specialists to discuss the available loan options and assistance programs for first-time buyers."),
    ("Tell me about the benefits of a student loan consolidation.",
     "Consolidating student loans can simplify repayment by combining multiple loans into one. Contact our loan specialists to explore the benefits and options for student loan consolidation."),
    ("How does the loan underwriting process work?",
     "Loan underwriting involves assessing your creditworthiness and financial situation. Contact our loan specialists for a detailed explanation of the loan underwriting process and criteria."),
    ("Can I get a loan with a co-signer?",
     "Yes, having a co-signer can enhance your chances of loan approval, especially if you have limited credit history. Contact our loan specialists to discuss the requirements and benefits of having a co-signer."),
    ("Tell me about the benefits of a business expansion loan.",
     "A business expansion loan provides funds for growing your business operations. Contact our business loan specialists to explore the benefits and terms of a business expansion loan."),
    ("What is the loan-to-value (LTV) ratio for a home loan?",
     "The loan-to-value ratio for a home loan is the ratio of the loan amount to the appraised value of the property. Contact our mortgage specialists for information on LTV ratios and how they impact your loan terms."),
    ("Is there a grace period for loan payments?",
     "Some loans may have a grace period, allowing for a brief delay in payments without incurring penalties. Contact our loan specialists to understand the terms related to grace periods for your specific loan."),
    ("Can I refinance my existing mortgage?",
     "Yes, you can refinance your existing mortgage to potentially get a lower interest rate or change the loan terms. Contact our mortgage specialists to discuss the refinancing options available to you."),
    ("Tell me about the benefits of a business equipment loan.",
     "A business equipment loan provides funds for purchasing or upgrading business equipment. Contact our business loan specialists to explore the benefits and terms of a business equipment loan."),
    ("How can I improve my credit score to qualify for a loan?",
     "Improving your credit score involves managing debts, making timely payments, and monitoring your credit report. Contact our credit counseling services for guidance on improving your credit score for loan eligibility."),
    ("What is the eligibility criteria for a personal loan?",
     "The eligibility criteria for a personal loan include factors like income, credit score, and employment status. Contact our loan specialists to understand the specific eligibility criteria for personal loans."),
    ("Tell me about the benefits of an FHA loan for homebuyers.",
     "An FHA loan is a government-backed mortgage with lower down payment requirements. Contact our mortgage specialists to explore the benefits and eligibility criteria for FHA loans."),
    ("Can I get a loan for debt consolidation if I have multiple debts?",
     "Yes, debt consolidation loans help combine multiple debts into a single loan with a potentially lower interest rate. Contact our loan specialists to discuss options for debt consolidation based on your financial situation."),
    ("How does the loan disbursement process work?",
     "The loan disbursement process involves transferring approved loan funds to your account. Contact our loan department for details on the disbursement process and the timeline for receiving your loan amount."),
    ("What is the maximum loan tenure for a personal loan?",
     "The maximum loan tenure for a personal loan depends on the lender's policies and the type of personal loan. Contact our loan specialists for information on the maximum tenure available for personal loans."),
    ("Tell me about the benefits of an unsecured business loan.",
     "An unsecured business loan doesn't require collateral and offers flexibility. Contact our business loan specialists to explore the benefits and terms of an unsecured business loan."),
    ("Can I get a loan if I'm retired and on a fixed income?",
     "Yes, retirees can qualify for loans based on their fixed income and creditworthiness. Contact our loan specialists to discuss loan options tailored to retirees and the specific documentation required."),
    ("What is the difference between a fixed-rate and adjustable-rate mortgage?",
     "A fixed-rate mortgage has a constant interest rate, while an adjustable-rate mortgage may change over time. Contact our mortgage specialists to understand the differences and determine which option suits your needs."),
    ("Is there a prepayment penalty for home loans?",
     "Some home loans may have prepayment penalties. Contact our mortgage specialists to understand the terms related to prepayment penalties and whether they apply to your specific home loan."),
    ("Tell me about the benefits of a business line of credit.",
     "A business line of credit provides flexible access to funds for day-to-day operations. Contact our business loan specialists to explore the benefits and terms of a business line of credit."),
    ("How can I check my credit score before applying for a loan?",
     "You can check your credit score through credit reporting agencies. Contact our credit counseling services for guidance on obtaining and understanding your credit report before applying for a loan."),
    ("What is the process for getting a personal loan with bad credit?",
     "While having bad credit can pose challenges, there are options for obtaining a personal loan. Contact our loan specialists to discuss available solutions and steps to secure a personal loan with bad credit."),
    ("Can I get a loan for home renovation projects?",
     "Yes, we offer loans specifically for home renovation projects. Contact our loan specialists to discuss the available loan options and terms for financing your home improvement plans."),
    ("Tell me about the benefits of a jumbo mortgage for luxury homes.",
     "A jumbo mortgage is designed for high-value homes, offering larger loan amounts. Contact our mortgage specialists to explore the benefits and eligibility criteria for jumbo mortgages."),
    ("How does the loan application process work for business loans?",
     "The business loan application process involves submitting financial documents and a business plan. Contact our business loan specialists for a step-by-step guide on applying for business loans and the required documentation."),
    ("What is the difference between a secured and unsecured loan?",
     "A secured loan requires collateral, while an unsecured loan does not. Contact our loan specialists to understand the differences and determine which type of loan aligns with your financial needs."),
    ("Is there a minimum credit score requirement for mortgage loans?",
     "Mortgage lenders may have minimum credit score requirements. Contact our mortgage specialists to understand the specific credit score requirements for mortgage loans and how they may impact your eligibility."),
    ("Tell me about the benefits of an SBA loan for small businesses.",
     "SBA loans provide government-backed funding for small businesses. Contact our business loan specialists to explore the benefits and eligibility criteria for Small Business Administration (SBA) loans."),
    ("How to apply for an education loan?",
     "To apply for an education loan, visit our website or the nearest branch to access the application form. Complete the required documentation, including proof of admission and course details. Our loan specialists will guide you through the process."),
    ("What is the interest rate on education loans?",
     "The interest rate on education loans varies and depends on factors such as the loan amount, tenure, and your credit history. Contact our loan department or check our website for the most up-to-date interest rates."),
    ("How is the EMI cost calculated for education loans?",
     "The EMI (Equated Monthly Installment) for education loans is calculated based on the loan amount, interest rate, and tenure. You can use our online EMI calculator on the website to get an estimate of your monthly installment."),
    ("How can I exit my education loan?",
     "To exit your education loan, you need to repay the entire outstanding amount. You can choose to make prepayments or pay the loan in full. Contact our loan department for details on the repayment process and any applicable charges."),
    ("Is there a difference between SWIFT and IBAN for international transfers?",
     "Yes, SWIFT (Society for Worldwide Interbank Financial Telecommunication) and IBAN (International Bank Account Number) serve different purposes. SWIFT is used for bank identification, while IBAN is the recipient's unique account number. Both are essential for international transfers."),
    ("Can I initiate an international money transfer over the phone?",
     "Yes, you can initiate an international money transfer over the phone by contacting our customer support. Follow the instructions provided by the customer support representative to verify your identity and complete the transfer."),
    ("How can I initiate an international money transfer?",
     "To initiate an international money transfer, log in to our online banking portal and navigate to the 'International Transfers' or 'Send Money Abroad' section. Follow the prompts, provide the recipient's details, and select the destination country."),
    ("What information do I need for an international money transfer?",
     "For an international money transfer, you typically need the recipient's full name, international bank account number (IBAN), bank name, branch details, and the destination country. Double-check the details to ensure a smooth transfer."),
    ("Is there a fee for international money transfers?",
    "Yes, international money transfers usually involve fees. Review our fee schedule or contact our customer support for information on fees associated with sending money abroad. Currency conversion fees may also apply."),
    ######
    #Trasactions
    ("How can I check my recent transactions?",
    "You can check your recent transactions by logging in to your online banking account and navigating to the 'Transactions' or 'Account Activity' section. There, you'll find a list of your most recent transactions with details like date, amount, and merchant information."),

    ("Where can I view my transaction history?",
    "Your transaction history is available in your online banking account. Simply log in and go to the 'Transaction History' or 'Account Activity' section to view a comprehensive list of your past transactions."),

    ("How do I access my bank statements?",
    "To access your bank statements, log in to your online banking account and go to the 'Statements' or 'Documents' section. You can download and view your statements in a printable format."),

    ("How can I check if I've received money from someone?",
    "You can check if you've received money by reviewing your transaction history in the online banking account. Look for incoming transactions or payments from other users."),

    ("What is the process for transferring money to a friend?",
    "To transfer money to a friend, use the 'Transfer' or 'Send Money' feature in your online banking. Enter your friend's account details and the amount you want to transfer. Follow the on-screen prompts to complete the transaction."),









]

# Train the machine learning models
banking_chatbot.train_ml_models(dataset)

#
# print(" ")
# print("Welcome to the Banking Chatbot!")
# print(" ")
# print("We're delighted to assist you with all your queries and needs.\nFeel free to ask any questions or seek help regarding our services. ")
# print(" ")
# print("Type 'quit', 'exit', 'q' to exit the chatbot")
# print("____________________________________________________________________")
# print(" ")
#
# while True:
#     user_input = input('You: ')
#
#     # Check if the user wants to quit
#     if user_input.lower() in ['quit', 'exit', 'q']:
#         print('Exiting the chatbot. Goodbye!')
#         break
#
#     # Get the chatbot's response
#     response = banking_chatbot.respond(user_input)
#
#     # Print the response
#     print('Bot:', response)

def print_with_delay(text, delay=0.03):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def print_welcome_message():
    print(" ")
    print_with_delay("Welcome to the Banking Chatbot!")
    print(" ")
    print("We're delighted to assist you with all your queries and needs.\nFeel free to ask any questions or seek help regarding our services.")
    print(" ")
    print_with_delay("Type 'quit', 'exit', or 'q' to exit the chatbot")
    print("____________________________________________________________________")
    print(" ")

def ask_question(question):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] You: {question}")

# Call the function to print the welcome message
print_welcome_message()

while True:
    user_input = input('You: ')

    # Add timestamp for user input
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] You: {user_input}")

    # Check if the user wants to quit
    if user_input.lower() in ['quit', 'exit', 'q']:
        print('Bot: Exiting the chatbot. Goodbye!')
        break

    # Get the chatbot's response
    response = banking_chatbot.respond(user_input)

    # Add timestamp for bot response
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] Bot: {response}")
