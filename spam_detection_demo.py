# CountVectorizer: turns text into numbers (counts of words).
# MultinomialNB: a simple machine learning model good for text classification.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 1) Example data (a bit more than before)
texts = [
    "win a free prize",          # spam
    "you won money",             # spam
    "click here to get cash",    # spam
    "limited time offer",        # spam
    "let's meet tomorrow",       # ham
    "can we have lunch",         # ham
    "see you in class",          # ham
    "hi how are you",            # ham
]

labels = [
    "spam",
    "spam",
    "spam",
    "spam",
    "ham",
    "ham",
    "ham",
    "ham",
]

# 2) Split data into training and test sets (simple manual split)
train_texts = texts[:6]   # first 6 for training
train_labels = labels[:6]

test_texts = texts[6:]    # last 2 for testing
test_labels = labels[6:]

# 3) Turn text into numbers using training data
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_texts)

# 4) Train the model
model = MultinomialNB()
model.fit(X_train, train_labels)

# 5) EVALUATE MODEL PERFORMANCE ON TEST DATA
X_test = vectorizer.transform(test_texts)
test_predictions = model.predict(X_test)

# accuracy = how many it got right / total
accuracy = accuracy_score(test_labels, test_predictions)
print("=== Model Evaluation ===")
print("Test messages:")
for msg, true_label, pred_label in zip(test_texts, test_labels, test_predictions):
    print(f"  Text: {msg}")
    print(f"  True label: {true_label}, Predicted: {pred_label}")
    print()

print(f"Overall accuracy on test data: {accuracy * 100:.2f}%")
print("========================\n")

# 6) USER INPUT LOOP (for actual classification / filtering)
while True:
    text = input("Enter a message to classify (or type 'quit' to stop): ")
    if text.lower() == "quit":
        break

    X_user = vectorizer.transform([text])
    label = model.predict(X_user)[0]  # "spam" or "ham"

    print("Message:", text)
    print("ML classification:", label)
    print("-" * 40)
