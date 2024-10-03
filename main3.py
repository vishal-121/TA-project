# Import required libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data directly inside the code
data = [
    {"text": "I just love getting stuck in traffic for hours.", "label": 1},
    {"text": "Wow, what a beautiful sunny day!", "label": 0},
    {"text": "Oh great, another meeting that could have been an email.", "label": 1},
    {"text": "I am so thrilled to clean the house.", "label": 1},
    {"text": "I can't wait to see that movie!", "label": 0},
    {"text": "This is the best meal I've ever had.", "label": 1},
    {"text": "The weather is perfect for a beach day.", "label": 0},
    {"text": "Fantastic, now my phone is dead.", "label": 1},
    {"text": "I truly enjoy working overtime on weekends.", "label": 1},
    {"text": "This vacation is going to be amazing.", "label": 0},
    {"text": "I just love getting stuck in traffic for hours.", "label": 1},
    {"text": "Wow, what a beautiful sunny day!", "label": 0},
    {"text": "Oh great, another meeting that could have been an email.", "label": 1},
    {"text": "I am so thrilled to clean the house.", "label": 1},
    {"text": "I can't wait to see that movie!", "label": 0},
    {"text": "This is the best meal I've ever had.", "label": 1},
    {"text": "The weather is perfect for a beach day.", "label": 0},
    {"text": "Fantastic, now my phone is dead.", "label": 1},
    {"text": "I truly enjoy working overtime on weekends.", "label": 1},
    {"text": "This vacation is going to be amazing.", "label": 0},
    {"text": "Oh great, another flat tire!", "label": 1},
    {"text": "I can't believe I forgot my umbrella on a rainy day.", "label": 1},
    {"text": "This is exactly what I wanted for my birthday.", "label": 1},
    {"text": "I'm so excited for another day at the office!", "label": 1},
    {"text": "Yay, more emails to respond to!", "label": 1},
    {"text": "This is the easiest exam I've ever taken.", "label": 1},
    {"text": "What a surprise! I didn't expect that!", "label": 1},
    {"text": "I've always wanted to wait in long lines.", "label": 1},
    {"text": "This new policy makes everything so much easier!", "label": 1},
    {"text": "I love it when my plans get canceled last minute.", "label": 1},
    {"text": "What a lovely day for a hike!", "label": 0},
    {"text": "I just adore stepping in gum on the sidewalk.", "label": 1},
    {"text": "Fantastic! Another day of doing nothing!", "label": 1},
    {"text": "I can't wait to spend my weekend doing laundry!", "label": 1},
    {"text": "How wonderful, my internet is down again!", "label": 1},
    {"text": "This traffic jam is just what I needed!", "label": 1},
    {"text": "So glad to have a dentist appointment today!", "label": 1},
    {"text": "What a delightful surprise to find my coffee cold.", "label": 1},
    {"text": "I'm absolutely thrilled to hear my neighbor's music at 2 AM.", "label": 1},
    {"text": "This is the most thrilling book I've ever read!", "label": 1},
    {"text": "Wow, this is the best service I've ever received!", "label": 0},
    {"text": "So glad I stayed up all night for this!", "label": 1},
    {"text": "Can't wait to get back to work after this vacation.", "label": 1},
    {"text": "This salad is just bursting with flavor!", "label": 0},
    {"text": "Oh, how I love waking up to the sound of construction!", "label": 1},
    {"text": "What a fantastic way to start the week!", "label": 0},
    {"text": "I'm just loving this winter weather!", "label": 1},
    {"text": "So glad my phone battery died right when I needed it!", "label": 1},
    {"text": "I just love being stuck in traffic for hours.", "label": 1},
    {"text": "Wow, what a beautiful sunny day!", "label": 0},
    {"text": "Oh great, another meeting that could have been an email.", "label": 1},
    {"text": "I'm so thrilled to clean the house.", "label": 1},
    {"text": "I can't wait to see that movie!", "label": 0},
    {"text": "This is the best meal I've ever had.", "label": 1},
    {"text": "The weather is perfect for a beach day.", "label": 0},
    {"text": "Fantastic, now my phone is dead.", "label": 1},
    {"text": "I truly enjoy working overtime on weekends.", "label": 1},
    {"text": "This vacation is going to be amazing.", "label": 0},
    {"text": "Oh great, another flat tire!", "label": 1},
    {"text": "I can't believe I forgot my umbrella on a rainy day.", "label": 1},
    {"text": "This is exactly what I wanted for my birthday.", "label": 1},
    {"text": "I'm so excited for another day at the office!", "label": 1},
    {"text": "Yay, more emails to respond to!", "label": 1},
    {"text": "This is the easiest exam I've ever taken.", "label": 1},
    {"text": "What a surprise! I didn't expect that!", "label": 1},
    {"text": "I've always wanted to wait in long lines.", "label": 1},
    {"text": "This new policy makes everything so much easier!", "label": 1},
    {"text": "I love it when my plans get canceled last minute.", "label": 1},
    {"text": "What a lovely day for a hike!", "label": 0},
    {"text": "I just adore stepping in gum on the sidewalk.", "label": 1},
    {"text": "Fantastic! Another day of doing nothing!", "label": 1},
    {"text": "I can't wait to spend my weekend doing laundry!", "label": 1},
    {"text": "How wonderful, my internet is down again!", "label": 1},
    {"text": "This traffic jam is just what I needed!", "label": 1},
    {"text": "So glad to have a dentist appointment today!", "label": 1},
    {"text": "What a delightful surprise to find my coffee cold.", "label": 1},
    {"text": "I'm absolutely thrilled to hear my neighbor's music at 2 AM.", "label": 1},
    {"text": "This is the most thrilling book I've ever read!", "label": 1},
    {"text": "Wow, this is the best service I've ever received!", "label": 0},
    {"text": "So glad I stayed up all night for this!", "label": 1},
    {"text": "Can't wait to get back to work after this vacation.", "label": 1},
    {"text": "This salad is just bursting with flavor!", "label": 0},
    {"text": "Oh, how I love waking up to the sound of construction!", "label": 1},
    {"text": "What a fantastic way to start the week!", "label": 0},
    {"text": "I'm just loving this winter weather!", "label": 1},
    {"text": "So glad my phone battery died right when I needed it!", "label": 1},
    {"text": "Oh great, my car broke down again.", "label": 1},
    {"text": "Just what I needed, more rain!", "label": 1},
    {"text": "I can't wait to hear my alarm clock go off tomorrow!", "label": 1},
    {"text": "This is just the kind of day I was hoping for.", "label": 0},
    {"text": "I've always wanted to trip over my own feet.", "label": 1},
    {"text": "What a surprise to find the store closed!", "label": 1},
    {"text": "I'm so happy to sit in a waiting room for hours.", "label": 1},
    {"text": "Oh fantastic, I just lost my wallet.", "label": 1},
    {"text": "I love being in the middle of a family argument!", "label": 1},
    {"text": "This dinner is absolutely what I was craving!", "label": 0},
    {"text": "Yay, my flight was delayed!", "label": 1},
    {"text": "I'm just thrilled to do taxes this year!", "label": 1},
    {"text": "What a wonderful surprise, another unexpected expense!", "label": 1},
    {"text": "I'm really enjoying this never-ending lecture.", "label": 1},
]

# Split data into text and labels
texts = [item['text'] for item in data]
labels = [item['label'] for item in data]

# Use CountVectorizer to convert text to a bag-of-words model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Use Logistic Regression for sarcasm detection
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Function to predict sarcasm for new sentences
def predict_sarcasm(new_sentences):
    # Transform the new sentences to match the training data format
    new_X = vectorizer.transform(new_sentences)
    predictions = model.predict(new_X)
    return predictions

# Example of predicting sarcasm for new sentences
new_sentences = [
    "I love being stuck in the rain.",
    "This is the worst day ever!",
    "Can't wait to do my taxes!",
]

predictions = predict_sarcasm(new_sentences)
for sentence, prediction in zip(new_sentences, predictions):
    print(f"Sentence: '{sentence}' - Sarcasm Detected: {'Yes' if prediction == 1 else 'No'}")
