import pandas as pd
from sklearn.model_selection import train_test_split
import string

# Function to tokenize text into sentences


def tokenize_sentences(text):
    return '. '.join(text.split('. '))


# Step 1: Load the data
data = pd.read_csv('comedy-transcripts-with-sentences.csv')

# Print first 5 rows to understand the data
print(data.head())

# Checking for missing values
print(data.isnull().sum())

# Step 2: Text Cleaning (optional, based on your needs)
# Converting text to lowercase
# Removing punctuation
data['transcript'] = data['transcript'].str.lower()
data['transcript'] = data['transcript'].str.replace(
    '[{}]'.format(string.punctuation), '')

# Tokenize the transcripts into sentences
data['sentences'] = data['transcript'].apply(tokenize_sentences)

# Step 3: Splitting the data
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Save the data
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)

print("Training data saved as train.csv")
print("Testing data saved as test.csv")
