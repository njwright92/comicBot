import pandas as pd
from sklearn.model_selection import train_test_split

# Function to tokenize text into sentences


def tokenize_sentences(text):
    return '. '.join(text.split('. '))


# Step 1: Load the data
data = pd.read_csv('comedy-dad-jokes-combined.csv')

# Print first 5 rows to understand the data
print(data.head())

# Checking for missing values
print(data.isnull().sum())

# Tokenize the setup into sentences
data['setup_sentences'] = data['setup'].apply(tokenize_sentences)

# Step 3: Splitting the data
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Save the data
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)

print("Training data saved as train.csv")
print("Testing data saved as test.csv")
