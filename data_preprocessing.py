import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Load the data
data = pd.read_csv('comedy-transcripts.csv')

# Print first 5 rows to understand the data
print(data.head())

# Checking for missing values
print(data.isnull().sum())

# Step 2: Splitting the data
# Note that test_size=0.25 for a 25% test set, leaving 75% for training
train, test = train_test_split(data, test_size=0.25, random_state=42)

# Step 3: Save the data
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)

print("Training data saved as train.csv")
print("Testing data saved as test.csv")
