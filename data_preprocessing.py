import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Load the data from a text file
with open('scraped_transcripts.txt', 'r', encoding='utf-8') as file:
    # Assuming each transcript is separated by two new lines
    transcripts = file.read().split('\n')

# Remove any empty transcripts
transcripts = [t for t in transcripts if t.strip()]

# Convert list to DataFrame
data = pd.DataFrame({'transcript': transcripts})

# Step 2: Splitting the data

train, test = train_test_split(data, test_size=0.30, random_state=42)

# Step 3: Save the data to text files
# We join the transcripts back with two new lines as separator
train_transcripts = '\n'.join(train['transcript'])
test_transcripts = '\n'.join(test['transcript'])

with open('train.txt', 'w', encoding='utf-8') as file:
    file.write(train_transcripts)

with open('test.txt', 'w', encoding='utf-8') as file:
    file.write(test_transcripts)

print("Training data saved as train.txt")
print("Testing data saved as test.txt")
