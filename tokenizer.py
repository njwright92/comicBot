import json
import random


def add_markers(text: str, segment_length: int = 612) -> str:
    """Add start and end markers to the text every specified number of characters."""
    # Initialize variables for chunking
    new_text = ""
    start = 0

    # Randomly select a token pair for the entire text
    token_pairs = [
        ("<STANDUP_START>", "<STANDUP_END>"),
        ("<COMEDY_START>", "<COMEDY_END>"),
        ("<JOKES_START>", "<JOKES_END>")
    ]
    start_token, end_token = random.choice(token_pairs)

    # Append the start token at the beginning
    new_text += start_token + " <s> "

    while start < len(text):
        end = min(start + segment_length, len(text))
        new_text += text[start:end]

        # Check if this is not the last segment; then, add end and start tokens for the next segment
        if end < len(text):
            new_text += " </s> <s> "
        start = end

    # Close the last segment with end token and then append the end token for the entire text
    new_text += " </s> " + end_token

    return new_text


def process_transcripts(file_path: str) -> None:
    """Process each transcript in the JSON file with start and end markers."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    processed_data = []
    for entry in data:
        # Wrap the title with metadata markers without altering the transcript
        metadata_title = "<METADATA>" + entry['title'] + "</METADATA>"
        # Add markers to the transcript text without tokenization
        updated_transcript = add_markers(entry['transcript'])
        processed_entry = {
            "url": entry["url"],
            "title": metadata_title,  # Updated to wrap only the title with metadata
            "transcript": updated_transcript  # Transcript text with added markers
        }
        processed_data.append(processed_entry)

    # Write processed data back to file
    with open('marked_yt_transcripts.json', 'w', encoding='utf-8') as file:
        json.dump(processed_data, file, ensure_ascii=False, indent=4)


# Example usage
file_path = 'yt_transcripts.json'
process_transcripts(file_path)
