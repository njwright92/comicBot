from datasets import load_dataset


def main():
    # Load the dataset
    datasets = load_dataset(
        'csv', data_files={'train': 'train.csv', 'test': 'test.csv'})

    # Print the dataset
    print(datasets)

    # Check the size of the train and test datasets
    print(f'Train size: {len(datasets["train"])}')
    print(f'Test size: {len(datasets["test"])}')

    # Print the first example from the train and test datasets
    print(f'First training example: {datasets["train"][0]}')
    print(f'First test example: {datasets["test"][0]}')


if __name__ == "__main__":
    main()
