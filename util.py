import csv

import matplotlib.pyplot as plt
import numpy as np
import json


def load_review_dataset(csv_path):
    """Load the spam dataset from a CSV file

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        messages: A list of string values containing the text of each message.
        labels: The binary labels (0 or 1) for each message. A 1 indicates spam.
    """

    messages = []
    labels = []

    with open(csv_path, 'r', newline='', encoding='utf8') as csv_file:
        reader = csv.reader(csv_file)

        for message, label in reader:
            messages.append(message)
            labels.append(1 if label == 'positive' else 0)

    return messages, np.array(labels)


def write_json(filename, value):
    """Write the provided value as JSON to the given filename"""
    with open(filename, 'w') as f:
        json.dump(value, f)