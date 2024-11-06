from torch.utils.data import Dataset
import torch
import random
import re


class ImageCaptionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        """
        Initialize the dataset with conversation data and tokenizer.

        Args:
            data (list): List of dictionaries containing conversation data
            tokenizer: Tokenizer instance for encoding text
            max_length (int): Maximum length for tokenized sequences
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Return the total number of conversations in the dataset."""
        return len(self.data)

    def clean_text(self, text):
        """
        Clean the text by removing image tags and extra whitespace.

        Args:
            text (str): Input text to clean

        Returns:
            str: Cleaned text
        """
        # Remove <image> tags
        text = re.sub(r"<image>\s*", "", text)
        # Remove extra whitespace
        text = " ".join(text.split())
        return text

    def __getitem__(self, idx):
        """
        Get a single conversation pair from the dataset.

        Args:
            idx (int): Index of the conversation

        Returns:
            dict: Dictionary containing image_id, input_ids, attention_mask, and labels
        """
        item = self.data[idx]
        image_id = item["id"]
        conversations = item["conversations"]

        # Randomly select a question-answer pair
        max_pairs = len(conversations) // 2
        selected_pair = random.randrange(max_pairs)
        question_idx = selected_pair * 2

        # Get the human question and AI response
        question = self.clean_text(conversations[question_idx]["value"])
        answer = self.clean_text(conversations[question_idx + 1]["value"])

        input_ids = self.tokenizer.encode(question, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        target_ids = self.tokenizer.encode(answer, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        
        input_ids = input_ids.squeeze(0)
        target_ids = target_ids.squeeze(0)
        
        return image_id, input_ids, target_ids
