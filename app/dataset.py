import pandas as pd
from transformers import AutoTokenizer
import json
from typing import Literal
import torch

class PreprocessPodcastDataset:

    """
    Maintains all code related to ingestion and preprocessing of data required to fine-tune
    a model to predict the tokens of text that form the introduction of a podcast from its audio transcript
    """

    def __init__(self, fpath: str):

        print("Loading dataset...")
        self.dataset = pd.read_csv(fpath, sep='\t')
        # convert timestamps all to milliseconds
        self.dataset['episode_intro_start'] = self.dataset['episode_intro_start'] * 1000
        self.dataset['episode_intro_end'] = self.dataset['episode_intro_end'] * 1000
        self.dataset = self.dataset.sample(frac=1).reset_index(drop=True) # randomly shuffle rows prior to train test split
        self.labeled_dataset = []
        self.train_pairs = []
        self.eval_pairs = []
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def _annotate_data(self, intro_start: float, intro_end: float, transcription: str) -> dict:
        """
        Take the start and end timestamps for podcast introduction and annotate the audio transcription
        data with this information to create token level labels
        """

        # Find first word after intro_start
        # Find last word before intro_end
        # Create list of labels where the length of the list is equal to the number of word tokens in the dataset
        data = json.loads(transcription)
        word_timestamps = data['words']
        start_found, end_found = False, False
        i = 0

        # if this were a repeated process, this could be optimzied by performing binary search
        while not start_found or not end_found:
            if i >= len(word_timestamps): # the end time stamp is at the end of the podcast
                intro_end_idx = i - 1
                end_found = True
            else:
                word_obj = word_timestamps[i]     
                starttime = word_obj['startTime']
                if not start_found:
                    if starttime >= intro_start:
                        intro_start_idx = i
                        start_found = True
                if not end_found:
                    if starttime >= intro_end:
                        # take previous word
                        intro_end_idx = i-1
                        end_found = True
                i += 1
        
        labels = [0] * len(transcription.split(" "))
        labels[intro_start_idx:intro_end_idx+1] = [1] * (intro_end_idx - intro_start_idx + 1)
        return {"text": data['transcription'], "labels": labels}

    def create_labels(self):

        # apply function to annotate data on each row of dataframe
        num_rows = self.dataset.shape[0]
        for i, row in self.dataset.iterrows():

            train_pair = self._annotate_data(row['episode_intro_start'], row['episode_intro_end'], row['transcription'])
            self.labeled_dataset.append(train_pair)
            if (i/num_rows) >= 0.70: # use 30% of dataset for evaluation
                self.eval_pairs.append(train_pair)
            else:
                self.train_pairs.append(train_pair)

    def _tokenize_and_preserve_labels(self, document_obj: dict):

        """
        Preprocessing step for text field of data. Since BERT model will
        perform subword tokenization and we need a label for each token,
        the labels and tokens need to be aligned so they are of the same
        length
        """        

        tokenized_doc = []
        labels = []
        # labels = [-100] # assign label -100 to special tokens [CLS] and [SEP] so they're ignored by loss function
        document, document_labels = document_obj['text'], document_obj['labels']
        for word, label in zip(document.split(" "), document_labels):

            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = self.tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)

            # Add the tokenized word to the final tokenized document
            tokenized_doc.extend(tokenized_word)

            # Add the same label to the new list of labels `n_subwords` times
            labels.extend([label] * n_subwords)

         # labels.append(-100)
        return labels

    def tokenize_documents(self, dataset: Literal["train", "eval"]):
        
        if dataset == 'train':
            data = self.train_pairs
        elif dataset == 'eval':
            data = self.eval_pairs
        else:
            raise Exception("Dataset input must be one of ['train', 'eval']")
        
        new_labels = []
        document_texts = []
        for i in range(len(data)):
            doc = data[i]['text']
            document_texts.append(doc)
            new_label = self._tokenize_and_preserve_labels(data[i])
            new_labels.append(new_label)

        tokenized_inputs = self.tokenizer(document_texts, add_special_tokens=False) # don't include special tokens as they will be added when dataset is chunked
        tokenized_inputs['labels'] = new_labels
        return tokenized_inputs


class PodcastDataset(torch.utils.data.Dataset):

    def __init__(self, tokenized_dataset):
        self.tokenized_dataset = tokenized_dataset

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
        # item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.tokenized_dataset['labels'])