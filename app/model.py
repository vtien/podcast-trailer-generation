from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
import evaluate
import torch
from typing import List
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve(strict=True).parent

class PodcastIntroExtractorModel:

    """
    Token classification model that predicts the start and end of the introduction / overview 
    of a podcast. This is to be used as the first component of an automatically generated podcast trailer
    and is inspired from by the approach in Jing et. al., 2021
    """

    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.max_input_length = self.tokenizer.model_max_length

    def compute_metrics(self, p: List[tuple]):

        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [str(p) for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [str(l) for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        # load evaluation metric
        seqeval = evaluate.load("seqeval")
        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def train(self, tokenized_train_dataset, tokenized_eval_dataset, dataset_class):

        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)      
        tokenized_train_dataset = self.chunk_train(tokenized_train_dataset)
        tokenized_eval_dataset = self.chunk_train(tokenized_eval_dataset)  
        train_dataset = dataset_class(tokenized_train_dataset)
        eval_dataset = dataset_class(tokenized_eval_dataset)

        model = AutoModelForTokenClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2)  

        training_args = TrainingArguments(
            output_dir="model/",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=2,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

    def chunk_train(self, tokenized_dataset):

        """
        Split training dataset into chunks of 512 tokens with 128 tokens of overlap
        """

        token_overlap = 128
        chunk_length = self.max_input_length - 2 # leave space for adding [CLS] and [SEP] token
        new_vals = {k: [] for k in tokenized_dataset}

        # first chunk labels to maintain which chunks we will include in training (since we don't want to include chunks with only negative examples)
        labels = tokenized_dataset['labels']
        for i, label_list in enumerate(labels):
            inpt_ids = tokenized_dataset['input_ids'][i]
            attn_mask = tokenized_dataset['attention_mask'][i]            
            while len(label_list) > chunk_length or (len(label_list) < chunk_length and len(label_list) != 0):
                # labels
                new_labels = [-100]
                new_labels.extend(label_list[:chunk_length])
                new_labels.append(-100)

                # input_ids
                input_ids = [101]
                input_ids.extend(inpt_ids[:chunk_length])
                input_ids.append(102)

                # attention mask
                attention_mask = [1]
                attention_mask.extend(attn_mask[:chunk_length])
                attention_mask.append(1)

                # add data chunk - only include chunks that include positive examples
                if np.sum(label_list[:chunk_length]) > 0:
                    
                    # pad chunk if necessary
                    pad_length = self.max_input_length - len(new_labels)
                    if pad_length > 0:
                        # labels
                        new_labels.extend([-100]*pad_length)
                        # input ids
                        input_ids.extend([0]*pad_length)
                        # attention mask
                        attention_mask.extend([0]*pad_length)

                    new_vals['labels'].append(new_labels)
                    new_vals['input_ids'].append(input_ids)
                    new_vals['attention_mask'].append(attention_mask)

                # slide window
                label_list = label_list[chunk_length-token_overlap:]
                inpt_ids = inpt_ids[chunk_length-token_overlap:]
                attn_mask = attn_mask[chunk_length-token_overlap:]

        # update tokenized dataset with new vals
        for k in tokenized_dataset:
            tokenized_dataset[k] = new_vals[k]

        return tokenized_dataset

    def chunk_input_text(self, tokenized_dataset):

        """
        Enables splitting text provided to the model at inference time into chunks of 512 tokens
        to account for long podcast transcripts
        """

        new_dataset = {k: [] for k in tokenized_dataset}
        token_overlap = 128
        chunk_length = self.max_input_length - 2
        inpt_ids = tokenized_dataset['input_ids']
        attn_mask = tokenized_dataset['attention_mask']
        input_ids = inpt_ids
        
        while len(inpt_ids) > chunk_length or (len(inpt_ids) < chunk_length and len(inpt_ids) != 0):
            # input_ids
            input_ids = [101]
            input_ids.extend(inpt_ids[:chunk_length])
            input_ids.append(102)

            # attention mask
            attention_mask = [1]
            attention_mask.extend(attn_mask[:chunk_length])
            attention_mask.append(1)

            # pad chunk if necessary
            pad_length = self.max_input_length - len(input_ids)
            if pad_length > 0:
                # input ids
                input_ids.extend([0]*pad_length)
                # attention mask
                attention_mask.extend([0]*pad_length)

            new_dataset['input_ids'].append(torch.Tensor(input_ids).flatten().long())
            new_dataset['attention_mask'].append(torch.Tensor(attention_mask).flatten().long())

            # slide window
            inpt_ids = inpt_ids[chunk_length-token_overlap:]
            attn_mask = attn_mask[chunk_length-token_overlap:]

        return new_dataset

    def _predict(self, model, tokenized_chunk):

        """
        Perform inference according to the approach followed in Jing et. al., 2021, where a window size of
        k is used to find the most probable start and end boundaries for the introduction
        """

        token_ids = tokenized_chunk['input_ids'][0]
        with torch.no_grad():
            logits = model(**tokenized_chunk).logits

        k = 5 # window size
        start_probs = []
        end_probs = []
        for i in range(len(logits[0])):
            prob1_sum_forward, prob1_sum_backward = 0,0
            for j in range(1,k):
                if i+j < len(logits[0]):
                    prob1_sum_forward += logits[0][i+j][1]
                if i-j >= 0:
                    prob1_sum_backward += logits[0][i-j][1]
            prob1_sum = (prob1_sum_forward / k) - (prob1_sum_backward / k)
            prob1_sum_end = -1 * prob1_sum
            start_probs.append(prob1_sum)
            end_probs.append(prob1_sum_end)            

        start_prediction = np.argmax(start_probs)
        start_val = start_probs[start_prediction]
        end_prediction = np.argmax(end_probs[start_prediction:]) + start_prediction
        end_val = end_probs[end_prediction]
        words = self.tokenizer.convert_ids_to_tokens(token_ids)
        prediction = " ".join(words[start_prediction:end_prediction+1])
        confidence = float((start_val + end_val) / 2)
        return confidence, prediction

    def predict(self, text: str):

        """
        Podcast transcripts are longer than 512 tokens, the maximum input length of the BERT model used
        here. Thus, podcast transcripts will be split into chunks of 512 tokens and be processed
        one after the other
        """
        model = AutoModelForTokenClassification.from_pretrained(f"{BASE_DIR}/model/checkpoint-40", local_files_only=True)
        tokenized_text = self.tokenizer(text, add_special_tokens=False)
        chunked_dataset = self.chunk_input_text(tokenized_text)
        max_conf = float("-inf")
        best = {}
        for i in range(len(chunked_dataset['input_ids'])):

            chunk_dataset = {"input_ids": chunked_dataset['input_ids'][i].unsqueeze(0).long(),
                            "attention_mask": chunked_dataset['attention_mask'][i].unsqueeze(0).long()}
            conf, pred = self._predict(model, chunk_dataset)
            if conf > max_conf:
                max_conf = conf
                best['confidence'] = conf
                best['prediction'] = pred

        return best