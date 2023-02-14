from app.model import PodcastIntroExtractorModel
from app.dataset import PreprocessPodcastDataset, PodcastDataset

def train():

    # Load model and dataset
    model = PodcastIntroExtractorModel()
    dataset = PreprocessPodcastDataset(fpath="./podcast_intro_data_pub.tsv")
    dataset.create_labels()
    tokenized_train_dataset, tokenized_eval_dataset = dataset.tokenize_documents("train"), dataset.tokenize_documents("eval")

    # Train model
    model.train(tokenized_train_dataset, tokenized_eval_dataset, PodcastDataset)


if __name__ == "__main__":

    train()