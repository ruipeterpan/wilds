from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
)

# Load the full dataset, and download it if necessary
dataset = get_dataset(dataset="amazon", download=True)

# Get the training set
train_data = dataset.get_subset(
    "train",
)

# https://huggingface.co/LiYuan/amazon-review-sentiment-analysis
tokenizer = BertTokenizer.from_pretrained("LiYuan/amazon-review-sentiment-analysis")
model = BertForSequenceClassification.from_pretrained("LiYuan/amazon-review-sentiment-analysis")

