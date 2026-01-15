import torch
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_PATH = "model/bert_model.pt"   # keep filename consistent

def load_emotion_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load model with correct number of labels
    num_labels = 6  # update this if your dataset changes
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

    # Load weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    return model, tokenizer, device
