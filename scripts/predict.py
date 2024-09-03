from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from transformers import pipeline, BertTokenizer, BertForSequenceClassification


def predict(text, model, tokenizer):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")

    outputs = model(**inputs)

    probs = outputs[0].softmax(1)

    """ Explanation outputs: The BERT model returns a tuple containing the output logits (and possibly other elements depending on the model configuration). In this case, the output logits are the first element in the tuple, which is why we access it using outputs[0].
    outputs[0]: This is a tensor containing the raw output logits for each class. The shape of the tensor is (batch_size, num_classes) where batch_size is the number of input samples (in this case, 1, as we are predicting for a single input text) and num_classes is the number of target classes.
    softmax(1): The softmax function is applied along dimension 1 (the class dimension) to convert the raw logits into class probabilities. Softmax normalizes the logits so that they sum to 1, making them interpretable as probabilities. """

    pred_label_idx = probs.argmax()
    pred_label = model.config.id2label[pred_label_idx.item()]

    return probs, pred_label_idx, pred_label

if __name__ == "__main__":

    model_path = "vanilla-classification-model"
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer= BertTokenizer.from_pretrained(model_path)
    nlp= pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    # hard coded examples
    text = "ungebeizt, nicht geglüht"
    prediction = nlp(text)
    print ("{} -> {}".format(text, prediction))

    text = "C100S"
    prediction = nlp(text)
    print ("{} -> {}".format(text, prediction))