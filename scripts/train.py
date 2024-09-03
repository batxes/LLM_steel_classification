import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class DataLoader(Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Retrieve tokenized data for the given index
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Add the label for the given index to the item dictionary
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def prepare_data_for_training(df, tokenizer):

    # 60, 20, 20
    df_full_train, df_test = train_test_split(df, test_size=0.2, stratify=df['labels'], random_state=42)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, stratify=df_full_train['labels'], random_state=42)

    train_texts = df_train['text'].tolist()
    train_texts = [str(text) for text in train_texts]
    train_labels = df_train['labels'].tolist()

    val_texts = df_val['text'].tolist()
    val_texts = [str(text) for text in val_texts]
    val_labels = df_val['labels'].tolist()

    test_texts = df_test['text'].tolist()
    test_texts = [str(text) for text in test_texts]
    test_labels = df_test['labels'].tolist()

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings  = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataloader = DataLoader(train_encodings, train_labels)
    val_dataloader = DataLoader(val_encodings, val_labels)
    test_dataloader = DataLoader(test_encodings, test_labels)
    return train_dataloader, val_dataloader, test_dataloader



def compute_metrics(pred):

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }

def train(model, train_dataloader, val_dataloader, test_dataloader):

    training_args = TrainingArguments(
    # The output directory where the model predictions and checkpoints will be written
    output_dir='./Model_checkpoints', 
    do_train=True,
    do_eval=True,
    #  The number of epochs, defaults to 3.0 
    num_train_epochs=3,              
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=32,
    # Number of steps used for a linear warmup
    warmup_steps=100,                
    weight_decay=0.01,
    logging_strategy='steps',
   # TensorBoard log directory                 
    logging_dir='./multi-class-logs',            
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps", 
    fp16=True,
    load_best_model_at_end=True
    )

    trainer = Trainer(
    # the pre-trained model that will be fine-tuned 
    model=model,
     # training arguments that we defined above                        
    args=training_args,                 
    train_dataset=train_dataloader,         
    eval_dataset=val_dataloader,            
    compute_metrics= compute_metrics
    )
    trainer.train()

    q=[trainer.evaluate(eval_dataset=df_org) for df_org in [train_dataloader, val_dataloader, test_dataloader]]

    check = pd.DataFrame(q, index=["train","val","test"]).iloc[:,:5]

    print (check)
    return trainer

def save_model(trainer, tokenizer):
    model_path = "vanilla-classification-model"
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)