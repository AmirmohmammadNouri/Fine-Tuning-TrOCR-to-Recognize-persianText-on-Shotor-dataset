# Introduction

We will train the TrOCR model on this dataset and run inference again to analyze the results. This will provide us with a comprehensive idea of how far we can push the boundaries of the TrOCR models for different use cases.

## Steps

- Prepare and analyze the curved text images dataset.
- Load the TrOCR Small Printed model from Hugging Face.
- Initialize the Hugging Face Sequence to Sequence Trainer API.
- Define the evaluation metric
- Train the model and run inference.




## The Shotor Persian Text Dataset


Shotor (means camel in Persian) is a free synthetic dataset for Word Level OCR.

![Alt text](https://raw.githubusercontent.com/amirabbasasadi/Shotor/master/demo.png)

[Shotor Dataset GitHub Page](https://github.com/amirabbasasadi/Shotor)



## Installing and Importing Required Libraries

```python
import foobar

!pip install -q transformers
!pip install -q sentencepiece
!pip install -q jiwer
!pip install -q datasets
!pip install -q evaluate
!pip install -q -U accelerate
 
 
!pip install -q matplotlib
!pip install -q protobuf==3.20.1
!pip install -q tensorboard
```
- transformers: This is the Hugging Face transformers library that gives us access to hundreds of transformer based models including the TrOCR model. 
- sentencepiece: This is the sentencepiece tokenizer library that is needed to convert words into tokens and numbers. This is also part of the Hugging Face family.
- jiwer: The jiwer library gives us access to several speech recognition and language recognition metrics. These include WER (Word Error Rate) and CER (Character Error Rate). We will use the CER metric to evaluate the model while training.


## Defining Configurations

```python
import foobar

@dataclass(frozen=True)
class TrainingConfig:
    BATCH_SIZE:    int = 48
    EPOCHS:        int = 35
    LEARNING_RATE: float = 0.00005
 
@dataclass(frozen=True)
class DatasetConfig:
    DATA_ROOT:     str = 'shotor_data'
 
@dataclass(frozen=True)
class ModelConfig:
    MODEL_NAME: str = 'microsoft/trocr-small-printed'
```
The model will undergo 35 epochs of training using a batch size of 48.  The learning rate for the optimizer is set at 0.00005. Higher learning rates can make the training process unstable leading to higher loss from the beginning.

Please make sure to update tests as appropriate.

## Preparing the Dataset

```python
train_df = pd.read_fwf(
    os.path.join(DatasetConfig.DATA_ROOT, 'shotor_train.txt'), header=None
)
train_df.rename(columns={0: 'file_name', 1: 'text'}, inplace=True)
test_df = pd.read_fwf(
    os.path.join(DatasetConfig.DATA_ROOT, 'shotor_test.txt'), header=None
)
test_df.rename(columns={0: 'file_name', 1: 'text'}, inplace=True)
```

## defining the augmentations

The next step is defining the augmentations.

```python
# Augmentations.
train_transforms = transforms.Compose([
    transforms.ColorJitter(brightness=.5, hue=.3),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
])
```

We apply ColorJitter and GaussianBlur to the images. There is no need to apply any rotation of flipping to the images as there is already enough variability in the original dataset.

The best way to prepare the dataset is to write a custom dataset class. This allows us to have finer control over the inputs. The following code block defines a CustomOCRDataset class to prepare the dataset.


```python
class CustomOCRDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length
 
 
    def __len__(self):
        return len(self.df)
 
 
    def __getitem__(self, idx):
        # The image file name.
        file_name = self.df['file_name'][idx]
        # The text (label).
        text = self.df['text'][idx]
        # Read the image, apply augmentations, and get the transformed pixels.
        image = Image.open(self.root_dir + file_name).convert('RGB')
        image = train_transforms(image)
        pixel_values = self.processor(image, return_tensors='pt').pixel_values
        # Pass the text through the tokenizer and get the labels,
        # i.e. tokenized labels.
        labels = self.processor.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_target_length
        ).input_ids
        # We are using -100 as the padding token.
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding
```
## Optimizer and Evaluation Metric

For optimizing the model weights, we choose the AdamW optimizer with a weight decay of 0.0005.

```python
2
3
	
optimizer = optim.AdamW(
    model.parameters(), lr=TrainingConfig.LEARNING_RATE, weight_decay=0.0005
)

cer_metric = evaluate.load('cer')
 
 
def compute_cer(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
 
 
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
 
 
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
 
 
    return {"cer": cer}
```

Without elaborating further, CER is basically the number of characters that the model did not predict correctly. The lower the CER, the better the performance of the model.


## Training and Validation of TrOCR

The training arguments must be initialized before the training can begin.

```python
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy='epoch',
    per_device_train_batch_size=TrainingConfig.BATCH_SIZE,
    per_device_eval_batch_size=TrainingConfig.BATCH_SIZE,
    fp16=True,
    output_dir='seq2seq_model_printed/',
    logging_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=5,
    report_to='tensorboard',
    num_train_epochs=TrainingConfig.EPOCHS
)
```

```python
res = trainer.train()
```
