# Flan-T5-Base Finetuned with PEFT LoRA for Chat & Dialogue Summarization

This project used the following blog as a reference: https://www.philschmid.de/fine-tune-flan-t5-peft

Instead of finetuning 'flan-t5-base' as is, I chose to further finetune a finetuned model 'braindao/flan-t5-cnn' on chat & dialogue SAMSum dataset. 'braindao/flan-t5-cnn' had been already finetuned on the CNN Dailymail dataset. 


## Demo
Please visit my Hugging Face Space for Gradio API, which takes YouTube Videio_ID and gives a summary: [sooolee/summarize-transcripts-gradio](https://huggingface.co/spaces/sooolee/summarize-transcripts-gradio)

You can watch my quick demo [video](https://www.loom.com/share/8c7f3dbbf4964e46bf13350a19b3ca6f).  

## Model Description
* This model further finetuned 'braindao/flan-t5-cnn' on the more conversational samsum dataset.
* Huggingface PEFT Library LoRA (r = 16) and bitsandbytes int-8 was used to speed up training and reduce the model size.
* Only 1.7M parameters were trained (0.71% of original flan-t5-base 250M parameters).
* The model checkpoint is just 7MB.

## Intended Uses & Limitations
* Summarize transcripts such as YouTube transcripts.
* While the model size is small, if you want to process a long transcript (say, more than 15-20 minutes worth of talk), it will still require a small GPU like T4. 

## Training Hyperparameters
The following hyperparameters were used during training:

- learning_rate: 0.001
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 5

## Training Results
- train_loss: 1.47

## Evaluation Results
- rogue1: 46.819522%
- rouge2: 20.898074%
- rougeL: 37.300937%
- rougeLsum: 37.271341%


## How to use
* [model in hugging space](https://huggingface.co/sooolee/flan-t5-base-cnn-samsum-lora)
* Note 'max_new_tokens=60' is used in the below example to control the length of the summary. FLAN-T5 model has max generation length = 200 and min generation length = 20 (default).

```python
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load peft config for pre-trained checkpoint etc.
peft_model_id = "sooolee/flan-t5-base-cnn-samsum-lora"
config = PeftConfig.from_pretrained(peft_model_id)

# load base LLM model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, device_map='auto') # load_in_8bit=True, 
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id, device_map='auto')

# Tokenize the text inputs
texts = "<e.g. Part of YouTube Transcript>"
inputs = tokenizer(texts, return_tensors="pt", padding=True, ) # truncation=True

# Make inferences
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with torch.no_grad():    
    output = self.model.generate(input_ids=inputs["input_ids"].to(device), max_new_tokens=60, do_sample=True, top_p=0.9)
    summary = self.tokenizer.batch_decode(output.detach().cpu().numpy(), skip_special_tokens=True)

summary
```

## Other
I further finetuned BART-Large-CNN as well on the same SAMSum datasets for the same purpose. Please check out [sooolee/bart-large-cnn-finetuned-samsum-lora](https://huggingface.co/sooolee/bart-large-cnn-samsum-lora) the model fine-tuned 
