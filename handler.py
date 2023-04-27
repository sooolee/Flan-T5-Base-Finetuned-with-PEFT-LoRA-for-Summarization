from typing import  List, Any
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from youtube_transcript_api import YouTubeTranscriptApi

def preprocessing(data):          
    texts = list()
    i = 0
    if len(data) <= i+3000:
        texts = data
    else:
        while len(data[i:]) != 0:
            if len(data[i:]) > 3000:
                string = str(data[i:i+3000])
                texts.append(string)
                i = i + 2800
            else:
                string = str(data[i:])
                texts.append(string)
                break    
    return texts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
peft_model_id = "sooolee/flan-t5-base-cnn-samsum-lora"
config = PeftConfig.from_pretrained(peft_model_id)

class EndpointHandler:
    def __init__(self, path=""):
        # load model and tokenizer from path
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        self.model = PeftModel.from_pretrained(self.model, path, device_map='auto')

    def __call__(self, data: Any) -> List[str]:
        
        video_id = data.pop("inputs", data)
        dict = YouTubeTranscriptApi.get_transcript(video_id)
        
        transcript = ""

        for i in range(len(dict)):
            transcript += dict[i]['text']

        # process input
        texts = preprocessing(transcript)
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, ) # truncation=True
        
        with torch.no_grad():    
            output = self.model.generate(input_ids=inputs["input_ids"].to(device), max_new_tokens=60, do_sample=True, top_p=0.9)
            summary = self.tokenizer.batch_decode(output.detach().cpu().numpy(), skip_special_tokens=True)
        
        return summary