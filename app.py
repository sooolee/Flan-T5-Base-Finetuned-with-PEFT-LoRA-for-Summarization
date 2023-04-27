import os
import gradio as gr
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from youtube_transcript_api import YouTubeTranscriptApi

# def load_data(file_obj):
#     """
#     Load data from the file object of the gr.File() inputs
#     """
#     path = file_obj.name
#     with open(path, "r") as f:
#         data = f.read()

#     return data

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
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, device_map='auto') # load_in_8bit=True,
model = PeftModel.from_pretrained(model, peft_model_id, device_map='auto')

def summarize(video_id):
    # transcript = load_data(file_obj)
    dict = YouTubeTranscriptApi.get_transcript(video_id)
        
    transcript = ""

    for i in range(len(dict)):
        transcript += dict[i]['text']

    texts = preprocessing(transcript)
    inputs = tokenizer(texts, return_tensors="pt", padding=True, )

    with torch.no_grad():
        output_tokens = model.generate(input_ids=inputs["input_ids"].to(device), max_new_tokens=60, do_sample=True, top_p=0.9)
        outputs = tokenizer.batch_decode(output_tokens.detach().cpu().numpy(), skip_special_tokens=True)
    
    return outputs

gr.Interface(
    fn=summarize,
    title = 'Summarize Transcripts',
    # inputs = gr.File(file_types=["text"], label="Upload a text file.", interactive=True),
    inputs = gr.Textbox(label="Video_ID", interactive=True),
    outputs = gr.Textbox(label="Summary", max_lines=120, interactive=False),
).launch(debug=True)
