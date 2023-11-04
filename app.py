import os
import requests
import numpy as np

fp16_model_path = 'clip-vit-base-patch16.xml'
int8_model_path = 'clip-vit-base-patch16_int8.xml'

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
max_length = model.config.text_config.max_position_embeddings
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

from PIL import Image
from pathlib import Path
from scipy.special import softmax
from openvino.runtime import compile_model
from urllib.request import urlretrieve

from flask import Flask
app = Flask(__name__)

DREAMS = ['I am going to win this event and prove myself']

@app.route('/')
def build():
    url = 'https://versatilevats.com/openshift/labels.txt'
response = requests.get(url)


if response.status_code == 200:
    labels = response.text.split('\n')

    print(labels)

    sample_path = Path("data/coco.jpg")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(
        "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg",
        sample_path
    )
    image = Image.open(sample_path)
    
    input_labels = labels
    text_descriptions = [f"This is a photo of a {label}" for label in input_labels]
    
    inputs = processor(text=text_descriptions, images=[image], return_tensors="pt", padding=True)
    compiled_model = compile_model(int8_model_path)
    logits_per_image_out = compiled_model.output(0)
    ov_logits_per_image = compiled_model(dict(inputs))[logits_per_image_out]
    probs = softmax(ov_logits_per_image, axis=1)
    
    sorted_indices = np.argsort(probs[0])
    sorted_indices_reverse = sorted_indices[::-1]
    sorted_probs_reverse = probs[0][sorted_indices_reverse]
    
    #Making a dictionary
    label_to_value = dict(zip(input_labels, probs[0]))
    sorted_label_to_value = dict(sorted(label_to_value.items(), key=lambda item: item[1], reverse= True))
    first_3_results = dict(list(sorted_label_to_value.items())[:3])
    print(first_3_results)
    return first_3_results
  
@app.route('/dreams')
def dreams():
    return DREAMS
  
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
