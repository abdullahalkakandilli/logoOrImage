from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel
import pandas as pd

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")


df = pd.read_csv(r'C:\Users\alka\Masaüstü\imagesT.csv')

for index, row in df.iterrows():

    url = row['Logo']
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(

        text=["logo", "not logo"], images=image, return_tensors="pt", padding=True

    )

    outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score

    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    if (probs[0][1] > 0.40):
        df.at[index, 'Logo'] = 'not Logo'