from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel
import streamlit as st
import pandas as pd


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

def _max_width_():
    max_width_str = f"max-width: 1800px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )

st.set_page_config(page_icon="✂️", page_title="Logo or Not Logo")


c1, c2 = st.columns([1, 6])


with c2:

    st.caption("")
    st.title("Logo or Not Logo")

with c2:

    uploaded_file = st.file_uploader(
        " ",
        key="1",
        help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
        label_visibility='collapsed',
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)
        
        for index, row in df.iterrows():

            url = row['Logo']
            image = Image.open(requests.get(url, stream=True).raw)

            inputs = processor(

                text=["logo", "not logo"], images=image, return_tensors="pt", padding=True

            )

            outputs = model(**inputs)

            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score

            probs = logits_per_image.softmax(dim=1)
            if (probs[0][1] > 0.40):
                df.at[index, 'Logo'] = 'not Logo'
        
        file_container = st.expander("Check your uploaded .csv")
        file_container.write(df)
        st.stop()



