from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel
import streamlit as st
import pandas as pd
from functionforDownloadButtons import download_button

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
    )

    if uploaded_file is not None:
        file_container = st.expander("Check your uploaded .csv")
        df = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)
        file_container.write(df)
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

    else:
        st.info(
            f"""
                👆 Upload a .csv file first. Sample to try: [biostats.csv](https://people.sc.fsu.edu/~jburkardt/data/csv/biostats.csv)
                """
        )

        st.stop()

cs, c1 = st.columns([2, 2])

# The code below is for the download button
# Cache the conversion to prevent computation on every rerun

with cs:

    @st.experimental_memo
    def convert_df(df):
        return df.to_csv().encode("utf-8")


    csv = convert_df(df)

    st.caption("")

    st.download_button(
        label="Download results",
        data=csv,
        file_name="classification_results.csv",
        mime="text/csv",
    )



