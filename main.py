from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel
import streamlit as st
import pandas as pd
from functionforDownloadButtons import download_button

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
        df = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)


        file_container = st.expander("Check your uploaded .csv")
        file_container.write(df)


    else:
        st.info(
            f"""
                👆 Upload a .csv file first. Sample to try: [biostats.csv](https://people.sc.fsu.edu/~jburkardt/data/csv/biostats.csv)
                """
        )

        st.stop()


def get_values(column_names):

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    df_after = pd.DataFrame()

    for index, row in df.iterrows():

        url = row[column_names]
        image = Image.open(requests.get(url, stream=True).raw)

        inputs = processor(

            text=["logo", "not logo"], images=image, return_tensors="pt", padding=True

        )

        outputs = model(**inputs)

        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score

        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        if (column_names == 'Logo'):
            if (probs[0][1] > 0.40):
                df.at[index, column_names] = 'not Logo'
                df_after.append(df.index)
        else:
            if (probs[0][1] < 0.60):
                df.at[index, column_names] = 'not Image'
                df_after.append(df.index)
    return
#df = final result
form = st.form(key="annotation")
with form:

    column_names = st.selectbox(
        "Column name:", list(df.columns)
    )

    submitted = st.form_submit_button(label="Submit")
result_df = pd.DataFrame()
if submitted:

    result = get_values(column_names)


c29, c30, c31 = st.columns([1, 1, 2])

with c29:

    CSVButton = download_button(
        df,
        "File.csv",
        "Download to CSV",
    )
