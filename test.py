
import pandas as pd


df = pd.read_csv(r'C:\Users\alka\Masaüstü\imagesT.csv')

for index, row in df.iterrows():
    print(row['Logo'])

# The code below is for the download button
# Cache the conversion to prevent computation on every rerun

with c2:

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
