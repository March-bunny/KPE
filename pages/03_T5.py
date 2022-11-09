import streamlit as st
import numpy as np
from pandas import DataFrame
from keybert import KeyBERT
# For Flair (Keybert)
from flair.embeddings import TransformerDocumentEmbeddings
import seaborn as sns
# For download buttons
from pages.func.functionforDownloadButtons import download_button
import os
import json
from keyphrase_vectorizers import KeyphraseCountVectorizer
from keyphrasetransformer import KeyPhraseTransformer

st.set_page_config(
    page_title="T5 Keyword Extractor",
    page_icon="🎈",
)


def _max_width_():
    max_width_str = f"max-width: 1400px;"
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


_max_width_()

c30, c31, c32 = st.columns([10, 1, 3])

with c30:
    # st.image("logo.png", width=400)
    st.title("T5 Keyword Extractor")
    st.header("")



with st.expander("ℹ️ - About this app", expanded=True):

    st.write(
        """     
-   This is an implementation of T5 Keyword Extractor.
	    """
    )

    st.markdown("")

st.markdown("")
st.markdown("## **📌 Paste document **")
with st.form(key="my_form"):


    ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])

    with c2:
        doc = st.text_area(
            "Paste your text below (max 500 words)",
            height=510,
        )

        MAX_WORDS = 1000
        import re
        res = len(re.findall(r"\w+", doc))
        if res > MAX_WORDS:
            st.warning(
                "⚠️ Your text contains "
                + str(res)
                + " words."
                + " Only the first 1000 words will be reviewed. Stay tuned as increased allowance is coming! 😊"
            )

            doc = doc[:MAX_WORDS]

        submit_button = st.form_submit_button(label="✨ Get me the data!")

if not submit_button:
    st.stop()

kp = KeyPhraseTransformer()
keywords = kp.get_key_phrases(doc)

st.markdown("## **🎈 Check & download results **")

st.header("")

cs, c1, c2, c3, cLast = st.columns([2, 1.5, 1.5, 1.5, 2])

with c1:
    CSVButton2 = download_button(keywords, "Data.csv", "📥 Download (.csv)")
with c2:
    CSVButton2 = download_button(keywords, "Data.txt", "📥 Download (.txt)")
with c3:
    CSVButton2 = download_button(keywords, "Data.json", "📥 Download (.json)")

st.header("")

df = DataFrame(keywords, columns=["Keyword/Keyphrase"])

#df.index += 1

# Add styling
cmGreen = sns.light_palette("green", as_cmap=True)
cmRed = sns.light_palette("red", as_cmap=True)

c1, c2, c3 = st.columns([1, 3, 1])

with c2:
    st.dataframe(df)
