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
from sentence_transformers import SentenceTransformer
import numpy as np

st.set_page_config(
    page_title="Original-method Keyword Extractor",
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

c30, c31, c32 = st.columns([30, 1, 3])

with c30:
    # st.image("logo.png", width=400)
    st.title("Original-method Keyword Extractor")
    st.header("")



with st.expander("ℹ️ - About this app", expanded=True):

    st.write(
        """     
-   This is the implementation of Original-method's KPE algorithm introduced in WS.
	    """
    )

    st.markdown("")

st.markdown("")
st.markdown("## **📌 Paste document **")
with st.form(key="my_form"):


    ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
    with c1:
        top_N = st.slider(
            "# N of results",
            min_value=0,
            max_value=30,
            value=10,
            help="You can choose the number of keywords/keyphrases to display. Between 1 and 30, default number is 10. If you choose 0, it will be unlimited.",
        )
        threshold = st.slider(
            "# threshold",
            min_value=0,
            max_value=100,
            value=20,
            help="Set a lower bound for cosine similarity. Between 0(%) and 100(%), default number is 20(%).",
        )

    with c2:
        doc = st.text_area(
            "Paste your text below (max 1000 words)",
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

@st.cache(allow_output_mutation=True)
def alice_kpe(doc):
    #cos 類似度
    def cos_sim(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    #T5 KPE
    kp = KeyPhraseTransformer()
    keywords = kp.get_key_phrases(doc)

    #BERT
    model = SentenceTransformer('all-MiniLM-L6-v2')

    #Keywords emb
    kw_embeddings = model.encode(keywords)

    #main_sentence emb
    main_sentence = model.encode(doc)

    keywords_list = list()
    for sentence, embedding in zip(keywords, kw_embeddings):
        keyword_tuple = sentence , float(cos_sim(embedding, main_sentence))
        keywords_list.append(keyword_tuple)
    keywords_list.sort(key = lambda x: x[1],reverse = True) 
    keywords = keywords_list
    return keywords

keywords = alice_kpe(doc)
#threshold
keywords = [e for e in keywords if e[1] >= (threshold / 100)]
#TopN
if top_N != 0:
    keywords = keywords[0:top_N]

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

df = (
    DataFrame(keywords, columns=["Keyword/Keyphrase", "Relevancy"])
    .sort_values(by="Relevancy", ascending=False)
    .reset_index(drop=True)
)

df.index += 1

# Add styling
cmGreen = sns.light_palette("green", as_cmap=True)
cmRed = sns.light_palette("red", as_cmap=True)
df = df.style.background_gradient(
    cmap=cmGreen,
    subset=[
        "Relevancy",
    ],
)

c1, c2, c3 = st.columns([1, 3, 1])

format_dictionary = {
    "Relevancy": "{:.1%}",
}

df = df.format(format_dictionary)

with c2:
    st.table(df)
