import time
import os

import pandas as pd
import streamlit as st
import openai

from src.utils import make_more_columns

st.set_page_config(page_title="🔄 Step 2 Data Translate", page_icon="🔄")

st.subheader('🔄 Define translation parameters')
my_expander2 = st.expander(label='🛠️ Translation parameters', expanded=True)

# Add API Key input for OpenAI
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password", 
                               help="Required for GPT-4o Mini translation")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = api_key

# Only GPT-4o Mini is available
translation_engine = "GPT-4o Mini"
st.session_state.translation_engine = translation_engine

if 'loaded_data' in st.session_state and st.session_state.loaded_data.shape[0] > 0:
    st.session_state.data = st.session_state.loaded_data.copy()
    with my_expander2:
        # limit the number of rows
        col1_, col2_ = st.columns(2)
        with col1_:
            top_n: int = int(
                st.number_input(
                    'Take last N rows',
                    min_value=1,
                    max_value=st.session_state.data.shape[0],
                    value=st.session_state.data.shape[0],
                )
            )
        with col2_:
            col_by = st.selectbox(
                'Sort data by', options=['Timestamp', 'Word'], help='Select the column to sort the data by'
            )

        st.session_state.data = st.session_state.data.sort_values(col_by)[-top_n:]
        d = st.date_input(
            label='Starting date',
            value=pd.to_datetime(st.session_state.data['Timestamp']).dt.date.min(),
            min_value=pd.to_datetime(st.session_state.data['Timestamp']).dt.date.min(),
            max_value=pd.to_datetime(st.session_state.data['Timestamp']).dt.date.max(),
            help='Change this value if you want to limit the st.session_state.data by the start date',
        )
        st.session_state.data = st.session_state.data.loc[
            pd.to_datetime(st.session_state.data['Timestamp']).dt.date >= d
        ]
        
        # Language selection for GPT-4o Mini
        col1__, col2__ = st.columns(2)
        langs_list = [
            "German", "English", "Spanish", "French", "Italian", "Portuguese", 
            "Russian", "Japanese", "Chinese", "Korean", "Arabic", "Dutch", 
            "Hindi", "Swedish", "Turkish", "Polish", "Danish", "Norwegian"
        ]
        with col1__:
            lang = st.selectbox('Lang to translate into', options=langs_list, index=0)
        
        translate_options = ['Use context', 'Word only']
        with col2__:
            translate_option = st.selectbox(
                'Word translation style',
                options=translate_options,
                help='Translate the word by itself or use the whole phrase as a context',
            )

        to_translate = st.multiselect(
            label='What to translate (select one or multiple)',
            options=['Word', 'Stem', 'Sentence'],
            default=['Word'],
            help='Select the columns that will be translated',
        )
        
        # Update the text to show which translation service will be used
        translation_service = "GPT-4o Mini"
        
        # ...existing code for filters and preview...
        books = st.multiselect(
            label='Filter by books',
            options=st.session_state.data['Book title'].unique(),
            default=st.session_state.data['Book title'].unique(),
            help='Select the books that will be translated',
        )
        if len(books) > 0:
            st.session_state.data = st.session_state.data.loc[st.session_state.data['Book title'].isin(books)]
        authors = st.multiselect(
            label='Filter by authors',
            options=st.session_state.data['Authors'].unique(),
            default=st.session_state.data['Authors'].unique(),
            help='Select the Authors that will be translated',
        )
        if len(authors) > 0:
            st.session_state.data = st.session_state.data.loc[st.session_state.data['Authors'].isin(authors)]

        langs_from = st.multiselect(
            label='Languages to translate',
            options=st.session_state.data['Word language'].unique(),
            default=st.session_state.data['Word language'].unique(),
            help='Select the languages that will be translated',
        )
        if len(langs_from) > 0:
            st.session_state.data = st.session_state.data.loc[st.session_state.data['Word language'].isin(langs_from)]

        st.write(f'{st.session_state.data.shape[0]} texts will be translated (using {translation_service})')
        st.session_state.loaded_data = st.session_state.data
        st.dataframe(
            st.session_state.data.reset_index(drop=True).drop(
                [col for col in st.session_state.data.columns if 'with' in col or 'translated' in col], axis=1
            )
        )
    if st.session_state.data is None:
        st.session_state.data = st.session_state.loaded_data

    st.session_state.translate = st.button(
        'Translate', on_click=make_more_columns, args=(st.session_state.data, lang, to_translate, translate_option)
    )

    if st.session_state.translate or st.session_state.load_state:
        time.sleep(1)

        st.session_state.load_state = True
        translated_data = st.session_state.translated_df
        st.success('Translation finished!', icon='✅')

        st.subheader("Enhanced Translation Results")
        tab1, tab2 = st.tabs(["Card Preview", "Full Data"])
        with tab1:
            for i, row in translated_data.head(3).iterrows():
                with st.expander(f"Card {i+1}: {row['Word']} → {row.get('translated_word', '')}"):
                    cols = st.columns(2)
                    with cols[0]:
                        st.markdown("**Original**")
                        st.markdown(f"- Word: **{row['Word']}**")
                        st.markdown(f"- Sentence: {row['Sentence']}")
                        st.markdown(f"- Source: {row.get('Book title', 'Unknown')}")
                    with cols[1]:
                        st.markdown("**Translation**")
                        st.markdown(f"- Translated word: **{row.get('translated_word', '')}**")
                        st.markdown(f"- Sentence with blank: {row.get('sentence_with_blank', '')}")
                        if 'synonyms' in row and row['synonyms']:
                            st.markdown(f"- Synonyms: {row['synonyms']}")
                        if 'disambiguation' in row and row['disambiguation']:
                            st.markdown(f"- Usage notes: {row['disambiguation']}")
                    st.markdown("---")
                    st.markdown("**Anki Card Preview**")
                    cols_preview = st.columns(2)
                    with cols_preview[0]:
                        st.markdown("**Front:**")
                        st.markdown(row.get('anki_front', ''), unsafe_allow_html=True)
                    with cols_preview[1]:
                        st.markdown("**Back:**")
                        st.markdown(row.get('anki_back', ''), unsafe_allow_html=True)
        with tab2:
            st.dataframe(translated_data)
else:
    st.write('You need to upload some data in order to translate it.')
