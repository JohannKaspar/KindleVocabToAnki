import os
import asyncio
from openai import OpenAI, AsyncOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt
import json

import datetime
import sqlite3
import tempfile
from typing import List

import altair as alt
import pandas as pd
import streamlit as st
from deep_translator import GoogleTranslator
from pydantic import BaseModel, Field


def get_data_from_vocab(db: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    """
    Extract the data from vocab.db and convert it into pandas DataFrame.

    Args:
        db: uploaded vocab.db

    Returns:
        extracted data.

    """
    with tempfile.NamedTemporaryFile() as fp:
        fp.write(db.getvalue())
        con = sqlite3.connect(fp.name)

    cur = con.cursor()

    sql = """
        SELECT WORDS.word, WORDS.stem, WORDS.lang, LOOKUPS.usage, BOOK_INFO.title, BOOK_INFO.authors, LOOKUPS.timestamp
          FROM LOOKUPS
          LEFT JOIN WORDS
            ON WORDS.id = LOOKUPS.word_key
          LEFT JOIN BOOK_INFO
            ON BOOK_INFO.id = LOOKUPS.book_key
         ORDER BY WORDS.stem, LOOKUPS.timestamp
    """

    cur.execute(sql)
    data_sql = cur.fetchall()
    data = pd.DataFrame(
        data_sql, columns=['Word', 'Stem', 'Word language', 'Sentence', 'Book title', 'Authors', 'Timestamp']
    )
    data['Timestamp'] = data['Timestamp'].apply(
        lambda t: datetime.datetime.fromtimestamp(t / 1000).strftime('%Y-%m-%d %H:%M:%S')
    )
    data = data.sort_values('Timestamp').reset_index(drop=True)
    return data


@retry(wait=wait_exponential(multiplier=1, min=2, max=60), stop=stop_after_attempt(5))
def translate_with_gpt4o_mini(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate text using GPT-4o Mini model.
    
    Args:
        text: Text to translate
        source_lang: Source language
        target_lang: Target language
        
    Returns:
        Translated text
    """
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using GPT-4o Mini model
            messages=[
                {"role": "system", "content": f"You are a translator from {source_lang} to {target_lang}. Translate the text accurately, preserving the meaning and context."},
                {"role": "user", "content": f"Translate this text from {source_lang} to {target_lang}: {text}"}
            ],
            temperature=0.2,  # Lower temperature for more consistent translations
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"GPT-4o Mini translation error: {e}")
        # Fallback to Google Translate
        return GoogleTranslator(source=source_lang, target=target_lang).translate(text)


class TranslationResult(BaseModel):
    translated_word: str = Field(..., description="The translation of the word/phrase into the target language (use standard/base form)")
    sentence_with_blank: str = Field(..., description="The example sentence with the word/phrase replaced by a blank")
    synonyms: list[str] = Field(..., description="2-3 synonyms for the original word/phrase in the original language (use standard forms)")
    disambiguation: str = Field(..., description="Disambiguation between the original word/phrase and its synonyms, with usage notes. Use only HTML formatting (like <b>word</b>), never markdown.")
    original_word: str = Field(..., description="The original word/phrase in standard form (infinitive for verbs, singular for nouns)")


async def translate_single_item_async(client: AsyncOpenAI, item: tuple, lang: str) -> dict:
    """
    Translate a single item asynchronously using GPT-4o Mini.
    """
    if len(item) >= 4:
        text_lang, text, word, book_title = item
    else:
        text_lang, text, word = item
        book_title = ""
    
    # Remove the word from the sentence for the cloze
    sentence_with_blank = text.replace(word, "___")
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    f"You are an expert linguist and language teacher helping someone learn vocabulary using Anki cards. "
                    f"You excel at detecting phrasal verbs, collocations, and multi-word expressions in context. "
                    f"You always provide translations in standard forms (infinitive verbs, singular nouns). "
                    f"You ONLY use HTML formatting (like <b>word</b>) and NEVER use markdown formatting (like **word**). "
                    f"Given a sentence and a target word, return a JSON object matching this Pydantic model: "
                    f"{TranslationResult.schema_json()}"
                )},
                {"role": "user", "content": (
                    f"Target word: '{word}'\n"
                    f"Sentence: '{text}'\n"
                    f"IMPORTANT INSTRUCTIONS:\n"
                    f"1. SMART PHRASE DETECTION: Look for phrasal verbs, collocations, and multi-word expressions around '{word}'. "
                    f"   Examples: 'abide by' → 'sich halten an', 'look up' → 'nachschlagen', 'give up' → 'aufgeben'\n"
                    f"2. STANDARD FORM: Always use the base/infinitive form of verbs and singular form of nouns in both languages.\n"
                    f"   Examples: 'abided' → 'abide', 'running' → 'run', 'children' → 'child'\n"
                    f"3. CONTEXT-AWARE TRANSLATION: Consider the full phrase or collocation, not just the isolated word.\n\n"
                    f"Based on the sentence context, determine if '{word}' is part of a larger phrase or expression. "
                    f"If so, translate the complete phrase/expression into {lang} as 'translated_word'. "
                    f"If it's a standalone word, translate just the word but use its standard form (infinitive/singular). "
                    f"Return the standard form of the original word/phrase as 'original_word'. "
                    f"Return the sentence with the word/phrase replaced by a blank as 'sentence_with_blank'. "
                    f"Return 2-3 synonyms for the original word/phrase (in {text_lang}) as 'synonyms'. "
                    f"Return a disambiguation section as 'disambiguation', explaining when to use the original word/phrase vs. the synonyms. "
                    f"IMPORTANT: Use only HTML formatting (like <b>word</b>) in ALL fields, never use markdown formatting like **word**. "
                    f"Format all fields for direct use in an Anki card with HTML only."
                )}
            ],
            temperature=0.2,
            max_tokens=400,
            response_format={"type": "json_object"}
        )
        result = response.choices[0].message.content
        result_dict = json.loads(result)
        # Validate and coerce using Pydantic
        validated = TranslationResult(**result_dict).dict()
        
        # Post-process to ensure no markdown formatting remains
        for key in ['disambiguation', 'translated_word', 'original_word', 'sentence_with_blank']:
            if key in validated and validated[key]:
                # Convert markdown bold to HTML bold using regex
                import re
                validated[key] = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', validated[key])
        
        validated["book_title"] = book_title
        return validated
    except Exception as e:
        print(f"GPT-4o Mini translation error: {e}")
        return {
            "translated_word": "",
            "sentence_with_blank": sentence_with_blank,
            "synonyms": [],
            "disambiguation": "",
            "original_word": word,
            "book_title": book_title
        }


async def translate_with_context_async(data: List, lang: str) -> list[dict]:
    """
    Enhanced async translation using GPT-4o Mini with structured output (Pydantic schema).
    Args:
        data: List of tuples with (text_lang, text, word, book_title)
        lang: Target language for translation
    Returns:
        List of dictionaries with comprehensive translation data
    """
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    
    # Create semaphore to limit concurrent requests (adjust based on API limits)
    semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent requests
    
    async def translate_with_semaphore(item):
        async with semaphore:
            return await translate_single_item_async(client, item, lang)
    
    # Create progress tracking
    total_items = len(data)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create all tasks
    tasks = [translate_with_semaphore(item) for item in data]
    
    # Process tasks with progress updates
    results = []
    for i, task in enumerate(asyncio.as_completed(tasks)):
        result = await task
        results.append(result)
        
        # Update progress
        progress = (i + 1) / total_items
        progress_bar.progress(progress)
        status_text.text(f"Translating... {i + 1}/{total_items}")
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return results


def translate_with_context(data: List, lang: str) -> list[dict]:
    """
    Sync wrapper for the async translation function.
    Enhanced translation using GPT-4o Mini with structured output (Pydantic schema).
    Args:
        data: List of tuples with (text_lang, text, word, book_title)
        lang: Target language for translation
    Returns:
        List of dictionaries with comprehensive translation data
    """
    # Run the async function in the event loop
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an event loop (like in Streamlit), 
            # we need to run in a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, translate_with_context_async(data, lang))
                return future.result()
        else:
            # If no event loop is running, we can run directly
            return loop.run_until_complete(translate_with_context_async(data, lang))
    except RuntimeError:
        # If no event loop exists, create one
        return asyncio.run(translate_with_context_async(data, lang))


def make_more_columns(data: pd.DataFrame, lang: str, to_translate: List[str], translate_option: str) -> pd.DataFrame:
    """
    Create additional columns with enhanced translation data using GPT-4o Mini context-aware output.
    Deduplicate words (keep first occurrence), use HTML formatting, and handle partial/failed translations gracefully.
    """
    # (removed) do not deduplicate here, keep each word–book row intact

    # If previous translations exist, reuse them
    prev_results = getattr(st.session_state, 'translated_df', None)
    prev_map = {}
    if prev_results is not None and 'Word' in prev_results.columns:
        for _, row in prev_results.iterrows():
            prev_map[row['Word']] = row

    # Prepare translation input and results
    translation_input = []
    for idx, row in data.iterrows():
        word = row['Word']
        prev = prev_map.get(word)
        if prev is not None and all(
            prev.get(col, '') for col in ['translated_word', 'sentence_with_blank', 'synonyms', 'disambiguation', 'original_word']
        ):
            # Reuse previous translation if all fields are non-empty
            translation_input.append(None)  # Mark for reuse
        else:
            translation_input.append((row['Word language'], row['Sentence'], row['Word'], row.get('Book title', '')))

    # Batch translate only the needed rows
    to_translate_items = [v for v in translation_input if v is not None]
    batch_results = translate_with_context(to_translate_items, lang) if to_translate_items else []

    # Merge reused and new results
    translation_results = []
    batch_idx = 0
    for list_idx, (idx, row) in enumerate(data.iterrows()):
        word = row['Word']
        prev = prev_map.get(word)
        if translation_input[list_idx] is None and prev is not None:
            translation_results.append({
                'translated_word': prev.get('translated_word', ''),
                'sentence_with_blank': prev.get('sentence_with_blank', ''),
                'synonyms': prev.get('synonyms', ''),
                'disambiguation': prev.get('disambiguation', ''),
                'original_word': prev.get('original_word', ''),
                'book_title': row.get('Book title', ''),
            })
        else:
            result = batch_results[batch_idx]
            translation_results.append({
                'translated_word': result.get('translated_word', ''),
                'sentence_with_blank': result.get('sentence_with_blank', ''),
                'synonyms': ', '.join(result.get('synonyms', [])) if isinstance(result.get('synonyms'), list) else '',
                'disambiguation': result.get('disambiguation', ''),
                'original_word': result.get('original_word', ''),
                'book_title': row.get('Book title', ''),
            })
            batch_idx += 1

    # Assign results to DataFrame
    data['translated_word'] = [r['translated_word'] for r in translation_results]
    data['sentence_with_blank'] = [r['sentence_with_blank'] for r in translation_results]
    data['synonyms'] = [r['synonyms'] for r in translation_results]
    data['disambiguation'] = [r['disambiguation'] for r in translation_results]
    data['original_word'] = [r['original_word'] for r in translation_results]

    # Ensure 'book_title' column exists for HTML formatting
    if 'Book title' in data.columns:
        data['book_title'] = data['Book title']
    else:
        data['book_title'] = ''

    # HTML formatting for Anki
    def html_anki_front(row):
        return (
            f"<div style='font-size:1.5em;'><b>{row['translated_word']}</b></div>"
            f"<div style='margin:1em 0;'>{row['sentence_with_blank']}</div>"
            f"<div style='color:#555;'>syn. <b>{row['synonyms']}</b></div>"
            f"<div style='font-size:small;color:gray;'>Book: {row['book_title']}</div>"
        )
    def html_anki_back(row):
        return (
            f"<div style='font-size:1.5em;'><b>{row['original_word']}</b></div>"
            f"<div style='margin:1em 0;'>{row['disambiguation']}</div>"
        )
    data['anki_front'] = data.apply(html_anki_front, axis=1)
    data['anki_back'] = data.apply(html_anki_back, axis=1)

    st.session_state.translated_df = data.reset_index(drop=True)
    return data


def show_vocabulary_stats(df: pd.DataFrame) -> None:
    """
    Show various statistics based on the data.

    Args:
        df: dataframe with data

    Returns:
        Nothing
    """
    df['date'] = pd.to_datetime(df['Timestamp']).dt.date
    df['Year-month'] = pd.to_datetime(df['Timestamp']).dt.strftime('%Y-%m')
    d = df.groupby('Year-month')['Sentence'].count().sort_index().reset_index()
    d['count'] = d['Sentence'].cumsum()

    chart = (
        alt.Chart(d)
        .mark_line(point=True, strokeWidth=3)
        .encode(x=alt.X('Year-month:T', timeUnit='yearmonth'), y='count:Q')
        .configure_point(size=20)
        .properties(title='Number of word over time')
        .configure_point(size=50)
        .interactive()
    )

    pie_df = df['Book title'].value_counts().reset_index().head(5)

    base = alt.Chart(pie_df).encode(
        alt.Theta('count:Q').stack(True), alt.Color('Book title:N'), tooltip=['Book title', 'count']
    )
    pie = base.mark_arc(outerRadius=120).properties(title='Number of words in top 5 books').interactive()
    text = base.mark_text(radius=140, size=12).encode(text='count:Q')

    st.altair_chart(chart, use_container_width=True)
    st.altair_chart(pie + text, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label='Word count', value=df.shape[0], help='Unique word count in the vocabulary')
    col2.metric(label='Book count', value=df['Book title'].nunique(), help='Book count in the vocabulary')
    col3.metric(
        label='Language count',
        value=df['Word language'].nunique(),
        help='Language count in the vocabulary',
    )
    col4.metric(
        label='Days with looked up works', value=df['date'].nunique(), help='Days with at least one word looked up'
    )
