import streamlit as st
import os

st.set_page_config(
    page_title='Kindle Vocabulary to Anki converter',
    page_icon='📚',
)

st.title('📚 Kindle Vocabulary to Anki converter')

# Check for OpenAI API key
api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    st.error("⚠️ **OpenAI API Key Required**")
    st.markdown("""
    This app uses GPT-4o Mini for intelligent, context-aware translations. To use the app, you need to set up your OpenAI API key:
    
    **Option 1: Environment Variable (Recommended)**
    ```bash
    export OPENAI_API_KEY="your-api-key-here"
    ```
    
    **Option 2: Add to your shell profile**
    Add the following line to your `~/.zshrc` or `~/.bash_profile`:
    ```bash
    export OPENAI_API_KEY="your-api-key-here"
    ```
    
    **Get your API key:**
    1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
    2. Sign in to your account
    3. Create a new API key
    4. Set the environment variable and restart this app
    
    **Cost:** GPT-4o Mini is very affordable - typically costs less than $0.01 per 100 words translated.
    """)
    st.stop()
else:
    st.success("✅ OpenAI API key detected - ready to translate!")

st.markdown("""
### 🚀 **Modern AI-Powered Translation**
This app converts Kindle vocabulary files into rich Anki flashcards using **GPT-4o Mini** for intelligent, context-aware translations.

### ✨ **Key Features:**
- **Smart phrase detection**: Automatically detects phrasal verbs and collocations (e.g., "abide by" → "sich halten an")
- **Context-aware translation**: Uses the full sentence context for accurate translations
- **Rich card format**: Includes synonyms, disambiguation notes, and example sentences
- **Standard word forms**: Converts to infinitive verbs and singular nouns
- **HTML formatting**: Ready-to-import Anki cards with proper styling

### 📱 **Getting Your Kindle Data:**
To get started, you need the `vocab.db` file from your Kindle device:

1. Connect your Kindle device to your computer via USB cable
2. Navigate to `Kindle/system/vocabulary/` on your Kindle
3. Copy the `vocab.db` file to your computer
4. Upload it using the file uploader in **Step 1**

### 🔒 **Privacy:**
- Your data is processed locally and not stored permanently
- Only vocabulary words are sent to OpenAI for translation
- All data is cleared when your session ends

### 🔗 **More Information:**
- **Project:** [GitHub Repository](https://github.com/Erlemar/KindleVocabToAnki)
- **Blog:** [andlukyane.com/blog/kindlevocabtoanki](https://andlukyane.com/blog/kindlevocabtoanki)

---
**Ready to start?** Navigate to **Step 1: Data Upload** in the sidebar to begin! 👆
""")

# If you want to show navigation, use the sidebar or rely on Streamlit's built-in multipage support.
# Remove show_pages and rely on the new Streamlit navigation.
