import requests
import json
import streamlit as st

def get_gemini_sentiment(text_to_analyze, api_key):
    """
    Function to call Gemini Pro (Flash) for sentiment analysis.
    This makes an actual API call to Gemini.
    """
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    chat_history = []
    # Refined prompt for sentiment analysis
    prompt = f"Analyze the sentiment of the following financial news headline or article snippet. Respond with a single word: 'Positive', 'Negative', or 'Neutral'. Do not add any other text.\n\nText: '{text_to_analyze}'"
    chat_history.append({ "role": "user", "parts": [{ "text": prompt }] })
    
    payload = { "contents": chat_history }
    headers = { 'Content-Type': 'application/json' }

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        result = response.json()
        
        if result and 'candidates' in result and len(result['candidates']) > 0 and \
           'content' in result['candidates'][0] and 'parts' in result['candidates'][0]['content'] and \
           len(result['candidates'][0]['content']['parts']) > 0:
            sentiment = result['candidates'][0]['content']['parts'][0]['text'].strip()
            # Basic validation for expected sentiment words
            sentiment_score = 0.0
            if sentiment == "Positive":
                sentiment_score = 1.0
            elif sentiment == "Negative":
                sentiment_score = -1.0

            return {"sentiment_label": sentiment, "sentiment_score": sentiment_score, "summary": f"Sentiment analysis: {sentiment}"}
        else:
            st.warning("Unexpected Gemini API response structure. Defaulting to Neutral.")
            return {"sentiment_label": "Neutral", "sentiment_score": 0.0, "summary": "Sentiment analysis: Neutral"}
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Gemini API for sentiment: {e}. Defaulting to Neutral.")
        return {"sentiment_label": "Neutral", "sentiment_score": 0.0, "summary": f"Error: {e}"}
    except Exception as e:
        st.error(f"An unexpected error occurred during Gemini sentiment analysis: {e}. Defaulting to Neutral.")
        return {"sentiment_label": "Neutral", "sentiment_score": 0.0, "summary": f"Error: {e}"}