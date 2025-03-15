import torch
import streamlit as st
import warnings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import speech_recognition as sr
import pyttsx3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime
import wikipedia

# Suppress unnecessary warnings
warnings.filterwarnings("ignore")

# Set wide layout for Streamlit
st.set_page_config(layout="wide")

# App title and description
st.title("Dynamic Intent Chatbot with RAG")
st.markdown("Chat with a bot that adapts to any intent using conversation memory and external knowledge!")

# Detect device (GPU or CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.write(f"Running on: {device}")

# Load the model and tokenizer (swap with "facebook/bart-large" if no fine-tuned model)
model_path = "chatbot_finetuned"  # Replace if needed
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, use_safetensors=True, torch_dtype=torch.float16)
model = model.to(device)
model.eval()

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

# Generic phrases to refine responses (only for non-search intents)
generic_responses = [
    "please provide more details",
    "could you please provide",
    "i need more information"
]

# Initialize user profile
if 'user_profile' not in st.session_state:
    st.session_state['user_profile'] = {
        'name': None,
        'email': None,
        'phone': None,
        'username': None,
        'preferences': {},
        'last_updated': {}
    }

# Fetch Wikipedia snippets
def search_wikipedia(query, num_results=3):
    print(f"Searching Wikipedia for: {query}")
    clean_query = re.sub(r'could you search (?:about|for)?|on wikipedia\??', '', query, flags=re.IGNORECASE).strip()
    print(f"Cleaned query: {clean_query}")
    try:
        search_results = wikipedia.search(clean_query, results=num_results)
        snippets = []
        for result in search_results:
            try:
                page = wikipedia.page(result, auto_suggest=False)
                snippets.append(page.summary[:2000])
            except wikipedia.exceptions.DisambiguationError:
                pass
        return snippets
    except Exception as e:
        print(f"Error searching Wikipedia: {e}")
        return []

# Generate response
def generate_contextual_response(prompt, model, tokenizer, max_length=80, num_beams=5):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "what is your name" in response.lower() and st.session_state['user_profile']['name']:
        response = response.replace("What is your name?", f"Thanks for letting me know your name is {st.session_state['user_profile']['name']}.")
    if "what is your email" in response.lower() and st.session_state['user_profile']['email']:
        response = response.replace("What is your email?", f"I already have your email as {st.session_state['user_profile']['email']}.")
    if "what is your phone" in response.lower() and st.session_state['user_profile']['phone']:
        response = response.replace("What is your phone number?", f"I have your phone number as {st.session_state['user_profile']['phone']}.")
    
    return response

# Transcribe voice input
def transcribe_voice():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    try:
        with microphone as source:
            st.info("Listening... Speak now!")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
            st.success("Voice input received. Processing...")
            return recognizer.recognize_google(audio)
    except sr.WaitTimeoutError:
        st.warning("No voice detected. Please try again.")
    except sr.UnknownValueError:
        st.warning("Could not understand audio. Please speak clearly.")
    except Exception as e:
        st.error(f"Error: {str(e)}")
    return ""

# Parse instruction
def parse_instruction(intent, instruction):
    intent_patterns = {
        "payment_issue": r'payment issue in (.+)',
        "place_order": r'order (.+)',
        "track_order": r'track order (\d+)',
        "cancel_order": r'cancel order (\d+)',
        "create_account": r'full name:\s*([\w\s]+)|email:\s*([\w\.\@]+)|username:\s*(\w+)',
        "search": r'search (?:about|for)?\s*(.+?)(?:\s*on\s*wikipedia)?'
    }
    
    user_info_patterns = {
        'name': r'(?:my name is|i am|i\'m|call me) ([\w\s]+)',
        'email': r'(?:my email is|reach me at|contact me at) ([\w\.\@]+)',
        'phone': r'(?:my phone is|my number is|call me at) (\d[\d\s\-\(\)]+)'
    }
    
    parsed_detail = None
    
    if intent in intent_patterns:
        match = re.search(intent_patterns[intent], instruction.lower())
        if match:
            if intent == "search":
                parsed_detail = match.group(1).strip()
            elif intent == "create_account":
                parsed_detail = " ".join(filter(None, match.groups())).strip()
                if match.group(1):
                    st.session_state['user_profile']['name'] = match.group(1).strip()
                    st.session_state['user_profile']['last_updated']['name'] = datetime.now()
                if match.group(2):
                    st.session_state['user_profile']['email'] = match.group(2).strip()
                    st.session_state['user_profile']['last_updated']['email'] = datetime.now()
                if match.group(3):
                    st.session_state['user_profile']['username'] = match.group(3).strip()
                    st.session_state['user_profile']['last_updated']['username'] = datetime.now()
            else:
                parsed_detail = match.group(1).strip()
    
    for info_type, pattern in user_info_patterns.items():
        match = re.search(pattern, instruction.lower())
        if match:
            value = match.group(1).strip()
            st.session_state['user_profile'][info_type] = value
            st.session_state['user_profile']['last_updated'][info_type] = datetime.now()
    
    return parsed_detail

# Build prompt
def build_prompt(intent, instruction, parsed_detail, history):
    system_message = "You are a helpful assistant. For 'search' intent, summarize Wikipedia info directly. Otherwise, use details to respond specifically, asking for more only if needed."
    
    user_context = ""
    if st.session_state['user_profile']['name']:
        user_context += f" The user's name is {st.session_state['user_profile']['name']}."
    
    snippets = search_wikipedia(instruction)
    external_info = " | ".join(snippets) if snippets else "No additional info found."
    
    if history:
        past_texts = [f"Intent: {item['intent']} Instruction: {item['instruction']} Response: {item['response']}" 
                      for item in history]
        current_text = f"Intent: {intent} Instruction: {instruction}"
        vectorizer = TfidfVectorizer()
        past_vectors = vectorizer.fit_transform(past_texts)
        current_vector = vectorizer.transform([current_text])
        similarities = cosine_similarity(current_vector, past_vectors)
        top_indices = similarities.argsort()[0][-3:][::-1]
        relevant_history = " | ".join([past_texts[i] for i in top_indices])
        prompt = f"{system_message} | User context: {user_context} | Relevant history: {relevant_history} | External info: {external_info} | Current: Intent: {intent} Instruction: {instruction}"
    else:
        prompt = f"{system_message} | User context: {user_context} | External info: {external_info} | Intent: {intent} Instruction: {instruction}"
    
    if parsed_detail:
        prompt += f" | User provided detail: {parsed_detail}"
    elif intent != "search":
        prompt += " | If details are missing, ask for more only if not in user context."
    
    return prompt

# Initialize session state
if 'instruction' not in st.session_state:
    st.session_state['instruction'] = ""
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []
if 'conversation_state' not in st.session_state:
    st.session_state['conversation_state'] = {}

# State management
def get_state(intent):
    return st.session_state['conversation_state'].get(intent, "initial")

def set_state(intent, state):
    st.session_state['conversation_state'][intent] = state

# UI
st.subheader("Input Fields")

predefined_intents = [
    'cancel_order', 'change_order', 'change_shipping_address', 'check_cancellation_fee',
    'check_invoice', 'check_payment_methods', 'check_refund_policy', 'complaint',
    'contact_customer_service', 'contact_human_agent', 'create_account', 'delete_account',
    'delivery_options', 'delivery_period', 'edit_account', 'get_invoice', 'get_refund',
    'newsletter_subscription', 'payment_issue', 'place_order', 'recover_password',
    'registration_problems', 'review', 'search', 'set_up_shipping_address', 'switch_account',
    'track_order', 'track_refund'
]
col1, col2 = st.columns([1, 2])
with col1:
    intent_selection = st.selectbox("Select an Intent:", predefined_intents)
with col2:
    custom_intent = st.text_input("Or enter a custom intent:", placeholder="e.g., order_pizza")

intent = custom_intent.strip() if custom_intent.strip() else intent_selection

col3, col4 = st.columns([1, 2])
with col3:
    if st.button("Record Instruction"):
        transcribed_text = transcribe_voice()
        if transcribed_text:
            st.session_state['instruction'] = transcribed_text
with col4:
    instruction = st.text_area(
        "Instruction:",
        value=st.session_state['instruction'],
        placeholder="e.g., Could you search about RAG on Wikipedia?"
    )

st.subheader("Current User Profile")
if any(value is not None for key, value in st.session_state['user_profile'].items() if key != 'last_updated' and key != 'preferences'):
    profile_data = {k: v for k, v in st.session_state['user_profile'].items() if k not in ['last_updated', 'preferences'] and v is not None}
    import pandas as pd
    st.table(pd.DataFrame([profile_data]))
else:
    st.write("No user profile information available yet.")

max_length = st.slider("Response Max Length", min_value=50, max_value=500, value=150, step=10)
num_beams = st.slider("Number of Beams", min_value=1, max_value=20, value=5)

st.subheader("Conversation History")
if st.session_state['conversation_history']:
    for i, item in enumerate(st.session_state['conversation_history']):
        st.write(f"**You ({i+1})**: Intent: {item['intent']} | Instruction: {item['instruction']}")
        st.write(f"**Bot**: {item['response']}")
else:
    st.write("No conversation history yet.")

col5, col6, col7 = st.columns([1, 1, 1])
with col5:
    if st.button("Generate Response"):
        if not intent or not instruction.strip():
            st.warning("Please provide both an intent and an instruction.")
        else:
            st.session_state['instruction'] = instruction.strip()
            parsed_detail = parse_instruction(intent, instruction)
            current_state = get_state(intent)
            with st.spinner("Searching and generating response..."):
                snippets = search_wikipedia(instruction)
                prompt = build_prompt(intent, instruction, parsed_detail, st.session_state['conversation_history'])
                st.write(f"Debug - Prompt: {prompt}")  # Keep this for sanity checks
                if intent == "search":
                    if snippets:
                        full_text = "Here’s what I found about " + instruction + ": " + " ".join(snippets)
                        # Convert max_length (tokens) to chars (rough estimate: 1 token ≈ 4 chars)
                        char_limit = max_length * 4
                        if len(full_text) > char_limit:
                            trimmed_text = full_text[:char_limit].rsplit('.', 1)[0] + '.'  # Last full sentence
                            if len(trimmed_text) > char_limit or trimmed_text == full_text[:char_limit] + '.':
                                trimmed_text = full_text[:char_limit].rsplit(' ', 1)[0] + '...'  # Last word + ellipsis
                            response = trimmed_text
                        else:
                            response = full_text
                    else:
                        suggestions = wikipedia.search(instruction, results=3)
                        response = f"No exact match found. Suggestions: {', '.join(suggestions)}." if suggestions else "Sorry, I couldn’t find anything on Wikipedia about that."
                else:
                    response = generate_contextual_response(prompt, model, tokenizer, max_length, num_beams)
                    if any(phrase in response.lower() for phrase in generic_responses):
                        if intent == "create_account" and st.session_state['user_profile']['name']:
                            response = f"Thanks {st.session_state['user_profile']['name']}! Could you specify what else you need help with?"
                        else:
                            response = "Could you specify what exactly you're having trouble with? For example, is it a payment issue, order tracking, or something else?"
                        set_state(intent, "awaiting_details")
                    elif current_state == "initial" and not parsed_detail:
                        set_state(intent, "awaiting_details")
                    else:
                        set_state(intent, "initial")
            st.subheader("Chatbot Response")
            st.success(response)
            st.session_state['conversation_history'].append({
                'intent': intent,
                'instruction': instruction,
                'response': response
            })
            if not tts_engine.isBusy():
                tts_engine.say(response)
                tts_engine.runAndWait()
                tts_engine.stop()
                         
with col6:
    if st.button("Reset Instruction"):
        st.session_state['instruction'] = ""
        st.rerun()

with col7:
    if st.button("Clear All"):
        st.session_state['conversation_history'] = []
        st.session_state['instruction'] = ""
        st.session_state['conversation_state'] = {}
        st.session_state['user_profile'] = {
            'name': None,
            'email': None,
            'phone': None,
            'username': None,
            'preferences': {},
            'last_updated': {}
        }
        st.rerun()