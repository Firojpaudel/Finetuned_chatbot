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
import sqlite3
import bcrypt
import random
import string

# Suppress unnecessary warnings
warnings.filterwarnings("ignore")

# Set wide layout for Streamlit
st.set_page_config(layout="wide")

# --- Database Setup ---
def initialize_database():
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            phone TEXT,
            username TEXT UNIQUE,
            password TEXT,
            preferences TEXT,
            last_updated TEXT,
            reset_code TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ConversationHistory (
            conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            thread_id INTEGER,
            intent TEXT,
            instruction TEXT,
            response TEXT,
            timestamp TEXT,
            FOREIGN KEY (user_id) REFERENCES Users (user_id)
        )
    ''')
    
    conn.commit()
    conn.close()

initialize_database()

# --- Database Helper Functions ---
def update_user_profile(user_id, field, value):
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()
    cursor.execute(f"UPDATE Users SET {field} = ?, last_updated = ? WHERE user_id = ?",
                   (value, datetime.now().isoformat(), user_id))
    conn.commit()
    conn.close()

def insert_conversation_history(user_id, thread_id, intent, instruction, response):
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO ConversationHistory (user_id, thread_id, intent, instruction, response, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                   (user_id, thread_id, intent, instruction, response, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def load_conversation_history(user_id, thread_id=None):
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()
    if thread_id is None:
        cursor.execute("SELECT intent, instruction, response, timestamp, thread_id FROM ConversationHistory WHERE user_id = ? ORDER BY timestamp",
                       (user_id,))
    else:
        cursor.execute("SELECT intent, instruction, response, timestamp, thread_id FROM ConversationHistory WHERE user_id = ? AND thread_id = ? ORDER BY timestamp",
                       (user_id, thread_id))
    history = [{'intent': row[0], 'instruction': row[1], 'response': row[2], 'timestamp': row[3], 'thread_id': row[4]} for row in cursor.fetchall()]
    conn.close()
    return history

def get_thread_headings(user_id):
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()
    cursor.execute("SELECT thread_id, MIN(timestamp) FROM ConversationHistory WHERE user_id = ? GROUP BY thread_id ORDER BY MIN(timestamp)",
                   (user_id,))
    threads = [(row[0], row[1]) for row in cursor.fetchall()]
    conn.close()
    return {thread_id: f"Thread {i+1} - {timestamp}" for i, (thread_id, timestamp) in enumerate(threads)}

def get_next_thread_id(user_id):
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(thread_id) FROM ConversationHistory WHERE user_id = ?", (user_id,))
    max_id = cursor.fetchone()[0]
    conn.close()
    return (max_id + 1) if max_id is not None else 1

def create_new_thread(user_id):
    new_thread_id = get_next_thread_id(user_id)
    # Insert a placeholder entry to make the thread visible
    insert_conversation_history(user_id, new_thread_id, "None", "", "New thread started")
    return new_thread_id

def generate_reset_code(length=6):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def store_reset_code(user_id, reset_code):
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()
    cursor.execute("UPDATE Users SET reset_code = ? WHERE user_id = ?", (reset_code, user_id))
    conn.commit()
    conn.close()

def verify_reset_code(username, reset_code):
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, reset_code FROM Users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    if result and result[1] == reset_code:
        return result[0]
    return None

def reset_password(user_id, new_password):
    hashed_pw = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()
    cursor.execute("UPDATE Users SET password = ?, reset_code = NULL WHERE user_id = ?",
                   (hashed_pw, user_id))
    conn.commit()
    conn.close()

# --- Chatbot Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.write(f"Running on: {device}")

model_path = "chatbot_finetuned"  # Adjust to your model path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, use_safetensors=True, torch_dtype=torch.float16)
model = model.to(device)
model.eval()

tts_engine = pyttsx3.init()

generic_responses = ["please provide more details", "could you please provide", "i need more information"]
require_details_intents = ["track_order", "cancel_order", "change_order"]

# --- Wikipedia Search ---
def search_wikipedia(query, num_results=3):
    clean_query = re.sub(r'could you search (?:about|for)?|on wikipedia\??', '', query, flags=re.IGNORECASE).strip()
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
        return []

# --- Response Generation ---
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
    
    if "what is your name" in response.lower() and st.session_state['user_profile'].get('name'):
        response = response.replace("What is your name?", f"Thanks for letting me know your name is {st.session_state['user_profile']['name']}.")
    if "what is your email" in response.lower() and st.session_state['user_profile'].get('email'):
        response = response.replace("What is your email?", f"I already have your email as {st.session_state['user_profile']['email']}.")
    if "what is your phone" in response.lower() and st.session_state['user_profile'].get('phone'):
        response = response.replace("What is your phone number?", f"I have your phone number as {st.session_state['user_profile']['phone']}.")
    
    return response

# --- Voice Input ---
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

# --- Intent Parsing ---
def parse_instruction(intent, instruction):
    intent_patterns = {
        "payment_issue": r'payment issue (.+)',
        "place_order": r'order (.+)',
        "track_order": r'(?:track order|order number|my order number is)[\s:]*([\w!@]+)',
        "cancel_order": r'(?:cancel order|order number|my order number is)[\s:]*([\w!@]+)',
        "change_order": r'(?:change order|order number|my order number is)[\s:]*([\w!@]+)',
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
                parsed_detail = match.group(1).strip() if match.group(1) else instruction.strip()
            elif intent == "create_account":
                parsed_detail = " ".join(filter(None, match.groups())).strip()
                if match.group(1):
                    st.session_state['user_profile']['name'] = match.group(1).strip()
                    st.session_state['user_profile']['last_updated']['name'] = datetime.now()
                    update_user_profile(st.session_state['user_id'], 'name', match.group(1).strip())
                if match.group(2):
                    st.session_state['user_profile']['email'] = match.group(2).strip()
                    st.session_state['user_profile']['last_updated']['email'] = datetime.now()
                    update_user_profile(st.session_state['user_id'], 'email', match.group(2).strip())
                if match.group(3):
                    st.session_state['user_profile']['username'] = match.group(3).strip()
                    st.session_state['user_profile']['last_updated']['username'] = datetime.now()
                    update_user_profile(st.session_state['user_id'], 'username', match.group(3).strip())
            else:
                parsed_detail = match.group(1).strip()
    
    for info_type, pattern in user_info_patterns.items():
        match = re.search(pattern, instruction.lower())
        if match:
            value = match.group(1).strip()
            st.session_state['user_profile'][info_type] = value
            st.session_state['user_profile']['last_updated'][info_type] = datetime.now()
            update_user_profile(st.session_state['user_id'], info_type, value)
    
    return parsed_detail

# --- Prompt Building ---
def build_prompt(intent, instruction, parsed_detail, history):
    system_message = "You are a helpful assistant. Use details to respond specifically, asking for more only if needed."
    user_context = ""
    if st.session_state['user_profile'].get('name'):
        user_context += f" The user's name is {st.session_state['user_profile']['name']}."
    
    if history:
        past_texts = [f"Intent: {item['intent']} Instruction: {item['instruction']} Response: {item['response']}" 
                      for item in history if item['instruction']]  # Exclude placeholder entries
        current_text = f"Intent: {intent} Instruction: {instruction}"
        vectorizer = TfidfVectorizer()
        past_vectors = vectorizer.fit_transform(past_texts)
        current_vector = vectorizer.transform([current_text])
        similarities = cosine_similarity(current_vector, past_vectors)
        top_indices = similarities.argsort()[0][-3:][::-1]
        relevant_history = " | ".join([past_texts[i] for i in top_indices])
        prompt = f"{system_message} | User context: {user_context} | Relevant history: {relevant_history} | Current: Intent: {intent} Instruction: {instruction}"
    else:
        prompt = f"{system_message} | User context: {user_context} | Intent: {intent} Instruction: {instruction}"
    
    if parsed_detail:
        prompt += f" | User provided detail: {parsed_detail}"
    elif intent != "search":
        prompt += " | If details are missing, ask for more only if not in user context."
    
    return prompt

# --- Session State Initialization ---
if 'user_profile' not in st.session_state:
    st.session_state['user_profile'] = {}
if 'instruction' not in st.session_state:
    st.session_state['instruction'] = ""
if 'conversation_state' not in st.session_state:
    st.session_state['conversation_state'] = {}
if 'reset_stage' not in st.session_state:
    st.session_state['reset_stage'] = None
if 'current_thread_id' not in st.session_state:
    st.session_state['current_thread_id'] = None

def get_state(intent):
    return st.session_state['conversation_state'].get(intent, "initial")

def set_state(intent, state):
    st.session_state['conversation_state'][intent] = state

# --- Sidebar for Login/Logout and Threads ---
if st.session_state.get('logged_in', False):
    if 'user_profile' in st.session_state and st.session_state['user_profile'].get('username'):
        st.sidebar.write(f"Logged in as: {st.session_state['user_profile']['username']}")
    else:
        st.sidebar.write("Loading profile...")
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state.pop('user_id', None)
        st.session_state['user_profile'] = {}
        st.session_state['instruction'] = ""
        st.session_state['conversation_state'] = {}
        st.session_state['reset_stage'] = None
        st.session_state['current_thread_id'] = None
        st.sidebar.success("Logged out successfully.")
        st.rerun()
    
    st.sidebar.subheader("Conversation Threads")
    if 'user_id' in st.session_state:
        thread_headings = get_thread_headings(st.session_state['user_id'])
        if thread_headings:
            selected_thread = st.sidebar.selectbox("Select a Thread", options=list(thread_headings.values()), index=0)
            st.session_state['current_thread_id'] = next(thread_id for thread_id, heading in thread_headings.items() if heading == selected_thread)
        else:
            st.sidebar.write("No threads yet.")
        
        if st.sidebar.button("New Chat"):
            st.session_state['current_thread_id'] = create_new_thread(st.session_state['user_id'])
            st.session_state['instruction'] = ""
            st.session_state['conversation_state'] = {}
            st.rerun()
    else:
        st.sidebar.write("No user ID found.")

else:
    option = st.sidebar.selectbox("Choose an action", ["Login", "Create Account", "Forgot Password"])

# --- Main App ---
if st.session_state.get('logged_in', False):
    st.title("Dynamic Intent Chatbot with RAG")
    st.markdown("Chat with a finetuned-bot that adapts to any intent using conversation memory and external knowledge!")
    st.markdown("---")
    st.markdown("Finetuned model: `facebook/bart-base`")
    st.markdown("Finetuned on: `Bitext-customer-support-llm-chatbot-training-dataset`")

    # Initialize thread if none exists
    if 'user_id' in st.session_state and st.session_state['current_thread_id'] is None:
        st.session_state['current_thread_id'] = create_new_thread(st.session_state['user_id'])

    # UI Components
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
        intent_selection = st.selectbox(
            "Select an Intent:",
            ["None"] + predefined_intents,
            key=f"intent_select_{st.session_state['current_thread_id']}"
        )
    with col2:
        custom_intent = st.text_input("Or enter a custom intent:", placeholder="e.g., order_pizza")

    intent = custom_intent.strip() if custom_intent.strip() else (intent_selection if intent_selection != "None" else None)

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

    st.subheader(f"Conversation History - Thread {st.session_state['current_thread_id']}")
    history = load_conversation_history(st.session_state['user_id'], st.session_state['current_thread_id'])
    if history:
        for i, item in enumerate([h for h in history if h['instruction']]):  # Skip placeholder entries
            st.write(f"**You ({i+1})**: Intent: {item['intent']} | Instruction: {item['instruction']}")
            st.write(f"**Bot**: {item['response']}")
    else:
        st.write("No conversation history in this thread yet.")

    col5, col6, col7 = st.columns([1, 1, 1])
    with col5:
        if st.button("Generate Response"):
            if not instruction.strip():
                st.warning("Please provide an instruction.")
            else:
                st.session_state['instruction'] = instruction.strip()
                parsed_detail = parse_instruction(intent, instruction) if intent else None
                with st.spinner("Generating response..."):
                    if intent == "search":
                        snippets = search_wikipedia(instruction)
                        if snippets:
                            search_topic = parsed_detail if parsed_detail else instruction.strip()
                            full_text = f"Here’s what I found about {search_topic}: " + " ".join(snippets)
                            char_limit = max_length * 4
                            if len(full_text) > char_limit:
                                trimmed_text = full_text[:char_limit].rsplit('.', 1)[0] + '.'
                                if len(trimmed_text) > char_limit or trimmed_text == full_text[:char_limit] + '.':
                                    trimmed_text = full_text[:char_limit].rsplit(' ', 1)[0] + '...'
                                response = trimmed_text
                            else:
                                response = full_text
                        else:
                            suggestions = wikipedia.search(instruction, results=3)
                            response = f"No exact match found. Suggestions: {', '.join(suggestions)}." if suggestions else "Sorry, I couldn’t find anything on Wikipedia about that."
                    else:
                        if intent in require_details_intents and parsed_detail is None:
                            if intent == "track_order":
                                response = "Please provide your order number to track."
                            elif intent == "cancel_order":
                                response = "Please provide your order number to cancel."
                            elif intent == "change_order":
                                response = "Please provide your order number to change."
                            set_state(intent, "awaiting_details")
                        else:
                            prompt = build_prompt(intent or "None", instruction, parsed_detail, history)
                            response = generate_contextual_response(prompt, model, tokenizer, max_length, num_beams)
                            set_state(intent or "None", "initial")
                    st.subheader("Chatbot Response")
                    st.success(response)
                    insert_conversation_history(st.session_state['user_id'], st.session_state['current_thread_id'], intent or "None", instruction, response)
                    if not tts_engine.isBusy():
                        tts_engine.say(response)
                        tts_engine.runAndWait()
                        tts_engine.stop()

    with col6:
        if st.button("Reset Instruction"):
            st.session_state['instruction'] = ""
            st.rerun()

    with col7:
        if st.button("Clear Thread"):
            st.session_state['instruction'] = ""
            st.session_state['conversation_state'] = {}
            conn = sqlite3.connect('chatbot.db')
            cursor = conn.cursor()
            cursor.execute("DELETE FROM ConversationHistory WHERE user_id = ? AND thread_id = ?",
                           (st.session_state['user_id'], st.session_state['current_thread_id']))
            conn.commit()
            conn.close()
            st.rerun()

else:
    if option == "Create Account":
        st.subheader("Create Account")
        name = st.text_input("Name")
        email = st.text_input("Email")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Create Account"):
            conn = sqlite3.connect('chatbot.db')
            cursor = conn.cursor()
            cursor.execute("SELECT user_id FROM Users WHERE username = ?", (username,))
            if cursor.fetchone():
                st.error("Username already taken.")
            else:
                hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
                cursor.execute("INSERT INTO Users (name, email, username, password, last_updated) VALUES (?, ?, ?, ?, ?)",
                               (name, email, username, hashed_pw, datetime.now().isoformat()))
                conn.commit()
                conn.close()
                st.success("Account created successfully. Please log in.")

    elif option == "Login":
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            conn = sqlite3.connect('chatbot.db')
            cursor = conn.cursor()
            cursor.execute("SELECT user_id, password FROM Users WHERE username = ?", (username,))
            result = cursor.fetchone()
            if result:
                user_id, stored_hash = result
                if bcrypt.checkpw(password.encode('utf-8'), stored_hash):
                    st.session_state['logged_in'] = True
                    st.session_state['user_id'] = user_id
                    cursor.execute("SELECT name, email, phone, username FROM Users WHERE user_id = ?", (user_id,))
                    user_data = cursor.fetchone()
                    st.session_state['user_profile'] = {
                        'name': user_data[0],
                        'email': user_data[1],
                        'phone': user_data[2],
                        'username': user_data[3],
                        'preferences': {},
                        'last_updated': {}
                    }
                    conn.close()
                    st.session_state['current_thread_id'] = create_new_thread(user_id)
                    st.success("Logged in successfully.")
                    st.rerun()
                else:
                    st.error("Incorrect password.")
            else:
                st.error("Username not found.")
            conn.close()

    elif option == "Forgot Password":
        st.subheader("Forgot Password")
        if st.session_state['reset_stage'] == "code_sent":
            username = st.text_input("Username (re-enter if needed)")
            reset_code = st.text_input("Enter Reset Code")
            new_password = st.text_input("New Password", type="password")
            if st.button("Reset Password"):
                user_id = verify_reset_code(username, reset_code)
                if user_id:
                    reset_password(user_id, new_password)
                    st.session_state['reset_stage'] = None
                    st.success("Password reset successfully. Please log in with your new password.")
                else:
                    st.error("Invalid reset code.")
        else:
            username = st.text_input("Username")
            email = st.text_input("Email")
            if st.button("Send Reset Code"):
                conn = sqlite3.connect('chatbot.db')
                cursor = conn.cursor()
                cursor.execute("SELECT user_id, email FROM Users WHERE username = ?", (username,))
                result = cursor.fetchone()
                if result and result[1] == email:
                    user_id = result[0]
                    reset_code = generate_reset_code()
                    store_reset_code(user_id, reset_code)
                    st.session_state['reset_stage'] = "code_sent"
                    st.success(f"Reset code generated: {reset_code} (In a real app, this would be emailed to {email})")
                else:
                    st.error("Username or email not found.")
                conn.close()

    st.write("Please log in, create an account, or reset your password to use the chatbot.")