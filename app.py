import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import speech_recognition as sr
import pyttsx3

# Set page configuration for wider layout
st.set_page_config(layout="wide")

# Title of the app
st.title("Custom Fine-Tuned Model Chatbot")
st.markdown("Provide the intent and instruction to get a response from your fine-tuned model.")

# Load model and tokenizer
model_path = "chatbot_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Initialize TTS engine
tts_engine = pyttsx3.init()

# Function to generate a response
def generate_response(prompt, model, tokenizer, max_length=50, num_beams=5):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Function to transcribe voice input
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

# Initialize session state for instruction if not already present
if 'instruction' not in st.session_state:
    st.session_state['instruction'] = ""

# User Inputs Section
st.subheader("Input Fields")

# Dropdown and Custom Intent Together
predefined_intents = [
    "book_flight",
    "track_order",
    "cancel_reservation",
    "get_weather",
    "general_query"
]
col1, col2 = st.columns([1, 2])
with col1:
    intent_selection = st.selectbox("Select an Intent:", predefined_intents)
with col2:
    custom_intent = st.text_input("Or enter a custom intent (overrides selection):", placeholder="e.g., book_hotel")

# Determine intent
intent = custom_intent if custom_intent.strip() else intent_selection

# Instruction Field with Voice Integration
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
        placeholder="e.g., Find a flight to New York"
    )

# Sliders for customization
max_length = st.slider("Response Max Length", min_value=20, max_value=200, value=50, step=10)
num_beams = st.slider("Number of Beams (Beam Search)", min_value=1, max_value=10, value=5)

# Generate Response and Reset Buttons
col5, col6 = st.columns([1, 1])
with col5:
    if st.button("Generate Response"):
        if not intent or not st.session_state['instruction']:
            st.warning("Please fill out both 'Intent' and 'Instruction' fields.")
        else:
            # Combine intent and instruction
            input_prompt = f"Intent: {intent} | Instruction: {st.session_state['instruction']}"
            
            # Generate the response
            with st.spinner("Generating response..."):
                response = generate_response(input_prompt, model, tokenizer, max_length, num_beams)
            
            # Display the response
            st.subheader("Chatbot Response")
            st.success(response)
            
            # Immediate TTS Playback
            if not tts_engine.isBusy():
                tts_engine.say(response)
                tts_engine.runAndWait()
                tts_engine.stop()
with col6:
    if st.button("Reset"):
        st.session_state['instruction'] = ""
        st.experimental_rerun()
