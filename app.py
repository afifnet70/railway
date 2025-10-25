import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import pandas as pd
from datetime import datetime
import json
import os

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Medical Dialogue Summarizer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/medical-dialogue-summarization',
        'Report a bug': "https://github.com/yourusername/medical-dialogue-summarization/issues",
        'About': "## Medical Dialogue Summarization\nAI-powered tool for converting doctor-patient dialogues into structured SOAP notes."
    }
)

# ==================== SESSION STATE ====================
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'usage_history' not in st.session_state:
    st.session_state.usage_history = []

# ==================== MODEL LOADING ====================
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the fine-tuned medical model"""
    try:
        with st.spinner("🔄 Loading medical AI model..."):
            model_path = "./small_llm_medical"
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            
            return tokenizer, model, device
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return None, None, None

# ==================== GENERATION FUNCTION ====================
def generate_soap_note(dialogue, max_length=150, temperature=0.7):
    """Generate SOAP note from medical dialogue"""
    if not st.session_state.model or not st.session_state.tokenizer:
        return "Error: Model not loaded"
    
    try:
        prompt = f"Summarize this medical dialogue:\n{dialogue}\n\nSummary:"
        inputs = st.session_state.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(st.session_state.device)
        
        with torch.no_grad():
            outputs = st.session_state.model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True,
                pad_token_id=st.session_state.tokenizer.eos_token_id,
                eos_token_id=st.session_state.tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        generated_text = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Summary:" in generated_text:
            summary = generated_text.split("Summary:")[-1].strip()
        else:
            summary = generated_text.replace(prompt, "").strip()
        
        return summary
        
    except Exception as e:
        return f"Generation error: {str(e)}"

# ==================== SIDEBAR ====================
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")
    
    # Model Configuration
    st.subheader("Model Settings")
    max_length = st.slider("Max Summary Length", 50, 300, 150, help="Maximum length of generated SOAP note")
    temperature = st.slider("Creativity", 0.1, 1.0, 0.7, help="Higher values = more creative, Lower values = more focused")
    
    # Load Model Button
    if not st.session_state.model_loaded:
        if st.button("🚀 Load AI Model", use_container_width=True):
            st.session_state.tokenizer, st.session_state.model, st.session_state.device = load_model()
            if st.session_state.model:
                st.session_state.model_loaded = True
                st.success("✅ Model loaded successfully!")
                st.rerun()
    else:
        st.success("✅ Model Loaded")
        if st.button("🔄 Reload Model", use_container_width=True):
            st.session_state.model_loaded = False
            st.rerun()
    
    st.markdown("---")
    
    # Quick Examples
    st.subheader("💡 Quick Examples")
    example_dialogues = {
        "Fever & Cough": "Patient: I have had fever and cough for 3 days with body aches. Doctor: Any breathing difficulties? Patient: Some shortness of breath when walking.",
        "Headache & Nausea": "Patient: I've been experiencing headache and nausea since morning. Doctor: Any vomiting or vision changes? Patient: No vomiting, but lights seem too bright.",
        "Chest Pain": "Patient: I feel chest pain when I walk fast or climb stairs. Doctor: Does the pain radiate anywhere? Patient: Sometimes to my left arm."
    }
    
    for name, dialogue in example_dialogues.items():
        if st.button(f"📝 {name}", use_container_width=True):
            st.session_state.dialogue_input = dialogue
            st.rerun()
    
    st.markdown("---")
    
    # About Section
    st.subheader("ℹ️ About")
    st.markdown("""
    This AI tool converts doctor-patient dialogues 
    into structured **SOAP notes**:
    
    - **S**ubjective
    - **O**bjective  
    - **A**ssessment
    - **P**lan
    
    *Fine-tuned on medical dialogue data*
    """)

# ==================== MAIN INTERFACE ====================
st.title("🏥 Medical Dialogue Summarizer")
st.markdown("Convert doctor-patient conversations into structured SOAP notes using AI")

# Initialize dialogue input
if 'dialogue_input' not in st.session_state:
    st.session_state.dialogue_input = ""

# Main Input Area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📋 Medical Dialogue Input")
    dialogue = st.text_area(
        "Enter doctor-patient conversation:",
        value=st.session_state.dialogue_input,
        height=200,
        placeholder="Patient: I have been experiencing symptoms...\nDoctor: When did this start?...\nPatient: About 3 days ago...",
        help="Enter the complete dialogue between doctor and patient"
    )

with col2:
    st.subheader("🎯 Generation Parameters")
    
    # Real-time preview
    if dialogue:
        word_count = len(dialogue.split())
        st.metric("Word Count", word_count)
        
        if word_count > 300:
            st.warning("⚠️ Long input may affect quality")
        elif word_count < 20:
            st.warning("⚠️ Very short input")
        else:
            st.success("✅ Good input length")
    
    st.metric("Model Status", "Loaded" if st.session_state.model_loaded else "Not Loaded")

# Generate Button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    generate_clicked = st.button(
        "🎯 Generate SOAP Note", 
        type="primary", 
        use_container_width=True,
        disabled=not st.session_state.model_loaded or not dialogue.strip()
    )

# ==================== RESULTS SECTION ====================
if generate_clicked and dialogue.strip():
    st.markdown("---")
    
    with st.spinner("🔄 Generating SOAP note... This may take a few seconds."):
        start_time = time.time()
        soap_note = generate_soap_note(dialogue, max_length, temperature)
        generation_time = time.time() - start_time
    
    # Results Columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Generated SOAP Note")
        
        # Display SOAP note in a nice box
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                border-radius: 10px;
                color: white;
                margin: 10px 0;
            ">
                <h4 style="color: white; margin: 0 0 15px 0;">🧬 SOAP Note</h4>
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 5px;">
                    {soap_note.replace(chr(10), '<br>')}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Performance metrics
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        with col_metric1:
            st.metric("Generation Time", f"{generation_time:.2f}s")
        with col_metric2:
            st.metric("Note Length", f"{len(soap_note.split())} words")
        with col_metric3:
            st.metric("Input Length", f"{len(dialogue.split())} words")
    
    with col2:
        st.subheader("🔍 Analysis")
        
        # SOAP Components Breakdown (simulated)
        soap_components = {
            "Subjective (S)": "Patient-reported symptoms and history",
            "Objective (O)": "Clinical findings and observations", 
            "Assessment (A)": "Diagnosis and clinical impression",
            "Plan (P)": "Treatment plan and follow-up"
        }
        
        for component, description in soap_components.items():
            with st.expander(f"📌 {component}"):
                st.caption(description)
                # Here you could add component-specific analysis
        
        # Quality Assessment
        st.subheader("📊 Quality Metrics")
        
        # Simulated quality scores (in real app, you'd calculate these)
        quality_data = {
            "Medical Accuracy": 85,
            "Completeness": 78, 
            "Clarity": 92,
            "Structure": 88
        }
        
        for metric, score in quality_data.items():
            st.progress(score/100, text=f"{metric}: {score}%")
    
    # Action Buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💾 Save Result", use_container_width=True):
            # Save to session history
            record = {
                "timestamp": datetime.now().isoformat(),
                "dialogue": dialogue,
                "soap_note": soap_note,
                "generation_time": generation_time
            }
            st.session_state.usage_history.append(record)
            st.success("✅ Result saved to history")
    
    with col2:
        st.download_button(
            label="📥 Download SOAP",
            data=soap_note,
            file_name=f"soap_note_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col3:
        if st.button("🔄 Regenerate", use_container_width=True):
            st.rerun()
    
    with col4:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.dialogue_input = ""
            st.rerun()

# ==================== USAGE HISTORY ====================
if st.session_state.usage_history:
    st.markdown("---")
    st.subheader("📈 Usage History")
    
    # Convert to DataFrame for nice display
    history_df = pd.DataFrame(st.session_state.usage_history)
    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
    history_df['date'] = history_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    history_df['dialogue_preview'] = history_df['dialogue'].str[:50] + '...'
    history_df['soap_preview'] = history_df['soap_note'].str[:50] + '...'
    
    # Display recent entries
    for i, record in enumerate(reversed(st.session_state.usage_history[-5:])):
        with st.expander(f"📄 {record['timestamp'][:16]} - {record['soap_note'][:30]}..."):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.text_area("Dialogue", record['dialogue'], height=100, key=f"dial_{i}")
            with col2:
                st.text_area("SOAP Note", record['soap_note'], height=100, key=f"soap_{i}")
            st.caption(f"Generated in {record['generation_time']:.2f}s")

# ==================== FOOTER ====================
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**🏥 Medical AI Tool**")
    st.caption("Fine-tuned for medical dialogue summarization")

with footer_col2:
    st.markdown("**🔧 Model Info**")
    st.caption("DialoGPT-small • 117M parameters")

with footer_col3:
    st.markdown("**📊 Statistics**")
    st.caption(f"Total generations: {len(st.session_state.usage_history)}")

# ==================== REQUIREMENTS ====================
# requirements.txt for Streamlit:
"""
streamlit>=1.28.0
torch>=2.0.0
transformers>=4.30.0
sentencepiece>=0.1.97
accelerate>=0.20.0
pandas>=1.5.0
"""

# ==================== DEPLOYMENT ====================
# For Streamlit Cloud, create these files:
# 1. app.py (this file)
# 2. requirements.txt (above)
# 3. small_llm_medical/ (your model folder)
# 4. .streamlit/config.toml (optional)

# .streamlit/config.toml:
"""
[theme]
primaryColor = "#ff4b4b"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
address = "0.0.0.0"
port = 8501
"""
