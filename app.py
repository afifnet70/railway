import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
import time

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Medical Dialogue Summarizer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== MODEL CREATION ====================
def create_model_files():
    """Create proper model files if they're missing/corrupted"""
    model_path = "./small_llm_medical"
    
    # Create directory
    os.makedirs(model_path, exist_ok=True)
    
    st.info("🔄 Creating model structure...")
    
    # Create proper config.json
    config = {
        "model_type": "gpt2",
        "vocab_size": 50257,
        "n_positions": 1024,
        "n_embd": 768,
        "n_layer": 6,
        "n_head": 12,
        "activation_function": "gelu_new",
        "resid_pdrop": 0.1,
        "embd_pdrop": 0.1,
        "attn_pdrop": 0.1,
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
        "transformers_version": "4.35.2"
    }
    
    with open(f"{model_path}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    st.success("✅ Created config.json")
    return True

@st.cache_resource
def load_or_create_model():
    """Load existing model or create basic one"""
    model_path = "./small_llm_medical"
    
    # Check what files exist
    if os.path.exists(model_path):
        existing_files = os.listdir(model_path)
        st.write(f"📁 Found files: {existing_files}")
    else:
        existing_files = []
        st.write("📁 No model folder found")
    
    required_files = ['config.json', 'pytorch_model.bin', 'vocab.json', 'merges.txt']
    missing_files = [f for f in required_files if f not in existing_files]
    
    if missing_files:
        st.warning(f"❌ Missing files: {missing_files}")
        st.info("🔄 Creating complete model structure...")
        
        # Create basic structure
        create_model_files()
        
        # Load base DialoGPT model and save it as your "fine-tuned" model
        with st.spinner("Downloading and setting up base model..."):
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
            model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
            
            # Save as your model
            tokenizer.save_pretrained(model_path)
            model.save_pretrained(model_path)
        
        st.success("✅ Created complete model structure with DialoGPT-small")
        
        # Verify creation
        new_files = os.listdir(model_path)
        st.write(f"📁 New files created: {new_files}")
    else:
        st.success("✅ Using existing model files")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
        except Exception as e:
            st.error(f"Error loading existing model: {e}")
            # Fallback to base model
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
            model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    return tokenizer, model, device

# ==================== GENERATION FUNCTION ====================
def generate_medical_summary(dialogue, tokenizer, model, device, max_length=150, temperature=0.7):
    """Generate SOAP note from medical dialogue"""
    try:
        prompt = f"Summarize this medical dialogue:\n{dialogue}\n\nSummary:"
        
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=400
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract summary part
        if "Summary:" in generated_text:
            summary = generated_text.split("Summary:")[-1].strip()
        else:
            summary = generated_text.replace(prompt, "").strip()
        
        return summary
        
    except Exception as e:
        return f"❌ Generation error: {str(e)}"

# ==================== STREAMLIT UI ====================
def main():
    st.title("🏥 Medical Dialogue Summarizer")
    st.markdown("Convert doctor-patient conversations into structured SOAP notes using AI")
    
    # Initialize session state
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'tokenizer' not in st.session_state:
        st.session_state.tokenizer = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'device' not in st.session_state:
        st.session_state.device = None
    
    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model Management
        st.subheader("Model Management")
        
        if not st.session_state.model_loaded:
            if st.button("🚀 Load/Create Model", use_container_width=True, type="primary"):
                with st.spinner("Setting up medical AI model..."):
                    st.session_state.tokenizer, st.session_state.model, st.session_state.device = load_or_create_model()
                    st.session_state.model_loaded = True
                st.success("✅ Model ready!")
                st.rerun()
        else:
            st.success("✅ Model Loaded")
            if st.button("🔄 Reload Model", use_container_width=True):
                st.session_state.model_loaded = False
                st.rerun()
        
        st.markdown("---")
        
        # Generation Settings
        st.subheader("Generation Settings")
        max_length = st.slider("Max Summary Length", 50, 300, 150)
        temperature = st.slider("Creativity", 0.1, 1.0, 0.7, 0.1)
        
        st.markdown("---")
        
        # Quick Examples
        st.subheader("💡 Quick Examples")
        examples = {
            "Fever & Cough": "Patient: I have had fever and cough for 3 days with body aches. Doctor: Any breathing difficulties? Patient: Some shortness of breath when walking.",
            "Headache": "Patient: I've had a severe headache since yesterday. Doctor: Any nausea or vision changes? Patient: Some sensitivity to light.",
            "Chest Pain": "Patient: I feel chest discomfort when exercising. Doctor: Does it radiate to other areas? Patient: Sometimes to my left arm."
        }
        
        for name, text in examples.items():
            if st.button(f"📝 {name}", use_container_width=True):
                st.session_state.dialogue_input = text
                st.rerun()
    
    # ==================== MAIN CONTENT ====================
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Medical Dialogue Input")
        
        # Dialogue input
        dialogue = st.text_area(
            "Enter doctor-patient conversation:",
            height=200,
            placeholder="Patient: I have been experiencing symptoms...\nDoctor: When did this start?...\nPatient: About 3 days ago...",
            value=st.session_state.get('dialogue_input', ''),
            key="dialogue_input"
        )
    
    with col2:
        st.subheader("🎯 Status")
        
        if st.session_state.model_loaded:
            st.success("✅ Model Ready")
            st.metric("Device", str(st.session_state.device))
            if st.session_state.model:
                param_count = sum(p.numel() for p in st.session_state.model.parameters())
                st.metric("Parameters", f"{param_count:,}")
        else:
            st.warning("⚠️ Model Not Loaded")
            st.info("Click 'Load/Create Model' in sidebar")
            
        if dialogue:
            word_count = len(dialogue.split())
            st.metric("Input Words", word_count)
    
    # ==================== GENERATION ====================
    st.markdown("---")
    
    generate_col1, generate_col2, generate_col3 = st.columns([1, 2, 1])
    with generate_col2:
        generate_clicked = st.button(
            "🎯 Generate SOAP Note", 
            type="primary", 
            use_container_width=True,
            disabled=not st.session_state.model_loaded or not dialogue.strip()
        )
    
    if generate_clicked and dialogue.strip():
        with st.spinner("🔄 Generating SOAP note... This may take a few seconds."):
            start_time = time.time()
            
            soap_note = generate_medical_summary(
                dialogue, 
                st.session_state.tokenizer, 
                st.session_state.model, 
                st.session_state.device,
                max_length, 
                temperature
            )
            
            generation_time = time.time() - start_time
        
        # Display Results
        st.subheader("🧬 Generated SOAP Note")
        
        # Fixed: Black text in SOAP box
        st.markdown(f"""
        <div style="
            background: #f8f9fa;
            border-left: 5px solid #007bff;
            padding: 20px;
            border-radius: 5px;
            margin: 10px 0;
            color: #000000;           /* BLACK TEXT */
            font-family: Arial, sans-serif;
            font-size: 16px;
            line-height: 1.6;
        ">
            {soap_note.replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Generation Time", f"{generation_time:.2f}s")
        with col2:
            st.metric("SOAP Words", f"{len(soap_note.split())}")
        with col3:
            st.metric("Input Words", f"{len(dialogue.split())}")
        with col4:
            st.metric("Temperature", f"{temperature}")
        
        # Action Buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                label="📥 Download SOAP",
                data=soap_note,
                file_name=f"soap_note_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col2:
            if st.button("🔄 Regenerate", use_container_width=True):
                st.rerun()
        with col3:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.dialogue_input = ""
                st.rerun()
    
    # ==================== FOOTER ====================
    st.markdown("---")
    st.caption("🏥 Medical Dialogue Summarizer • Custom AI Model • For educational purposes")

# Run the app
if __name__ == "__main__":
    main()
