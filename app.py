from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medical Dialogue Summarization API",
    description="Convert doctor-patient dialogues into SOAP notes using fine-tuned DialoGPT-small",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and tokenizer
tokenizer = None
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
async def startup_event():
    """Load model on startup with safetensors support"""
    global tokenizer, model
    try:
        logger.info("Loading fine-tuned medical model (safetensors format)...")
        
        # Load your fine-tuned model with safetensors
        model_path = "./small_llm_medical"
        
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model with safetensors support
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            use_safetensors=True,  # ✅ Explicitly use safetensors
            local_files_only=True  # ✅ Only use local files
        )
        
        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.to(device)
        model.eval()
        
        logger.info("✅ Model loaded successfully with safetensors!")
        logger.info(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"🎯 Using device: {device}")
        
        # Verify model files
        model_files = os.listdir(model_path)
        logger.info(f"📁 Model files: {model_files}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        # Try alternative loading method
        try:
            logger.info("Trying alternative loading method...")
            model = AutoModelForCausalLM.from_pretrained(model_path)
            model.to(device)
            model.eval()
            logger.info("✅ Model loaded with alternative method!")
        except Exception as e2:
            logger.error(f"❌ Alternative loading also failed: {e2}")
            raise e

class DialogueRequest(BaseModel):
    dialogue: str
    max_length: int = 100
    temperature: float = 0.7

class SummaryResponse(BaseModel):
    summary: str
    status: str
    model: str = "dialogpt-medical-finetuned"

def generate_medical_summary(dialogue: str, max_length: int = 100, temperature: float = 0.7) -> str:
    """Generate SOAP note from medical dialogue using your fine-tuned model"""
    try:
        # Use the same prompt format as your training
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
        
        # Extract summary part (same as your training)
        if "Summary:" in generated_text:
            summary = generated_text.split("Summary:")[-1].strip()
        else:
            summary = generated_text.replace(prompt, "").strip()
        
        return summary
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

# ==================== API ROUTES ====================

@app.get("/")
async def root():
    return {
        "message": "Medical Dialogue Summarization API",
        "status": "active", 
        "model": "DialoGPT-small (Fine-tuned for Medical Dialogues)",
        "model_format": "safetensors",
        "endpoints": {
            "POST /summarize": "Generate SOAP note from dialogue",
            "GET /health": "Health check",
            "GET /model-info": "Model information",
            "GET /demo": "Web interface",
            "GET /docs": "API documentation"
        }
    }

@app.post("/summarize", response_model=SummaryResponse)
async def summarize_dialogue(request: DialogueRequest):
    """Summarize medical dialogue into SOAP note using your fine-tuned model"""
    try:
        if not model or not tokenizer:
            raise HTTPException(status_code=503, detail="Model not loaded")
            
        summary = generate_medical_summary(
            request.dialogue, 
            request.max_length, 
            request.temperature
        )
        return SummaryResponse(summary=summary, status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model and tokenizer else "not loaded"
    return {
        "status": "healthy", 
        "model_loaded": model_status,
        "model_format": "safetensors",
        "device": str(device),
        "service": "medical-dialogue-summarization"
    }

@app.get("/model-info")
async def model_info():
    """Get information about the fine-tuned model"""
    model_files = os.listdir("./small_llm_medical") if os.path.exists("./small_llm_medical") else []
    return {
        "model_name": "DialoGPT-small",
        "fine_tuned_for": "Medical dialogue summarization",
        "model_format": "safetensors",
        "parameters": f"{sum(p.numel() for p in model.parameters()):,}" if model else "unknown",
        "training_data": "Medical dialogues with SOAP notes",
        "model_files": model_files,
        "input_format": "Doctor-patient dialogue",
        "output_format": "SOAP note summary"
    }

@app.get("/example")
async def get_example():
    """Get example usage"""
    example_dialogue = "Patient: I have had fever and cough for 3 days with some chest pain. Doctor: Any breathing difficulties? Patient: Yes, I feel short of breath when walking."
    
    if model and tokenizer:
        example_summary = generate_medical_summary(example_dialogue)
    else:
        example_summary = "SOAP Note: Fever, cough, chest pain, shortness of breath. Assessment: Possible respiratory infection."
    
    return {
        "example_dialogue": example_dialogue,
        "example_summary": example_summary,
        "usage": {
            "curl": """curl -X POST "https://your-app.railway.app/summarize" -H "Content-Type: application/json" -d '{"dialogue": "Patient has fever and cough..."}'""",
            "python": """
import requests
response = requests.post(
    "https://your-app.railway.app/summarize",
    json={"dialogue": "Patient has fever and cough..."}
)
print(response.json())
            """
        }
    }

# Serve simple HTML frontend (optional)
@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    """Simple web interface for testing"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Medical Dialogue Summarizer</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 20px; border-radius: 10px; }
            textarea { width: 100%; height: 150px; margin: 10px 0; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            .result { background: white; padding: 15px; border-radius: 5px; margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏥 Medical Dialogue Summarizer</h1>
            <p>Enter a doctor-patient dialogue to generate a SOAP note:</p>
            <textarea id="dialogue" placeholder="Patient: I have fever and cough...&#10;Doctor: Any other symptoms?..."></textarea>
            <br>
            <button onclick="summarize()">Generate SOAP Note</button>
            <div id="result" class="result" style="display:none;"></div>
        </div>
        
        <script>
            async function summarize() {
                const dialogue = document.getElementById('dialogue').value;
                const resultDiv = document.getElementById('result');
                
                if (!dialogue) {
                    alert('Please enter a medical dialogue');
                    return;
                }
                
                resultDiv.innerHTML = 'Generating summary...';
                resultDiv.style.display = 'block';
                
                try {
                    const response = await fetch('/summarize', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ dialogue: dialogue })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        resultDiv.innerHTML = '<strong>Generated SOAP Note:</strong><br>' + data.summary;
                    } else {
                        resultDiv.innerHTML = 'Error: ' + data.detail;
                    }
                } catch (error) {
                    resultDiv.innerHTML = 'Error: ' + error.message;
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
