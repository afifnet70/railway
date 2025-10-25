from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medical Dialogue Summarization API",
    description="Convert doctor-patient dialogues into SOAP notes using AI",
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

# Load model globally
try:
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("./trained_model")
    model = AutoModelForCausalLM.from_pretrained("./trained_model")
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise e

class DialogueRequest(BaseModel):
    dialogue: str
    max_length: int = 150
    temperature: float = 0.7

class SummaryResponse(BaseModel):
    summary: str
    status: str
    model: str = "distilgpt2-medical"

def generate_summary(dialogue: str, max_length: int = 150, temperature: float = 0.7) -> str:
    """Generate SOAP note from medical dialogue"""
    try:
        prompt = f"Medical Dialogue: {dialogue}\nSummary:"
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
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
            summary = generated_text.split("Summary:")[1].strip()
        else:
            summary = generated_text.replace("Medical Dialogue:", "").strip()
        
        return summary
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

# API Routes
@app.get("/")
async def root():
    return {
        "message": "Medical Dialogue Summarization API",
        "status": "active",
        "endpoints": {
            "POST /summarize": "Generate SOAP note from dialogue",
            "GET /health": "Health check",
            "GET /docs": "API documentation"
        }
    }

@app.post("/summarize", response_model=SummaryResponse)
async def summarize_dialogue(request: DialogueRequest):
    """Summarize medical dialogue into SOAP note"""
    try:
        summary = generate_summary(
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
    return {
        "status": "healthy", 
        "model_loaded": True,
        "service": "medical-summarization-api"
    }

@app.get("/model-info")
async def model_info():
    """Get model information"""
    return {
        "model_type": "DistilGPT2",
        "parameters": "82M",
        "purpose": "Medical dialogue summarization",
        "input_format": "Doctor-patient dialogue",
        "output_format": "SOAP note"
    }

# Example usage endpoint
@app.get("/example")
async def get_example():
    """Get example usage"""
    example_dialogue = "Patient: I have had fever and cough for 3 days. Doctor: Any breathing difficulties? Patient: Some shortness of breath when walking fast."
    example_summary = generate_summary(example_dialogue)
    
    return {
        "example_dialogue": example_dialogue,
        "example_summary": example_summary,
        "usage": {
            "curl": 'curl -X POST "https://your-app.railway.app/summarize" -H "Content-Type: application/json" -d \'{"dialogue": "Your medical dialogue here"}\'',
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
