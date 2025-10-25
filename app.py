from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

app = FastAPI(title="Medical Summarization API")

# Load model
tokenizer = AutoTokenizer.from_pretrained("your-username/medical-dialogue-summarizer")
model = AutoModelForCausalLM.from_pretrained("your-username/medical-dialogue-summarizer")

class DialogueRequest(BaseModel):
    dialogue: str

class SummaryResponse(BaseModel):
    summary: str
    status: str

@app.post("/summarize")
async def summarize(request: DialogueRequest):
    prompt = f"Medical Dialogue: {request.dialogue}\nSummary:"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=len(inputs[0]) + 100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Summary:" in summary:
        summary = summary.split("Summary:")[1].strip()
    
    return SummaryResponse(summary=summary, status="success")

@app.get("/")
async def root():
    return {"message": "Medical Dialogue Summarization API"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
