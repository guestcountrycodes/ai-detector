from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# ── Load model ────────────────────────────────────────────────────────────
MODEL_ID  = "Hello-SimpleAI/chatgpt-detector-roberta"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
detector  = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    truncation=True,
    max_length=512,
)

app = FastAPI()

# ── Core logic ────────────────────────────────────────────────────────────
def analyse_text(text: str) -> dict:
    text = text.strip()
    if not text:
        return {"error": "Empty text"}

    words  = text.split()
    chunks = [" ".join(words[i:i+400]) for i in range(0, len(words), 400)]

    ai_scores = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        r     = detector(chunk)[0]
        label = r["label"].upper()
        score = r["score"]
        if label in ("CHATGPT", "FAKE", "AI", "LABEL_1"):
            ai_scores.append(score)
        else:
            ai_scores.append(1.0 - score)

    if not ai_scores:
        return {"error": "Could not process text"}

    ai_pct    = round(sum(ai_scores) / len(ai_scores) * 100, 1)
    human_pct = round(100 - ai_pct, 1)

    if ai_pct >= 80:
        verdict = "AI Generated"
    elif ai_pct >= 50:
        verdict = "Likely AI Generated"
    elif ai_pct >= 30:
        verdict = "Mixed / Uncertain"
    else:
        verdict = "Human Written"

    return {
        "ai_score":    ai_pct,
        "human_score": human_pct,
        "verdict":     verdict,
        "word_count":  len(words),
    }

# ── Routes ────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ok", "service": "GuestCountry AI Detector"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/detect")
async def detect(request: Request):
    try:
        body = await request.json()
        text = str(body.get("text", "")).strip()
    except Exception:
        return JSONResponse({"ok": False, "error": "Invalid JSON"}, status_code=400)

    if not text:
        return JSONResponse({"ok": False, "error": "No text provided"}, status_code=400)

    text   = text[:5000]
    result = analyse_text(text)

    if "error" in result:
        return JSONResponse({"ok": False, "error": result["error"]}, status_code=500)

    return JSONResponse({
        "ok":          True,
        "ai_score":    result["ai_score"],
        "human_score": result["human_score"],
        "verdict":     result["verdict"],
        "word_count":  result["word_count"],
    })
