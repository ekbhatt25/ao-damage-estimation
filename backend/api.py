"""
Auto-Owners AI Claims API
CV Detection: Mask R-CNN (parts) + Mask R-CNN (damage types)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pathlib import Path
import shutil
import time

from cv_detector import CVDetector

app = FastAPI(title="Auto-Owners AI Claims API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("\nLoading CV models...")
cv_detector = CVDetector()
print("Models ready.\n")


@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    """
    Upload a vehicle photo. Returns detected parts and damage types as JSON.
    """
    if not image.filename:
        raise HTTPException(status_code=400, detail="No image provided")

    temp_path = Path(f"temp/temp_{image.filename}")
    temp_path.parent.mkdir(exist_ok=True)

    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        t0 = time.perf_counter()
        detections = cv_detector.detect(str(temp_path))
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        return {
            "detections": detections,
            "summary": {
                "total_damaged_parts": len(detections),
                "parts": list({d["part"] for d in detections}),
                "damage_types": list({d["damage_type"] for d in detections}),
            },
            "inference_ms": elapsed_ms,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.get("/health")
def health():
    return {"status": "healthy", "models": "parts + damage Mask R-CNN"}


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CV Damage Detection</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 700px; margin: 50px auto; padding: 20px; background: #f5f5f5; }
            .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h1 { color: #003d7a; }
            input[type=file], button { display: block; width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
            button { background: #003d7a; color: white; border: none; cursor: pointer; font-size: 16px; }
            button:hover { background: #005ca8; }
            pre { background: #f8f9fa; padding: 15px; border-radius: 4px; overflow-x: auto; font-size: 13px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>CV Damage Detection</h1>
            <p>Mask R-CNN parts detection + damage type detection</p>
            <form id="form">
                <input type="file" name="image" accept="image/*" required>
                <button type="submit">Detect Damage</button>
            </form>
            <div id="result"></div>
        </div>
        <script>
            document.getElementById('form').onsubmit = async (e) => {
                e.preventDefault();
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = '<p>Running detection...</p>';
                const formData = new FormData(e.target);
                try {
                    const res = await fetch('/detect', { method: 'POST', body: formData });
                    const data = await res.json();
                    if (!res.ok) { resultDiv.innerHTML = `<p><b>Error:</b> ${data.detail}</p>`; return; }
                    resultDiv.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
                } catch (err) {
                    resultDiv.innerHTML = `<p><b>Error:</b> ${err.message}</p>`;
                }
            };
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
