from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from openai_coco import analyze_damage as gpt4_analyze
import tempfile

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/analyze")
async def analyze_damage(image: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        content = await image.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        gpt_result = gpt4_analyze(tmp_path)
        
        # analyze_damage returns a set like {'front_bumper', 'door'}
        if isinstance(gpt_result, set):
            parts_list = [part.replace('_', ' ').title() for part in gpt_result]
        elif isinstance(gpt_result, dict):
            damaged_parts = gpt_result.get('damaged_parts', [])
            parts_list = [p['part'].replace('_', ' ').title() for p in damaged_parts]
        else:
            parts_list = []
        
        return {
            "damageType": ", ".join(parts_list) if parts_list else "No damage detected",
            "severity": "Moderate",
            "estimatedCost": "Pending estimate",
            "recommendation": f"Detected damage to: {', '.join(parts_list)}" if parts_list else "No damage found"
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            "damageType": "Error",
            "severity": "N/A",
            "estimatedCost": "N/A",
            "recommendation": f"Analysis failed: {str(e)}"
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/")
async def root():
    return {"message": "Vehicle Damage API - use POST /api/analyze"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)