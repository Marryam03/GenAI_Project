## Run
export PROJECT_ROOT="/home/mabdrabou/Desktop/GenAI Project"
export GEMINI_API_KEY="GEMINI API KEY"
export GEN_MAX_TOKENS=2048
export GEN_MAX_RETRIES=3
export GEN_WORKERS=4

python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000