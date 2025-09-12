FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-fetch weights so the first request is fast
RUN python - <<'PY'
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
print("EfficientNet-B0 weights cached.")
PY

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]