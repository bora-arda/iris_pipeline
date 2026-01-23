FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -m scripts.prepare_data
RUN mkdir -p models

CMD ["python", "-m" , "training.train"]