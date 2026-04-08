FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install openenv-core fastapi uvicorn gradio requests

EXPOSE 8000

CMD ["uvicorn", "openenv.core.env_server:app", "--host", "0.0.0.0", "--port", "8000"]