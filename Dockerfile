FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install openenv-core fastapi uvicorn gradio requests

EXPOSE 8000

CMD ["python", "-m", "openenv.core.env_server", "server.environment:APIEnvironment"]