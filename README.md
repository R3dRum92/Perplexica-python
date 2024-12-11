1. Python libraries:
```bash
python3 -m venv venv
source ./venv/bin/activate
pip install fastapi pydantic requests colorlog langchain langchain_openai langchain_google_genai langchain_anthropic langchain_ollama transformers
```

2. Searxng:
[Searxng](https://github.com/searxng/searxng-docker)

```bash
git clone https://github.com/searxng/searxng-docker.git
cd searxng-docker
```
Generate secret key:
```bash
sed -i "s|ultrasecretkey|$(openssl rand -hex 32)|g" searxng/settings.yml
```
Edit searxng/settings.yml:
Set limiter to false
Add the following lines:
```
search:
  formats:
    - html
    - json
```
