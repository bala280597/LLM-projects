import os
import requests
from bs4 import BeautifulSoup
from IPython.display import Markdown, display

def chat_completion(sys_prom, user_prom):
    url = "http://localhost:11434/api/chat"
    headers = {"Content-Type": "application/json"}
    payload = {
            "model": "mistral",
            "messages": [
                        {"role": "system", "content": sys_prom},
                        {"role": "user", "content": user_prom}
            ],
            "stream": False
    }
    
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    
    data = response.json()
    
                
    return data['message']['content']

headers = {
 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

class Website:

    def __init__(self, url):
        """
        Create this Website object from the given url using the BeautifulSoup library
        """
        self.url = url
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

def user_prompt_for(website):
    user_prompt = f"You are looking at a website titled {website.title}"
    user_prompt += "\nThe contents of this website is as follows; \
please provide a short summary of this website in markdown. \
If it includes news or announcements, then summarize these too.\n\n"
    user_prompt += website.text
    return user_prompt

def messages_for(website):
    
    system_prompt = "You are an assistant that analyzes the contents of a website \
    and provides a short summary, ignoring text that might be navigation related. \
    Respond in markdown."
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(website)}
    ]

def summarize(url):
    website = Website(url)
    response = chat_completion(messages_for(website))
    return response

def display_summary(url):
    summary = summarize(url)
    display(Markdown(summary))

def chat_bot(messages):
    import requests
    import json

    url = "http://localhost:11434/api/chat"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "mistral",
        "messages": messages,
        "stream": True
    }

    try:
        response = requests.post(url, json=payload, headers=headers, stream=True)
        for line in response.iter_lines():
            if line:
                chunk = line.decode('utf-8')
                if chunk.startswith('data:'):
                    chunk = chunk[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    data = json.loads(chunk)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    yield f"[Invalid JSON]: {chunk}"
    except Exception as e:
        yield f"[Error]: {str(e)}"


display_summary("https://cnn.com")

# Stream chat
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of France?"}
]

for chunk in chat_bot(messages):
    print(chunk, end='', flush=True)
