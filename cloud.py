#pip install --upgrade "openai>=1.0.0,<2.0.0" "pydantic>=2.0,<3.0"

#check credit at
#https://platform.openai.com/settings/organization/billing/overview

"""
chatgpt_image_text_bridge.py

Python example that asks a ChatGPT-style model whether a user input should produce an image or only text
and then either:
 - generates a black-and-white sun image (for Slovak prompt "Nakresli mi slniečko") and returns the short text
 - returns short text only (for e.g. "co je hlavne mesto Slovenska")

Requirements:
 - Install the official OpenAI Python client (and keep it up-to-date):
     pip install openai
 - Set your API key in the environment variable OPENAI_API_KEY

This script is intentionally deterministic by giving the assistant explicit examples and by asking for strict JSON output.
Adapt model names if your environment uses different model ids.
"""

import os
import json
from openai import OpenAI
import numpy as np
import cv2

# init client using OPENAI_API_KEY env var
with open('api-key.txt','r') as f:
    api_key = f.read()
    
if not api_key:
    raise RuntimeError("Set OPENAI_API_KEY environment variable before running.")

client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = (
    "You are the robot NICO with two hands and no legs, working as a concise assistant in Brno city. Respond only with JSON (no extra text) using the exact schema below. "
    "If the user request asks to draw or generate an image (in Slovak, Czech or English), return:"
    "{\"type\":\"image\", \"caption\":<short text>, \"image_prompt\":<prompt-for-image>, \"size\":<size>} . "
    "If the user request asks only for text, return: {\"type\":\"text\", \"content\":<short text>}. "
    "Language of returned text should match the user's language (Slovak, Czech or English). Keep texts very short. "
    "Do NOT include explanations, markup, or any extra fields."
)

# Provide two concrete examples to make responses predictable.
EXAMPLE_USER_1 = "Nakresli mi slniečko"
EXAMPLE_ASSISTANT_1 = json.dumps({
    "type": "image",
    "caption": "fajn, nakreslime to",
    "image_prompt": "Black-and-white simple line drawing of a smiling sun. Minimalist, clean lines, transparent background, centered composition, svg-like style, no text.",
    "size": "1024x1024"
})

EXAMPLE_USER_2 = "co je hlavne mesto Slovenska"
EXAMPLE_ASSISTANT_2 = json.dumps({
    "type": "text",
    "content": "Bratislava"
})


def ask_chat_to_classify(user_text: str, language = None) -> dict:
    """Send the user's text to the chat model and parse strict JSON response."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": EXAMPLE_USER_1},
        {"role": "assistant", "content": EXAMPLE_ASSISTANT_1},
        {"role": "user", "content": EXAMPLE_USER_2},
        {"role": "assistant", "content": EXAMPLE_ASSISTANT_2},
        {"role": "user", "content": user_text},
    ]

    resp = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.0,
        max_tokens=200,
    )

    # The assistant should respond with a single JSON object string
    assistant_text = resp.choices[0].message.content.strip()

    try:
        parsed = json.loads(assistant_text)
    except Exception as e:
        raise RuntimeError(f"Failed to parse assistant JSON: {e}\nRaw response:\n{assistant_text}")

    return parsed


def generate_image_from_prompt(image_prompt: str, size: str = "1024x1024") -> bytes:
    """Call the Images API to generate an image and return binary bytes (PNG/JPEG depending on engine).
    Adjust model name if your account uses a different image model id.
    """
    img_resp = client.images.generate(
        model="gpt-image-1",
        prompt=image_prompt,
        size=size,
        background=None  # keep transparent background if supported
    )

    # API returns base64-encoded images in data[0].b64_json
    b64 = img_resp.data[0].b64_json
    import base64
    img_bytes = base64.b64decode(b64)
    return img_bytes

def classify(user_text: str):
    classification = ask_chat_to_classify(user_text)
    kind = classification.get("type") 
    if kind == "image":
        return kind, classification.get("image_prompt"), classification.get("caption"), classification.get("size", "1024x1024")
    elif kind == "text":
        return kind, classification.get("content")
    else:
        return kind

def generate(image_prompt, size):
    img_bytes = generate_image_from_prompt(image_prompt, size=size)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(240,240))
    return img
    
def handle_user_input(user_text: str):
    print('question:',user_text)
    #global img, content, caption
    features = classify(user_text)
    kind = features[0]
    if kind == "image":
        # produce image and display/save it
        image_prompt = features[1]
        caption = features[2]
        size = features[3]

        if not image_prompt or not caption:
            raise RuntimeError("Assistant returned invalid image response (missing prompt/caption).")

        print('answer:',caption,'moment...')
        img = generate(image_prompt, size)
        
        cv2.imwrite('generated.png',img)

        #out_filename = "generated_image.png"
        #with open(out_filename, "wb") as f:
        #    f.write(img_bytes)
        #print(f"Image saved to: {out_filename}")

    elif kind == "text":
        content = features[1]
        if not content:
            raise RuntimeError("Assistant returned invalid text response (missing content).")
            
        print('answer:',content)


if __name__ == "__main__":
    # Example runs
    #print("=== Example: Nakresli mi slniečko ===")
    #handle_user_input("Nakresli mi slniečko")
    print("\n=== Example: co je hlavne mesto Slovenska ===")
    handle_user_input("co je hlavne mesto Slovenska")

    # For interactive use, uncomment below:
    # user = input('Napíš otázku (SK/CZ/EN): ').strip()
    # handle_user_input(user)
