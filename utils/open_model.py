import openai
import requests


deployment_name = "gpt-4o-2024-05-13"
openai.api_type = "azure"
openai.azure_endpoint ="https://questionanswer.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "af7a5446c36847f09fad6578420d026a"


def get_openai_response(prompt, deployment_name, temperature=0.5, max_tokens=30):
 

  response = openai.chat.completions.create(
    # deployment_id=deployment_name, # Updated parameter name
    messages=[{"role": "user", "content": prompt}], # New message format
    temperature=temperature,
    model=deployment_name,
    max_tokens=max_tokens)

  return response.choices[0].message.content.strip(" \n")


def process_files_and_create_json(text_):
    results = []
    prompt = f"{text_}' Provide a suitable answer"
    gtp3_result =get_openai_response(prompt, deployment_name)
    results.append({"message": gtp3_result})
    return results