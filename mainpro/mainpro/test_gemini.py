from google import genai
import os

# Retrieve API key from environment variable
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Please set your GEMINI_API_KEY environment variable")

# Initialize the client
client = genai.Client(api_key=api_key)

# Generate content using Gemini 2.5 Pro model
response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="which model you are and who is the owner of this account"
)

# Print the generated code
print(response.text)