import os
import openai
from dotenv import load_dotenv

load_dotenv()

def get_llm_client():
    api_key = os.getenv("LLM_API_KEY")
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    if not api_key:
        raise ValueError("LLM_API_KEY not found in environment variables.")

    if provider == "openai":
        return openai.OpenAI(api_key=api_key)
    else:
        # Placeholder for other providers
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

def generate_linkedin_post(topic="latest trends in Artificial Intelligence"):
    """
    Generates a LinkedIn post about the given topic using an LLM.
    """
    client = get_llm_client()
    
    prompt = (
        f"Write a professional, engaging, and insightful LinkedIn post about {topic}. "
        "The post should be suitable for a tech-savvy audience but accessible to general professionals. "
        "Include 3-5 relevant hashtags at the end. "
        "Keep the tone optimistic and forward-looking. "
        "Do not include any preamble like 'Here is a post', just return the post content."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o", # Using a high-quality model
            messages=[
                {"role": "system", "content": "You are an expert AI tech influencer on LinkedIn."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating post: {e}")
        return None

if __name__ == "__main__":
    # Test run
    print(generate_linkedin_post())
