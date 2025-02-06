import os
import asyncio
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client with OpenRouter configuration
openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_API_BASE")
)
embeddings_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def test_embeddings(text: str) -> list:
    """Test getting embeddings from OpenRouter."""
    try:
        response = await embeddings_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )

        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return None

async def test_title_summary(text: str, url: str) -> dict:
    """Test getting title and summary from OpenRouter."""
    try:
        system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
        Return a JSON object with 'title' and 'summary' keys.
        For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
        For the summary: Create a concise summary of the main points in this chunk.
        Keep both title and summary concise but informative."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"URL: {url}\n\nContent:\n{text[:1000]}..."}
        ]

        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return None

async def main():
    # Test data
    test_text = """# Welcome to the Documentation
    This is a sample documentation page that we'll use for testing.
    It contains multiple paragraphs and some technical content.
    
    ## Features
    - Feature 1: Description of feature 1
    - Feature 2: Description of feature 2
    """
    test_url = "https://example.com/docs/test"

    print("\n=== Testing Embeddings ===")
    embeddings = await test_embeddings(test_text)
    if embeddings:
        print(f"✅ Successfully got embeddings")
        print(f"Embedding length: {len(embeddings)}")
        print(f"First few values: {embeddings[:5]}")
    else:
        print("❌ Failed to get embeddings")

    print("\n=== Testing Title/Summary Generation ===")
    title_summary = await test_title_summary(test_text, test_url)
    if title_summary:
        print("✅ Successfully got title and summary")
        print(f"Title: {title_summary.get('title')}")
        print(f"Summary: {title_summary.get('summary')}")
    else:
        print("❌ Failed to get title and summary")

if __name__ == "__main__":
    asyncio.run(main()) 