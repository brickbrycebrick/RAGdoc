import os
import asyncio
import json
from typing import List, Dict
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import create_client

from crawler import BatchProcessor, TitleSummaryBatchProcessor, RateLimiter

load_dotenv()

# Rate limiting configuration
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 0.5  # seconds
MAX_RETRY_DELAY = 64    # seconds
BATCH_SIZE = 5          # number of requests to batch together
RATE_LIMIT_WINDOW = 60  # seconds
MAX_REQUESTS_PER_WINDOW = 50  # adjust based on your rate limits

# Initialize clients
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_API_BASE")
)

model = OpenAIModel(os.getenv('LLM_MODEL', 'gpt-4o-mini'))


system_prompt = """
You are an expert at extracting technical keywords and their definitions from documentation.
Your task is to analyze the provided content and summary to identify important technical terms, concepts, and their definitions unique to the content.

Rules for extraction:
1. Only extract keywords that have clear technical meaning or significance
2. Each keyword must have a clear, concise definition based on the context
3. Focus on programming terms, library features, and technical concepts
4. Avoid extracting common terms unless they have specific technical meaning related to the documentation
5. Definitions should be brief but complete enough to understand the concept
6. If a keyword appears multiple times, combine the context to create a comprehensive definition

Output must be a JSON array of objects, each containing:
- keyword: The technical term or concept
- definition: A clear, concise explanation of the term

Example output:
{
    "keywords": [
        {
            "keyword": "async/await",
            "definition": "Python syntax for writing asynchronous code that looks synchronous, allowing concurrent operations without blocking."
        }
    ]
}

If no valid technical keywords are found, return an empty array:
{
    "keywords": []
}
"""

keyword_limiter = RateLimiter(RATE_LIMIT_WINDOW, MAX_REQUESTS_PER_WINDOW)
keyword_processor = TitleSummaryBatchProcessor(BATCH_SIZE, keyword_limiter)

class ProcessingDeps:
    def __init__(self):
        self.supabase = supabase
        self.openai = openai_client


async def extract_keywords(content: str, summary: str) -> List[Dict]:
    """Extract keywords with proper argument handling"""
    item = {
        "system_prompt": system_prompt,
        "user_content": f"Summary: {summary}\n\nContent:\n{content}"
    }

    for attempt in range(5):
        try:
            result = await keyword_processor.add_to_batch(item)
            # We need to await the result if it's a Future
            if asyncio.isfuture(result):
                result = await result
            return json.loads(result).get("keywords", [])
        except Exception as e:
            if attempt == 4:
                print(f"Failed after 5 attempts: {e}")
                return []
            await asyncio.sleep(2 ** attempt)

async def process_chunks(chunks: List[Dict]):
    """Process chunks in parallel batches"""
    # Create all LLM requests first without awaiting
    llm_tasks = []
    for chunk in chunks:
        item = {
            "system_prompt": system_prompt,
            "user_content": f"Summary: {chunk['summary']}\n\nContent:\n{chunk['content']}"
        }
        llm_tasks.append(keyword_processor.add_to_batch(item))
    
    # Wait for all LLM requests to complete
    try:
        results = await asyncio.gather(*llm_tasks)
        print(f"Debug - Raw results: {results}")  # Debug log
        
        # Process results and insert into database
        for chunk, result in zip(chunks, results):
            try:
                if asyncio.isfuture(result):
                    result = await result
                
                # The result might be a string or already parsed JSON
                if isinstance(result, str):
                    parsed_result = json.loads(result)
                else:
                    parsed_result = result
                
                keywords = parsed_result.get("keywords", [])
                print(f"Debug - Parsed keywords: {keywords}")  # Debug log
                
                for keyword in keywords:
                    print(f"Inserting keyword: {keyword['keyword']} - {keyword['definition']}")
                    # Execute database insertion immediately instead of collecting tasks
                    await supabase.table("keywords").insert({
                        "url": chunk["url"],
                        "chunk_number": chunk["chunk_number"],
                        "keyword": keyword["keyword"],
                        "definition": keyword["definition"]
                    }).execute()
                    
                print(f"Processed chunk {chunk['chunk_number']} from {chunk['url']} - Found {len(keywords)} keywords")
            except Exception as e:
                print(f"Error processing individual result: {e}")
                print(f"Problematic result: {result}")
    
    except Exception as e:
        print(f"Error processing batch: {e}")

async def main():
    chunks = supabase.rpc('get_pages_without_keywords').execute().data
    
    # Process chunks in batches of BATCH_SIZE
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        await process_chunks(batch)
        print(f"Progress: {i+BATCH_SIZE}/{len(chunks)} ({min((i+BATCH_SIZE)/len(chunks)*100, 100):.1f}%)")

if __name__ == "__main__":
    asyncio.run(main())