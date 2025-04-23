import os
import sys
import json
import asyncio
import aiohttp
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any, TypeVar, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
import random
import time
from collections import deque

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI, RateLimitError
from supabase.lib.client_options import ClientOptions
from postgrest import AsyncPostgrestClient
from supabase import create_async_client, AsyncClient

load_dotenv()

# Simplified configuration
MAX_CONCURRENT_PROCESSES = 10
MAX_CONCURRENT_CRAWLS = 3

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Global variables
supabase: AsyncClient = None

async def initialize_supabase() -> AsyncClient:
    try:
        options = ClientOptions(
            postgrest_client_timeout=10,
            storage_client_timeout=10
        )
        
        client = await create_async_client(
            supabase_url=os.getenv("SUPABASE_URL"),
            supabase_key=os.getenv("SUPABASE_SERVICE_KEY"),
            options=options
        )
        
        # Test the connection
        await client.table("crawl_queue").select("*").limit(1).execute()
        return client
    except Exception as e:
        logging.error(f"Error initializing Supabase client: {e}")
        raise

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4 with batching and retries."""
    
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""
    
    item = {
        "system_prompt": system_prompt,
        "user_content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."  # Send first 1000 chars for context
    }
    
    retry_count = 0
    while retry_count < 5:
        try:
            response = await openai_client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4"),
                messages=[
                    {"role": "system", "content": item["system_prompt"]},
                    {"role": "user", "content": item["user_content"]}
                ],
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            if isinstance(result, dict) and "title" in result and "summary" in result:
                return result
            raise ValueError("Invalid response format from title_summary_processor")
        except Exception as e:
            retry_count += 1
            if retry_count == 5:
                logging.error(f"Max retries reached for title/summary generation: {e}")
                return {"title": "Error processing title", "summary": "Error processing summary"}
            
            delay = min(1 * (2 ** retry_count) + random.uniform(0, 1), 64)
            logging.error(f"Error getting title and summary, retrying in {delay:.2f} seconds...")
            await asyncio.sleep(delay)

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI with batching and retries."""
    retry_count = 0
    while retry_count < 5:
        try:
            response = await openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=[text]
            )
            return response.data[0].embedding
        except Exception as e:
            retry_count += 1
            if retry_count == 5:
                logging.error(f"Max retries reached for embedding generation: {e}")
                return [0] * 1536
            
            delay = min(1 * (2 ** retry_count) + random.uniform(0, 1), 64)
            logging.error(f"Error getting embedding, retrying in {delay:.2f} seconds...")
            await asyncio.sleep(delay)

async def process_and_store_document(url: str, markdown: str):
    chunks = chunk_text(markdown)
    for i, chunk in enumerate(chunks):
        logging.info(f"Processing chunk {i} for {url}")
        if not chunk.strip():
            logging.warning(f"Empty chunk detected at position {i}")
            continue
        
        extracted = await get_title_and_summary(chunk, url)
        logging.debug(f"Title extraction result: {extracted}")
        if 'title' not in extracted or 'summary' not in extracted:
            logging.error(f"Invalid extraction format for {url} chunk {i}")
            continue
        
        embedding = await get_embedding(chunk)
        logging.info(f"Storing chunk {i} with title: {extracted['title'][:50]}...")
        await insert_chunk(ProcessedChunk(
            url=url,
            chunk_number=i,
            title=extracted['title'],
            summary=extracted['summary'],
            content=chunk,
            metadata={},
            embedding=embedding
        ))
    return True

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        await supabase.table("site_pages").insert(data).execute()
    except Exception as e:
        logging.error(f"Error inserting chunk {chunk.chunk_number} for {chunk.url}: {e}")
        raise

async def update_url_status(url: str, status: str, error_message: str = None):
    try:
        logging.info(f"Attempting to update status for {url} to {status}")
        
        query_result = supabase.table("crawl_queue").select("attempts").eq("url", url)
        logging.debug(f"Query object: {query_result}")

        single_result = query_result.single()
        logging.debug(f"Single result object: {single_result}")

        execute_result = await single_result.execute()
        logging.info(f"Execute result: {execute_result}")
        logging.debug(f"Result data structure: {dir(execute_result)}")

        current_attempts = execute_result.data['attempts']
        logging.info(f"Current attempts: {current_attempts}")

        data = {
            "status": status,
            "last_attempt": datetime.now(timezone.utc).isoformat(),
            "attempts": current_attempts + 1
        }
        
        if error_message:
            data["error_message"] = error_message
            
        update_result = await supabase.table("crawl_queue").update(data).eq("url", url).execute()
        logging.info(f"Update status successful for {url}")
        
    except Exception as e:
        logging.error(f"Error updating status for {url}: {str(e)}")
        raise

async def add_urls_to_queue(urls: List[str]):
    """Add URLs to the crawl queue if they don't exist."""
    try:
        # Get existing URLs in a single query
        existing_urls = await supabase.table("crawl_queue").select("url").in_("url", urls).execute()
        existing_set = {item['url'] for item in existing_urls.data}
        
        # Filter out existing URLs
        new_urls = [url for url in urls if url not in existing_set]
        
        if new_urls:
            # Prepare batch data
            batch_data = [
                {
                    "url": url,
                    "status": "pending",
                    "attempts": 0,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "last_attempt": datetime.now(timezone.utc).isoformat()
                }
                for url in new_urls
            ]
            
            # Insert in batches
            for i in range(0, len(batch_data), 10):
                batch = batch_data[i:i + 10]
                await supabase.table("crawl_queue").insert(batch).execute()
                
        return len(new_urls)
    except Exception as e:
        logging.error(f"Error adding URLs to queue: {e}")
        raise

# async def get_queue_urls(max_attempts: int = 40):
#     result = await supabase.from_('crawl_queue').select('url').eq('attempts', max_attempts).execute()
#     return [row['url'] for row in result.data]

async def get_queue_urls(max_attempts: int = 30) -> List[str]:
    """Get pending URLs or failed URLs with attempts < max_attempts"""
    try:
        logging.info(f"🔍 Fetching queue URLs (max_attempts={max_attempts})")

        # Execute raw query with parameter binding
        result = await supabase.from_('crawl_queue').select('url').lt('attempts', max_attempts).execute()

        print(result)
        
        # Validate response structure
        if not hasattr(result, 'data'):
            logging.error("❌ Unexpected response format from Supabase")
            logging.debug(f"🔎 Response keys: {list(result.keys())}")
            return []

        urls = [row['url'] for row in result.data]
        logging.info(f"✅ Found {len(urls)} URLs in queue")
        
        if urls:
            logging.debug(f"🔗 First 5 URLs: {urls[:5]}")
        else:
            logging.warning("⚠️ No URLs found matching criteria")
            
        return urls
        
    except Exception as e:
        logging.error(f"❌ Queue query error: {str(e)}", exc_info=True)
        return []

async def crawl_parallel(urls: List[str]):
    """Crawl multiple URLs in parallel with optimized concurrency control."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )

    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_CRAWLS)
        
        async def process_url(url: str):
            try:
                async with semaphore:
                    # Check for existing crawl in batch
                    response = await supabase.table("site_pages").select("url").eq("url", url).execute()
                    if response.data:
                        await update_url_status(url, "completed")
                        return

                    result = await crawler.arun(crawl_config)
                    if result and result.success:
                        if await process_and_store_document(url, result.markdown_v2.raw_markdown):
                            await update_url_status(url, "completed")
                    else:
                        await update_url_status(url, "failed", result.error_message)
            except Exception as e:
                await update_url_status(url, "failed", str(e))
        
        await asyncio.gather(*[process_url(url) for url in urls])

async def get_site_urls(base_url: str) -> List[str]:
    """Get URLs from a website's sitemap.xml or return the base URL if no sitemap exists.
    
    Args:
        base_url: The website's base URL (e.g., 'https://ai.pydantic.dev')
        
    Returns:
        List of URLs to crawl
    """
    # Normalize the base URL
    if base_url.endswith('/'):
        base_url = base_url[:-1]
    
    # Try common sitemap paths
    sitemap_paths = [
        '/sitemap.xml',
        '/sitemap_index.xml',
        '/sitemap/',
        '/sitemap/sitemap.xml'
    ]
    
    async def try_sitemap(sitemap_url: str) -> Optional[List[str]]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(sitemap_url) as response:
                    if response.status != 200:
                        return None
                    
                    content = await response.text()
                    # Parse the XML
                    root = ElementTree.fromstring(content)
                    
                    # Extract all URLs from the sitemap
                    namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
                    urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
                    
                    if urls:
                        logging.info(f"Found sitemap at: {sitemap_url}")
                        return urls
                        
        except (aiohttp.ClientError, ElementTree.ParseError) as e:
            logging.error(f"Error processing sitemap {sitemap_url}: {e}")
            return None
    
    for path in sitemap_paths:
        sitemap_url = base_url + path
        urls = await try_sitemap(sitemap_url)
        if urls:
            return urls
    
    # If no sitemap found or no URLs extracted, return the base URL
    logging.info(f"No sitemap found. Using base URL: {base_url}")
    return [base_url]

async def get_site_urls_recursive(base_url: str, max_depth: int = 2) -> List[str]:
    """Get URLs from a website by recursively crawling internal links using crawl4ai.
    
    Args:
        base_url: The website's base URL (e.g., 'https://ai.pydantic.dev')
        max_depth: Maximum depth for recursive crawling (default: 2)
        
    Returns:
        List of URLs to crawl
    """
    # Normalize the base URL
    if base_url.endswith('/'):
        base_url = base_url[:-1]
    
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        visited_urls = set()
        urls_to_crawl = [(base_url, 0)]  # (url, depth) pairs
        base_domain = urlparse(base_url).netloc
        
        while urls_to_crawl:
            current_url, depth = urls_to_crawl.pop(0)
            
            # Skip if URL already visited or depth exceeded
            if current_url in visited_urls or depth > max_depth:
                continue
            
            visited_urls.add(current_url)
            logging.info(f"Discovering links from: {current_url} (depth {depth})")
            
            try:
                result = await crawler.arun(current_url, crawl_config)
                if result and result.success:
                    # Process both internal and external links
                    for link_type in ['internal', 'external']:
                        for link in result.links.get(link_type, []):
                            link_url = link['href']
                            
                            # Normalize the URL
                            if link_url.endswith('/'):
                                link_url = link_url[:-1]
                            
                            # Skip invalid or already processed URLs
                            if (not link_url or 
                                'undefined' in link_url or 
                                link_url in visited_urls):
                                continue
                            
                            # Check if URL belongs to same domain
                            link_domain = urlparse(link_url).netloc
                            if link_domain == base_domain:
                                urls_to_crawl.append((link_url, depth + 1))
                else:
                    logging.error(f"Failed to crawl {current_url}: {result.error_message}")
            
            except Exception as e:
                logging.error(f"Error crawling {current_url}: {str(e)}")
                continue
    
    # Return list of discovered URLs
    return list(visited_urls)

async def with_error_logging(coro, context=""):
    try:
        return await coro
    except Exception as e:
        logging.error(f"Failed {context}: {str(e)}")
        raise

async def main():
    try:
        # Initialize Supabase client
        global supabase
        logging.info("Initializing Supabase client...")
        supabase = await initialize_supabase()
        logging.info("Supabase client initialized successfully")
        
        # Get URLs from command line or use default
        base_urls = sys.argv[1:] if len(sys.argv) > 1 else ["https://docs.cohere.com/"]
        
        all_urls = []
        for base_url in base_urls:
            # Try to get URLs from sitemap first
            urls = await get_site_urls(base_url)
            if not urls:
                # If no sitemap found, try recursive crawling
                urls = await get_site_urls_recursive(base_url)
            all_urls.extend(urls)
        
        if not all_urls:
            logging.info("No URLs found to crawl")
            return
        
        # Add URLs to queue
        num_added = await add_urls_to_queue(all_urls)
        logging.info(f"Added {num_added} new URLs to queue")
        
        # Get URLs to crawl
        urls_to_crawl = await get_queue_urls()
        if not urls_to_crawl:
            logging.info("No URLs in queue to crawl")
            return
        
        # Start crawling
        await crawl_parallel(urls_to_crawl)
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())