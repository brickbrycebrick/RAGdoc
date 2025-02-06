import os
import sys
import json
import asyncio
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

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI, RateLimitError
from supabase import create_client, Client

load_dotenv()


# openai_client = AsyncOpenAI(
#     api_key=os.getenv("OPENROUTER_API_KEY"),
#     base_url=os.getenv("OPENROUTER_API_BASE")
# )

openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

embeddings_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Rate limiting configuration
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1  # seconds
MAX_RETRY_DELAY = 64    # seconds
BATCH_SIZE = 5          # number of requests to batch together
RATE_LIMIT_WINDOW = 60  # seconds
MAX_REQUESTS_PER_WINDOW = 50  # adjust based on your rate limits

T = TypeVar('T')

class RateLimiter:
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = deque()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.time()
            
            # Remove old requests
            while self.requests and self.requests[0] <= now - self.window_size:
                self.requests.popleft()
            
            # If we're at the limit, wait
            if len(self.requests) >= self.max_requests:
                wait_time = self.requests[0] + self.window_size - now
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    return await self.acquire()
            
            # Add current request
            self.requests.append(now)

class BatchProcessor:
    def __init__(self, batch_size: int, rate_limiter: RateLimiter):
        self.batch_size = batch_size
        self.rate_limiter = rate_limiter
        self.batch_queue: List[Dict] = []
        self.lock = asyncio.Lock()
        self.processed_count = 0
        self.last_process_time = time.time()
        self.processing = False
        self.process_timer = None

    async def add_to_batch(self, item: Dict) -> asyncio.Future:
        future = asyncio.Future()
        async with self.lock:
            print(f"[{time.strftime('%H:%M:%S')}] Adding item to batch. Queue size: {len(self.batch_queue)}")
            self.batch_queue.append({"item": item, "future": future})
            
            # Cancel existing timer if there is one
            if self.process_timer:
                self.process_timer.cancel()
            
            # Schedule batch processing if we have enough items
            if len(self.batch_queue) >= self.batch_size:
                if not self.processing:
                    self.processing = True
                    print(f"[{time.strftime('%H:%M:%S')}] Batch size reached, processing immediately")
                    asyncio.create_task(self._process_and_reset())
            else:
                # Schedule processing after a delay if we don't have a full batch
                self.process_timer = asyncio.create_task(self._schedule_processing())
        
        return future

    async def _schedule_processing(self):
        """Schedule processing after a delay if items are still pending."""
        try:
            await asyncio.sleep(5)  # Wait 5 seconds
            async with self.lock:
                if self.batch_queue and not self.processing:
                    print(f"[{time.strftime('%H:%M:%S')}] Processing incomplete batch after delay")
                    self.processing = True
                    asyncio.create_task(self._process_and_reset())
        except asyncio.CancelledError:
            # Timer was cancelled, do nothing
            pass

    async def _process_and_reset(self):
        try:
            print(f"[{time.strftime('%H:%M:%S')}] Starting batch processing")
            await self.process_batch()
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Error in _process_and_reset: {str(e)}")
        finally:
            async with self.lock:
                self.processing = False
                self.last_process_time = time.time()
                # Check if there are remaining items to process
                if self.batch_queue:
                    print(f"[{time.strftime('%H:%M:%S')}] Items remaining in queue: {len(self.batch_queue)}")
                    self.process_timer = asyncio.create_task(self._schedule_processing())

    async def process_batch(self):
        current_batch = []
        async with self.lock:
            if not self.batch_queue:
                return
            
            # Take all remaining items if less than batch_size
            current_batch = self.batch_queue[:self.batch_size]
            self.batch_queue = self.batch_queue[self.batch_size:]
            print(f"Processing batch of size {len(current_batch)}. Remaining in queue: {len(self.batch_queue)}")

        if not current_batch:
            return

        try:
            print(f"Acquiring rate limiter for batch of size {len(current_batch)}")
            await self.rate_limiter.acquire()
            print(f"Processing batch items...")
            results = await self._process_batch([item["item"] for item in current_batch])
            
            # Set results for futures
            for item, result in zip(current_batch, results):
                if not item["future"].done():
                    item["future"].set_result(result)
                    print(f"Set result for batch item")
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            # Set exception for all futures in the batch
            for item in current_batch:
                if not item["future"].done():
                    item["future"].set_exception(e)
            # Return items to queue if it's a connection error
            if "Connection error" in str(e):
                async with self.lock:
                    print(f"[{time.strftime('%H:%M:%S')}] Returning {len(current_batch)} items to queue due to connection error")
                    self.batch_queue = current_batch + self.batch_queue

class EmbeddingBatchProcessor(BatchProcessor):
    async def _process_batch(self, items: List[Dict]) -> List[List[float]]:
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                print(f"[{time.strftime('%H:%M:%S')}] Processing embedding batch of size {len(items)}, attempt {retry_count + 1}")
                response = await embeddings_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=[item["text"] for item in items]
                )
                print(f"[{time.strftime('%H:%M:%S')}] Successfully got embeddings for {len(response.data)} items")
                return [data.embedding for data in response.data]
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    print(f"[{time.strftime('%H:%M:%S')}] Max retries reached for batch embedding")
                    raise
                delay = min(INITIAL_RETRY_DELAY * (2 ** retry_count) + random.uniform(0, 1), MAX_RETRY_DELAY)
                print(f"[{time.strftime('%H:%M:%S')}] Batch embedding error: {str(e)}. Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)

class TitleSummaryBatchProcessor(BatchProcessor):
    async def _process_batch(self, items: List[Dict]) -> List[Dict[str, str]]:
        try:
            print(f"Processing title/summary batch of size {len(items)}")
            messages = []
            for item in items:
                messages.append([
                    {"role": "system", "content": item["system_prompt"]},
                    {"role": "user", "content": item["user_content"]}
                ])
            
            # Create multiple completion requests using openai_client (OpenRouter)
            print("Creating completion requests...")
            tasks = [
                openai_client.chat.completions.create(
                    model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                    messages=msg,
                    response_format={"type": "json_object"}
                )
                for msg in messages
            ]
            
            print("Awaiting completion responses...")
            responses = await asyncio.gather(*tasks)
            print(f"Successfully processed {len(responses)} title/summary requests")
            return [json.loads(resp.choices[0].message.content) for resp in responses]
        except Exception as e:
            print(f"Error in batch title/summary processing: {str(e)}")
            raise

# Initialize rate limiters and batch processors
embedding_rate_limiter = RateLimiter(RATE_LIMIT_WINDOW, MAX_REQUESTS_PER_WINDOW)
title_summary_rate_limiter = RateLimiter(RATE_LIMIT_WINDOW, MAX_REQUESTS_PER_WINDOW)

embedding_processor = EmbeddingBatchProcessor(BATCH_SIZE, embedding_rate_limiter)
title_summary_processor = TitleSummaryBatchProcessor(BATCH_SIZE, title_summary_rate_limiter)

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4 with batching and retries."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""
    
    item = {
        "system_prompt": system_prompt,
        "user_content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."
    }
    
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            result = await title_summary_processor.add_to_batch(item)
            return await result
        except RateLimitError:
            retry_count += 1
            if retry_count == MAX_RETRIES:
                print(f"Max retries reached for title/summary generation")
                return {"title": "Error processing title", "summary": "Error processing summary"}
            
            delay = min(INITIAL_RETRY_DELAY * (2 ** retry_count) + random.uniform(0, 1), MAX_RETRY_DELAY)
            print(f"Rate limit hit, retrying in {delay:.2f} seconds...")
            await asyncio.sleep(delay)
        except Exception as e:
            print(f"Error getting title and summary: {e}")
            return {"title": "Error processing title", "summary": "Error processing summary"}

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI with batching and retries."""
    item = {"text": text}
    
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            result = await embedding_processor.add_to_batch(item)
            return await result
        except RateLimitError:
            retry_count += 1
            if retry_count == MAX_RETRIES:
                print(f"Max retries reached for embedding generation")
                return [0] * 1536
            
            delay = min(INITIAL_RETRY_DELAY * (2 ** retry_count) + random.uniform(0, 1), MAX_RETRY_DELAY)
            print(f"Rate limit hit, retrying in {delay:.2f} seconds...")
            await asyncio.sleep(delay)
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return [0] * 1536

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    try:
        print(f"Processing chunk {chunk_number} for {url}")
        # Get title and summary
        extracted = await get_title_and_summary(chunk, url)
        print(f"Succesfully processed chunk {chunk_number} of {url}")
        
        # Get embedding
        embedding = await get_embedding(chunk)
        print(f"Got embedding for chunk {chunk_number} of {url}")
        
        # Create metadata
        metadata = {
            "source": urlparse(url).netloc,
            "chunk_size": len(chunk),
            "crawled_at": datetime.now(timezone.utc).isoformat(),
            "url_path": urlparse(url).path
        }
        
        return ProcessedChunk(
            url=url,
            chunk_number=chunk_number,
            title=extracted['title'],
            summary=extracted['summary'],
            content=chunk,
            metadata=metadata,
            embedding=embedding
        )
    except Exception as e:
        print(f"Error processing chunk {chunk_number} for {url}: {e}")
        raise

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
        
        result = supabase.table("site_pages").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

async def process_and_store_document(url: str, markdown: str):
    """Process a document and store its chunks in parallel."""
    try:
        print(f"Starting to process document: {url}")
        # Split into chunks
        chunks = chunk_text(markdown)
        print(f"Split document into {len(chunks)} chunks")
        
        # Process chunks in parallel
        print(f"Processing chunks for {url}")
        tasks = [
            process_chunk(chunk, i, url) 
            for i, chunk in enumerate(chunks)
        ]
        processed_chunks = await asyncio.gather(*tasks)
        print(f"Finished processing {len(processed_chunks)} chunks for {url}")
        
        # Store chunks in parallel
        print(f"Starting to store chunks for {url}")
        insert_tasks = [
            insert_chunk(chunk) 
            for chunk in processed_chunks
        ]
        await asyncio.gather(*insert_tasks)
        print(f"Completed storing all chunks for {url}")
    except Exception as e:
        print(f"Error in process_and_store_document for {url}: {e}")
        raise

async def get_queue_urls(max_attempts: int = 3) -> List[str]:
    """Get URLs that need to be crawled (pending or failed with fewer than max attempts)."""
    try:
        result = supabase.table("crawl_queue").select("url").or_(
            f"status.eq.pending,and(status.eq.failed,attempts.lt.{max_attempts})"
        ).execute()
        
        return [row['url'] for row in result.data]
    except Exception as e:
        print(f"Error getting queue URLs: {e}")
        return []

async def update_url_status(url: str, status: str, error_message: str = None):
    """Update status and increment attempts for a URL in the queue."""
    try:
        data = {
            "status": status,
            "last_attempt": datetime.now(timezone.utc).isoformat(),
            "attempts": supabase.table("crawl_queue").select("attempts").eq("url", url).execute().data[0]["attempts"] + 1,
            "error_message": error_message
        }
        
        result = supabase.table("crawl_queue").update(data).eq("url", url).execute()
        if not result.data:
            print(f"Warning: No rows updated for URL: {url}")
    except Exception as e:
        print(f"Error updating URL status: {e}")

async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )

    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    crawler = AsyncWebCrawler(config=browser_config)
    
    try:
        await crawler.start()
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_url(url: str):
            try:
                async with semaphore:
                    print(f"Starting to process URL: {url}")
                    # Check if URL has been successfully crawled before
                    existing = supabase.table("site_pages").select("url").eq("url", url).execute()
                    if existing.data:
                        print(f"Skipping {url} - already crawled")
                        await update_url_status(url, "completed")
                        return

                    result = await crawler.arun(
                        url=url,
                        config=crawl_config,
                        session_id="session1"
                    )
                    
                    if result.success:
                        print(f"Successfully crawled: {url}")
                        print(f"Starting document processing for: {url}")
                        await process_and_store_document(url, result.markdown_v2.raw_markdown)
                        print(f"Completed processing document for: {url}")
                        await update_url_status(url, "completed")
                        print(f"Updated status to completed for: {url}")
                    else:
                        print(f"Failed: {url} - Error: {result.error_message}")
                        await update_url_status(url, "failed", result.error_message)
            except Exception as e:
                error_msg = str(e)
                print(f"Error processing {url}: {error_msg}")
                await update_url_status(url, "failed", error_msg)
                
        print(f"Starting to process {len(urls)} URLs")
        await asyncio.gather(*[process_url(url) for url in urls])
        print("Completed processing all URLs")
    finally:
        try:
            await crawler.close()
            print("Crawler closed successfully")
        except Exception as e:
            print(f"Warning: Browser cleanup error: {e}")

def get_site_urls(base_url: str) -> List[str]:
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
    
    for path in sitemap_paths:
        sitemap_url = base_url + path
        try:
            response = requests.get(sitemap_url)
            response.raise_for_status()
            
            # Parse the XML
            root = ElementTree.fromstring(response.content)
            
            # Extract all URLs from the sitemap
            namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
            
            if urls:
                print(f"Found sitemap at: {sitemap_url}")
                return urls
                
        except requests.RequestException:
            continue
        except ElementTree.ParseError:
            continue
    
    # If no sitemap found or no URLs extracted, return the base URL
    print(f"No sitemap found. Using base URL: {base_url}")
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
            print(f"Discovering links from: {current_url} (depth {depth})")
            
            try:
                result = await crawler.arun(
                    url=current_url,
                    config=crawl_config,
                    session_id="discovery"
                )
                
                if result.success:
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
                    print(f"Failed to crawl {current_url}: {result.error_message}")
            
            except Exception as e:
                print(f"Error crawling {current_url}: {str(e)}")
                continue
    
    # Return list of discovered URLs
    return list(visited_urls)

async def main():
    # Get URLs from command line or use default
    base_urls = sys.argv[1:] if len(sys.argv) > 1 else [
        #"https://supabase.com/docs",
        "https://e2b.dev/docs",
        "https://apify.com/templates",
        "https://ai.pydantic.dev/",
        "https://crawl4ai.com/mkdocs/"

    ]
    
    # Process all base URLs to get their sitemaps or crawl recursively
    all_urls = []
    for base_url in base_urls:
        # Try sitemap first, if no results then try recursive crawling
        urls = get_site_urls(base_url)
        if len(urls) <= 1:  # Only base URL was returned
            print(f"No sitemap found for {base_url}, trying recursive crawling...")
            urls = await get_site_urls_recursive(base_url)
        
        print(f"Found {len(urls)} URLs to crawl for {base_url}")
        all_urls.extend(urls)
    
    # Add URLs to queue
    await add_urls_to_queue(all_urls)
    
    # Get URLs that need to be crawled
    urls_to_crawl = await get_queue_urls()
    print(f"Processing {len(urls_to_crawl)} URLs from queue")
    
    if urls_to_crawl:
        await crawl_parallel(urls_to_crawl)
    else:
        print("No URLs in queue need processing")

async def add_urls_to_queue(urls: List[str]):
    """Add URLs to the crawl queue if they don't exist."""
    try:
        # Get ALL existing URLs from both tables with a single query using 'in' filter
        existing_result = supabase.table("crawl_queue").select("url").execute()
        existing_urls = {row['url'] for row in existing_result.data}
        
        # Create a list of truly new URLs, filtering out duplicates
        new_urls = []
        for url in urls:
            if (url 
                and url not in existing_urls  # Check against existing URLs
                and not url.endswith('/undefined')
                and 'undefined' not in url
                and url not in new_urls):  # Ensure no duplicates within the new batch
                new_urls.append(url)
        
        if not new_urls:
            print("No new URLs to add to queue")
            return
        
        print(f"Adding {len(new_urls)} new URLs to queue")
        
        # Process URLs one at a time to ensure no duplicates
        successful_adds = 0
        for url in new_urls:
            try:
                # Check one more time if URL exists before inserting
                check_existing = supabase.table("crawl_queue").select("url").eq("url", url).execute()
                if not check_existing.data:
                    queue_data = {
                        "url": url,
                        "status": "pending",
                        "attempts": 0,
                        "last_attempt": None,
                        "error_message": None
                    }
                    result = supabase.table("crawl_queue").insert(queue_data).execute()
                    if result.data:
                        successful_adds += 1
            except Exception as e:
                if "duplicate key value" not in str(e):  # Ignore duplicate key errors
                    print(f"Error adding URL {url}: {e}")
                continue
        
        print(f"Successfully added {successful_adds} new URLs to queue")
                
    except Exception as e:
        print(f"Error in add_urls_to_queue: {e}")

if __name__ == "__main__":
    asyncio.run(main())