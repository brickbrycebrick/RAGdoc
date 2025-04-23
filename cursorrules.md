We are building a RAG system that does the following:
1. Scrapes a list of websites for documentation
2. Stores the documentation in a supabase database
3. Retrieves relevant information from the database based on the user's query
4. Provides that information to the LLM as context to answer the user's question

We are using the following technologies:
- Python
- Supabase
- Pydantic
- Crawl4AI

In supabase, the site_pages table has the following columns:
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    title varchar not null,
    summary varchar not null,
    content text not null,  -- Added content column
    metadata jsonb not null default '{}'::jsonb,  -- Added metadata column
    embedding vector(1536),  -- OpenAI embeddings are 1536 dimensions
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    unique(url, chunk_number)

The metadata column has the following structure:
```
{
  "source": "docs.api.com",
  "url_path": "/api/client/python/sitemap.xml",
  "chunk_size": 5000,
  "crawled_at": "..."
}
```

You will always write clean, modular code that is easy to understand and maintain.
