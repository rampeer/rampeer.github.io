{% include menu.markdown %}

## Examples

### Web crawler

Allows user / external input to add pages to index.

`CrawlQueue` keeps queue of incoming crawling requests

`CrawlThrottler` prevents crawling recently crawled pages.

`PageParser` is the component that does all the heavy-lifting. 

We can estimate that 1M crawled pages will require about 100Mb, and if our crawl speed estimation is correct,
it will take around 2 days to collect that volume of data. Fun!

```
CrawledPage: schema (
    id: int [id]
    ts: timestamp
    url: str.url [index: filter]
    parsed_data: str.json
    
    size ~~ 100 bytes
)

CrawlRequest: schema(
    url: url
)

Сrawler: service (
    # No need to define internal functions - they are apparent
    PageParser: component [parallel: 5]
    
    |> enqueue_page:one[CrawlRequest] >
    
    > next_page:CrawlRequest --> CrawlThrottler --> PageParser@t -- (
      -- CrawlThrottler --> found_urls: batch[CrawlRequest] > 
      -- crawled_data: one[CrawledPage]  lag[t ~~ 1s]  >
    )
    
    
    CrawlQueue: storage.queue (
        > enqueue_page, found_urls
        < next_page
    )
    
    CrawledPagesDB: storage.mongodb [tablename: "crawled"] (
        > crawled_data
        schema: CrawledPage
        size ~~ 100Mb
        records ~~ 1M
    )
    
    CrawlThrottler: cache[ttl=1day]
)
```


### Recommender system

