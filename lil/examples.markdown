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
    
    |> enqueue_page:one[CrawlRequest] > CrawlQueue
    CrawlQueue >| @t > CrawlThrottler > [next_page:CrawlRequest] > PageParser > (
        > CrawlThrottler > batch[found_urls: CrawlRequest] > CrawlQueue
        > one[crawled_data: CrawledPage] lag[t ~~ 1s] > CrawledPagesDB
    )
)
```


### Recommender system

Site collects events from visitors. Then, 

```
User: schema (
    id: int [id]
    name: str
)

Session: schema (
    id: int [id]
    user_id: User.id?
    session_start: timestamp
    session_end: timestamp?
)


Product: schema (
    id: int [id]
    name: str
)

View: schema(
    view_id: int [id]
    product_id: Product.id?
    session_id: Session.id
    ts: timestamp  [ts <= Session.session_end, ts >= Session.session_start]
)

Purchase: schema(
    order_id: int [id]
    cart: List[Product.id]
    session_id: Session.id
    ts: timestamp
)


Browser: [external] app (
    < tracked_views: stream[View] <| ...
    < tracked_purchases: stream[Purchase] <| ...
)

EventCollector: node (
    EventWriter: component
    
    tracked_views > EventWriter > eventDB.views
    tracked_purchases > EventWriter > eventDB.purchases
    
    get_recommendations > List[Product.id] ---- user: User.id <|
)

DataScienceMachine: node (
    triggered[daily] > model_update 
    EventDB.views >
)


EventDB: storage.postgre (
    Size ~~ 1Tb
    
)


ModelDB: storage.kv (
    Size ~~ 1Gb
    precomputed_recs : storage[schema] (
        user_id: int [key]
        recommendations: List[Product.id]
    )
)
```