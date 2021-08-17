Each symbol has its own semantics.

### Types and hierarchy

`()` and `{}` denote hierarchy. Since they look very similar, they can be used interchangeably.

`a: B`, `B a` means that `a` belongs to an object we are describing, and we also assign some metadata `B` to `a`.
`B[a]` means that there is `B of a` belonging to an object describing. 

Primitive and complex types are borrowed from Python. `Map` can be defined as `Map[a, b]` or 
`a -> b` (Scala-style).

Example:

```
MyMovieLibrary: schema(
    id: int
    
    owner_name: str
    
    MoveEntry: schema(
        id: int
        movie_name: str
    )
    Review: schema(
        rating: float
        review: str
    )
    
    movies: List[MoveEntry.id]
    my_ratings: MoveEntry.id -> Review    
)
```

### Timing

`@`, `@now`, `@current` - current moment (when executing code). Assumed implicitly if timing is not defined.

`@t` : at moment t. When output is written, when input is received, state of object at moment t

`@t1..t2` : between moments t1 and t2.

`lag[t1 .. t2]` : time from moment t1 to t2. `lag[t]` : time from t1 to now

`obj'` : changes of obj.

`@t +- N u` : `N` units of time `u` after/before `t` (`t` may be omitted if it unambiguous)

` \ ` : distributed

`obj @ t: last` - state of obj at last known/available moment `t`

Examples:

`obj:X @t` or `obj @t :X` means object `obj` of type `X` at moment of time `t`

`obj'@t1..t2` : changes (update message) of obj from moment t1 to moment t2

`obj@-10s` : state of `obj` 10 seconds before current moment

`stream [obj:X @t1 ] @t2` : state of object `obj` at moment `t1` of type `X` appeared on stream at moment `t2`

`stream[obj @-10s : X] @t` : state of object `obj` at moment `t1` of type `X` appeared on stream with 10-second lag. 

`stream [obj: X] @+10s` : state of object `obj` will appear on stream in 10 seconds 

`stream[X] \ N(A, B)` means 'time between events in the stream is distributed normally'

`stream[X] \ Exp(A)` means 'time between events in the stream is distributed exponentially'

`stream[X] \ every N u` means 'time between events in the stream is `N` units of `u`'

`lag[in .. out]` means time from `in` to `out`

### Comparisons and restrictions

`>` is "more than"

`<` and "less than"

`=` "equals strictly"

`%` divided evenly

`~` means "approximately" or "typical".  Typical is >95% of the cases

`>~`, `~<` means "typically more/less than".

`~=` "typically equal to"

`~N` means "a number around N, +- 25%". If N is a formula, then linear proportion is assumed.

`x: Y [id] [pk] [unique]` - x of type Y is unique identifier / part of a primary key / unique among objects Y

`len[...]` - number of elements, `len` number of records, `unique[..]` unique elements

Examples:

`a: int [a > 0]` : a is a positive integer

`a : int ~~ 3` or `a : int [a ~~ 3]` : a is around 3

`len ~~ 5 per unique[Y]` means "there are 5 records per each unique Y"

`a = ~5` is "a is greater than, let's say, around 5"

`b:batch[obj:Obj@t]  lag[t] ~< 10s` state of object `obj` of type `Obj` in batch `b` is 10 seconds old or less.    

## Data flow

`>` and `<` show direction of the flow. `|` means "active side" or "trigger".

`/` describes paths. `*` is a wildcard. So, `storageA/dir1/dir2/*/files` means
"files from storageA at path dir1/dir2/<anything>". 

`a -c-> b` or `a |-c-> b` means "data `c` flows from `a` to `b`.

`a -c->| b` means `a` triggers with parameter `b` / writes `c` to `b`.

`a - -c- ->| b` or `a |- -c- -> b` means `a` reads data data `c` from `b`.

`a - -d- > b/c` means `a` calls `b` with argument `c` and gets `d` in response.

## Structural pieces

### Data

`schema` is a data type. It may contain data fields, provide some info about restrictions and expected values.
Schemas can be nested, showing that inner schema cannot be used without / has a strong contextual connection to the
outer schema.

Schema fields can then be used as a data type:

```
schema Product(
    id: int
    name: str
)

schema Order(
    orderId: int
    products: List[Product.id]
)
```

If a schema is used as a data type inside another schema, then it is embedded inside it. You may access its fields
directly, or, if field name has `*` in it - then substituting `*` with the field name.

### Software
We can define
- `x: Node` - machine `a`
- `d: storage`s ; `a:app`s (`e: service`s and `p: pipeline`s) in a node, along with connections between them.
- `cmp: component`s and `d: storage`s in an app, service or pipeline, along with connections between them.
- `cmp: component`s, `d: storage`s, and their connections in a component if we want to break it down.

`node` is a machine, virtual or physical, with its own CPU, memory and disk resources. 
We can define inside:
- `app`s and `storage`s
- connections between them and between external resources (using `-->` and `<---`)
- resource limitation (`cpu: N`, `disk: M`, `ram: N`)
- parallelism level `parallel: Q`. In that case, requests to that node can be sharded: `parallel: 5 by obj.id`
Sharding should be deterministic if node is stateful. If sharding is not stated, then distribution of requests is
arbitrary/random.

`app` is a running software on a `node`. We can define:
- open inputs `>` and outputs `<`, of the app. 
- `component`s and `storage`s inside it
- Connections between them and between external resources (using `-->` and `<---`).
- Parallelism level with sharding option `parallel: R by obj.id`.

Apps can be:
- `app/pipeline`: instantiated by schedule/external call. Shuts down after finishing work. ETL pipeline job or
ML model training is a pipeline.
- `app/service` : instantiated on deployment and expected to keep running. Web server is a service; crawler is a service.
Data collector or ETL stream transformer is a service.

#### Storage
`storage` is something that holds arrays of data, and supports "insert" and "select"  operations. We can define:
- resources (`cpu: N`, `disk: M`, `ram: N`)
- if distributed, then parallelism `parallel: Q` and `replication: R` 
- Schemas inside. Each storage can hold objects of multiple schemas.
- Schema partition keys `schemaA: partitions[pk1, pk2, pk3]`
- Data retention `retention[10d]` or `retention[1000 records]`
- Which moment can the states of stored objects be queried:
  - "Ground truth", "latest object state": `@now`
  - One of the latest / last known available object state `@last`
  - Periodic snapshots `@t [t: now .. -20d % 8h]`
  - Update stream `@t [t: now .. -1d]` 
- Default database settings `schemaA: group[groupname] table[tname] queue [qname]`

It is assumed that each `storage` comes will all connectors, data access layers and validators.
So, instead of running SQL queries, you call it specifying partition keys, or just write objects into it.
That's why | in component definition means "some programmatic interface"

Partitions and primary keys can be defined outside:
` --- stream[x: Type] ---> storageA / x.value1:partitionKey1 / x.value2: partitionKey2 / x.pkval: primaryKeyColumn`

`medium` is something on top of the data stream which modifies/defines it in some way.
It can ba a request splitter (`medium/shard`), a transport (`medium/json` or `medium/rest`), or a modifier app (`medium/cache`, `medium/queue`). 

`component` is an `app` piece that transforms data.
`component/library` `component/external`
- It is a large piece of logic which is not shared across different apps (usually).
- `component` has inputs, outputs and triggers as well, and defines some SLAs and global properties.
You do not care what's happening inside, unless it really matters or you are its developer.
- You may break it down into a set of storages and (sub)components (treat it as an app), 
and define relationships between them and your component.

Component can be triggered externally by data appearing at the input, or internally by a timer.

Inputs and outputs formats just wrapped schemas: 
- `Q: stream[x@(t-10s): A]@t`. Q is an infinite sequence of objects of type `A`, with the state object x at moment `t-10s`
 which came to the stream at moment `t` (i.e. there is a 10 second lag)
- `Q: batch[x: A]@t`. Q is a finite sequence of objects of type `A`. Batch can be a file, a query result or just a set of
objects.
- `Q: single[A]` or `Q: A`. Q is a singular A (~singleton). It is expected that these objects do not support lineage,
and they exist "just now" or "always". It can be just a singular object (REST API request), or a config file.


- `every T u >`: schedules
- `> every X records > `: stream to batch
- `( )` : can be called simultaneously
- ` --c--` : `c` is being transferred
- ` > ( ) for a in Q > ` : batch / stream to singleton 
- `a -c- > b` or `a > b:c` or `b < a:c` : `a` writes/sends data `c` to `b`
- `a - -c- - > b` or `a |> b` : `b` requests data `c` from `a`.
Function call, or send and await. component to component
- `|...` : storage read/write query, or "do actively"

`|>` actively read. component to storage / to storage ("get response from")
`STORAGE |> inName:mode[X}`

`outName: mode[X] > COMPONENT |> inName:mode[X}`

- `COMPONENT.out > inName:mode[X}` await request, and start doing stuff when it happened. component to component

`<|` actively write. component to storage (sometimes component; "send and forget")
`STORAGE |< outName:mode[X}`

`COMPONENT < outName:mode[X}`

open input with open active output= api definition. Await request, send response. funName(X)=Y
`> funName:mode[Request] / mode[Response]`

`> funName:mode[Request] --mode[Response]-->`
`> funName:mode[Request] >| mode[Response]`

open output = external api call (defined in parent). Send request, await response, do stuff.
`< callName: mode[Request] < ....`
`> callName / mode[Response] > ....`
`b.callName > a.funName > b.CallName`


`< callName: mode[Request] < ....`
`> callName / mode[Response] > ....`
`...  < a.funName: mode[Response] < callName : mode[Request] < ...`

```
|> inName > transform1 > transform2
```
