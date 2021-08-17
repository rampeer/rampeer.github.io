# Alphabet


| Symbol       | Meaning                                                                                              | Example                                                        |
|--------------|------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| :            | defines a <property-property type> pair                                                              | a is an integer<br>`a: int`                                    |
| =            | defines a <property-property value> pair                                                             | hostname is "google.com"<br>`hostname = google.com`            |
| /            | defines paths, database names, table names, or partitions                                            | objects inside storage `db`'s database `my_db`'s table `messages` with partition key `request.date`:<br> `db: storage`<br> `db/my_db/messages/ partition[messagedate_pkey: request.date]`                                                              |
| () and .     | parentheses define hierarchy, and dot is used to access it                                           |                                                                |
| []           | brackets define complex types, or add meta-information to an object that they follow. Syntax:<br>[modifierValue]<br>[modifierType: modifierValue]<br>[modifierType: modifierValue1, modifierValue2]<br>modifierType[modifierValue]<br>modifierType[modifierSubtype:modifierValue] |  #a is a stream of messages in json format. They are coming through a HTTP Websocket.<br> `> a: stream [format:json] source[http: websocket]`                                                             |
| > < =        | strict comparisons that define domain of values                                                      | age is a positive int<br>age: int [a >= 0]                   |
| >~ <~ ~ ~=     | fuzzy comparisons that define reasonable domain of values (typical values)                           | age is typically between 18 and 30<br>a: int [18 ~< a ~< 30] |
| ~~           | order of magnitude     | size ~~ 10Gb |
| >--><br><--< | data flow arrows                                                                                     | a writes to storage b in batch<br> a ----> b                                                              |
| _            | Meta-commands (include, import, ...)                                                                           |                                                                |
| !            | Deploy mode                                                                                                    |                                                                |
| ^            | Version or version draft (=proposal)                                                                                          |                                                                |
| #            | Docstring                                                                                                 |                                                                |
| \             | Distribution                                                                                                     |                                                                |
| "x" | treat x as a literal | | 
| $x | treat x as a reference 
| @t | at moment x
| x' | change/new value of x

`modifierSubtype`, `modifierType`, pieces of full object qualifier (`.` path, like `a.b.c`) can be omitted if
 they can be resolved unambiguously. `"` and `$` can be skipped as well. Values will be 

# Abstractions

## Schema

Schema defines data structure. It is a state of an object (typically on business level), which is valid for multiple
applications and services. Data from schemas may be scattered on several databases. Schemas may be nested when
children belong only to one parent.

```
User: schema (
    [version: 5]
    [maintainer: test@gmail.com]
    
    user_id: int [key]
    name: str
    age: int
    purchases: List[Purchase]
    Purchase: schema (
        order_id: int [key]
        items: List[Product]
    )
)

Product: schema (
    product_id: int [key]
    name: str
    ts: timestamp
)
```

We may reference schema's fields as data types to add semantics. 
For example, by writing `oid: Purchase.order_id` we are saying that variable `oid` has type `int`, 
but actually it is a reference to `Purchase`'s `order_id`.

Definitions can be split into multiple files. Deployment (`~`) or variants (`!`) modify existing values.
To "extend" definition outside of the original file, you just start with _.

This adds two fields to the Product definition

```
_Product (
    image_url: str
    page_url: str
)
```

#### schema.config
subtype of schema that describes configuration file format.

For that subtype, modifier `FORMAT[filepath]` can be set for the schema itself to specify  along with
`[env: ENVVAR_NAME]` or `[config: <config file path and option name>]`, or hardcoded values `hostname: str = sample` 
for each field. 

`FORMAT` is one of: json, yaml, ini, xml, toml

Example:

```
UserMongoCreds: config json[env: "MONGO_CRED_PATH"] (
    username: str
    password: str [env: SECRET_MONGO_PASSWORD]
    hostname: str
    port: int
    
    hostname~prod = "prod-user-mongo"
    hostname~dev = "prod-user-mongo"
)
```

#### schema.enum
 
Advanced enum. It can be used to assign non-trivial values to enums.
It is useful when different components have different enums, and they must be mapped to each other.

Example:

```
GenderInDB: enum(
    male = "male"
    female = "female"
)

GenderOnSite: enum(
    adult
    _as GenderInDB
    adult.male = "m" same[_.male]
    adult.female = "f" same[_.female]
    adult.unisex = "u" similar[_.female, _.male]
    kids = "k"
    kids.girls = "g" similar[_.female]
    kids.boys = "b" similar[_.male]
)
```

### Conditions

Conditions may use: 
- `> < =`, to specify valid ranges
- `>~ ~< =~` to specify typical ranges (reasonable bounds)
- `~~` to define order of magnitude or asymptotic line 
- `=` or `=~` to define expected value or typical value
- `start..` , `..end`, `start..end` to define ranges, for example `0..100`
- `%` to specify divisibility. "a is ranges from 0 to 1000 with step 10" is written as `a: int [0..1000, a % 10]`
- `len`, `start.. end` when applied to strings, specify string length or substrings (which can then be used in conditions) 


### Basic data types and units

int
- B|Billions
- M|Millions
- K|Thousands

float

str
str.nonempty
str.identifier (valid program variable name)

str.email
str.password
str.phone
str.username

str.url
str.hostname
str.filename

str.json
str.xml
str.base64

blob
blob.picture
blob.sound
blob.model


timestamp
date
datetime

timespan
ms|milliseconds
s|seconds
m|minutes
h|hours
d|days
w|weeks
mon|months

size
b|byte
Kb, Mb, Gb, Tb 

### Complex types

- `List[x]`, `Map[x, y]`, `Array[x]` are complex data types. They wrap around x, adding extra properties to the `field_name`
(`field_name.len` for all complex types; and `field_name.keys`/`field_name.values` for `Map`).

- `embedded[SchemaName: fields1, fieldsN]` includes part of another schema into that one. 
If no fields are specified, then all are used. Asterisk `*` in the field name is used as a placeholder. 
It define a place where the original variable's name is placed 
(if there is no asterisk, embedded schema's field names are appended to the `field_name`).
  - For example, 
`user_*: Embedded[User: name, age]` becomes `user_name: User.name` , `user_age: User.age`

- `enum[EnumName: value1, value2, ... valueN]` defines enum `EnumName` with a list of possible values.

### Modifiers

Modifiers supply with meta-information. They look like

- `modifierType[modifierSubtype: modifierValue]`
  - or `modifierType[modifierSubtype: modifierValue, sub2: value2, sub3: value3a, value3b, value3c]`
  - or `modifierType[modifierSubtype: value1, value2, value3]`
- `[modifierSubtype: modifierValue]`
  - or `[modifierSubtype: value1, value2, value3]`
- `modifierType[modifierValue]`
  - or`modifierType[value1, value2, value3]`
- or just `[modifierValue]`

They go right after the object definition:

`field_name: fieldtype [modifierSubtype: modifierValue]`

or

`appname: app [modifierSubtype: modifierValue] ( ... )`

they often replace `=`. Example above can be rewritten as

```
appname: app (
    modifierSubtype: modifierValue
    ...
)
```  

#### Database-related modifiers

- `[partition]` informs that this field is a partition key. Rows with the same values will end up in the same node. 

- `[key]` flags fields that constitute table's primary key.

- `[unique]` says that values in that field is unique for each object

- `index[<index_name>: <range|equal|...>]` says that this field is a part of index `<index_name>`,
and in that index it can be effectively queried using "equals", "not equals" and "in" (`equal`) or using ">" and "<"
or "between" (`range`). Custom index types are supported; for example, to define database-specific index
(such as fulltext-search indices).

#### Validation modifiers

- `subset[P]`, `superset[P]`, `all[P]`, where P is a reference be another schema's field.
`subset`, `superset` say that the set of all values of field at hand is a subset/superset of P. 
`all` means that for each value of P there is a corresponding record in the schema at hand.
  - Example: a map which contains some Transaction id's as a key:
  - `prices:Map[Transaction.id, int] subset[p.keys, Transaction.id]`

- `always[a > 0]` or just `[a > 0]` or `[>0]` states domain on valid ranges. 

- `typically[a > 0, a <100]` or `[a >~ 0, a <~100]` or `[>~ 0, <~100]` define typical domain. "~" is a shorthand
of "typically" here. It is useful (for sharding, for example) to know the real range and whether the field 
is dominated by a single value: `typically[a = null]`

- `if[a > 0] then[x = 5] else[x = 3]` and `if[a > 0] then[y > ~5]` are conditional domain specification.

- `defined[field1, field...]`, `md5[...]` to inform that this field is functionally dependent on other fields 
 
- `substring[..4, = "test"]` to substring 


#### Semantics and formats

- `same[W]` and `similar[W]` states that this field has same/similar semantic meaning as W.

- `utctime["timesamp"/"iso"/formatstring]` specifies time format of the field (UTC)

- `zonetime[time: "timestamp"iso"/formatstring, zone: TIMEZONE]` specifies time format of the field
along with timezone. Useful for keeping client's time. `TIMEZONE` is an ISO string that represents a timezone.

- `unit[UNITNAME]` or `unit[UNITNAME / 1000]` defines property's unit. It is useful when some fields deal with one
unit (milliseconds, dollars, bytes) and some have other (seconds, cents, kilobytes). 

- `default[VALUE]` defines default value of the field. It is used when field is omitted during deserialization.

- `format[FORMAT]` to assure format inside this string. `a: str format[json]` can be rewritten canonically into
`a: str.json` or `a: json`.  

#### Time-related

- `lag[t]`, `lag[t1..t2]`, `since[t]`

- `first[t>...]`, `last[t<...]`

- `first since[]`, `last before[t]`

## Projects, clusters, nodes, applications
Project is a complete collection of repos, machines and services that can be interacted by an external user.
For instance, your online shop is a "project" 

Cluster is a set of machines, and node is a single machine (virtual or dedicated). 

Application is a piece of software that runs on a node.
There are subtypes of applications:
- services (`app.service`) which are expected to start accepting requests on start-up, and hang indefinitely
- pipelines (`app.pipeline`) which are launched by a trigger or schedule. They transform data, and halt.

```
Project: projectname
MyCluster: cluster(
    autoscale: ...
    nodenames: ...
    kubernetes_config_path: ..
    MyNode: node(
        hostname: ...
        region: ...
        bandwidth: ...
        ram: ...
        hdd: ...
        sdd: ...
        gpu: ...
        cpu: ...
        docker_image: ...
        dockerfile: ...
        MyApp: app(
            triggered: <schedule>
            >-- inName : mode[type] -->
            |>-- inName : mode[type] -->
            <-- outName: mode[type] --<
            |<-- outName: mode[type] --<
            |< apiCallName:   argmode[argtype] 
            |> apiCallName:   argmode[argtype]
            - - - > apiDefinition: argmode[argtype] < retval: argmode[argtype]
        )
    )
)
```

We do not have to define all these parts of hierarchy. "Default" project/cluster/node/app is 
assumed if not stated explicitly.

## Storages, components and pipelines

Apps are made of components and stateful storages.

Component is logic that processes data. It is defined by its input data streams, output data streams, exposed APIs,
and imported APIs from other components.

We do not care what's going on inside the component.
If we have to describe what's inside the component, we break it down into components and storages as well.

If there are no storages inside the component (even existing but invisible because we did not
state what's inside the component), then it must be stateless.

Storages hold data. One storage may hold multiple schemas.  

```
storageA: storage (
    sometable: schema@t (
        replication: ...
        size ~~ 1Mb
        records ~~ 1M
        retention: 30d
        retention: 1M records
        retention: 10Mb
        transaction_id: int [key]
        order_id: Purchase.id
        somedata: str
        Purchase.state@t' = "done"
    ) 
)
```

`storageA/sometable filter[transaction_id = 123]`

`storageA/sometable filter[somedata = "qwe"]`

### Data streams

- `stream[x]` or just `x` is an infinite sequence of objects of type X; and we are looking at
 objects one by one. `stream[X@t2]@t` means that now, at moment `t`, an object state was read from
the stream. This state was captured at moment `t2` (and `t2<t`)

- `batch[x]@t1..t2` or `batch[x]#N` is a finite sequence of objects, drawn from an infinite set.
We are looking at all objects in the batch simultaneously. We can specify time range of objects in the batch, 
or number of records.

- `one[x]` or `single[x]` means that there is a single x is passed, drawn from a finite set (for example,
config file).


## Access modifiers

`external` - defined here, but you do not control that

`public` - default; exposed to everyone who can read the doc

`private` - invisible to all who can read the doc.

## Template commands

`_as: QWE`; then `_.x` means `QWE.x` 

`_|` resolved
`_include: <localfile>`

`_import_FORMAT: git+<git url>/path^X`. X `branch` or `hash` or `release`. `FORMAT` is `lil`, `docker` ,
`docker-compose`, ...

`_code: path/sourcefile.ext#Lnum [arg1: sourcearg1] [arg2: sourcearg2] [connclass: sourceclassname]`

`_uml (...)` defines internal structure, class or activity diagrams in `plant uml` format.
