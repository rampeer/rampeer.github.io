### Schema

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