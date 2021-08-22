{% include menu.markdown %}

## Quick reference

### Syntax

`a: b` property `a` of type `b`

`a: b someMod[modType: value]` property `a` of type `b`, and also we know that `a` has setting `someMod`, and
that setting's`modType` is set to `value`. Modifiers can:

- specify filtering parameters and variable domains

  `a:int always[a> 5]` or just `a:int [a>5]` defines possible value range

  `a:int typically[a<10]` or `[a ~< 10]` define reasonable value range

- add semantic info, specify data format or middlewares.  `comp1.inName -- format[json] ---> comp2.outName`

`hostname~dev = dev.mytool.com` tilde (`~`) is used to define deploy-specific (or machine/user-specific) values.

`hostname ~username:user ~nodename:node` 

`MyComponent^V1` MyComponent of version 1.

`MyComponent!varX` MyComponent variant (design draft) `varX`:

```
MyComponent!varX :component(
    -completely_removed_stuff: str
    
    -old_field: int
    +new_replacing_field: int
    
    kept_stuff: float
    
    +completely_new: str
)
```

`_source: src/app/server.py#L30..50` defines binding to source files

`_import: URI/`

- `[public]` share with everyone
- `[private]` do not share outside of dev group 
- `[external]` described here, but we do not control that

### Objects

defines business data structure
```
MyDataSchema: schema [size ~ 30 bytes] [version=V1] (
    a: int [id]
    b: float [b>0]
    dt: str
    d: enum[MyEnum: value1, value2, value3]
)
```

Cluster is a set of nodes (think of autoscaling group).

Node is a machine, virtual or dedicated.

Storages store data. Stored schema is usually different than business logic's schema.

Apps are running services or scheduled periodically firing pipelines. Apps are made of components, which define pieces of logic.
Usually we do not care what happens inside them 

Layers can be skipped.

- `-->` is "read from"
- `<` is "write to"
- `|` is "triggered from outside"; `| something |` states that `something` happens outside of the app / component.
- `|> apiName: intype > outtype >|` - api definition
-
`stream` is a sequence of objects which we dispatch one by one

`batch` is a  

```
MyApp: app (
    comp1: component (
        |> config: sigle[config]
        > inName: stream[datatype]
        < outName: batch[datatype]
        |> apiName: stream[indatatype] > stream[outdatatype] >|
        <| callName: stream[outDataType] < stream[inDataType] |<
    )
)
```

Storages store data. Not all data inside belongs to business schema; some fields are technical.

```
MyDB: storage (
    mytable: schema (
        id: MyDataSchema.id
        
    )
)
```
