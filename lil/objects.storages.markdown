### Storages, components and pipelines

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
