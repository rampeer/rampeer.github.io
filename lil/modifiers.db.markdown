### Database modifiers

- `[partition]` informs that this field is a partition key. Rows with the same values will end up in the same node. 

- `[key]` flags fields that constitute table's primary key.

- `[unique]` says that values in that field is unique for each object

- `index[<index_name>: <range|equal|...>]` says that this field is a part of index `<index_name>`,
and in that index it can be effectively queried using "equals", "not equals" and "in" (`equal`) or using ">" and "<"
or "between" (`range`). Custom index types are supported; for example, to define database-specific index
(such as fulltext-search indices).
