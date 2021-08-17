### Validation modifiers

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