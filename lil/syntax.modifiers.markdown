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

### Conditions

For each property, we may use modifiers to specify some conditions.

- `> < =`, to specify valid ranges
- `>~ ~< =~` to specify typical ranges (reasonable bounds)
- `~~` to define order of magnitude or asymptotic line 
- `=` or `=~` to define expected value or typical value
- `start..` , `..end`, `start..end` to define ranges, for example `0..100`
- `%` to specify divisibility. "a is ranges from 0 to 1000 with step 10" is written as `a: int [0..1000, a % 10]`
- `len`, `start.. end` when applied to strings, specify string length or substrings (which can then be used in conditions) 
