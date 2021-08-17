### Semantics and formats

- `same[W]` and `similar[W]` states that this field has same/similar semantic meaning as W.

- `utctime["timesamp"/"iso"/formatstring]` specifies time format of the field (UTC)

- `zonetime[time: "timestamp"iso"/formatstring, zone: TIMEZONE]` specifies time format of the field
along with timezone. Useful for keeping client's time. `TIMEZONE` is an ISO string that represents a timezone.

- `unit[UNITNAME]` or `unit[UNITNAME / 1000]` defines property's unit. It is useful when some fields deal with one
unit (milliseconds, dollars, bytes) and some have other (seconds, cents, kilobytes). 

- `default[VALUE]` defines default value of the field. It is used when field is omitted during deserialization.

- `format[FORMAT]` to assure format inside this string. `a: str format[json]` can be rewritten canonically into
`a: str.json` or `a: json`.  

### Time-related

- `lag[t]`, `lag[t1..t2]`, `since[t]`

- `first[t>...]`, `last[t<...]`

- `first since[]`, `last before[t]`

## Data streams

- `stream[x]` or just `x` is an infinite sequence of objects of type X; and we are looking at
 objects one by one. `stream[X@t2]@t` means that now, at moment `t`, an object state was read from
the stream. This state was captured at moment `t2` (and `t2<t`)

- `batch[x]@t1..t2` or `batch[x]#N` is a finite sequence of objects, drawn from an infinite set.
We are looking at all objects in the batch simultaneously. We can specify time range of objects in the batch, 
or number of records.

- `one[x]` or `single[x]` means that there is a single x is passed, drawn from a finite set (for example,
config file, or credentials of available nodes).

## Access modifiers

`[external]` - defined here, but you do not control that

`[public]` - default; exposed to everyone who can read the doc

`[private]` - invisible to all who can read the doc.