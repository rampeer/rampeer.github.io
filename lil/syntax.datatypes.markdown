### Basic data types

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

### Basic units

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
