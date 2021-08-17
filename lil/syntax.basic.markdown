### Alphabet


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