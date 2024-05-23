# chilean-humor

Describe your project here.

```shell
.venv\Scripts\activate
sqlite-utils insert humor.db routines data/FICVN-routines.csv --csv --detect-types
sqlite-utils insert humor.db comedians data/FICVN-comedians.csv --csv --detect-types
sqlite-utils transform humor.db routines --pk ROUTINEID
sqlite-utils transform humor.db comedians --pk COMEDIANID
sqlite-utils extract humor.db routines SHOW --table shows
sqlite-utils add-foreign-key humor.db shows SHOW comedians SHOW
```