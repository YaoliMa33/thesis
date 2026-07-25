# New Recorded Data Status

The local folder checked during packaging was `D:\enose_data`, not
`D:\enose_dataL`. The actual data subfolder was:

```text
D:\enose_data\enose_data
```

Compared with the current training set copied from `D:\thesis\enose_data`, this
newer folder contains extra CSV records:

- `acetone`: 10 extra CSV files.
- `air`: 10 extra CSV files.
- `alcohol`: 10 extra CSV files.
- `reference`: 9 CSV files.

The file-level list is stored in:

```text
docs/new_untrained_records_manifest.csv
```

## Training Status

These new records have not been used in any current training, validation, test,
figure, report, metric table, or conclusion in this repository.

Do not merge them into `data/training_enose_data/` silently. Before using them,
define a new explicit protocol, for example:

- append as a future chronological batch,
- use only after updating feature extraction,
- regenerate split definitions,
- rerun model selection without touching old test conclusions,
- clearly version the resulting tables and figures.

The current repository preserves the old trained result boundary so that the
published conclusions remain reproducible.

