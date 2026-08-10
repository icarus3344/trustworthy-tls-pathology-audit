# Data availability

No whole-slide image, patch, embedding, model weight, clinical record or
object-level prediction is distributed in this repository.

The study used open-access TCGA-STAD diagnostic slides from the NCI Genomic
Data Commons. The public manifest records slide identifiers and, where retained
in the original audit package, GDC file identifiers and MD5 checksums. Users
must download slides directly from GDC and follow its current data-use terms.

The five expansion slides were selected as a second engineering set. Their
public slide identifiers are included, but their local download records were
not part of the retained public package; GDC fields are therefore marked
"not archived" rather than reconstructed from memory.

The manifest is for reproducibility and is not a pathology ground-truth table.

## Local object inventory for the YOLO audit

`tls_audit.yolo_robustness` accepts a local JSON inventory instead of assuming
an institutional directory layout. Keep that inventory outside version
control: it contains WSI locations and level-0 object coordinates. The payload
is either a list or an object with an `objects` list. Each object requires:

- `cohort`, `sample_id` and `object_id`;
- `wsi` (absolute or relative to the inventory file);
- `polygon_coordinates`, `level0_bbox`, `mpp_x` and `mpp_y`;
- two `existing_records`, one each for 10% and 25% padding, with
  `padding_ratio` and `sha256` fields.

The run manifest records only the inventory filename and SHA-256, never its
absolute path or its contents.
