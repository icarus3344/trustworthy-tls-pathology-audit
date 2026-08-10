# Third-party notices

This repository contains original adapter and audit code. It does not
redistribute third-party source repositories, model weights, embeddings,
whole-slide images or pathology annotations.

## Pipeline components

| Component | Role in the research workflow | Upstream terms |
|---|---|---|
| [HookNet-TLS](https://github.com/DIAGNijmegen/pathology-hooknet-tls) | Candidate TLS and germinal-centre segmentation | MIT |
| [Pan-Cancer Spatial Atlas of TLS](https://github.com/Coolgenome/Pan-Cancer_Spatial_Atlas_TLS) | Released maturity cascade and study workflow | GNU AGPL v3 |
| [PathPrism](https://github.com/KatherLab/PathPrism) | Nine-class tissue semantics and spatial features | CC BY-NC 4.0 |
| [STAMP](https://github.com/KatherLab/STAMP) | WSI tiling and feature extraction | MIT |
| [UNI](https://huggingface.co/MahmoodLab/UNI) | Frozen pathology image encoder used by PathPrism | gated CC BY-NC-ND 4.0 |
| [Ultralytics](https://github.com/ultralytics/ultralytics) | Runtime used to evaluate released YOLO classifiers | GNU AGPL v3 or commercial license |
| [NCI Genomic Data Commons](https://portal.gdc.cancer.gov/) | Source of public TCGA-STAD diagnostic slides | GDC data-use policies |

The optional PrismNet adapter reads a checkpoint supplied by the user and
implements matrix operations needed for inference. No PathPrism checkpoint or
UNI-derived embedding is included. Users must obtain every external asset
directly from its official source and comply with the corresponding terms.

## Figures

Figures in this repository were generated from the author's aggregate audit
tables. They are not copied from the cited papers. No individual H&E field is
included.

## Attribution

Use of a component does not imply endorsement by its original authors. Please
cite the corresponding papers and repositories when reproducing any part of
the workflow.
