# Project brief

## Title

**Reliability and uncertainty auditing of a multi-model TLS pathology
workflow**

## Motivation

Tertiary lymphoid structures are spatially organized immune aggregates whose
morphology and maturity may be clinically informative. Published pipelines can
detect candidate regions and assign maturity labels, but an apparently
confident label does not show that the model is stable, attends to the intended
region or generalizes across routine preprocessing choices.

This project asked a reliability-first question:

> When should a model-generated TLS candidate and maturity label be trusted
> enough to enter biological analysis or expert review?

## Technical work

During a six-week summer project, I reconstructed and connected:

- HookNet-TLS candidate segmentation;
- a released two-stage YOLO maturity cascade;
- PathPrism tissue semantics generated from STAMP/UNI features and a PrismNet
  linear probe.

I aligned polygons and semantic tiles in physical coordinates and designed a
stress-test suite covering crop context, encoding, colour, stain
normalization, Grad-CAM, occlusion and counterfactual image interventions. I
also implemented radial tissue-context features, deterministic same-slide
matched controls, slide-clustered summaries and a manual-review priority score.

## Pilot evidence

Ten public TCGA-STAD slides were screened. Eight contained 84 postprocessed
model candidates. Only 41 of 84 hard-cascade outputs remained stable over
0–40% context padding; 10 of 84 changed under mild colour or encoding
perturbations, and Macenko normalization changed 48 of 84. Step-1 Grad-CAM was
enriched inside the HookNet polygon in 75 of 84 objects, whereas Step 2 met the
same criterion in only 20 of 84.

PathPrism showed a positive inner LYM difference relative to same-slide
matched controls in all eight detected slides. Because all models observed the
same H&E image, this supports cross-model spatial consistency rather than
independent biological validation.

## Research outcome

The project did not produce a validated TLS classifier. It produced:

1. a traceable multi-model engineering workflow;
2. quantitative evidence of context and preprocessing fragility;
3. an uncertainty-aware review queue;
4. explicit stopping rules for unsupported biological claims.

The main lesson was that trustworthy computational pathology requires
patient/slide-aware statistics, independent validation, provenance and the
ability to preserve negative results.

## Next step

A blinded pathologist review should first confirm candidate validity and
follicular/germinal-centre morphology. The calibrated workflow can then be
tested on an external GI cohort and compared across pathology foundation
models, with calibration, abstention and high-confidence failure analysis as
primary endpoints.
