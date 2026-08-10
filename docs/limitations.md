# Limitations and interpretation boundary

## No pathology ground truth

The 84 regions are postprocessed model candidates, not expert-confirmed TLS.
There is no complete object-level reference for candidate validity, follicular
organization or germinal-centre status. Accuracy, sensitivity, specificity and
calibration cannot be estimated.

## Shared H&E source

HookNet, YOLO and PathPrism all derive information from the same H&E image.
Agreement can arise from shared morphology or shared artefacts. The PathPrism
result is therefore cross-model spatial concordance, not independent
biological validation.

## Small and nested sample

Ten public slides were screened and eight contributed candidates. Objects from
one slide are correlated. The slide, not the object, is the relevant
independent unit; eight detected slides cannot support population or outcome
claims.

## Domain and scale shift

The released models were not calibrated here for gastric cancer. PathPrism
outputs differed between 112- and 256-micrometre tiles, and the exact original
WSI preprocessing parameters were not fully specified in the upstream
materials available during the audit.

## Working maturity rule

The continuous E/P/S score is an engineering derivation for perturbation
curves. It is not an author-confirmed calibrated probability. Hard labels also
remain unvalidated against pathology.

## Interpretability is not proof

Grad-CAM has limited spatial resolution. Occlusion and blurred
counterfactuals introduce artificial inputs. Convergent attribution evidence
can increase confidence that a model used a region, but it cannot prove a
correct diagnosis or biological mechanism.

## Reproducibility constraints

Third-party weights, UNI embeddings and WSI data cannot be bundled. End-to-end
reproduction requires independent access to upstream assets and a compatible
OpenSlide/PyTorch environment. This repository reproduces code paths,
synthetic checks and public aggregate summaries, not the entire compute
environment.

## Manual review

Engineering visual review was used to flag tissue edges, artefacts, confluent
aggregates and uncertain morphology. It is not a substitute for a blinded
pathologist reference.

## Appropriate conclusion

The workflow is useful for surfacing unstable or implausibly supported model
outputs and for designing a pathology review set. It is not ready for clinical
use or a biological claim about TLS maturity.
