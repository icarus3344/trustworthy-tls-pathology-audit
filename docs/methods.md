# Methods

## 1. Candidate geometry

HookNet-TLS annotations were parsed as level-zero WSI polygons. Every polygon
was checked against slide dimensions before cropping. The default working view
was the polygon bounding box plus 10% context; 0%, 5%, 15%, 25% and 40% were
added for perturbation analysis.

Coordinates were converted into micrometres using the slide MPP rather than
matched by file order or raster index.

## 2. Maturity cascade

Step 1 emits Immature and Mature evidence used by the working cascade. Step 2
separates Primary and Secondary for objects entering the mature branch.

For continuous perturbation curves, the audit defines an explicitly
engineering-derived score:

    p(E) = p1(Immature)
    p(P) = p1(Mature) × p2(Primary)
    p(S) = p1(Mature) × p2(Secondary)

The relevant branch values are normalized first. These scores sum to one but
are not calibrated pathology probabilities and are not presented as an
upstream author-confirmed fusion rule.

Uncertainty is summarized with normalized Shannon entropy and the top-1
probability margin.

## 3. Robustness audit

### Context

Each object is re-evaluated over six polygon-padding fractions. The audit
records class transitions, probability range, maximum entropy and minimum
margin.

### Colour and encoding

The 10% crop is evaluated after lossless PNG, JPEG quality changes, mild hue,
saturation and brightness changes, Reinhard normalization and Macenko
normalization. Normalization is treated separately from mild technical
perturbation because it can impose a large distribution shift.

### Evidence localization

Grad-CAM is computed for both cascade stages. Saliency enrichment is:

    saliency mass inside polygon / polygon area fraction

An enrichment above one indicates more heat inside the polygon than expected
from area alone. Occlusion repeats inference after local image intervention and
provides a non-gradient comparison.

TLS-only and context-only blurred images compare dependence on polygon
morphology with dependence on the surrounding crop. The manipulated images may
be out of distribution; their role is diagnostic rather than causal.

## 4. PathPrism adapter

UNI feature vectors are projected using the released PrismNet PCA parameters,
then passed through the released linear probe for eight biological tissue
classes plus BACK:

    z = (x - pca_mean) × pca_projection
    logits = z × weight_transpose + bias
    probabilities = softmax(logits)

The classes are ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR and TUM. No UNI
embedding or checkpoint is redistributed.

## 5. Spatial features and controls

Candidate centres and PathPrism tile centres are represented in physical
coordinates. Primary analysis uses the 0–250 micrometre inner region at a
112-micrometre semantic tile scale; 256 micrometres is a sensitivity analysis.

Features include:

- hard-label and continuous tissue fractions;
- radial LYM, STR and TUM composition;
- connected-component counts;
- nearest-tissue distances;
- direct four-neighbour interface length;
- centroid-proximity features reported separately from physical contact.

Three deterministic same-slide controls are selected per candidate outside a
TLS exclusion zone while approximately matching broad tissue composition.

## 6. Statistical unit

Objects are nested inside slides. Object-level results are descriptive; slide
means are the primary independent summaries. Bootstrap intervals resample
slides rather than pretending that 84 candidates are 84 independent patients.
