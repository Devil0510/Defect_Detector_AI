# Severity Index Formulation Audit

## 1. Technical Inspection of `src/severity/index_calculator.py`

The index calculator implements the equation:
$$S = 100 \times \left( w_A \cdot \frac{\min(A, A_{\max})}{A_{\max}} + w_L \cdot \frac{\min(L, L_{\max})}{L_{\max}} + w_C \cdot C + w_T \cdot \frac{\min(T, T_{\max})}{T_{\max}} \right)$$

Every parameter in this equation was audited for empirical or standardized justification.

---

## 2. Parameter Audit Table

| Parameter | Symbol | Current Value | Source | Empirical / Literature Evidence | Status & Scientific Classification |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Area Weight** | $w_A$ | `0.40` | Internal Heuristic | None. Intuition that defect surface area is the primary visual damage factor. | **Unjustified** (*Research assumption*) |
| **Length Weight** | $w_L$ | `0.25` | Internal Heuristic | None. Intuition that elongated defects (e.g. scratches, cracks) propagate stress. | **Unjustified** (*Research assumption*) |
| **Contrast Weight** | $w_C$ | `0.20` | Internal Heuristic | None. Inferred that deeper surface discoloration indicates greater disruption. | **Unjustified** (*Research assumption*) |
| **Texture Weight** | $w_T$ | `0.15` | Internal Heuristic | None. Inferred that micro-roughness reflects surface degradation. | **Unjustified** (*Research assumption*) |
| **Area Normalization Cap** | $A_{\max}$ | `4000.0 px²` | Internal Heuristic | Fixed at 10% of total image area ($200 \times 200 = 40,000\text{ px}^2$). Not calibrated to critical flaw size. | **Unjustified** (*Research assumption*) |
| **Length Normalization Cap**| $L_{\max}$ | `200.0 px` | Image Geometry | Maximum pixel span of image dimension ($W=200$). Geometric upper bound, not physical length. | **Partially Justified** (*Image-relative constraint*) |
| **Texture Normalization Cap**| $T_{\max}$ | `50.0` | Internal Heuristic | Arbitrary divisor based on inspection of raw GLCM contrast ranges in dataset. | **Unjustified** (*Research assumption*) |
| **Low / Moderate Threshold**| $T_{\text{low}}$ | `25.0` | Arbitrary Quartile | Linear quartile division $[0, 25)$. No metallurgical significance. | **Unjustified** (*Research assumption*) |
| **Moderate / High Threshold**| $T_{\text{mod}}$ | `50.0` | Arbitrary Quartile | Linear quartile division $[25, 50)$. No metallurgical significance. | **Unjustified** (*Research assumption*) |
| **High / Critical Threshold**| $T_{\text{high}}$ | `75.0` | Arbitrary Quartile | Linear quartile division $[50, 75)$. Resulted in severe class imbalance ($<1\%$ Critical). | **Unjustified** (*Research assumption*) |

---

## 3. Methodological Critique

1. **Absence of Domain Failure Models**: In physical metallurgy and fracture mechanics (e.g., linear elastic fracture mechanics), defect criticality depends non-linearly on stress concentration factors ($K_t$), crack tip radius, orientation relative to rolling/tensile stress direction, and material thickness. The linear weighting model $S = \sum w_i x_i$ is a proxy for visual surface disruption, not mechanical criticality.
2. **Arbitrary Bounding of Critical Class**: The threshold $S \ge 75$ creates an extreme minority class where only **19 out of 4,126 defects (0.46%)** qualify as Critical across the entire dataset. In the test set, only 5 Critical defects exist, on which the ML models achieved only 20% recall (1 out of 5 detected).
3. **Equal Weighting Across Defect Typologies**: The same weight vector $(0.40, 0.25, 0.20, 0.15)$ is applied uniformly to all 6 defect types. In reality, a scratch of length $L$ has a radically different stress concentration profile than an oxide inclusion or rolled-in scale of identical pixel area.

---

## 4. Required Research Refinement
- If pseudo-labels are utilized, thresholds must either be calibrated via quantile stratification (e.g., 25th, 50th, 75th percentiles of empirical distributions) or explicitly framed as a **synthetic benchmark proxy**.
- The index must be clearly documented in all reports as an **Exploratory Geometric Severity Index (EGSI)** rather than an established industrial rating.
