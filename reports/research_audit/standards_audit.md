# Standards & Industrial Specifications Audit

## 1. Repository Standards Search Results

A comprehensive regex search across all source code, docstrings, scripts, configurations, and documentation for terms `ASTM`, `ISO`, `EN`, `standard`, `grade`, and `rating` was conducted:

| Standard / Keyword | Repository Occurrences | Context Found | Scientific Classification |
| :--- | :---: | :--- | :--- |
| **ASTM Standards** (e.g. ASTM E45, ASTM A6) | 0 occurrences | No explicit standard cited or fabricated in code. | **Directly Supported** (*No false claims made*) |
| **ISO Standards** (e.g. ISO 4967) | 0 occurrences | No explicit standard cited or fabricated in code. | **Directly Supported** (*No false claims made*) |
| **EN Standards** (e.g. EN 10163) | 0 occurrences | No explicit standard cited or fabricated in code. | **Directly Supported** (*No false claims made*) |
| **Defect Class Standard** | Mentioned in README | References Northeastern University (NEU) defect database terminology. | **Directly Supported** (*Academic benchmark*) |

---

## 2. Examination of Relevant Industrial Surface Inspection Standards

To ensure complete clarity in the undergraduate research paper/report, we document how actual international steel standards approach surface defects:

### A. EN 10163 (Delivery conditions for surface finish of hot-rolled steel plates, wide flats and sections)
- **Scope**: Defines acceptance criteria for surface defects (cracks, shell, seams, laps, rolled-in scale, inclusions).
- **Measurement Principles**: Divides defects into discontinuous vs continuous, defining permissible repair depths (grinding or welding) based on nominal product thickness $t$ (e.g. max depth $\le 0.2t$ or remaining thickness $\ge t_{\min}$).
- **Formula Availability**: **Does NOT provide a 2D single-image pixel severity equation.** It requires absolute caliper/depth gauge measurements.

### B. ASTM E45 (Standard Test Methods for Determining the Inclusion Content of Steel)
- **Scope**: Microscopic inclusion rating using reference comparison charts (JK chart method: Types A, B, C, D rating severity from 0.5 to 3.0 based on inclusion length and thickness).
- **Measurement Principles**: Requires cross-sectional metallographic preparation at $100\times$ magnification.
- **Applicability to NEU-DET**: **Not directly applicable.** NEU-DET contains macroscopic/low-magnification 2D strip surface images, not metallographic microsections.

### C. ASTM A6 / A6M (Standard Specification for General Requirements for Rolled Structural Steel Bars, Plates, Shapes, and Sheet Piling)
- **Scope**: Defines surface imperfection limits based on area percentage and depth relative to thickness.

---

## 3. Standards Audit Conclusion

1. **No Standard Was Falsified**: The codebase correctly avoids asserting that an ASTM or ISO standard defines our formula.
2. **Scientific Finding**: There is **no universal international standard that defines a closed-form numerical severity score from 2D macroscopic planar grayscale surface photographs** without thickness/depth calibration.
3. **Requirement for Paper / Thesis**: The research report must explicitly state that the formulated severity index is an *algorithmic proxy* designed for research exploration and comparison, not a certified metallurgical standard.
