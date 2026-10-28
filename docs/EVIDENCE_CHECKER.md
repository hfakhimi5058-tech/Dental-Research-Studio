# Evidence & Uncertainty Checker

Open **Reports → Generate draft & check**. The app generates one volume statement per segmented label. Each statement records its voxel count, physical voxel volume, calculation method, label and an actual labelled voxel. This is a measurement draft, not a diagnostic report.

Use **Add a draft finding** for a claim from another report. Enter one claim, choose a segmented structure or name an anatomical region, and optionally capture the workspace crosshair. A structure supplies a labelled voxel if no location was supplied. An unlocated or mismatched claim stays flagged. A label's absence does not establish that a tooth is anatomically absent.

Select a finding to view three source-grid slices. These can be oblique. Grayscale shows source intensity; the window comes from the workspace and is not assumed to be calibrated HU. **Locate in source images** opens the linked workspace for zooming, window changes and wider inspection. The slices establish an evidence location, not the truth of the claim.

For each finding, record:

1. Whether the finding is visible on the source image.
2. Whether it is tied to a segmented structure or named image region (checked automatically against the current grid and label).
3. Whether image quality and model/segmentation output are adequate for this particular claim.
4. Whether to request specialist review, record a research review, or reject the finding.

Provide a reviewer name/identifier and evidence notes. Assessments are self-recorded and are not authenticated professional sign-off. An acceptance choice cannot clear missing-source, synthetic-image, stale-location, calibration-assumption, boundary-contact or disconnected-component flags. Component checks appear when quantitative analysis is available. Reference Dice is displayed if available; it is overlap, not a probability that a clinical claim is true. No calibrated model confidence is available.

## Persistence and invalidation

Changes are saved locally in `data/cases/<case-id>/evidence.json`, with an append-only event history inside that file. Removed findings and superseded measurement drafts remain in the history. This is an editable local audit record, not a tamper-proof audit system. Use **Save case & corrections** to preserve the source case and any edited mask too.

Image/mask revisions, case reopenings and changed analysis/reference evidence invalidate the relevant previous assessments. A stale draft claim must be replaced with a finding grounded in the current image. Regenerating measurements resets their reviews and preserves manually added claims with their original revisions. A version check rejects updates from an outdated browser view. Old source-slice requests are rejected if the active case or revision changed.

**Evidence + audit JSON**, the full JSON report and printable HTML report retain unresolved flags and rejected findings. A report exported before running the checker explicitly says it has not been run. Local evidence and scans are excluded from Git.

## Relationship to the supplied paper

The supplied paper, *From Voxels to Views to Reports: A Segmentation-Guided VLM Pipeline for CBCT Report Generation* (Singh, Lu and Sagl; `ODIN_challenges_024.pdf`), describes segmentation-guided views, schema-constrained VQA, a fine-tuned VLM and post-processing/report assembly in sections 2.1–2.3.

This implementation adds an explicit, conservative review stage around our existing quantitative reports and manually entered draft claims. It does **not** reproduce that paper's fine-tuned VLM, panoramic reconstructions, multi-anatomy segmentation, clinical phrasing corpus, validation-based source selection, or low-precision suppression rules. No paper performance results or thresholds are transferred to this app. Missing or uncertain evidence is retained for review rather than omitted from exports.

The checker has no network or model dependency. It does not automatically decide that pathology is visible, evaluate all image artifacts, parse arbitrary narrative claims, verify diagnostic correctness or confer clinical clearance.
