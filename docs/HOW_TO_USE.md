# How to use Dental Research Studio

For installation and model setup, start with the [README](../README.md). Open the app with `Start.cmd`, then visit `http://127.0.0.1:8517/`. The **?** button in the top right opens the built-in guide.

## From a scan to tooth segments

1. Select **Import images**. For an unsegmented scan, put the `.nii`, `.nii.gz` or `.mha` file in **Raw scan** and leave **Existing segmentation mask** empty. For large files, choose **Local paths** and use **Scan path**. Conventional multi-file DICOM is available through **DICOM folder → Find image series**.
2. If you already have a labelled mask, use **Existing segmentation mask** instead. A matching scan and a label-definitions JSON file can accompany it. A raw intensity image is not a segmentation mask, even when both use `.nii.gz`.
3. For a raw scan, open **Models & jobs**. Choose **Use** beside the installed ToothSeg model, keep `checkpoint_final.pth`, review its source and labels, then select **Run local segmentation**. If the model is missing, follow **Enable automated segmentation** in the README.
4. Follow the stage and activity panel under the run button or above the workspace images. A block percentage measures the **current prediction pass**, not the entire pipeline. Other stages show their name and activity without inventing a percentage. **Completed** confirms the whole job finished. Errors and the **Live stage log** are in **Models & jobs**.
5. Return to **Workspace**. A separate progress counter shows how many selected surfaces have been prepared for the 3D view.

**A raw scan does not need labels beforehand. Segmentation creates the labels.** An empty 3D surface view before segmentation is expected; the source slices can still be reviewed.

The sidebar search **filters existing structure names and IDs**; typing a command there does not run it. Use **Clear filter** if your search hides all structures. **Models & jobs** gives setup instructions when weights or the runtime are missing.

## Datasets and pulp cavities

**Datasets** separates current-machine capabilities from the research source catalogue. Listed anatomy describes the source dataset, not what an installed model predicts. ToothSeg predicts permanent teeth; its output has no separate pulp, nerve, jaw or sinus labels.

For Pulpy3D, use **Import images → Pulpy3D → Inspect folder**, choose a patient and annotation variant, then review and confirm its label encoding. The [Pulpy3D guide](PULPY3D.md) explains original FDI IDs, compact training IDs, binary masks and custom metadata. Importing preserves the original mask and opens its structures for 3D review, measurements and export. Pulp segments are not whole-tooth segments. This feature does not run automatic pulp prediction or train on the dataset.

## View and measure individual teeth

- Use the **Structure** selector above the images to show a label separately. **Show only this structure** isolates the current selection; the arrow buttons move to the previous or next available label. **Show all structures** restores the complete prediction.
- Drag inside the 3D view to rotate, or use its directional buttons. Use the panel's expand button for a larger view. Scroll a source slice to move through the scan.
- The measurement strip shows the selected segment's **volume in mm³**, **labelled voxel count** and **bounds in mm**.
- Open **Segment sizes** for visual cards for every label. Search by tooth name or label, sort by label or volume, and use **View this structure separately** to inspect a card's segment. **Export sizes CSV** downloads the measurements.

Volume is labelled voxel count multiplied by voxel volume from the image's spatial transform. Bounds are dimensions along R (right), A (anterior) and S (superior); they are **not anatomical tooth lengths**. Names and IDs identify labels when display colours repeat.

## Review questionable results and save work

A predicted label can contain disconnected regions, including a stray fragment far from the tooth. Isolating a label deliberately retains all of its regions. A fragment affects both volume and bounds; a plausible name or shape does not establish that it is a real tooth.

Check the source slices and use **Analysis → Run quantitative analysis** for component and boundary checks. Size cards flag scan-edge contact and show disconnected-component warnings when that analysis is available. These checks are review prompts, not diagnoses. Do not remove a region merely to improve appearance.

Use **Correct** for deliberate mask edits and **Undo** to reverse a correction. Select **Save case & corrections** to retain edits; reopen cases from **Cases**. Original input files are preserved. **Reports** provides printable reports, NIfTI masks, selected-surface OBJ and JSON/CSV exports. The [Evidence & Uncertainty Checker](EVIDENCE_CHECKER.md) records source links and reviewer judgments.

This is a research prototype. Model predictions, image calibration and measurements need review; the app is not clinically validated. Processing is local, but a project placed in a synced folder may be synchronized by that service.
