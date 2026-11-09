# Pulpy3D: pulp and nerve annotation review

Pulpy3D support opens **existing dataset annotations** alongside their CBCT scan. You can rotate the surfaces, isolate an individual labelled pulp cavity, inspect source slices, calculate volume and physical bounds, and export measurements. It does not generate pulp predictions from a new unlabelled scan.

Obtain the dataset from its [official portal](https://ditto.ing.unimore.it/pulpy3d/) and check the release's access terms and annotation provenance. Keep extracted files in `data/Pulpy3D/` inside your installation. Data, model weights and the publisher's code are not included. Do not assume that every distributed annotation was manually verified.

## Open a case

1. Select **Import images → Pulpy3D** (also available from **Datasets**).
2. Enter the extracted dataset folder or a single patient folder, then select **Inspect folder**.
3. Select a patient and annotation variant. The author's [documented layout](https://github.com/mahmoudgamal0/Pulpy3D#dataset-structure) pairs `data.nii.gz` with one or more of `gt_instance_ian.nii.gz`, `gt_instance.nii.gz`, `gt_pulp_mandible.nii.gz`, `gt_pulp.nii.gz` or `gt_ian.nii.gz` in each patient folder. `.nii` is also accepted. A paired single-channel `imagesTr` / `labelsTr` layout is supported through matching label metadata.
4. Review the encoding before confirming it. Original files and compact training masks use different IDs; selecting a different encoding changes the names, not the source voxels. For a different release, select **Matching label-definitions JSON** and provide that release's exact definitions.
5. Select **Import & analyse**. Unknown IDs, empty masks, invalid spatial units and scan/mask grid mismatches produce an error instead of an automatic relabelling or resampling. Only confirm missing spatial units as millimetres when the source calibration establishes that.
6. Use the structure selector, **Show only this structure**, **Segment sizes**, **Analysis** and **Reports**. Saved cases retain the chosen labels and annotation provenance; original inputs are not overwritten.

The importer loads one annotation variant at a time. Reimport another variant to inspect it as a separate case. Binary masks contain a single foreground class, so they cannot distinguish individual teeth. Pulp cavity volume is not whole-tooth volume; R/A/S bounds are not anatomical canal lengths.

## Label encodings

The [author's dataset metadata](https://github.com/mahmoudgamal0/Pulpy3D/blob/main/datasets/Pulpy3D/dataset.json) and [training transformations](https://github.com/mahmoudgamal0/Pulpy3D/blob/main/dataloader/Transformation.py) define the original-to-training mapping below. These are quoted source IDs, not measurements.

| Encoding | Pulp cavity IDs | Other IDs |
| --- | --- | --- |
| Original FDI | 31–38 and 41–48 | 0 background; 50 source anomaly; 1 nerve in the combined annotation |
| Compact training | 2–9 correspond to FDI 31–38; 10–17 to FDI 41–48 | 0 background; 18 source anomaly; 1 nerve in the combined annotation |
| Binary | 1 foreground, with its meaning determined by the selected pulp or nerve variant | 0 background |
| Custom JSON | Exact definitions supplied for the release | Every observed ID must be defined |

For custom JSON, use either `{"labels":{"background":0,"Pulp cavity FDI 31":31}}` or an ID-to-name object such as `{"0":"Background","31":"Pulp cavity FDI 31"}`. Supply all IDs present in your mask; the example is only a format illustration.

## Prediction and training status

The bundled ToothSeg integration predicts permanent teeth, not separate pulp or nerve structures. The Pulpy3D import feature requires no GPU or prediction model. The authors' [main repository](https://github.com/mahmoudgamal0/Pulpy3D/tree/main/checkpoints) and [seed repository](https://github.com/mahmoudgamal0/Pulpy3D-Seed/tree/main/checkpoints) had no distributed checkpoint files when reviewed on 2026-11-09. A compatible pretrained pulp model has not been integrated or evaluated here.

This update does not train a model or combine datasets. Pulpy3D extends ToothFairy, so overlapping patients must not cross training and test splits. Review source permissions, target labels, annotation provenance and independent evaluation before undertaking training. Clinical performance has not been established.
