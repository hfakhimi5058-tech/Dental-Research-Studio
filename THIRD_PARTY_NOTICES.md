# Third-party components

- `app/static/plotly.min.js`: Plotly.js v4.1.1, copyright 2012–2026 Plotly, Inc., MIT license. The MIT permission/warranty terms are retained in [licenses/Plotly-MIT.txt](licenses/Plotly-MIT.txt). Plotly.py's distributed notice is included there; the JavaScript bundle also retains its original copyright header.
- ToothSeg source is downloaded separately from the [official MIC-DKFZ repository](https://github.com/MIC-DKFZ/ToothSeg), pinned to commit `b29d1017fa124f89645fa56f98649e3f3f43bdb0`. The downloaded source retains its Apache-2.0 license. The local adapter uses its published ToothFairy2 pipeline and does not change network weights.
- ToothSeg model weights are downloaded separately from [Zenodo record 14893540](https://doi.org/10.5281/zenodo.14893540), under CC-BY-4.0. Creator attribution is in [docs/TOOTHSEG_ATTRIBUTION.txt](docs/TOOTHSEG_ATTRIBUTION.txt).
- Dataset registry/index files contain public publisher metadata, archive member names and source references, not medical images. Dataset access, licenses and permitted uses must be checked at the original source.
- Python dependencies are installed from their registries and retain their respective licenses. They are not vendored in this repository.
