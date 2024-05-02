# social-graph-analysis-2024

Codebase for tests and analysis of: 

Gajderowicz, B., Fisher, A., Mago, V.: (preperation) "Graph pruning for identifying COVID-19 misinformation dissemination patterns and indicators on Twitter/X"

# Downlaod data
1. Go to root folder.
1. Download data as a [zip files](https://zenodo.org/doi/10.5281/zenodo.11100128) into root folder. Should contain the following: data_root.zip, data-part.z01, data-part.z02, data-part.z03, data-part.z04, data-part.z05, data-part.zip.
1. `unzip 'data-part.*'`
1. `unzip data_root.zip -d data`

# Install
1. `conda env create -f environment.yml`
1. `conda activate pyTwitter`

# Run
1. `python -m src.clustering quebec`
1. `python -m src.clustering british_columbia`
1. `python -m src.clustering ontario`
1. `python -m src.analysis COVID`
1. `python -m src.analysis Healthcare`
1. `python -m src.stats`
