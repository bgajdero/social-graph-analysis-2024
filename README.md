# social-graph-analysis-2024

Codebase for tests and analysis of: 

Gajderowicz, B., Fisher, A., Mago, V.: (preperation) "Graph pruning for identifying COVID-19 misinformation dissemination patterns and indicators on Twitter/X"

# Install
1. `conda env create -f environment.yml`
1. `conda activate pyTwitter`
1. `Downlaod the data az a zip file from 
1. `unzip data.zip > data`

# Run
1. `python -m src.clustering quebec`
1. `python -m src.clustering british_columbia`
1. `python -m src.clustering ontario`
1. `python -m src.analysis COVID`
1. `python -m src.analysis Healthcare`
1. `python -m src.stats`