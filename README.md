# StreetLens

**StreetLens: Enabling Human-Centered AI Agents for Neighborhood Assessment from Street View Imagery**  

This is the official repository of **StreetLens**.  

StreetLens is a human-centered, researcher-configurable workflow for scalable neighborhood environmental assessments. It enables VLMs to mimic trained human coders by:

- **Domain-informed analysis:** Focuses on questions derived from established protocols.  
- **Scalable image retrieval and annotation:** Automatically retrieves relevant street view imagery (SVI) and generates semantic annotations, ranging from objective features (e.g., number of cars) to subjective perceptions (e.g., sense of disorder).  
- **Flexible integration of expertise and data:** Allows researchers to define the VLM’s role through domain-informed prompting and incorporate prior survey data for robust assessments across diverse contexts.

## Links

- **Demo:** [https://knowledge-computing.github.io/projects/streetlens](https://knowledge-computing.github.io/projects/streetlens)
- **ACM SIGSPATIAL GeoHCC'25 Workshop Paper:** [https://dl.acm.org/doi/10.1145/3764917.3771334](https://dl.acm.org/doi/10.1145/3764917.3771334)
- **arXiv Paper:** [https://arxiv.org/abs/2506.14670](https://arxiv.org/abs/2506.14670)  

## System Architecture and Examples

Below is an overview of the StreetLens workflow along with input examples from a case study:

![StreetLens System Architecture and Input Examples](./figures/system_architecture.png)  
*Figure: Input examples from a case study and system architecture of StreetLens showing the flow of VLM-based neighborhood assessment.*


## Colab Notebooks

We provide two Google Colab notebooks that can be run with a free GPU quota:

1. `1_data_exploration.ipynb` – Explore the input data  
2. `2_assess_neighborhood_environment.ipynb` – Run neighborhood environment assessment


## Paper & BibTeX Citation

For more details on the methodology, see the paper: [StreetLens: Enabling Human-Centered AI Agents for Neighborhood Assessment from Street View Imagery](https://dl.acm.org/doi/10.1145/3764917.3771334)

If you find this work useful, please cite it using the following BibTeX entry:

```bibtex
@inproceedings{10.1145/3764917.3771334,
author = {Kim, Jina and Jang, Leeje and Chiang, Yao-Yi and Wang, Guanyu and Pasco, Michelle C.},
title = {StreetLens: Enabling Human-Centered AI Agents for Neighborhood Assessment from Street View Imagery},
year = {2025},
isbn = {9798400721809},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3764917.3771334},
doi = {10.1145/3764917.3771334},
booktitle = {Proceedings of the 1st ACM SIGSPATIAL International Workshop on Human-Centered Geospatial Computing},
pages = {15–19},
numpages = {5},
keywords = {automatic workflow, neighborhood environment assessment, vision-language model, prompt engineering, in-context learning},
location = {Minneapolis, MN, USA},
series = {GeoHCC '25}
}
