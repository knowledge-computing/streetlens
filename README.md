# StreetLens

**StreetLens: Enabling Human-Centered AI Agents for Neighborhood Assessment from Street View Imagery**  

This is the official repository of **StreetLens**.  

StreetLens is a human-centered, researcher-configurable workflow for scalable neighborhood environmental assessments. It enables VLMs to mimic trained human coders by:

- **Domain-informed analysis:** Focuses on questions derived from established protocols.  
- **Scalable image retrieval and annotation:** Automatically retrieves relevant street view imagery (SVI) and generates semantic annotations, ranging from objective features (e.g., number of cars) to subjective perceptions (e.g., sense of disorder).  
- **Flexible integration of expertise and data:** Allows researchers to define the VLM’s role through domain-informed prompting and incorporate prior survey data for robust assessments across diverse contexts.



## 🔗 Links

- **Demo:** [https://knowledge-computing.github.io/projects/streetlens](https://knowledge-computing.github.io/projects/streetlens)
- **arXiv Paper:** [https://arxiv.org/abs/2506.14670](https://arxiv.org/abs/2506.14670)  
- StreetLens was accepted to the **GEOHCC Workshop at ACM SIGSPATIAL 2025**. Proceedings link will be updated once available.  


## 🖼️ System Architecture and Examples

Below is an overview of the StreetLens workflow along with input examples from a case study:

![StreetLens System Architecture and Input Examples](./figures/system_architecture_examples.png)  
*Figure: Input examples from a case study and system architecture of StreetLens showing the flow of VLM-based neighborhood assessment.*



## 📝 Colab Notebooks

We provide two Google Colab notebooks that can be run with a free GPU quota:

1. `1_data_exploration.ipynb` – Explore the input data  
2. `2_assess_neighborhood_environment.ipynb` – Run neighborhood environment assessment  
