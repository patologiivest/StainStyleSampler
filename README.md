# StainStyleSampler

## Overview

The StainStyleSampler is a Python tool designed for extracting color and stain features from histological images. It leverages various color conversion techniques, stain deconvolution, and clustering methods to build feature representations, generate 2D embeddings (via PCA or UMAP), create histograms, and select representative image references. The project includes local stain deconvolution and tissue/background masking utilities, avoiding a dependency on HistomicsTK.

## Features
- Features extraction:
  - Extracts image features using different color spaces (lab, rgb, hsi).
  - Supports stain deconvolution and splitting of stain features.
  - Support for automatic clustering algorithms.
- Features visualization:
  - Generates 2D embeddings using PCA or UMAP.
- Reference Selection:
  - Supports various reference selection methods including random, representative, grouped, and density-based approaches.

## Dependencies


## Installation
1. Clone the Repository:
```
   git clone https://github.com/patologiivest/StainStyleSampler.git
   cd StainStyleSampler
```
4. Create a Virtual Environment:
```
  conda env create -f "env-file.yml"
  source venv/bin/activate  # On Windows: venv\Scripts\activate
```
