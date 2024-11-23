# Statin-Associated Muscle Symptoms Study Metabolomics Analysis

This repository contains code and notebooks for metabolomics analysis of the statin-associated muscle symptoms study. It includes interactive notebooks for data wrangling, preprocessing, and analysis of metabolomics data.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Data](#data)
- [Results](#results)
- [Dependencies](#dependencies)

## Introduction

The goal of this project is to examine the effects of fish oil
supplements on lipid profiles, especially triglycerides. This repository provides a set of tools and notebooks for preprocessing, analyzing, and visualizing the metabolomics data.

## Project Structure

- **interactive/**
  - [sams_analysis.jl](https://rawcdn.githack.com/senresearch/mlm-metabolomics-supplement/ea303b23db6dee88ce9c4c4c70ba70ea287ceed5/SAMSstudy/interactive/sams_analysis.html): Interactive Pluto notebook to run the metabolomics analysis.

- **notebooks/**
  - **extract_annotations/**
    - [BuildZdataset.ipynb](https://github.com/senresearch/mlm-metabolomics-supplement/blob/main/SAMSstudy/notebooks/extract_annotations/BuildZdataset.ipynb): Notebook that carries out the extraction of the annotations of the matbolites.
  - **preprocessing/**
    - [PreprocessingLipo.ipynb](https://github.com/senresearch/mlm-metabolomics-supplement/blob/main/SAMSstudy/notebooks/preprocessing/PreprocessingLipo.ipynb): Notebook that carries out the data wrangling process for the lipidomics data.
    - [PreprocessingMeta.ipynb](https://github.com/senresearch/mlm-metabolomics-supplement/blob/main/SAMSstudy/notebooks/preprocessing/PreprocessingMeta.ipynb): Notebook that carries out the data wrangling process for the metabolomics data.
  - **enrichement_analysis/**
    - [EnrichmentAnalysis.ipynb](https://github.com/senresearch/mlm-metabolomics-supplement/blob/main/SAMSstudy/notebooks/enrichment_analysis/EnrichmentAnalysis.ipynb): Notebook that performs metabolomic analysis related to fish oil, including enrichment analysis using overrepresentation analysis (ORA), and generates visualizations.
    

- **src/**: Source code containing functions for preprocessing and analysis.

- **data/**
  - **processed/**: Directory for processed data files (after running the wrangling and preprocessing notebooks).

- **images/**: Directory for output images generated by the analysis.

## Getting Started

### Prerequisites

- **Julia**: Required for running the Pluto notebook.
- **R**: Required for running some function in the Jupyter notebooks.

### Installation

Clone the repository:

```bash
git clone https://github.com/senresearch/mlm-metabolomics-supplement.git
cd mlm-metabolomics-supplement/SAMSstudy
```

Install the required Julia packages for the Jupyter notebooks (*Note:Pluto notebooks environment is self-contained*):

```bash
using Pkg()
Pkg.activate(".")
Pkg.instantiate()
```

## Usage

### Data Wrangling and Preprocessing

1. **Place Raw Data**: Add your raw metabolomics data to the appropriate directory.

2. **Run Wrangling Notebooks**:
   - [BuildZdataset.ipynb](https://github.com/senresearch/mlm-metabolomics-supplement/blob/main/SAMSstudy/notebooks/extract_annotations/BuildZdataset.ipynb)

3. **Run Preprocessing Notebooks**:
   - [PreprocessingLipo.ipynb](https://github.com/senresearch/mlm-metabolomics-supplement/blob/main/SAMSstudy/notebooks/preprocessing/PreprocessingLipo.ipynb)
   - [PreprocessingMeta.ipynb](https://github.com/senresearch/mlm-metabolomics-supplement/blob/main/SAMSstudy/notebooks/preprocessing/PreprocessingMeta.ipynb)
   
### Metabolomics Analysis

To perform the metabolomics analysis using the interactive Pluto notebook:

1. **Start Julia and Install Pluto**:

   ```julia
   julia
   ]

   add Pluto
   using Pluto
   Pluto.run()
   ```

2. **Open Notebook**: In your web browser, navigate to `interactive/sams_analysis.jl`.

## Data

The processed data will be saved in the `data/processed/` directory after running the preprocessing notebooks.

**Note**: The `data/processed/` directory is empty by default and will contain the processed data files after running the notebooks.

## Results

All output images and figures generated during the analysis will be saved in the `images/` directory.

## Dependencies

The file `Project.toml` contains the list of the Julia packages necessary to run the Jfunctions and Jupyter notebooks.