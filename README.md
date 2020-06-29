# README

# Breathing *k*-means

This directory contains the reference implementation of "Breathing *k*-means" and supplementary material supporting a current conference submission (see preprint at arxiv (tbd))

## Installation using conda
create the conda environment

```
$ conda env create -f environment.yml
```
activate the created environment

```
$ conda activate bkmenv
```
## Running

```
$ python src/bkm.py
```

This makes a run on a built-in test problem. For a number of diverse examples please run the enclosed jupyter notebook (which makes use of bkm.py)

```
$ jupyter lab notebooks/bkm.ipynb
```

## Content
The top level folder "supplement" contains the following sub folders
* data/ - (data sets used in the notebook)
* notebooks/ - (contains a jupyter notebook with examples)
* src/  
  * bkm.py - reference implementation of breathing k-means
  * aux.py - plotting functions
  * mydataset.py - general class to administer problem data sets
  * runfunctions.py  - wrapper functions used in the notebook

## Acknowledgements
Kudos go the [scikit-learn](https://scikit-learn.org/) team  which accelerated their [sklearn.cluster.KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans) class immensely from  version 0.21.1 to 0.23.1. This lead to the decision to replace a previous GPU-based k-means implementation and use sklearn.cluster.KMeans as base class in the breathing k-means reference implementation making it very compact and portable while requiring only a CPU. 

