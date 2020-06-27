# README

# Breathing *k*-means

This directory contains the reference implementation of "Breathing *k*-means" and supplementary material supporting a current conference submission (see preprint at arxiv (tbd))

## Installation using conda
### create the conda environment

  >conda env create -f environment.yml

### activate the created environment

  >conda activate bkmenv


## Test Run


```
python src/bkm.py
```

This makes a run on a built-in test problem. For detailed simulations please see the jupyter notebook which makes use of bkm.py

### 

## Content
The top level folder "supplement" contains the following sub folders
* data  
(data sets used in the notebook)
* notebooks  
(contains a jupyter notebook with demonstrations and reconstructions of several figures in the paper)
* src  
contains the implementation and auxiliary files

  * src/bkm.py - stand-alone implementation of breathing k-means   
    can also be included as module


  * aux.py  
  plotting functions

  * mydataset.py  
    general class to administer data sets (clustering problems)

  * runfunctions.py  
    wrapper functions used in the notebook to run our algorithm and k-means++
    on problems and generate figures illustrating the result



* notebooks

    * bkm.ipynb - jupyter notebook with examples


## Training

(no training needed since the algorithm does not contain adaptive parameters).

## Evaluation

To evaluate the algorithm please open the accompanying notebook and follow the sugggestions there:

```eval
jupyter lab notebooks/bkm.ipynb
```
