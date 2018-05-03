# `gp_grid`
Tools for exact GP inference on massive image, video, spatial-temporal, or multi-output datasets.

# Installation & Setup
Only python 2 is supported.
An up-to-date Anaconda distribution is recommended in addition to the following non-standard dependencies (using `pip`):
```
pip install tabulate
pip install GPy # v1.6.1 tested but other versions should work
```

# Tutorials (under construction...)
* [Gaussian Process Inference on Full Grids](./tutorials/full_grid_tutorial.ipynb): This simple tutorial takes you though performing efficient Gaussian process inference on fully structure grids with no missing observations.
* [Gaussian Process Inference on Grids with Missing Observations](./tutorials/gappy_grid_tutorial.ipynb): This tutorial considers the general case where the input data is structured on a grid, however, some input response are missing. It will consider multi-output senarios as well as senarios where several input dimensions form a dimension of a grid.

# Citation
The underlying algorithms are based on the 2018 SDM paper:

```
@inproceedings{evans_gp_grid,
  title={Exploiting Structure for Fast Kernel Learning},
  author={Evans, Trefor W. and Prasanth B. Nair},
  booktitle={SIAM International Conference on Data Mining},
  year={2018},
  pages={414-422}
}
```
