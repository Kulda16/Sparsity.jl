# Sparsity.jl
Repository with Julia 1.6 version code used in Master's Thesis.
All script are available on master branch.

## Cloning from Master Branch
```
git clone --branch master https://github.com/Kulda16/Sparsity.jl
```

## FIRST EXPERIMENT - sparse_regression.jl, sparse_regression_l1_pareto.jl, v_optimizer.jl

* Part 1 - classic pipeline for ML in Julia language to train and validate model.
* Part 2 - L1 regularization with Pareto frontiers.
* Part 3 - variational methods applied to this experiment in order to find sparse solutions.

## SECOND EXPERIMENT - iris_various.jl

* same methods from experiment 1 applied to the Iris problem. Classification on deeper neural network with trying to prune it.

## THIRD EXPERIMENT - mill.jl

* same methods applied to multi-instance learning experiment with the Musk dataset. Again, the main goal of this experiment, is to prune this deep neural network to make it more interpretable.
