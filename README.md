# Agree-to-Disagree
Repository for the paper: "Agree to Disagree: Diversity through Disagreement for Better Transferability"

## Reproducing our results

### 2D toy setup

To reproduce the 2D D-BAT results (Figure 1), see [this notebook](https://github.com/mpagli/Agree-to-Disagree/blob/main/notebooks/D_BAT_2D_example.ipynb).

### Results on datasets with completely spurious correlation

* To reproduce the C-MNIST results, see [this notebook](https://github.com/mpagli/Agree-to-Disagree/blob/main/notebooks/D_BAT_C_MNIST.ipynb) (`notebooks/D_BAT_C_MNIST.ipynb`)
* To reproduce the MF-Dominoes results, see [this notebook](https://github.com/mpagli/Agree-to-Disagree/blob/main/notebooks/D_BAT_M_F_Dominoes.ipynb) (`notebooks/D_BAT_M_F_Dominoes.ipynb`)
* To reproduce the MC-Dominoes results (Table 1), see [this notebook](https://github.com/mpagli/Agree-to-Disagree/blob/main/notebooks/D_BAT_M_C_Dominoes.ipynb) (`notebooks/D_BAT_M_C_Dominoes.ipynb`)

### Results on "natural" datasets

Install WILDS: `pip install wilds`

* To reproduce the Waterbirds results, run `train-waterbird.sh`.
* To reproduce the Camelyon17 dataset, run `train-camelyon.sh`.
* To reproduce the results on the Office-Home dataset, first download the data (see `datasets/README.md` for link), then run `train-office-home-ood_is_test.sh` and `train-office-home-ood_is_not_test.sh`

### Results on uncertainty and OOD detection

* To reproduce results on the MNIST OOD setup (Figure 3), see [this notebook](https://github.com/mpagli/Agree-to-Disagree/blob/main/notebooks/D_BAT_MNIST_Uncertainty.ipynb) (`notebooks/D_BAT_MNIST_Uncertainty.ipynb`)
* To reproduce results on the CIFAR10 OOD setup (Figure 4), see [this notebook](https://github.com/mpagli/Agree-to-Disagree/blob/main/notebooks/D_BAT_CIFAR10_OOD_CIFAR100_MCDropout.ipynb) (`notebooks/D_BAT_CIFAR10_OOD_CIFAR100_MCDropout.ipynb`)
