## Gradient Bidding

```
███████╗ ██████╗ ██████╗     █████╗ ██╗
██╔════╝██╔═══██╗██╔══██╗   ██╔══██╗██║
█████╗  ██║   ██║██████╔╝   ███████║██║
██╔══╝  ██║   ██║██╔══██╗   ██╔══██║██║
██║     ╚██████╔╝██║  ██║██╗██║  ██║██║
╚═╝      ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝
```

## TL;DR
Conditional computation as a payoff maximization problem.

## Motivation

Various things including p2p networks.

---

## Experiments

[**FFNNBidding.ipynb**](https://github.com/unconst/GradientBidding/blob/master/FFNNBidding.ipynb): Bids are additional weights in a standard FFNN (flows). Neurons in the pre-sequent layers mask their output by the shifted bids to created a differentiable market.

[**SingleBidder.ipynb**](https://github.com/unconst/GradientBidding/blob/master/Singlebidder.ipynb): Single bidder using a sparsely gated mixture of experts layer. Bidder bids the load balance on the sparese gating. Children mask the outputs using the shifted bids.

[**MultipleBidder.ipynb**](https://github.com/unconst/GradientBidding/blob/master/Multibidder.ipynb): Multiple bidders and multiple sellers in a sparsely gated mixture of experts layer.

[**PriceOfAnarchy.ipynb**](https://github.com/unconst/GradientBidding/blob/master/PriceOfAnarchy.ipynb): Comparison of interactions between multiple bidders in either a cooperative setting or a competitive setting


## Run

To run this, there are two options:
**Option 1**: Clone the repository, set up a `virtualevn`, and open Jupyter Notebook
```
git clone https://github.com/unconst/GradientBidding.git
virtualenv env && source env/bin/activate && pip install -r requirements.txt
$ jupyter notebook
```
**Option #2**: Open one of the notebooks in Google Colab

| **Notebook Name**    | **Google Colab Link**                                                                                                                                                           |
|----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FFNNBidding.ipynb    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unconst/GradientBidding/blob/master/FFNNBidding.ipynb)    |
| SingleBidder.ipynb   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unconst/GradientBidding/blob/master/SingleBidder.ipynb)   |
| MultipleBidder.ipynb | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unconst/GradientBidding/blob/master/MultipleBidder.ipynb) |
| PriceOfAnarchy.ipynb | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unconst/GradientBidding/blob/master/PriceOfAnarchy.ipynb) |

---

## TODO 
(const) most things.
