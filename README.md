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

FFNNBidding: Bids are additional weights in a standard FFNN (flows). Neurons in the pre-sequent layers mask their output by the shifted bids to created a differentiable market.

SingleBidder: Single bidder using a sparsely gated mixture of experts layer. Bidder bids the load balance on the sparese gating. Children mask the outputs using the shifted bids.

MultipleBidder: Multiple bidders and multiple sellers in a sparsely gated mixture of experts layer.


## Run

```
virtualenv env && source env/bin/activate && pip install -r requirements.txt
$ jupyter notebook
```
---

## TODO 
(const) most things.
