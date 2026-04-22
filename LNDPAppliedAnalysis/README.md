# Applied Analysis of Local Node Differential Privacy (LNDP*)

This repository contains an empirical evaluation of the Local Node Differential Privacy (LNDP*) mechanism introduced in the paper "Local Node Differential Privacy" (arXiv:2602.15802) by Raskhodnikova, Smith, Wagaman, and Zavyalov.

The goal of this repository is to bridge theoretical privacy mathematics with applied systems engineering by testing the mechanism's behavior across different network topologies.

## Overview of the Experiment

The provided Python script (`ScaleFreeGraphExperiments.py`) implements the core LNDP* "blurry degree" transformation and evaluates it against two distinct graph structures:
1. Erdős–Rényi (Concentrated Topology): A standard theoretical baseline where node degrees form a bell curve.
2. Barabási–Albert (Scale-Free Topology): A heavy-tailed distribution common in real-world networks (e.g., social networks, healthcare affiliations) containing extreme outlier hubs.

### Key Components
1. LNDP Implementation: Simulates the local noise addition to bounded degree buckets.
2. Privacy Metric (TVD): Calculates the Total Variation Distance between the true blurry distribution and the noisy estimate.
3. Reconstruction Analysis: Uses the Erdős–Gallai theorem and a configuration model to test what structural information an adversary can reconstruct strictly from the stored aggregate data.

## Key Findings

The experiment demonstrates that the mechanism's effectiveness is highly dependent on the underlying graph topology:

Concentrated Graphs: The mechanism performs efficiently. Outliers naturally benefit from $k$-anonymity because they can "hide in the crowd" of the bell curve(bc no outliers, usual nodes hide amongst other usual nodes).
Heavy-Tailed (Scale-Free) Graphs: The mechanism faces an imbalanced utility-privacy trade-off. While the local noise significantly distorts the heavy tail (reducing utility for legitimate data analysis), the resulting aggregate shape still retains enough mathematical constraints for an adversary to effectively reconstruct and isolate extreme outliers. 

## Motivation

This code is intended for privacy engineers, applied scientists, and researchers to explore the engineering constraints of deploying local graph privacy mechanisms in real-world systems, particularly regarding threat models involving aggregate reconstruction.

## Requirements
* `numpy`
* `networkx`
* `matplotlib`

## Reference
* [Local Node Differential Privacy (arXiv:2602.15802)](https://arxiv.org/abs/2602.15802)
