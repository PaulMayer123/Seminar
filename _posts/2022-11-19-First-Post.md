---
title: "First-Post"
date: 2022-11-19
---

# Motivation
  - Example
  - Motivate manybody systems and sampling
  - Topic (What do we need/want?) statistically independent samples x from Boltzmann Distribution
  - old approach: simulations, many steps until new state, wanted states are often rare events

# New ML Approach: Boltzmann generators
  - "one shot" samples
  - We can compute Boltzmann weights of a given x but we do not have these samples x
  - Coordinate Transform from configuration states x to latent space z
    - There different states are close and can be easily sampled via Gaussian
    - Invertible Neural Networks

# ML part
  - Invertible Blocks
  - Training
    - 2 modes
    - by energy: procedure, explain Loss funtion
    - by example: procedure, explain Loss funtion
    - combine both
  - statistics
  - more details in appendix
  - 
  <p align="center">
  <img src="https://raw.githubusercontent.com/PaulMayer123/seminar/main/invertible2.png" width="350" title="hover text">
    </p>
<p align="center">
  <img src="https://raw.githubusercontent.com/PaulMayer123/seminar/main/Train-by-energy.gif" width="350" title="hover text">
    </p>
 
 # Examples
  - 5 Different Examples; as many depending how much time is left

# Conclusion
We can use the Boltzmann generators for rare-event sampling problems in many-body systems. Furthermore, we obtain
independent <b>one-shot</b> samples. And it is possible with dense systems with more than 1000 dimension, as we saw in 
the last example. But the approach is not ergodic, which means it does not cover the whole configuration space. Although
in the paper some ideas are broad up to combine the Boltzmann generators with classical sampling methods to fix this.
Moreover, the Networks are always very system specific, and therefore we have to train on every configuration space from
scratch. Ideally we could pretrain the Boltzmann Generators so that we only have to fine-tune it to every special use
case. We also end up with the trade of that we do not have to do the small simulation steps, but the complexer the system
the more difficult it is to reweight and the result are not that accurate anymore. The Boltzmann generators can be
used in many topics and there are some papers that build up on it. So whenever we want to sample from a known distribution
we can use this approach.

# References

