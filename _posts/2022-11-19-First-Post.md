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
  <p align="center">
  <img src="https://raw.githubusercontent.com/PaulMayer123/seminar/main/invertible2.png" width="350" title="hover text">
    </p>

# ML part
  - Invertible Blocks
  - Training
    - 2 modes
    - by energy: procedure, explain Loss funtion
    - by example: procedure, explain Loss funtion
    - combine both
  - statistics
  - more details in appendix
  ![Alt text](../../../blob/main/invertible2.png?raw=true "Block")
 
 # Examples
  - 5 Different Examples; as many depending how much time is left

# Discussion/Conclusion
 - limitations
