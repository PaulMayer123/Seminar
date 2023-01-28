---
title: "First-Post"
date: 2022-11-19
---

# Motivation
  - Example
  - Motivate manybody systems and sampling
  - Topic (What do we need/want?) statistically independent samples x from Boltzmann Distribution
  - old approach: simulations, many steps until new state, wanted states are often rare events


- - - -
<br></br>
# Boltzmann Generators
First let's take a look at what we can do. In our example we have all the positions and forces between our molecules. 
We can therefore compute the energy of the system. With this energy we can calculate the boltzmann weight and know the 
probability of this state. <b>But</b> we only have no or very few samples. Hence, our problem is sampling.

So how do these boltzmann generators work? The key idea is a coordinate transformation. From the configuration space X
(as seen before: the positions and forces of the molecules) to a so-called latent space Z. There different states are 
close to each other. And in such a way, that we can sample from there with a gaussian. 

Since in our example this results in an 76 dimensional gaussian(which is difficult to visualize). Let's look at a simpler
example that shows the principle better:

<p align="center">
  <img src="https://raw.githubusercontent.com/PaulMayer123/seminar/main/Transform-RealNVP.png" width="750" title="Transformation">
</p>

The left part(blue) shows the configuration space of our data. After the transform we have the samples repacked in a gaussian 
like shape (right blue part). The bends and stretches of the gray lines illustrate well the transformation. The key part 
is that we this transformation is invertible. That means we can also transform in the other direction. So if we have
samples in the latent space(red right part) we can transform them into our configuration space and get something similar
to our data back. In our case we want to draw a sample in the latent space via a gaussian and then transform the sample
to our configuration space to obtain a sample for our original problem. 

We do this via a Deep Invertible Neural Network. As illustrated in the next image:
1. Sample from Gaussian
2. Transform via Neural Network
3. Reweight

<p align="center">
  <img src="https://raw.githubusercontent.com/PaulMayer123/seminar/main/Boltzmann-with-Reweighting.png" width="350" title="hover text">
</p>

We start by drawing a sample from a gaussian distribution. Then we transform it through our Network and therefore get a
sample in our configuration space. We thus generate a distribution p<sub>x</sub>. This distribution is similar to the 
boltzmann distribution, but not exact. That's why some reweighting is needed. Our Network consists out of smaller blocks.
Which we now take a closer look at.

- - - -
<br></br>
## Invertible NN

For the invertible blocks, the boltzmann generators use RealNVP(link) transformations. It uses only trivial invertible
operations, like addition and multiplication. In the image, the blue part is for the direction from the latent space to the 
configuration space and the red part for the other direction. First the input is split into 2 channels (x<sub>1</sub>, x<sub>2</sub>).
One channel remains unchanged and is only used as input to change the second input. S and T are two
<b>non</b>-invertible networks. We use the first channel as input of these networks and then multiply or add it to the 
second channel. Even though the two Networks are not invertible, we know their input and therefore can recompute it and 
then divide or subtract it from the second channel to get our original inputs back. Note that we use the same network 
both directions. We can stack those blocks to obtain a deep neural network. In order to avoid that we only change one
half of the input we swap the channel that gets modified every other block.
<br></br>
<p align="center">
  <img src="https://raw.githubusercontent.com/PaulMayer123/seminar/main/invertible2.png" width="450" title="hover text">
</p>

## Training

Why do we need invertible Blocks? There are two ways to train our network, so that we really get good, realistic samples.
And each of it requires the other direction. The first mode is called training-by-energy:

### Training by energy
1. Sample from gaussian
2. Transform through NN and generate a distribution p<sub>x</sub>

<p align="center">
  <img src="https://raw.githubusercontent.com/PaulMayer123/seminar/main/Train-by-energy.gif" width="350" title="hover text">
</p>
In the beginning p<sub>x</sub> will be very different from the boltzmann distribution. We want to minimize this 
difference. We therefore use the Kullback-Leibler-Divergence which is derived from the difference between
two distributions. So we do not need samples from the configuration space for this training mode. But it tends to focus
on the most meta-stable state. 

### Training by example

1. Start with a valid configuration (from simulation or experiments)
2. Transform through NN in other direction

<p align="center">
  <img src="https://raw.githubusercontent.com/PaulMayer123/seminar/main/Train-by-example.gif" width="350" title="hover text">
</p>
This mode is as we all know we start with valid configuration. We use our transformation in the other direction.
Training by example is especially good in the early stages, but requires configurations.
So the best way is to combine both methods together

- - - - 
<br></br>
# Examples
  - 5 Different Examples; as many depending on how much time is left

- - - -
<br></br>
# Conclusion #
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


# New ML Approach: Boltzmann generators
  - "one shot" samples
  - We can compute Boltzmann weights of a given x but we do not have these samples x
  - Coordinate Transform from configuration states x to latent space z
    - There different states are close and can be easily sampled via Gaussian
    - Invertible Neural Networks




  - Invertible Blocks
  - Training
    - 2 modes
    - by energy: procedure, explain Loss funtion
    - by example: procedure, explain Loss funtion
    - combine both
  - statistics
  - more details in appendix
