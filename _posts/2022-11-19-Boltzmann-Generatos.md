---
title: "Boltzmann-Generators"
date: 2022-11-19
---

What is the probability that a protein will be folded at a given temperature? This and many more questions like this are
part of statistical mechanics, where we try to describe the average behaviour of many copies of the same system. But how
can we compute such probabilities? Simply look at all possible configurations of all folded and unfolded proteins? Sadly,
the enumeration of all these states is infeasible, and we therefore have to sample from their
equilibrium distribution to compute statistics about the system. 
<!--(What do we need/want?) statistically independent samples x from Boltzmann Distribution -->

In this blog I present the new approach to generate "one-shot" samples from the paper "Boltzmann generators: Sampling 
equilibrium states of many-body systems with deep learning". I will focus mainly on the machine learning tools they used
to achieve this.
- - - -
<br>
# Many-Body Problems
Many-body problems are a category of physical problems. They are about microscopic systems made of many interacting 
particles <!-- Quelle wiki -->. The underlying physical laws can be simple, but the resulting system as a whole is 
extremely complex. In condensed matter physics, the macroscopic and microscopic physical properties of matter are 
studied, in particular the solid and liquid phases formed by electromagnetic forces between atoms. The system can be
described via the equations of motion. The equations thus takes into account the mass, position, energy and forces of the 
particles. The equations of motion don't have just one solution, therefore we are talking about probabilities of certain 
states. Often the most interesting states are rare-events, like the transition of a protein from folded to unfolded or vice
versa. One example that we take a closer look at throughout this blog, is an open or closed dimer. This condensed matter
system consists of 36 molecules. The focus lies on the two colored particles in the picture <!-- ref -->. These can be in two
main states: closed (left) or open (right). The transition from one to the other is a rare but interesting event. Additionally,
a possible interesting statistic is the probability that the dimer is closed or open. 

<p align="center">
    <a name="ImageDenseSystem">
        <img src="https://raw.githubusercontent.com/PaulMayer123/seminar/main/Dense-closed.png" width="250" title="hover text">
        <img src="https://raw.githubusercontent.com/PaulMayer123/seminar/main/Dense-open.png" width="252.25" title="hover text">
    </a>
</p>
<br>
# Boltzmann Distribution <!-- Nochmal motivieren warum wir hiervon samplen wollen(was beschreibt sie,...) -->
The boltzmann distribution often appears in such problems. It takes into account the energy and temperature of the system.
The less energy of a state, the higher its probability is. In our example, the system has the lowest energy, when the 
dimer is closed or open. To transition from one to the other a high energy barrier must be overcome and therefore these
events are quite rare. If we have a given configuration of our system, we can compute the energy and thus can compute the
corresponding Boltzmann weight. <!-- Hier Beispiel U(x) angeben?? oder alles zusammen später -->

# Old Approach
The classical approach is to simulate the system. A numerical, iterative solution to the equations of motion is computed
with small steps. These steps can be in the order of femto seconds! Therefore, we need a long time to transition from one
meta-stable state to the other. For the transition from the open to closed dimer 10<sup>12</sup> simulation steps are 
needed. Furthermore, the obtained samples are often correlated to each other.

- - - -
<br>
# Boltzmann Generators
How can we use machine learning to improve the sampling? As in the name of the paper Boltzmann generators are used to 
obtain independent, "one shot" samples. So we no longer need small simulation steps.

First let's take a look at what we can do. In our example we have all the positions and forces, for a given sample,
between our molecules. We can therefore compute the energy of the system. With this energy we can calculate the boltzmann
weight and know the probability of this state. <b>But</b> we only have no or very few samples. Hence, our problem is 
sampling. <!-- HIer input? -->

So how do these boltzmann generators work? The key idea is a coordinate transformation. From the configuration space X
(as seen before: the positions and forces of the molecules) to a so-called latent space Z. There different states are 
close to each other. And in such a way, that we can sample from there with a gaussian. 

Since in our example this results in a 76 dimensional gaussian(which is difficult to visualize). Let's look at a simpler
example that shows the principle better:

<p align="center">
    <a name="ImageRealNvp">
        <img src="https://raw.githubusercontent.com/PaulMayer123/seminar/main/Transform-RealNVP.png" width="750" title="Transformation">
    </a>
</p>

The left part(blue) shows the configuration space of our data. After the transform we have the samples repacked in a gaussian 
like shape (right blue part). The bends and stretches of the gray lines illustrate well the transformation. The key part 
is that the transformation is invertible. That means we can also transform in the other direction. So if we have
samples in the latent space(red right part) we can transform them into our configuration space and get something similar
to our data back. In our case we want to draw a sample in the latent space via a gaussian and then transform the sample
to our configuration space to obtain a sample for our original problem. 

We do this via a deep invertible neural network. As illustrated in the next image:
1. Sample from Gaussian
2. Transform via Neural Network
3. Reweight

<p align="center">
    <a name="ImageWhole">
        <img src="https://raw.githubusercontent.com/PaulMayer123/seminar/main/Boltzmann-with-Reweight.png" width="350" title="hover text">
    </a>
</p>

We start by drawing a sample from a gaussian distribution. Then we transform it through our Network and therefore get a
sample in our configuration space. We thus generate a distribution p<sub>x</sub>. This distribution is similar to the 
boltzmann distribution, but not exact. That's why some reweighting is needed. Our Network consists out of smaller blocks.

- - - -
<br>
## Input
So how does a configuration and therefore input to our network look like? For our dimer example, we have n <sub>s</sub>
= 36 solvent particles and the two dimer molecules. The input vector is simply the alternating x and y position of each particle:
<p align="center">
    <a name="ImageInput">
        <img src="https://raw.githubusercontent.com/PaulMayer123/seminar/main/input-vector.png" width="350" title="hover text">
    </a>
</p>
With this input vector we can compute the energy of the system as follows:
<p align="center">
    <a name="ImageEquations">
        <img src="https://raw.githubusercontent.com/PaulMayer123/seminar/main/energy-equation.png" width="600" title="hover text">
    </a>
</p>

The details are not that important, but the first row are constraints for the center and y-position of the particle dimer.
The second row describes the interaction between the dimer molecules. The third and fourth line is for the box constraints
on the edges of our system (x and y direction). And the last row describes the interaction therefore repulsion of the other
particles

## Invertible NN
Let's look at the smaller blocks that make up our network. These blocks are invertible and the boltzmann generators use RealNVP transformations. It uses only trivial invertible
operations, like addition and multiplication. In the image, the blue part is for the direction from the latent space to the 
configuration space and the red part for the other direction. First the input is split into 2 channels (x<sub>1</sub>, x<sub>2</sub>).
One channel remains unchanged and is only used as input to change the second input. S and T are two
<b>non</b>-invertible networks. We use the first channel as input of these networks and then multiply or add it to the 
second channel. Even though the two Networks are not invertible, we know their input and therefore can recompute it and 
then divide or subtract it from the second channel to get our original inputs back. Note that we use the same network 
both directions. In order to avoid that we only change one half of the input we swap the channel that gets modified every other layer.
A block consist of 2 layers one modification of each channel. We can stack those blocks to obtain a deep neural network.
For our running example 8 blocks (with 2 layers each) were used. Furthermore, the networks S and T consist of 3 hidden
layers with 200 neurons.
<br>
<p align="center">
    <a name="ImageInvertible">
        <img src="https://raw.githubusercontent.com/PaulMayer123/seminar/main/invertible2.png" width="450" title="hover text">
    </a>
</p>

- - - -  
<br>
## Training

Why do we need invertible Blocks? There are two ways to train our network, so that we really get good, realistic samples.
And each of it requires the other direction. The first mode is called training-by-energy:

### Training by energy
1. Sample from gaussian
2. Transform through NN and generate a distribution p<sub>x</sub>

<p align="center">
    <a name="ImageTrainByEnergy">
    <img src="https://raw.githubusercontent.com/PaulMayer123/seminar/main/Train-by-energy.gif" width="400" title="hover text">
    </a>
</p>
In the beginning p<sub>x</sub> will be very different from the boltzmann distribution. We want to minimize this 
difference. We therefore use the Kullback-Leibler-Divergence which is derived from the difference between
two distributions. So we do not need samples from the configuration space for this training mode. But it tends to focus
on the most meta-stable state. 

### Training by example

1. Start with a valid configuration (from simulation or experiments)
2. Transform through NN in other direction

<p align="center">
    <a name="ImageTrainByExample">
        <img src="https://raw.githubusercontent.com/PaulMayer123/seminar/main/Train-by-example.gif" width="400" title="hover text">
    </a>
</p>
This mode is as we all know we start with valid configuration. We use our transformation in the other direction.
Training by example is especially good in the early stages, but requires configurations.
<b>So the best way is to combine both methods together.</b>
For the dimer example, we start with only 'training by example' for the first 20 epochs. After that the 'training by energy'
is also used and the whole network is trained for 2000 epochs.


## Reweigthing

Because of our network, we never have exactly the boltzmann distribution. Therefore, we need a bit of reweighting. The third
step of the boltzmann generators. Statistical mechanics offer many tools to generate the wanted distribution when p<sub>x
</sub> is sufficiently similar.
The easiest way is w(x)=e<sup>-u(x)</sup>/p<sub>x</sub>. Where e<sup>-u(x)</sup> is the boltzmann distribution that we can
compute, because we know the energy of the sample. To compute our statistics we use these new weights. And the more equal the distributions are,
the better and more accurate the statistics.

- - - - 
<br>

## Results
Let's look at the result for the system with the dimer. We recall that the dimer can be closed or open. And these states
are separated by a high energy barrier to transition from one to the other. In the latent space, we obtain a 76 dimensional
gaussian. One possible statistic is the free energy difference. In the following image we can see the black line that was
obtained by classical sampling methods. The green points are samples generated with the boltzmann generators.

<p align="center">
    <a name="FreeEnergy">
        <img src="https://raw.githubusercontent.com/PaulMayer123/seminar/main/Dense-FreeEnergyDiff.png" width="350" title="hover text">
    </a>
</p>

For one transition from one meta-stable state to the other and back, the simulation needs 10<sup>12</sup> steps. To get
the same precision as the boltzmann generators we need 100 of those transitions. On the other hand the boltzmann generators
need 2*10<sup>7</sup> energy evaluation in the training process. This is a significant speed-up by 7 orders of magnitude!
In addition, the samples are independent and "one-shot". That means we can draw as many samples as we want without significant
computations.

## Transition Paths
What else can we do with the transformation? If we take our 2 meta-stable states, we can do a linear interpolation in the
latent space. If we transform this path back to the configuration space, we obtain possible and realistic transition paths
from one to the other. One of these paths can be seen in the next image.


<p align="center">
    <a name="transitionPath">
        <img src="https://raw.githubusercontent.com/PaulMayer123/seminar/main/transition-paths.png" width="400" title="hover text">
    </a>
</p>

<!-- exploration -->

[Custom foo description](#transitionPath)

# Conclusion #
We can use the Boltzmann generators for rare-event sampling problems in many-body systems. Furthermore, we obtain
independent <b>one-shot</b> samples. And in the paper they show an example with a dense systems with more than 1000 dimension.
But the approach is not ergodic, which means it does not cover the whole configuration space. Although
in the paper some ideas are brought up to combine the boltzmann generators with classical sampling methods to fix this.
Moreover, the networks are always very system specific, and therefore we have to train on every configuration space from
scratch. Ideally we could pretrain the boltzmann generators so that we only have to fine-tune it to every special use
case. We also end up with the trade off that we do not have to do the small simulation steps, but the more complex the system
the more difficult it is to reweight and the results are not that accurate anymore. The Boltzmann generators can be
used in many topics and there are some papers that build up on it. So whenever we want to sample from a known distribution
we can use this approach. [[1]](#1)

- - - -
<br>
# References
- <a href="https://www.science.org/doi/10.1126/science.aaw1147" target="_blank" name="Boltzmann">[1]</a> F. Noé, S. Olsson, J. Köhler, H. Wu; Boltzmann generators: sampling equilibrium states of many-body systems with deep learning; Science, 365 (2019)
- <a name="RealNvp">Dinh, Laurent, Jascha Sohl-Dickstein, and Samy Bengio. "Density estimation using real nvp." arXiv preprint arXiv:1605.08803 (2016)</a>
- <a name="NoeYoutube">Frank Noe. (2020, 26. September). MLDS 2020 - 3 Boltzmann Generators. YouTube. https://youtu.be/WuXJRswYIaA</a>
- <a name="PhysicsYoutube">ICTP Condensed Matter and Statistical Physics. (2021, 16. December). Enhanced sampling in Molecular Dynamics: Why is it necessary?. Youtube. https://www.youtube.com/watch?v=2S3xYRLy2cI</a>

