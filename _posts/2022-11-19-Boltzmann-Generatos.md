---
title: "Boltzmann-Generators"
date: 2022-11-19
tags: "Paul Mayer"
---

<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

## What's the Problem?

What is the probability that a protein will be folded at a given temperature? This and many more questions like this are
part of statistical mechanics, where we try to describe the average behaviour of many copies of the same system. But how
can we compute such probabilities? Simply look at all possible configurations of all folded and unfolded proteins? Sadly,
the enumeration of all these states is infeasible, and we therefore have to sample from their
equilibrium distribution to compute statistics about the system. 
<!--(What do we need/want?) statistically independent samples x from Boltzmann Distribution -->

In this blog I present the new approach to generate "one-shot" samples from the paper 
<a href="https://www.science.org/doi/10.1126/science.aaw1147" target="_blank">Boltzmann generators: Sampling 
equilibrium states of many-body systems with deep learning</a>. I will focus mainly on the machine learning tools they used
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
system consists of 36 molecules. The focus lies on the two colored particles in figure <a href="#ImageDenseSystem">1</a>. These can be in two
main states: closed (left) or open (right). The transition from one to the other is a rare but interesting event. Additionally,
a possible interesting statistic is the probability that the dimer is closed or open. 

<figure align="center">
    <div id="ImageDenseSystem">
        <img src="https://raw.githubusercontent.com/PaulMayer123/seminar/main/Dense-closed.png" width="250" title="hover text">
        <img src="https://raw.githubusercontent.com/PaulMayer123/seminar/main/Dense-open.png" width="252.25" title="hover text">
        <figcaption>Fig.1 Repulsive particle system with bistable dimer. Closed (blue, left), Open (red, right)  - F. Noé, S. Olsson, J. Köhler, H. Wu. (2019) <a href="#Boltzmann">[1]</a></figcaption>
    </div>
</figure>
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

<figure align="center">
    <div name="ImageRealNvp">
        <img src="https://raw.githubusercontent.com/PaulMayer123/seminar/main/Transform-RealNVP.png" width="750" title="Transformation">
        <figcaption>Fig.2 Transformation -  Dinh, Laurent, Jascha Sohl-Dickstein, and Samy Bengio (2016) <a href="#RealNvp">[2]</a></figcaption>
    </div>
</figure>

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

<figure align="center">
    <div name="ImageWhole">
        <img src="https://raw.githubusercontent.com/PaulMayer123/seminar/main/Boltzmann-with-Reweight.png" width="350" title="hover text">
        <figcaption>Fig.3 Boltzmann Generators - F. Noé, S. Olsson, J. Köhler, H. Wu. (2019) <a href="#Boltzmann">[1]</a></figcaption>
    </div>
</figure>

We start by drawing a sample from a gaussian distribution. Then we transform it through our Network and therefore get a
sample in our configuration space. We thus generate a distribution p<sub>x</sub>. This distribution is similar to the 
boltzmann distribution, but not exact. That's why some reweighting is needed. Our Network consists out of smaller blocks.

- - - -
<br>
## Input
So how does a configuration and therefore input to our network look like? For our dimer example, we have \\( n_s = 36 \\)
solvent particles and the two dimer molecules. The input vector is simply the alternating x and y position of each particle:
<div align="center">
    <a name="ImageInput">
        <img src="https://raw.githubusercontent.com/PaulMayer123/seminar/main/input-vector.png" width="350" title="hover text">
    </a>
</div>
<br>
With this input vector we can compute the energy of the system as follows:
<div align="center">
    <a name="ImageEquations">
        <img src="https://raw.githubusercontent.com/PaulMayer123/seminar/main/energy-equation.png" width="600" title="hover text">
    </a>
</div>


&& \begin{align}
    U(x) &= k_d(x_{1x} + x_{2x})^2 + k_dx_{1y}^2 + k_dx_{2y}^2 \\
    &+ \frac{1}{4} a (d -d_0)^4 - \frac{1}{2} b (d - d_0)^2 + c(d - d_0)^4 \\
    &+ \sum_{i=1}^{n+2} h(-x_{ix} - l_{box})k_{box}(-x_{ix} - l_{box})^2 + \sum_{i=1}^{n+2} h(x_{ix} - l_{box})k_{box}(x_{ix} - l_{box})^2 \\
    &+ \sum_{i=1}^{n+2} h(-x_{iy} - l_{box})k_{box}(-x_{iy} - l_{box})^2 + \sum_{i=1}^{n+2} h(x_{iy} - l_{box})k_{box}(x_{iy} - l_{box})^2 \\
    &+ \epsilon \sum_{i=1}^{n+1} \sum_{j=i+1,j \neq 2}^{n+2} (\frac{\sigma}{ \lVert x_i - x_j \rVert })^{12} \\
    \end{align}
&&
 
\\[
x = [x_{1x}, x_{1y}, x_{2x}, x_{2y}, \dots, x_{(n_s+2)x}, x_{(n_s+2)y}]
\\]
Where \\( d = \lVert x_1 - x_2 \rVert \\) is the distance between the dimer particles, \\( k_d \\) the strength of the bond 
between the dimer particles, a,b and c are coefficients that describe the energy curve for the dimer, \\( d_0 \\) the equilibrium
distance between the dimer particles, \\( l_{box} \\) the length of the bounding box of the system, \\( k_{box} \\) the strength 
of the bond between the particles and the bound of the box, \\( \epsilon \\) strength of the repulsion between particles,
\\( \sigma \\) distance between two particles so that energy between them is zero and
h the step function. The first row are the energy cost of moving the dimer particles in x and y direction.
The second row describes the interaction between the dimer molecules. The third and fourth line is for the box constraints
on the edges of our system (x and y direction). And the last row describes the interaction therefore repulsion of the other
particles. The parameters were chosen as shown in table 1. With this equation we can therefore compute the energy given
a sample x and with the energy we can compute the corresponding boltzmann weight.

\\[
\begin{array}{|c|c|c|c|c|c|c|c|c|c|}
    \text{Parameter} & \epsilon & \sigma & k_d & d_0 & a & b & c & l_{box} & k_{box} \\
    \text{Value} & 1.0< & 1.1 & 20.0 & 1.5 & 25.0 & 10.0 & -0.5 & 3.0 & 100.0 \\
\end{array}
\\]


\\[
<div name="Table">
    <table>
        <tr>
            <th>Parameter</th>
            <th> \\( \epsilon \\) </th>
            <th><math>\sigma</math></th>
            <th><math> k_d </math></th>
            <th><math><mr> d <msub>0</msub> </mr></math></th>
            <th>a</th>
            <th>b</th>
            <th>c</th>
            <th><math>l <sub>box</sub> </math></th>
            <th>\\( k_{box} \\)</th>
        </tr>
        <tr>
            <th>Value</th>
            <td>1.0</td>
            <td>1.1</td>
            <td>20.0</td>
            <td>1.5</td>
            <td>25.0</td>
            <td>10.0</td>
            <td>-0.5</td>
            <td>3.0</td>
            <td>100.0</td>
        </tr>
    </table>
    <figcaption>Table 1 Parameters - F. Noé, S. Olsson, J. Köhler, H. Wu. (2019) <a href="#Boltzmann">[1]</a></figcaption>
</div>
\\]

<figure align="center">
    <div name="table1">
        <img src="https://raw.githubusercontent.com/PaulMayer123/seminar/main/table1.png" width="350" title="hover text">
        <figcaption>Table 1 Parameters - F. Noé, S. Olsson, J. Köhler, H. Wu. (2019) <a href="#Boltzmann">[1]</a></figcaption>
    </div>
</figure>


## Invertible NN
Let's look at the smaller blocks that make up our network. These blocks are invertible and the boltzmann generators use RealNVP transformations. It uses only trivial invertible
operations, like addition and multiplication. In the image, the blue part is for the direction from the latent space to the 
configuration space and the red part for the other direction. First the input is split into 2 channels \\( (x_1, x_2) \\).
One channel remains unchanged and is only used as input to change the second input. S and T are two
<b>non</b>-invertible networks. We use the first channel as input of these networks and then multiply or add it to the 
second channel. Even though the two Networks are not invertible, we know their input and therefore can recompute it and 
then divide or subtract it from the second channel to get our original inputs back. Note that we use the same network 
both directions. In order to avoid that we only change one half of the input we swap the channel that gets modified every other layer.
A block consist of 2 layers one modification of each channel. We can stack those blocks to obtain a deep neural network.
For our running example 8 blocks (with 2 layers each) were used. Furthermore, the networks S and T consist of 3 hidden
layers with 200 neurons.
<br>
<figure align="center">
    <div name="ImageInvertible">
        <img src="https://raw.githubusercontent.com/PaulMayer123/seminar/main/invertible2.png" width="450" title="hover text">
        <figcaption>Fig.4 Real NVP Transformation Block </figcaption>
    </div>
</figure>

- - - -  
<br>
## Training

Why do we need invertible Blocks? There are two ways to train our network, so that we really get good, realistic samples.
And each of it requires the other direction. The first mode is called training-by-energy:

### Training by energy
1. Sample from gaussian
2. Transform through NN and generate a distribution p<sub>x</sub>

<figure align="center">
    <div name="ImageTrainByEnergy">
        <img src="https://raw.githubusercontent.com/PaulMayer123/seminar/main/Train-by-energy.gif" width="400" title="hover text">
        <figcaption>Fig.5 Training by Energy - adapted from: F. Noé, S. Olsson, J. Köhler, H. Wu. (2019) <a href="#Boltzmann">[1]</a></figcaption>
    </div>
</figure>
In the beginning p<sub>x</sub> will be very different from the boltzmann distribution. We want to minimize this 
difference. We therefore use the Kullback-Leibler-Divergence which is derived from the difference between
two distributions. So we do not need samples from the configuration space for this training mode. But it tends to focus
on the most meta-stable state. 

### Training by example

1. Start with a valid configuration (from simulation or experiments)
2. Transform through NN in other direction

<figure align="center">
    <div name="ImageTrainByExample">
        <img src="https://raw.githubusercontent.com/PaulMayer123/seminar/main/Train-by-example.gif" width="400" title="hover text">
        <figcaption>Fig.6 Training by Example - adapted from: F. Noé, S. Olsson, J. Köhler, H. Wu. (2019) <a href="#Boltzmann">[1]</a></figcaption>
    </div>
</figure>
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

<figure align="center">
    <div name="FreeEnergy">
        <img src="https://raw.githubusercontent.com/PaulMayer123/seminar/main/Dense-FreeEnergyDiff.png" width="350" title="hover text">
        <figcaption>Fig.7 Free Energy Difference - F. Noé, S. Olsson, J. Köhler, H. Wu. (2019) <a href="#Boltzmann">[1]</a></figcaption>
    </div>
</figure>

For one transition from one meta-stable state to the other and back, the simulation needs 10<sup>12</sup> steps. To get
the same precision as the boltzmann generators we need 100 of those transitions. On the other hand the boltzmann generators
need 2*10<sup>7</sup> energy evaluation in the training process. This is a significant speed-up by 7 orders of magnitude!
In addition, the samples are independent and "one-shot". That means we can draw as many samples as we want without significant
computations.

## Transition Paths
What else can we do with the transformation? If we take our 2 meta-stable states, we can do a linear interpolation in the
latent space. If we transform this path back to the configuration space, we obtain possible and realistic transition paths
from one to the other. One of these paths can be seen in the next image.


<figure align="center">
    <div  name="transitionPath">
        <img src="https://raw.githubusercontent.com/PaulMayer123/seminar/main/transition-paths.png" width="400" title="hover text">
        <figcaption>Fig.8 Transition Path - F. Noé, S. Olsson, J. Köhler, H. Wu. (2019) <a href="#Boltzmann">[1]</a></figcaption>
    </div>
</figure>

<!-- exploration -->


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
we can use this approach. 

- - - -
<br>
# References
- <a href="https://www.science.org/doi/10.1126/science.aaw1147" target="_blank" id="Boltzmann">[1]</a> F. Noé, S. Olsson, J. Köhler, H. Wu; Boltzmann generators: sampling equilibrium states of many-body systems with deep learning; Science, 365 (2019)
- <a href="https://arxiv.org/abs/1605.08803" target="_blank" name="RealNvp">[2]</a> Dinh, Laurent, Jascha Sohl-Dickstein, and Samy Bengio. "Density estimation using real nvp." arXiv preprint arXiv:1605.08803 (2016)
- <a href="https://youtu.be/WuXJRswYIaA" target="_blank" name="NoeYoutube">[3]</a> Frank Noe. (2020, 26. September). MLDS 2020 - 3 Boltzmann Generators. YouTube. https://youtu.be/WuXJRswYIaA
- <a href="https://www.youtube.com/watch?v=2S3xYRLy2cI" target="_blank" name="PhysicsYoutube">[4]</a> ICTP Condensed Matter and Statistical Physics. (2021, 16. December). Enhanced sampling in Molecular Dynamics: Why is it necessary?. Youtube. https://www.youtube.com/watch?v=2S3xYRLy2cI

