# Probabilistic Artificial Intelligence 2023
My projects required for successful completion of the course Probabilistic Artifical Intelligence led by Prof. Andreas Krause at ETH Zurich, fall semester 2023. Partial code skeletons were provided and core logic and algorithms were implemented by the students. Evaluation was performed on black-box test data in a fixed environment, with a pass for successfully beating the performance baselines.

## Project 0 
Very simple exact bayesian inference outputting posterior probabilities, given fixed prior probabilities for sampling from three known distributions (Normal, Laplace, Student).

Baseline: 0.98
Grade: 1.0

## Project 1
Implimentation of Gaussian Process regression in order to model air pollution and predict the concentration of polution per cubic meter of air at previously unmeasured locations.

Easy Baseline: 45.48
Medium Baseline: 26.98
Hard Baseline: 21.791

Grade: 4.333

## Project 2
Implimentation of a SWA-Gaussian to classify land-use patterns from satellite images. The implimentation also solved the problem of ambigious where one image could fit in on multiple classifications. 

<p align="left">
  <img src="https://github.com/epichome/Probabilistic-Artificial-Intelligence-2023/blob/main/Media/satellite.png" height="320">  
</p>

Easy Baseline: 0.900969364643097
Medium Baseline: 0.8562244963645935
Hard Baseline: 0.840867350101471

Grade: 0.8389560456178626

## Project 3
Used Bayesian optimization in order to tune the structual features of a drug candidate, which affected its absorption and distribution. I used the Expected improvement (EI) as the aqusition function.

Easy Baseline: 0.6
Medium Baseline: 0.685
Hard Baseline: 0.785

Grade: 0.824272736002454

## Project 4
Implimetation of Soft Actor Critic (SAC), an off-policy Reinforcement learning algorithm, that by practicing on a simulator learned to control a policy for a Pendulum and make it swing up from an angle of π to 0. 

<p align="left">
  <img src="https://github.com/epichome/Probabilistic-Artificial-Intelligence-2023/blob/main/Media/pendulum_episode.gif" >  
</p>

Easy Baseline: -739.2
Medium Baseline: -534.0
Hard Baseline: -388.1

Grade: -366.1