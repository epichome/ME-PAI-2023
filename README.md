# Probabilistic Artificial Intelligence 2023
Projects for successful completion of the course Probabilistic Artifical Intelligence led by Prof. Andreas Krause at ETH Zurich, winter semester 2021. Partial code skeletons were provided and core logic and algorithms were implemented by the students. Evaluation was performed on black-box test data in a fixed environment, with a pass for successfully beating the performance baselines.

## Project 0 
Very simple exact bayesian inference outputting posterior probabilities, given fixed prior probabilities for sampling from three known distributions (Normal, Laplace, Student).

## project 1
Implimentation of Gaussian Process regression in order to model air pollution and predict the concentration of polution per cubic meter of air at previously unmeasured locations.

## project 2
Implimentation of a SWA-Gaussian to classify land-use patterns from satellite images. The implimentation also solved the problem of ambigious where one image could fit in on multiple classifications. 

![Classification of satellite images](Media/satellite.png?raw=true "Satellite images")

## project 3
Tuned the structual features of a drug candidate, which affects its absorption and distribution using Bayesian optimization. I used the Expected improvement (EI) as the aqusition function.

## project 4
implemented Soft Actor Critic (SAC), an off-policy Reinforcement learning algorithm, that by practicing on a simulator learned to control a policy for a Pendulum and make it swing up from an angle of π to 0. 

![Pendulum](Media/pendulum_episode.mp4?raw=true "Pendulum")