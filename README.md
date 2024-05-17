# Optimizer Visualization using MATLAB
## Overview
This project provides interactive visualizations to compare the convergence behavior of popular optimization algorithms that I implemented from scratch on differnet mathematical functions/terrains:
* __Gradient Descent__
* __Gradient Descent with Momentum (Momentum)__
* __Nesterov Accelerated Gradient (NAG)__
* __Adaptive Gradient (AdaGrad)__
* __RMSProp__
* __AdaDelta__
* __Adaptive Moment Estimation (Adam)__
* __Nesterov-accelerated Adaptive Moment Estimation (Nadam)__
* __AdamW__
* __Ranger__
  
By observing the animations, you can gain valuable insights into how these algorithms navigate towards the global minimum of a cost function.

## Motivation
Understanding the behavior and performance of optimization algorithms is essential in various fields such as machine learning, optimization problems, and mathematical modeling. This project aims to provide a visual comparison of commonly used optimizers to aid in understanding their strengths and weaknesses.

## Requirements
* MATLAB
* Global Optimization Toolbox in MATLAB
* Symbolic Math Toolbox in MATLAB

# Installation
1. Clone the repository:
```
git clone https://github.com/toan-ly/Optimizer-Visualization.git
```

# Usage
## Running the Simulations
To run the simulations, execute the optim_visualization script in MATLAB:
```
run('optim_visualization.m')
```
This will visulize the optimization process for different algorithms over multiple terrains.

## Customizing the Parameters
You can customize the parameters in the main script (`optim_visualization.m`):
* `learning_rates`: Adjust the learning rates for each optimizer.
* `grad_thres`: Gradient threshold for stopping criteria.
* `thres`: Loss threshold for convergence.
* `total_iter`: Total number of iterations for each optimizer.
* `map_size`: Size of the optimization landscape.
* `num_steps`: Number of repetitions for each function.

## Demo
