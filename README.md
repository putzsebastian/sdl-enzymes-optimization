# Optimization algorithms for enzymatic reaction conditions

This repository contains code, data and workflows for automated testing of multiple global optimization algorithms on enzyme kinetic data. The project accompanies the article:

> **Optimized Machine Learning for Autonomous Enzymatic Reaction Intensification in a Self-driving Lab**  
> Sebastian Putz, Niklas Teetz, Michael Abt, Pascal Jerono, Thomas Meurer, Matthias Franzreb* 

---

## Overview

This project implements a variety of optimization algorithms:
- **Bayesian Optimization (BO)**
- **Particle Swarm Optimization (PSO)**
- **Simulated Annealing (SA)**
- **Genetic Algorithm (GA)**
- **Random Search (RS)**
- **Response Surface Methodology (RSM)**

These tools are applied to maximize enzyme activity in simulated experimental campaigns on a linear interpolation model of the dataset, as described in the publication.

---

## Installation

### 1. Clone the repository

- Open a terminal and run: 
- git clone https://github.com/putzsebastian/sdl-enzymes-optimization.git
- cd <your-repo>

### 2. Create and activate the environment
- Make sure to have Anaconda or Miniconda installed. Then run:
- conda env create -f environment.yml
- conda activate biocar-enzyme-opt

## Usage
- You can execute the python files in /scripts/ directly.
- For BO, PSO, GA and SA scripts for grid search of optimal hyperparameters as well as for testing fixed hyperparameters with fixed number of iterations are included.
- For RSM and RS scripts for a fixed number of iterations only are included.
- Feel free to play around and edit the scripts to vary hyperparameters of the algorithms, number of iterations etc.
- You can also implement your own optimization algorithms locally and test it on the dataset, or adapt to use your own dataset and test which algorithm is best.

## License

This project is licensed under the MIT License. See the [LICENSE] file for details.
