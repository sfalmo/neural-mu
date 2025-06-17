# Neural $\mu$

This repository supplements the following publication:

**Determining the chemical potential via universal density functional learning**  
*Florian Sammüller and Matthias Schmidt.*


## Instructions

### Setup

A recent version of [Julia](https://julialang.org/downloads/) needs to be installed on your system (Julia 1.11.5 was used for development).

Launch the Julia interpreter within this directory and type `]` to enter the package manager.
Activate the environment and install the required packages as follows:

```julia
activate .
instantiate
```

Type backspace to exit the package manager.

### Data

Download and extract pregenerated datasets, models and predictions (~700MB):

```julia
include("get_data.jl")
```

### Usage

Models can be trained from scratch in `Train.ipynb`.
Trained models are used for predictions of self-consistent density profiles in `Predict.ipynb`.
The plots of the paper are reproduced in `Plot.ipynb`, including the comparison of chemical potential values.
Models and common functions are defined in `utils.jl`.
