/*!
An LSTM for predicting stock prices written in Rust using PyTorch bindings, as an experiment.

Designed to be run on [Polygon](https://polygon.io/) stock data, but also to be modular, extensible and easily modifiable.
Based off the [Julia](https://julialang.org/) code in the [stock-lstm](https://gitlab.com/tekne/stock-lstm) repository, which
was based off the [Knet](https://github.com/denizyuret/Knet.jl) machine learning framework.
*/
#![forbid(missing_docs)]

pub mod data;

/// The floating point type to be used for CPU calculations
pub type CpuFloat = f64;

/// The floating point type to be used for GPU calculations
pub type GpuFloat = f32;