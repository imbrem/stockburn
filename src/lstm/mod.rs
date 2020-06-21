/*!
The LSTM implementation: a rather direct translation of https://gitlab.com/tekne/stock-lstm
*/

use tch::nn::{VarStore, Sequential};

/// The StockLSTM model from https://gitlab.com/tekne/stock-lstm
pub type StockLSTM = Sequential;

/// A descriptor for an instance of the StockLSTM model
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct StockLSTMBuilder {
    /// The number of input neurons
    pub inputs: usize,
    /// The number of stock prices to predict
    pub stocks: usize,
    /// The hidden LSTM layers in use
    pub hidden: Vec<usize>
}

impl StockLSTMBuilder {
    /// The default hidden layer size
    pub const DEFAULT_HIDDEN: usize = 512;
    /// The default number of hidden layers
    pub const DEFAULT_NO_HIDDEN: usize = 2;
    /// Describe a `StockLSTM` with a given number of input neurons and output stock prices, having the default
    /// hidden layer configuration.
    pub fn new(inputs: usize, stocks: usize) -> StockLSTMBuilder {
        StockLSTMBuilder {
            inputs,
            stocks,
            hidden: vec![Self::DEFAULT_HIDDEN; Self::DEFAULT_NO_HIDDEN]
        }
    }
    /// Build a `StockLSTM` over a given `VarStore`
    pub fn build(&self, _vs: &VarStore) -> StockLSTM {
        unimplemented!()
    }
}