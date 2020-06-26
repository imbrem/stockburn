/*!
The LSTM implementation: a rather direct translation of https://gitlab.com/tekne/stock-lstm
*/

use tch::nn::{self, Linear, VarStore, LSTM};

/// The StockLSTM model from https://gitlab.com/tekne/stock-lstm
#[derive(Debug)]
pub struct StockLSTM {
    /// This model's LSTM layers
    pub lstm_layers: Vec<LSTM>,
    /// This model's linear layer
    pub linear_layer: Linear,
}

/// A descriptor for an instance of the StockLSTM model
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct StockLSTMBuilder {
    /// The number of input neurons
    pub inputs: usize,
    /// The number of stock prices to predict
    pub stocks: usize,
    /// The hidden LSTM layers in use
    pub hidden: Vec<usize>,
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
            hidden: vec![Self::DEFAULT_HIDDEN; Self::DEFAULT_NO_HIDDEN],
        }
    }
    /// Build a `StockLSTM` over a given `VarStore`
    pub fn build(&self, vs: &VarStore) -> StockLSTM {
        let mut lstm_layers = Vec::with_capacity(self.hidden.len());
        let mut input_size = self.inputs;
        for hidden in self.hidden.iter().copied() {
            let lstm = nn::lstm(&vs.root(), input_size as i64, hidden as i64, Default::default());
            lstm_layers.push(lstm);
            input_size = hidden;
        }
        let linear_layer = nn::linear(&vs.root(), input_size as i64, self.stocks as i64, Default::default());
        StockLSTM {
            lstm_layers,
            linear_layer
        }
    }
}
