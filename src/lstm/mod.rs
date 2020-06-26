/*!
The LSTM implementation: a rather direct translation of https://gitlab.com/tekne/stock-lstm
*/

use crate::data::{Prediction, Tick};
use tch::nn::{self, LSTMState, Linear, Module, RNNConfig, VarStore, LSTM, RNN};
use tch::{Reduction, Tensor};

/// The StockLSTM model from https://gitlab.com/tekne/stock-lstm
#[derive(Debug)]
pub struct StockLSTM {
    /// The number of additional inputs
    pub additional_inputs: usize,
    /// The number of stock inputs
    pub stock_inputs: usize,
    /// The number of stock outputs
    pub stock_outputs: usize,
    /// This model's LSTM layer
    pub lstm_layer: LSTM,
    /// This model's linear layer
    pub linear_layer: Linear,
}

impl StockLSTM {
    /// Compute the loss on a set of inputs and outputs, modifying LSTM state in the process
    pub fn loss(&self, xs: &Tensor, ys: &Tensor, state: &LSTMState) -> (Tensor, LSTMState) {
        let (yhat, state) = self.seq_init(xs, state);
        let loss = yhat.mse_loss(ys, Reduction::Sum);
        (loss, state)
    }
}

impl RNN for StockLSTM {
    type State = LSTMState;
    fn zero_state(&self, batch_dim: i64) -> LSTMState {
        self.lstm_layer.zero_state(batch_dim)
    }
    fn step(&self, input: &Tensor, state: &LSTMState) -> LSTMState {
        self.lstm_layer.step(input, state)
    }
    fn seq_init(&self, input: &Tensor, state: &LSTMState) -> (Tensor, LSTMState) {
        let (hidden, state) = self.lstm_layer.seq_init(input, state);
        let output = self.linear_layer.forward(&hidden);
        (output, state)
    }
    fn seq(&self, input: &Tensor) -> (Tensor, LSTMState) {
        let (hidden, state) = self.lstm_layer.seq(input);
        let output = self.linear_layer.forward(&hidden);
        (output, state)
    }
}

/// A descriptor for an instance of the StockLSTM model
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct StockLSTMDesc {
    /// The number of additional input neurons
    pub additional_inputs: usize,
    /// The number of input stocks
    pub stock_inputs: usize,
    /// The number of stock prices to predict
    pub stock_outputs: usize,
    /// The size of the hidden LSTM layers to use
    pub hidden: usize,
    /// The number of hidden LSTM layers to use
    pub layers: usize,
}

impl StockLSTMDesc {
    /// Build a `StockLSTM` over a given `VarStore `
    pub fn build(&self, vs: &VarStore) -> StockLSTM {
        let inputs = self.additional_inputs + self.stock_inputs * Tick::NN_FIELDS;
        let lstm_layer = nn::lstm(
            &vs.root(),
            inputs as i64,
            self.hidden as i64,
            RNNConfig {
                has_biases: true,
                num_layers: self.layers as i64,
                dropout: 0.,
                train: true,
                bidirectional: false,
                batch_first: true,
            },
        );
        let linear_layer = nn::linear(
            &vs.root(),
            self.hidden as i64,
            (self.stock_outputs * Prediction::NN_FIELDS) as i64,
            Default::default(),
        );
        StockLSTM {
            stock_inputs: self.stock_inputs,
            additional_inputs: self.additional_inputs,
            stock_outputs: self.stock_outputs,
            lstm_layer,
            linear_layer,
        }
    }
}
