/*!
The LSTM implementation: a rather direct translation of https://gitlab.com/tekne/stock-lstm
*/

use crate::data::{Prediction, Tick};
use num::NumCast;
use std::iter::Peekable;
use tch::nn::{self, LSTMState, Linear, Module, RNNConfig, VarStore, LSTM, RNN};
use tch::{Reduction, Tensor};

/// The StockLSTM model from https://gitlab.com/tekne/stock-lstm
#[derive(Debug)]
pub struct StockLSTM {
    /// The number of additional inputs
    pub additional_inputs: usize,
    /// The number of date inputs
    pub date_inputs: usize,
    /// The number of stock inputs
    pub stock_inputs: usize,
    /// The number of stock outputs
    pub stock_outputs: usize,
    /// The number of fields per prediction
    pub prediction_outputs: usize,
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
    /// Package a batch of sequences of ticks and additional data into a vector
    pub fn make_batches<'a, A, DF, I, D, F>(
        &self,
        mut additional: A,
        mut time_func: DF,
        tick_iterators: &mut [Peekable<I>],
        batch_size: usize,
        sequence_length: usize,
    ) -> Option<(Tensor, Tensor)>
    where
        A: Iterator<Item = &'a [f32]>,
        I: Iterator<Item = Tick<D, F>>,
        F: Copy + NumCast,
        D: Copy + Ord,
        DF: FnMut(D, &mut Vec<f32>),
    {
        // Step 1: verify basic invariants
        assert_eq!(
            tick_iterators.len(),
            self.stock_inputs,
            "Wrong number of input stocks!"
        );

        // Step 2: allocate space
        let rows = batch_size * sequence_length;
        let input_features = tick_iterators.len() * Tick::NN_FIELDS + self.additional_inputs;
        let input_size = rows * input_features;
        let mut input = Vec::<f32>::with_capacity(input_size);
        let output_features = tick_iterators.len() * Prediction::NN_FIELDS + self.additional_inputs;
        let output_size = rows * output_features;
        let mut output = Vec::<f32>::with_capacity(output_size);

        // Step 3: initialize counters
        let mut curr_t = tick_iterators
            .iter_mut()
            .filter_map(|ticks| ticks.peek().map(|tick| tick.t))
            .min()?;

        // Step 4: fill in rows
        for _row in 0..rows {
            // Step 4.a: fill in additional rows, zero filling on missing
            if let Some(additional) = additional.next() {
                let truncate_additional = additional.len().min(self.additional_inputs);
                input.extend_from_slice(&additional[..truncate_additional]);
                let additional_fill = self.additional_inputs - truncate_additional;
                input.extend(std::iter::repeat(0.0).take(additional_fill));
            } else {
                input.extend(std::iter::repeat(0.0).take(self.additional_inputs))
            }
            // Step 4.b: fill in time data
            time_func(curr_t, &mut input);
            // Step 4.c: fill in input tick data for the current date, zero filling on missing ticks
            let mut min_t: Option<D> = None;
            for ticks in tick_iterators.iter_mut() {
                if let Some(tick) = ticks.peek() {
                    // Check the date
                    if tick.t == curr_t {
                        // Write the tick, then
                        tick.push_tick(&mut input);
                        // Advance the tick iterator
                        ticks.next();
                        // Look at the time of the tick after this tick, and if necessary, update the minimum time
                        if let Some(tick) = ticks.peek() {
                            if let Some(t) = min_t {
                                if tick.t < t {
                                    min_t = Some(tick.t)
                                }
                            } else {
                                min_t = Some(tick.t)
                            }
                        }
                    } else {
                        // Mismatched time: zero fill without advancing the iterator
                        input.extend(std::iter::repeat(0.0).take(Tick::NN_FIELDS))
                    }
                } else {
                    // Empty iterator: zero fill
                    input.extend(std::iter::repeat(0.0).take(Tick::NN_FIELDS))
                }
            }
            // Step 4.d: if any ticks have been filled in, update the minimum time, moving it forwards
            if let Some(t) = min_t {
                curr_t = t;
            }
            // Step 4.e: fill in output tick data for the current date, zero filling on missing ticks
            for ticks in tick_iterators.iter_mut() {
                if let Some(tick) = ticks.peek() {
                    // Check the date
                    if tick.t == curr_t {
                        // Write the prediction associated with the tick
                        tick.pred().push_pred(&mut output)
                    } else {
                        // Mismatched time: zero fill without advancing the iterator
                        input.extend(std::iter::repeat(0.0).take(Prediction::NN_FIELDS))
                    }
                } else {
                    // Empty iterator: zero fill
                    input.extend(std::iter::repeat(0.0).take(Prediction::NN_FIELDS))
                }
            }
        }

        // Step 5: generate tensors from vectors
        let input = Tensor::from(&input[..]).view([
            batch_size as i64,
            sequence_length as i64,
            input_features as i64,
        ]);
        let output = Tensor::from(&output[..]).view([
            batch_size as i64,
            sequence_length as i64,
            output_features as i64,
        ]);

        // Return result!
        return Some((input, output))
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
    /// The number of date inputs
    pub date_inputs: usize,
    /// The number of input stocks
    pub stock_inputs: usize,
    /// The number of stock prices to predict
    pub stock_outputs: usize,
    /// The number of fields per output stock
    pub prediction_outputs: usize,
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
            (self.stock_outputs * self.prediction_outputs) as i64,
            Default::default(),
        );
        StockLSTM {
            stock_inputs: self.stock_inputs,
            additional_inputs: self.additional_inputs,
            date_inputs: self.date_inputs,
            stock_outputs: self.stock_outputs,
            prediction_outputs: self.prediction_outputs,
            lstm_layer,
            linear_layer,
        }
    }
}
