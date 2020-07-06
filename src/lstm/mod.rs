/*!
The LSTM implementation: a rather direct translation of https://gitlab.com/tekne/stock-lstm
*/

use crate::data::{Prediction, Tick};
use chrono::{DateTime, NaiveDateTime, Utc};
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
    /// The number of stocks to predict
    pub stocks: usize,
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
    /// Package a batch of sequences of ticks and additional data into tensors
    fn make_batches_impl<'a, A, DF, I, F>(
        additional_inputs: usize,
        stocks: usize,
        date_inputs: usize,
        mut additional: A,
        mut time_func: DF,
        tick_iterators: &mut [Peekable<I>],
        batch_size: usize,
        sequence_length: usize,
    ) -> Option<(Tensor, Tensor)>
    where
        A: Iterator<Item = &'a [f32]>,
        I: Iterator<Item = Tick<F>>,
        F: Copy + NumCast,
        DF: FnMut(DateTime<Utc>, &mut Vec<f32>),
    {
        // Step 1: verify basic invariants
        assert_eq!(
            tick_iterators.len(),
            stocks,
            "Wrong number of input stocks!"
        );

        // Step 2: allocate space
        let rows = batch_size * sequence_length;
        let input_features =
            tick_iterators.len() * Tick::NN_FIELDS + additional_inputs + date_inputs;
        let input_size = rows * input_features;
        let mut input = Vec::<f32>::with_capacity(input_size);
        let output_features = tick_iterators.len() * Prediction::NN_FIELDS;
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
                let truncate_additional = additional.len().min(additional_inputs);
                input.extend_from_slice(&additional[..truncate_additional]);
                let additional_fill = additional_inputs - truncate_additional;
                input.extend(std::iter::repeat(0.0).take(additional_fill));
            } else {
                input.extend(std::iter::repeat(0.0).take(additional_inputs));
            }
            // Step 4.b: fill in time data
            time_func(DateTime::from_utc(curr_t, Utc), &mut input);
            // Step 4.c: fill in input tick data for the current date, zero filling on missing ticks
            let mut min_t: Option<NaiveDateTime> = None;
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
                        input.extend(std::iter::repeat(0.0).take(Tick::NN_FIELDS));
                    }
                } else {
                    // Empty iterator: zero fill
                    input.extend(std::iter::repeat(0.0).take(Tick::NN_FIELDS));
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
                        tick.pred().push_pred(&mut output);
                    } else {
                        // Mismatched time: zero fill without advancing the iterator
                        output.extend(std::iter::repeat(0.0).take(Prediction::NN_FIELDS));
                    }
                } else {
                    // Empty iterator: zero fill
                    output.extend(std::iter::repeat(0.0).take(Prediction::NN_FIELDS));
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
        return Some((input, output));
    }
    /// Package a batch of sequences of ticks and additional data into tensors
    pub fn make_batches<'a, A, DF, I, F>(
        &self,
        additional: A,
        time_func: DF,
        tick_iterators: &mut [Peekable<I>],
        batch_size: usize,
        sequence_length: usize,
    ) -> Option<(Tensor, Tensor)>
    where
        A: Iterator<Item = &'a [f32]>,
        I: Iterator<Item = Tick<F>>,
        F: Copy + NumCast,
        DF: FnMut(DateTime<Utc>, &mut Vec<f32>),
    {
        Self::make_batches_impl(
            self.additional_inputs,
            self.stocks,
            self.date_inputs,
            additional,
            time_func,
            tick_iterators,
            batch_size,
            sequence_length,
        )
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
    /// The number of stocks to predict
    pub stocks: usize,
    /// The size of the hidden LSTM layers to use
    pub hidden: usize,
    /// The number of hidden LSTM layers to use
    pub layers: usize,
}

impl StockLSTMDesc {
    /// Build a `StockLSTM` over a given `VarStore `
    pub fn build(&self, vs: &VarStore) -> StockLSTM {
        let inputs = self.additional_inputs + self.stocks * Tick::NN_FIELDS;
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
            (self.stocks * Prediction::NN_FIELDS) as i64,
            Default::default(),
        );
        StockLSTM {
            stocks: self.stocks,
            additional_inputs: self.additional_inputs,
            date_inputs: self.date_inputs,
            lstm_layer,
            linear_layer,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{
        naive::{NaiveDate, NaiveDateTime, NaiveTime},
        DateTime, Duration, Timelike, Utc,
    };
    /// Test making batches of data
    #[test]
    fn batch_making_works() {
        let t = NaiveDateTime::new(
            NaiveDate::from_ymd(2020, 06, 22),
            NaiveTime::from_hms(22, 59, 33),
        );
        let fake_stock_1: &[Tick] = &[
            Tick {
                t,
                o: 40.0,
                h: 41.0,
                l: 39.0,
                c: 40.5,
                v: 300.0,
                vw: 39.5,
                n: 2.0,
            },
            Tick {
                t: t + Duration::minutes(1),
                o: 40.5,
                h: 41.5,
                l: 38.0,
                c: 40.0,
                v: 500.0,
                vw: 40.25,
                n: 4.0,
            },
            Tick {
                t: t + Duration::minutes(2),
                o: 40.0,
                h: 42.0,
                l: 39.5,
                c: 40.0,
                v: 1000.0,
                vw: 41.25,
                n: 7.0,
            },
            Tick {
                t: t + Duration::minutes(3),
                o: 40.0,
                h: 41.0,
                l: 39.0,
                c: 40.5,
                v: 300.0,
                vw: 39.5,
                n: 2.0,
            },
            Tick {
                t: t + Duration::minutes(5),
                o: 40.5,
                h: 41.0,
                l: 39.0,
                c: 40.0,
                v: 500.0,
                vw: 40.5,
                n: 4.0,
            },
        ];
        let fake_stock_2: &[Tick] = &[
            Tick {
                t: t + Duration::minutes(1),
                o: 30.0,
                h: 31.0,
                l: 29.0,
                c: 30.5,
                v: 300.0,
                vw: 39.5,
                n: 2.0,
            },
            Tick {
                t: t + Duration::minutes(2),
                o: 30.5,
                h: 31.5,
                l: 28.0,
                c: 30.0,
                v: 400.0,
                vw: 40.25,
                n: 4.0,
            },
            Tick {
                t: t + Duration::minutes(3),
                o: 30.0,
                h: 32.0,
                l: 29.5,
                c: 30.0,
                v: 900.0,
                vw: 31.25,
                n: 7.0,
            },
            Tick {
                t: t + Duration::minutes(4),
                o: 30.0,
                h: 32.0,
                l: 29.0,
                c: 30.5,
                v: 300.0,
                vw: 39.5,
                n: 2.0,
            },
            Tick {
                t: t + Duration::minutes(5),
                o: 30.5,
                h: 31.0,
                l: 28.0,
                c: 30.0,
                v: 400.0,
                vw: 40.25,
                n: 4.0,
            },
            Tick {
                t: t + Duration::minutes(6),
                o: 30.0,
                h: 31.0,
                l: 29.5,
                c: 30.0,
                v: 900.0,
                vw: 31.25,
                n: 7.0,
            },
        ];
        let fake_stocks = &mut [
            fake_stock_1.iter().copied().peekable(),
            fake_stock_2.iter().copied().peekable(),
        ];
        let additional_data: &[&[f32]] = &[
            &[1.0, 2.0, 400.0],
            &[3.0, 4.0],
            &[5.0, 6.0],
            &[7.0, 8.0],
            &[9.0, 10.0],
            &[11.0, 12.0],
            &[13.0, 14.0],
        ];
        let time_func = |d: DateTime<Utc>, v: &mut Vec<f32>| v.push(d.minute() as f32);
        let (input_data, output_data) = StockLSTM::make_batches_impl(
            3,
            2,
            1,
            additional_data.iter().copied(),
            time_func,
            fake_stocks,
            4,
            2,
        )
        .unwrap();
        assert_eq!(
            input_data.size3().unwrap(),
            (4, 2, 3 + 1 + 2 * Tick::NN_FIELDS as i64)
        );
        assert_eq!(
            output_data.size3().unwrap(),
            (4, 2, 2 * Prediction::NN_FIELDS as i64)
        );
    }
}
