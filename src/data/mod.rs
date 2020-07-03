/*!
Data processing and IO functions
*/
use crate::*;
use chrono::{DateTime, Utc};
use num::NumCast;
use serde::{Deserialize, Serialize};
use ta::{Close, High, Low, Open, Volume};

pub mod fake;
pub mod polygon;
pub mod scale;

/// Tick data for a stock
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Tick<D = DateTime<Utc>, F = CpuFloat> {
    /// This tick's timestamp
    pub t: D,
    /// The volume traded this tick
    pub v: F,
    /// The volume weighted average price of this tick
    pub vw: F,
    /// The opening price of this tick
    pub o: F,
    /// The closing price of this tick
    pub c: F,
    /// The high price of this tick
    pub h: F,
    /// The low price of this tick
    pub l: F,
    /// The number of trades which occured during this tick
    pub n: F,
}

impl Tick {
    /// The number of fields a tick feeds into a neural network. Time is *not* fed in.
    pub const NN_FIELDS: usize = 7; // (v, vw, o, c, h, l, n)
}

impl<D, F> Tick<D, F>
where
    F: Copy + NumCast,
{
    /// Push a tick's data points to an input vector. Guaranteed to write `NN_FIELDS` data points
    pub fn push_tick(&self, input: &mut Vec<f32>) {
        input.push(NumCast::from(self.o).unwrap_or(0.0));
        input.push(NumCast::from(self.h).unwrap_or(0.0));
        input.push(NumCast::from(self.l).unwrap_or(0.0));
        input.push(NumCast::from(self.c).unwrap_or(0.0));
        input.push(NumCast::from(self.v).unwrap_or(0.0));
        input.push(NumCast::from(self.vw).unwrap_or(0.0));
        input.push(NumCast::from(self.n).unwrap_or(0.0));
    }
    /// Get the prediction corresponding to a tick
    pub fn pred(&self) -> Prediction<F> {
        Prediction {
            c: self.c,
            v: self.v,
        }
    }
}

impl<D, F> Open for Tick<D, F>
where
    F: Copy + Into<f64>,
{
    #[inline]
    fn open(&self) -> f64 {
        self.o.into()
    }
}

impl<D, F> High for Tick<D, F>
where
    F: Copy + Into<f64>,
{
    #[inline]
    fn high(&self) -> f64 {
        self.h.into()
    }
}

impl<D, F> Low for Tick<D, F>
where
    F: Copy + Into<f64>,
{
    #[inline]
    fn low(&self) -> f64 {
        self.l.into()
    }
}

impl<D, F> Close for Tick<D, F>
where
    F: Copy + Into<f64>,
{
    #[inline]
    fn close(&self) -> f64 {
        self.c.into()
    }
}

impl<D, F> Volume for Tick<D, F>
where
    F: Copy + Into<f64>,
{
    #[inline]
    fn volume(&self) -> f64 {
        self.v.into()
    }
}

/// A predicted tick
pub struct Prediction<F = CpuFloat> {
    /// Predicted closing price
    pub c: F,
    /// Predicted volume
    pub v: F,
}

impl<F> Close for Prediction<F>
where
    F: Copy + Into<f64>,
{
    #[inline]
    fn close(&self) -> f64 {
        self.c.into()
    }
}

impl<F> Volume for Prediction<F>
where
    F: Copy + Into<f64>,
{
    #[inline]
    fn volume(&self) -> f64 {
        self.v.into()
    }
}


impl Prediction {
    /// The number of fields a neural network must predict to yield a tick prediction
    pub const NN_FIELDS: usize = 2;
}

impl<F> Prediction<F>
where
    F: Copy + NumCast,
{
    /// Push a predictions's data points to an input vector. Guaranteed to write `NN_FIELDS` data points
    pub fn push_pred(&self, input: &mut Vec<f32>) {
        input.push(NumCast::from(self.c).unwrap_or(0.0));
        input.push(NumCast::from(self.v).unwrap_or(0.0));
    }
}
