/*!
Data processing and IO functions
*/
use crate::*;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use ta::{Close, High, Low, Open, Volume};

pub mod fake;
pub mod polygon;

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

impl<D, F> Open for Tick<D, F> where F: Copy + Into<f64> {
    #[inline]
    fn open(&self) -> f64 {
        self.o.into()
    }
}

impl<D, F> High for Tick<D, F> where F: Copy + Into<f64> {
    #[inline]
    fn high(&self) -> f64 {
        self.h.into()
    }
}

impl<D, F> Low for Tick<D, F> where F: Copy + Into<f64> {
    #[inline]
    fn low(&self) -> f64 {
        self.l.into()
    }
}

impl<D, F> Close for Tick<D, F> where F: Copy + Into<f64> {
    #[inline]
    fn close(&self) -> f64 {
        self.c.into()
    }
}

impl<D, F> Volume for Tick<D, F> where F: Copy + Into<f64> {
    #[inline]
    fn volume(&self) -> f64 {
        self.v.into()
    }
}