/*!
Data processing and IO functions
*/
use crate::*;
use chrono::{DateTime, Duration, NaiveDate, NaiveDateTime, Utc};
use num::{Float, NumCast};
use serde::{Deserialize, Serialize};
use ta::{Close, High, Low, Open, Volume};
use util::to_ns;

pub mod fake;
pub mod polygon;
pub mod scale;

/// Tick data for a stock
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Tick<F = CpuFloat> {
    /// This tick's timestamp in UTC
    pub t: NaiveDateTime,
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

impl<F> Tick<F>
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

/// Push clock, with a period measured in seconds / 2 pi
pub fn push_clock_period<F>(period: F, time: DateTime<Utc>, dest: &mut Vec<F>)
where
    F: Float + Copy,
{
    let naive = time.naive_utc();
    let naive_0 = NaiveDate::from_ymd(2020, 1, 1).and_hms(1, 1, 1);
    let dt = naive - naive_0;
    let dt: F = to_ns(dt);
    let scaled_dt = dt / period;
    let (sin_dt, cos_dt) = scaled_dt.sin_cos();
    dest.push(sin_dt);
    dest.push(cos_dt);
}

/// Push a set of clocks, with duration periods
pub fn clocks<'a, F>(
    durations: &'a [Duration],
) -> (
    usize,
    impl FnMut(DateTime<Utc>, &mut Vec<F>) + Send + Sync + Copy + 'a,
)
where
    F: Float + Copy + NumCast,
{
    (
        durations.len() * 2, move |time, dest| {
        for duration in durations.iter() {
            let tau: F = NumCast::from(2.0 * std::f64::consts::PI).expect("Pi fits in F");
            let duration_ns: F = to_ns(*duration);
            let period = duration_ns / tau;
            push_clock_period(period, time, dest)
        }
    }
)
}

impl<F> Open for Tick<F>
where
    F: Copy + Into<f64>,
{
    #[inline]
    fn open(&self) -> f64 {
        self.o.into()
    }
}

impl<F> High for Tick<F>
where
    F: Copy + Into<f64>,
{
    #[inline]
    fn high(&self) -> f64 {
        self.h.into()
    }
}

impl<F> Low for Tick<F>
where
    F: Copy + Into<f64>,
{
    #[inline]
    fn low(&self) -> f64 {
        self.l.into()
    }
}

impl<F> Close for Tick<F>
where
    F: Copy + Into<f64>,
{
    #[inline]
    fn close(&self) -> f64 {
        self.c.into()
    }
}

impl<F> Volume for Tick<F>
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
