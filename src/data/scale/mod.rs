/*!
Input data scaling
*/
use super::Tick;
use crate::{CpuFloat, util::to_s};
use chrono::{Duration, DateTime, Utc};
use num::Float;

/// A window for exponential scaling
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ExpScaler<F = CpuFloat> {
    /// The exponential moving average of the input data
    pub average: F,
    /// The exponential moving average's decay parameter
    pub average_decay: F,
    /// The range of the input data
    pub range: F,
    /// The range decay parameter
    pub range_decay: F,
}

/// Clip a value within an absolute value range
pub fn clip<F: Copy + Float>(value: F, range: F) -> F {
    value.min(-range).max(range)
}

impl<F> ExpScaler<F>
where
    F: Copy + Float,
{
    /// Scale a value according to the current window
    #[inline]
    pub fn scale(&self, val: F) -> F {
        let clip_range = self.range + self.range + self.range;
        clip(val - self.average, clip_range) / self.range
    }
    /// Update a window given a value and a time difference
    #[inline]
    pub fn update(&mut self, val: F, dt: Duration) {
        // Caclulate dt in seconds
        let dt_s: F = to_s(dt);
        // Update range
        let diff = (val - self.average).abs();
        self.range = self.range.max(diff) * self.range_decay; //TODO: think about this...
        let old_proportion = self.average_decay.powf(dt_s);
        let new_proportion = F::one() - old_proportion;
        self.average = new_proportion * val + old_proportion * self.average;
    }
}

/// An exponential scaler for stock market ticks
pub struct TickExpScaler<F> {
    /// The current time
    pub t: DateTime<Utc>,
    /// The opening price scaler
    pub o: ExpScaler<F>,
    /// The high price scaler
    pub h: ExpScaler<F>,
    /// The low price scaler
    pub l: ExpScaler<F>,
    /// The closing price scaler
    pub c: ExpScaler<F>,
    /// The volume price scaler
    pub v: ExpScaler<F>,
    /// The VWAP scaler
    pub vw: ExpScaler<F>,
    /// The scaler for the number of trades
    pub n: ExpScaler<F>,
}

impl<F> TickExpScaler<F> {
    /// Create a new tick scaler from a base scaler
    pub fn new(t: DateTime<Utc>, base: ExpScaler<F>) -> TickExpScaler<F>
    where
        ExpScaler<F>: Clone,
    {
        TickExpScaler {
            t,
            o: base.clone(),
            h: base.clone(),
            l: base.clone(),
            c: base.clone(),
            v: base.clone(),
            vw: base.clone(),
            n: base.clone(),
        }
    }
}

impl<F: Float + Copy> TickExpScaler<F> {
    /// Scale a tick of data
    pub fn scale(&self, tick: Tick<DateTime<Utc>, F>) -> Tick<DateTime<Utc>, F> {
        Tick {
            t: tick.t,
            o: self.o.scale(tick.o),
            c: self.c.scale(tick.c),
            h: self.h.scale(tick.h),
            l: self.l.scale(tick.l),
            v: self.v.scale(tick.v),
            vw: self.vw.scale(tick.vw),
            n: self.n.scale(tick.n),
        }
    }
}
