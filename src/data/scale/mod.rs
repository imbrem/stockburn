/*!
Input data scaling
*/
use super::Tick;
use crate::{util::to_s, CpuFloat};
use chrono::{DateTime, Duration, NaiveDateTime, Utc};
use num::Float;
use std::fmt::Debug;

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
    value.max(-range).min(range)
}

impl<F> ExpScaler<F>
where
    F: Copy + Float,
{
    /// Create a new exponential scaler with a given starting value
    #[inline]
    pub fn start(start: F, average_decay: F, range_decay: F) -> ExpScaler<F> {
        ExpScaler {
            average: start,
            range: F::zero(),
            average_decay,
            range_decay,
        }
    }
    /// Scale a value according to the current window
    #[inline]
    pub fn scale(&self, val: F) -> F {
        // Return 0 for NaN and Inf
        if val.is_finite() {
            return F::zero()
        }
        if self.range == F::zero() {
            return F::zero();
        }
        let clip_range = self.range + self.range + self.range;
        let diff = val - self.average;
        let clipped = clip(diff, clip_range);
        clipped / self.range
    }
    /// Update a window given a value and a time difference
    #[inline]
    pub fn update(&mut self, val: F, dt: Duration) {
        // Ignore NaN and Inf
        if !val.is_finite() {
            return
        }
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
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct TickExpScaler<F> {
    /// The current time in Utc
    pub t: NaiveDateTime,
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
            t: t.naive_utc(),
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
    #[inline]
    pub fn scale(&self, tick: Tick<F>) -> Tick<F> {
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
    /// Update the scaler with a new tick of data
    #[inline]
    pub fn update(&mut self, tick: Tick<F>) {
        self.update_dt(tick, tick.t - self.t)
    }
    /// Update the scaler with a new tick of data, artificially fixing dt
    #[inline]
    pub fn update_dt(&mut self, tick: Tick<F>, dt: Duration) {
        self.t = tick.t;
        self.o.update(tick.o, dt);
        self.c.update(tick.c, dt);
        self.h.update(tick.h, dt);
        self.l.update(tick.l, dt);
        self.v.update(tick.v, dt);
        self.vw.update(tick.vw, dt);
        self.n.update(tick.n, dt);
    }
    /// Feed a tick into the scaler, and return the scaled tick
    #[inline]
    pub fn tick(&mut self, tick: Tick<F>) -> Tick<F> {
        let scaled_tick = self.scale(tick);
        self.update(tick);
        scaled_tick
    }
}
