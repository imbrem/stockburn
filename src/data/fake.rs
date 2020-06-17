/*!
Generate fake tick data, for testing purposes
*/
use super::Tick;
use chrono::{DateTime, Duration, Utc};
use rand::{distributions::Distribution, Rng};
use std::iter::Peekable;

/// Generate tick data using a price generator, volume generator and a time generator
#[derive(Debug, Clone)]
pub struct TickGen<D, P, V>
where
    D: Iterator<Item = DateTime<Utc>>,
    P: TimedGen<Item = f64>,
    V: TimedGen<Item = Volume>,
{
    /// The time generator in use
    pub time_gen: Peekable<D>,
    /// The price generator in use
    pub price_gen: P,
    /// The volume generator in use
    pub volume_gen: V,
}

/// A volume, number-of-trades pair
pub struct Volume {
    /// The volume of a tick
    pub v: f64,
    /// The number of trades of a tick
    pub n: f64,
}

impl<D, P, V> Iterator for TickGen<D, P, V>
where
    D: Iterator<Item = DateTime<Utc>>,
    P: TimedGen<Item = f64>,
    V: TimedGen<Item = Volume>,
{
    type Item = Tick<DateTime<Utc>, f64>;
    fn next(&mut self) -> Option<Tick<DateTime<Utc>, f64>> {
        use std::cmp::Ordering::*;
        let old_time = self.time_gen.next()?;
        let t = *self.time_gen.peek()?;
        let dt = t - old_time;
        let prices = [
            self.price_gen.next_after(dt / 4)?,
            self.price_gen.next_after(dt / 4)?,
            self.price_gen.next_after(dt / 4)?,
            self.price_gen.next_after(dt / 4)?,
        ];
        let o = prices[0];
        let h = *prices
            .iter()
            .max_by(|l, r| l.partial_cmp(r).unwrap_or(Less))
            .expect("Nonempty");
        let l = *prices
            .iter()
            .min_by(|l, r| l.partial_cmp(r).unwrap_or(Greater))
            .expect("Nonempty");
        let c = prices[3];
        let volumes = [
            self.volume_gen.next_after(dt / 4)?,
            self.volume_gen.next_after(dt / 4)?,
            self.volume_gen.next_after(dt / 4)?,
            self.volume_gen.next_after(dt / 4)?,
        ];
        let v = volumes.iter().map(|v| v.v).sum();
        let mut vw: f64 = volumes
            .iter()
            .zip(prices.iter())
            .map(|(v, p)| v.v * p)
            .sum();
        vw /= v;
        let n = volumes.iter().map(|v| v.n).sum();
        Some(Tick {
            t,
            o,
            h,
            l,
            c,
            v,
            vw,
            n,
        })
    }
}

/// A trait implemented by timed generators
pub trait TimedGen {
    /// The this object generates
    type Item;
    /// Generate a value, jumping forward a given duration
    fn next_after(&mut self, after: Duration) -> Option<Self::Item>;
}

/// Generate numbers using a time-weighted second-order random walk
#[derive(Debug, Copy, Clone)]
pub struct DistGen2<R, P, J> {
    /// The RNG used by this random walk
    pub rng: R,
    /// The current price
    pub price: f64,
    /// The price jitter distribution
    pub jitter: P,
    /// The current price velocity
    pub vel: f64,
    /// The current price acceleration
    pub acc: f64,
    /// The jerk distribution
    pub jerk: J,
}

impl<R, P, J> TimedGen for DistGen2<R, P, J>
where
    R: Rng,
    P: Distribution<f64>,
    J: Distribution<f64>,
{
    type Item = f64;
    fn next_after(&mut self, after: Duration) -> Option<f64> {
        let after = after
            .to_std()
            .expect("Duration out of bounds!")
            .as_secs_f64();
        self.acc += after * self.jerk.sample(&mut self.rng);
        self.vel += after * self.acc;
        self.price += after * self.vel + self.jitter.sample(&mut self.rng);
        Some(self.price)
    }
}

/// Generate random volumes by generating random numbers of trades (given a number of seconds),
/// and then generating random average trade sizes
pub struct VolumeGen<R, N, A> {
    /// The RNG used to generate these values
    pub rng: R,
    /// The distribution for the average trade size
    pub average: A,
    /// The distribution for the average number of trades / second
    pub no_trades: N,
}

impl<R, N, A> TimedGen for VolumeGen<R, N, A>
where
    R: Rng,
    N: Distribution<f64>,
    A: Distribution<f64>,
{
    type Item = Volume;
    fn next_after(&mut self, after: Duration) -> Option<Volume> {
        let after = after
            .to_std()
            .expect("Duration out of bounds!")
            .as_secs_f64();
        let no_trades = self.no_trades.sample(&mut self.rng) * after;
        let volume = no_trades * self.average.sample(&mut self.rng);
        let n = no_trades.round();
        let v = if n == 0.0 { 0.0 } else { volume };
        Some(Volume { v, n })
    }
}