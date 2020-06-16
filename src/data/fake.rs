/*!
Generate fake tick data, for testing purposes
*/
use chrono::{DateTime, Duration, Utc};
use rand::Rng;
use std::iter::Peekable;

/// Generate tick data using a price generator and a time generator
#[derive(Debug, Clone)]
pub struct TickGen<D: Iterator<Item = DateTime<Utc>>, P: PriceGen> {
    /// The time generator in use
    pub time_generator: Peekable<D>,
    /// The price generator in use
    pub price_generator: P,
}

/// A trait implemented by price generators
pub trait PriceGen {
    /// Generate a price, jumping forward a given number of seconds
    fn price_after(&mut self, after: Duration) -> f64;
}

/// Generate fake prices using a time-weighted random walk
#[derive(Debug, Copy, Clone)]
pub struct PriceRandomWalk<R> {
    /// The RNG used by this random walk
    pub rng: R,
}

impl<R: Rng> PriceGen for PriceRandomWalk<R> {
    fn price_after(&mut self, _after: Duration) -> f64 {
        unimplemented!()
    }
}
