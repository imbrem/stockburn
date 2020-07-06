/*!
Generate fake tick data, for testing purposes
*/
use super::Tick;
use chrono::{
    naive::{NaiveDate, NaiveDateTime},
    Date, DateTime, Datelike, Duration, TimeZone, Timelike, Utc, Weekday,
};
use rand::{distributions::Distribution, Rng, thread_rng};
use rand_distr::Normal;
use std::iter::Peekable;

/// Generate decent looking fake tick data using a provided RNG
pub fn cubic_fake_ticks() -> impl Iterator<Item = Tick> {
    let price_gen = DistGen2 {
        rng: thread_rng(),
        price: 40.0,
        jitter: Normal::new(0.0, 0.1).unwrap(),
        vel: 1e-7,
        acc: 1e-15,
        jerk: Normal::new(0.0, 1e-19).unwrap(),
    };
    let volume_gen = VolumeGen {
        rng: thread_rng(),
        average: Normal::new(200.0, 100.0).unwrap(),
        no_trades: Normal::new(0.03, 0.05).unwrap(),
    };
    let time_gen = NASDAQDays(Date::from_utc(NaiveDate::from_ymd(2020, 10, 10), Utc))
        .map(|date| NASDAQMinutes::for_date(date))
        .flatten()
        .peekable();
    TickGen {
        price_gen,
        volume_gen,
        time_gen,
        close: 0.0,
    }
}

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
    /// The previous closing price
    pub close: f64,
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
    type Item = Tick<f64>;
    fn next(&mut self) -> Option<Tick<f64>> {
        use std::cmp::Ordering::*;
        let old_time = self.time_gen.next()?;
        let t = *self.time_gen.peek()?;
        let dt = t - old_time;
        let volumes = [
            self.volume_gen.next_after(dt / 4)?,
            self.volume_gen.next_after(dt / 4)?,
            self.volume_gen.next_after(dt / 4)?,
            self.volume_gen.next_after(dt / 4)?,
        ];
        let v = volumes.iter().map(|v| v.v).sum();
        if v == 0.0 {
            // Zero volume special case
            return Some(Tick {
                t: t.naive_utc(),
                v,
                vw: self.close,
                o: self.close,
                h: self.close,
                l: self.close,
                c: self.close,
                n: 0.0,
            });
        }
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
        self.close = c;
        let mut vw: f64 = volumes
            .iter()
            .zip(prices.iter())
            .map(|(v, p)| v.v * p)
            .sum();
        vw /= v;
        let n = volumes.iter().map(|v| v.n).sum();
        Some(Tick {
            t: t.naive_utc(),
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
#[derive(Debug, Copy, Clone)]
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
        let no_trades = self.no_trades.sample(&mut self.rng).max(0.0) * after;
        let volume = no_trades * self.average.sample(&mut self.rng).max(0.0);
        let n = no_trades.round();
        let v = if n == 0.0 { 0.0 } else { volume };
        Some(Volume { v, n })
    }
}

/// Generate NASDAQ trading days starting at a given date
#[derive(Debug, Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub struct NASDAQDays(pub Date<Utc>);

/// Check if a naive UTC date is a NASDAQ trading day
pub fn naive_utc_is_nasdaq_trading_day(date: NaiveDate) -> bool {
    match date.weekday() {
        Weekday::Sat => false,
        Weekday::Sun => false,
        _ => true,
    }
}

/// Check if a date is a NASDAQ trading day
#[inline]
pub fn is_nasdaq_trading_day<Tz: TimeZone>(date: Date<Tz>) -> bool {
    naive_utc_is_nasdaq_trading_day(date.naive_utc())
}

/// Check if a naive UTC datetime is within NASDAQ trading hours
#[inline]
pub fn naitve_utc_is_nasdaq_trading_time(datetime: NaiveDateTime) -> bool {
    if !naive_utc_is_nasdaq_trading_day(datetime.date()) {
        return false;
    }
    match datetime.hour() {
        14 => datetime.minute() >= 30,
        15..=20 => true,
        21 => datetime.minute() == 0,
        _ => false,
    }
}

/// Check if a time is within NASDAQ trading hours
pub fn is_nasdaq_trading_time<Tz: TimeZone>(datetime: DateTime<Tz>) -> bool {
    naitve_utc_is_nasdaq_trading_time(datetime.naive_utc())
}

impl Iterator for NASDAQDays {
    type Item = Date<Utc>;
    fn next(&mut self) -> Option<Date<Utc>> {
        while !is_nasdaq_trading_day(self.0) {
            self.0 = self.0.succ()
        }
        let result = self.0;
        self.0 = self.0.succ();
        Some(result)
    }
}

/// Generate NASDAQ trading minutes on a given date, starting at a given time
#[derive(Debug, Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub struct NASDAQMinutes(pub DateTime<Utc>);

impl NASDAQMinutes {
    /// Create a new `NASDAQMinutes` iterator for a given date
    pub fn for_date(date: Date<Utc>) -> NASDAQMinutes {
        NASDAQMinutes(date.and_hms(14, 30, 00))
    }
}

impl Iterator for NASDAQMinutes {
    type Item = DateTime<Utc>;
    fn next(&mut self) -> Option<DateTime<Utc>> {
        if !is_nasdaq_trading_time(self.0) {
            return None;
        }
        let result = self.0;
        self.0 = self.0 + Duration::minutes(1);
        Some(result)
    }
}
