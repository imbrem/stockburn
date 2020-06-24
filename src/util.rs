/*!
Miscellaneous utilities for `stockburn`
*/

use chrono::Duration;
use num::{Float, NumCast};

/// Convert a `chrono::Duration` to a floating point containing the number of nanoseconds
pub fn to_ns<F: Float>(dur: Duration) -> F {
    NumCast::from(dur.num_nanoseconds().unwrap_or(i64::MAX)).expect("Floating type F overflowed")
}

/// Convert a `chrono::Duration` to a floating point containing the number of seconds
pub fn to_s<F: Float>(dur: Duration) -> F {
    let ns_in_sec: F = NumCast::from(1_000_000_000).expect("F cannot hold nanoseconds in second (10^9)");
    let dur_ns: F = to_ns(dur);
    dur_ns / ns_in_sec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_s() {
        assert_eq!(to_s::<f64>(Duration::minutes(1)), 60.0);
        assert_eq!(to_s::<f32>(Duration::days(1)), 60.0 * 60.0 * 24.0);
    }
}