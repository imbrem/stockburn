/*!
[Polygon](https://polygon.io/)-specific data processing code
*/
use crate::*;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Polygon tick data for a stock
#[derive(Debug, Serialize, Deserialize)]
pub struct PolygonTick<D = DateTime<Utc>, F = CpuFloat> {
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
