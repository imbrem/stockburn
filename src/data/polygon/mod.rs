/*!
[Polygon](https://polygon.io/)-specific data processing code
*/
use super::Tick;
use anyhow;
use chrono::NaiveDateTime;
use csv;
use std::io::{Read, Write};

/// Read polygon tick data from a Reader
pub fn read_ticks<R: Read>(rdr: R, date_format: Option<&str>) -> Vec<Tick> {
    unimplemented!()
}

/// Deserialize tick data
pub fn deserialize_ticks<R: Read>(rdr: R) -> Result<Vec<Tick>, csv::Error> {
    csv::Reader::from_reader(rdr).into_deserialize().collect()
}

/// Write tick data to a Writer
/// On success, return how many ticks were written
pub fn write_ticks<W, I>(wtr: W, ticks: I) -> Result<usize, csv::Error>
where
    W: Write,
    I: Iterator<Item = Tick>,
{
    let mut wtr = csv::Writer::from_writer(wtr);
    let mut written = 0;
    for tick in ticks {
        wtr.serialize(tick)?;
        written += 1;
    }
    Ok(written)
}
