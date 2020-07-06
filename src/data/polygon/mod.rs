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
    let date_format = if let Some(format) = date_format {
        format
    } else {
        return deserialize_ticks(rdr)
            .filter_map(|result| result.ok())
            .collect();
    };
    unimplemented!()
}

/// Deserialize tick data
pub fn deserialize_ticks<R: Read>(rdr: R) -> impl Iterator<Item = Result<Tick, csv::Error>> {
    csv::Reader::from_reader(rdr).into_deserialize()
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
