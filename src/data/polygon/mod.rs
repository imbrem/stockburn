/*!
[Polygon](https://polygon.io/)-specific data processing code
*/
use super::Tick;
use chrono::NaiveDateTime;
use csv;
use std::io::{Read, Write};
use std::str::FromStr;

/// The polygon DateTime format
pub const POLYGON_DATETIME: &str = "%Y-%m-%d %H:%M:%S";

/// Read polygon tick data from a Reader
pub fn read_ticks<R: Read>(rdr: R, date_format: Option<&str>) -> Vec<Tick> {
    let date_format = if let Some(format) = date_format {
        format
    } else {
        return deserialize_ticks(rdr)
            .filter_map(|result| result.ok())
            .collect();
    };
    csv::Reader::from_reader(rdr)
        .into_records()
        .filter_map(|result| {
            let record = result.ok()?;
            let mut record = record.iter();
            let first = record.next()?;
            let t = NaiveDateTime::parse_from_str(first, date_format).ok()?;
            let mut tick = Tick {
                t,
                v: f64::NAN,
                vw: f64::NAN,
                o: f64::NAN,
                c: f64::NAN,
                h: f64::NAN,
                l: f64::NAN,
                n: f64::NAN,
            };
            for (i, field) in record.enumerate().take(7) {
                match i {
                    0 => tick.v = f64::from_str(field).unwrap_or(f64::NAN),
                    1 => tick.vw = f64::from_str(field).unwrap_or(f64::NAN),
                    2 => tick.o = f64::from_str(field).unwrap_or(f64::NAN),
                    3 => tick.c = f64::from_str(field).unwrap_or(f64::NAN),
                    4 => tick.h = f64::from_str(field).unwrap_or(f64::NAN),
                    5 => tick.l = f64::from_str(field).unwrap_or(f64::NAN),
                    _ => tick.n = f64::from_str(field).unwrap_or(f64::NAN),
                }
            }
            Some(tick)
        })
        .collect()
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
