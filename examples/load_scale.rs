/*!
Load data and scale it exponentially
*/
use clap::{App, Arg};
use csv;
use io_enum::*;
use std::fs::File;
use std::io::{stdin, Stdin};
use std::path::Path;
use stockburn::data::*;

#[derive(Debug, Read)]
pub enum IoSources {
    Stdin(Stdin),
    File(File),
}

fn main() {
    let matches = App::new("Stock Data Scalar")
        .version("1.0")
        .author("Jad Elkhaleq Ghalayini <jad.ghalayini@mail.utoronto.ca>")
        .about("Loads stock data from a file (or optionally generates it randomly) and scales it using an exponential scaler")
        .arg(
            Arg::with_name("INPUT")
            .help("Sets the input file to use")
            .index(1)
        )
        .get_matches();
    let reader = if let Some(path) = matches.value_of("INPUT") {
        IoSources::File(File::open(Path::new(path)).expect("Failed to open input file"))
    } else {
        IoSources::Stdin(stdin())
    };
    let mut ticks = csv::Reader::from_reader(reader)
        .into_deserialize()
        .map(|tick| tick.expect("Error reading tick"))
        .peekable();
    let first_tick: Tick = if let Some(first) = ticks.peek() {
        *first
    } else {
        return;
    };
    println!("First tick {:?}", first_tick);
    while let Some(_tick) = ticks.next() {
        unimplemented!()
    }
}
