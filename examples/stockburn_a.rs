/*!
A very simple stock LSTM model
*/

use anyhow::format_err;
use chrono::Duration;
use clap::{App, Arg};
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::File;
use std::path::Path;
use stockburn::data::{clocks, polygon::read_ticks, Tick};
use stockburn::lstm::StockLSTMDesc;
use tch::nn::{OptimizerConfig, RNN};
use tch::{nn, Device};

const LEARNING_RATE: f64 = 0.01;
const HIDDEN_SIZE: usize = 256;
const LSTM_LAYERS: usize = 2;
const SEQ_LEN: usize = 180;
const BATCH_SIZE: usize = 256;
const EPOCHS: u64 = 100;

pub fn run_network(verbosity: usize, input_files: &[String], device: Device) -> anyhow::Result<()> {
    // Length check for input files
    let stocks = input_files.len();
    if stocks == 0 {
        return Err(format_err!(
            "StockLSTM needs at least one input file, recieved zero!"
        ));
    }

    // Load input files
    let input_files_style = ProgressStyle::default_bar()
        .template("Loading files: {bar:60} {pos:> 7}/{len:7}: {wide_msg}");
    let input_files_progress = ProgressBar::new(stocks as u64);
    let mut ticks: Vec<Vec<Tick>> = Vec::new();
    input_files_progress.set_style(input_files_style);
    for file in input_files {
        input_files_progress.set_message(file);
        let file = File::open(Path::new(file))?;
        ticks.push(read_ticks(file, None));
        input_files_progress.inc(1);
    }

    // Clock function setup
    let (date_inputs, clock_fn) = clocks::<f32>(&[
        Duration::minutes(5),
        Duration::minutes(10),
        Duration::minutes(30),
        Duration::hours(1),
        Duration::days(1),
        Duration::weeks(1),
        Duration::weeks(4),
        Duration::days(365),
    ]);

    // Network setup
    if verbosity >= 2 {
        eprintln!("Setting up network");
    }
    let vs = nn::VarStore::new(device);
    let lstm_desc = StockLSTMDesc {
        additional_inputs: 0,
        stocks,
        date_inputs,
        hidden: HIDDEN_SIZE,
        layers: LSTM_LAYERS,
    };
    let lstm = lstm_desc.build(&vs);

    if verbosity >= 2 {
        eprintln!("Initializing optimizer");
    }
    let mut opt = nn::Adam::default()
        .build(&vs, LEARNING_RATE)
        .map_err(|err| format_err!("Error building optimizer: {:#?}", err))?;

    if verbosity >= 1 {
        eprintln!("Beginning training");
    }

    let epochs_progress = ProgressBar::new(EPOCHS);

    for epoch in 0..EPOCHS {}

    Ok(())
}

pub fn main() -> anyhow::Result<()> {
    // Initialization, argument parsing
    let matches = App::new("Stockburn Alpha")
        .version("1.0")
        .author("Jad Elkhaleq Ghalayini <jad.ghalayini@mail.utoronto.ca>")
        .about("An LSTM which attempts to predict the price changes of stocks")
        .arg(
            Arg::with_name("STOCKS")
                .help("Input stock data in Polygon format")
                .required(true)
                .multiple(true),
        )
        .arg(
            Arg::with_name("device")
                .short("d")
                .long("device")
                .help("Device to use: cuda, cpu. Defaults to cuda")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("verbose")
                .short("v")
                .long("verbose")
                .help("Sets the level of verbosity")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("fake")
                .short("f")
                .long("fake")
                .help("A fake input stock")
        )
        .get_matches();

    let input_files = matches.values_of_lossy("STOCKS").expect("Required");
    let verbosity = matches
        .value_of("verbose")
        .map(|v| usize::from_str_radix(v, 10))
        .unwrap_or(Ok(0))?;

    let device: Device = match matches.value_of("device").unwrap_or("cuda") {
        "cuda" => Device::cuda_if_available(),
        "cpu" => Device::Cpu,
        device => Err(format_err!("Invalid value for device: {:?}", device))?,
    };
    if verbosity >= 1 {
        eprintln!("Device: {:?}", device);
    }

    run_network(verbosity, &input_files, device)
}
