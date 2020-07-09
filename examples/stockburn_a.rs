/*!
A very simple stock LSTM model
*/

use anyhow::format_err;
use chrono::Duration;
use clap::{App, Arg};
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::File;
use std::path::Path;
use stockburn::data::{
    clocks,
    polygon::{read_ticks, POLYGON_DATETIME},
    scale::TickExpScaler,
    Tick,
};
use stockburn::lstm::StockLSTMDesc;
use tch::nn::{OptimizerConfig, RNN};
use tch::{nn, Device};

const LEARNING_RATE: f64 = 0.01;
const AVERAGE_DECAY_RATE: f64 = 0.999;
const RANGE_DECAY_RATE: f64 = 0.999;
const HIDDEN_SIZE: usize = 256;
const LSTM_LAYERS: usize = 2;
const SEQ_LEN: usize = 180;
const BATCH_SIZE: usize = 256;
const EPOCHS: u64 = 100;
const TRAIN_TEST_RATIO: f64 = 0.95;

pub fn train_test_split(mut ticks: Vec<Vec<Tick>>, ratio: f64) -> (Vec<Vec<Tick>>, Vec<Vec<Tick>>) {
    let samples: usize = ticks.iter().map(|ticks| ticks.len()).max().unwrap_or(0);
    let train_samples: usize = (samples as f64 * ratio) as usize;
    let mut test_samples = Vec::with_capacity(ticks.len());
    for ticks in ticks.iter() {
        if ticks.len() <= train_samples {
            test_samples.push(Vec::new());
            continue;
        }
        test_samples.push(ticks[train_samples..].into());
    }
    for ticks in ticks.iter_mut() {
        ticks.truncate(train_samples)
    }
    (ticks, test_samples)
}

pub fn run_network(verbosity: usize, input_files: &[String], device: Device) -> anyhow::Result<()> {
    // Length check for input files
    let stocks = input_files.len();
    if stocks == 0 {
        return Err(format_err!(
            "StockLSTM needs at least one input file, recieved zero!"
        ));
    }

    // Load input files
    let input_files_style =
        ProgressStyle::default_bar().template("Loading files: {wide_bar} {pos}/{len}: {msg:15}");
    let input_files_progress = ProgressBar::new(stocks as u64);
    let mut ticks: Vec<Vec<Tick>> = Vec::new();

    input_files_progress.set_style(input_files_style);

    for filename in input_files {
        input_files_progress.set_message(filename);
        let file = File::open(Path::new(filename))?;
        let mut file_ticks = read_ticks(file, Some(POLYGON_DATETIME));
        if file_ticks.is_empty() {
            if verbosity >= 1 {
                eprintln!("WARNING: could not read any ticks from file {}", filename);
            }
        } else {
            let first = file_ticks[0];
            let mut scaler = TickExpScaler::with_start(first, AVERAGE_DECAY_RATE, RANGE_DECAY_RATE);
            for tick in file_ticks.iter_mut() {
                *tick = scaler.tick(*tick);
            }
            ticks.push(file_ticks);
        }
        input_files_progress.inc(1);
    }
    input_files_progress.finish_and_clear();

    // Clock function setup
    let clock_periods = &[
        Duration::minutes(5),
        Duration::minutes(10),
        Duration::minutes(30),
        Duration::hours(1),
        Duration::days(1),
        Duration::weeks(1),
        Duration::weeks(4),
        Duration::days(365),
    ];
    let (date_inputs, clock_fn) = clocks::<f32>(clock_periods);

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

    let (training_data, testing_data) = train_test_split(ticks, TRAIN_TEST_RATIO);

    // Get tick counts
    let total_training_ticks: usize = training_data.iter().map(|ticks| ticks.len()).sum();
    let total_testing_ticks: usize = testing_data.iter().map(|ticks| ticks.len()).sum();

    // Loop setup
    let epochs_progress = ProgressBar::new(EPOCHS);

    let data_progress_style =
        ProgressStyle::default_bar().template("[{msg:<15}] {wide_bar} {pos:> 7}/{len:7}");

    let mut training_ticks: Vec<_> = training_data
        .iter()
        .map(|ticks| ticks.iter().copied().peekable())
        .collect();
    let mut testing_ticks: Vec<_> = testing_data
        .iter()
        .map(|ticks| ticks.iter().copied().peekable())
        .collect();

    // Loop over the data
    for epoch in 0..EPOCHS {

        // === INITIALIZATION ===

        epochs_progress.println(format!("Epoch {}", epoch));

        // === TRAINING ===

        // Reset data progress
        let data_progress = ProgressBar::new(total_training_ticks as u64);
        data_progress.set_style(data_progress_style.clone());
        data_progress.set_message("no loss");

        let mut batch = 0;
        let mut sum_loss = 0.0;
        let mut max_loss = -f64::INFINITY;
        let mut min_loss = f64::INFINITY;

        // Pack training data as batches
        while let Some((input_batch, output_batch)) = lstm.make_batches(
            std::iter::repeat(&[][..]),
            clock_fn,
            &mut training_ticks,
            BATCH_SIZE,
            SEQ_LEN,
        ) {
            // Send everything to the GPU
            let input_batch = input_batch.to_device(device);
            let output_batch = output_batch.to_device(device);

            // Feedforward loss
            let (loss, _state) = lstm.loss(
                &input_batch,
                &output_batch,
                &lstm.zero_state(BATCH_SIZE as i64),
            );
            //lstm_state = state;

            // Optimize
            opt.backward_step_clip(&loss, 0.5);

            // Advance progress bar, set message
            batch += 1;
            let loss = f64::from(loss);
            sum_loss += loss;
            max_loss = max_loss.max(loss);
            min_loss = min_loss.min(loss);
            let ticks_left: usize = training_ticks.iter().map(|ticks| ticks.len()).sum();
            data_progress.set_position((total_training_ticks - ticks_left) as u64);
            data_progress.set_message(&format!("loss = {:.5}", loss));
        }

        // Destroy the batch progress bar
        data_progress.finish_and_clear();

        // Print training losses
        epochs_progress.println(format!(
            "Epoch {}: average training loss = {}, max training loss = {}, min training loss = {}",
            epoch,
            sum_loss / batch as f64,
            max_loss,
            min_loss
        ));

        // === TESTING ===

        let mut lstm_state = lstm.zero_state(BATCH_SIZE as i64);
        let data_progress = ProgressBar::new(total_testing_ticks as u64);
        data_progress.set_style(data_progress_style.clone());
        data_progress.set_message("no loss");

        let mut batch = 0;
        let mut sum_loss = 0.0;
        let mut max_loss = -f64::INFINITY;
        let mut min_loss = f64::INFINITY;

        while let Some((input_batch, output_batch)) = lstm.make_batches(
            std::iter::repeat(&[][..]),
            clock_fn,
            &mut testing_ticks,
            BATCH_SIZE,
            SEQ_LEN,
        ) {
            // Send everything to the GPU
            let input_batch = input_batch.to_device(device);
            let output_batch = output_batch.to_device(device);

            // Feedforward loss
            let (loss, state) = lstm.loss(&input_batch, &output_batch, &lstm_state);
            lstm_state = state;

            // Advance progress bar, set message
            batch += 1;
            let loss = f64::from(loss);
            sum_loss += loss;
            max_loss = max_loss.max(loss);
            min_loss = min_loss.min(loss);
            let ticks_left: usize = testing_ticks.iter().map(|ticks| ticks.len()).sum();
            data_progress.set_position((total_testing_ticks - ticks_left) as u64);
            data_progress.set_message(&format!("loss = {:.5}", loss));
        }

        // Destroy the batch progress bar
        data_progress.finish_and_clear();

        // Print testing losses
        epochs_progress.println(format!(
            "Epoch {}: average testing loss = {}, max testing loss = {}, min testing loss = {}\n",
            epoch,
            sum_loss / batch as f64,
            max_loss,
            min_loss
        ));

        // === CLEANUP ===

        // Tick forward the epoch counter
        epochs_progress.inc(1);

        // Reset iterators
        if epoch == EPOCHS - 1 {
            // Skip reset for last epoch
            break;
        }
        for (i, ticks) in training_data.iter().enumerate() {
            training_ticks[i] = ticks.iter().copied().peekable();
        }
        for (i, ticks) in testing_data.iter().enumerate() {
            testing_ticks[i] = ticks.iter().copied().peekable();
        }
    }

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
