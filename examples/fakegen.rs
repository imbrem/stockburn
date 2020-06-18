/*!
Generate and sample some fake tick data
*/
use chrono::{naive::NaiveDate, Date, Utc};
use clap::{App, Arg};
use rand::thread_rng;
use rand_distr::Normal;
use rustyline::error::ReadlineError;
use rustyline::Editor;
use stockburn::data::fake::*;

fn main() {
    let matches = App::new("Fake Tick Data Generator")
        .version("1.0")
        .author("Jad Elkhaleq Ghalayini <jad.ghalayini@mail.utoronto.ca>")
        .about("Generates fake tick data using a second order time-weighted random walk")
        .arg(
            Arg::with_name("no-ticks")
                .short("n")
                .help(
                    "The number of ticks to generate. If not provided, the user will be prompted.",
                )
                .takes_value(true),
        )
        .arg(Arg::with_name("no-header").help("Do not output a CSV header"))
        .get_matches();

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
    let mut tick_gen = TickGen {
        price_gen,
        volume_gen,
        time_gen,
        close: 0.0,
    };
    let mut rl = Editor::<()>::new();
    let n = if let Some(n) = matches.value_of("no-ticks") {
        usize::from_str_radix(&n, 10).expect("Invalid number of ticks!")
    } else {
        loop {
            match rl.readline("Points to generate: ") {
                Ok(line) => match usize::from_str_radix(&line, 10) {
                    Ok(points) => break points,
                    Err(_) => eprintln!("Invalid input: {:?}", line),
                },
                Err(ReadlineError::Interrupted) => {
                    eprintln!("CTRL-C");
                    return;
                }
                Err(ReadlineError::Eof) => {
                    eprintln!("CTRL-D");
                    return;
                }
                Err(err) => eprintln!("Error: {:?}", err),
            }
        }
    };
    if !matches.is_present("no-header") {
        println!("t,v,vw,o,c,h,l,n")
    }
    for _ in 0..n {
        let tick = tick_gen.next().unwrap();
        println!(
            "{},{},{},{},{},{},{},{}",
            tick.t, tick.v, tick.vw, tick.o, tick.c, tick.h, tick.l, tick.n
        );
    }
}
