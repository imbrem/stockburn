/*!
Generate and sample some fake tick data
*/
use chrono::{naive::NaiveDate, Date, Utc};
use rand::thread_rng;
use rand_distr::Normal;
use rustyline::error::ReadlineError;
use rustyline::Editor;
use stockburn::data::fake::*;

fn main() {
    let price_gen = DistGen2 {
        rng: thread_rng(),
        price: 40.0,
        jitter: Normal::new(0.0, 0.1).unwrap(),
        vel: 1e-5,
        acc: 1e-10,
        jerk: Normal::new(0.0, 1e-7).unwrap(),
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
    };
    let mut rl = Editor::<()>::new();
    let points = loop {
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
    };
    for _ in 0..points {
        println!("{:?}", tick_gen.next().unwrap());
        println!("State: {:#?}", tick_gen);
    }
}
