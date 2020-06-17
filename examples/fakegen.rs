/*!
Generate and sample some fake tick data
*/
use rand::thread_rng;
use rand_distr::Normal;
use stockburn::data::fake::*;

fn main() {
    let _price_gen = DistGen2 {
        rng: thread_rng(),
        price: 40.0,
        jitter: Normal::new(0.0, 0.1),
        vel: 0.1,
        acc: 0.01,
        jerk: Normal::new(0.0001, 0.001),
    };
    unimplemented!()
}
