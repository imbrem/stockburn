/*!
Test polygon IO
*/
use std::io::{Seek, SeekFrom};
use stockburn::data::{fake::*, polygon::*, *};
use tempfile::tempfile;

#[test]
fn fake_data_roundtrip() {
    const TEST_DATA_LENGTH: usize = 10000;
    let ticks: Vec<Tick> = cubic_fake_ticks().take(TEST_DATA_LENGTH).collect();
    let mut tmp = tempfile().expect("Tempfile creation should not fail!");
    write_ticks(&mut tmp, ticks.iter().copied()).expect("Writing test data should not fail!");
    tmp.seek(SeekFrom::Start(0)).expect("Seek should not fail");
    let read_ticks = read_ticks(&mut tmp).expect("Reading test data should not fail");
    assert_eq!(ticks, read_ticks);
}
