//! These tests demonstrate how to setup OpenVINO networks.
mod fixture;

use fixture::Fixture;
use openvino::Core;
use std::fs;

#[test]
fn read_network() {
    let mut core = Core::new(None).unwrap();
    let model = fs::read(Fixture::graph()).unwrap();
    let weights = fs::read(Fixture::weights()).unwrap();
    core.read_network_from_buffer(&model, &weights).unwrap();
}
