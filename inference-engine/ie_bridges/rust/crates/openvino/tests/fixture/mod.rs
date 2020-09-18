use std::path::PathBuf;

/// This structure encodes the paths necessary for running the classification example.
pub struct Fixture;
#[allow(dead_code)]
impl Fixture {
    fn dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixture")
            .canonicalize()
            .unwrap()
    }
    pub fn graph() -> PathBuf {
        Fixture::dir().join("frozen_inference_graph.xml")
    }
    pub fn weights() -> PathBuf {
        Fixture::dir().join("frozen_inference_graph.bin")
    }
    pub fn tensor() -> PathBuf {
        Fixture::dir().join("tensor-1x3x640x481-u8.bgr")
    }
}
