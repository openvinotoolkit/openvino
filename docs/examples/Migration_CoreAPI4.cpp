#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
//! [part4]
CNNNetwork network = core.ReadNetwork("model.onnx");
//! [part4]
return 0;
}
