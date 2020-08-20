#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
//! [part3]
CNNNetwork network = core.ReadNetwork(input_model);
//! [part3]
return 0;
}
