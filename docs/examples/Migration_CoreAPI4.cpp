#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
CNNNetwork network = core.ReadNetwork("model.onnx");

return 0;
}
