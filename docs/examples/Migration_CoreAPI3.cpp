#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
CNNNetwork network = core.ReadNetwork(input_model);

return 0;
}
