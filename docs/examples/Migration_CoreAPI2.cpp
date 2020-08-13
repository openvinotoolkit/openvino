#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
CNNNetReader network_reader;
network_reader.ReadNetwork(fileNameToString(input_model));
network_reader.ReadWeights(fileNameToString(input_model).substr(0, input_model.size() - 4) + ".bin");
CNNNetwork network = network_reader.getNetwork();
return 0;
}
