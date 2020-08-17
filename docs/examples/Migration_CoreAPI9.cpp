#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
auto execNetwork = core.LoadNetwork(network, deviceName, { });

return 0;
}
