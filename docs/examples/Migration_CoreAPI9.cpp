#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
//! [part9]
auto execNetwork = core.LoadNetwork(network, deviceName, { });
//! [part9]
return 0;
}
