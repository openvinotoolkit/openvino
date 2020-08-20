#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
//! [part8]
auto execNetwork = plugin.LoadNetwork(network, { });
//! [part8]
return 0;
}
