#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
auto execNetwork = plugin.LoadNetwork(network, { });
return 0;
}
