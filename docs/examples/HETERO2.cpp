#include <inference_engine.hpp>
#include "ie_plugin_config.hpp"
#include "hetero/hetero_plugin_config.hpp"

int main() {
using namespace InferenceEngine;
InferenceEngine::Core core
auto network = core.ReadNetwork("Model.xml");
auto executable_network = core.LoadNetwork(network, "HETERO:FPGA,CPU");
return 0;
}
