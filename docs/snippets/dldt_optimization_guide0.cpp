#include <inference_engine.hpp>
#include "ie_plugin_config.hpp"
#include "hetero/hetero_plugin_config.hpp"


int main() {
using namespace InferenceEngine;
//! [part0]
using namespace InferenceEngine::PluginConfigParams;
using namespace InferenceEngine::HeteroConfigParams;

Core ie;
auto network = ie.ReadNetwork("sample.xml");
// ...

auto execNetwork = ie.LoadNetwork(network, "HETERO:FPGA,CPU", { {KEY_HETERO_DUMP_GRAPH_DOT, YES} });
//! [part0]

return 0;
}
