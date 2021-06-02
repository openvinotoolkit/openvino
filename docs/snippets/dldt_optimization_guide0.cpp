#include <ie_core.hpp>

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
