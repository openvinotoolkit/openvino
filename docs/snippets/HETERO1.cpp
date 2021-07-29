#include <ie_core.hpp>
#include <ngraph/function.hpp>
#include <ngraph/variant.hpp>

int main() {
using namespace InferenceEngine;
using namespace ngraph;
//! [part1]
InferenceEngine::Core core;
auto network = core.ReadNetwork("sample.xml");
auto function = network.getFunction();

// This example demonstrates how to perform default affinity initialization and then
// correct affinity manually for some layers
const std::string device = "HETERO:FPGA,CPU";

// QueryNetworkResult object contains map layer -> device
InferenceEngine::QueryNetworkResult res = core.QueryNetwork(network, device, { });

// update default affinities
res.supportedLayersMap["layerName"] = "CPU";

// set affinities to network
for (auto&& node : function->get_ops()) {
    auto& affinity = res.supportedLayersMap[node->get_friendly_name()];
    // Store affinity mapping using node runtime information
    node->get_rt_info()["affinity"] = std::make_shared<ngraph::VariantWrapper<std::string>>(affinity);
}

// load network with affinities set before
auto executable_network = core.LoadNetwork(network, device);
//! [part1]
return 0;
}
