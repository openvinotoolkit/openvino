#include <openvino/runtime/core.hpp>
#include <openvino/core/model.hpp>


int main() {
ov::Core core;
auto model = core.read_model("sample.xml");
//! [part0]
for (auto && op : model->get_ops()) {
    op->get_rt_info()["affinity"] = "CPU";
}
//! [part0]

//! [part1]
// This example demonstrates how to perform default affinity initialization and then
// correct affinity manually for some layers
const std::string device = "HETERO:GPU,CPU";

// query_model result contains maping of supported operations to device
auto supported = core.query_model(model, device);

// update default affinities
supported["layerName"] = "CPU";

// set affinities to network
for (auto&& node : model->get_ops()) {
    auto& affinity = supported[node->get_friendly_name()];
    // Store affinity mapping using node runtime information
    node->get_rt_info()["affinity"] = affinity;
}

// load network with affinities set before
auto compiled_model = core.compile_model(model, device);
//! [part1]

//! [part2]
{
    auto compiled_model = core.compile_model(model, "HETERO:GPU,CPU");
}
{
    auto compiled_model = core.compile_model(model, "HETERO", ov::device::priorities("GPU", "CPU"));
}
{
    auto compiled_model = core.compile_model(model, "HETERO", ov::device::priorities("GPU,CPU"));
}
//! [part2]
return 0;
}
