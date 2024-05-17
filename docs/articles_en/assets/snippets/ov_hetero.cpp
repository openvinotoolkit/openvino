#include <openvino/runtime/core.hpp>

int main() {
ov::Core core;
auto model = core.read_model("sample.xml");
//! [set_manual_affinities]
for (auto && op : model->get_ops()) {
    op->get_rt_info()["affinity"] = "CPU";
}
//! [set_manual_affinities]

//! [fix_automatic_affinities]
// This example demonstrates how to perform default affinity initialization and then
// correct affinity manually for some layers
const std::string device = "HETERO:GPU,CPU";

// query_model result contains mapping of supported operations to devices
auto supported_ops = core.query_model(model, device);

// update default affinities manually for specific operations
supported_ops["operation_name"] = "CPU";

// set affinities to a model
for (auto&& node : model->get_ops()) {
    auto& affinity = supported_ops[node->get_friendly_name()];
    // Store affinity mapping using op runtime information
    node->get_rt_info()["affinity"] = affinity;
}

// load model with manually set affinities
auto compiled_model = core.compile_model(model, device);
//! [fix_automatic_affinities]

{
//! [compile_model]
auto compiled_model = core.compile_model(model, "HETERO:GPU,CPU");
// or with ov::device::priorities with multiple args
compiled_model = core.compile_model(model, "HETERO", ov::device::priorities("GPU", "CPU"));
// or with ov::device::priorities with a single argument
compiled_model = core.compile_model(model, "HETERO", ov::device::priorities("GPU,CPU"));
//! [compile_model]
}

{
//! [configure_fallback_devices]
auto compiled_model = core.compile_model(model, "HETERO",
    // GPU with fallback to CPU
    ov::device::priorities("GPU", "CPU"),
    // profiling is enabled only for GPU
    ov::device::properties("GPU", ov::enable_profiling(true)),
    // FP32 inference precision only for CPU
    ov::device::properties("CPU", ov::hint::inference_precision(ov::element::f32))
);
//! [configure_fallback_devices]
}

{
//! [set_pipeline_parallelism]
std::set<ov::hint::ModelDistributionPolicy> model_policy = {ov::hint::ModelDistributionPolicy::PIPELINE_PARALLEL};
auto compiled_model =
    core.compile_model(model, "HETERO:GPU.1,GPU.2", ov::hint::model_distribution_policy(model_policy));
//! [set_pipeline_parallelism]
}
return 0;
}
