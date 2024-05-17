#include <openvino/runtime/core.hpp>

int main() {
//! [get_available_devices]
ov::Core core;
std::vector<std::string> available_devices = core.get_available_devices();
//! [get_available_devices]

//! [hetero_priorities]
auto device_priorites = core.get_property("HETERO", ov::device::priorities);
//! [hetero_priorities]

//! [cpu_device_name]
auto cpu_device_name = core.get_property("CPU", ov::device::full_name);
//! [cpu_device_name]

auto model = core.read_model("sample.xml");
{
//! [compile_model_with_property]
auto compiled_model = core.compile_model(model, "CPU",
    ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
    ov::hint::inference_precision(ov::element::f32));
//! [compile_model_with_property]
}

{
//! [optimal_number_of_infer_requests]
auto compiled_model = core.compile_model(model, "CPU");
auto nireq = compiled_model.get_property(ov::optimal_number_of_infer_requests);
//! [optimal_number_of_infer_requests]
}
{
//! [core_set_property_then_compile]
// set letency hint is a default for CPU
core.set_property("CPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
// compiled with latency configuration hint
auto compiled_model_latency = core.compile_model(model, "CPU");
// compiled with overriden ov::hint::performance_mode value
auto compiled_model_thrp = core.compile_model(model, "CPU",
    ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
//! [core_set_property_then_compile]
}

{
//! [inference_num_threads]
auto compiled_model = core.compile_model(model, "CPU");
auto nthreads = compiled_model.get_property(ov::inference_num_threads);
//! [inference_num_threads]
}

{
//! [multi_device]
auto compiled_model = core.compile_model(model, "MULTI",
    ov::device::priorities("CPU", "GPU"));
// change the order of priorities
compiled_model.set_property(ov::device::priorities("GPU", "CPU"));
//! [multi_device]
}
return 0;
}
