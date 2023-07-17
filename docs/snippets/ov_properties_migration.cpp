#include <openvino/runtime/core.hpp>
#ifndef IN_OV_COMPONENT
#    define IN_OV_COMPONENT
#    define WAS_OV_LIBRARY_DEFINED
#endif

#include <inference_engine.hpp>

#ifdef WAS_OV_LIBRARY_DEFINED
#    undef IN_OV_COMPONENT
#    undef WAS_OV_LIBRARY_DEFINED
#endif

int main_new() {
    ov::Core core;

//! [core_get_ro_property]
// 'auto' is automatically deduced as std::string
// since the type is stored in the property
auto full_device_name = core.get_property("CPU", ov::device::full_name);
//! [core_get_ro_property]

//! [core_get_rw_property]
// 'auto' is automatically deduced as ov::streams::Num
// since the type is stored in the property
auto num_streams = core.get_property("CPU", ov::streams::num);
//! [core_get_rw_property]

//! [core_set_property]
core.set_property("CPU", ov::enable_profiling(true));
//! [core_set_property]

auto model = core.read_model("sample.xml");
//! [core_compile_model]
auto compiled_model = core.compile_model(model, "MULTI",
    ov::device::priorities("GPU", "CPU"),
    ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
    ov::hint::inference_precision(ov::element::f32));
//! [core_compile_model]

//! [compiled_model_set_property]
// turn CPU off for multi-device execution
compiled_model.set_property(ov::device::priorities("GPU"));
//! [compiled_model_set_property]

{
//! [compiled_model_get_ro_property]
// 'auto' is deduced to 'uint32_t'
auto nireq = compiled_model.get_property(ov::optimal_number_of_infer_requests);
//! [compiled_model_get_ro_property]
}

{
//! [compiled_model_get_rw_property]
ov::hint::PerformanceMode perf_mode = compiled_model.get_property(ov::hint::performance_mode);
//! [compiled_model_get_rw_property]
}


return 0;
}


int main_old() {
    InferenceEngine::Core core;
//! [core_get_metric]
auto full_device_name = core.GetMetric("CPU", METRIC_KEY(FULL_DEVICE_NAME)).as<std::string>();
//! [core_get_metric]

//! [core_get_config]
// a user has to parse std::string after
auto num_streams = core.GetConfig("CPU", CONFIG_KEY(CPU_THROUGHPUT_STREAMS)).as<std::string>();
//! [core_get_config]

//! [core_set_config]
core.SetConfig({ { CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES) } }, "CPU");
//! [core_set_config]

auto model = core.ReadNetwork("sample.xml");
//! [core_load_network]
auto exec_network = core.LoadNetwork(model, "MULTI", {
    { MULTI_CONFIG_KEY(DEVICE_PRIORITIES), "CPU, GPU" },
    { CONFIG_KEY(PERFORMANCE_HINT), CONFIG_VALUE(THROUGHPUT) },
    { CONFIG_KEY(ENFORCE_BF16), CONFIG_VALUE(NO) } });
//! [core_load_network]

//! [executable_network_set_config]
// turn CPU off for multi-device execution
exec_network.SetConfig({ { MULTI_CONFIG_KEY(DEVICE_PRIORITIES), "GPU" } });
//! [executable_network_set_config]

{
//! [executable_network_get_metric]
auto nireq = exec_network.GetMetric(EXEC_NETWORK_METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<uint32_t>();
//! [executable_network_get_metric]
}

{
//! [executable_network_get_config]
std::string perf_model = exec_network.GetConfig(CONFIG_KEY(PERFORMANCE_HINT)).as<std::string>();
//! [executable_network_get_config]
}

return 0;
}
