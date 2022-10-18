#include <c_api/ie_c_api.h>
#include <openvino/c/openvino.h>

int main_new() {
ov_core_t* core = ov_core_create(&core);
//! [core_get_ro_property]
char* ret = nullptr;
ov_core_get_property(core, "CPU", ov_property_key_device_full_name, &ret);
//! [core_get_ro_property]

//! [core_get_rw_property]
char* ret = nullptr;
ov_core_get_property(core, "CPU", ov_property_key_num_streams, &ret);
//! [core_get_rw_property]

//! [core_set_property]
ov_core_set_property(core, "CPU", ov_property_key_enable_profiling, "TRUE");
//! [core_set_property]

ov_model_t* model = nullptr;
ov_core_read_model(core, "sample.xml", nullptr, &model);
//! [core_compile_model]
ov_compiled_model_t* compiled_model = nullptr;
ov_core_compile_model(core, model, "MULTI", 2, &compiled_model, ov_property_key_device_priorities, "TRUE");
//! [core_compile_model]

//! [compiled_model_set_property]
// turn CPU off for multi-device execution
ov_compiled_model_set_property(compiled_model, ov_property_key_device_priorities, "CPU, GPU");
//! [compiled_model_set_property]

{
//! [compiled_model_get_ro_property]
char* result = nullptr;
ov_compiled_model_get_property(compiled_model, ov_property_key_hint_num_requests, &result);
//! [compiled_model_get_ro_property]
}

{
//! [compiled_model_get_rw_property]
char* result = nullptr;
ov_compiled_model_get_property(compiled_model, ov_property_key_hint_performance_mode, &result);
//! [compiled_model_get_rw_property]
}

return 0;
}


int main_old() {
ie_core_t *core = ie_core_create("", &core);
    
//! [core_get_metric]
ie_param_t param;
param.params = nullptr;
ie_core_get_metric(core, "CPU", "SUPPORTED_CONFIG_KEYS", &param);
//! [core_get_metric]

//! [core_get_config]
ie_param_t param;
param.params = nullptr;
ie_core_get_config(core, "CPU", "CPU_THREADS_NUM", &param);
//! [core_get_config]

//! [core_set_config]
ie_config_t config = {"PERF_COUNT", "YES", nullptr};
ie_core_set_config(core, &config, "CPU");
//! [core_set_config]

ie_network_t *network = nullptr;
ie_core_read_network(core, "sample.xml", "sample.bin", &network);
//! [core_load_network]
ie_config_t config = {"DEVICE_PRIORITIES", "CPU, GPU", nullptr};
ie_executable_network_t *exe_network = nullptr;
ie_core_load_network(core, network, "MULTI", &config, &exe_network);
//! [core_load_network]

//! [executable_network_set_config]
// turn CPU off for multi-device executio
ie_config_t config_param = {"DEVICE_PRIORITIES", "GPU", nullptr};
ie_exec_network_set_config(exe_network, &config_param);
//! [executable_network_set_config]

{
//! [executable_network_get_metric]
ie_param_t param;
param.params = nullptr;
ie_exec_network_get_metric(exe_network, "SUPPORTED_CONFIG_KEYS", &param);
//! [executable_network_get_metric]
}

{
//! [executable_network_get_config]
ie_param_t param;
param.params = nullptr;
ie_exec_network_get_config(exe_network, "CPU_THREADS_NUM", &param);
//! [executable_network_get_config]
}

return 0;
}
