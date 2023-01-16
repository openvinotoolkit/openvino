#include <c_api/ie_c_api.h>
#include <openvino/c/openvino.h>

int ov_properties_migration_main_new_c() {
ov_core_t* core = NULL;
ov_core_create(&core);

//! [core_get_ro_property]
char* full_device_name = NULL;
ov_core_get_property(core, "CPU", ov_property_key_device_full_name, &full_device_name);
ov_free(full_device_name);
//! [core_get_ro_property]

//! [core_get_rw_property]
char* num_streams = NULL;
ov_core_get_property(core, "CPU", ov_property_key_num_streams, &num_streams);
ov_free(num_streams);
//! [core_get_rw_property]

//! [core_set_property]
ov_core_set_property(core, "CPU", ov_property_key_enable_profiling, "TRUE");
//! [core_set_property]

ov_model_t* model = NULL;
ov_core_read_model(core, "sample.xml", NULL, &model);
//! [core_compile_model]
ov_compiled_model_t* compiled_model = NULL;
ov_core_compile_model(core, model, "MULTI", 6, &compiled_model,
    ov_property_key_device_priorities, "CPU, CPU",
    ov_property_key_hint_performance_mode, "THROUGHPUT",
    ov_property_key_hint_inference_precision, "f32");
//! [core_compile_model]

//! [compiled_model_set_property]
// turn CPU off for multi-device execution
ov_compiled_model_set_property(compiled_model, ov_property_key_device_priorities, "GPU");
//! [compiled_model_set_property]

{
//! [compiled_model_get_ro_property]
char* nireq = NULL;
ov_compiled_model_get_property(compiled_model, ov_property_key_hint_num_requests, &nireq);
ov_free(nireq);
//! [compiled_model_get_ro_property]
}

{
//! [compiled_model_get_rw_property]
char* perf_mode = NULL;
ov_compiled_model_get_property(compiled_model, ov_property_key_hint_performance_mode, &perf_mode);
ov_free(perf_mode);
//! [compiled_model_get_rw_property]
}
ov_compiled_model_free(compiled_model);
ov_model_free(model);
ov_core_free(core);
return 0;
}


int ov_properties_migration_main_old_c() {
ie_core_t *core = NULL;
ie_core_create("", &core);
    
//! [core_get_metric]
ie_param_t full_device_name;
full_device_name.params = NULL;
ie_core_get_metric(core, "CPU", "FULL_DEVICE_NAME", &full_device_name);
ie_param_free(&full_device_name);
//! [core_get_metric]

//! [core_get_config]
ie_param_t num_streams;
num_streams.params = NULL;
ie_core_get_config(core, "CPU", "CPU_THROUGHPUT_STREAMS", &num_streams);
ie_param_free(&num_streams);
//! [core_get_config]

//! [core_set_config]
ie_config_t config = {"PERF_COUNT", "YES", NULL};
ie_core_set_config(core, &config, "CPU");
//! [core_set_config]

ie_network_t *network = NULL;
ie_core_read_network(core, "sample.xml", "sample.bin", &network);
//! [core_load_network]
ie_config_t config_1 = {"DEVICE_PRIORITIES", "CPU, GPU", NULL};
ie_config_t config_2 = {"PERFORMANCE_HINT", "THROUGHPUT", &config_1};
ie_config_t config_3 = {"ENFORCE_BF16", "NO", &config_2};
ie_executable_network_t *exe_network = NULL;
ie_core_load_network(core, network, "MULTI", &config_3, &exe_network);
//! [core_load_network]

//! [executable_network_set_config]
// turn CPU off for multi-device executio
ie_config_t config_param = {"DEVICE_PRIORITIES", "GPU", NULL};
ie_exec_network_set_config(exe_network, &config_param);
//! [executable_network_set_config]

{
//! [executable_network_get_metric]
ie_param_t nireq;
nireq.params = NULL;
ie_exec_network_get_metric(exe_network, "OPTIMAL_NUMBER_OF_INFER_REQUESTS", &nireq);
ie_param_free(&nireq);
//! [executable_network_get_metric]
}

{
//! [executable_network_get_config]
ie_param_t perf_model;
perf_model.params = NULL;
ie_exec_network_get_config(exe_network, "PERFORMANCE_HINT", &perf_model);
//! [executable_network_get_config]
}
ie_exec_network_free(&exe_network);
ie_network_free(&network);
ie_core_free(&core);
return 0;
}
