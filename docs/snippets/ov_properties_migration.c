#include <openvino/c/openvino.h>

int main_new() {
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

