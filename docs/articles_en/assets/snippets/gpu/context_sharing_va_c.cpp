// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_LIBVA
#include <openvino/c/openvino.h>
#include <openvino/c/gpu/gpu_plugin_properties.h>
#include <openvino/runtime/intel_gpu/ocl/va.hpp>

VADisplay get_va_display();
VASurfaceID decode_va_surface();

int main() {
    ov_core_t* core = NULL;
    ov_model_t* model = NULL;
    ov_compiled_model_t* compiled_model = NULL;
    ov_infer_request_t* infer_request = NULL;
    ov_remote_context_t* shared_va_context = NULL;
    ov_tensor_t* remote_tensor = NULL;
    ov_preprocess_prepostprocessor_t* preprocess = NULL;
    ov_preprocess_input_info_t* preprocess_input_info = NULL;
    ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info = NULL;
    ov_preprocess_preprocess_steps_t* preprocess_input_steps = NULL;
    ov_preprocess_input_model_info_t* preprocess_input_model_info = NULL;
    ov_layout_t* layout = NULL;
    ov_model_t* new_model = NULL;

    ov_output_const_port_t* input_port = NULL;
    char* in_tensor_name = NULL;
    char* out_tensor_name = NULL;
    ov_shape_t* input_shape = NULL;
    ov_element_type_e input_type;

    const int height = 480;
    const int width = 640;

    // initialize the objects
    ov_core_create(&core);
    ov_core_read_model(core, "model.xml", "model.bin", &model);

    // ...

    //! [context_sharing_va]

    // ...

    ov_preprocess_prepostprocessor_create(model, &preprocess);
    ov_preprocess_prepostprocessor_get_input_info(preprocess, &preprocess_input_info);
    ov_preprocess_input_info_get_tensor_info(preprocess_input_info, &preprocess_input_tensor_info);
    ov_preprocess_input_tensor_info_set_element_type(preprocess_input_tensor_info, U8);
    ov_preprocess_input_tensor_info_set_color_format_with_subname(preprocess_input_tensor_info,
                                                                  NV12_TWO_PLANES,
                                                                  2,
                                                                  "y",
                                                                  "uv");
    ov_preprocess_input_tensor_info_set_memory_type(preprocess_input_tensor_info, "GPU_SURFACE");
    ov_preprocess_input_tensor_info_set_spatial_static_shape(preprocess_input_tensor_info, height, width);
    ov_preprocess_input_info_get_preprocess_steps(preprocess_input_info, &preprocess_input_steps);
    ov_preprocess_preprocess_steps_convert_color(preprocess_input_steps, BGR);
    ov_preprocess_preprocess_steps_resize(preprocess_input_steps, RESIZE_LINEAR);
    ov_preprocess_input_info_get_model_info(preprocess_input_info, &preprocess_input_model_info);
    ov_layout_create("NCHW", &layout);
    ov_preprocess_input_model_info_set_layout(preprocess_input_model_info, layout);
    ov_preprocess_prepostprocessor_build(preprocess, &new_model);

    VADisplay display = get_va_display();
    // create the shared context object
    ov_core_create_context(core,
                           "GPU",
                           4,
                           &shared_va_context,
                           ov_property_key_intel_gpu_context_type,
                           "VA_SHARED",
                           ov_property_key_intel_gpu_va_device,
                           display);

    // compile model within a shared context
    ov_core_compile_model_with_context(core, new_model, shared_va_context, 0, &compiled_model);

    ov_output_const_port_t* port_0 = NULL;
    char* input_name_0 = NULL;
    ov_model_const_input_by_index(new_model, 0, &port_0);
    ov_port_get_any_name(port_0, &input_name_0);

    ov_output_const_port_t* port_1 = NULL;
    char* input_name_1 = NULL;
    ov_model_const_input_by_index(new_model, 1, &port_1);
    ov_port_get_any_name(port_1, &input_name_1);

    ov_shape_t shape_y = {0, NULL};
    ov_shape_t shape_uv = {0, NULL};
    ov_const_port_get_shape(port_0, &shape_y);
    ov_const_port_get_shape(port_1, &shape_uv);

    // execute decoding and obtain decoded surface handle
    VASurfaceID va_surface = decode_va_surface();
    //     ...
    //wrap decoder output into RemoteBlobs and set it as inference input
    
    ov_tensor_t* remote_tensor_y = NULL;
    ov_tensor_t* remote_tensor_uv = NULL;
    ov_remote_context_create_tensor(shared_va_context,
                                    U8,
                                    shape_y,
                                    6,
                                    &remote_tensor_y,
                                    ov_property_key_intel_gpu_shared_mem_type,
                                    "VA_SURFACE",
                                    ov_property_key_intel_gpu_dev_object_handle,
                                    va_surface,
                                    ov_property_key_intel_gpu_va_plane,
                                    0);
    ov_remote_context_create_tensor(shared_va_context,
                                    U8,
                                    shape_uv,
                                    6,
                                    &remote_tensor_uv,
                                    ov_property_key_intel_gpu_shared_mem_type,
                                    "VA_SURFACE",
                                    ov_property_key_intel_gpu_dev_object_handle,
                                    va_surface,
                                    ov_property_key_intel_gpu_va_plane,
                                    1);

    ov_compiled_model_create_infer_request(compiled_model, &infer_request);
    ov_infer_request_set_tensor(infer_request, input_name_0, remote_tensor_y);
    ov_infer_request_set_tensor(infer_request, input_name_1, remote_tensor_uv);
    ov_infer_request_infer(infer_request);
    //! [context_sharing_va]

    // deinitialization
    ov_free(input_name_0);
    ov_free(input_name_1);
    ov_output_const_port_free(port_0);
    ov_output_const_port_free(port_1);
    ov_layout_free(layout);
    ov_preprocess_input_model_info_free(preprocess_input_model_info);
    ov_preprocess_preprocess_steps_free(preprocess_input_steps);
    ov_preprocess_input_tensor_info_free(preprocess_input_tensor_info);
    ov_preprocess_input_info_free(preprocess_input_info);
    ov_model_free(new_model);
    ov_preprocess_prepostprocessor_free(preprocess);
    ov_tensor_free(remote_tensor_y);
    ov_tensor_free(remote_tensor_uv);
    ov_shape_free(&shape_y);
    ov_shape_free(&shape_uv);
    ov_infer_request_free(infer_request);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_remote_context_free(shared_va_context);
    ov_core_free(core);

    return 0;
}
#endif  // ENABLE_LIBVA
