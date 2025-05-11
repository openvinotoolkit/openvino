// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/c/openvino.h>
#include <openvino/c/gpu/gpu_plugin_properties.h>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>

cl::Image2D get_y_image();
cl::Image2D get_uv_image();

int main() {
    ov_core_t* core = NULL;
    ov_model_t* model = NULL;
    ov_compiled_model_t* compiled_model = NULL;
    ov_infer_request_t* infer_request = NULL;
    ov_preprocess_prepostprocessor_t* preprocess = NULL;
    ov_preprocess_input_info_t* preprocess_input_info = NULL;
    ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info = NULL;
    ov_preprocess_preprocess_steps_t* preprocess_input_steps = NULL;
    ov_preprocess_input_model_info_t* preprocess_input_model_info = NULL;
    ov_layout_t* layout = NULL;
    ov_model_t* model_with_preproc = NULL;
    ov_remote_context_t* gpu_context = NULL;
    char* input_name0 = NULL;
    char* input_name1 = NULL;
    ov_output_const_port_t* input_port0 = NULL;
    ov_output_const_port_t* input_port1 = NULL;
    size_t height = 480;
    size_t width = 640;

    ov_core_create(&core);
    ov_core_read_model(core, "model.xml", "model.bin", &model);

    //! [init_preproc]
    ov_preprocess_prepostprocessor_create(model, &preprocess);
    ov_preprocess_prepostprocessor_get_input_info(preprocess, &preprocess_input_info);
    ov_preprocess_input_info_get_tensor_info(preprocess_input_info, &preprocess_input_tensor_info);
    ov_preprocess_input_tensor_info_set_element_type(preprocess_input_tensor_info, ov_element_type_e::U8);
    ov_preprocess_input_tensor_info_set_color_format_with_subname(preprocess_input_tensor_info,
                                                                  ov_color_format_e::NV12_TWO_PLANES,
                                                                  2,
                                                                  "y",
                                                                  "uv");
    ov_preprocess_input_tensor_info_set_memory_type(preprocess_input_tensor_info, "GPU_SURFACE");
    ov_preprocess_input_tensor_info_set_spatial_static_shape(preprocess_input_tensor_info, height, width);
    ov_preprocess_input_info_get_preprocess_steps(preprocess_input_info, &preprocess_input_steps);
    ov_preprocess_preprocess_steps_convert_color(preprocess_input_steps, ov_color_format_e::BGR);
    ov_preprocess_preprocess_steps_resize(preprocess_input_steps, RESIZE_LINEAR);
    ov_preprocess_input_info_get_model_info(preprocess_input_info, &preprocess_input_model_info);
    ov_layout_create("NCHW", &layout);
    ov_preprocess_input_model_info_set_layout(preprocess_input_model_info, layout);
    ov_preprocess_prepostprocessor_build(preprocess, &model_with_preproc);
    //! [init_preproc]

    ov_core_compile_model(core, model_with_preproc, "GPU", 0, &compiled_model);
    ov_compiled_model_get_context(compiled_model, &gpu_context);
    ov_compiled_model_create_infer_request(compiled_model, &infer_request);

    {
        //! [single_batch]
        ov_model_const_input_by_index(model, 0, &input_port0);
        ov_model_const_input_by_index(model, 1, &input_port1);
        ov_port_get_any_name(input_port0, &input_name0);
        ov_port_get_any_name(input_port1, &input_name1);

        ov_shape_t shape_y, shape_uv;
        ov_tensor_t* remote_tensor_y = NULL;
        ov_tensor_t* remote_tensor_uv = NULL;
        ov_const_port_get_shape(input_port0, &shape_y);
        ov_const_port_get_shape(input_port1, &shape_uv);

        cl::Image2D image_y = get_y_image();
        cl::Image2D image_uv = get_uv_image();
        ov_remote_context_create_tensor(gpu_context,
                                        ov_element_type_e::U8,
                                        shape_y,
                                        4,
                                        &remote_tensor_y,
                                        ov_property_key_intel_gpu_shared_mem_type,
                                        "OCL_IMAGE2D",
                                        ov_property_key_intel_gpu_mem_handle,
                                        image_y.get());

        ov_remote_context_create_tensor(gpu_context,
                                        ov_element_type_e::U8,
                                        shape_uv,
                                        4,
                                        &remote_tensor_y,
                                        ov_property_key_intel_gpu_shared_mem_type,
                                        "OCL_IMAGE2D",
                                        ov_property_key_intel_gpu_mem_handle,
                                        image_uv.get());

        ov_infer_request_set_tensor(infer_request, input_name0, remote_tensor_y);
        ov_infer_request_set_tensor(infer_request, input_name1, remote_tensor_uv);
        ov_infer_request_infer(infer_request);
        //! [single_batch]

        ov_free(input_name0);
        ov_free(input_name1);
        ov_output_const_port_free(input_port0);
        ov_output_const_port_free(input_port1);

        ov_layout_free(layout);
        ov_preprocess_input_model_info_free(preprocess_input_model_info);
        ov_preprocess_preprocess_steps_free(preprocess_input_steps);
        ov_preprocess_input_tensor_info_free(preprocess_input_tensor_info);
        ov_preprocess_input_info_free(preprocess_input_info);
        ov_preprocess_prepostprocessor_free(preprocess);

        ov_tensor_free(remote_tensor_y);
        ov_tensor_free(remote_tensor_uv);
        ov_shape_free(&shape_y);
        ov_shape_free(&shape_uv);

        ov_infer_request_free(infer_request);
        ov_compiled_model_free(compiled_model);
        ov_model_free(model);
        ov_model_free(model_with_preproc);
        ov_remote_context_free(gpu_context);
        ov_core_free(core);
    }

    return 0;
}
