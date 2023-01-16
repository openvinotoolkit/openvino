// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <openvino/c/openvino.h>

int main_new() {
    char* model_path = NULL;
    char* tensor_name = NULL;

    ov_core_t* core = NULL;
    ov_core_create(&core);
    
    ov_model_t* model = NULL;
    ov_core_read_model(core, model_path, NULL, &model);

    {
    //! [ov_mean_scale]
    ov_preprocess_prepostprocessor_t* preprocess = NULL;
    ov_preprocess_input_info_t* input_info = NULL;
    ov_preprocess_input_model_info_t* input_model = NULL;
    ov_preprocess_preprocess_steps_t* input_process = NULL;
    ov_layout_t* layout = NULL;
    ov_model_t* new_model = NULL;

    ov_preprocess_prepostprocessor_create(model, &preprocess);
    ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info);
    ov_preprocess_input_info_get_model_info(input_info, &input_model);
    // we only need to know where is C dimension
    ov_layout_create("...C", &layout);
    ov_preprocess_input_model_info_set_layout(input_model, layout);
    // specify scale and mean values, order of operations is important
    ov_preprocess_input_info_get_preprocess_steps(input_info, &input_process);
    ov_preprocess_preprocess_steps_mean(input_process, 116.78f);
    ov_preprocess_preprocess_steps_scale(input_process, 57.21f);
    // insert preprocessing operations to the 'model'
    ov_preprocess_prepostprocessor_build(preprocess, &new_model);

    ov_layout_free(layout);
    ov_model_free(new_model);
    ov_preprocess_input_model_info_free(input_model);
    ov_preprocess_preprocess_steps_free(input_process);
    ov_preprocess_input_info_free(input_info);
    ov_preprocess_prepostprocessor_free(preprocess);
    //! [ov_mean_scale]
    }

    {
    //! [ov_conversions]
    ov_preprocess_prepostprocessor_t* preprocess = NULL;
    ov_preprocess_input_info_t* input_info = NULL;
    ov_preprocess_input_tensor_info_t* input_tensor_info = NULL;
    ov_layout_t* layout_nhwc = NULL;
    ov_preprocess_input_model_info_t* input_model = NULL;
    ov_layout_t* layout_nchw = NULL;
    ov_model_t* new_model = NULL;

    ov_preprocess_prepostprocessor_create(model, &preprocess);
    ov_preprocess_prepostprocessor_get_input_info_by_name(preprocess, tensor_name, &input_info);
    ov_preprocess_input_info_get_tensor_info(input_info, &input_tensor_info);

    ov_layout_create("NHWC", &layout_nhwc);
    ov_preprocess_input_tensor_info_set_layout(input_tensor_info, layout_nhwc);
    ov_preprocess_input_tensor_info_set_element_type(input_tensor_info, ov_element_type_e::U8);

    ov_preprocess_input_info_get_model_info(input_info, &input_model);
    ov_layout_create("NCHW", &layout_nchw);
    ov_preprocess_input_model_info_set_layout(input_model, layout_nchw);
    // layout and precision conversion is inserted automatically,
    // because tensor format != model input format
    ov_preprocess_prepostprocessor_build(preprocess, &new_model);

    ov_layout_free(layout_nchw);
    ov_layout_free(layout_nhwc);
    ov_model_free(new_model);
    ov_preprocess_input_model_info_free(input_model);
    ov_preprocess_input_tensor_info_free(input_tensor_info);
    ov_preprocess_input_info_free(input_info);
    ov_preprocess_prepostprocessor_free(preprocess);
    //! [ov_conversions]
    }

    {
    //! [ov_color_space]
    ov_preprocess_prepostprocessor_t* preprocess = NULL;
    ov_preprocess_input_info_t* input_info = NULL;
    ov_preprocess_input_tensor_info_t* input_tensor_info = NULL;
    ov_preprocess_preprocess_steps_t* input_process = NULL;
    ov_model_t* new_model = NULL;

    ov_preprocess_prepostprocessor_create(model, &preprocess);
    ov_preprocess_prepostprocessor_get_input_info_by_name(preprocess, tensor_name, &input_info);
    ov_preprocess_input_info_get_tensor_info(input_info, &input_tensor_info);
    ov_preprocess_input_tensor_info_set_color_format(input_tensor_info, ov_color_format_e::NV12_TWO_PLANES);
    // add NV12 to BGR conversion
    ov_preprocess_input_info_get_preprocess_steps(input_info, &input_process);
    ov_preprocess_preprocess_steps_convert_color(input_process, ov_color_format_e::BGR);
    // and insert operations to the model
    ov_preprocess_prepostprocessor_build(preprocess, &new_model);

    ov_preprocess_input_tensor_info_free(input_tensor_info);
    ov_preprocess_preprocess_steps_free(input_process);
    ov_preprocess_input_info_free(input_info);
    ov_preprocess_prepostprocessor_free(preprocess);
    ov_model_free(new_model);
    //! [ov_color_space]
    }

    {
    //! [ov_image_scale]
    ov_preprocess_prepostprocessor_t* preprocess = NULL;
    ov_preprocess_input_info_t* input_info = NULL;
    ov_preprocess_input_tensor_info_t* input_tensor_info = NULL;
    ov_preprocess_input_model_info_t* input_model = NULL;
    ov_layout_t* layout = NULL;
    ov_preprocess_preprocess_steps_t* input_process = NULL;
    ov_model_t* new_model = NULL;

    ov_preprocess_prepostprocessor_create(model, &preprocess);
    ov_preprocess_prepostprocessor_get_input_info_by_name(preprocess, tensor_name, &input_info);
    ov_preprocess_input_info_get_tensor_info(input_info, &input_tensor_info);
    // scale from the specified tensor size
    ov_preprocess_input_tensor_info_set_spatial_static_shape(input_tensor_info, 448, 448);
    // need to specify H and W dimensions in model, others are not important
    ov_preprocess_input_info_get_model_info(input_info, &input_model);
    ov_layout_create("??HW", &layout);
    ov_preprocess_input_model_info_set_layout(input_model, layout);
    // scale to model shape
    ov_preprocess_input_info_get_preprocess_steps(input_info, &input_process);
    ov_preprocess_preprocess_steps_resize(input_process, ov_preprocess_resize_algorithm_e::RESIZE_LINEAR);
    // and insert operations to the model
    ov_preprocess_prepostprocessor_build(preprocess, &new_model);

    ov_layout_free(layout);
    ov_preprocess_preprocess_steps_free(input_process);
    ov_preprocess_input_model_info_free(input_model);
    ov_preprocess_input_tensor_info_free(input_tensor_info);
    ov_preprocess_input_info_free(input_info);
    ov_model_free(new_model);
    ov_preprocess_prepostprocessor_free(preprocess);
    //! [ov_image_scale]
    ov_model_free(model);
    ov_core_free(core);
    }

    return 0;
}

int main_old() {
    {
    //! [c_api_ppp]
    // No preprocessing related interfaces provided by C API 1.0
    //! [c_api_ppp]
    }
    return 0;
}
