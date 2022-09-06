// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ov_test.hpp"

TEST(ov_preprocess, ov_preprocess_prepostprocessor_create) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_prepostprocessor_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_prepostprocessor_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_prepostprocessor_get_input_info) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_prepostprocessor_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t* input_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_get_input_info(preprocess, &input_info));
    ASSERT_NE(nullptr, input_info);

    ov_preprocess_input_info_free(input_info);
    ov_preprocess_prepostprocessor_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_prepostprocessor_get_input_info_by_name) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_prepostprocessor_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t* input_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_get_input_info_by_name(preprocess, "data", &input_info));
    ASSERT_NE(nullptr, input_info);

    ov_preprocess_input_info_free(input_info);
    ov_preprocess_prepostprocessor_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_prepostprocessor_get_input_info_by_index) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_prepostprocessor_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t* input_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    ASSERT_NE(nullptr, input_info);

    ov_preprocess_input_info_free(input_info);
    ov_preprocess_prepostprocessor_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_input_info_get_tensor_info) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_prepostprocessor_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t* input_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    ASSERT_NE(nullptr, input_info);

    ov_preprocess_input_tensor_info_t* input_tensor_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_input_info_get_tensor_info(input_info, &input_tensor_info));
    ASSERT_NE(nullptr, input_tensor_info);

    ov_preprocess_input_tensor_info_free(input_tensor_info);
    ov_preprocess_input_info_free(input_info);
    ov_preprocess_prepostprocessor_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_input_info_get_preprocess_steps) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_prepostprocessor_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t* input_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    ASSERT_NE(nullptr, input_info);

    ov_preprocess_preprocess_steps_t* input_process = nullptr;
    OV_ASSERT_OK(ov_preprocess_input_info_get_preprocess_steps(input_info, &input_process));
    ASSERT_NE(nullptr, input_process);

    ov_preprocess_preprocess_steps_free(input_process);
    ov_preprocess_input_info_free(input_info);
    ov_preprocess_prepostprocessor_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_preprocess_steps_resize) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_prepostprocessor_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t* input_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    ASSERT_NE(nullptr, input_info);

    ov_preprocess_preprocess_steps_t* input_process = nullptr;
    OV_ASSERT_OK(ov_preprocess_input_info_get_preprocess_steps(input_info, &input_process));
    ASSERT_NE(nullptr, input_process);

    OV_ASSERT_OK(ov_preprocess_preprocess_steps_resize(input_process, ov_preprocess_resize_algorithm_e::RESIZE_LINEAR));

    ov_preprocess_preprocess_steps_free(input_process);
    ov_preprocess_input_info_free(input_info);
    ov_preprocess_prepostprocessor_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_input_tensor_info_set_element_type) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_prepostprocessor_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t* input_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    ASSERT_NE(nullptr, input_info);

    ov_preprocess_input_tensor_info_t* input_tensor_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_input_info_get_tensor_info(input_info, &input_tensor_info));
    ASSERT_NE(nullptr, input_tensor_info);

    OV_ASSERT_OK(ov_preprocess_input_tensor_info_set_element_type(input_tensor_info, ov_element_type_e::F32));

    ov_preprocess_input_tensor_info_free(input_tensor_info);
    ov_preprocess_input_info_free(input_info);
    ov_preprocess_prepostprocessor_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_input_tensor_info_set_from) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_prepostprocessor_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t* input_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    ASSERT_NE(nullptr, input_info);

    ov_preprocess_input_tensor_info_t* input_tensor_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_input_info_get_tensor_info(input_info, &input_tensor_info));
    ASSERT_NE(nullptr, input_tensor_info);

    ov_tensor_t* tensor = nullptr;
    ov_shape_t shape;
    int64_t dims[4] = {1, 416, 416, 4};
    OV_ASSERT_OK(ov_shape_create(4, dims, &shape));

    OV_ASSERT_OK(ov_tensor_create(ov_element_type_e::F32, shape, &tensor));
    OV_ASSERT_OK(ov_preprocess_input_tensor_info_set_from(input_tensor_info, tensor));

    ov_preprocess_input_tensor_info_free(input_tensor_info);
    ov_preprocess_input_info_free(input_info);
    ov_preprocess_prepostprocessor_free(preprocess);
    ov_shape_free(&shape);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_input_tensor_info_set_layout) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_prepostprocessor_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t* input_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    ASSERT_NE(nullptr, input_info);

    ov_preprocess_input_tensor_info_t* input_tensor_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_input_info_get_tensor_info(input_info, &input_tensor_info));
    ASSERT_NE(nullptr, input_tensor_info);

    ov_layout_t* layout = nullptr;
    const char* input_layout_desc = "NCHW";
    OV_ASSERT_OK(ov_layout_create(input_layout_desc, &layout));
    OV_ASSERT_OK(ov_preprocess_input_tensor_info_set_layout(input_tensor_info, layout));

    ov_preprocess_input_tensor_info_free(input_tensor_info);
    ov_preprocess_input_info_free(input_info);
    ov_preprocess_prepostprocessor_free(preprocess);
    ov_layout_free(layout);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_input_tensor_info_set_color_format) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_prepostprocessor_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t* input_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    ASSERT_NE(nullptr, input_info);

    ov_preprocess_input_tensor_info_t* input_tensor_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_input_info_get_tensor_info(input_info, &input_tensor_info));
    ASSERT_NE(nullptr, input_tensor_info);

    OV_ASSERT_OK(
        ov_preprocess_input_tensor_info_set_color_format(input_tensor_info, ov_color_format_e::NV12_SINGLE_PLANE));

    ov_preprocess_input_tensor_info_free(input_tensor_info);
    ov_preprocess_input_info_free(input_info);
    ov_preprocess_prepostprocessor_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_input_tensor_info_set_spatial_static_shape) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_prepostprocessor_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t* input_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    ASSERT_NE(nullptr, input_info);

    ov_preprocess_input_tensor_info_t* input_tensor_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_input_info_get_tensor_info(input_info, &input_tensor_info));
    ASSERT_NE(nullptr, input_tensor_info);

    size_t input_height = 500;
    size_t input_width = 500;
    OV_ASSERT_OK(
        ov_preprocess_input_tensor_info_set_spatial_static_shape(input_tensor_info, input_height, input_width));

    ov_preprocess_input_tensor_info_free(input_tensor_info);
    ov_preprocess_input_info_free(input_info);
    ov_preprocess_prepostprocessor_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_preprocess_steps_convert_element_type) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_prepostprocessor_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t* input_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    ASSERT_NE(nullptr, input_info);

    ov_preprocess_preprocess_steps_t* input_process = nullptr;
    OV_ASSERT_OK(ov_preprocess_input_info_get_preprocess_steps(input_info, &input_process));
    ASSERT_NE(nullptr, input_process);

    ov_preprocess_input_tensor_info_t* input_tensor_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_input_info_get_tensor_info(input_info, &input_tensor_info));
    ASSERT_NE(nullptr, input_tensor_info);

    OV_ASSERT_OK(ov_preprocess_input_tensor_info_set_element_type(input_tensor_info, ov_element_type_e::U8));
    OV_ASSERT_OK(ov_preprocess_preprocess_steps_convert_element_type(input_process, ov_element_type_e::F32));

    ov_preprocess_input_tensor_info_free(input_tensor_info);
    ov_preprocess_preprocess_steps_free(input_process);
    ov_preprocess_input_info_free(input_info);
    ov_preprocess_prepostprocessor_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_preprocess_steps_convert_color) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_prepostprocessor_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t* input_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    ASSERT_NE(nullptr, input_info);

    ov_preprocess_preprocess_steps_t* input_process = nullptr;
    OV_ASSERT_OK(ov_preprocess_input_info_get_preprocess_steps(input_info, &input_process));
    ASSERT_NE(nullptr, input_process);

    ov_preprocess_input_tensor_info_t* input_tensor_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_input_info_get_tensor_info(input_info, &input_tensor_info));
    ASSERT_NE(nullptr, input_tensor_info);

    OV_ASSERT_OK(
        ov_preprocess_input_tensor_info_set_color_format(input_tensor_info, ov_color_format_e::NV12_SINGLE_PLANE));
    OV_ASSERT_OK(ov_preprocess_preprocess_steps_convert_color(input_process, ov_color_format_e::BGR));

    ov_preprocess_input_tensor_info_free(input_tensor_info);
    ov_preprocess_preprocess_steps_free(input_process);
    ov_preprocess_input_info_free(input_info);
    ov_preprocess_prepostprocessor_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_prepostprocessor_get_output_info) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_prepostprocessor_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_output_info_t* output_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_get_output_info(preprocess, &output_info));
    ASSERT_NE(nullptr, output_info);

    ov_preprocess_output_info_free(output_info);
    ov_preprocess_prepostprocessor_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_prepostprocessor_get_output_info_by_index) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_prepostprocessor_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_output_info_t* output_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_get_output_info_by_index(preprocess, 0, &output_info));
    ASSERT_NE(nullptr, output_info);

    ov_preprocess_output_info_free(output_info);
    ov_preprocess_prepostprocessor_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_prepostprocessor_get_output_info_by_name) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_prepostprocessor_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_output_info_t* output_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_get_output_info_by_name(preprocess, "fc_out", &output_info));
    ASSERT_NE(nullptr, output_info);

    ov_preprocess_output_info_free(output_info);
    ov_preprocess_prepostprocessor_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_output_info_get_tensor_info) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_prepostprocessor_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_output_info_t* output_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_get_output_info_by_index(preprocess, 0, &output_info));
    ASSERT_NE(nullptr, output_info);

    ov_preprocess_output_tensor_info_t* output_tensor_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_output_info_get_tensor_info(output_info, &output_tensor_info));
    ASSERT_NE(nullptr, output_tensor_info);

    ov_preprocess_output_tensor_info_free(output_tensor_info);
    ov_preprocess_output_info_free(output_info);
    ov_preprocess_prepostprocessor_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_output_set_element_type) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_prepostprocessor_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_output_info_t* output_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_get_output_info_by_index(preprocess, 0, &output_info));
    ASSERT_NE(nullptr, output_info);

    ov_preprocess_output_tensor_info_t* output_tensor_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_output_info_get_tensor_info(output_info, &output_tensor_info));
    ASSERT_NE(nullptr, output_tensor_info);

    OV_ASSERT_OK(ov_preprocess_output_set_element_type(output_tensor_info, ov_element_type_e::F32));

    ov_preprocess_output_tensor_info_free(output_tensor_info);
    ov_preprocess_output_info_free(output_info);
    ov_preprocess_prepostprocessor_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_input_info_get_model_info) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_prepostprocessor_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t* input_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    ASSERT_NE(nullptr, input_info);

    ov_preprocess_input_model_info_t* input_model = nullptr;
    OV_ASSERT_OK(ov_preprocess_input_info_get_model_info(input_info, &input_model));
    ASSERT_NE(nullptr, input_model);

    ov_preprocess_input_model_info_free(input_model);
    ov_preprocess_input_info_free(input_info);
    ov_preprocess_prepostprocessor_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_input_model_info_set_layout) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_prepostprocessor_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t* input_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    ASSERT_NE(nullptr, input_info);

    ov_preprocess_input_model_info_t* input_model = nullptr;
    OV_ASSERT_OK(ov_preprocess_input_info_get_model_info(input_info, &input_model));
    ASSERT_NE(nullptr, input_model);

    ov_layout_t* layout = nullptr;
    const char* layout_desc = "NCHW";
    OV_ASSERT_OK(ov_layout_create(layout_desc, &layout));
    OV_ASSERT_OK(ov_preprocess_input_model_info_set_layout(input_model, layout));
    ov_layout_free(layout);

    ov_preprocess_input_model_info_free(input_model);
    ov_preprocess_input_info_free(input_info);
    ov_preprocess_prepostprocessor_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_prepostprocessor_build) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_prepostprocessor_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_model_t* new_model = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_build(preprocess, &new_model));
    ASSERT_NE(nullptr, new_model);

    ov_model_free(new_model);
    ov_preprocess_prepostprocessor_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_prepostprocessor_build_apply) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_prepostprocessor_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t* input_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    ASSERT_NE(nullptr, input_info);

    ov_preprocess_input_tensor_info_t* input_tensor_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_input_info_get_tensor_info(input_info, &input_tensor_info));
    ASSERT_NE(nullptr, input_tensor_info);
    ov_tensor_t* tensor = nullptr;
    ov_shape_t shape;
    int64_t dims[4] = {1, 416, 416, 3};
    OV_ASSERT_OK(ov_shape_create(4, dims, &shape));

    OV_ASSERT_OK(ov_tensor_create(ov_element_type_e::U8, shape, &tensor));
    OV_ASSERT_OK(ov_preprocess_input_tensor_info_set_from(input_tensor_info, tensor));

    const char* layout_desc = "NHWC";
    ov_layout_t* layout = nullptr;
    OV_ASSERT_OK(ov_layout_create(layout_desc, &layout));
    OV_ASSERT_OK(ov_preprocess_input_tensor_info_set_layout(input_tensor_info, layout));
    ov_layout_free(layout);

    ov_preprocess_preprocess_steps_t* input_process = nullptr;
    OV_ASSERT_OK(ov_preprocess_input_info_get_preprocess_steps(input_info, &input_process));
    ASSERT_NE(nullptr, input_process);
    OV_ASSERT_OK(ov_preprocess_preprocess_steps_resize(input_process, ov_preprocess_resize_algorithm_e::RESIZE_LINEAR));

    ov_preprocess_input_model_info_t* input_model = nullptr;
    OV_ASSERT_OK(ov_preprocess_input_info_get_model_info(input_info, &input_model));
    ASSERT_NE(nullptr, input_model);

    const char* model_layout_desc = "NCHW";
    ov_layout_t* model_layout = nullptr;
    OV_ASSERT_OK(ov_layout_create(model_layout_desc, &model_layout));
    OV_ASSERT_OK(ov_preprocess_input_model_info_set_layout(input_model, model_layout));
    ov_layout_free(model_layout);

    ov_preprocess_output_info_t* output_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_get_output_info_by_index(preprocess, 0, &output_info));
    ASSERT_NE(nullptr, output_info);
    ov_preprocess_output_tensor_info_t* output_tensor_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_output_info_get_tensor_info(output_info, &output_tensor_info));
    ASSERT_NE(nullptr, output_tensor_info);
    OV_ASSERT_OK(ov_preprocess_output_set_element_type(output_tensor_info, ov_element_type_e::F32));

    ov_model_t* new_model = nullptr;
    OV_ASSERT_OK(ov_preprocess_prepostprocessor_build(preprocess, &new_model));
    ASSERT_NE(nullptr, new_model);

    ov_preprocess_input_tensor_info_free(input_tensor_info);
    ov_tensor_free(tensor);
    ov_shape_free(&shape);
    ov_preprocess_preprocess_steps_free(input_process);
    ov_preprocess_input_model_info_free(input_model);
    ov_preprocess_output_tensor_info_free(output_tensor_info);
    ov_preprocess_output_info_free(output_info);
    ov_preprocess_input_info_free(input_info);
    ov_model_free(new_model);
    ov_preprocess_prepostprocessor_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}