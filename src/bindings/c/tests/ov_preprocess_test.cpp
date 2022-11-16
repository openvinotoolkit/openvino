// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ov_test.hpp"

class ov_preprocess : public ::testing::Test {
protected:
    void SetUp() override {
        core = nullptr;
        model = nullptr;
        preprocess = nullptr;
        input_info = nullptr;
        input_tensor_info = nullptr;
        input_process = nullptr;
        tensor = nullptr;
        output_info = nullptr;
        output_tensor_info = nullptr;
        input_model = nullptr;

        OV_EXPECT_OK(ov_core_create(&core));
        EXPECT_NE(nullptr, core);

        OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
        EXPECT_NE(nullptr, model);
    }
    void TearDown() override {
        ov_preprocess_input_model_info_free(input_model);
        ov_preprocess_output_tensor_info_free(output_tensor_info);
        ov_preprocess_output_info_free(output_info);
        ov_tensor_free(tensor);
        ov_preprocess_preprocess_steps_free(input_process);
        ov_preprocess_input_tensor_info_free(input_tensor_info);
        ov_preprocess_input_info_free(input_info);
        ov_preprocess_prepostprocessor_free(preprocess);
        ov_model_free(model);
        ov_core_free(core);
    }

public:
    ov_core_t* core;
    ov_model_t* model;
    ov_preprocess_prepostprocessor_t* preprocess;
    ov_preprocess_input_info_t* input_info;
    ov_preprocess_input_tensor_info_t* input_tensor_info;
    ov_preprocess_preprocess_steps_t* input_process;
    ov_tensor_t* tensor;
    ov_preprocess_output_info_t* output_info;
    ov_preprocess_output_tensor_info_t* output_tensor_info;
    ov_preprocess_input_model_info_t* input_model;
};

TEST_F(ov_preprocess, ov_preprocess_prepostprocessor_create) {
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);
}

TEST_F(ov_preprocess, ov_preprocess_prepostprocessor_get_input_info) {
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);

    OV_EXPECT_OK(ov_preprocess_prepostprocessor_get_input_info(preprocess, &input_info));
    EXPECT_NE(nullptr, input_info);
}

TEST_F(ov_preprocess, ov_preprocess_prepostprocessor_get_input_info_by_name) {
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);

    OV_EXPECT_OK(ov_preprocess_prepostprocessor_get_input_info_by_name(preprocess, "data", &input_info));
    EXPECT_NE(nullptr, input_info);
}

TEST_F(ov_preprocess, ov_preprocess_prepostprocessor_get_input_info_by_index) {
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);

    OV_EXPECT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    EXPECT_NE(nullptr, input_info);
}

TEST_F(ov_preprocess, ov_preprocess_input_info_get_tensor_info) {
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);

    OV_EXPECT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    EXPECT_NE(nullptr, input_info);

    OV_EXPECT_OK(ov_preprocess_input_info_get_tensor_info(input_info, &input_tensor_info));
    EXPECT_NE(nullptr, input_tensor_info);
}

TEST_F(ov_preprocess, ov_preprocess_input_info_get_preprocess_steps) {
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);

    OV_EXPECT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    EXPECT_NE(nullptr, input_info);

    OV_EXPECT_OK(ov_preprocess_input_info_get_preprocess_steps(input_info, &input_process));
    EXPECT_NE(nullptr, input_process);
}

TEST_F(ov_preprocess, ov_preprocess_preprocess_steps_resize) {
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);

    OV_EXPECT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    EXPECT_NE(nullptr, input_info);

    OV_EXPECT_OK(ov_preprocess_input_info_get_preprocess_steps(input_info, &input_process));
    EXPECT_NE(nullptr, input_process);

    OV_EXPECT_OK(ov_preprocess_preprocess_steps_resize(input_process, ov_preprocess_resize_algorithm_e::RESIZE_LINEAR));
}

TEST_F(ov_preprocess, ov_preprocess_preprocess_steps_scale) {
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);

    OV_EXPECT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    EXPECT_NE(nullptr, input_info);

    OV_EXPECT_OK(ov_preprocess_input_info_get_preprocess_steps(input_info, &input_process));
    EXPECT_NE(nullptr, input_process);

    OV_EXPECT_OK(ov_preprocess_preprocess_steps_scale(input_process, 2.0f));
}

TEST_F(ov_preprocess, ov_preprocess_preprocess_steps_mean) {
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);

    OV_EXPECT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    EXPECT_NE(nullptr, input_info);

    OV_EXPECT_OK(ov_preprocess_input_info_get_preprocess_steps(input_info, &input_process));
    EXPECT_NE(nullptr, input_process);

    OV_EXPECT_OK(ov_preprocess_preprocess_steps_mean(input_process, 2.0f));
}

TEST_F(ov_preprocess, ov_preprocess_preprocess_steps_crop) {
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);

    OV_EXPECT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    EXPECT_NE(nullptr, input_info);

    OV_EXPECT_OK(ov_preprocess_input_info_get_preprocess_steps(input_info, &input_process));
    EXPECT_NE(nullptr, input_process);

    int32_t begin[] = {0, 0, 5, 10};
    int32_t end[] = {1, 3, 15, 20};
    OV_EXPECT_OK(ov_preprocess_preprocess_steps_crop(input_process, begin, 4, end, 4));
}

TEST_F(ov_preprocess, ov_preprocess_preprocess_steps_convert_layout) {
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);

    OV_EXPECT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    EXPECT_NE(nullptr, input_info);

    OV_EXPECT_OK(ov_preprocess_input_info_get_preprocess_steps(input_info, &input_process));
    EXPECT_NE(nullptr, input_process);

    ov_layout_t* layout = nullptr;
    const char* input_layout_desc = "NCHW";
    OV_EXPECT_OK(ov_layout_create(input_layout_desc, &layout));
    OV_EXPECT_OK(ov_preprocess_preprocess_steps_convert_layout(input_process, layout));

    ov_layout_free(layout);
}

TEST_F(ov_preprocess, ov_preprocess_preprocess_steps_reverse_channels) {
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);

    OV_EXPECT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    EXPECT_NE(nullptr, input_info);

    OV_EXPECT_OK(ov_preprocess_input_info_get_preprocess_steps(input_info, &input_process));
    EXPECT_NE(nullptr, input_process);

    OV_EXPECT_OK(ov_preprocess_preprocess_steps_reverse_channels(input_process));
}

TEST_F(ov_preprocess, ov_preprocess_input_tensor_info_set_element_type) {
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);

    OV_EXPECT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    EXPECT_NE(nullptr, input_info);

    OV_EXPECT_OK(ov_preprocess_input_info_get_tensor_info(input_info, &input_tensor_info));
    EXPECT_NE(nullptr, input_tensor_info);

    OV_EXPECT_OK(ov_preprocess_input_tensor_info_set_element_type(input_tensor_info, ov_element_type_e::F32));
}

TEST_F(ov_preprocess, ov_preprocess_input_tensor_info_set_from) {
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);

    OV_EXPECT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    EXPECT_NE(nullptr, input_info);

    OV_EXPECT_OK(ov_preprocess_input_info_get_tensor_info(input_info, &input_tensor_info));
    EXPECT_NE(nullptr, input_tensor_info);

    ov_shape_t shape;
    int64_t dims[4] = {1, 416, 416, 4};
    OV_EXPECT_OK(ov_shape_create(4, dims, &shape));

    OV_EXPECT_OK(ov_tensor_create(ov_element_type_e::F32, shape, &tensor));
    OV_EXPECT_OK(ov_preprocess_input_tensor_info_set_from(input_tensor_info, tensor));

    ov_shape_free(&shape);
}

TEST_F(ov_preprocess, ov_preprocess_input_tensor_info_set_layout) {
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);

    OV_EXPECT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    EXPECT_NE(nullptr, input_info);

    OV_EXPECT_OK(ov_preprocess_input_info_get_tensor_info(input_info, &input_tensor_info));
    EXPECT_NE(nullptr, input_tensor_info);

    ov_layout_t* layout = nullptr;
    const char* input_layout_desc = "NCHW";
    OV_EXPECT_OK(ov_layout_create(input_layout_desc, &layout));
    OV_EXPECT_OK(ov_preprocess_input_tensor_info_set_layout(input_tensor_info, layout));

    ov_layout_free(layout);
}

TEST_F(ov_preprocess, ov_preprocess_input_tensor_info_set_color_format) {
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);

    OV_EXPECT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    EXPECT_NE(nullptr, input_info);

    OV_EXPECT_OK(ov_preprocess_input_info_get_tensor_info(input_info, &input_tensor_info));
    EXPECT_NE(nullptr, input_tensor_info);

    OV_EXPECT_OK(
        ov_preprocess_input_tensor_info_set_color_format(input_tensor_info, ov_color_format_e::NV12_SINGLE_PLANE));
}

TEST_F(ov_preprocess, ov_preprocess_input_tensor_info_set_spatial_static_shape) {
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);

    OV_EXPECT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    EXPECT_NE(nullptr, input_info);

    OV_EXPECT_OK(ov_preprocess_input_info_get_tensor_info(input_info, &input_tensor_info));
    EXPECT_NE(nullptr, input_tensor_info);

    size_t input_height = 500;
    size_t input_width = 500;
    OV_EXPECT_OK(
        ov_preprocess_input_tensor_info_set_spatial_static_shape(input_tensor_info, input_height, input_width));
}

TEST_F(ov_preprocess, ov_preprocess_preprocess_steps_convert_element_type) {
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);

    OV_EXPECT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    EXPECT_NE(nullptr, input_info);

    OV_EXPECT_OK(ov_preprocess_input_info_get_preprocess_steps(input_info, &input_process));
    EXPECT_NE(nullptr, input_process);

    OV_EXPECT_OK(ov_preprocess_input_info_get_tensor_info(input_info, &input_tensor_info));
    EXPECT_NE(nullptr, input_tensor_info);

    OV_EXPECT_OK(ov_preprocess_input_tensor_info_set_element_type(input_tensor_info, ov_element_type_e::U8));
    OV_EXPECT_OK(ov_preprocess_preprocess_steps_convert_element_type(input_process, ov_element_type_e::F32));
}

TEST_F(ov_preprocess, ov_preprocess_preprocess_steps_convert_color) {
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);

    OV_EXPECT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    EXPECT_NE(nullptr, input_info);

    OV_EXPECT_OK(ov_preprocess_input_info_get_preprocess_steps(input_info, &input_process));
    EXPECT_NE(nullptr, input_process);

    OV_EXPECT_OK(ov_preprocess_input_info_get_tensor_info(input_info, &input_tensor_info));
    EXPECT_NE(nullptr, input_tensor_info);

    OV_EXPECT_OK(
        ov_preprocess_input_tensor_info_set_color_format(input_tensor_info, ov_color_format_e::NV12_SINGLE_PLANE));
    OV_EXPECT_OK(ov_preprocess_preprocess_steps_convert_color(input_process, ov_color_format_e::BGR));
}

TEST_F(ov_preprocess, ov_preprocess_prepostprocessor_get_output_info) {
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);

    OV_EXPECT_OK(ov_preprocess_prepostprocessor_get_output_info(preprocess, &output_info));
    EXPECT_NE(nullptr, output_info);
}

TEST_F(ov_preprocess, ov_preprocess_prepostprocessor_get_output_info_by_index) {
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);

    OV_EXPECT_OK(ov_preprocess_prepostprocessor_get_output_info_by_index(preprocess, 0, &output_info));
    EXPECT_NE(nullptr, output_info);
}

TEST_F(ov_preprocess, ov_preprocess_prepostprocessor_get_output_info_by_name) {
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);

    OV_EXPECT_OK(ov_preprocess_prepostprocessor_get_output_info_by_name(preprocess, "fc_out", &output_info));
    EXPECT_NE(nullptr, output_info);
}

TEST_F(ov_preprocess, ov_preprocess_output_info_get_tensor_info) {
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);

    OV_EXPECT_OK(ov_preprocess_prepostprocessor_get_output_info_by_index(preprocess, 0, &output_info));
    EXPECT_NE(nullptr, output_info);

    OV_EXPECT_OK(ov_preprocess_output_info_get_tensor_info(output_info, &output_tensor_info));
    EXPECT_NE(nullptr, output_tensor_info);
}

TEST_F(ov_preprocess, ov_preprocess_output_set_element_type) {
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);

    OV_EXPECT_OK(ov_preprocess_prepostprocessor_get_output_info_by_index(preprocess, 0, &output_info));
    EXPECT_NE(nullptr, output_info);

    OV_EXPECT_OK(ov_preprocess_output_info_get_tensor_info(output_info, &output_tensor_info));
    EXPECT_NE(nullptr, output_tensor_info);

    OV_EXPECT_OK(ov_preprocess_output_set_element_type(output_tensor_info, ov_element_type_e::F32));
}

TEST_F(ov_preprocess, ov_preprocess_input_info_get_model_info) {
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);

    OV_EXPECT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    EXPECT_NE(nullptr, input_info);

    OV_EXPECT_OK(ov_preprocess_input_info_get_model_info(input_info, &input_model));
    EXPECT_NE(nullptr, input_model);
}

TEST_F(ov_preprocess, ov_preprocess_input_model_info_set_layout) {
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);

    OV_EXPECT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    EXPECT_NE(nullptr, input_info);

    OV_EXPECT_OK(ov_preprocess_input_info_get_model_info(input_info, &input_model));
    EXPECT_NE(nullptr, input_model);

    ov_layout_t* layout = nullptr;
    const char* layout_desc = "NCHW";
    OV_EXPECT_OK(ov_layout_create(layout_desc, &layout));
    OV_EXPECT_OK(ov_preprocess_input_model_info_set_layout(input_model, layout));

    ov_layout_free(layout);
}

TEST_F(ov_preprocess, ov_preprocess_prepostprocessor_build) {
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);

    ov_model_t* new_model = nullptr;
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_build(preprocess, &new_model));
    EXPECT_NE(nullptr, new_model);

    ov_model_free(new_model);
}

TEST_F(ov_preprocess, ov_preprocess_prepostprocessor_build_apply) {
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);

    OV_EXPECT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
    EXPECT_NE(nullptr, input_info);

    OV_EXPECT_OK(ov_preprocess_input_info_get_tensor_info(input_info, &input_tensor_info));
    EXPECT_NE(nullptr, input_tensor_info);

    ov_shape_t shape;
    int64_t dims[4] = {1, 416, 416, 3};
    OV_EXPECT_OK(ov_shape_create(4, dims, &shape));

    OV_EXPECT_OK(ov_tensor_create(ov_element_type_e::U8, shape, &tensor));
    OV_EXPECT_OK(ov_preprocess_input_tensor_info_set_from(input_tensor_info, tensor));

    const char* layout_desc = "NHWC";
    ov_layout_t* layout = nullptr;
    OV_EXPECT_OK(ov_layout_create(layout_desc, &layout));
    OV_EXPECT_OK(ov_preprocess_input_tensor_info_set_layout(input_tensor_info, layout));
    ov_layout_free(layout);

    OV_EXPECT_OK(ov_preprocess_input_info_get_preprocess_steps(input_info, &input_process));
    EXPECT_NE(nullptr, input_process);
    OV_EXPECT_OK(ov_preprocess_preprocess_steps_resize(input_process, ov_preprocess_resize_algorithm_e::RESIZE_LINEAR));

    OV_EXPECT_OK(ov_preprocess_input_info_get_model_info(input_info, &input_model));
    EXPECT_NE(nullptr, input_model);

    const char* model_layout_desc = "NCHW";
    ov_layout_t* model_layout = nullptr;
    OV_EXPECT_OK(ov_layout_create(model_layout_desc, &model_layout));
    OV_EXPECT_OK(ov_preprocess_input_model_info_set_layout(input_model, model_layout));
    ov_layout_free(model_layout);

    OV_EXPECT_OK(ov_preprocess_prepostprocessor_get_output_info_by_index(preprocess, 0, &output_info));
    EXPECT_NE(nullptr, output_info);
    OV_EXPECT_OK(ov_preprocess_output_info_get_tensor_info(output_info, &output_tensor_info));
    EXPECT_NE(nullptr, output_tensor_info);
    OV_EXPECT_OK(ov_preprocess_output_set_element_type(output_tensor_info, ov_element_type_e::F32));

    ov_model_t* new_model = nullptr;
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_build(preprocess, &new_model));
    EXPECT_NE(nullptr, new_model);

    ov_shape_free(&shape);
    ov_model_free(new_model);
}