// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ov_test.hpp"

class ov_compiled_model : public ::testing::TestWithParam<std::string> {};
INSTANTIATE_TEST_SUITE_P(device_name, ov_compiled_model, ::testing::Values("CPU"));
TEST_P(ov_compiled_model, ov_compiled_model_inputs_size) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), nullptr, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    size_t input_size;
    OV_ASSERT_OK(ov_compiled_model_inputs_size(compiled_model, &input_size));
    ASSERT_NE(0, input_size);

    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model, ov_compiled_model_input) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), nullptr, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    ov_output_const_port_t* input_port = nullptr;
    OV_ASSERT_OK(ov_compiled_model_input(compiled_model, &input_port));
    ASSERT_NE(nullptr, input_port);

    ov_output_const_port_free(input_port);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model, ov_compiled_model_input_by_index) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), nullptr, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    ov_output_const_port_t* input_port = nullptr;
    OV_ASSERT_OK(ov_compiled_model_input_by_index(compiled_model, 0, &input_port));
    ASSERT_NE(nullptr, input_port);

    ov_shape_t shape;
    OV_ASSERT_OK(ov_const_port_get_shape(input_port, &shape));
    ov_shape_free(&shape);

    ov_output_const_port_free(input_port);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model, ov_compiled_model_input_by_name) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), nullptr, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    ov_output_const_port_t* input_port = nullptr;
    OV_ASSERT_OK(ov_compiled_model_input_by_name(compiled_model, "data", &input_port));
    ASSERT_NE(nullptr, input_port);

    ov_shape_t shape;
    OV_ASSERT_OK(ov_const_port_get_shape(input_port, &shape));
    ov_shape_free(&shape);

    ov_output_const_port_free(input_port);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model, ov_compiled_model_outputs_size) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), nullptr, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    size_t output_size;
    OV_ASSERT_OK(ov_compiled_model_outputs_size(compiled_model, &output_size));
    ASSERT_NE(0, output_size);

    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model, ov_compiled_model_output) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), nullptr, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    ov_output_const_port_t* output_port = nullptr;
    OV_ASSERT_OK(ov_compiled_model_output(compiled_model, &output_port));
    ASSERT_NE(nullptr, output_port);

    ov_output_const_port_free(output_port);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model, ov_compiled_model_output_by_index) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), nullptr, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    ov_output_const_port_t* output_port = nullptr;
    OV_ASSERT_OK(ov_compiled_model_output_by_index(compiled_model, 0, &output_port));
    ASSERT_NE(nullptr, output_port);

    ov_shape_t shape;
    OV_ASSERT_OK(ov_const_port_get_shape(output_port, &shape));
    ov_shape_free(&shape);

    ov_output_const_port_free(output_port);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model, ov_compiled_model_output_by_name) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), nullptr, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    ov_output_const_port_t* output_port = nullptr;
    OV_ASSERT_OK(ov_compiled_model_output_by_name(compiled_model, "fc_out", &output_port));
    ASSERT_NE(nullptr, output_port);

    ov_shape_t shape;
    OV_ASSERT_OK(ov_const_port_get_shape(output_port, &shape));
    ov_shape_free(&shape);

    ov_output_const_port_free(output_port);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model, get_runtime_model) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), nullptr, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    ov_model_t* runtime_model = nullptr;
    OV_EXPECT_OK(ov_compiled_model_get_runtime_model(compiled_model, &runtime_model));
    EXPECT_NE(nullptr, runtime_model);

    ov_model_free(runtime_model);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model, get_runtime_model_error_handling) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), nullptr, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    ov_model_t* runtime_model = nullptr;
    OV_EXPECT_NOT_OK(ov_compiled_model_get_runtime_model(nullptr, &runtime_model));
    OV_EXPECT_NOT_OK(ov_compiled_model_get_runtime_model(compiled_model, nullptr));

    ov_model_free(runtime_model);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model, create_infer_request) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), nullptr, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    ov_infer_request_t* infer_request = nullptr;
    OV_EXPECT_OK(ov_compiled_model_create_infer_request(compiled_model, &infer_request));
    EXPECT_NE(nullptr, infer_request);

    ov_infer_request_free(infer_request);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model, create_infer_request_error_handling) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), nullptr, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    ov_infer_request_t* infer_request = nullptr;
    OV_EXPECT_NOT_OK(ov_compiled_model_create_infer_request(nullptr, &infer_request));
    OV_EXPECT_NOT_OK(ov_compiled_model_create_infer_request(compiled_model, nullptr));

    ov_infer_request_free(infer_request);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}