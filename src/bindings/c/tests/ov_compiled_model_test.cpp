// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ov_test.hpp"

class ov_compiled_model : public ::testing::TestWithParam<std::string> {};
INSTANTIATE_TEST_SUITE_P(device_name, ov_compiled_model, ::testing::Values("CPU"));
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

TEST_P(ov_compiled_model, get_inputs) {
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

    ov_output_const_node_list_t input_ports;
    input_ports.ports = nullptr;
    input_ports.size = 0;
    OV_EXPECT_OK(ov_compiled_model_inputs(compiled_model, &input_ports));
    EXPECT_NE(nullptr, input_ports.ports);
    EXPECT_NE(0, input_ports.size);

    ov_output_const_node_list_free(&input_ports);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model, get_inputs_error_handling) {
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

    ov_output_const_node_list_t input_ports;
    input_ports.ports = nullptr;
    input_ports.size = 0;
    OV_EXPECT_NOT_OK(ov_compiled_model_inputs(nullptr, &input_ports));
    OV_EXPECT_NOT_OK(ov_compiled_model_inputs(compiled_model, nullptr));

    ov_output_const_node_list_free(&input_ports);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model, get_outputs) {
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

    ov_output_const_node_list_t output_ports;
    output_ports.ports = nullptr;
    output_ports.size = 0;
    OV_EXPECT_OK(ov_compiled_model_outputs(compiled_model, &output_ports));
    EXPECT_NE(nullptr, output_ports.ports);
    EXPECT_NE(0, output_ports.size);

    ov_output_const_node_list_free(&output_ports);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model, get_outputs_error_handling) {
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

    ov_output_const_node_list_t output_ports;
    output_ports.ports = nullptr;
    output_ports.size = 0;
    OV_EXPECT_NOT_OK(ov_compiled_model_outputs(nullptr, &output_ports));
    OV_EXPECT_NOT_OK(ov_compiled_model_outputs(compiled_model, nullptr));

    ov_output_const_node_list_free(&output_ports);
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