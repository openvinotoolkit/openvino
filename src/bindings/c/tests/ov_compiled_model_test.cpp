// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_test.hpp"

namespace {

class ov_compiled_model_test : public ov_capi_test_base {
    void SetUp() override {
        ov_capi_test_base::SetUp();
    }

    void TearDown() override {
        ov_capi_test_base::TearDown();
    }
};

INSTANTIATE_TEST_SUITE_P(ov_compiled_model, ov_compiled_model_test, ::testing::Values("CPU"));

TEST_P(ov_compiled_model_test, ov_compiled_model_inputs_size) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), 0, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    size_t input_size;
    OV_EXPECT_OK(ov_compiled_model_inputs_size(compiled_model, &input_size));
    EXPECT_NE(0, input_size);

    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model_test, ov_compiled_model_input) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), 0, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    ov_output_const_port_t* input_port = nullptr;
    OV_EXPECT_OK(ov_compiled_model_input(compiled_model, &input_port));
    EXPECT_NE(nullptr, input_port);

    ov_output_const_port_free(input_port);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model_test, ov_compiled_model_input_by_index) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), 0, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    ov_output_const_port_t* input_port = nullptr;
    OV_EXPECT_OK(ov_compiled_model_input_by_index(compiled_model, 0, &input_port));
    EXPECT_NE(nullptr, input_port);

    ov_shape_t shape;
    OV_ASSERT_OK(ov_const_port_get_shape(input_port, &shape));
    ov_shape_free(&shape);

    ov_output_const_port_free(input_port);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model_test, ov_compiled_model_input_by_name) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), 0, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    ov_output_const_port_t* input_port = nullptr;
    OV_EXPECT_OK(ov_compiled_model_input_by_name(compiled_model, "data", &input_port));
    EXPECT_NE(nullptr, input_port);

    ov_shape_t shape;
    OV_ASSERT_OK(ov_const_port_get_shape(input_port, &shape));
    ov_shape_free(&shape);

    ov_output_const_port_free(input_port);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model_test, get_property) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), 0, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    const char* key = ov_property_key_supported_properties;
    char* result = nullptr;
    OV_EXPECT_OK(ov_compiled_model_get_property(compiled_model, key, &result));
    ov_free(result);

    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model_test, set_property) {
    auto device = GetParam();
    std::string device_name = "BATCH:" + device + "(4)";
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), 0, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    const char* key = ov_property_key_auto_batch_timeout;
    char* result = nullptr;

    OV_EXPECT_OK(ov_compiled_model_get_property(compiled_model, key, &result));
    EXPECT_STREQ("1000", result);  // default value from /src/plugins/auto_batch/src/plugin.cpp
    OV_EXPECT_OK(ov_compiled_model_set_property(compiled_model, key, "2000"));
    OV_EXPECT_OK(ov_compiled_model_get_property(compiled_model, key, &result));

    EXPECT_STREQ("2000", result);
    ov_free(result);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model_test, create_compiled_model_with_property) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    const char* key = ov_property_key_hint_performance_mode;
    const char* num = "LATENCY";
    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), 2, &compiled_model, key, num));
    EXPECT_NE(nullptr, compiled_model);
    char* result = nullptr;
    OV_EXPECT_OK(ov_compiled_model_get_property(compiled_model, key, &result));
    EXPECT_STREQ(result, "LATENCY");
    ov_free(result);

    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model_test, ov_compiled_model_outputs_size) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), 0, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    size_t output_size;
    OV_EXPECT_OK(ov_compiled_model_outputs_size(compiled_model, &output_size));
    EXPECT_NE(0, output_size);

    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model_test, ov_compiled_model_output) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), 0, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    ov_output_const_port_t* output_port = nullptr;
    OV_EXPECT_OK(ov_compiled_model_output(compiled_model, &output_port));
    EXPECT_NE(nullptr, output_port);

    ov_output_const_port_free(output_port);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model_test, ov_compiled_model_output_by_index) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), 0, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    ov_output_const_port_t* output_port = nullptr;
    OV_EXPECT_OK(ov_compiled_model_output_by_index(compiled_model, 0, &output_port));
    EXPECT_NE(nullptr, output_port);

    ov_shape_t shape;
    OV_ASSERT_OK(ov_const_port_get_shape(output_port, &shape));
    ov_shape_free(&shape);

    ov_output_const_port_free(output_port);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model_test, ov_compiled_model_output_by_name) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), 0, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    ov_output_const_port_t* output_port = nullptr;
    OV_EXPECT_OK(ov_compiled_model_output_by_name(compiled_model, "relu", &output_port));
    EXPECT_NE(nullptr, output_port);

    ov_shape_t shape;
    OV_ASSERT_OK(ov_const_port_get_shape(output_port, &shape));
    ov_shape_free(&shape);

    ov_output_const_port_free(output_port);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model_test, get_runtime_model) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), 0, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    ov_model_t* runtime_model = nullptr;
    OV_EXPECT_OK(ov_compiled_model_get_runtime_model(compiled_model, &runtime_model));
    EXPECT_NE(nullptr, runtime_model);

    ov_model_free(runtime_model);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model_test, get_runtime_model_error_handling) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), 0, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    ov_model_t* runtime_model = nullptr;
    OV_EXPECT_NOT_OK(ov_compiled_model_get_runtime_model(nullptr, &runtime_model));
    OV_EXPECT_NOT_OK(ov_compiled_model_get_runtime_model(compiled_model, nullptr));

    ov_model_free(runtime_model);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model_test, create_infer_request) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), 0, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    ov_infer_request_t* infer_request = nullptr;
    OV_EXPECT_OK(ov_compiled_model_create_infer_request(compiled_model, &infer_request));
    EXPECT_NE(nullptr, infer_request);

    ov_infer_request_free(infer_request);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model_test, create_infer_request_error_handling) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), 0, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    ov_infer_request_t* infer_request = nullptr;
    OV_EXPECT_NOT_OK(ov_compiled_model_create_infer_request(nullptr, &infer_request));
    OV_EXPECT_NOT_OK(ov_compiled_model_create_infer_request(compiled_model, nullptr));

    ov_infer_request_free(infer_request);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

}  // namespace
