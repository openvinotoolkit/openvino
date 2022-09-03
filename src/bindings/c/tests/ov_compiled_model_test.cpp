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
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), false, &compiled_model));
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
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), false, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    ov_model_t* runtime_model = nullptr;
    OV_EXPECT_NOT_OK(ov_compiled_model_get_runtime_model(nullptr, &runtime_model));
    OV_EXPECT_NOT_OK(ov_compiled_model_get_runtime_model(compiled_model, nullptr));

    ov_model_free(runtime_model);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model, set_and_get_property) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), false, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

#if 0
    // Test enum
    const char* key_1 = ov_property_key_affinity;
    const char* type_1 = ov_property_value_type_enum;
    ov_affinity_e affinity = ov_affinity_e::HYBRID_AWARE;
    OV_EXPECT_OK(ov_compiled_model_set_property(compiled_model, key_1, type_1, affinity));
    char* result_1 = nullptr;
    OV_EXPECT_OK(ov_compiled_model_get_property(compiled_model, key_1, &result_1));
    EXPECT_STREQ(result_1, "HYBRID_AWARE");
    ov_free(result_1);

    // Test uint32
    const char* key_2 = ov_property_key_inference_num_threads;
    const char* type_2 = ov_property_value_type_int32;
    int num = 7;
    OV_EXPECT_OK(ov_compiled_model_set_property(compiled_model, key_2, type_2, num));
    char* result_2 = nullptr;
    OV_EXPECT_OK(ov_compiled_model_get_property(compiled_model, key_2, &result_2));
    EXPECT_STREQ(result_2, "7");
    ov_free(result_2);
#endif

    char* result_3 = nullptr;
    const char* key_3 = ov_property_key_supported_properties;
    OV_EXPECT_OK(ov_compiled_model_get_property(compiled_model, key_3, &result_3));
    ov_free(result_3);

    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model, create_compiled_model_with_property) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    const char* key = ov_property_key_hint_num_requests;
    const char* type = ov_property_value_type_uint32;
    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), true, &compiled_model, key, type, 9));
    EXPECT_NE(nullptr, compiled_model);
    char* result = nullptr;
    OV_EXPECT_OK(ov_compiled_model_get_property(compiled_model, key, &result));
    EXPECT_STREQ(result, "9");
    ov_free(result);

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
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), false, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    ov_output_node_list_t input_nodes;
    input_nodes.output_nodes = nullptr;
    input_nodes.size = 0;
    OV_EXPECT_OK(ov_compiled_model_inputs(compiled_model, &input_nodes));
    EXPECT_NE(nullptr, input_nodes.output_nodes);
    EXPECT_NE(0, input_nodes.size);

    ov_output_node_list_free(&input_nodes);
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
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), false, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    ov_output_node_list_t input_nodes;
    input_nodes.output_nodes = nullptr;
    input_nodes.size = 0;
    OV_EXPECT_NOT_OK(ov_compiled_model_inputs(nullptr, &input_nodes));
    OV_EXPECT_NOT_OK(ov_compiled_model_inputs(compiled_model, nullptr));

    ov_output_node_list_free(&input_nodes);
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
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), false, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    ov_output_node_list_t output_nodes;
    output_nodes.output_nodes = nullptr;
    output_nodes.size = 0;
    OV_EXPECT_OK(ov_compiled_model_outputs(compiled_model, &output_nodes));
    EXPECT_NE(nullptr, output_nodes.output_nodes);
    EXPECT_NE(0, output_nodes.size);

    ov_output_node_list_free(&output_nodes);
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
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), false, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    ov_output_node_list_t output_nodes;
    output_nodes.output_nodes = nullptr;
    output_nodes.size = 0;
    OV_EXPECT_NOT_OK(ov_compiled_model_outputs(nullptr, &output_nodes));
    OV_EXPECT_NOT_OK(ov_compiled_model_outputs(compiled_model, nullptr));

    ov_output_node_list_free(&output_nodes);
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
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), false, &compiled_model));
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
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), false, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    ov_infer_request_t* infer_request = nullptr;
    OV_EXPECT_NOT_OK(ov_compiled_model_create_infer_request(nullptr, &infer_request));
    OV_EXPECT_NOT_OK(ov_compiled_model_create_infer_request(compiled_model, nullptr));

    ov_infer_request_free(infer_request);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}