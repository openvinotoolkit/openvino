// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ov_test.hpp"

class ov_model_test : public ov_capi_test_base {
    void SetUp() override {
        ov_capi_test_base::SetUp();
    }

    void TearDown() override {
        ov_capi_test_base::TearDown();
    }
};

INSTANTIATE_TEST_SUITE_P(ov_model, ov_model_test, ::testing::Values(""));

TEST_P(ov_model_test, ov_model_const_input) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    ov_output_const_port_t* input_port = nullptr;
    OV_EXPECT_OK(ov_model_const_input(model, &input_port));
    EXPECT_NE(nullptr, input_port);

    ov_output_const_port_free(input_port);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_model_test, ov_model_const_input_by_name) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    ov_output_const_port_t* input_port = nullptr;
    OV_EXPECT_OK(ov_model_const_input_by_name(model, "data", &input_port));
    EXPECT_NE(nullptr, input_port);

    ov_shape_t shape;
    OV_ASSERT_OK(ov_const_port_get_shape(input_port, &shape));
    ov_shape_free(&shape);

    ov_output_const_port_free(input_port);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_model_test, ov_model_const_input_by_index) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    ov_output_const_port_t* input_port = nullptr;
    OV_EXPECT_OK(ov_model_const_input_by_index(model, 0, &input_port));
    EXPECT_NE(nullptr, input_port);

    ov_shape_t shape;
    OV_ASSERT_OK(ov_const_port_get_shape(input_port, &shape));
    ov_shape_free(&shape);

    ov_output_const_port_free(input_port);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_model_test, ov_model_input) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    ov_output_port_t* input_port = nullptr;
    OV_EXPECT_OK(ov_model_input(model, &input_port));
    EXPECT_NE(nullptr, input_port);

    ov_output_port_free(input_port);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_model_test, ov_model_input_by_name) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    ov_output_port_t* input_port = nullptr;
    OV_EXPECT_OK(ov_model_input_by_name(model, "data", &input_port));
    EXPECT_NE(nullptr, input_port);

    ov_shape_t shape;
    OV_ASSERT_OK(ov_port_get_shape(input_port, &shape));
    ov_shape_free(&shape);

    ov_output_port_free(input_port);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_model_test, ov_model_input_by_index) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    ov_output_port_t* input_port = nullptr;
    OV_EXPECT_OK(ov_model_input_by_index(model, 0, &input_port));
    EXPECT_NE(nullptr, input_port);

    ov_shape_t shape;
    OV_ASSERT_OK(ov_port_get_shape(input_port, &shape));
    ov_shape_free(&shape);

    ov_output_port_free(input_port);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_model_test, ov_model_const_output) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    ov_output_const_port_t* output_port = nullptr;
    OV_EXPECT_OK(ov_model_const_output(model, &output_port));
    EXPECT_NE(nullptr, output_port);

    ov_output_const_port_free(output_port);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_model_test, ov_model_const_output_by_index) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    ov_output_const_port_t* output_port = nullptr;
    OV_EXPECT_OK(ov_model_const_output_by_index(model, 0, &output_port));
    EXPECT_NE(nullptr, output_port);

    ov_shape_t shape;
    OV_ASSERT_OK(ov_const_port_get_shape(output_port, &shape));
    ov_shape_free(&shape);

    ov_output_const_port_free(output_port);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_model_test, ov_model_const_output_by_name) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    ov_output_const_port_t* output_port = nullptr;
    OV_EXPECT_OK(ov_model_const_output_by_name(model, "relu", &output_port));
    EXPECT_NE(nullptr, output_port);

    ov_shape_t shape;
    OV_ASSERT_OK(ov_const_port_get_shape(output_port, &shape));
    ov_shape_free(&shape);

    ov_output_const_port_free(output_port);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_model_test, ov_model_output) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    ov_output_port_t* output_port = nullptr;
    OV_EXPECT_OK(ov_model_output(model, &output_port));
    EXPECT_NE(nullptr, output_port);

    ov_output_port_free(output_port);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_model_test, ov_model_output_by_index) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    ov_output_port_t* output_port = nullptr;
    OV_EXPECT_OK(ov_model_output_by_index(model, 0, &output_port));
    EXPECT_NE(nullptr, output_port);

    ov_shape_t shape;
    OV_ASSERT_OK(ov_port_get_shape(output_port, &shape));
    ov_shape_free(&shape);

    ov_output_port_free(output_port);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_model_test, ov_model_output_by_name) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    ov_output_port_t* output_port = nullptr;
    OV_EXPECT_OK(ov_model_output_by_name(model, "relu", &output_port));
    EXPECT_NE(nullptr, output_port);

    ov_shape_t shape;
    OV_ASSERT_OK(ov_port_get_shape(output_port, &shape));
    ov_shape_free(&shape);

    ov_output_port_free(output_port);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_model_test, ov_model_inputs_size) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    size_t input_size;
    OV_EXPECT_OK(ov_model_inputs_size(model, &input_size));
    EXPECT_NE(0, input_size);

    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_model_test, ov_model_outputs_size) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    size_t output_size;
    OV_EXPECT_OK(ov_model_outputs_size(model, &output_size));
    EXPECT_NE(0, output_size);

    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_model_test, ov_model_is_dynamic) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    EXPECT_NO_THROW(ov_model_is_dynamic(model));

    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_model_test, ov_model_reshape_input_by_name) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    ov_output_const_port_t* input_port_1 = nullptr;
    OV_EXPECT_OK(ov_model_const_input(model, &input_port_1));
    EXPECT_NE(nullptr, input_port_1);

    char* tensor_name = nullptr;
    OV_EXPECT_OK(ov_port_get_any_name(input_port_1, &tensor_name));

    ov_shape_t shape = {0, nullptr};
    int64_t dims[4] = {1, 3, 896, 896};
    OV_EXPECT_OK(ov_shape_create(4, dims, &shape));

    ov_partial_shape_t partial_shape;
    OV_EXPECT_OK(ov_shape_to_partial_shape(shape, &partial_shape));
    OV_EXPECT_OK(ov_model_reshape_input_by_name(model, tensor_name, partial_shape));

    ov_output_const_port_t* input_port_2 = nullptr;
    OV_EXPECT_OK(ov_model_const_input(model, &input_port_2));
    EXPECT_NE(nullptr, input_port_2);

    EXPECT_NE(input_port_1, input_port_2);

    ov_shape_free(&shape);
    ov_partial_shape_free(&partial_shape);
    ov_free(tensor_name);
    ov_output_const_port_free(input_port_1);
    ov_output_const_port_free(input_port_2);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_model_test, ov_model_reshape) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    ov_output_const_port_t* input_port_1 = nullptr;
    OV_EXPECT_OK(ov_model_const_input(model, &input_port_1));
    EXPECT_NE(nullptr, input_port_1);

    char* tensor_name = nullptr;
    OV_EXPECT_OK(ov_port_get_any_name(input_port_1, &tensor_name));
    const char* const_tensor_name = tensor_name;

    ov_shape_t shape = {0, nullptr};
    int64_t dims[4] = {1, 3, 896, 896};
    OV_EXPECT_OK(ov_shape_create(4, dims, &shape));

    ov_partial_shape_t partial_shape;
    OV_EXPECT_OK(ov_shape_to_partial_shape(shape, &partial_shape));
    OV_EXPECT_OK(ov_model_reshape(model, &const_tensor_name, &partial_shape, 1));

    ov_output_const_port_t* input_port_2 = nullptr;
    OV_EXPECT_OK(ov_model_const_input(model, &input_port_2));
    EXPECT_NE(nullptr, input_port_2);

    EXPECT_NE(input_port_1, input_port_2);

    ov_shape_free(&shape);
    ov_partial_shape_free(&partial_shape);
    ov_free(tensor_name);
    ov_output_const_port_free(input_port_1);
    ov_output_const_port_free(input_port_2);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_model_test, ov_model_reshape_by_port_indexes) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    size_t port_indexs[] = {0};

    ov_shape_t shape = {0, nullptr};
    int64_t dims[4] = {1, 3, 896, 896};
    OV_EXPECT_OK(ov_shape_create(4, dims, &shape));

    ov_partial_shape_t partial_shape;
    OV_EXPECT_OK(ov_shape_to_partial_shape(shape, &partial_shape));
    OV_EXPECT_OK(ov_model_reshape_by_port_indexes(model, port_indexs, &partial_shape, 1));

    ov_output_const_port_t* input_port_2 = nullptr;
    OV_EXPECT_OK(ov_model_const_input(model, &input_port_2));
    EXPECT_NE(nullptr, input_port_2);

    ov_shape_free(&shape);
    ov_partial_shape_free(&partial_shape);
    ov_output_const_port_free(input_port_2);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_model_test, ov_model_reshape_single_input) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    ov_shape_t shape = {0, nullptr};
    int64_t dims[4] = {1, 3, 896, 896};
    OV_EXPECT_OK(ov_shape_create(4, dims, &shape));

    ov_partial_shape_t partial_shape;
    OV_EXPECT_OK(ov_shape_to_partial_shape(shape, &partial_shape));
    OV_EXPECT_OK(ov_model_reshape_single_input(model, partial_shape));

    ov_output_const_port_t* input_port_2 = nullptr;
    OV_EXPECT_OK(ov_model_const_input(model, &input_port_2));
    EXPECT_NE(nullptr, input_port_2);

    ov_shape_free(&shape);
    ov_partial_shape_free(&partial_shape);
    ov_output_const_port_free(input_port_2);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_model_test, ov_model_reshape_by_ports) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    ov_output_port_t* input_port_1 = nullptr;
    OV_EXPECT_OK(ov_model_input(model, &input_port_1));
    EXPECT_NE(nullptr, input_port_1);
    const ov_output_port_t* input_ports = input_port_1;

    ov_shape_t shape = {0, nullptr};
    int64_t dims[4] = {1, 3, 896, 896};
    OV_EXPECT_OK(ov_shape_create(4, dims, &shape));

    ov_partial_shape_t partial_shape;
    OV_EXPECT_OK(ov_shape_to_partial_shape(shape, &partial_shape));
    OV_EXPECT_OK(ov_model_reshape_by_ports(model, &input_ports, &partial_shape, 1));

    ov_output_const_port_t* input_port_2 = nullptr;
    OV_EXPECT_OK(ov_model_const_input(model, &input_port_2));
    EXPECT_NE(nullptr, input_port_2);

    ov_shape_free(&shape);
    ov_partial_shape_free(&partial_shape);
    ov_output_port_free(input_port_1);
    ov_output_const_port_free(input_port_2);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_model_test, ov_model_get_friendly_name) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    char* friendly_name = nullptr;
    OV_EXPECT_OK(ov_model_get_friendly_name(model, &friendly_name));
    EXPECT_NE(nullptr, friendly_name);

    ov_free(friendly_name);
    ov_model_free(model);
    ov_core_free(core);
}
