// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ov_test.hpp"

TEST(ov_model, ov_model_outputs) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_output_node_list_t output_node_list;
    output_node_list.output_nodes = nullptr;
    OV_ASSERT_OK(ov_model_outputs(model, &output_node_list));
    ASSERT_NE(nullptr, output_node_list.output_nodes);

    ov_output_node_list_free(&output_node_list);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_model, ov_model_inputs) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_output_node_list_t input_node_list;
    input_node_list.output_nodes = nullptr;
    OV_ASSERT_OK(ov_model_inputs(model, &input_node_list));
    ASSERT_NE(nullptr, input_node_list.output_nodes);

    ov_output_node_list_free(&input_node_list);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_model, ov_model_input_by_name) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_output_const_node_t* input_node = nullptr;
    OV_ASSERT_OK(ov_model_input_by_name(model, "data", &input_node));
    ASSERT_NE(nullptr, input_node);

    ov_shape_t shape;
    OV_ASSERT_OK(ov_node_get_shape(input_node, &shape));
    ov_shape_free(&shape);

    ov_output_node_free(input_node);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_model, ov_model_input_by_index) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_output_const_node_t* input_node = nullptr;
    OV_ASSERT_OK(ov_model_input_by_index(model, 0, &input_node));
    ASSERT_NE(nullptr, input_node);

    ov_shape_t shape;
    OV_ASSERT_OK(ov_node_get_shape(input_node, &shape));
    ov_shape_free(&shape);

    ov_output_node_free(input_node);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_model, ov_model_is_dynamic) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ASSERT_NO_THROW(ov_model_is_dynamic(model));

    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_model, ov_model_reshape_input_by_name) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_output_node_list_t input_node_list1;
    input_node_list1.output_nodes = nullptr;
    OV_ASSERT_OK(ov_model_inputs(model, &input_node_list1));
    ASSERT_NE(nullptr, input_node_list1.output_nodes);

    char* tensor_name = nullptr;
    OV_ASSERT_OK(ov_node_list_get_any_name_by_index(&input_node_list1, 0, &tensor_name));

    ov_shape_t shape = {0, nullptr};
    int64_t dims[4] = {1, 3, 896, 896};
    OV_ASSERT_OK(ov_shape_create(4, dims, &shape));

    ov_partial_shape_t partial_shape;
    OV_ASSERT_OK(ov_shape_to_partial_shape(shape, &partial_shape));
    OV_ASSERT_OK(ov_model_reshape_input_by_name(model, tensor_name, partial_shape));

    ov_output_node_list_t input_node_list2;
    input_node_list2.output_nodes = nullptr;
    OV_ASSERT_OK(ov_model_inputs(model, &input_node_list2));
    ASSERT_NE(nullptr, input_node_list2.output_nodes);

    EXPECT_NE(input_node_list1.output_nodes, input_node_list2.output_nodes);

    ov_shape_free(&shape);
    ov_partial_shape_free(&partial_shape);
    ov_free(tensor_name);
    ov_output_node_list_free(&input_node_list1);
    ov_output_node_list_free(&input_node_list2);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_model, ov_model_get_friendly_name) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    char* friendly_name = nullptr;
    OV_ASSERT_OK(ov_model_get_friendly_name(model, &friendly_name));
    ASSERT_NE(nullptr, friendly_name);

    ov_free(friendly_name);
    ov_model_free(model);
    ov_core_free(core);
}