// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_case.hpp"
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"

using namespace ov;
using namespace ov::frontend;
using namespace ov::frontend::onnx::tests;

static std::string s_manifest = onnx_backend_manifest("${MANIFEST}");

OPENVINO_TEST(onnx_editor, topological_sort_two_nodes_swap) {
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/topological_sort/two_nodes_swap.onnx", &front_end);
    OV_ASSERT_NO_THROW(front_end->convert(input_model));
}

OPENVINO_TEST(onnx_editor, topological_sort_completely_unsorted) {
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/topological_sort/completely_unsorted.onnx", &front_end);
    OV_ASSERT_NO_THROW(front_end->convert(input_model));
}

OPENVINO_TEST(onnx_editor, topological_sort_completely_unsorted_2) {
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/topological_sort/completely_unsorted_2.onnx", &front_end);
    OV_ASSERT_NO_THROW(front_end->convert(input_model));
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
OPENVINO_TEST(onnx_editor, topological_sort_completely_unsorted_2_wstring) {
    FrontEnd::Ptr front_end;
    auto input_model = load_model(L"model_editor/topological_sort/completely_unsorted_2.onnx", &front_end);
    OV_ASSERT_NO_THROW(front_end->convert(input_model));
}
#endif

OPENVINO_TEST(onnx_editor, topological_sort_constant_node_in_the_graph) {
    const std::string rel_path_to_model = "model_editor/topological_sort/add_abc_const_node_unsorted.onnx";
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/topological_sort/completely_unsorted_2.onnx", &front_end);

    OV_ASSERT_NO_THROW(front_end->convert(input_model));
}

OPENVINO_TEST(onnx_editor, topological_sort_multioutput_node) {
    const std::string rel_path_to_model = "model_editor/topological_sort/multioutput_split_unsorted.onnx";
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/topological_sort/completely_unsorted_2.onnx", &front_end);

    OV_ASSERT_NO_THROW(front_end->convert(input_model));
}

/*
// No suitable functionality yet
OPENVINO_TEST(onnx_editor, topological_sort_graph_not_changed_if_the_same_name_of_unsorted_node_and_initializer) {
    const std::string rel_path_to_model =
        "model_editor/topological_sort/same_name_of_unsorted_node_and_initializer.onnx";
    ONNXModelEditor editor{
        ov::util::path_join({ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, rel_path_to_model})};

    // topological sorting is called only via Editor importing
    const auto ref_model =
        ov::util::path_join({ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, rel_path_to_model});

    const auto result = compare_onnx_models(editor.model_string(), ref_model);
    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, topological_sort_graph_not_changed_if_empty_input_name) {
    const std::string rel_path_to_model = "model_editor/topological_sort/empty_input_name.onnx";
    ONNXModelEditor editor{
        ov::util::path_join({ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, rel_path_to_model})};

    // topological sorting is called only via Editor importing
    const auto ref_model =
        ov::util::path_join({ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, rel_path_to_model});

    const auto result = compare_onnx_models(editor.model_string(), ref_model);
    EXPECT_TRUE(result.is_ok) << result.error_message;
}
*/