// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>

#include "common_test_utils/file_utils.hpp"
#include "editor.hpp"
#include "engines_util/test_case.hpp"
#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "onnx_test_util.hpp"
#include "util/test_control.hpp"

using namespace ov;
using namespace ov::onnx_editor;
using namespace ngraph::test;

static std::string s_manifest = "${MANIFEST}";

NGRAPH_TEST(onnx_editor, topological_sort_two_nodes_swap) {
    ONNXModelEditor editor{ngraph::file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/topological_sort/two_nodes_swap.onnx")};
    ASSERT_NO_THROW(editor.get_function());
}

NGRAPH_TEST(onnx_editor, topological_sort_completely_unsorted) {
    ONNXModelEditor editor{ngraph::file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/topological_sort/completely_unsorted.onnx")};
    editor.get_function();
}

NGRAPH_TEST(onnx_editor, topological_sort_completely_unsorted_2) {
    ONNXModelEditor editor{
        ngraph::file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                     SERIALIZED_ZOO,
                                     "onnx/model_editor/topological_sort/completely_unsorted_2.onnx")};
    ASSERT_NO_THROW(editor.get_function());
}

NGRAPH_TEST(onnx_editor, topological_sort_graph_not_changed_if_the_same_name_of_unsorted_node_and_initializer) {
    const std::string rel_path_to_model =
        "onnx/model_editor/topological_sort/same_name_of_unsorted_node_and_initializer.onnx";
    ONNXModelEditor editor{
        ngraph::file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, rel_path_to_model)};

    // topological sorting is called only via Editor importing
    const auto ref_model =
        ngraph::file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, rel_path_to_model);

    const auto result = compare_onnx_models(editor.model_string(), ref_model);
    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, topological_sort_graph_not_changed_if_empty_input_name) {
    const std::string rel_path_to_model = "onnx/model_editor/topological_sort/empty_input_name.onnx";
    ONNXModelEditor editor{
        ngraph::file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, rel_path_to_model)};

    // topological sorting is called only via Editor importing
    const auto ref_model =
        ngraph::file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, rel_path_to_model);

    const auto result = compare_onnx_models(editor.model_string(), ref_model);
    EXPECT_TRUE(result.is_ok) << result.error_message;
}
