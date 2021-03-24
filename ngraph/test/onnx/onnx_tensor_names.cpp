// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "onnx_import/onnx.hpp"
#include "onnx_import/onnx_utils.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";

using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;

NGRAPH_TEST(onnx_tensor_names, simple_model)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/tensor_names.prototxt"));

    auto ops = function->get_ordered_ops();
    ASSERT_EQ(ops[0]->get_friendly_name(), "input");
    ASSERT_EQ(ops[0]->get_output_tensor(0).get_names(), std::unordered_set<std::string>{"input"});
    ASSERT_EQ(ops[1]->get_friendly_name(), "relu_t");
    ASSERT_EQ(ops[1]->get_output_tensor(0).get_names(), std::unordered_set<std::string>{"relu_t"});
    // ops[2] is a constant created in the ONNX importer as part of Identity operator
    ASSERT_EQ(ops[3]->get_friendly_name(), "final_output");
    ASSERT_EQ(ops[3]->get_output_tensor(0).get_names(),
              std::unordered_set<std::string>{"final_output"});
    ASSERT_EQ(ops[4]->get_friendly_name(), "final_output");

    ASSERT_EQ(function->get_result()->get_input_tensor(0).get_names(),
              std::unordered_set<std::string>{"final_output"});
    ASSERT_EQ(function->get_result()->input_value(0).get_tensor().get_names(),
              std::unordered_set<std::string>{"final_output"});
}

NGRAPH_TEST(onnx_tensor_names, node_multiple_outputs)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/top_k.prototxt"));

    auto ops = function->get_ordered_ops();

    ASSERT_EQ(ops[0]->get_friendly_name(), "x");
    ASSERT_EQ(ops[0]->get_output_tensor(0).get_names(), std::unordered_set<std::string>{"x"});
    // ops[1] is a constant created in the ONNX importer as part of TopK operator(K value)
    ASSERT_EQ(ops[2]->get_friendly_name(), "indices");
    ASSERT_EQ(ops[2]->get_output_tensor(0).get_names(), std::unordered_set<std::string>{"values"});
    ASSERT_EQ(ops[2]->get_output_tensor(1).get_names(), std::unordered_set<std::string>{"indices"});
    // result nodes are generated in different order than function results.
    ASSERT_EQ(ops[3]->get_friendly_name(), "indices");
    ASSERT_EQ(ops[4]->get_friendly_name(), "values");

    ASSERT_EQ(function->get_results()[0]->get_input_tensor(0).get_names(),
              std::unordered_set<std::string>{"values"});
    ASSERT_EQ(function->get_results()[1]->get_input_tensor(0).get_names(),
              std::unordered_set<std::string>{"indices"});
    ASSERT_EQ(function->get_results()[0]->input_value(0).get_tensor().get_names(),
              std::unordered_set<std::string>{"values"});
    ASSERT_EQ(function->get_results()[1]->input_value(0).get_tensor().get_names(),
              std::unordered_set<std::string>{"indices"});
}
