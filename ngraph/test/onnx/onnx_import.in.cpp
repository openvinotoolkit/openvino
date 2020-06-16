//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif
#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include "gtest/gtest.h"
#include "ngraph/frontend/onnx_import/onnx.hpp"
#include "ngraph/frontend/onnx_import/onnx_utils.hpp"
#include "ngraph/frontend/onnx_import/default_opset.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"
#include "util/type_prop.hpp"

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";

using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;

// ############################################################################ CORE TESTS
NGRAPH_TEST(${BACKEND_NAME}, onnx_test_test_case)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({1});
    test_case.add_input<float>({2});
    test_case.add_input<float>({3});
    test_case.add_expected_output<float>(Shape{1}, {6});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_test_test_case_mutliple_inputs)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(Inputs{{1}, {2}, {3}});
    test_case.add_expected_output(Shape{1}, std::vector<float>{6});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_output_names_check)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/split_equal_parts_default.prototxt"));

    std::size_t size = function->get_output_size();
    for (std::size_t i{0}; i < size; ++i)
    {
        std::shared_ptr<Node> node = function->get_output_op(i);
        EXPECT_EQ(node->get_friendly_name(), "output_" + std::to_string(i + 1));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_node_names_check)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc.prototxt"));

    // Filter out Add nodes from the function graph
    std::vector<std::shared_ptr<Node>> additions;
    auto ordered_ops = function->get_ordered_ops();
    std::copy_if(
        ordered_ops.begin(),
        ordered_ops.end(),
        std::back_inserter(additions),
        [](std::shared_ptr<Node> op) { return std::string(op->get_type_name()) == "Add"; });

    EXPECT_EQ(additions.size(), 2);
    EXPECT_EQ(additions.at(0)->get_friendly_name(), "X");
    EXPECT_EQ(additions.at(1)->get_friendly_name(), "Y");
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_add_abc)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(Inputs{{1}, {2}, {3}});
    test_case.add_expected_output(Shape{1}, std::vector<float>{6});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_binary_add_abc)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc.onnx"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(Inputs{{1}, {2}, {3}});
    test_case.add_expected_output(Shape{1}, std::vector<float>{6});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_bool_const_op)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/bool_const_op.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_expected_output(std::vector<bool>{1, 0, 0, 1});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_bool_init_and)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/bool_init_and.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_expected_output(std::vector<bool>{1});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_bool_input_or)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/bool_input_or.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input(std::vector<bool>{true, false, true, false});
    test_case.add_input(std::vector<bool>{false, false, true, true});
    test_case.add_expected_output(std::vector<bool>{1, 0, 1, 1});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_bool_init_raw)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/bool_init_raw.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_expected_output(std::vector<bool>{true, false, true});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_add_abc_initializers)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc_initializers.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({1, 2, 3, 4});
    test_case.add_expected_output<float>({3, 6, 9, 12});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_override_op)
{
    onnx_import::register_operator(
        "FalseAdd", 1, "", [](const onnx_import::Node& node) -> NodeVector {
            NodeVector ng_inputs{node.get_ng_inputs()};
            return {std::make_shared<ngraph::op::Add>(ng_inputs.at(0), ng_inputs.at(1))};
        });

    onnx_import::register_operator(
        "FalseAdd", 1, "", [](const onnx_import::Node& node) -> NodeVector {
            NodeVector ng_inputs{node.get_ng_inputs()};
            return {std::make_shared<ngraph::op::Subtract>(ng_inputs.at(0), ng_inputs.at(1))};
        });

    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/override_op.prototxt"));

    Inputs inputs;
    inputs.emplace_back(std::vector<float>{0.f, 1.f, 2.f, 3.f});
    inputs.emplace_back(std::vector<float>{3.f, 2.f, 1.f, 0.f});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output<float>({-3.f, -1.f, 1.f, 3.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_import_non_existing_file)
{
    try
    {
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/i.dont.exist"));
    }
    catch (const std::runtime_error& exc)
    {
        // asserts that an exception was thrown and that the error message contains the file name
        std::string msg{exc.what()};
        EXPECT_TRUE(msg.find("i.dont.exist") != std::string::npos);
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_unsupported_op)
{
    try
    {
        onnx_import::import_onnx_model(
            file_util::path_join(SERIALIZED_ZOO, "onnx/unsupported_op.prototxt"));
        FAIL() << "Expected ngraph::ngraph_error";
    }
    catch (ngraph::ngraph_error const& err)
    {
        std::string what{err.what()};
        EXPECT_NE(what.find("nGraph does not support"), std::string::npos);
        EXPECT_NE(what.find("FakeOpName"), std::string::npos);
        EXPECT_NE(what.find("AnotherFakeOpName"), std::string::npos);
    }
    catch (...)
    {
        FAIL() << "Expected ngraph::ngraph_error";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_custom_op)
{
    onnx_import::register_operator(
        "AddQ", 1, "com.intel.ai", [](const onnx_import::Node& node) -> NodeVector {
            NodeVector ng_inputs{node.get_ng_inputs()};
            return {std::make_shared<ngraph::op::Add>(ng_inputs.at(0), ng_inputs.at(1))};
        });

    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/custom_operator.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>({3.f, 6.f, 9.f, 12.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_custom_op_default_domain)
{
    onnx_import::register_operator(
        "AddQ", 1, "com.intel.ai", [](const onnx_import::Node& node) -> NodeVector {
            NodeVector ng_inputs{node.get_ng_inputs()};
            return {std::make_shared<ngraph::op::Add>(ng_inputs.at(0), ng_inputs.at(1))};
        });

    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/custom_operator_default_domain.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>({3.f, 6.f, 9.f, 12.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_is_op_supported)
{
    // Simple case
    EXPECT_TRUE(onnx_import::is_operator_supported("Sum", 1, "ai.onnx"));
    // With fallback
    EXPECT_TRUE(onnx_import::is_operator_supported("Sum", 100, "ai.onnx"));

    // Different opset versions
    EXPECT_TRUE(onnx_import::is_operator_supported("Add", 1, "ai.onnx"));
    EXPECT_TRUE(onnx_import::is_operator_supported("Add", 7, "ai.onnx"));

    // Default domain name
    EXPECT_TRUE(onnx_import::is_operator_supported("Sum", 1));

    // Unregistered operator
    EXPECT_FALSE(onnx_import::is_operator_supported("DummyOp", 1));
    EXPECT_FALSE(onnx_import::is_operator_supported("DummyOp", 1, "ai.onnx"));
    EXPECT_FALSE(onnx_import::is_operator_supported("DummyOp", 10, "ai.onnx"));

    // Operator with bad domain name
    EXPECT_FALSE(onnx_import::is_operator_supported("Sum", 1, "bad.domain"));

    // Registered custom operator
    onnx_import::register_operator(
        "AddQ", 1, "com.intel.ai", [](const onnx_import::Node& node) -> NodeVector {
            NodeVector ng_inputs{node.get_ng_inputs()};
            return {std::make_shared<ngraph::op::Add>(ng_inputs.at(0), ng_inputs.at(1))};
        });
    EXPECT_TRUE(onnx_import::is_operator_supported("AddQ", 1, "com.intel.ai"));
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_missing_op_domain)
{
    onnx_import::register_operator(
        "CustomAdd", 1, "custom.op", [](const onnx_import::Node& node) -> NodeVector {
            NodeVector ng_inputs{node.get_ng_inputs()};
            return {std::make_shared<ngraph::op::Add>(ng_inputs.at(0), ng_inputs.at(1))};
        });

    EXPECT_TRUE(onnx_import::is_operator_supported("CustomAdd", 1, "custom.op"));

    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/missing_op_domain.prototxt"));

    Inputs inputs;
    inputs.emplace_back(std::vector<float>{0.f, 1.f, 2.f, 3.f});
    inputs.emplace_back(std::vector<float>{0.f, 1.f, 2.f, 3.f});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output<float>({0.f, 2.f, 4.f, 6.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_unknown_domain)
{
    // the importer should not throw when it encounters an unknown domain in the model
    EXPECT_NO_THROW(onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/unknown_domain.prototxt")));
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_op_in_unknown_domain)
{
    try
    {
        onnx_import::import_onnx_model(
            file_util::path_join(SERIALIZED_ZOO, "onnx/unknown_domain_add.prototxt"));

        FAIL() << "The onnx_importer did not throw for unknown domain and op";
    }
    catch (const ngraph::ngraph_error& e)
    {
        const std::string msg = e.what();

        EXPECT_NE(msg.find("unknown.domain.Add"), std::string::npos)
            << "The error message should contain domain and op name: unknown.domain.Add";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_missing_input)
{
    onnx_import::register_operator(
        "TestMissingInOut", 1, "com.intel.ai", [](const onnx_import::Node& node) -> NodeVector {
            NodeVector ng_inputs{node.get_ng_inputs()};
            std::shared_ptr<ngraph::Node> A = ng_inputs.at(0);
            std::shared_ptr<ngraph::Node> B = ng_inputs.at(1);
            std::shared_ptr<ngraph::Node> C = ng_inputs.at(2);

            A = A * C;
            if (!B->is_null())
            {
                B = B / C;
            }

            C = C + C;
            return {A, B, C};
        });

    onnx_import::register_operator(
        "TestMissingIn", 1, "com.intel.ai", [](const onnx_import::Node& node) -> NodeVector {
            NodeVector ng_inputs{node.get_ng_inputs()};
            std::shared_ptr<ngraph::Node> result = std::make_shared<ngraph::op::Constant>(
                element::f32, ngraph::Shape{2, 2}, std::vector<float>{1, 1, 1, 1});

            for (const auto& ng_input : ng_inputs)
            {
                if (!ng_input->is_null())
                {
                    result = ng_input * result;
                }
            }

            return {result};
        });

    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/missing_input.prototxt"));

    Inputs inputs{{1, 2, 3, 4}, {5, 6, 7, 8}};

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output<float>({50, 144, 294, 512});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_initializer_wo_input)
{
    // This test checks a model which has an initializer, but no input with the same name
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/initializer_wo_input.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({0, 1, 2, 3, 4, 5});
    test_case.add_expected_output<float>({0, 2, 6, 12, 20, 30});
    test_case.run();
}

// ############################################################################ OPERATOR TESTS
NGRAPH_TEST(${BACKEND_NAME}, onnx_model_addmul_abc)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/addmul_abc.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({9, 10, 11, 12});
    test_case.add_input<float>({5, 6, 7, 8});
    test_case.add_input<float>({1, 2, 3, 4});
    test_case.add_expected_output<float>(Shape{1, 2, 2}, {46, 62, 80, 100});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_argmin_no_keepdims)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/argmin_no_keepdims.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({2, 1, 3, 10});
    test_case.add_expected_output<float>(Shape{2}, {1, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_batch_norm_default)
{
    // Batch Normalization with default parameters
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/batchnorm_default.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({-1.f, 0.f, 1.f, 2.f, 3.f, 4.f}); // data {1, 2, 1, 3}
    test_case.add_input<float>({1.f, 1.5f});                     // scale
    test_case.add_input<float>({0.f, 1.f});                      // bias
    test_case.add_input<float>({0.f, 3.f});                      // mean
    test_case.add_input<float>({1.f, 1.5f});                     // var
    test_case.add_expected_output<float>(
        Shape{1, 2, 1, 3}, {-0.999995f, 0.f, 0.999995f, -0.22474074f, 1.f, 2.2247407f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_relu)
{
    // Simple ReLU test
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/relu.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({-1, -2, 0, 1, 2, 3});
    test_case.add_expected_output<float>({0, 0, 0, 1, 2, 3});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_sum_opset1)
{
    // Simple Sum test for opset1.
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/sum_opset1.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({3.f, 0.f, 2.f});
    test_case.add_input<float>({1.f, 3.f, 4.f});
    test_case.add_input<float>({2.f, 6.f, 6.f});
    test_case.add_expected_output<float>(Shape{3}, {6.f, 9.f, 12.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_sum)
{
    // Simple Sum test for opset8.
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/sum.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({3.f});
    test_case.add_input<float>({1.f, 3.f, 4.f});
    test_case.add_input<float>({2.f, 6.f, 6.f});
    test_case.add_expected_output<float>(Shape{3}, {6.f, 12.f, 13.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_sum_one_input)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/sum_one_input.prototxt"));

    // input data shape (3, )
    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({3.f, 0.f, 2.f});
    test_case.add_expected_output<float>({3.f, 0.f, 2.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_cum_sum_1d)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/cum_sum_1d.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({1.f, 2.f, 3.f});
    test_case.add_expected_output<float>(Shape{3}, {1.f, 3.f, 6.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_cum_sum_2d_axis_input)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/cum_sum_2d_axis_input.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    test_case.add_expected_output<float>(Shape{2, 3}, {1.f, 3.f, 6.f, 4.f, 9.f, 15.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_cum_sum_2d_dynamic_axis_input)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/cum_sum_2d_dynamic_axis_input.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    test_case.add_input<std::int32_t>({1});
    test_case.add_expected_output<float>(Shape{2, 3}, {1.f, 3.f, 6.f, 4.f, 9.f, 15.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_cum_sum_3d_exclusive_reverse)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/cum_sum_3d_exclusive_reverse.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({1.f,  2.f,  3.f,  4.f,  5.f,  6.f,  7.f,  8.f,
                                9.f,  10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f,
                                17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f});
    test_case.add_expected_output<float>(
        Shape{2, 3, 4}, {13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f,
                         0.f,  0.f,  0.f,  0.f,  0.f,  0.f,  0.f,  0.f,  0.f,  0.f,  0.f,  0.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_min_two_inputs_opset1)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/min_two_inputs_opset1.prototxt"));

    // input data shape (3, )
    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({1.f, 2.f, 1.f});
    test_case.add_input<float>({1.f, 4.f, 4.f});
    test_case.add_expected_output<float>({1.f, 2.f, 1.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_min_two_inputs)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/min_two_inputs.prototxt"));

    // input data shape (3, )
    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({2.f});
    test_case.add_input<float>({1.f, 4.f, 4.f});
    test_case.add_expected_output<float>({1.f, 2.f, 2.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_max_opset1)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/max_opset1.prototxt"));

    // input data shape (3, )
    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({3.f, 2.f, 1.f});
    test_case.add_input<float>({1.f, 4.f, 4.f});
    test_case.add_input<float>({2.f, 5.f, 3.f});

    test_case.add_expected_output<float>({3.f, 5.f, 4.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_max)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/max.prototxt"));

    // input data shape (3, )
    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({1.f, 4.f, 4.f});
    test_case.add_input<float>({3.f});
    test_case.add_input<float>({2.f, 5.f, 3.f});

    test_case.add_expected_output<float>({3.f, 5.f, 4.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_mean_opset1)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/mean_opset1.prototxt"));

    // input data shape (3, )
    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({3.f, 0.f, 2.f});
    test_case.add_input<float>({1.f, 3.f, 4.f});
    test_case.add_input<float>({2.f, 6.f, 6.f});

    test_case.add_expected_output<float>({2.f, 3.f, 4.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_mean)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/mean.prototxt"));

    // input data shape (3, )
    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({3.f});
    test_case.add_input<float>({1.f, 2.f, 5.f});
    test_case.add_input<float>({2.f, 7.f, 7.f});

    test_case.add_expected_output<float>({2.f, 4.f, 5.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_gemm_abc)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/gemm_abc.prototxt"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 2>(
                            {{1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12}, {13, 14, 15, 16, 17, 18}})
                            .get_vector());

    inputs.emplace_back(test::NDArray<float, 2>({{19, 20, 21, 22},
                                                 {23, 24, 25, 26},
                                                 {27, 28, 29, 30},
                                                 {31, 32, 33, 34},
                                                 {35, 36, 37, 38},
                                                 {39, 40, 41, 42}})
                            .get_vector());

    inputs.emplace_back(
        test::NDArray<float, 2>({{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}).get_vector());

    auto expected_output =
        test::NDArray<float, 2>(
            {{340, 350.5, 361, 371.5}, {862, 890.5, 919, 947.5}, {1384, 1430.5, 1477, 1523.5}})
            .get_vector();

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_matmul)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/matmul.prototxt"));

    std::vector<std::vector<float>> inputs;

    inputs.emplace_back(
        test::NDArray<float, 2>({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}).get_vector());

    inputs.emplace_back(
        test::NDArray<float, 2>({{13, 14, 15}, {16, 17, 18}, {19, 20, 21}, {22, 23, 24}})
            .get_vector());

    auto expected_output =
        test::NDArray<float, 2>({{190, 200, 210}, {470, 496, 522}, {750, 792, 834}}).get_vector();

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_softmax_0D)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/softmax_0D.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({3.141592});
    test_case.add_expected_output<float>({1.0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_softmax_1D)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/softmax_1D.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({-1.0, 0.0, 1.0});
    test_case.add_expected_output<float>({0.09003058, 0.24472848, 0.66524094});
    test_case.run();
}
namespace
{
    // common input for all Softmax 3D test cases (Shape = {3,4,5})
    const std::vector<float> SOFTMAX_INPUT = {
        2.75793882,  -0.50841322, 0.82013929,  -0.62409912, -0.96136118, 0.21004745,  1.38337255,
        1.19030397,  2.0940445,   -0.03551657, -0.78686039, 1.992782,    0.04300319,  -0.29230777,
        -0.56797112, -1.26732165, -0.61935399, 0.57670432,  0.92844898,  2.82469233,

        0.98721677,  -0.05100663, -1.21178917, -0.17530157, 1.40051805,  -0.13259761, -1.14313018,
        0.2673723,   -0.87996154, 1.29053106,  1.55,        0.8396538,   1.20729817,  0.23727845,
        -0.89113606, -1.70909842, 0.26460363,  -0.70566808, 2.383518,    1.07024615,

        -1.21722605, 0.82919357,  0.55765697,  0.12657686,  0.63432172,  0.75425957,  -2.43721014,
        -1.24478184, 2.65316853,  1.19509542,  -0.95523998, 0.5149006,   -0.01151649, 0.68327026,
        -0.4589638,  -0.46554745, 0.21055324,  0.39266729,  2.05098086,  1.83207919};
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_softmax_axis_0)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/softmax_axis_0.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>(SOFTMAX_INPUT);

    test_case.add_expected_output<float>(
        {0.09683057, 0.00369363, 0.01394559, 0.00329012, 0.00234823, 0.00757665, 0.02449322,
         0.02019284, 0.04985249, 0.00592694, 0.00279593, 0.04505148, 0.00641108, 0.00458466,
         0.00348007, 0.00172928, 0.00330577, 0.01093237, 0.01554086, 0.10351497,

         0.01648154, 0.00583583, 0.00182802, 0.00515374, 0.02491679, 0.00537859, 0.00195794,
         0.00802367, 0.00254737, 0.0223216,  0.02893419, 0.0142204,  0.02053893, 0.00778581,
         0.00251907, 0.00111174, 0.00800149, 0.0030324,  0.06658917, 0.0179084,

         0.00181811, 0.01407243, 0.01072611, 0.0069699,  0.01158077, 0.01305647, 0.00053677,
         0.0017687,  0.08719896, 0.02028982, 0.00236265, 0.01027717, 0.0060709,  0.01216173,
         0.00388087, 0.00385541, 0.00758048, 0.00909469, 0.04775123, 0.03836337});

    test_case.run(6);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_softmax_axis_1)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/softmax_axis_1.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>(SOFTMAX_INPUT);

    test_case.add_expected_output<float>(
        {0.22757064, 0.00868076, 0.03277484, 0.00773243, 0.0055188,  0.0178066,  0.05756383,
         0.04745709, 0.11716303, 0.01392945, 0.00657097, 0.10587974, 0.01506727, 0.01077484,
         0.00817884, 0.00406413, 0.00776921, 0.0256932,  0.03652405, 0.24328028,

         0.06217413, 0.02201481, 0.00689594, 0.01944171, 0.09399488, 0.02028993, 0.00738604,
         0.03026811, 0.00960958, 0.08420492, 0.10914991, 0.05364435, 0.07748005, 0.02937079,
         0.0095028,  0.00419387, 0.03018442, 0.01143929, 0.2511977,  0.06755678,

         0.00587593, 0.04548053, 0.0346656,  0.02252594, 0.03742775, 0.04219705, 0.00173478,
         0.00571623, 0.2818174,  0.06557446, 0.00763582, 0.03321466, 0.01962049, 0.03930537,
         0.01254255, 0.01246025, 0.02449929, 0.02939305, 0.15432668, 0.12398617});

    test_case.run(4);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_softmax_invalid_axis_1D)
{
    ASSERT_THROW(onnx_import::import_onnx_model(
                     file_util::path_join(SERIALIZED_ZOO, "onnx/softmax_invalid_axis_1D.prototxt")),
                 ngraph::ngraph_error)
        << "Softmax model with invalid axis was successfully imported while it should have thrown.";
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_softmax_invalid_axis_3D)
{
    ASSERT_THROW(onnx_import::import_onnx_model(
                     file_util::path_join(SERIALIZED_ZOO, "onnx/softmax_invalid_axis_3D.prototxt")),
                 ngraph::ngraph_error)
        << "Softmax model with invalid axis was successfully imported while it should have thrown.";
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_sub)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/sub.prototxt"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 3>({{{1, 2, 3}}}).get_vector());

    inputs.emplace_back(test::NDArray<float, 3>({{{4, 5, 7}}}).get_vector());

    auto expected_output = test::NDArray<float, 3>({{{-3, -3, -4}}}).get_vector();

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_div)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/div.prototxt"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 3>({{{1, 2, 3}}}).get_vector());
    inputs.emplace_back(test::NDArray<float, 3>({{{1, 4, 12}}}).get_vector());

    auto expected_output = test::NDArray<float, 3>({{{1, 0.5, 0.25}}}).get_vector();

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_add_bcast)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/add_bcast.prototxt"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 3>(
                            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                             {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                             {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
                            .get_vector());

    inputs.emplace_back(test::NDArray<float, 1>({1, 2, 3, 4, 5}).get_vector());

    auto expected_output =
        test::NDArray<float, 4>(
            {{{{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}},
              {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}},
              {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}}}})
            .get_vector();

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_log_sum)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_log_sum.prototxt"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}})
            .get_vector()};

    // output data shape (1,)
    auto expected_output = test::NDArray<float, 4>({{{{2.77258872f}}}}).get_vector();

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_log_sum_exp)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_log_sum_exp.prototxt"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}})
            .get_vector()};

    // output data shape (1,)
    auto expected_output = test::NDArray<float, 4>({{{{3.77258872f}}}}).get_vector();

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_l1)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_l1.prototxt"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}})
            .get_vector()};

    // output data shape (1,)
    auto expected_output = test::NDArray<float, 4>({{{{16}}}}).get_vector();

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_l2)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_l2.prototxt"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}})
            .get_vector()};

    // output data shape (1,)
    auto expected_output = test::NDArray<float, 4>({{{{4}}}}).get_vector();

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_max)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_max.prototxt"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}}})
            .get_vector()};

    // output data shape (1,)
    auto expected_output = test::NDArray<float, 4>({{{{16}}}}).get_vector();

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_mean)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_mean.prototxt"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}})
            .get_vector()};

    // output data shape (1,)
    auto expected_output = test::NDArray<float, 4>({{{{1}}}}).get_vector();

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(Shape{}, expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_min)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_min.prototxt"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}}})
            .get_vector()};

    // output data shape (1,)
    auto expected_output = test::NDArray<float, 4>({{{{1}}}}).get_vector();

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_prod)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_prod.prototxt"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}})
            .get_vector()};

    // output data shape (1,)
    auto expected_output = test::NDArray<float, 4>({{{{1}}}}).get_vector();

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_sum)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_sum.prototxt"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}})
            .get_vector()};

    // output data shape (1,)
    auto expected_output = test::NDArray<float, 4>({{{{16}}}}).get_vector();

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_square)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_sum_square.prototxt"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}})
            .get_vector()};

    // output data shape (1,)
    auto expected_output = test::NDArray<float, 4>({{{{16}}}}).get_vector();

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_resize_opset10_import_only)
{
    const auto resize_fn = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize_opset10.prototxt"));

    // Input data shape (1, 2, 3, 4)
    // Scales input constant values {4, 3, 2, 1}

    Shape expected_output_shape{4, 6, 6, 4};
    EXPECT_EQ(resize_fn->get_output_size(), 1);
    EXPECT_EQ(resize_fn->get_output_shape(0), expected_output_shape);
    EXPECT_EQ(count_ops_of_type<onnx_import::default_opset::Interpolate>(resize_fn), 1);
    EXPECT_EQ(count_ops_of_type<onnx_import::default_opset::Constant>(resize_fn), 1);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_shape)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/shape.prototxt"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 3>(
                            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                             {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                             {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
                            .get_vector());

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output<int64_t>({3, 4, 5});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_elu)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/elu.prototxt"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 3>(
            {{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
             {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
             {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    auto expected_output = test::NDArray<float, 3>({{{-1.999753180391830f,
                                                      -1.999329074744190f,
                                                      -1.998176236068890f,
                                                      -1.995042495646670f,
                                                      -1.986524106001830f},
                                                     {-1.963368722222530f,
                                                      -1.900425863264270f,
                                                      -1.729329433526770f,
                                                      -1.264241117657120f,
                                                      0},
                                                     {1, 2, 3, 4, 5},
                                                     {6, 7, 8, 9, 10}},
                                                    {{-1.963368722222530f,
                                                      -1.900425863264270f,
                                                      -1.729329433526770f,
                                                      -1.264241117657120f,
                                                      0},
                                                     {1, 2, 3, 4, 5},
                                                     {6, 7, 8, 9, 10},
                                                     {11, 12, 13, 14, 15}},
                                                    {{1, 1, 1, 1, 1},
                                                     {-1.264241117657120f,
                                                      -1.264241117657120f,
                                                      -1.264241117657120f,
                                                      -1.264241117657120f,
                                                      -1.264241117657120f},
                                                     {0, 0, 0, 0, 0},
                                                     {2, 2, 2, 2, 2}}})
                               .get_vector();

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_leaky_relu)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/leaky_relu.prototxt"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 3>(
            {{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
             {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
             {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    auto expected_output = test::NDArray<float, 3>({{{-0.9f, -0.8f, -0.7f, -0.6f, -0.5f},
                                                     {-0.4f, -0.3f, -0.2f, -0.1f, 0},
                                                     {1, 2, 3, 4, 5},
                                                     {6, 7, 8, 9, 10}},
                                                    {{-0.4f, -0.3f, -0.2f, -0.1f, 0},
                                                     {1, 2, 3, 4, 5},
                                                     {6, 7, 8, 9, 10},
                                                     {11, 12, 13, 14, 15}},
                                                    {{1, 1, 1, 1, 1},
                                                     {-0.1f, -0.1f, -0.1f, -0.1f, -0.1f},
                                                     {0, 0, 0, 0, 0},
                                                     {2, 2, 2, 2, 2}}})
                               .get_vector();

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_prelu)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/prelu.prototxt"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 3>(
            {{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
             {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
             {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    inputs.emplace_back(test::NDArray<float, 3>(
                            {{{1, 0, 1, 0, 1}, {0, 1, 0, 1, 0}, {1, 0, 1, 0, 1}, {0, 1, 0, 1, 0}},
                             {{0, 1, 0, 1, 0}, {1, 0, 1, 0, 1}, {0, 1, 0, 1, 0}, {1, 0, 1, 0, 1}},
                             {{1, 0, 1, 0, 1}, {0, 1, 0, 1, 0}, {1, 0, 1, 0, 1}, {0, 1, 0, 1, 0}}})
                            .get_vector());

    auto expected_output =
        test::NDArray<float, 3>(
            {{{-9, 0, -7, 0, -5}, {0, -3, 0, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
             {{0, -3, 0, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
             {{1, 1, 1, 1, 1}, {0, -1, 0, -1, 0}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector();

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_selu)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/selu.prototxt"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 3>(
            {{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
             {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
             {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    auto expected_output =
        test::NDArray<float, 3>(
            {{{-5.99925954117548f,
               -5.99798722423258f,
               -5.99452870820667f,
               -5.98512748694000f,
               -5.95957231800549f},
              {-5.89010616666759f, -5.70127758979282f, -5.18798830058032f, -3.79272335297135f, 0},
              {3, 6, 9, 12, 15},
              {18, 21, 24, 27, 30}},
             {{-5.89010616666759f, -5.70127758979282f, -5.18798830058032f, -3.79272335297135f, 0},
              {3, 6, 9, 12, 15},
              {18, 21, 24, 27, 30},
              {33, 36, 39, 42, 45}},
             {{3, 3, 3, 3, 3},
              {-3.79272335297135f,
               -3.79272335297135f,
               -3.79272335297135f,
               -3.79272335297135f,
               -3.79272335297135f},
              {0, 0, 0, 0, 0},
              {6, 6, 6, 6, 6}}})
            .get_vector();

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_sigmoid)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/sigmoid.prototxt"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 3>(
            {{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
             {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
             {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    auto expected_output = test::NDArray<float, 3>({{{0.00012339457598623f,
                                                      0.00033535013046648f,
                                                      0.00091105119440065f,
                                                      0.00247262315663477f,
                                                      0.00669285092428486f},
                                                     {0.01798620996209160f,
                                                      0.04742587317756680f,
                                                      0.119202922022118f,
                                                      0.268941421369995f,
                                                      0.5f},
                                                     {0.731058578630005f,
                                                      0.880797077977882f,
                                                      0.952574126822433f,
                                                      0.982013790037908f,
                                                      0.993307149075715f},
                                                     {0.997527376843365f,
                                                      0.999088948805599f,
                                                      0.999664649869534f,
                                                      0.999876605424014f,
                                                      0.999954602131298f}},
                                                    {{0.01798620996209160f,
                                                      0.04742587317756680f,
                                                      0.119202922022118f,
                                                      0.268941421369995f,
                                                      0.5f},
                                                     {0.731058578630005f,
                                                      0.880797077977882f,
                                                      0.952574126822433f,
                                                      0.982013790037908f,
                                                      0.993307149075715f},
                                                     {0.997527376843365f,
                                                      0.999088948805599f,
                                                      0.999664649869534f,
                                                      0.999876605424014f,
                                                      0.999954602131298f},
                                                     {0.999983298578152f,
                                                      0.999993855825398f,
                                                      0.999997739675702f,
                                                      0.999999168471972f,
                                                      0.999999694097773f}},
                                                    {{0.731058578630005f,
                                                      0.731058578630005f,
                                                      0.731058578630005f,
                                                      0.731058578630005f,
                                                      0.731058578630005f},
                                                     {0.268941421369995f,
                                                      0.268941421369995f,
                                                      0.268941421369995f,
                                                      0.268941421369995f,
                                                      0.268941421369995f},
                                                     {0.5f, 0.5f, 0.5f, 0.5f, 0.5f},
                                                     {0.880797077977882f,
                                                      0.880797077977882f,
                                                      0.880797077977882f,
                                                      0.880797077977882f,
                                                      0.880797077977882f}}})
                               .get_vector();

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_tanh)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/tanh.prototxt"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 3>(
            {{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
             {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
             {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    auto expected_output = test::NDArray<float, 3>({{{-0.999999969540041f,
                                                      -0.999999774929676f,
                                                      -0.999998336943945f,
                                                      -0.999987711650796f,
                                                      -0.999909204262595f},
                                                     {-0.999329299739067f,
                                                      -0.995054753686731f,
                                                      -0.964027580075817f,
                                                      -0.761594155955765f,
                                                      0},
                                                     {0.761594155955765f,
                                                      0.964027580075817f,
                                                      0.995054753686731f,
                                                      0.999329299739067f,
                                                      0.999909204262595f},
                                                     {0.999987711650796f,
                                                      0.999998336943945f,
                                                      0.999999774929676f,
                                                      0.999999969540041f,
                                                      0.999999995877693f}},
                                                    {{-0.999329299739067f,
                                                      -0.995054753686731f,
                                                      -0.964027580075817f,
                                                      -0.761594155955765f,
                                                      0},
                                                     {0.761594155955765f,
                                                      0.964027580075817f,
                                                      0.995054753686731f,
                                                      0.999329299739067f,
                                                      0.999909204262595f},
                                                     {0.999987711650796f,
                                                      0.999998336943945f,
                                                      0.999999774929676f,
                                                      0.999999969540041f,
                                                      0.999999995877693f},
                                                     {0.999999999442106f,
                                                      0.999999999924497f,
                                                      0.999999999989782f,
                                                      0.999999999998617f,
                                                      0.999999999999813f}},
                                                    {{0.761594155955765f,
                                                      0.761594155955765f,
                                                      0.761594155955765f,
                                                      0.761594155955765f,
                                                      0.761594155955765f},
                                                     {-0.761594155955765f,
                                                      -0.761594155955765f,
                                                      -0.761594155955765f,
                                                      -0.761594155955765f,
                                                      -0.761594155955765f},
                                                     {0, 0, 0, 0, 0},
                                                     {0.964027580075817f,
                                                      0.964027580075817f,
                                                      0.964027580075817f,
                                                      0.964027580075817f,
                                                      0.964027580075817f}}})
                               .get_vector();

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_thresholded_relu)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/thresholded_relu.prototxt"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 3>(
            {{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
             {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
             {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    auto expected_output =
        test::NDArray<float, 3>(
            {{{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 3, 4, 5}, {6, 7, 8, 9, 10}},
             {{0, 0, 0, 0, 0}, {0, 0, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
             {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}}})
            .get_vector();

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_matmul_vec_ten3d)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/matmul_vec_ten3d.prototxt"));

    Inputs inputs;
    inputs.emplace_back(std::vector<float>{0.f, 1.f});
    inputs.emplace_back(
        test::NDArray<float, 3>{{{0.f}, {1.f}}, {{2.f}, {3.f}}, {{4.f}, {5.f}}}.get_vector());

    auto expected_output = test::NDArray<float, 2>{{1.f}, {3.f}, {5.f}}.get_vector();

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_softplus)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/softplus.prototxt"));

    // -1.0f, 0, 1.0f, 10.f,                    normal input values for activation
    // 100.0f, -100.0f, 1000.0f, -1000.0f,      input values that leads to exp() overflow
    // FLT_MIN, FLT_MIN / 16, -FLT_MIN / 16,    min, denorm, -denorm
    // FLT_MAX, -FLT_MAX,                       max, -max;
    Inputs inputs{std::vector<float>{-1.0f,
                                     0,
                                     1.0f,
                                     10.f,
                                     100.0f,
                                     -100.0f,
                                     1000.0f,
                                     -1000.0f,
                                     FLT_MIN,
                                     FLT_MIN / 16,
                                     -FLT_MIN / 16,
                                     FLT_MAX,
                                     -FLT_MAX}};

    std::vector<float>& input = inputs.back();
    std::vector<float> output;
    auto softplus_impl = [](float x) -> float {
        if (x > 0)
        {
            return x + std::log(std::exp(-x) + 1);
        }
        else
        {
            return std::log(std::exp(x) + 1);
        }
    };

    std::transform(std::begin(input), std::end(input), std::back_inserter(output), softplus_impl);

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_softplus_infinity)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/softplus.prototxt"));

    std::vector<float> input(13, std::numeric_limits<float>::infinity());
    std::vector<float> expected_output(13, std::numeric_limits<float>::infinity());

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input(input);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_sum_opset8)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/sum_opset8.prototxt"));

    Inputs inputs;
    inputs.emplace_back(std::vector<float>{1.0f, 2.0f, 3.0f});
    inputs.emplace_back(test::NDArray<float, 2>{{10.0f}, {20.0f}, {30.0f}}.get_vector());
    inputs.emplace_back(test::NDArray<float, 3>{{{100.0f}}, {{200.0f}}, {{300.0f}}}.get_vector());

    auto expected_output =
        test::NDArray<float, 3>{
            {{111.0f, 112.0f, 113.0f}, {121.0f, 122.0f, 123.0f}, {131.0f, 132.0f, 133.0f}},

            {{211.0f, 212.0f, 213.0f}, {221.0f, 222.0f, 223.0f}, {231.0f, 232.0f, 233.0f}},

            {{311.0f, 312.0f, 313.0f}, {321.0f, 322.0f, 323.0f}, {331.0f, 332.0f, 333.0f}}}
            .get_vector();

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_argmax_int32)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/argmax_int32.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<std::int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    test_case.add_expected_output<std::int32_t>({1, 1, 1, 1, 1, 1});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_argmin_int32)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/argmin_int32.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<std::int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    test_case.add_expected_output<std::int32_t>({0, 0, 0, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_top_k)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/top_k.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    test_case.add_expected_output<float>(Shape{3, 3}, {3, 2, 1, 7, 6, 5, 11, 10, 9}); // values
    test_case.add_expected_output<std::int64_t>(Shape{3, 3},
                                                {3, 2, 1, 3, 2, 1, 3, 2, 1}); // indices
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_top_k_opset_10)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/top_k_opset_10.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    test_case.add_input<int64_t>({3});

    test_case.add_expected_output<float>(Shape{3, 3}, {3, 2, 1, 7, 6, 5, 11, 10, 9}); // values
    test_case.add_expected_output<std::int64_t>(Shape{3, 3},
                                                {3, 2, 1, 3, 2, 1, 3, 2, 1}); // indices
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_top_k_opset_10_const_k)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/top_k_opset_10_const_k.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

    test_case.add_expected_output<float>(Shape{3, 3}, {3, 2, 1, 7, 6, 5, 11, 10, 9}); // values
    test_case.add_expected_output<std::int64_t>(Shape{3, 3},
                                                {3, 2, 1, 3, 2, 1, 3, 2, 1}); // indices
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_top_k_opset_11_const_k_smallest)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/top_k_opset_11_const_k_smallest.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8});

    test_case.add_expected_output<float>(Shape{3, 3}, {0, 1, 2, 4, 5, 6, 8, 9, 10}); // values
    test_case.add_expected_output<std::int64_t>(Shape{3, 3},
                                                {0, 1, 2, 0, 1, 2, 3, 2, 1}); // indices
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_acosh)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/acosh.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>(Shape{1, 3}, {1.0f, 2.5f, 4.3f});
    test_case.add_expected_output<float>(Shape{1, 3}, {0.0f, 1.5667993f, 2.13795861f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_asinh)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/asinh.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>(Shape{1, 3}, {-1.0f, 0.0f, 1.0f});
    test_case.add_expected_output<float>(Shape{1, 3}, {-0.88137358f, 0.0f, 0.88137358f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_atanh)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/atanh.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>(Shape{1, 3}, {-0.9f, 0.0f, 0.9f});
    test_case.add_expected_output<float>(Shape{1, 3}, {-1.4722194f, 0.0f, 1.4722194f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_sinh)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/sinh.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({-1.0f, 0.0f, 1.0f});
    test_case.add_expected_output<float>({-1.1752012f, 0.f, 1.1752012f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_cosh)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/cosh.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({-1.0f, 0.0f, 1.0f});
    test_case.add_expected_output<float>({1.54308069f, 1.f, 1.54308069f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_sign)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/sign.prototxt"));

    Inputs inputs{std::vector<float>{-std::numeric_limits<float>::infinity(),
                                     -3.141592f,
                                     0.0f,
                                     2.71828f,
                                     std::numeric_limits<float>::infinity()}};

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output<float>({-1.0f, -1.0f, 0.0f, 1.0f, 1.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_one_hot_with_axis)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/one_hot_axis.prototxt"));

    Inputs inputs{{1.0, 9.0, 2.0, 4.0}, {1.0, 3.0}};
    std::vector<float> expected_output{{1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0,
                                        1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 3.0,
                                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_one_hot_without_axis)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/one_hot_no_axis.prototxt"));

    std::vector<std::vector<std::int64_t>> inputs{{0, 7, 8}, {2, 5}};
    std::vector<std::int64_t> expected_output{5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                              2, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 2, 2, 2};

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_where)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/where.prototxt"));

    // conditions tensor - 3x3x3
    auto condition = std::vector<int>{
        {0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0}};

    // 1x3 tensor of "1"
    auto x1 = std::vector<int>{1, 1, 1};
    // 3x1 tensor of "2"
    auto x2 = std::vector<int>{2, 2, 2};

    std::vector<std::vector<int>> inputs;
    inputs.push_back(std::move(condition));
    inputs.push_back(std::move(x1));
    inputs.push_back(std::move(x2));

    // y = 3x3x3
    std::vector<int> expected_output{2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2,
                                     1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2};

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_erf)
{
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/erf.prototxt"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 2>{
        {-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()},
        {-3.141592f, 0.0f},
        {0.5f, 1.0f}}.get_vector());

    const std::vector<float> expected_output = test::NDArray<float, 2>{
        {-1.0f, 1.0f},
        {-0.99999112f, 0.0f},
        {0.52049988f, 0.84270079f}}.get_vector();

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_erf_int32)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/erf_int32.prototxt"));

    const std::vector<std::vector<int32_t>> inputs{
        {-std::numeric_limits<int32_t>::max(), -1, 0, 1, std::numeric_limits<int32_t>::max()}};

    const std::vector<int32_t> expected_output{-1, 0, 0, 0, 1};

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_shrink_float)
{
    const auto shrink_fn = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/shrink_float.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(shrink_fn, "${BACKEND_NAME}");
    test_case.add_input<float>(
        {-2.0f, -1.6f, -1.5f, -1.4f, -1.0f, 0.0f, 1.0f, 1.4f, 1.5f, 1.6f, 2.0f});
    test_case.add_expected_output<float>(
        Shape{11}, {-1.5f, -1.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.1f, 1.5f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_shrink_int)
{
    const auto shrink_fn = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/shrink_int.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(shrink_fn, "${BACKEND_NAME}");
    test_case.add_input<int>({-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5});
    test_case.add_expected_output<int>(Shape{11}, {-4, -3, -2, -1, 0, 0, 0, 1, 2, 3, 4});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_lp_norm_p1)
{
    const auto lp_norm_fn = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/lp_norm_p1.prototxt"));

    Shape data_shape{2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    std::iota(std::begin(data), std::end(data), 1);

    auto test_case = ngraph::test::NgraphTestCase(lp_norm_fn, "${BACKEND_NAME}");
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(
        data_shape, {0.07142857f, 0.125f,      0.16666667f, 0.2f,    0.22727273f, 0.25f,
                     0.26923078f, 0.2857143f,  0.3f,        0.3125f, 0.32352942f, 0.33333334f,
                     0.9285714f,  0.875f,      0.8333333f,  0.8f,    0.77272725f, 0.75f,
                     0.7307692f,  0.71428573f, 0.7f,        0.6875f, 0.6764706f,  0.6666667f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_lp_norm_p2)
{
    const auto lp_norm_fn = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/lp_norm_p2.prototxt"));

    Shape data_shape{2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    std::iota(std::begin(data), std::end(data), 1);

    auto test_case = ngraph::test::NgraphTestCase(lp_norm_fn, "${BACKEND_NAME}");
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(
        data_shape, {0.0766965f,  0.14142136f, 0.19611613f, 0.24253564f, 0.28216633f, 0.31622776f,
                     0.34570536f, 0.37139067f, 0.39391932f, 0.41380295f, 0.4314555f,  0.4472136f,
                     0.9970545f,  0.98994946f, 0.9805807f,  0.97014254f, 0.9593655f,  0.9486833f,
                     0.9383431f,  0.9284767f,  0.91914505f, 0.9103665f,  0.9021342f,  0.8944272f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_lp_norm_default)
{
    const auto lp_norm_fn = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/lp_norm_default.prototxt"));

    Shape data_shape{2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    std::iota(std::begin(data), std::end(data), 1);

    auto test_case = ngraph::test::NgraphTestCase(lp_norm_fn, "${BACKEND_NAME}");
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(
        data_shape, {0.18257418f, 0.36514837f, 0.5477225f,  0.73029673f, 0.37904903f, 0.45485884f,
                     0.5306686f,  0.60647845f, 0.42616236f, 0.47351375f, 0.5208651f,  0.5682165f,
                     0.4469492f,  0.48132992f, 0.51571065f, 0.5500913f,  0.45862272f, 0.48560053f,
                     0.5125783f,  0.53955615f, 0.46609157f, 0.4882864f,  0.51048124f, 0.5326761f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_instance_normalization)
{
    const auto instance_norm_fn = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/instance_norm.prototxt"));

    Shape data_shape{1, 2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    std::iota(std::begin(data), std::end(data), 1);

    auto test_case = ngraph::test::NgraphTestCase(instance_norm_fn, "${BACKEND_NAME}");
    test_case.add_input<float>(data);
    test_case.add_input<float>(std::vector<float>{2.134f, 3.256f});
    test_case.add_input<float>(std::vector<float>{0.765f, 1.055f});
    test_case.add_expected_output<float>(
        data_shape, {-2.6335807f, -2.015657f,  -1.3977331f, -0.77980936f, -0.16188562f, 0.45603812f,
                     1.0739619f,  1.6918856f,  2.3098092f,  2.927733f,    3.5456567f,   4.1635804f,
                     -4.130463f,  -3.1876516f, -2.2448401f, -1.3020288f,  -0.35921717f, 0.5835942f,
                     1.5264057f,  2.469217f,   3.4120288f,  4.35484f,     5.2976513f,   6.240463f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_eye_like)
{
    const auto eye_like_fn = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/eye_like.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(eye_like_fn, "${BACKEND_NAME}");
    test_case.add_input<float>({0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f});
    test_case.add_expected_output<float>(
        Shape{3, 4}, {0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reverse_sequence_0_batch_1)
{
    const auto reverse_sequence_fn = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reverse_sequence_time_0_batch_1.prototxt"));
    auto test_case = ngraph::test::NgraphTestCase(reverse_sequence_fn, "${BACKEND_NAME}");

    test_case.add_input<float>(
        {0.f, 4.f, 8.f, 12.f, 1.f, 5.f, 9.f, 13.f, 2.f, 6.f, 10.f, 14.f, 3.f, 7.f, 11.f, 15.f});
    test_case.add_input<int>({4, 3, 2, 1});
    test_case.add_expected_output<float>(
        Shape{4, 4},
        {3.f, 6.f, 9.f, 12.f, 2.f, 5.f, 8.f, 13.f, 1.f, 4.f, 10.f, 14.f, 0.f, 7.f, 11.f, 15.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reverse_sequence_1_batch_0)
{
    const auto reverse_sequence_fn = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reverse_sequence_time_1_batch_0.prototxt"));
    auto test_case = ngraph::test::NgraphTestCase(reverse_sequence_fn, "${BACKEND_NAME}");

    test_case.add_input<float>(
        {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f});
    test_case.add_input<int>({1, 2, 3, 4});
    test_case.add_expected_output<float>(
        Shape{4, 4},
        {0.f, 1.f, 2.f, 3.f, 5.f, 4.f, 6.f, 7.f, 10.f, 9.f, 8.f, 11.f, 15.f, 14.f, 13.f, 12.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reverse_sequence_incorrect_batch_axis)
{
    EXPECT_THROW(onnx_import::import_onnx_model(file_util::path_join(
                     SERIALIZED_ZOO, "onnx/reverse_sequence_incorrect_batch_axis.prototxt")),
                 ngraph_error)
        << "ReverseSequence batch_axis attribute can only equal 0 or 1. Value of '2' is not "
           "accepted.";
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reverse_sequence_incorrect_time_axis)
{
    EXPECT_THROW(onnx_import::import_onnx_model(file_util::path_join(
                     SERIALIZED_ZOO, "onnx/reverse_sequence_incorrect_time_axis.prototxt")),
                 ngraph_error)
        << "ReverseSequence time_axis attribute can only equal 0 or 1. Value of '2' is not "
           "accepted.";
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reverse_sequence_time_and_batch_axis_equal)
{
    EXPECT_THROW(onnx_import::import_onnx_model(file_util::path_join(
                     SERIALIZED_ZOO, "onnx/reverse_sequence_time_and_batch_axis_equal.prototxt")),
                 ngraph_error)
        << "ReverseSequence 'time_axis' and 'batch_axis' can't be equal.";
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_matmul_float_type)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/matmul_float.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>(std::vector<float>{0, 1, 2, 3, 4, 5});
    test_case.add_input<float>(std::vector<float>{0, 1});
    test_case.add_expected_output<float>(Shape{3, 1}, std::vector<float>{1, 3, 5});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_mod)
{
    const auto mod_fn = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/mod_sign.prototxt"));
    auto test_case = ngraph::test::NgraphTestCase(mod_fn, "${BACKEND_NAME}");

    test_case.add_input<int64_t>({-8, 3, 4, 9, -17, 1});
    test_case.add_input<int64_t>({22, -13, 8, -3, 7, 2});
    test_case.add_expected_output<int64_t>(Shape{6}, {-8, 3, 4, 0, -3, 1});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_scatterND)
{
    const auto scatterND_fn = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/scatter_nd.prototxt"));
    auto test_case = ngraph::test::NgraphTestCase(scatterND_fn, "${BACKEND_NAME}");

    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    test_case.add_input<int64_t>({4, 3, 1, 7});
    test_case.add_input<float>({9.f, 10.f, 11.f, 12.f});
    test_case.add_expected_output<float>(Shape{8}, {1.f, 11.f, 3.f, 10.f, 9.f, 6.f, 7.f, 12.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_gatherND_int32)
{
    const auto gatherND_fn = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/gatherND_int32.prototxt"));
    auto test_case = ngraph::test::NgraphTestCase(gatherND_fn, "${BACKEND_NAME}");

    test_case.add_input<int32_t>({0, 1, 2, 3});
    test_case.add_input<int64_t>({1, 0});
    test_case.add_expected_output<int32_t>(Shape{2, 2}, {2, 3, 0, 1});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_gatherND_float)
{
    const auto gatherND_fn = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/gatherND_float.prototxt"));
    auto test_case = ngraph::test::NgraphTestCase(gatherND_fn, "${BACKEND_NAME}");

    test_case.add_input<float>({0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f});
    test_case.add_input<int64_t>({0, 1, 1, 0});
    test_case.add_expected_output<float>(Shape{2, 2}, {2.f, 3.f, 4.f, 5.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_pad_constant)
{
    const auto pad_fn = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/pad_constant.prototxt"));
    auto test_case = ngraph::test::NgraphTestCase(pad_fn, "${BACKEND_NAME}");

    test_case.add_input<float>({1.f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f});
    test_case.add_expected_output<float>(
        Shape{3, 4}, {0.f, 0.f, 1.f, 1.2f, 0.f, 0.f, 2.3f, 3.4f, 0.f, 0.f, 4.5f, 5.7f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reciprocal)
{
    const auto reciprocal_fn = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reciprocal.prototxt"));
    auto test_case = ngraph::test::NgraphTestCase(reciprocal_fn, "${BACKEND_NAME}");

    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    test_case.add_expected_output<float>(Shape{3, 2},
                                         {1.f, 1 / 2.f, 1 / 3.f, 1 / 4.f, 1 / 5.f, 1 / 6.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_round)
{
    const auto round_fn =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/round.prototxt"));
    auto test_case = ngraph::test::NgraphTestCase(round_fn, "${BACKEND_NAME}");

    test_case.add_input<float>({0.1f,
                                0.5f,
                                0.9f,
                                1.2f,
                                1.5f,
                                1.8f,
                                2.3f,
                                2.5f,
                                2.7f,
                                -1.1f,
                                -1.5f,
                                -1.9f,
                                -2.2f,
                                -2.5f,
                                -2.8f});
    test_case.add_expected_output<float>(
        {0.f, 0.f, 1.f, 1.f, 2.f, 2.f, 2.f, 2.f, 3.f, -1.f, -2.f, -2.f, -2.f, -2.f, -3.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_scatter10_import_only)
{
    const auto scatter_fn = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/scatter_opset10.prototxt"));

    const Shape data_shape{2, 2};

    EXPECT_EQ(scatter_fn->get_output_size(), 1);
    EXPECT_EQ(scatter_fn->get_output_shape(0), data_shape);
    EXPECT_EQ(count_ops_of_type<op::v3::ScatterElementsUpdate>(scatter_fn), 1);
    EXPECT_EQ(count_ops_of_type<op::v0::Constant>(scatter_fn), 4);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_scatter_elements_import_only)
{
    const auto scatter_fn = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/scatter_elements_opset11.prototxt"));

    const Shape data_shape{1, 5};

    EXPECT_EQ(scatter_fn->get_output_size(), 1);
    EXPECT_EQ(scatter_fn->get_output_shape(0), data_shape);
    EXPECT_EQ(count_ops_of_type<op::v3::ScatterElementsUpdate>(scatter_fn), 1);
    EXPECT_EQ(count_ops_of_type<op::v0::Constant>(scatter_fn), 4);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_upsample8_import_only)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/upsample8_nearest.prototxt"));

    // Input data shape (1, 1, 2, 2)
    // Scales attribute values {1.0, 1.0, 2.0, 3.0}

    Shape expected_output_shape{1, 1, 4, 6};
    EXPECT_EQ(function->get_output_size(), 1);
    EXPECT_EQ(function->get_output_shape(0), expected_output_shape);
    EXPECT_EQ(count_ops_of_type<onnx_import::default_opset::Interpolate>(function), 1);
    EXPECT_EQ(count_ops_of_type<onnx_import::default_opset::Constant>(function), 1);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_upsample8_nearest_infer)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/upsample8_nearest.prototxt"));

    // Input data shape (1, 1, 2, 2)
    // Scales attribute values {1.0, 1.0, 2.0, 3.0}
    // mode: nearest

    Shape expected_output_shape{1, 1, 4, 6};
    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({1.0, 2.0, 3.0, 4.0});
    test_case.add_expected_output<float>(
        expected_output_shape, {1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
                                3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_upsample8_linear_infer)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/upsample8_linear.prototxt"));

    // Input data shape (1, 1, 2, 2)
    // Scales attribute values {1.0, 1.0, 2.0, 2.0}
    // mode: linear

    Shape expected_output_shape{1, 1, 4, 4};
    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({1.0, 2.0, 3.0, 4.0});
    test_case.add_expected_output<float>(
        expected_output_shape,
        {1.0, 1.5, 2.0, 2.0, 2.0, 2.5, 3.0, 3.0, 3.0, 3.5, 4.0, 4.0, 3.0, 3.5, 4.0, 4.0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_upsample9_scales_const_import_only)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/upsample9_scales_const_nearest.prototxt"));

    // Input data shape (1, 1, 2, 2)
    // Input const scales values {1.0, 1.0, 2.0, 3.0}

    Shape expected_output_shape{1, 1, 4, 6};
    EXPECT_EQ(function->get_output_size(), 1);
    EXPECT_EQ(function->get_output_shape(0), expected_output_shape);
    EXPECT_EQ(count_ops_of_type<onnx_import::default_opset::Interpolate>(function), 1);
    EXPECT_EQ(count_ops_of_type<onnx_import::default_opset::Constant>(function), 1);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_upsample9_scales_const_nearest_infer)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/upsample9_scales_const_nearest.prototxt"));

    // Input data shape (1, 1, 2, 2)
    // Input const scales values {1.0, 1.0, 2.0, 3.0}
    // mode: nearest

    Shape expected_output_shape{1, 1, 4, 6};
    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({1.0, 2.0, 3.0, 4.0});
    test_case.add_expected_output<float>(
        expected_output_shape, {1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
                                3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_upsample9_scales_const_linear_infer)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/upsample9_scales_const_linear.prototxt"));

    // Input data shape (1, 1, 2, 2)
    // Input const scales values {1.0, 1.0, 2.0, 2.0}
    // mode: linear

    Shape expected_output_shape{1, 1, 4, 4};
    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({1.0, 2.0, 3.0, 4.0});
    test_case.add_expected_output<float>(
        expected_output_shape,
        {1.0, 1.5, 2.0, 2.0, 2.0, 2.5, 3.0, 3.0, 3.0, 3.5, 4.0, 4.0, 3.0, 3.5, 4.0, 4.0});
    test_case.run();
}