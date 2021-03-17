//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include "core/null_node.hpp"
#include "gtest/gtest.h"
#include "onnx_import/onnx.hpp"
#include "onnx_import/onnx_utils.hpp"
#include "default_opset.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_tools.hpp"
#include "util/type_prop.hpp"
#include <cpp/ie_cnn_network.h>

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";

using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;

using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

// ############################################################################ CORE TESTS
NGRAPH_TEST(${BACKEND_NAME}, onnx_test_test_case)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
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
    EXPECT_EQ(additions.at(0)->get_output_tensor(0).get_names(),
              std::unordered_set<std::string>{"X"});
    EXPECT_EQ(additions.at(1)->get_friendly_name(), "Y");
    EXPECT_EQ(additions.at(1)->get_output_tensor(0).get_names(),
              std::unordered_set<std::string>{"Y"});
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_add_abc)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_multiple_inputs(Inputs{{1}, {2}, {3}});
    test_case.add_expected_output(Shape{1}, std::vector<float>{6});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_binary_add_abc)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_multiple_inputs(Inputs{{1}, {2}, {3}});
    test_case.add_expected_output(Shape{1}, std::vector<float>{6});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_bool_const_op)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/bool_const_op.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_expected_output(std::vector<bool>{1, 0, 0, 1});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_bool_init_and)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/bool_init_and.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_expected_output(std::vector<bool>{1});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_bool_input_or)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/bool_input_or.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input(std::vector<bool>{true, false, true, false});
    test_case.add_input(std::vector<bool>{false, false, true, true});
    test_case.add_expected_output(std::vector<bool>{1, 0, 1, 1});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_bool_init_raw)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/bool_init_raw.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_expected_output(std::vector<bool>{true, false, true});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_add_abc_initializers)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc_initializers.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({1, 2, 3, 4});
    test_case.add_expected_output<float>({3, 6, 9, 12});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_override_op)
{
    onnx_import::register_operator(
        "FalseAdd", 1, "", [](const onnx_import::Node& node) -> OutputVector {
            OutputVector ng_inputs{node.get_ng_inputs()};
            return {std::make_shared<ngraph::op::v1::Add>(ng_inputs.at(0), ng_inputs.at(1))};
        });

    onnx_import::register_operator(
        "FalseAdd", 1, "", [](const onnx_import::Node& node) -> OutputVector {
            OutputVector ng_inputs{node.get_ng_inputs()};
            return {std::make_shared<ngraph::op::v1::Subtract>(ng_inputs.at(0), ng_inputs.at(1))};
        });

    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/override_op.prototxt"));

    Inputs inputs;
    inputs.emplace_back(std::vector<float>{0.f, 1.f, 2.f, 3.f});
    inputs.emplace_back(std::vector<float>{3.f, 2.f, 1.f, 0.f});

    auto test_case = test::TestCase<TestEngine>(function);
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
        "AddQ", 1, "com.intel.ai", [](const onnx_import::Node& node) -> OutputVector {
            OutputVector ng_inputs{node.get_ng_inputs()};
            return {std::make_shared<ngraph::op::v1::Add>(ng_inputs.at(0), ng_inputs.at(1))};
        });

    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/custom_operator.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>({3.f, 6.f, 9.f, 12.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_custom_op_register_unregister)
{
    onnx_import::register_operator(
        "AddQ", 1, "com.intel.ai", [](const onnx_import::Node& node) -> OutputVector {
            OutputVector ng_inputs{node.get_ng_inputs()};
            return {std::make_shared<ngraph::op::v1::Add>(ng_inputs.at(0), ng_inputs.at(1))};
        });

    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/custom_operator.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>({3.f, 6.f, 9.f, 12.f});
    test_case.run();

    onnx_import::unregister_operator("AddQ", 1, "com.intel.ai");
    try
    {
        auto function = onnx_import::import_onnx_model(
            file_util::path_join(SERIALIZED_ZOO, "onnx/custom_operator.prototxt"));
        FAIL() << "Expected ngraph::ngraph_error";
    }
    catch (ngraph::ngraph_error const& err)
    {
        std::string what{err.what()};
        EXPECT_NE(what.find("Check 'unknown_operators.empty()' failed"), std::string::npos);
    }
    catch (...)
    {
        FAIL() << "Expected ngraph::ngraph_error";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_custom_op_default_domain)
{
    onnx_import::register_operator(
        "AddQ", 1, "com.intel.ai", [](const onnx_import::Node& node) -> OutputVector {
            OutputVector ng_inputs{node.get_ng_inputs()};
            return {std::make_shared<ngraph::op::v1::Add>(ng_inputs.at(0), ng_inputs.at(1))};
        });

    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/custom_operator_default_domain.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
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
        "AddQ", 1, "com.intel.ai", [](const onnx_import::Node& node) -> OutputVector {
            OutputVector ng_inputs{node.get_ng_inputs()};
            return {std::make_shared<ngraph::op::v1::Add>(ng_inputs.at(0), ng_inputs.at(1))};
        });
    EXPECT_TRUE(onnx_import::is_operator_supported("AddQ", 1, "com.intel.ai"));
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_missing_op_domain)
{
    onnx_import::register_operator(
        "CustomAdd", 1, "custom.op", [](const onnx_import::Node& node) -> OutputVector {
            OutputVector ng_inputs{node.get_ng_inputs()};
            return {std::make_shared<ngraph::op::v1::Add>(ng_inputs.at(0), ng_inputs.at(1))};
        });

    EXPECT_TRUE(onnx_import::is_operator_supported("CustomAdd", 1, "custom.op"));

    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/missing_op_domain.prototxt"));

    Inputs inputs;
    inputs.emplace_back(std::vector<float>{0.f, 1.f, 2.f, 3.f});
    inputs.emplace_back(std::vector<float>{0.f, 1.f, 2.f, 3.f});

    auto test_case = test::TestCase<TestEngine>(function);
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
        "TestMissingInOut", 1, "com.intel.ai", [](const onnx_import::Node& node) -> OutputVector {
            OutputVector ng_inputs{node.get_ng_inputs()};
            Output<ngraph::Node> A = ng_inputs.at(0);
            Output<ngraph::Node> B = ng_inputs.at(1);
            Output<ngraph::Node> C = ng_inputs.at(2);

            A = std::make_shared<op::v1::Multiply>(A, C);
            if (!ngraph::op::is_null(B))
            {
                B = std::make_shared<op::v1::Divide>(B, C);
            }

            C = std::make_shared<ngraph::op::v1::Add>(C, C);
            return {A, B, C};
        });

    onnx_import::register_operator(
        "TestMissingIn", 1, "com.intel.ai", [](const onnx_import::Node& node) -> OutputVector {
            OutputVector ng_inputs{node.get_ng_inputs()};
            std::shared_ptr<ngraph::Node> result = std::make_shared<ngraph::op::Constant>(
                element::f32, ngraph::Shape{2, 2}, std::vector<float>{1, 1, 1, 1});

            for (const auto& ng_input : ng_inputs)
            {
                if (!ngraph::op::is_null(ng_input))
                {
                    result = std::make_shared<op::v1::Multiply>(ng_input, result);
                }
            }

            return {result};
        });

    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/missing_input.prototxt"));

    Inputs inputs{{1, 2, 3, 4}, {5, 6, 7, 8}};

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output<float>({50, 144, 294, 512});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_initializer_wo_input)
{
    // This test checks a model which has an initializer, but no input with the same name
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/initializer_wo_input.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({0, 1, 2, 3, 4, 5});
    test_case.add_expected_output<float>({0, 2, 6, 12, 20, 30});
    test_case.run();
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, onnx_expand_function)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/quantization/dynamicquantizelinear.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({-1.f, -2.1f, -1.3f, -2.5f, -3.34f, -4.f});
    test_case.add_expected_output<uint8_t>(Shape{6}, {191, 121, 172, 96, 42, 0});
    test_case.add_expected_output<float>(Shape{}, {0.0156862754f});
    test_case.add_expected_output<uint8_t>(Shape{}, {255});
    test_case.run();
}

// ############################################################################ OPERATOR TESTS
NGRAPH_TEST(${BACKEND_NAME}, onnx_model_addmul_abc)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/addmul_abc.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({2, 1, 3, 10});
    test_case.add_expected_output<float>(Shape{2}, {1, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_batch_norm_default)
{
    // Batch Normalization with default parameters
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/batchnorm_default.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({-1, -2, 0, 1, 2, 3});
    test_case.add_expected_output<float>({0, 0, 0, 1, 2, 3});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_sum_opset1)
{
    // Simple Sum test for opset1.
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/sum_opset1.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
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
    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({3.f, 0.f, 2.f});
    test_case.add_expected_output<float>({3.f, 0.f, 2.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_cum_sum_1d)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/cum_sum_1d.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({1.f, 2.f, 3.f});
    test_case.add_expected_output<float>(Shape{3}, {1.f, 3.f, 6.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_cum_sum_2d_axis_input)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/cum_sum_2d_axis_input.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    test_case.add_expected_output<float>(Shape{2, 3}, {1.f, 3.f, 6.f, 4.f, 9.f, 15.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_cum_sum_2d_dynamic_axis_input)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/cum_sum_2d_dynamic_axis_input.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    test_case.add_input<std::int32_t>({1});
    test_case.add_expected_output<float>(Shape{2, 3}, {1.f, 3.f, 6.f, 4.f, 9.f, 15.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_cum_sum_3d_exclusive_reverse)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/cum_sum_3d_exclusive_reverse.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
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
    auto test_case = test::TestCase<TestEngine>(function);
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
    auto test_case = test::TestCase<TestEngine>(function);
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
    auto test_case = test::TestCase<TestEngine>(function);
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
    auto test_case = test::TestCase<TestEngine>(function);
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
    auto test_case = test::TestCase<TestEngine>(function);
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
    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_softmax_0D)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/softmax_0D.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_expected_output<float>({1.0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_softmax_1D)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/softmax_1D.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
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
} // namespace

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_softmax_axis_0)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/softmax_axis_0.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>(SOFTMAX_INPUT);

    test_case.add_expected_output<float>(
        Shape{3, 4, 5},
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

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(function);
    test_case.add_input<float>(SOFTMAX_INPUT);

    test_case.add_expected_output<float>(
        Shape{3, 4, 5},
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

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_nonmaxsuppression_center_point_box_format)
{
    auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/nonmaxsuppression_center_point_box_format.prototxt"));

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(function);

    test_case.add_input(std::vector<float>(
        {0.5f, 0.5f,  1.0f, 1.0f, 0.5f, 0.6f,  1.0f, 1.0f, 0.5f, 0.4f,   1.0f, 1.0f,
         0.5f, 10.5f, 1.0f, 1.0f, 0.5f, 10.6f, 1.0f, 1.0f, 0.5f, 100.5f, 1.0f, 1.0f})); // boxes
    test_case.add_input(std::vector<float>({0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f}));    // scores
    test_case.add_input(std::vector<int64_t>({3}));  // max_output_boxes_per_class
    test_case.add_input(std::vector<float>({0.5f})); // iou_threshold
    test_case.add_input(std::vector<float>({0.0f})); // score_threshold

    test_case.add_expected_output<int64_t>(Shape{3, 3}, {0, 0, 3, 0, 0, 0, 0, 0, 5});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_nonmaxsuppression_single_box)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/nonmaxsuppression_single_box.prototxt"));

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(function);

    test_case.add_input(std::vector<float>({0.0f, 0.0f, 1.0f, 1.0f})); // boxes
    test_case.add_input(std::vector<float>({0.9f}));                   // scores
    test_case.add_input(std::vector<int64_t>({3}));                    // max_output_boxes_per_class
    test_case.add_input(std::vector<float>({0.5f}));                   // iou_threshold
    test_case.add_input(std::vector<float>({0.0f}));                   // score_threshold

    test_case.add_expected_output<int64_t>(Shape{1, 3}, {0, 0, 0});
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

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_max_invalid_axes)
{
    EXPECT_THROW(onnx_import::import_onnx_model(
                     file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_max_invalid_axes.prototxt")),
                 ngraph::ngraph_error);
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

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_dynamic_rank_input)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_sum_dynamic_rank_input.prototxt"));
    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(function);
    test_case.add_input<float>(Shape{1, 1, 4, 4},
                               {1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f});

    test_case.add_expected_output<float>(Shape{1, 1, 1, 1}, {16.0f});
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

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_as_constant)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_sum_13_axes_as_constant.prototxt"));

    Inputs inputs{test::NDArray<float, 4>({{{{1.0f, 1.0f, 1.0f, 1.0f},
                                             {1.0f, 1.0f, 1.0f, 1.0f},
                                             {1.0f, 1.0f, 1.0f, 1.0f},
                                             {1.0f, 1.0f, 1.0f, 1.0f}}}})
                      .get_vector()};

    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_expected_output<float>(Shape{1, 1, 1, 1}, {16.0f});

    test_case.add_multiple_inputs(inputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_as_constant_single_axis)
{
    auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/reduce_sum_13_axes_as_constant_single_axis.prototxt"));

    Inputs inputs{
        test::NDArray<float, 3>({{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}}).get_vector()};

    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_expected_output<float>(Shape{2, 1, 3}, {5.0f, 7.0f, 9.0f, 17.0f, 19.0f, 21.0f});

    test_case.add_multiple_inputs(inputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_as_constant_keepdims_off)
{
    auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/reduce_sum_13_axes_as_constant_keepdims_off.prototxt"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{test::NDArray<float, 4>({{{{1.0f, 1.0f, 1.0f, 1.0f},
                                             {1.0f, 1.0f, 1.0f, 1.0f},
                                             {1.0f, 1.0f, 1.0f, 1.0f},
                                             {1.0f, 1.0f, 1.0f, 1.0f}}}})
                      .get_vector()};

    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_expected_output<float>(Shape{}, {16.0f});

    test_case.add_multiple_inputs(inputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_as_input)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_sum_13_axes_as_input.prototxt"));

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(function);
    test_case.add_input<float>({1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f});
    test_case.add_input<int64_t>({0, 1, 2, 3});

    test_case.add_expected_output<float>(Shape{1, 1, 1, 1}, {16.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_as_0_dim_input)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_sum_13_axes_as_0_dim_input.prototxt"));

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(function);
    test_case.add_input<float>(
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

    test_case.add_expected_output<float>(
        Shape{3, 2, 2},
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_input_dynamic)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_sum_13_input_dynamic.prototxt"));

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(function);
    test_case.add_input<int64_t>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

    test_case.add_expected_output<int64_t>(Shape{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                                           {5});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_empty)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_sum_13_axes_empty.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f});

    test_case.add_expected_output<float>(Shape{1, 1, 1, 1}, {16.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_empty_dynamic_rank_input)
{
    auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/reduce_sum_13_axes_empty_dynamic_rank_input.prototxt"));

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(function);
    test_case.add_input<float>(Shape{1, 1, 4, 4},
                               {1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f});

    test_case.add_expected_output<float>(Shape{1, 1, 1, 1}, {16.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_empty_with_noop)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_sum_13_axes_empty_with_noop.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({1.f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f});

    test_case.add_expected_output<float>(Shape{1, 1, 4, 4},
                                         {1.f,
                                          1.0f,
                                          1.0f,
                                          1.0f,
                                          1.0f,
                                          1.0f,
                                          1.0f,
                                          1.0f,
                                          1.0f,
                                          1.0f,
                                          1.0f,
                                          1.0f,
                                          1.0f,
                                          1.0f,
                                          1.0f,
                                          1.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_empty_without_noop)
{
    auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/reduce_sum_13_axes_empty_without_noop.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({1.f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f,
                                1.0f});

    test_case.add_expected_output<float>(Shape{1, 1, 1, 1}, {16.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_empty_constant_as_input)
{
    // this model contains a Constant node with an empty underlying tensor
    // this node is connected to the "roi" input of the Resize op but this input should be
    // ignored since the Resize coordinate_transformation_mode is set to asymmetric
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize11_empty_constant_as_input.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> input_data{1.0f, 3.0f, 4.0f, 8.0f, 6.0f, 2.0f, 7.0f, 11.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(
        Shape{1, 2, 4, 8},
        {1.0f,  1.5f,  2.0f, 2.5f, 3.0f, 3.0f,  3.0f,  3.0f,  2.5f,  3.25f, 4.0f,
         4.75f, 5.5f,  5.5f, 5.5f, 5.5f, 4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  8.0f,
         8.0f,  8.0f,  4.0f, 5.0f, 6.0f, 7.0f,  8.0f,  8.0f,  8.0f,  8.0f,

         6.0f,  5.0f,  4.0f, 3.0f, 2.0f, 2.0f,  2.0f,  2.0f,  6.5f,  6.5f,  6.5f,
         6.5f,  6.5f,  6.5f, 6.5f, 6.5f, 7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 11.0f,
         11.0f, 11.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 11.0f, 11.0f, 11.0f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize10_down_scales_const_nearest)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize10_down_scales_const_nearest.prototxt"));

    // Input data shape (1, 1, 2, 4)
    // Input const scales values {1.0, 1.0, 0.6, 0.6}
    // mode: linear

    Shape expected_output_shape{1, 1, 1, 2};
    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    test_case.add_expected_output<float>(expected_output_shape, {1.0, 3.0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize10_up_scales_const_linear)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize10_up_scales_const_linear.prototxt"));

    // Input data shape (1, 1, 2, 2)
    // Input const scales values {1.0, 1.0, 2.0, 2.0}
    // mode: nearest

    Shape expected_output_shape{1, 1, 4, 4};
    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({1.0, 2.0, 3.0, 4.0});
    test_case.add_expected_output<float>(
        expected_output_shape,
        {1.0, 1.5, 2.0, 2.0, 2.0, 2.5, 3.0, 3.0, 3.0, 3.5, 4.0, 4.0, 3.0, 3.5, 4.0, 4.0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize10_up_scales_const_nearest)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize10_up_scales_const_nearest.prototxt"));

    // Input data shape (1, 1, 2, 2)
    // Input const scales values {1.0, 1.0, 2.0, 3.0}
    // mode: linear

    Shape expected_output_shape{1, 1, 4, 6};
    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({1.0, 2.0, 3.0, 4.0});
    test_case.add_expected_output<float>(
        expected_output_shape, {1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
                                3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_down_scales_linear_asymmetric)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/resize11_down_scales_linear_asymmetric.prototxt"));

    const Shape expected_output_shape{1, 1, 1, 2};
    auto test_case = test::TestCase<TestEngine>(function);
    const size_t input_size = 8;
    std::vector<float> input_data(input_size);
    std::iota(std::begin(input_data), std::end(input_data), 1.0f);
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(expected_output_shape, {1.0f, 2.66666651f});

    test_case.run_with_tolerance_as_fp();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_scales_nearest_asymmetric_floor_dynamic_sizes)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/resize11_scales_nearest_asymmetric_floor_dynamic_scales.prototxt"));

    const Shape expected_output_shape{2, 1, 4, 1};
    auto test_case = test::TestCase<TestEngine>(function);
    const std::vector<float> input_data{1.0f, 3.0f, 4.0f, 8.0f, 6.0f, 2.0f, 7.0f, 11.0f};
    test_case.add_input<float>(input_data);
    test_case.add_input<float>(
        std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}); // roi
    test_case.add_input<float>(std::vector<float>{1.0f, 1.0f, 2.0f, 0.5f});  // scales
    test_case.add_expected_output<float>(expected_output_shape,
                                         {1.0f, 1.0f, 4.0f, 4.0f, 6.0f, 6.0f, 7.0f, 7.0f});

    test_case.run_with_tolerance_as_fp();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_up_scales_linear_asymmetric)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize11_up_scales_linear_asymmetric.prototxt"));

    const Shape expected_output_shape{2, 1, 4, 8};
    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> input_data{1.0f, 3.0f, 4.0f, 8.0f, 6.0f, 2.0f, 7.0f, 11.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(
        expected_output_shape,
        {1.0f,  1.5f,  2.0f, 2.5f, 3.0f, 3.0f,  3.0f,  3.0f,  2.5f,  3.25f, 4.0f,
         4.75f, 5.5f,  5.5f, 5.5f, 5.5f, 4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  8.0f,
         8.0f,  8.0f,  4.0f, 5.0f, 6.0f, 7.0f,  8.0f,  8.0f,  8.0f,  8.0f,

         6.0f,  5.0f,  4.0f, 3.0f, 2.0f, 2.0f,  2.0f,  2.0f,  6.5f,  6.5f,  6.5f,
         6.5f,  6.5f,  6.5f, 6.5f, 6.5f, 7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 11.0f,
         11.0f, 11.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 11.0f, 11.0f, 11.0f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_scales_nearest_asymmetric_floor)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/resize11_scales_nearest_asymmetric_floor.prototxt"));

    const Shape expected_output_shape{2, 1, 4, 1};
    auto test_case = test::TestCase<TestEngine>(function);
    const std::vector<float> input_data{1.0f, 3.0f, 4.0f, 8.0f, 6.0f, 2.0f, 7.0f, 11.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(expected_output_shape,
                                         {1.0f, 1.0f, 4.0f, 4.0f, 6.0f, 6.0f, 7.0f, 7.0f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_up_scales_cubic_align_corners)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/resize11_up_scales_cubic_align_corners.prototxt"));

    const Shape expected_output_shape{1, 1, 8, 8};
    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> input_data{1.0f,
                                  2.0f,
                                  3.0f,
                                  4.0f,
                                  5.0f,
                                  6.0f,
                                  7.0f,
                                  8.0f,
                                  9.0f,
                                  10.0f,
                                  11.0f,
                                  12.0f,
                                  13.0f,
                                  14.0f,
                                  15.0f,
                                  16.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(
        expected_output_shape,
        {
            1.0f,         1.34110787f,  1.80029155f,  2.32944606f,  2.67055394f,  3.19970845f,
            3.65889213f,  4.0f,         2.36443149f,  2.70553936f,  3.16472303f,  3.69387755f,
            4.03498542f,  4.56413994f,  5.02332362f,  5.36443149f,  4.20116618f,  4.54227405f,
            5.00145773f,  5.53061224f,  5.87172012f,  6.40087464f,  6.86005831f,  7.20116618f,
            6.31778426f,  6.65889213f,  7.1180758f,   7.64723032f,  7.98833819f,  8.51749271f,
            8.97667638f,  9.31778426f,  7.68221574f,  8.02332362f,  8.48250729f,  9.01166181f,
            9.35276968f,  9.8819242f,   10.34110787f, 10.68221574f, 9.79883382f,  10.13994169f,
            10.59912536f, 11.12827988f, 11.46938776f, 11.99854227f, 12.45772595f, 12.79883382f,
            11.63556851f, 11.97667638f, 12.43586006f, 12.96501458f, 13.30612245f, 13.83527697f,
            14.29446064f, 14.6355685f,  13.0f,        13.34110787f, 13.80029155f, 14.32944606f,
            14.67055394f, 15.19970845f, 15.65889213f, 16.0f,
        });
    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_up_scales_tf_half_pixel)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize11_up_scales_tf_half_pixel.prototxt"));

    const Shape expected_output_shape{1, 1, 8, 8};
    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> input_data{1.0f,
                                  2.0f,
                                  3.0f,
                                  4.0f,
                                  5.0f,
                                  6.0f,
                                  7.0f,
                                  8.0f,
                                  9.0f,
                                  10.0f,
                                  11.0f,
                                  12.0f,
                                  13.0f,
                                  14.0f,
                                  15.0f,
                                  16.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(
        expected_output_shape,
        {1.95703f, 2.43359f, 3.0625f,  3.46875f, 4.09766f, 4.57422f, 4.87109f, 4.80078f,
         3.86328f, 4.33984f, 4.96875f, 5.375f,   6.00391f, 6.48047f, 6.77734f, 6.70703f,
         6.37891f, 6.85547f, 7.48438f, 7.89063f, 8.51953f, 8.99609f, 9.29297f, 9.22266f,
         8.00391f, 8.48047f, 9.10938f, 9.51563f, 10.1445f, 10.6211f, 10.918f,  10.8477f,
         10.5195f, 10.9961f, 11.625f,  12.0313f, 12.6602f, 13.1367f, 13.4336f, 13.3633f,
         12.4258f, 12.9023f, 13.5313f, 13.9375f, 14.5664f, 15.043f,  15.3398f, 15.2695f,
         13.6133f, 14.0898f, 14.7188f, 15.125f,  15.7539f, 16.2305f, 16.5273f, 16.457f,
         13.332f,  13.8086f, 14.4375f, 14.8438f, 15.4727f, 15.9492f, 16.2461f, 16.1758f});
    test_case.run_with_tolerance_as_fp(2.0e-2f);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_all_attributes_default)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/resize11_up_sizes_all_attributes_default.prototxt"));

    const Shape expected_output_shape{1, 1, 7, 8};
    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> input_data{1.0f, 2.0f, 3.0f, 4.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(
        expected_output_shape,
        {1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f,
         2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f, 3.0f, 3.0f,
         3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f, 3.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f});
    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_sizes_nearest_asymmetric_floor)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/resize11_sizes_nearest_asymmetric_floor.prototxt"));

    const Shape expected_output_shape{2, 1, 4, 1};
    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> input_data{1.0f, 3.0f, 4.0f, 8.0f, 6.0f, 2.0f, 7.0f, 11.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(expected_output_shape,
                                         {1.0f, 1.0f, 4.0f, 4.0f, 6.0f, 6.0f, 7.0f, 7.0f});

    test_case.run_with_tolerance_as_fp();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_linear_asymmetric)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize11_up_sizes_linear_asymmetric.prototxt"));

    const Shape expected_output_shape{2, 1, 4, 8};
    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> input_data{2.0f, 4.0f, 1.0f, 3.0f, 7.0f, 8.0f, 9.0f, 6.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(
        expected_output_shape,
        {2.0f, 2.5f, 3.0f,  3.5f, 4.0f,  4.0f,  4.0f, 4.0f,  1.5f, 2.0f,  2.5f,  3.0f, 3.5f,
         3.5f, 3.5f, 3.5f,  1.0f, 1.5f,  2.0f,  2.5f, 3.0f,  3.0f, 3.0f,  3.0f,  1.0f, 1.5f,
         2.0f, 2.5f, 3.0f,  3.0f, 3.0f,  3.0f,  7.0f, 7.25f, 7.5f, 7.75f, 8.0f,  8.0f, 8.0f,
         8.0f, 8.0f, 7.75f, 7.5f, 7.25f, 7.0f,  7.0f, 7.0f,  7.0f, 9.0f,  8.25f, 7.5f, 6.75f,
         6.0f, 6.0f, 6.0f,  6.0f, 9.0f,  8.25f, 7.5f, 6.75f, 6.0f, 6.0f,  6.0f,  6.0f});

    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_down_sizes_cubic_half_pixel)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize11_down_sizes_cubic_half_pixel.prototxt"));

    const Shape expected_output_shape{1, 1, 3, 3};
    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> input_data{1.0f,
                                  2.0f,
                                  3.0f,
                                  4.0f,
                                  5.0f,
                                  6.0f,
                                  7.0f,
                                  8.0f,
                                  9.0f,
                                  10.0f,
                                  11.0f,
                                  12.0f,
                                  13.0f,
                                  14.0f,
                                  15.0f,
                                  16.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(expected_output_shape,
                                         {1.6307871,
                                          3.0046299,
                                          4.3784733,
                                          7.1261587,
                                          8.5,
                                          9.873844,
                                          12.621532,
                                          13.995373,
                                          15.369216});

    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_down_sizes_linear_pytorch_half_pixel)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/resize11_down_sizes_linear_pytorch_half_pixel.prototxt"));

    const Shape expected_output_shape{1, 1, 3, 1};
    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> input_data{1.0f,
                                  2.0f,
                                  3.0f,
                                  4.0f,
                                  5.0f,
                                  6.0f,
                                  7.0f,
                                  8.0f,
                                  9.0f,
                                  10.0f,
                                  11.0f,
                                  12.0f,
                                  13.0f,
                                  14.0f,
                                  15.0f,
                                  16.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(expected_output_shape, {1.666666f, 7.0f, 12.333333f});

    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_cubic_half_pixel)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize11_up_sizes_cubic_half_pixel.prototxt"));

    const Shape expected_output_shape{1, 1, 9, 10};
    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> input_data{1.0f,
                                  2.0f,
                                  3.0f,
                                  4.0f,
                                  5.0f,
                                  6.0f,
                                  7.0f,
                                  8.0f,
                                  9.0f,
                                  10.0f,
                                  11.0f,
                                  12.0f,
                                  13.0f,
                                  14.0f,
                                  15.0f,
                                  16.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(
        expected_output_shape,
        {0.45507922f,  0.64057922f,  0.97157922f,  1.42257922f,  1.90732922,   2.22332922f,
         2.70807922f,  3.15907922f,  3.49007922f,  3.67557922,   1.39437963f,  1.57987963f,
         1.91087963f,  2.36187963f,  2.84662963,   3.16262963f,  3.64737963f,  4.09837963f,
         4.42937963f,  4.61487963,   2.95130693f,  3.13680693f,  3.46780693f,  3.91880693f,
         4.40355693,   4.71955693f,  5.20430693f,  5.65530693f,  5.98630693f,  6.17180693,
         5.20525069f,  5.39075069f,  5.72175069f,  6.17275069f,  6.65750069,   6.97350069f,
         7.45825069f,  7.90925069f,  8.24025069f,  8.42575069,   6.88975f,     7.07525f,
         7.40625f,     7.85725f,     8.342,        8.658f,       9.14275f,     9.59375f,
         9.92475f,     10.11025f,    8.57424931f,  8.75974931f,  9.09074931f,  9.54174931f,
         10.02649931,  10.34249931f, 10.82724931f, 11.27824931f, 11.60924931f, 11.79474931,
         10.82819307f, 11.01369307f, 11.34469307f, 11.79569307f, 12.28044307,  12.59644307f,
         13.08119307f, 13.53219307f, 13.86319307f, 14.04869307,  12.38512037f, 12.57062037f,
         12.90162037f, 13.35262037f, 13.83737037,  14.15337037f, 14.63812037f, 15.08912037f,
         15.42012037f, 15.60562037,  13.32442078f, 13.50992078f, 13.84092078f, 14.29192078f,
         14.77667078,  15.09267078f, 15.57742078f, 16.02842078f, 16.35942078f, 16.54492078});
    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_cubic_half_pixel_dynamic_sizes)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/resize11_up_sizes_cubic_half_pixel_dynamic_sizes.prototxt"));

    const Shape expected_output_shape{1, 1, 9, 10};
    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> input_data{1.0f,
                                  2.0f,
                                  3.0f,
                                  4.0f,
                                  5.0f,
                                  6.0f,
                                  7.0f,
                                  8.0f,
                                  9.0f,
                                  10.0f,
                                  11.0f,
                                  12.0f,
                                  13.0f,
                                  14.0f,
                                  15.0f,
                                  16.0f};
    test_case.add_input<float>(input_data);
    test_case.add_input<float>(std::vector<float>{1, 1, 9, 10}); // sizes
    test_case.add_expected_output<float>(
        expected_output_shape,
        {0.45507922f,  0.64057922f,  0.97157922f,  1.42257922f,  1.90732922,   2.22332922f,
         2.70807922f,  3.15907922f,  3.49007922f,  3.67557922,   1.39437963f,  1.57987963f,
         1.91087963f,  2.36187963f,  2.84662963,   3.16262963f,  3.64737963f,  4.09837963f,
         4.42937963f,  4.61487963,   2.95130693f,  3.13680693f,  3.46780693f,  3.91880693f,
         4.40355693,   4.71955693f,  5.20430693f,  5.65530693f,  5.98630693f,  6.17180693,
         5.20525069f,  5.39075069f,  5.72175069f,  6.17275069f,  6.65750069,   6.97350069f,
         7.45825069f,  7.90925069f,  8.24025069f,  8.42575069,   6.88975f,     7.07525f,
         7.40625f,     7.85725f,     8.342,        8.658f,       9.14275f,     9.59375f,
         9.92475f,     10.11025f,    8.57424931f,  8.75974931f,  9.09074931f,  9.54174931f,
         10.02649931,  10.34249931f, 10.82724931f, 11.27824931f, 11.60924931f, 11.79474931,
         10.82819307f, 11.01369307f, 11.34469307f, 11.79569307f, 12.28044307,  12.59644307f,
         13.08119307f, 13.53219307f, 13.86319307f, 14.04869307,  12.38512037f, 12.57062037f,
         12.90162037f, 13.35262037f, 13.83737037,  14.15337037f, 14.63812037f, 15.08912037f,
         15.42012037f, 15.60562037,  13.32442078f, 13.50992078f, 13.84092078f, 14.29192078f,
         14.77667078,  15.09267078f, 15.57742078f, 16.02842078f, 16.35942078f, 16.54492078});
    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_nearest_round_prefer_floor_half_pixel)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/resize11_up_sizes_nearest_round_prefer_floor_half_pixel.prototxt"));

    const Shape expected_output_shape{1, 1, 7, 8};
    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> input_data{1.0f, 2.0f, 3.0f, 4.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(
        expected_output_shape,
        {1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f,
         2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f, 3.0f, 3.0f,
         3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f, 3.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f});
    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_nearest_prefer_ceil_asymmetric)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/resize11_up_sizes_nearest_prefer_ceil_asymmetric.prototxt"));

    const Shape expected_output_shape{1, 1, 8, 8};
    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> input_data{1.0f,
                                  2.0f,
                                  3.0f,
                                  4.0f,
                                  5.0f,
                                  6.0f,
                                  7.0f,
                                  8.0f,
                                  9.0f,
                                  10.0f,
                                  11.0f,
                                  12.0f,
                                  13.0f,
                                  14.0f,
                                  15.0f,
                                  16.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(
        expected_output_shape,
        {
            1.0f,  2.0f,  2.0f,  3.0f,  3.0f,  4.0f,  4.0f,  4.0f,  5.0f,  6.0f,  6.0f,
            7.0f,  7.0f,  8.0f,  8.0f,  8.0f,  5.0f,  6.0f,  6.0f,  7.0f,  7.0f,  8.0f,
            8.0f,  8.0f,  9.0f,  10.0f, 10.0f, 11.0f, 11.0f, 12.0f, 12.0f, 12.0f, 9.0f,
            10.0f, 10.0f, 11.0f, 11.0f, 12.0f, 12.0f, 12.0f, 13.0f, 14.0f, 14.0f, 15.0f,
            15.0f, 16.0f, 16.0f, 16.0f, 13.0f, 14.0f, 14.0f, 15.0f, 15.0f, 16.0f, 16.0f,
            16.0f, 13.0f, 14.0f, 14.0f, 15.0f, 15.0f, 16.0f, 16.0f, 16.0f,
        });
    test_case.run_with_tolerance_as_fp(2.0e-2f);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_nearest_ceil_half_pixel)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/resize11_up_sizes_nearest_ceil_half_pixel.prototxt"));

    const Shape expected_output_shape{1, 1, 8, 8};
    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> input_data{1.0f,
                                  2.0f,
                                  3.0f,
                                  4.0f,
                                  5.0f,
                                  6.0f,
                                  7.0f,
                                  8.0f,
                                  9.0f,
                                  10.0f,
                                  11.0f,
                                  12.0f,
                                  13.0f,
                                  14.0f,
                                  15.0f,
                                  16.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(
        expected_output_shape,
        {1.0f,  2.0f,  2.0f,  3.0f,  3.0f,  4.0f,  4.0f,  4.0f,  5.0f,  6.0f,  6.0f,  7.0f,  7.0f,
         8.0f,  8.0f,  8.0f,  5.0f,  6.0f,  6.0f,  7.0f,  7.0f,  8.0f,  8.0f,  8.0f,  9.0f,  10.0f,
         10.0f, 11.0f, 11.0f, 12.0f, 12.0f, 12.0f, 9.0f,  10.0f, 10.0f, 11.0f, 11.0f, 12.0f, 12.0f,
         12.0f, 13.0f, 14.0f, 14.0f, 15.0f, 15.0f, 16.0f, 16.0f, 16.0f, 13.0f, 14.0f, 14.0f, 15.0f,
         15.0f, 16.0f, 16.0f, 16.0f, 13.0f, 14.0f, 14.0f, 15.0f, 15.0f, 16.0f, 16.0f, 16.0f});
    test_case.run_with_tolerance_as_fp(2.0e-2f);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_nearest_floor_align_corners)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/resize11_up_sizes_nearest_floor_align_corners.prototxt"));

    const Shape expected_output_shape{1, 1, 8, 8};
    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> input_data{1.0f,
                                  2.0f,
                                  3.0f,
                                  4.0f,
                                  5.0f,
                                  6.0f,
                                  7.0f,
                                  8.0f,
                                  9.0f,
                                  10.0f,
                                  11.0f,
                                  12.0f,
                                  13.0f,
                                  14.0f,
                                  15.0f,
                                  16.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(
        expected_output_shape,
        {1.0f,  1.0f,  1.0f,  2.0f,  2.0f,  3.0f,  3.0f,  4.0f,  1.0f,  1.0f,  1.0f,  2.0f, 2.0f,
         3.0f,  3.0f,  4.0f,  1.0f,  1.0f,  1.0f,  2.0f,  2.0f,  3.0f,  3.0f,  4.0f,  5.0f, 5.0f,
         5.0f,  6.0f,  6.0f,  7.0f,  7.0f,  8.0f,  5.0f,  5.0f,  5.0f,  6.0f,  6.0f,  7.0f, 7.0f,
         8.0f,  9.0f,  9.0f,  9.0f,  10.0f, 10.0f, 11.0f, 11.0f, 12.0f, 9.0f,  9.0f,  9.0f, 10.0f,
         10.0f, 11.0f, 11.0f, 12.0f, 13.0f, 13.0f, 13.0f, 14.0f, 14.0f, 15.0f, 15.0f, 16.0f});
    test_case.run_with_tolerance_as_fp(2.0e-2f);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_down_sizes_tf_half_pixel)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize11_down_sizes_tf_half_pixel.prototxt"));

    const Shape expected_output_shape{1, 1, 3, 2};
    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> input_data{1.0f,
                                  2.0f,
                                  3.0f,
                                  4.0f,
                                  5.0f,
                                  6.0f,
                                  7.0f,
                                  8.0f,
                                  9.0f,
                                  10.0f,
                                  11.0f,
                                  12.0f,
                                  13.0f,
                                  14.0f,
                                  15.0f,
                                  16.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(expected_output_shape,
                                         {6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f});
    test_case.run_with_tolerance_as_fp(2.0e-2f);
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

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
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

    const auto inf = std::numeric_limits<float>::infinity();
    std::vector<float> output{0.3132616579532623291,
                              0.6931471824645996094,
                              1.313261628150939941,
                              10.0000457763671875,
                              inf,
                              0.0,
                              inf,
                              0.0,
                              0.6931471824645996094,
                              0.6931471824645996094,
                              0.6931471824645996094,
                              inf,
                              0.0};

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_argmax_int32)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/argmax_int32.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<std::int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    test_case.add_expected_output<std::int32_t>({1, 1, 1, 1, 1, 1});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_argmin_int32)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/argmin_int32.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<std::int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    test_case.add_expected_output<std::int32_t>({0, 0, 0, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_argmax_float)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/argmax_float.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({4, 0.1, 2, 3, -3, 1, -0.9, 0, 1, 2, 3, 0});
    test_case.add_expected_output<std::int64_t>({0, 3, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_argmin_float)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/argmin_float.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({4, 0.1, 2, 3, -3, 1, -0.9, 0, 1, 2, 3, 0});
    test_case.add_expected_output<std::int64_t>({1, 1, 0, 2});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_argmax_select_last_index)
{
    try
    {
        auto function = onnx_import::import_onnx_model(
            file_util::path_join(SERIALIZED_ZOO, "onnx/argmax_select_last_index.prototxt"));
        FAIL() << "Expected exception was not thrown";
    }
    catch (const ngraph::ngraph_error& e)
    {
        EXPECT_HAS_SUBSTRING(
            e.what(),
            std::string(
                "Mode 'select_last_index=1' is not supported by current implementation of ArgMax"));
    }
    catch (...)
    {
        FAIL() << "Expected OnnxNodeValidationFailure exception was not thrown";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_argmin_select_last_index)
{
    try
    {
        auto function = onnx_import::import_onnx_model(
            file_util::path_join(SERIALIZED_ZOO, "onnx/argmin_select_last_index.prototxt"));
        FAIL() << "Expected exception was not thrown";
    }
    catch (const ngraph::ngraph_error& e)
    {
        EXPECT_HAS_SUBSTRING(
            e.what(),
            std::string(
                "Mode 'select_last_index=1' is not supported by current implementation of ArgMin"));
        std::string what{e.what()};
    }
    catch (...)
    {
        FAIL() << "Expected OnnxNodeValidationFailure exception was not thrown";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_top_k)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/top_k.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8});

    test_case.add_expected_output<float>(Shape{3, 3}, {0, 1, 2, 4, 5, 6, 8, 9, 10}); // values
    test_case.add_expected_output<std::int64_t>(Shape{3, 3},
                                                {0, 1, 2, 0, 1, 2, 3, 2, 1}); // indices
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_top_k_opset_11_const_k_smallest_negative_axis)
{
    auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/top_k_opset_11_const_k_smallest_negative_axis.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>(Shape{1, 3}, {1.0f, 2.5f, 4.3f});
    test_case.add_expected_output<float>(Shape{1, 3}, {0.0f, 1.5667993f, 2.13795861f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_asinh)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/asinh.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>(Shape{1, 3}, {-1.0f, 0.0f, 1.0f});
    test_case.add_expected_output<float>(Shape{1, 3}, {-0.88137358f, 0.0f, 0.88137358f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_atanh)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/atanh.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>(Shape{1, 3}, {-0.9f, 0.0f, 0.9f});
    test_case.add_expected_output<float>(Shape{1, 3}, {-1.4722194f, 0.0f, 1.4722194f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_sinh)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/sinh.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({-1.0f, 0.0f, 1.0f});
    test_case.add_expected_output<float>({-1.1752012f, 0.f, 1.1752012f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_cosh)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/cosh.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
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

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_shrink_float)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/shrink_float.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>(
        {-2.0f, -1.6f, -1.5f, -1.4f, -1.0f, 0.0f, 1.0f, 1.4f, 1.5f, 1.6f, 2.0f});
    test_case.add_expected_output<float>(
        Shape{11}, {-1.5f, -1.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.1f, 1.5f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_shrink_int)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/shrink_int.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<int>({-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5});
    test_case.add_expected_output<int>(Shape{11}, {-4, -3, -2, -1, 0, 0, 0, 1, 2, 3, 4});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_lp_norm_p1)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/lp_norm_p1.prototxt"));

    Shape data_shape{2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    std::iota(std::begin(data), std::end(data), 1);

    auto test_case = test::TestCase<TestEngine>(function);
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
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/lp_norm_p2.prototxt"));

    Shape data_shape{2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    std::iota(std::begin(data), std::end(data), 1);

    auto test_case = test::TestCase<TestEngine>(function);
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
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/lp_norm_default.prototxt"));

    Shape data_shape{2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    std::iota(std::begin(data), std::end(data), 1);

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(
        data_shape, {0.18257418f, 0.36514837f, 0.5477225f,  0.73029673f, 0.37904903f, 0.45485884f,
                     0.5306686f,  0.60647845f, 0.42616236f, 0.47351375f, 0.5208651f,  0.5682165f,
                     0.4469492f,  0.48132992f, 0.51571065f, 0.5500913f,  0.45862272f, 0.48560053f,
                     0.5125783f,  0.53955615f, 0.46609157f, 0.4882864f,  0.51048124f, 0.5326761f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_lp_norm_default_dynamic)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/lp_norm_default_dynamic.prototxt"));

    Shape data_shape{2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    std::iota(std::begin(data), std::end(data), 1);

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(function);
    test_case.add_input<float>(data_shape, data);
    test_case.add_expected_output<float>(
        data_shape, {0.18257418f, 0.36514837f, 0.5477225f,  0.73029673f, 0.37904903f, 0.45485884f,
                     0.5306686f,  0.60647845f, 0.42616236f, 0.47351375f, 0.5208651f,  0.5682165f,
                     0.4469492f,  0.48132992f, 0.51571065f, 0.5500913f,  0.45862272f, 0.48560053f,
                     0.5125783f,  0.53955615f, 0.46609157f, 0.4882864f,  0.51048124f, 0.5326761f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_instance_normalization)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/instance_norm.prototxt"));

    Shape data_shape{1, 2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    std::iota(std::begin(data), std::end(data), 1);

    auto test_case = test::TestCase<TestEngine>(function);

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
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/eye_like.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_expected_output<float>(
        Shape{3, 4}, {0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reverse_sequence_0_batch_1)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reverse_sequence_time_0_batch_1.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

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
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reverse_sequence_time_1_batch_0.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

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

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>(std::vector<float>{0, 1, 2, 3, 4, 5});
    test_case.add_input<float>(std::vector<float>{0, 1});
    test_case.add_expected_output<float>(Shape{3, 1}, std::vector<float>{1, 3, 5});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_mod)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/mod_sign.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<int64_t>({-8, 3, 4, 9, -17, 1});
    test_case.add_input<int64_t>({22, -13, 8, -3, 7, 2});
    test_case.add_expected_output<int64_t>(Shape{6}, {-8, 3, 4, 0, -3, 1});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_scatterND_param_i64_indices)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/scatter_nd_param_i64_indices.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    test_case.add_input<int64_t>({4, 3, 1, 7});
    test_case.add_input<float>({9.f, 10.f, 11.f, 12.f});
    test_case.add_expected_output<float>(Shape{8}, {1.f, 11.f, 3.f, 10.f, 9.f, 6.f, 7.f, 12.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_scatterND_const_i32_indices)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/scatter_nd_const_i32_indices.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    test_case.add_input<float>({9.f, 10.f, 11.f, 12.f});
    test_case.add_expected_output<float>(Shape{8}, {1.f, 11.f, 3.f, 10.f, 9.f, 6.f, 7.f, 12.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_gather_elements_float_1D)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/gather_elements_float_1D.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>(Shape{3}, {1, 2, 3});
    test_case.add_input<int64_t>(Shape{1}, {1});
    test_case.add_expected_output<float>(Shape{1}, {2});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_gather_elements_int8_axis_1)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/gather_elements_int8_axis_1.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<int8_t>(Shape{2, 2}, {1, 2, 3, 4});
    test_case.add_input<int32_t>(Shape{2, 2}, {0, 0, 1, 0});
    test_case.add_expected_output<int8_t>(Shape{2, 2}, {1, 1, 4, 3});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_gather_elements_int32_axis_0)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/gather_elements_int32_axis_0.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<int32_t>(Shape{3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    test_case.add_input<int64_t>(Shape{2, 3}, {1, 2, 0, 2, 0, 0});
    test_case.add_expected_output<int32_t>(Shape{2, 3}, {4, 8, 3, 7, 2, 3});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_gather_elements_float_negative_axis)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/gather_elements_float_negative_axis.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>(Shape{2, 2}, {1, 2, 3, 4});
    test_case.add_input<int64_t>(Shape{2, 2}, {1, 1, 1, 0});
    test_case.add_expected_output<float>(Shape{2, 2}, {2, 2, 4, 3});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_gather_elements_float_3D_axis_2)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/gather_elements_float_3D_axis_2.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>(Shape{2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    test_case.add_input<int64_t>(Shape{2, 2, 1}, {0, 1, 0, 1});
    test_case.add_expected_output<float>(Shape{2, 2, 1}, {1, 4, 5, 8});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_gatherND_int32)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/gatherND_int32.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<int32_t>({0, 1, 2, 3});
    test_case.add_input<int64_t>({1, 0});
    test_case.add_expected_output<int32_t>(Shape{2, 2}, {2, 3, 0, 1});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_gatherND_float)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/gatherND_float.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>({0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f});
    test_case.add_input<int64_t>({0, 1, 1, 0});
    test_case.add_expected_output<float>(Shape{2, 2}, {2.f, 3.f, 4.f, 5.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_pad_constant)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/pad_constant.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>({1.f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f});
    test_case.add_expected_output<float>(
        Shape{3, 4}, {0.f, 0.f, 1.f, 1.2f, 0.f, 0.f, 2.3f, 3.4f, 0.f, 0.f, 4.5f, 5.7f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_pow_float32_float32)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/pow_float32_float32.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f}); // base
    test_case.add_input<float>({3.5f});               // exponent

    test_case.add_expected_output<float>(Shape{1, 4}, {1.f, 11.313708f, 46.765373f, 128.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_pow_float32_int32)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/pow_float32_int32.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f}); // base
    test_case.add_input<int>({3});                    // exponent

    test_case.add_expected_output<float>(Shape{1, 4}, {1.f, 8.f, 27.f, 64.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_pow_int32_float32)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/pow_int32_float32.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<int>({1, 2, 3, 4}); // base
    test_case.add_input<float>({3.5f});     // exponent

    test_case.add_expected_output<int>(Shape{1, 4}, {1, 11, 46, 128});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reciprocal)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reciprocal.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    test_case.add_expected_output<float>(Shape{3, 2},
                                         {1.f, 1 / 2.f, 1 / 3.f, 1 / 4.f, 1 / 5.f, 1 / 6.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_round)
{
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/round.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>(
        {0.1f, 0.9f, 1.2f, 1.5f, 1.8f, 2.3f, 2.7f, -1.1f, -1.9f, -2.2f, -2.8f});
    test_case.add_expected_output<float>(
        {0.f, 1.f, 1.f, 2.f, 2.f, 2.f, 3.f, -1.f, -2.f, -2.f, -3.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_round_half_nearest_even)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/round_half_nearest_even.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>({0.5f, 2.5f, -1.5f, -2.5f});
    test_case.add_expected_output<float>({0.f, 2.f, -2.f, -2.f});

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

NGRAPH_TEST(${BACKEND_NAME}, onnx_upsample6_nearest_infer)
{
    // clang-format off
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/upsample6_nearest.prototxt"));
    // height_scale: 2.0
    // width_scale: 3.0
    // mode: nearest
    const Shape input_shape          {1, 1, 2, 2};
    const Shape expected_output_shape{1, 1, 4, 6};

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>(input_shape,
        {   1.f, 2.f,
            3.f, 4.f    });
    test_case.add_expected_output<float>(expected_output_shape,
        {   1.f, 1.f, 1.f, 2.f, 2.f, 2.f,
            1.f, 1.f, 1.f, 2.f, 2.f, 2.f,
            3.f, 3.f, 3.f, 4.f, 4.f, 4.f,
            3.f, 3.f, 3.f, 4.f, 4.f, 4.f    });
    test_case.run();
    // clang-format on
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_upsample6_bilinear_infer)
{
    // clang-format off
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/upsample6_bilinear.prototxt"));
    // height_scale: 2.0
    // width_scale: 3.0
    // mode: bilinear
    const Shape input_shape          {1, 1, 2, 2};
    const Shape expected_output_shape{1, 1, 4, 6};

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>(input_shape,
        {   1.f, 2.f,
            3.f, 4.f    });
    test_case.add_expected_output<float>(expected_output_shape,
        {   1.f,  4.f/3,  5.f/3, 2.f, 2.f, 2.f,
            2.f,  7.f/3,  8.f/3, 3.f, 3.f, 3.f,
            3.f, 10.f/3, 11.f/3, 4.f, 4.f, 4.f,
            3.f, 10.f/3, 11.f/3, 4.f, 4.f, 4.f  });
    test_case.run();
    // clang-format on
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_upsample6_dynamic)
{
    // clang-format off
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/upsample6_dynamic.prototxt"));
    // height_scale: 1.5
    // width_scale: 2.5
    // mode: nearest
    //
    //  X ───╤══> Reshape ──R──> Upsample ──> Y
    //  S ───┘

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(function);

    test_case.add_input<float>(Shape {4},                      // X
        {   1.f, 2.f, 3.f, 4.f  });
    test_case.add_input<int64_t>(Shape {4},    {1, 1, 2, 2});  // S
    test_case.add_expected_output<float>(Shape {1, 1, 3, 5},   // Y
        {   1.f, 1.f, 1.f, 2.f, 2.f,
            1.f, 1.f, 1.f, 2.f, 2.f,
            3.f, 3.f, 3.f, 4.f, 4.f    });
    test_case.run();
    // clang-format on
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_upsample8_nearest_infer)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/upsample8_nearest.prototxt"));

    // Input data shape (1, 1, 2, 2)
    // Scales attribute values {1.0, 1.0, 2.0, 3.0}
    // mode: nearest

    const Shape expected_output_shape{1, 1, 4, 6};
    auto test_case = test::TestCase<TestEngine>(function);
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

    const Shape expected_output_shape{1, 1, 4, 4};
    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({1.0, 2.0, 3.0, 4.0});
    test_case.add_expected_output<float>(
        expected_output_shape,
        {1.0, 1.5, 2.0, 2.0, 2.0, 2.5, 3.0, 3.0, 3.0, 3.5, 4.0, 4.0, 3.0, 3.5, 4.0, 4.0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_upsample9_scales_const_nearest_infer)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/upsample9_scales_const_nearest.prototxt"));

    // Input data shape (1, 1, 2, 2)
    // Input const scales values {1.0, 1.0, 2.0, 3.0}
    // mode: nearest

    const Shape expected_output_shape{1, 1, 4, 6};
    auto test_case = test::TestCase<TestEngine>(function);
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

    const Shape expected_output_shape{1, 1, 4, 4};
    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({1.0, 2.0, 3.0, 4.0});
    test_case.add_expected_output<float>(
        expected_output_shape,
        {1.0, 1.5, 2.0, 2.0, 2.0, 2.5, 3.0, 3.0, 3.0, 3.5, 4.0, 4.0, 3.0, 3.5, 4.0, 4.0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_image_scaler)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/image_scaler.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0});
    test_case.add_expected_output<float>(Shape{1, 2, 2, 2},
                                         {12.0, 14.0, 16.0, 18.0, 21.0, 41.0, 61.0, 81.0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_size_op_single)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/size_op_single.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>(Shape{2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    test_case.add_expected_output<int>(Shape{}, {6});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_size_op_graph_end)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/size_op_graph_end.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({1.0, 2.0, 3.0, 4.0});
    test_case.add_expected_output<int>(Shape{}, {4});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_size_op_graph_middle)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/size_op_graph_middle.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({1.0, 2.0, 3.0, 4.0});
    test_case.add_expected_output<float>(Shape{}, {4.0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_size_op_on_input_graph_middle)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/size_op_on_input_graph_middle.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>(Shape{1, 2, 4, 1, 3},
                               {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.});
    test_case.add_expected_output<float>(
        Shape{1, 2, 4, 1, 3}, {24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24.,
                               24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24.});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_empty_initializers_handling)
{
    // int this test the "scales" input of the Resize operator is set to an empty initializer
    // this input should be ignored since the "sizes" optional input is provided
    // and the inference should use the data from the latter
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/empty_initializers_handling.prototxt"));

    const Shape expected_output_shape{2, 1, 4, 8};
    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> input_data{2.0f, 4.0f, 1.0f, 3.0f, 7.0f, 8.0f, 9.0f, 6.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(
        expected_output_shape,
        {2.0f, 2.5f, 3.0f,  3.5f, 4.0f,  4.0f,  4.0f, 4.0f,  1.5f, 2.0f,  2.5f,  3.0f, 3.5f,
         3.5f, 3.5f, 3.5f,  1.0f, 1.5f,  2.0f,  2.5f, 3.0f,  3.0f, 3.0f,  3.0f,  1.0f, 1.5f,
         2.0f, 2.5f, 3.0f,  3.0f, 3.0f,  3.0f,  7.0f, 7.25f, 7.5f, 7.75f, 8.0f,  8.0f, 8.0f,
         8.0f, 8.0f, 7.75f, 7.5f, 7.25f, 7.0f,  7.0f, 7.0f,  7.0f, 9.0f,  8.25f, 7.5f, 6.75f,
         6.0f, 6.0f, 6.0f,  6.0f, 9.0f,  8.25f, 7.5f, 6.75f, 6.0f, 6.0f,  6.0f,  6.0f});

    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_roi_align_f32)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/roi_align_f32.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12.,
                                13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.,
                                26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38.,
                                39., 40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 50., 51.,
                                52., 53., 54., 55., 56., 57., 58., 59., 60., 61., 62., 63., 64.,
                                65., 66., 67., 68., 69., 70., 71., 72., 73., 74.});

    test_case.add_input<float>({7.,   5.,  7.,  5., -15., -15., -15., -15., -10., 21.,
                                -10., 21., 13., 8., 13.,  8.,   -14., 19.,  -14., 19.});

    test_case.add_input<int32_t>({0, 0, 0, 0, 0});
    test_case.add_expected_output<float>(
        Shape{5, 3, 3, 4},
        {2.95833f, 3.20833f, 3.45833f, 3.70833f, 4.625f,   4.875f,   5.125f,   5.375f,   6.29167f,
         6.54167f, 6.79167f, 7.04167f, 27.9583f, 28.2083f, 28.4583f, 28.7083f, 29.625f,  29.875f,
         30.125f,  30.375f,  31.2917f, 31.5417f, 31.7917f, 32.0417f, 52.9583f, 53.2083f, 53.4583f,
         53.7083f, 54.625f,  54.875f,  55.125f,  55.375f,  56.2917f, 56.5417f, 56.7917f, 57.0417f,
         0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,
         0.f,      0.f,      0.f,      25.f,     25.f,     25.f,     25.f,     25.f,     25.f,
         25.f,     25.f,     25.f,     25.f,     25.f,     25.f,     50.f,     50.f,     50.f,
         50.f,     50.f,     50.f,     50.f,     50.f,     50.f,     50.f,     50.f,     50.f,
         7.39583f, 7.39583f, 7.42708f, 7.64583f, 9.0625f,  9.0625f,  9.09375f, 9.3125f,  10.7292f,
         10.7292f, 10.7604f, 10.9792f, 32.3958f, 32.3958f, 32.4271f, 32.6458f, 34.0625f, 34.0625f,
         34.0938f, 34.3125f, 35.7292f, 35.7292f, 35.7604f, 35.9792f, 57.3958f, 57.3958f, 57.4271f,
         57.6458f, 59.0625f, 59.0625f, 59.0938f, 59.3125f, 60.7292f, 60.7292f, 60.7604f, 60.9792f,
         4.27083f, 4.52083f, 4.77083f, 5.02083f, 5.9375f,  6.1875f,  6.4375f,  6.6875f,  7.60417f,
         7.85417f, 8.10417f, 8.35417f, 29.2708f, 29.5208f, 29.7708f, 30.0208f, 30.9375f, 31.1875f,
         31.4375f, 31.6875f, 32.6042f, 32.8542f, 33.1042f, 33.3542f, 54.2708f, 54.5208f, 54.7708f,
         55.0208f, 55.9375f, 56.1875f, 56.4375f, 56.6875f, 57.6042f, 57.8542f, 58.1042f, 58.3542f,
         6.77083f, 6.77083f, 6.77083f, 6.80208f, 8.4375f,  8.4375f,  8.4375f,  8.46875f, 10.1042f,
         10.1042f, 10.1042f, 10.1354f, 31.7708f, 31.7708f, 31.7708f, 31.8021f, 33.4375f, 33.4375f,
         33.4375f, 33.4688f, 35.1042f, 35.1042f, 35.1042f, 35.1354f, 56.7708f, 56.7708f, 56.7708f,
         56.8021f, 58.4375f, 58.4375f, 58.4375f, 58.4688f, 60.1042f, 60.1042f, 60.1042f, 60.1354f});
    test_case.run_with_tolerance_as_fp(1.0e-4f);
}

NGRAPH_TEST(${BACKEND_NAME}, quant_dequant_pattern)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/quant_dequant_pattern.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);
    // scale == 3.0
    // zero point == 10
    test_case.add_input<float>({9.0, 10.0, 15.0, 20.0, 30.0});
    test_case.add_input<float>({1});
    test_case.add_expected_output<float>(Shape{5}, {9.0, 9.0, 15.0, 21.0, 30.0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, quant_dequant_pattern_axis)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/quant_dequant_pattern_axis.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);
    // axis = 1
    // scale == {2.0, 3.0, 4.0}
    // zero point == {10, 20, 30}
    test_case.add_input<float>({1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 40.0, 50.0, 100.0});
    test_case.add_expected_output<float>(Shape{3, 3}, {0, 3, 4, 10, 21, 32, 40, 51, 100});
    test_case.add_input<float>({1});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_logsoftmax_0D)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/softmax_0D.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({3.141592});
    test_case.add_expected_output<float>({0.0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_logsoftmax_1D)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/logsoftmax_1D.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>({-1.0f, 0.0f, 1.0f});
    test_case.add_expected_output<float>(Shape{3}, {-2.4076061, -1.407606, -0.407606});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_logsoftmax13_1D)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/logsoftmax13_1D.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>({-1.0f, 0.0f, 1.0f});
    test_case.add_expected_output<float>(Shape{3}, {-2.4076061, -1.407606, -0.407606});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_logsoftmax13_2D)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/logsoftmax13_2D.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>({0.0f, 1.0f, 2.0f, 3.0f, 10000, 10001, 10002, 10003});
    test_case.add_expected_output<float>(Shape{2, 4},
                                         {-3.4401896,
                                          -2.4401896,
                                          -1.4401896,
                                          -0.44018966,
                                          -3.4401896,
                                          -2.4401896,
                                          -1.4401896,
                                          -0.44018966});
    test_case.run_with_tolerance_as_fp();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_logsoftmax13_2D_reshape)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/logsoftmax13_2D.prototxt"));
    InferenceEngine::CNNNetwork net(function);
    InferenceEngine::ICNNNetwork::InputShapes shapes = {};
    InferenceEngine::SizeVector shape = {1, 1, 4000};
    shapes[net.getInputsInfo().begin()->first] = shape;
    EXPECT_NO_THROW(net.reshape(shapes));
    ASSERT_EQ(shape, net.getOutputsInfo().begin()->second->getDims());
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_hard_sigmoid)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/hard_sigmoid.prototxt"));

    const auto inf = std::numeric_limits<float>::infinity();
    const auto neg_inf = -std::numeric_limits<float>::infinity();

    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>({inf, neg_inf, 0.0f, 1.0f});
    test_case.add_expected_output<float>(Shape{4}, {1.0f, 0.0f, 0.5f, 0.699999988079071f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_mul_v6)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/mul_v6.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>({1.0f, 2.0f, 3.0f});
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(Shape{3}, {3.0f, 8.0f, 15.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_mul_v6_broadcast_axis_1)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/mul_v6_broadcast_axis_1.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    Shape shape{1, 3, 2, 2};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(
        shape, {3.0f, 6.0f, 9.0f, 12.0f, 20.0f, 24.0f, 28.0f, 32.0f, 45.0f, 50.0f, 55.0f, 60.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_mul_v6_broadcast_axes_1_2)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/mul_v6_broadcast_axes_1_2.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    Shape shape{1, 3, 2, 2};
    std::vector<float> A(shape_size(shape), -1.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    test_case.add_expected_output<float>(
        shape, {-3.f, -3.f, -4.f, -4.f, -5.f, -5.f, -6.f, -6.f, -7.f, -7.f, -8.f, -8.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_mul_v6_broadcast_no_axis)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/mul_v6_broadcast_no_axis.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    Shape shape{2, 2};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f});
    test_case.add_expected_output<float>(shape, {3.0f, 6.0f, 9.0f, 12.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_mul_v7)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/mul_v7.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>({1.0f, 2.0f, 3.0f});
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(Shape{3}, {3.0f, 8.0f, 15.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_mul_v7_broadcast)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/mul_v7_broadcast.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    Shape shape{1, 2, 3};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(shape, {3.0f, 8.0f, 15.0f, 12.0f, 20.0f, 30.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_add_v6_broadcast_axis_1)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/add_v6_broadcast_axis_1.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    Shape shape{1, 3, 2, 2};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(
        shape, {4.0f, 5.0f, 6.0f, 7.0f, 9.0f, 10.0f, 11.0f, 12.0f, 14.0f, 15.0f, 16.0f, 17.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_add_v6_broadcast_axes_1_2)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/add_v6_broadcast_axes_1_2.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    Shape shape{1, 3, 2, 2};
    std::vector<float> A(shape_size(shape), 0.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    test_case.add_expected_output<float>(
        shape, {3.f, 3.f, 4.f, 4.f, 5.f, 5.f, 6.f, 6.f, 7.f, 7.f, 8.f, 8.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_add_v6_broadcast_no_axis)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/add_v6_broadcast_no_axis.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    Shape shape{2, 2};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f});
    test_case.add_expected_output<float>(shape, {4.0f, 5.0f, 6.0f, 7.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_add_v7)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/add_v7.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>({1.0f, 2.0f, 3.0f});
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(Shape{3}, {4.0f, 6.0f, 8.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_sub_v6_broadcast_axis_1)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/sub_v6_broadcast_axis_1.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    Shape shape{1, 3, 2, 2};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(
        shape, {-2.0f, -1.0f, 0.0f, 1.0f, 1.0f, 2.0f, 3.0f, 4.0f, 4.0f, 5.0f, 6.0f, 7.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_sub_v6_broadcast_axes_1_2)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/sub_v6_broadcast_axes_1_2.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    Shape shape{1, 3, 2, 2};
    std::vector<float> A(shape_size(shape), 0.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    test_case.add_expected_output<float>(
        shape, {-3.f, -3.f, -4.f, -4.f, -5.f, -5.f, -6.f, -6.f, -7.f, -7.f, -8.f, -8.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_sub_v6_broadcast_no_axis)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/sub_v6_broadcast_no_axis.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    Shape shape{2, 2};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f});
    test_case.add_expected_output<float>(shape, {-2.0f, -1.0f, 0.0f, 1.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_sub_v7)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/sub_v7.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>({1.0f, 2.0f, 3.0f});
    test_case.add_input<float>({3.0f, 8.0f, 7.0f});
    test_case.add_expected_output<float>(Shape{3}, {-2.0f, -6.0f, -4.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_sub_v7_broadcast)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/sub_v7_broadcast.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    Shape shape{1, 2, 3};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(shape, {-2.0f, -2.0f, -2.0f, 1.0f, 1.0f, 1.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_div_v6_broadcast_axis_1)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/div_v6_broadcast_axis_1.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    Shape shape{1, 3, 2, 2};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(
        shape,
        {0.3333333f, 0.6666666f, 1.0f, 1.333333f, 1.25f, 1.5f, 1.75f, 2.0f, 1.8f, 2.0, 2.2f, 2.4f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_div_v6_broadcast_axes_1_2)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/div_v6_broadcast_axes_1_2.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    Shape shape{1, 3, 2, 2};
    std::vector<float> A(shape_size(shape), 840.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    test_case.add_expected_output<float>(
        shape,
        {280.f, 280.f, 210.f, 210.f, 168.f, 168.f, 140.f, 140.f, 120.f, 120.f, 105.f, 105.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_div_v6_broadcast_no_axis)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/div_v6_broadcast_no_axis.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    Shape shape{2, 2};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1);
    test_case.add_input<float>(A);
    test_case.add_input<float>({2.0f});
    test_case.add_expected_output<float>(shape, {0.5f, 1.0f, 1.5f, 2.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_div_v7)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/div_v7.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>({1.0f, 2.0f, 3.0f});
    test_case.add_input<float>({3.0f, 8.0f, 7.0f});
    test_case.add_expected_output<float>(Shape{3}, {0.3333333f, 0.25f, 0.4285714f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_div_v7_broadcast)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/div_v7_broadcast.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    Shape shape{1, 2, 3};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(shape, {0.3333333f, 0.5f, 0.6f, 1.3333333f, 1.25f, 1.2f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_dangling_parameter)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dangling_parameter.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>({-1.0f, 2.0f, -3.0f});
    test_case.add_expected_output<float>(Shape{3}, {1.0f, 2.0f, 3.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_clip_inbounds)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/test_clip_inbounds.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    const std::vector<int32_t> data{-1, 0, 1, -9999, 9999};
    test_case.add_input<int32_t>(data);
    test_case.add_expected_output<int32_t>(Shape{data.size()}, data);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_mvn_v6)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/mvn_v6.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>(
        {0.8439683,  0.5665144, 0.05836735, 0.02916367, 0.12964272, 0.5060197, 0.79538304,
         0.9411346,  0.9546573, 0.17730942, 0.46192095, 0.26480448, 0.6746842, 0.01665257,
         0.62473077, 0.9240844, 0.9722341,  0.11965699, 0.41356155, 0.9129373, 0.59330076,
         0.81929934, 0.7862604, 0.11799799, 0.69248444, 0.54119414, 0.07513223});
    test_case.add_expected_output<float>(
        Shape{3, 3, 3, 1},
        {1.3546423,  0.33053496, -1.5450814,  -1.2106764,  -0.8925952,  0.29888135, 0.38083088,
         0.81808794, 0.85865635, -1.1060555,  -0.05552877, -0.78310335, 0.83281356, -1.250282,
         0.67467856, 0.7669372,  0.9113869,   -1.6463585,  -0.23402764, 1.6092131,  0.42940593,
         1.2906139,  1.1860244,  -0.92945826, 0.0721334,   -0.38174,    -1.7799333});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dropout1_no_training_no_return_mask)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dropout1_no_training_no_return_mask.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    const std::vector<float> data(3 * 4 * 5, 2.0f);
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(Shape{3, 4, 5}, data);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dropout1_no_training_return_mask)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dropout1_no_training_return_mask.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    const std::vector<float> data(3 * 4 * 5, 2.0f);
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(Shape{3, 4, 5}, data);
    test_case.add_expected_output<int32_t>(
        Shape{3, 4, 5}, std::vector<int32_t>(3 * 4 * 5, 1)); // // bool converted to i32
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dropout7_no_return_mask)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dropout7_no_return_mask.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    const std::vector<float> data(3 * 4 * 5, 2.0f);
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(Shape{3, 4, 5}, data);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dropout12_no_training_no_return_mask)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dropout12_no_training_no_return_mask.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    const std::vector<float> data(3 * 4 * 5, 2.0f);
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(Shape{3, 4, 5}, data);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dropout12_no_training_return_mask)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dropout12_no_training_return_mask.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    const std::vector<float> data(3 * 4 * 5, 2.0f);
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(Shape{3, 4, 5}, data);
    test_case.add_expected_output<int32_t>(
        Shape{3, 4, 5}, std::vector<int32_t>(3 * 4 * 5, 1)); // bool converted to i32
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dropout12_no_traning_no_const_rato)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dropout12_no_traning_no_const_rato.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({1, 2, 3, 4});
    // test_case.add_input<float>(Shape{}, {0.5}); // ratio input is ignored

    test_case.add_expected_output<float>(Shape{1, 4}, {1., 2., 3., 4.});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dropout12_training_mode)
{
    try
    {
        auto function = onnx_import::import_onnx_model(
            file_util::path_join(SERIALIZED_ZOO, "onnx/dropout12_training_mode.prototxt"));
        FAIL() << "Expected exception was not thrown";
    }
    catch (const ngraph::ngraph_error& e)
    {
        EXPECT_HAS_SUBSTRING(e.what(),
                             std::string("Training mode is not supported for Dropout op"));
    }
    catch (...)
    {
        FAIL() << "Expected ngraph_error exception was not thrown";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dropout12_not_const_training_mode)
{
    try
    {
        auto function = onnx_import::import_onnx_model(file_util::path_join(
            SERIALIZED_ZOO, "onnx/dropout12_not_const_training_mode.prototxt"));
        FAIL() << "Expected exception was not thrown";
    }
    catch (const ngraph::ngraph_error& e)
    {
        EXPECT_HAS_SUBSTRING(e.what(),
                             std::string("Non-constant training_mode input is not supported."));
    }
    catch (...)
    {
        FAIL() << "Expected ngraph_error exception was not thrown";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_multiple_slices_last_layer)
{
    std::vector<float> data(1 * 30 * 320 * 320);
    std::fill(data.begin(), data.end(), 1);

    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/multiple_slices_last_layer.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> o1(1 * 320 * 320 * 21);
    std::fill(o1.begin(), o1.end(), 1);

    std::vector<float> o2(1 * 320 * 320 * 9);
    std::fill(o2.begin(), o2.end(), 1);

    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(Shape{1, 320, 320, 21}, o1);
    test_case.add_expected_output<float>(Shape{1, 320, 320, 9}, o2);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_softmax_crossentropy_loss_mean)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/softmax_crossentropy_loss_mean.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({0.54881352186203,
                                0.7151893377304077,
                                0.6027633547782898,
                                0.5448831915855408,
                                0.42365479469299316,
                                0.6458941102027893,
                                0.4375872015953064,
                                0.891772985458374,
                                0.9636627435684204,
                                0.3834415078163147,
                                0.7917250394821167,
                                0.5288949012756348,
                                0.5680445432662964,
                                0.9255966544151306,
                                0.07103605568408966});
    test_case.add_input<int64_t>({1, 4, 3});
    test_case.add_expected_output<float>(Shape{}, {1.561384797096252441});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_negativelog_likelihood_loss)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/negativelog_likelihood_loss.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({
        0.54881352186203,     0.7151893377304077, 0.6027633547782898,  0.5448831915855408,
        0.42365479469299316,  0.6458941102027893, 0.4375872015953064,  0.891772985458374,
        0.9636627435684204,   0.3834415078163147, 0.7917250394821167,  0.5288949012756348,
        0.5680445432662964,   0.9255966544151306, 0.07103605568408966, 0.08712930232286453,
        0.020218396559357643, 0.832619845867157,  0.7781567573547363,  0.8700121641159058,
        0.978618323802948,    0.7991585731506348, 0.4614793658256531,  0.7805292010307312,
        0.11827442795038223,  0.6399210095405579, 0.14335328340530396, 0.9446688890457153,
        0.5218483209609985,   0.4146619439125061,
    });
    test_case.add_input<int64_t>({3, 3, 2, 4, 2, 0});
    test_case.add_expected_output<float>(Shape{}, {-0.531306922435760498});
    test_case.run();
}
