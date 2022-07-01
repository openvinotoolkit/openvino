// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cpp/ie_cnn_network.h>

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
#include "common_test_utils/ngraph_test_utils.hpp"
#include "default_opset.hpp"
#include "engines_util/test_case.hpp"
#include "engines_util/test_engines.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "ngraph/pass/manager.hpp"
#include "onnx_import/core/null_node.hpp"
#include "onnx_import/onnx.hpp"
#include "onnx_import/onnx_utils.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"
#include "util/type_prop.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";
static std::string s_device = test::backend_name_to_device("${BACKEND_NAME}");

using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;

// ############################################################################ CORE TESTS
NGRAPH_TEST(${BACKEND_NAME}, onnx_test_test_case) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1});
    test_case.add_input<float>({2});
    test_case.add_input<float>({3});
    test_case.add_expected_output<float>(Shape{1}, {6});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_test_test_case_mutliple_inputs) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(Inputs{{1}, {2}, {3}});
    test_case.add_expected_output(Shape{1}, std::vector<float>{6});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_output_names_check) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/split_equal_parts_default.onnx"));

    std::size_t size = function->get_output_size();
    for (std::size_t i{0}; i < size; ++i) {
        std::shared_ptr<Node> node = function->get_output_op(i);
        EXPECT_EQ(node->get_friendly_name(), "output_" + std::to_string(i + 1) + "/sink_port_0");
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_node_names_check) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc.onnx"));

    // Filter out Add nodes from the function graph
    std::vector<std::shared_ptr<Node>> additions;
    auto ordered_ops = function->get_ordered_ops();
    std::copy_if(ordered_ops.begin(), ordered_ops.end(), std::back_inserter(additions), [](std::shared_ptr<Node> op) {
        return std::string(op->get_type_name()) == "Add";
    });

    EXPECT_EQ(additions.size(), 2);
    EXPECT_EQ(additions.at(0)->get_friendly_name(), "add_node1");
    EXPECT_EQ(additions.at(0)->get_output_tensor(0).get_names(), std::unordered_set<std::string>{"X"});
    EXPECT_EQ(additions.at(1)->get_friendly_name(), "add_node2");
    EXPECT_EQ(additions.at(1)->get_output_tensor(0).get_names(), std::unordered_set<std::string>{"Y"});
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_duplicated_output_name) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/duplicated_output_name.onnx"));
    EXPECT_EQ(function->get_output_size(), 2);

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(Inputs{{1}, {2}, {3}});
    test_case.add_expected_output(Shape{1}, std::vector<float>{6});
    test_case.add_expected_output(Shape{1}, std::vector<float>{6});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_duplicated_more_output_names) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/duplicated_more_output_names.onnx"));
    EXPECT_EQ(function->get_output_size(), 4);

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(Inputs{{1, 2}, {2}, {3}});
    test_case.add_expected_output(Shape{1}, std::vector<float>{6});
    test_case.add_expected_output(Shape{1}, std::vector<float>{6});
    test_case.add_expected_output(Shape{1}, std::vector<float>{7});
    test_case.add_expected_output(Shape{1}, std::vector<float>{6});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_binary_add_abc) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(Inputs{{1}, {2}, {3}});
    test_case.add_expected_output(Shape{1}, std::vector<float>{6});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_bool_const_op) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/bool_const_op.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output(std::vector<bool>{1, 0, 0, 1});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_bool_init_and) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/bool_init_and.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output(std::vector<bool>{1});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_bool_input_or) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/bool_input_or.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input(std::vector<bool>{true, false, true, false});
    test_case.add_input(std::vector<bool>{false, false, true, true});
    test_case.add_expected_output(std::vector<bool>{1, 0, 1, 1});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_bool_init_raw) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/bool_init_raw.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output(std::vector<bool>{true, false, true});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_add_abc_initializers) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc_initializers.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1, 2, 3, 4});
    test_case.add_expected_output<float>({3, 6, 9, 12});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_override_op) {
    onnx_import::register_operator("FalseAdd", 1, "", [](const onnx_import::Node& node) -> OutputVector {
        OutputVector ng_inputs{node.get_ng_inputs()};
        return {std::make_shared<ngraph::op::v1::Add>(ng_inputs.at(0), ng_inputs.at(1))};
    });

    onnx_import::register_operator("FalseAdd", 1, "", [](const onnx_import::Node& node) -> OutputVector {
        OutputVector ng_inputs{node.get_ng_inputs()};
        return {std::make_shared<ngraph::op::v1::Subtract>(ng_inputs.at(0), ng_inputs.at(1))};
    });

    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/override_op.onnx"));

    Inputs inputs;
    inputs.emplace_back(std::vector<float>{0.f, 1.f, 2.f, 3.f});
    inputs.emplace_back(std::vector<float>{3.f, 2.f, 1.f, 0.f});

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output<float>({-3.f, -1.f, 1.f, 3.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_import_non_existing_file) {
    try {
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/i.dont.exist"));
    } catch (const std::runtime_error& exc) {
        // asserts that an exception was thrown and that the error message contains the file name
        std::string msg{exc.what()};
        EXPECT_TRUE(msg.find("i.dont.exist") != std::string::npos);
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_unsupported_op) {
    try {
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/unsupported_op.onnx"));
        FAIL() << "Expected ngraph::ngraph_error";
    } catch (ngraph::ngraph_error const& err) {
        std::string what{err.what()};
        EXPECT_NE(what.find("OpenVINO does not support"), std::string::npos);
        EXPECT_NE(what.find("FakeOpName"), std::string::npos);
        EXPECT_NE(what.find("AnotherFakeOpName"), std::string::npos);
    } catch (...) {
        FAIL() << "Expected ngraph::ngraph_error";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_custom_op) {
    onnx_import::register_operator("AddQ", 1, "com.intel.ai", [](const onnx_import::Node& node) -> OutputVector {
        OutputVector ng_inputs{node.get_ng_inputs()};
        return {std::make_shared<ngraph::op::v1::Add>(ng_inputs.at(0), ng_inputs.at(1))};
    });

    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/custom_operator.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>({3.f, 6.f, 9.f, 12.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_custom_op_register_unregister) {
    onnx_import::register_operator("AddQ", 1, "com.intel.ai", [](const onnx_import::Node& node) -> OutputVector {
        OutputVector ng_inputs{node.get_ng_inputs()};
        return {std::make_shared<ngraph::op::v1::Add>(ng_inputs.at(0), ng_inputs.at(1))};
    });

    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/custom_operator.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>({3.f, 6.f, 9.f, 12.f});
    test_case.run();

    onnx_import::unregister_operator("AddQ", 1, "com.intel.ai");
    try {
        auto function =
            onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/custom_operator.onnx"));
        FAIL() << "Expected ngraph::ngraph_error";
    } catch (ngraph::ngraph_error const& err) {
        std::string what{err.what()};
        EXPECT_NE(what.find("Check 'unknown_operators.empty()' failed"), std::string::npos);
    } catch (...) {
        FAIL() << "Expected ngraph::ngraph_error";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_custom_op_default_domain) {
    onnx_import::register_operator("AddQ", 1, "com.intel.ai", [](const onnx_import::Node& node) -> OutputVector {
        OutputVector ng_inputs{node.get_ng_inputs()};
        return {std::make_shared<ngraph::op::v1::Add>(ng_inputs.at(0), ng_inputs.at(1))};
    });

    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/custom_operator_default_domain.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>({3.f, 6.f, 9.f, 12.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_is_op_supported) {
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
    onnx_import::register_operator("AddQ", 1, "com.intel.ai", [](const onnx_import::Node& node) -> OutputVector {
        OutputVector ng_inputs{node.get_ng_inputs()};
        return {std::make_shared<ngraph::op::v1::Add>(ng_inputs.at(0), ng_inputs.at(1))};
    });
    EXPECT_TRUE(onnx_import::is_operator_supported("AddQ", 1, "com.intel.ai"));
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_missing_op_domain) {
    onnx_import::register_operator("CustomAdd", 1, "custom.op", [](const onnx_import::Node& node) -> OutputVector {
        OutputVector ng_inputs{node.get_ng_inputs()};
        return {std::make_shared<ngraph::op::v1::Add>(ng_inputs.at(0), ng_inputs.at(1))};
    });

    EXPECT_TRUE(onnx_import::is_operator_supported("CustomAdd", 1, "custom.op"));

    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/missing_op_domain.onnx"));

    Inputs inputs;
    inputs.emplace_back(std::vector<float>{0.f, 1.f, 2.f, 3.f});
    inputs.emplace_back(std::vector<float>{0.f, 1.f, 2.f, 3.f});

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output<float>({0.f, 2.f, 4.f, 6.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_custom_op_in_supported_operators) {
    onnx_import::register_operator("CustomAdd", 1, "custom.op", [](const onnx_import::Node& node) -> OutputVector {
        OutputVector ng_inputs{node.get_ng_inputs()};
        return {std::make_shared<ngraph::op::v1::Add>(ng_inputs.at(0), ng_inputs.at(1))};
    });

    const auto& supported_ops = onnx_import::get_supported_operators(1, "custom.op");
    EXPECT_NE(std::find(std::begin(supported_ops), std::end(supported_ops), "CustomAdd"), std::end(supported_ops));
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_unknown_domain) {
    // the importer should not throw when it encounters an unknown domain in the model
    EXPECT_NO_THROW(onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/unknown_domain.onnx")));
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_op_in_unknown_domain) {
    try {
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/unknown_domain_add.onnx"));

        FAIL() << "The onnx_importer did not throw for unknown domain and op";
    } catch (const ngraph::ngraph_error& e) {
        const std::string msg = e.what();

        EXPECT_NE(msg.find("unknown.domain.Add"), std::string::npos)
            << "The error message should contain domain and op name: unknown.domain.Add";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_missing_input) {
    onnx_import::register_operator("TestMissingInOut",
                                   1,
                                   "com.intel.ai",
                                   [](const onnx_import::Node& node) -> OutputVector {
                                       OutputVector ng_inputs{node.get_ng_inputs()};
                                       Output<ngraph::Node> A = ng_inputs.at(0);
                                       Output<ngraph::Node> B = ng_inputs.at(1);
                                       Output<ngraph::Node> C = ng_inputs.at(2);

                                       A = std::make_shared<op::v1::Multiply>(A, C);
                                       if (!ngraph::op::is_null(B)) {
                                           B = std::make_shared<op::v1::Divide>(B, C);
                                       }

                                       C = std::make_shared<ngraph::op::v1::Add>(C, C);
                                       return {A, B, C};
                                   });

    onnx_import::register_operator("TestMissingIn",
                                   1,
                                   "com.intel.ai",
                                   [](const onnx_import::Node& node) -> OutputVector {
                                       OutputVector ng_inputs{node.get_ng_inputs()};
                                       std::shared_ptr<ngraph::Node> result =
                                           std::make_shared<ngraph::op::Constant>(element::f32,
                                                                                  ngraph::Shape{2, 2},
                                                                                  std::vector<float>{1, 1, 1, 1});

                                       for (const auto& ng_input : ng_inputs) {
                                           if (!ngraph::op::is_null(ng_input)) {
                                               result = std::make_shared<op::v1::Multiply>(ng_input, result);
                                           }
                                       }

                                       return {result};
                                   });

    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/missing_input.onnx"));

    Inputs inputs{{1, 2, 3, 4}, {5, 6, 7, 8}};

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output<float>({50, 144, 294, 512});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_initializer_wo_input) {
    // This test checks a model which has an initializer, but no input with the same name
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/initializer_wo_input.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({0, 1, 2, 3, 4, 5});
    test_case.add_expected_output<float>({0, 2, 6, 12, 20, 30});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_expand_function_dependency_to_created_subgraph) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/transformations/greater_or_equal.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{5}, {3.f, 5.f, 3.f, 3.f, 6.f});
    test_case.add_input<float>(Shape{5}, {1.f, 4.f, 3.f, 7.f, 8.f});
    test_case.add_expected_output<int32_t>(Shape{5}, {1, 1, 1, 0, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_expand_function_greater_or_equal_inside_if) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/transformations/greater_or_equal_inside_if.onnx"));

    auto test_case = test::TestCase(function, s_device);

    // case when condition == true and any(x >= y)
    // expected value == x * y
    std::vector<float> x(40, 2);
    std::vector<float> y(40);
    std::iota(y.begin(), y.end(), -20);
    std::vector<float> expected;
    std::transform(x.begin(), x.end(), y.begin(), std::back_inserter(expected), [](float i, float j) -> float {
        return i * j;
    });
    test_case.add_input<bool>({true});  // condition
    test_case.add_input<float>(x);
    test_case.add_input<float>(y);
    test_case.add_expected_output<float>(expected);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_expand_context_dependent_function) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/transformations/softmax_crossentropy_consumed.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{3, 5},
                               {0.54881352186203,
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
    test_case.add_input<int64_t>(Shape{3}, {1, 4, 3});
    test_case.add_expected_output<int32_t>(Shape{}, {1});
    test_case.run();
}

// ############################################################################ OPERATOR TESTS
NGRAPH_TEST(${BACKEND_NAME}, onnx_model_addmul_abc) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/addmul_abc.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({9, 10, 11, 12});
    test_case.add_input<float>({5, 6, 7, 8});
    test_case.add_input<float>({1, 2, 3, 4});
    test_case.add_expected_output<float>(Shape{1, 2, 2}, {46, 62, 80, 100});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_argmin_no_keepdims) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/argmin_no_keepdims.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({2, 1, 3, 10});
    test_case.add_expected_output<int64_t>(Shape{2}, {1, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_batch_norm_default) {
    // Batch Normalization with default parameters
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/batchnorm_default.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({-1.f, 0.f, 1.f, 2.f, 3.f, 4.f});  // data {1, 2, 1, 3}
    test_case.add_input<float>({1.f, 1.5f});                      // scale
    test_case.add_input<float>({0.f, 1.f});                       // bias
    test_case.add_input<float>({0.f, 3.f});                       // mean
    test_case.add_input<float>({1.f, 1.5f});                      // var
    test_case.add_expected_output<float>(Shape{1, 2, 1, 3},
                                         {-0.999995f, 0.f, 0.999995f, -0.22474074f, 1.f, 2.2247407f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_relu) {
    // Simple ReLU test
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/relu.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({-1, -2, 0, 1, 2, 3});
    test_case.add_expected_output<float>({0, 0, 0, 1, 2, 3});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_sum_opset1) {
    // Simple Sum test for opset1.
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/sum_opset1.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({3.f, 0.f, 2.f});
    test_case.add_input<float>({1.f, 3.f, 4.f});
    test_case.add_input<float>({2.f, 6.f, 6.f});
    test_case.add_expected_output<float>(Shape{3}, {6.f, 9.f, 12.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_sum) {
    // Simple Sum test for opset8.
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/sum.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({3.f});
    test_case.add_input<float>({1.f, 3.f, 4.f});
    test_case.add_input<float>({2.f, 6.f, 6.f});
    test_case.add_expected_output<float>(Shape{3}, {6.f, 12.f, 13.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_sum_one_input) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/sum_one_input.onnx"));

    // input data shape (3, )
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({3.f, 0.f, 2.f});
    test_case.add_expected_output<float>({3.f, 0.f, 2.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_cum_sum_1d) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/cum_sum_1d.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f});
    test_case.add_expected_output<float>(Shape{3}, {1.f, 3.f, 6.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_cum_sum_2d_axis_input) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/cum_sum_2d_axis_input.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    test_case.add_expected_output<float>(Shape{2, 3}, {1.f, 3.f, 6.f, 4.f, 9.f, 15.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_cum_sum_2d_dynamic_axis_input) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/cum_sum_2d_dynamic_axis_input.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    test_case.add_input<std::int32_t>({1});
    test_case.add_expected_output<float>(Shape{2, 3}, {1.f, 3.f, 6.f, 4.f, 9.f, 15.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_cum_sum_3d_exclusive_reverse) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/cum_sum_3d_exclusive_reverse.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.f,  2.f,  3.f,  4.f,  5.f,  6.f,  7.f,  8.f,  9.f,  10.f, 11.f, 12.f,
                                13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f});
    test_case.add_expected_output<float>(Shape{2, 3, 4},
                                         {13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f,
                                          0.f,  0.f,  0.f,  0.f,  0.f,  0.f,  0.f,  0.f,  0.f,  0.f,  0.f,  0.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_min_two_inputs_opset1) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/min_two_inputs_opset1.onnx"));

    // input data shape (3, )
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.f, 2.f, 1.f});
    test_case.add_input<float>({1.f, 4.f, 4.f});
    test_case.add_expected_output<float>({1.f, 2.f, 1.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_min_two_inputs) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/min_two_inputs.onnx"));

    // input data shape (3, )
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({2.f});
    test_case.add_input<float>({1.f, 4.f, 4.f});
    test_case.add_expected_output<float>({1.f, 2.f, 2.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_max_opset1) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/max_opset1.onnx"));

    // input data shape (3, )
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({3.f, 2.f, 1.f});
    test_case.add_input<float>({1.f, 4.f, 4.f});
    test_case.add_input<float>({2.f, 5.f, 3.f});

    test_case.add_expected_output<float>({3.f, 5.f, 4.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_max) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/max.onnx"));

    // input data shape (3, )
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.f, 4.f, 4.f});
    test_case.add_input<float>({3.f});
    test_case.add_input<float>({2.f, 5.f, 3.f});

    test_case.add_expected_output<float>({3.f, 5.f, 4.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_mean_opset1) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/mean_opset1.onnx"));

    // input data shape (3, )
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({3.f, 0.f, 2.f});
    test_case.add_input<float>({1.f, 3.f, 4.f});
    test_case.add_input<float>({2.f, 6.f, 6.f});

    test_case.add_expected_output<float>({2.f, 3.f, 4.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_mean) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/mean.onnx"));

    // input data shape (3, )
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({3.f});
    test_case.add_input<float>({1.f, 2.f, 5.f});
    test_case.add_input<float>({2.f, 7.f, 7.f});

    test_case.add_expected_output<float>({2.f, 4.f, 5.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_gemm_abc) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/gemm_abc.onnx"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 2>({{1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12}, {13, 14, 15, 16, 17, 18}}).get_vector());

    inputs.emplace_back(test::NDArray<float, 2>({{19, 20, 21, 22},
                                                 {23, 24, 25, 26},
                                                 {27, 28, 29, 30},
                                                 {31, 32, 33, 34},
                                                 {35, 36, 37, 38},
                                                 {39, 40, 41, 42}})
                            .get_vector());

    inputs.emplace_back(test::NDArray<float, 2>({{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}).get_vector());

    auto expected_output =
        test::NDArray<float, 2>({{340, 350.5, 361, 371.5}, {862, 890.5, 919, 947.5}, {1384, 1430.5, 1477, 1523.5}})
            .get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_matmul) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/matmul.onnx"));

    std::vector<std::vector<float>> inputs;

    inputs.emplace_back(test::NDArray<float, 2>({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}).get_vector());

    inputs.emplace_back(test::NDArray<float, 2>({{13, 14, 15}, {16, 17, 18}, {19, 20, 21}, {22, 23, 24}}).get_vector());

    auto expected_output = test::NDArray<float, 2>({{190, 200, 210}, {470, 496, 522}, {750, 792, 834}}).get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_softmax_0D) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/softmax_0D.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<float>({1.0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_softmax_1D) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/softmax_1D.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({-1.0, 0.0, 1.0});
    test_case.add_expected_output<float>({0.09003058, 0.24472848, 0.66524094});
    test_case.run();
}
namespace {
// common input for all Softmax 3D test cases (Shape = {3,4,5})
// clang-format off
const std::vector<float> SOFTMAX_INPUT = {
    2.75793882,  -0.50841322, 0.82013929,  -0.62409912, -0.96136118,
    0.21004745,  1.38337255,  1.19030397,  2.0940445,   -0.03551657,
    -0.78686039, 1.992782,    0.04300319,  -0.29230777, -0.56797112,
    -1.26732165, -0.61935399, 0.57670432,  0.92844898,  2.82469233,

    0.98721677,  -0.05100663, -1.21178917, -0.17530157, 1.40051805,
    -0.13259761, -1.14313018, 0.2673723,   -0.87996154, 1.29053106,
    1.55,        0.8396538,   1.20729817,  0.23727845,  -0.89113606,
    -1.70909842, 0.26460363,  -0.70566808, 2.383518,    1.07024615,

    -1.21722605, 0.82919357,  0.55765697,  0.12657686,  0.63432172,
    0.75425957,  -2.43721014, -1.24478184, 2.65316853,  1.19509542,
    -0.95523998, 0.5149006,   -0.01151649, 0.68327026,  -0.4589638,
    -0.46554745, 0.21055324,  0.39266729,  2.05098086,  1.83207919};
}  // namespace
// clang-format on

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_softmax_axis_0) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/softmax_axis_0.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(SOFTMAX_INPUT);

    // clang-format off
    test_case.add_expected_output<float>(
        Shape{3, 4, 5},
        {0.09683057, 0.00369363, 0.01394559, 0.00329012, 0.00234823,
         0.00757665, 0.02449322, 0.02019284, 0.04985249, 0.00592694,
         0.00279593, 0.04505148, 0.00641108, 0.00458466, 0.00348007,
         0.00172928, 0.00330577, 0.01093237, 0.01554086, 0.10351497,

         0.01648154, 0.00583583, 0.00182802, 0.00515374, 0.02491679,
         0.00537859, 0.00195794, 0.00802367, 0.00254737, 0.0223216,
         0.02893419, 0.0142204,  0.02053893, 0.00778581, 0.00251907,
         0.00111174, 0.00800149, 0.0030324,  0.06658917, 0.0179084,

         0.00181811, 0.01407243, 0.01072611, 0.0069699,  0.01158077,
         0.01305647, 0.00053677, 0.0017687,  0.08719896, 0.02028982,
         0.00236265, 0.01027717, 0.0060709,  0.01216173, 0.00388087,
         0.00385541, 0.00758048, 0.00909469, 0.04775123, 0.03836337});
    // clang-format on

    test_case.run(6);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_softmax_axis_1) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/softmax_axis_1.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(SOFTMAX_INPUT);

    // clang-format off
    test_case.add_expected_output<float>(
        Shape{3, 4, 5},
        {0.22757064, 0.00868076, 0.03277484, 0.00773243, 0.0055188,
         0.0178066,  0.05756383, 0.04745709, 0.11716303, 0.01392945,
         0.00657097, 0.10587974, 0.01506727, 0.01077484, 0.00817884,
         0.00406413, 0.00776921, 0.0256932,  0.03652405, 0.24328028,

         0.06217413, 0.02201481, 0.00689594, 0.01944171, 0.09399488,
         0.02028993, 0.00738604, 0.03026811, 0.00960958, 0.08420492,
         0.10914991, 0.05364435, 0.07748005, 0.02937079, 0.0095028,
         0.00419387, 0.03018442, 0.01143929, 0.2511977,  0.06755678,

         0.00587593, 0.04548053, 0.0346656,  0.02252594, 0.03742775,
         0.04219705, 0.00173478, 0.00571623, 0.2818174,  0.06557446,
         0.00763582, 0.03321466, 0.01962049, 0.03930537, 0.01254255,
         0.01246025, 0.02449929, 0.02939305, 0.15432668, 0.12398617});
    // clang-format on

    test_case.run(4);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_softmax_axis_1_opset11) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/softmax_axis_1_opset11.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(SOFTMAX_INPUT);

    // clang-format off
    test_case.add_expected_output<float>(
        Shape{3, 4, 5},
        {0.88890495, 0.04825497, 0.27088348, 0.04490523, 0.02037154,
         0.06955369, 0.31998834, 0.39223197, 0.68041159, 0.05141776,
         0.02566661, 0.5885689,  0.12453075, 0.06257374, 0.03019055,
         0.01587475, 0.0431878,  0.21235381, 0.21210944, 0.89802015,

         0.31752626, 0.19442629, 0.0546935,  0.06279221, 0.36823282,
         0.10362164, 0.06523066, 0.24006419, 0.03103672, 0.32987983,
         0.55743381, 0.473766,   0.61451431, 0.09486084, 0.03722801,
         0.02141829, 0.26657706, 0.090728,   0.81131024, 0.26465935,

         0.08619648, 0.43343993, 0.3877785,  0.04523505, 0.15625437,
         0.61900597, 0.01653285, 0.06394322, 0.56592636, 0.27376196,
         0.11201305, 0.31654337, 0.21947994, 0.07893034, 0.05236297,
         0.18278451, 0.23348385, 0.32879834, 0.30990825, 0.5176207});
    // clang-format on

    test_case.run(4);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_softmax_axis_negative_1_opset11) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/softmax_axis_negative_1_opset11.onnx"));

    auto test_case = test::TestCase(function);
    test_case.add_input<float>(SOFTMAX_INPUT);

    // clang-format off
    test_case.add_expected_output<float>(
        Shape{3, 4, 5},
        {0.80619484, 0.03075256, 0.1161086,  0.027393,   0.01955098,
         0.07012683, 0.22670066, 0.18689778, 0.4614171,  0.05485764,
         0.04486171, 0.7228683,  0.10286818, 0.07356264, 0.05583908,
         0.01280724, 0.02448298, 0.08096659, 0.11509769, 0.76664555,

         0.30399805, 0.10764059, 0.03371745, 0.09505949, 0.4595844,
         0.13369875, 0.04866969, 0.19944906, 0.0633215,  0.554861,
         0.39101103, 0.19217177, 0.27755913, 0.10521588, 0.03404216,
         0.01150354, 0.08279411, 0.03137731, 0.6890207,  0.18530433,

         0.0402528,  0.31156224, 0.23747502, 0.15431291, 0.25639707,
         0.10627912, 0.00436928, 0.01439711, 0.7097961,  0.16515835,
         0.06798343, 0.29571748, 0.17468554, 0.34994435, 0.11166911,
         0.03615172, 0.07108136, 0.08527993, 0.4477579,  0.35972902});
    // clang-format on

    test_case.run(6);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_softmax_axis_negative_1_opset13) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/softmax_axis_negative_1_opset13.onnx"));

    auto test_case = test::TestCase(function);
    test_case.add_input<float>(SOFTMAX_INPUT);

    // clang-format off
    test_case.add_expected_output<float>(
        Shape{3, 4, 5},
        {0.80619484, 0.03075256, 0.1161086,  0.027393,   0.01955098,
         0.07012683, 0.22670066, 0.18689778, 0.4614171,  0.05485764,
         0.04486171, 0.7228683,  0.10286818, 0.07356264, 0.05583908,
         0.01280724, 0.02448298, 0.08096659, 0.11509769, 0.76664555,

         0.30399805, 0.10764059, 0.03371745, 0.09505949, 0.4595844,
         0.13369875, 0.04866969, 0.19944906, 0.0633215,  0.554861,
         0.39101103, 0.19217177, 0.27755913, 0.10521588, 0.03404216,
         0.01150354, 0.08279411, 0.03137731, 0.6890207,  0.18530433,

         0.0402528,  0.31156224, 0.23747502, 0.15431291, 0.25639707,
         0.10627912, 0.00436928, 0.01439711, 0.7097961,  0.16515835,
         0.06798343, 0.29571748, 0.17468554, 0.34994435, 0.11166911,
         0.03615172, 0.07108136, 0.08527993, 0.4477579,  0.35972902});
    // clang-format on

    test_case.run(6);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_sub) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/sub.onnx"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 3>({{{1, 2, 3}}}).get_vector());

    inputs.emplace_back(test::NDArray<float, 3>({{{4, 5, 7}}}).get_vector());

    auto expected_output = test::NDArray<float, 3>({{{-3, -3, -4}}}).get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_div) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/div.onnx"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 3>({{{1, 2, 3}}}).get_vector());
    inputs.emplace_back(test::NDArray<float, 3>({{{1, 4, 12}}}).get_vector());

    auto expected_output = test::NDArray<float, 3>({{{1, 0.5, 0.25}}}).get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_add_bcast) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/add_bcast.onnx"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 3>({{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                                 {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                                 {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
                            .get_vector());

    inputs.emplace_back(test::NDArray<float, 1>({1, 2, 3, 4, 5}).get_vector());

    auto expected_output =
        test::NDArray<float, 4>({{{{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}},
                                  {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}},
                                  {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}}}})
            .get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_nonmaxsuppression_center_point_box_format) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/nonmaxsuppression_center_point_box_format.onnx"));

    auto test_case = test::TestCase(function, s_device);

    test_case.add_input(
        std::vector<float>({0.5f, 0.5f,  1.0f, 1.0f, 0.5f, 0.6f,  1.0f, 1.0f, 0.5f, 0.4f,   1.0f, 1.0f,
                            0.5f, 10.5f, 1.0f, 1.0f, 0.5f, 10.6f, 1.0f, 1.0f, 0.5f, 100.5f, 1.0f, 1.0f}));  // boxes
    test_case.add_input(std::vector<float>({0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f}));                        // scores
    test_case.add_input(std::vector<int64_t>({3}));   // max_output_boxes_per_class
    test_case.add_input(std::vector<float>({0.5f}));  // iou_threshold
    test_case.add_input(std::vector<float>({0.0f}));  // score_threshold

    test_case.add_expected_output<int64_t>(Shape{3, 3}, {0, 0, 3, 0, 0, 0, 0, 0, 5});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_nonmaxsuppression_single_box) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/nonmaxsuppression_single_box.onnx"));

    auto test_case = test::TestCase(function, s_device);

    test_case.add_input(std::vector<float>({0.0f, 0.0f, 1.0f, 1.0f}));  // boxes
    test_case.add_input(std::vector<float>({0.9f}));                    // scores
    test_case.add_input(std::vector<int64_t>({3}));                     // max_output_boxes_per_class
    test_case.add_input(std::vector<float>({0.5f}));                    // iou_threshold
    test_case.add_input(std::vector<float>({0.0f}));                    // score_threshold

    test_case.add_expected_output<int64_t>(Shape{1, 3}, {0, 0, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_log_sum) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_log_sum.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = test::NDArray<float, 4>({{{{2.77258872f}}}}).get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_log_sum_exp) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_log_sum_exp.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = test::NDArray<float, 4>({{{{3.77258872f}}}}).get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_l1) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_l1.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = test::NDArray<float, 4>({{{{16}}}}).get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_l2) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_l2.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = test::NDArray<float, 4>({{{{4}}}}).get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_max) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_max.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = test::NDArray<float, 4>({{{{16}}}}).get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_max_invalid_axes) {
    EXPECT_THROW(
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_max_invalid_axes.onnx")),
        ngraph::ngraph_error);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_mean) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_mean.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = test::NDArray<float, 4>({{{{1}}}}).get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(Shape{}, expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_min) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_min.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = test::NDArray<float, 4>({{{{1}}}}).get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_prod) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_prod.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = test::NDArray<float, 4>({{{{1}}}}).get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_sum) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_sum.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = test::NDArray<float, 4>({{{{16}}}}).get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_dynamic_rank_input) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_sum_dynamic_rank_input.onnx"));
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(
        Shape{1, 1, 4, 4},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

    test_case.add_expected_output<float>(Shape{1, 1, 1, 1}, {16.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_square) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_sum_square.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = test::NDArray<float, 4>({{{{16}}}}).get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_as_constant) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_sum_13_axes_as_constant.onnx"));

    Inputs inputs{test::NDArray<float, 4>({{{{1.0f, 1.0f, 1.0f, 1.0f},
                                             {1.0f, 1.0f, 1.0f, 1.0f},
                                             {1.0f, 1.0f, 1.0f, 1.0f},
                                             {1.0f, 1.0f, 1.0f, 1.0f}}}})
                      .get_vector()};

    auto test_case = test::TestCase(function, s_device);

    test_case.add_expected_output<float>(Shape{1, 1, 1, 1}, {16.0f});

    test_case.add_multiple_inputs(inputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_as_constant_single_axis) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_sum_13_axes_as_constant_single_axis.onnx"));

    Inputs inputs{test::NDArray<float, 3>({{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}}).get_vector()};

    auto test_case = test::TestCase(function, s_device);

    test_case.add_expected_output<float>(Shape{2, 1, 3}, {5.0f, 7.0f, 9.0f, 17.0f, 19.0f, 21.0f});

    test_case.add_multiple_inputs(inputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_as_constant_keepdims_off) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_sum_13_axes_as_constant_keepdims_off.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{test::NDArray<float, 4>({{{{1.0f, 1.0f, 1.0f, 1.0f},
                                             {1.0f, 1.0f, 1.0f, 1.0f},
                                             {1.0f, 1.0f, 1.0f, 1.0f},
                                             {1.0f, 1.0f, 1.0f, 1.0f}}}})
                      .get_vector()};

    auto test_case = test::TestCase(function, s_device);

    test_case.add_expected_output<float>(Shape{}, {16.0f});

    test_case.add_multiple_inputs(inputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_as_input) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_sum_13_axes_as_input.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.0f, 2.0f, 3.0f, 4.0f});
    test_case.add_input<int64_t>({1});

    test_case.add_expected_output<float>(Shape{2, 1}, {3.0f, 7.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_as_0_dim_input) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_sum_13_axes_as_0_dim_input.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

    test_case.add_expected_output<float>(Shape{3, 2, 2},
                                         {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_input_dynamic) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_sum_13_input_dynamic.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<int64_t>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

    test_case.add_expected_output<int64_t>(Shape{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {5});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_empty) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_sum_13_axes_empty.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

    test_case.add_expected_output<float>(Shape{1, 1, 1, 1}, {16.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_empty_dynamic_rank_input) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_sum_13_axes_empty_dynamic_rank_input.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(
        Shape{1, 1, 4, 4},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

    test_case.add_expected_output<float>(Shape{1, 1, 1, 1}, {16.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_empty_with_noop) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_sum_13_axes_empty_with_noop.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(
        {1.f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

    test_case.add_expected_output<float>(
        Shape{1, 1, 4, 4},
        {1.f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_empty_without_noop) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_sum_13_axes_empty_without_noop.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(
        {1.f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

    test_case.add_expected_output<float>(Shape{1, 1, 1, 1}, {16.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize10_asymertic_last_dim) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/resize10_asymertic_last_dim.onnx"));

    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input_data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(Shape{1, 1, 1, 19},
                                         {1.0f,
                                          1.0f,
                                          2.0f,
                                          2.0f,
                                          3.0f,
                                          3.0f,
                                          4.0f,
                                          4.0f,
                                          5.0f,
                                          5.0f,
                                          6.0f,
                                          6.0f,
                                          7.0f,
                                          7.0f,
                                          8.0f,
                                          8.0f,
                                          9.0f,
                                          9.0f,
                                          10.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize10_asymertic_dim_in_the_middle) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize10_asymertic_dim_in_the_middle.onnx"));

    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input_data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(Shape{1, 1, 19, 1},
                                         {1.0f,
                                          1.0f,
                                          2.0f,
                                          2.0f,
                                          3.0f,
                                          3.0f,
                                          4.0f,
                                          4.0f,
                                          5.0f,
                                          5.0f,
                                          6.0f,
                                          6.0f,
                                          7.0f,
                                          7.0f,
                                          8.0f,
                                          8.0f,
                                          9.0f,
                                          9.0f,
                                          10.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_empty_constant_as_input) {
    // this model contains a Constant node with an empty underlying tensor
    // this node is connected to the "roi" input of the Resize op but this input should be
    // ignored since the Resize coordinate_transformation_mode is set to asymmetric
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize11_empty_constant_as_input.onnx"));

    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input_data{1.0f, 3.0f, 4.0f, 8.0f, 6.0f, 2.0f, 7.0f, 11.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(
        Shape{1, 2, 4, 8},
        {1.0f, 1.5f, 2.0f, 2.5f,  3.0f,  3.0f,  3.0f,  3.0f,  2.5f, 3.25f, 4.0f, 4.75f, 5.5f,  5.5f,  5.5f,  5.5f,
         4.0f, 5.0f, 6.0f, 7.0f,  8.0f,  8.0f,  8.0f,  8.0f,  4.0f, 5.0f,  6.0f, 7.0f,  8.0f,  8.0f,  8.0f,  8.0f,

         6.0f, 5.0f, 4.0f, 3.0f,  2.0f,  2.0f,  2.0f,  2.0f,  6.5f, 6.5f,  6.5f, 6.5f,  6.5f,  6.5f,  6.5f,  6.5f,
         7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 11.0f, 11.0f, 11.0f, 7.0f, 8.0f,  9.0f, 10.0f, 11.0f, 11.0f, 11.0f, 11.0f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize10_down_scales_const_nearest) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize10_down_scales_const_nearest.onnx"));

    // Input data shape (1, 1, 2, 4)
    // Input const scales values {1.0, 1.0, 0.6, 0.6}
    // mode: linear

    Shape expected_output_shape{1, 1, 1, 2};
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    test_case.add_expected_output<float>(expected_output_shape, {1.0, 3.0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize10_up_scales_const_linear) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize10_up_scales_const_linear.onnx"));

    // Input data shape (1, 1, 2, 2)
    // Input const scales values {1.0, 1.0, 2.0, 2.0}
    // mode: nearest

    Shape expected_output_shape{1, 1, 4, 4};
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.0, 2.0, 3.0, 4.0});
    test_case.add_expected_output<float>(
        expected_output_shape,
        {1.0, 1.5, 2.0, 2.0, 2.0, 2.5, 3.0, 3.0, 3.0, 3.5, 4.0, 4.0, 3.0, 3.5, 4.0, 4.0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize10_up_scales_const_nearest) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize10_up_scales_const_nearest.onnx"));

    // Input data shape (1, 1, 2, 2)
    // Input const scales values {1.0, 1.0, 2.0, 3.0}
    // mode: linear

    Shape expected_output_shape{1, 1, 4, 6};
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.0, 2.0, 3.0, 4.0});
    test_case.add_expected_output<float>(expected_output_shape,
                                         {1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
                                          3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_down_scales_linear_asymmetric) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize11_down_scales_linear_asymmetric.onnx"));

    const Shape expected_output_shape{1, 1, 1, 2};
    auto test_case = test::TestCase(function, s_device);
    const size_t input_size = 8;
    std::vector<float> input_data(input_size);
    std::iota(std::begin(input_data), std::end(input_data), 1.0f);
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(expected_output_shape, {1.0f, 2.66666651f});

    test_case.run_with_tolerance_as_fp();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_scales_nearest_asymmetric_floor_dynamic_sizes) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize11_scales_nearest_asymmetric_floor_dynamic_scales.onnx"));

    const Shape expected_output_shape{2, 1, 4, 1};
    auto test_case = test::TestCase(function, s_device);
    const std::vector<float> input_data{1.0f, 3.0f, 4.0f, 8.0f, 6.0f, 2.0f, 7.0f, 11.0f};
    test_case.add_input<float>(input_data);
    test_case.add_input<float>(std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});  // roi
    test_case.add_input<float>(std::vector<float>{1.0f, 1.0f, 2.0f, 0.5f});                          // scales
    test_case.add_expected_output<float>(expected_output_shape, {1.0f, 1.0f, 4.0f, 4.0f, 6.0f, 6.0f, 7.0f, 7.0f});

    test_case.run_with_tolerance_as_fp();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_up_scales_linear_asymmetric) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize11_up_scales_linear_asymmetric.onnx"));

    const Shape expected_output_shape{2, 1, 4, 8};
    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input_data{1.0f, 3.0f, 4.0f, 8.0f, 6.0f, 2.0f, 7.0f, 11.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(
        expected_output_shape,
        {1.0f, 1.5f, 2.0f, 2.5f,  3.0f,  3.0f,  3.0f,  3.0f,  2.5f, 3.25f, 4.0f, 4.75f, 5.5f,  5.5f,  5.5f,  5.5f,
         4.0f, 5.0f, 6.0f, 7.0f,  8.0f,  8.0f,  8.0f,  8.0f,  4.0f, 5.0f,  6.0f, 7.0f,  8.0f,  8.0f,  8.0f,  8.0f,

         6.0f, 5.0f, 4.0f, 3.0f,  2.0f,  2.0f,  2.0f,  2.0f,  6.5f, 6.5f,  6.5f, 6.5f,  6.5f,  6.5f,  6.5f,  6.5f,
         7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 11.0f, 11.0f, 11.0f, 7.0f, 8.0f,  9.0f, 10.0f, 11.0f, 11.0f, 11.0f, 11.0f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_scales_nearest_asymmetric_floor) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize11_scales_nearest_asymmetric_floor.onnx"));

    const Shape expected_output_shape{2, 1, 4, 1};
    auto test_case = test::TestCase(function, s_device);
    const std::vector<float> input_data{1.0f, 3.0f, 4.0f, 8.0f, 6.0f, 2.0f, 7.0f, 11.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(expected_output_shape, {1.0f, 1.0f, 4.0f, 4.0f, 6.0f, 6.0f, 7.0f, 7.0f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_up_scales_cubic_align_corners) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize11_up_scales_cubic_align_corners.onnx"));

    const Shape expected_output_shape{1, 1, 8, 8};
    auto test_case = test::TestCase(function, s_device);
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
            1.0f,         1.34110787f,  1.80029155f,  2.32944606f,  2.67055394f,  3.19970845f,  3.65889213f,
            4.0f,         2.36443149f,  2.70553936f,  3.16472303f,  3.69387755f,  4.03498542f,  4.56413994f,
            5.02332362f,  5.36443149f,  4.20116618f,  4.54227405f,  5.00145773f,  5.53061224f,  5.87172012f,
            6.40087464f,  6.86005831f,  7.20116618f,  6.31778426f,  6.65889213f,  7.1180758f,   7.64723032f,
            7.98833819f,  8.51749271f,  8.97667638f,  9.31778426f,  7.68221574f,  8.02332362f,  8.48250729f,
            9.01166181f,  9.35276968f,  9.8819242f,   10.34110787f, 10.68221574f, 9.79883382f,  10.13994169f,
            10.59912536f, 11.12827988f, 11.46938776f, 11.99854227f, 12.45772595f, 12.79883382f, 11.63556851f,
            11.97667638f, 12.43586006f, 12.96501458f, 13.30612245f, 13.83527697f, 14.29446064f, 14.6355685f,
            13.0f,        13.34110787f, 13.80029155f, 14.32944606f, 14.67055394f, 15.19970845f, 15.65889213f,
            16.0f,
        });
    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_up_scales_tf_half_pixel) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize11_up_scales_tf_half_pixel.onnx"));

    const Shape expected_output_shape{1, 1, 8, 8};
    auto test_case = test::TestCase(function, s_device);
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
        {1.95703f, 2.43359f, 3.0625f,  3.46875f, 4.09766f, 4.57422f, 4.87109f, 4.80078f, 3.86328f, 4.33984f, 4.96875f,
         5.375f,   6.00391f, 6.48047f, 6.77734f, 6.70703f, 6.37891f, 6.85547f, 7.48438f, 7.89063f, 8.51953f, 8.99609f,
         9.29297f, 9.22266f, 8.00391f, 8.48047f, 9.10938f, 9.51563f, 10.1445f, 10.6211f, 10.918f,  10.8477f, 10.5195f,
         10.9961f, 11.625f,  12.0313f, 12.6602f, 13.1367f, 13.4336f, 13.3633f, 12.4258f, 12.9023f, 13.5313f, 13.9375f,
         14.5664f, 15.043f,  15.3398f, 15.2695f, 13.6133f, 14.0898f, 14.7188f, 15.125f,  15.7539f, 16.2305f, 16.5273f,
         16.457f,  13.332f,  13.8086f, 14.4375f, 14.8438f, 15.4727f, 15.9492f, 16.2461f, 16.1758f});
    test_case.run_with_tolerance_as_fp(2.0e-2f);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_all_attributes_default) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize11_up_sizes_all_attributes_default.onnx"));

    const Shape expected_output_shape{1, 1, 7, 8};
    auto test_case = test::TestCase(function, s_device);
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

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_sizes_nearest_asymmetric_floor) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize11_sizes_nearest_asymmetric_floor.onnx"));

    const Shape expected_output_shape{2, 1, 4, 1};
    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input_data{1.0f, 3.0f, 4.0f, 8.0f, 6.0f, 2.0f, 7.0f, 11.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(expected_output_shape, {1.0f, 1.0f, 4.0f, 4.0f, 6.0f, 6.0f, 7.0f, 7.0f});

    test_case.run_with_tolerance_as_fp();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_linear_asymmetric) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize11_up_sizes_linear_asymmetric.onnx"));

    const Shape expected_output_shape{2, 1, 4, 8};
    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input_data{2.0f, 4.0f, 1.0f, 3.0f, 7.0f, 8.0f, 9.0f, 6.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(
        expected_output_shape,
        {2.0f, 2.5f,  3.0f, 3.5f,  4.0f, 4.0f, 4.0f, 4.0f, 1.5f, 2.0f,  2.5f, 3.0f,  3.5f, 3.5f, 3.5f, 3.5f,
         1.0f, 1.5f,  2.0f, 2.5f,  3.0f, 3.0f, 3.0f, 3.0f, 1.0f, 1.5f,  2.0f, 2.5f,  3.0f, 3.0f, 3.0f, 3.0f,
         7.0f, 7.25f, 7.5f, 7.75f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 7.75f, 7.5f, 7.25f, 7.0f, 7.0f, 7.0f, 7.0f,
         9.0f, 8.25f, 7.5f, 6.75f, 6.0f, 6.0f, 6.0f, 6.0f, 9.0f, 8.25f, 7.5f, 6.75f, 6.0f, 6.0f, 6.0f, 6.0f});

    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_down_sizes_cubic_half_pixel) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize11_down_sizes_cubic_half_pixel.onnx"));

    const Shape expected_output_shape{1, 1, 3, 3};
    auto test_case = test::TestCase(function, s_device);
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
        {1.6307871, 3.0046299, 4.3784733, 7.1261587, 8.5, 9.873844, 12.621532, 13.995373, 15.369216});

    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_down_sizes_linear_pytorch_half_pixel) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize11_down_sizes_linear_pytorch_half_pixel.onnx"));

    const Shape expected_output_shape{1, 1, 3, 1};
    auto test_case = test::TestCase(function, s_device);
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

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_cubic_half_pixel) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize11_up_sizes_cubic_half_pixel.onnx"));

    const Shape expected_output_shape{1, 1, 9, 10};
    auto test_case = test::TestCase(function, s_device);
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
        {0.45507922f,  0.64057922f,  0.97157922f,  1.42257922f,  1.90732922,   2.22332922f,  2.70807922f,  3.15907922f,
         3.49007922f,  3.67557922,   1.39437963f,  1.57987963f,  1.91087963f,  2.36187963f,  2.84662963,   3.16262963f,
         3.64737963f,  4.09837963f,  4.42937963f,  4.61487963,   2.95130693f,  3.13680693f,  3.46780693f,  3.91880693f,
         4.40355693,   4.71955693f,  5.20430693f,  5.65530693f,  5.98630693f,  6.17180693,   5.20525069f,  5.39075069f,
         5.72175069f,  6.17275069f,  6.65750069,   6.97350069f,  7.45825069f,  7.90925069f,  8.24025069f,  8.42575069,
         6.88975f,     7.07525f,     7.40625f,     7.85725f,     8.342,        8.658f,       9.14275f,     9.59375f,
         9.92475f,     10.11025f,    8.57424931f,  8.75974931f,  9.09074931f,  9.54174931f,  10.02649931,  10.34249931f,
         10.82724931f, 11.27824931f, 11.60924931f, 11.79474931,  10.82819307f, 11.01369307f, 11.34469307f, 11.79569307f,
         12.28044307,  12.59644307f, 13.08119307f, 13.53219307f, 13.86319307f, 14.04869307,  12.38512037f, 12.57062037f,
         12.90162037f, 13.35262037f, 13.83737037,  14.15337037f, 14.63812037f, 15.08912037f, 15.42012037f, 15.60562037,
         13.32442078f, 13.50992078f, 13.84092078f, 14.29192078f, 14.77667078,  15.09267078f, 15.57742078f, 16.02842078f,
         16.35942078f, 16.54492078});
    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_cubic_half_pixel_dynamic_sizes) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize11_up_sizes_cubic_half_pixel_dynamic_sizes.onnx"));

    const Shape expected_output_shape{1, 1, 9, 10};
    auto test_case = test::TestCase(function, s_device);
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
    test_case.add_input<float>(std::vector<float>{1, 1, 9, 10});  // sizes
    test_case.add_expected_output<float>(
        expected_output_shape,
        {0.45507922f,  0.64057922f,  0.97157922f,  1.42257922f,  1.90732922,   2.22332922f,  2.70807922f,  3.15907922f,
         3.49007922f,  3.67557922,   1.39437963f,  1.57987963f,  1.91087963f,  2.36187963f,  2.84662963,   3.16262963f,
         3.64737963f,  4.09837963f,  4.42937963f,  4.61487963,   2.95130693f,  3.13680693f,  3.46780693f,  3.91880693f,
         4.40355693,   4.71955693f,  5.20430693f,  5.65530693f,  5.98630693f,  6.17180693,   5.20525069f,  5.39075069f,
         5.72175069f,  6.17275069f,  6.65750069,   6.97350069f,  7.45825069f,  7.90925069f,  8.24025069f,  8.42575069,
         6.88975f,     7.07525f,     7.40625f,     7.85725f,     8.342,        8.658f,       9.14275f,     9.59375f,
         9.92475f,     10.11025f,    8.57424931f,  8.75974931f,  9.09074931f,  9.54174931f,  10.02649931,  10.34249931f,
         10.82724931f, 11.27824931f, 11.60924931f, 11.79474931,  10.82819307f, 11.01369307f, 11.34469307f, 11.79569307f,
         12.28044307,  12.59644307f, 13.08119307f, 13.53219307f, 13.86319307f, 14.04869307,  12.38512037f, 12.57062037f,
         12.90162037f, 13.35262037f, 13.83737037,  14.15337037f, 14.63812037f, 15.08912037f, 15.42012037f, 15.60562037,
         13.32442078f, 13.50992078f, 13.84092078f, 14.29192078f, 14.77667078,  15.09267078f, 15.57742078f, 16.02842078f,
         16.35942078f, 16.54492078});
    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_nearest_round_prefer_floor_half_pixel) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize11_up_sizes_nearest_round_prefer_floor_half_pixel.onnx"));

    const Shape expected_output_shape{1, 1, 7, 8};
    auto test_case = test::TestCase(function, s_device);
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

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_nearest_prefer_ceil_asymmetric) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize11_up_sizes_nearest_prefer_ceil_asymmetric.onnx"));

    const Shape expected_output_shape{1, 1, 8, 8};
    auto test_case = test::TestCase(function, s_device);
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
            1.0f,  2.0f,  2.0f,  3.0f,  3.0f,  4.0f,  4.0f,  4.0f,  5.0f,  6.0f,  6.0f,  7.0f,  7.0f,
            8.0f,  8.0f,  8.0f,  5.0f,  6.0f,  6.0f,  7.0f,  7.0f,  8.0f,  8.0f,  8.0f,  9.0f,  10.0f,
            10.0f, 11.0f, 11.0f, 12.0f, 12.0f, 12.0f, 9.0f,  10.0f, 10.0f, 11.0f, 11.0f, 12.0f, 12.0f,
            12.0f, 13.0f, 14.0f, 14.0f, 15.0f, 15.0f, 16.0f, 16.0f, 16.0f, 13.0f, 14.0f, 14.0f, 15.0f,
            15.0f, 16.0f, 16.0f, 16.0f, 13.0f, 14.0f, 14.0f, 15.0f, 15.0f, 16.0f, 16.0f, 16.0f,
        });
    test_case.run_with_tolerance_as_fp(2.0e-2f);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_nearest_ceil_half_pixel) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize11_up_sizes_nearest_ceil_half_pixel.onnx"));

    const Shape expected_output_shape{1, 1, 8, 8};
    auto test_case = test::TestCase(function, s_device);
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

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_nearest_floor_align_corners) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize11_up_sizes_nearest_floor_align_corners.onnx"));

    const Shape expected_output_shape{1, 1, 8, 8};
    auto test_case = test::TestCase(function, s_device);
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
        {1.0f, 1.0f, 1.0f, 2.0f,  2.0f,  3.0f,  3.0f,  4.0f,  1.0f,  1.0f,  1.0f,  2.0f,  2.0f,  3.0f,  3.0f,  4.0f,
         1.0f, 1.0f, 1.0f, 2.0f,  2.0f,  3.0f,  3.0f,  4.0f,  5.0f,  5.0f,  5.0f,  6.0f,  6.0f,  7.0f,  7.0f,  8.0f,
         5.0f, 5.0f, 5.0f, 6.0f,  6.0f,  7.0f,  7.0f,  8.0f,  9.0f,  9.0f,  9.0f,  10.0f, 10.0f, 11.0f, 11.0f, 12.0f,
         9.0f, 9.0f, 9.0f, 10.0f, 10.0f, 11.0f, 11.0f, 12.0f, 13.0f, 13.0f, 13.0f, 14.0f, 14.0f, 15.0f, 15.0f, 16.0f});
    test_case.run_with_tolerance_as_fp(2.0e-2f);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_resize11_down_sizes_tf_half_pixel) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/resize11_down_sizes_tf_half_pixel.onnx"));

    const Shape expected_output_shape{1, 1, 3, 2};
    auto test_case = test::TestCase(function, s_device);
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
    test_case.add_expected_output<float>(expected_output_shape, {6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f});
    test_case.run_with_tolerance_as_fp(2.0e-2f);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_shape) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/shape.onnx"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 3>({{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                                 {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                                 {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
                            .get_vector());

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output<int64_t>({3, 4, 5});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_elu) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/elu.onnx"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 3>({{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
                                 {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
                                 {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    auto expected_output =
        test::NDArray<float, 3>(
            {{{-1.999753180391830f, -1.999329074744190f, -1.998176236068890f, -1.995042495646670f, -1.986524106001830f},
              {-1.963368722222530f, -1.900425863264270f, -1.729329433526770f, -1.264241117657120f, 0},
              {1, 2, 3, 4, 5},
              {6, 7, 8, 9, 10}},
             {{-1.963368722222530f, -1.900425863264270f, -1.729329433526770f, -1.264241117657120f, 0},
              {1, 2, 3, 4, 5},
              {6, 7, 8, 9, 10},
              {11, 12, 13, 14, 15}},
             {{1, 1, 1, 1, 1},
              {-1.264241117657120f, -1.264241117657120f, -1.264241117657120f, -1.264241117657120f, -1.264241117657120f},
              {0, 0, 0, 0, 0},
              {2, 2, 2, 2, 2}}})
            .get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_leaky_relu) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/leaky_relu.onnx"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 3>({{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
                                 {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
                                 {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    auto expected_output =
        test::NDArray<float, 3>(
            {{{-0.9f, -0.8f, -0.7f, -0.6f, -0.5f}, {-0.4f, -0.3f, -0.2f, -0.1f, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
             {{-0.4f, -0.3f, -0.2f, -0.1f, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
             {{1, 1, 1, 1, 1}, {-0.1f, -0.1f, -0.1f, -0.1f, -0.1f}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_prelu_nd) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/prelu.onnx"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 3>({{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
                                 {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
                                 {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    inputs.emplace_back(test::NDArray<float, 3>({{{1, 0, 1, 0, 1}, {0, 1, 0, 1, 0}, {1, 0, 1, 0, 1}, {0, 1, 0, 1, 0}},
                                                 {{0, 1, 0, 1, 0}, {1, 0, 1, 0, 1}, {0, 1, 0, 1, 0}, {1, 0, 1, 0, 1}},
                                                 {{1, 0, 1, 0, 1}, {0, 1, 0, 1, 0}, {1, 0, 1, 0, 1}, {0, 1, 0, 1, 0}}})
                            .get_vector());

    auto expected_output =
        test::NDArray<float, 3>({{{-9, 0, -7, 0, -5}, {0, -3, 0, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
                                 {{0, -3, 0, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
                                 {{1, 1, 1, 1, 1}, {0, -1, 0, -1, 0}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_prelu_batch_nd_elementwise) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/prelu_batch_nd.onnx"));

    Inputs inputs;
    // Shape{2, 3, 4, 5}
    inputs.emplace_back(std::vector<float>{
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.});

    // Shape{2, 3, 4, 5}
    std::vector<float> slope(shape_size(Shape{2, 3, 4, 5}));
    std::iota(std::begin(slope), std::end(slope), 0);
    inputs.emplace_back(slope);

    // Shape{2, 3, 4, 5}
    auto expected_output = std::vector<float>{
        -0.,   -1.,   -2.,   -3.,   -4.,   -5.,   -6.,   -7.,   -8.,   -9.,   -10.,  -11.,  -12.,  -13.,  -14.,
        -15.,  -16.,  -17.,  -18.,  -19.,  -20.,  -21.,  -22.,  -23.,  -24.,  -25.,  -26.,  -27.,  -28.,  -29.,
        -30.,  -31.,  -32.,  -33.,  -34.,  -35.,  -36.,  -37.,  -38.,  -39.,  -40.,  -41.,  -42.,  -43.,  -44.,
        -45.,  -46.,  -47.,  -48.,  -49.,  -50.,  -51.,  -52.,  -53.,  -54.,  -55.,  -56.,  -57.,  -58.,  -59.,
        -60.,  -61.,  -62.,  -63.,  -64.,  -65.,  -66.,  -67.,  -68.,  -69.,  -70.,  -71.,  -72.,  -73.,  -74.,
        -75.,  -76.,  -77.,  -78.,  -79.,  -80.,  -81.,  -82.,  -83.,  -84.,  -85.,  -86.,  -87.,  -88.,  -89.,
        -90.,  -91.,  -92.,  -93.,  -94.,  -95.,  -96.,  -97.,  -98.,  -99.,  -100., -101., -102., -103., -104.,
        -105., -106., -107., -108., -109., -110., -111., -112., -113., -114., -115., -116., -117., -118., -119.};

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_prelu_1d) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/prelu_1d.onnx"));

    Inputs inputs;
    // Shape{2, 3, 4, 5}
    inputs.emplace_back(std::vector<float>{
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.});

    // Shape{5}
    inputs.emplace_back(std::vector<float>{0, 1, 2, 3, 4});

    // Shape{2, 3, 4, 5}
    auto expected_output = std::vector<float>{
        -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4.,
        -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4.,
        -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4.,
        -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4.,
        -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4.,
        -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4.};

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_prelu_C_1_1) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/prelu_c_1_1.onnx"));

    Inputs inputs;
    // Shape{2, 3, 4, 5}
    inputs.emplace_back(std::vector<float>{
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.});

    // Shape{3, 1, 1}
    inputs.emplace_back(std::vector<float>{0, 1, 2});

    // Shape{2, 3, 4, 5}
    auto expected_output = std::vector<float>{
        -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2.,
        -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2.};

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_selu) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/selu.onnx"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 3>({{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
                                 {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
                                 {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    auto expected_output =
        test::NDArray<float, 3>(
            {{{-5.99925954117548f, -5.99798722423258f, -5.99452870820667f, -5.98512748694000f, -5.95957231800549f},
              {-5.89010616666759f, -5.70127758979282f, -5.18798830058032f, -3.79272335297135f, 0},
              {3, 6, 9, 12, 15},
              {18, 21, 24, 27, 30}},
             {{-5.89010616666759f, -5.70127758979282f, -5.18798830058032f, -3.79272335297135f, 0},
              {3, 6, 9, 12, 15},
              {18, 21, 24, 27, 30},
              {33, 36, 39, 42, 45}},
             {{3, 3, 3, 3, 3},
              {-3.79272335297135f, -3.79272335297135f, -3.79272335297135f, -3.79272335297135f, -3.79272335297135f},
              {0, 0, 0, 0, 0},
              {6, 6, 6, 6, 6}}})
            .get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_sigmoid) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/sigmoid.onnx"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 3>({{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
                                 {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
                                 {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    auto expected_output =
        test::NDArray<float, 3>(
            {{{0.00012339457598623f,
               0.00033535013046648f,
               0.00091105119440065f,
               0.00247262315663477f,
               0.00669285092428486f},
              {0.01798620996209160f, 0.04742587317756680f, 0.119202922022118f, 0.268941421369995f, 0.5f},
              {0.731058578630005f, 0.880797077977882f, 0.952574126822433f, 0.982013790037908f, 0.993307149075715f},
              {0.997527376843365f, 0.999088948805599f, 0.999664649869534f, 0.999876605424014f, 0.999954602131298f}},
             {{0.01798620996209160f, 0.04742587317756680f, 0.119202922022118f, 0.268941421369995f, 0.5f},
              {0.731058578630005f, 0.880797077977882f, 0.952574126822433f, 0.982013790037908f, 0.993307149075715f},
              {0.997527376843365f, 0.999088948805599f, 0.999664649869534f, 0.999876605424014f, 0.999954602131298f},
              {0.999983298578152f, 0.999993855825398f, 0.999997739675702f, 0.999999168471972f, 0.999999694097773f}},
             {{0.731058578630005f, 0.731058578630005f, 0.731058578630005f, 0.731058578630005f, 0.731058578630005f},
              {0.268941421369995f, 0.268941421369995f, 0.268941421369995f, 0.268941421369995f, 0.268941421369995f},
              {0.5f, 0.5f, 0.5f, 0.5f, 0.5f},
              {0.880797077977882f, 0.880797077977882f, 0.880797077977882f, 0.880797077977882f, 0.880797077977882f}}})
            .get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_tanh) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/tanh.onnx"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 3>({{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
                                 {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
                                 {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    auto expected_output =
        test::NDArray<float, 3>(
            {{{-0.999999969540041f, -0.999999774929676f, -0.999998336943945f, -0.999987711650796f, -0.999909204262595f},
              {-0.999329299739067f, -0.995054753686731f, -0.964027580075817f, -0.761594155955765f, 0},
              {0.761594155955765f, 0.964027580075817f, 0.995054753686731f, 0.999329299739067f, 0.999909204262595f},
              {0.999987711650796f, 0.999998336943945f, 0.999999774929676f, 0.999999969540041f, 0.999999995877693f}},
             {{-0.999329299739067f, -0.995054753686731f, -0.964027580075817f, -0.761594155955765f, 0},
              {0.761594155955765f, 0.964027580075817f, 0.995054753686731f, 0.999329299739067f, 0.999909204262595f},
              {0.999987711650796f, 0.999998336943945f, 0.999999774929676f, 0.999999969540041f, 0.999999995877693f},
              {0.999999999442106f, 0.999999999924497f, 0.999999999989782f, 0.999999999998617f, 0.999999999999813f}},
             {{0.761594155955765f, 0.761594155955765f, 0.761594155955765f, 0.761594155955765f, 0.761594155955765f},
              {-0.761594155955765f, -0.761594155955765f, -0.761594155955765f, -0.761594155955765f, -0.761594155955765f},
              {0, 0, 0, 0, 0},
              {0.964027580075817f, 0.964027580075817f, 0.964027580075817f, 0.964027580075817f, 0.964027580075817f}}})
            .get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_thresholded_relu) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/thresholded_relu.onnx"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 3>({{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
                                 {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
                                 {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    auto expected_output =
        test::NDArray<float, 3>({{{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 3, 4, 5}, {6, 7, 8, 9, 10}},
                                 {{0, 0, 0, 0, 0}, {0, 0, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
                                 {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}}})
            .get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_matmul_vec_ten3d) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/matmul_vec_ten3d.onnx"));

    Inputs inputs;
    inputs.emplace_back(std::vector<float>{0.f, 1.f});
    inputs.emplace_back(test::NDArray<float, 3>{{{0.f}, {1.f}}, {{2.f}, {3.f}}, {{4.f}, {5.f}}}.get_vector());

    auto expected_output = test::NDArray<float, 2>{{1.f}, {3.f}, {5.f}}.get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_softplus) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/softplus.onnx"));

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
                              100.0,
                              0.0,
                              1000.0,
                              0.0,
                              0.6931471824645996094,
                              0.6931471824645996094,
                              0.6931471824645996094,
                              inf,
                              0.0};

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_softplus_infinity) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/softplus.onnx"));

    std::vector<float> input(13, std::numeric_limits<float>::infinity());
    std::vector<float> expected_output(13, std::numeric_limits<float>::infinity());

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input(input);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_sum_opset8) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/sum_opset8.onnx"));

    Inputs inputs;
    inputs.emplace_back(std::vector<float>{1.0f, 2.0f, 3.0f});
    inputs.emplace_back(test::NDArray<float, 2>{{10.0f}, {20.0f}, {30.0f}}.get_vector());
    inputs.emplace_back(test::NDArray<float, 3>{{{100.0f}}, {{200.0f}}, {{300.0f}}}.get_vector());

    auto expected_output =
        test::NDArray<float, 3>{{{111.0f, 112.0f, 113.0f}, {121.0f, 122.0f, 123.0f}, {131.0f, 132.0f, 133.0f}},

                                {{211.0f, 212.0f, 213.0f}, {221.0f, 222.0f, 223.0f}, {231.0f, 232.0f, 233.0f}},

                                {{311.0f, 312.0f, 313.0f}, {321.0f, 322.0f, 323.0f}, {331.0f, 332.0f, 333.0f}}}
            .get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_argmax_int32) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/argmax_int32.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<std::int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    test_case.add_expected_output<std::int64_t>({1, 1, 1, 1, 1, 1});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_argmin_int32) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/argmin_int32.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<std::int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    test_case.add_expected_output<std::int64_t>({0, 0, 0, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_argmax_float) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/argmax_float.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({4, 0.1, 2, 3, -3, 1, -0.9, 0, 1, 2, 3, 0});
    test_case.add_expected_output<std::int64_t>({0, 3, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_argmin_float) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/argmin_float.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({4, 0.1, 2, 3, -3, 1, -0.9, 0, 1, 2, 3, 0});
    test_case.add_expected_output<std::int64_t>({1, 1, 0, 2});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_argmax_select_last_index) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/argmax_select_last_index.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{4, 3}, {1, 1, 1, 0.5, 3, 4, 0.5, 1, 1.1, 0, 3, 0});
    test_case.add_expected_output<std::int64_t>(Shape{1, 3}, {0, 3, 1});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_argmin_select_last_index) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/argmin_select_last_index.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{4, 3}, {1, 1, 1, 2, 3, 4, 2, 1, 1.1, 3, 3, 8});
    test_case.add_expected_output<std::int64_t>(Shape{4}, {2, 0, 1, 1});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_top_k) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/top_k.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    test_case.add_expected_output<float>(Shape{3, 3}, {3, 2, 1, 7, 6, 5, 11, 10, 9});       // values
    test_case.add_expected_output<std::int64_t>(Shape{3, 3}, {3, 2, 1, 3, 2, 1, 3, 2, 1});  // indices
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_top_k_opset_10) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/top_k_opset_10.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    test_case.add_input<int64_t>({3});

    test_case.add_expected_output<float>(Shape{3, 3}, {3, 2, 1, 7, 6, 5, 11, 10, 9});       // values
    test_case.add_expected_output<std::int64_t>(Shape{3, 3}, {3, 2, 1, 3, 2, 1, 3, 2, 1});  // indices
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_top_k_opset_10_const_k) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/top_k_opset_10_const_k.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

    test_case.add_expected_output<float>(Shape{3, 3}, {3, 2, 1, 7, 6, 5, 11, 10, 9});       // values
    test_case.add_expected_output<std::int64_t>(Shape{3, 3}, {3, 2, 1, 3, 2, 1, 3, 2, 1});  // indices
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_top_k_opset_11_const_k_smallest) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/top_k_opset_11_const_k_smallest.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8});

    test_case.add_expected_output<float>(Shape{3, 3}, {0, 1, 2, 4, 5, 6, 8, 9, 10});        // values
    test_case.add_expected_output<std::int64_t>(Shape{3, 3}, {0, 1, 2, 0, 1, 2, 3, 2, 1});  // indices
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_top_k_opset_11_const_k_smallest_negative_axis) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/top_k_opset_11_const_k_smallest_negative_axis.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8});

    test_case.add_expected_output<float>(Shape{3, 3}, {0, 1, 2, 4, 5, 6, 8, 9, 10});        // values
    test_case.add_expected_output<std::int64_t>(Shape{3, 3}, {0, 1, 2, 0, 1, 2, 3, 2, 1});  // indices
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_acosh) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/acosh.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{1, 3}, {1.0f, 2.5f, 4.3f});
    test_case.add_expected_output<float>(Shape{1, 3}, {0.0f, 1.5667993f, 2.13795861f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_asinh) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/asinh.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{1, 3}, {-1.0f, 0.0f, 1.0f});
    test_case.add_expected_output<float>(Shape{1, 3}, {-0.88137358f, 0.0f, 0.88137358f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_atanh) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/atanh.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{1, 3}, {-0.9f, 0.0f, 0.9f});
    test_case.add_expected_output<float>(Shape{1, 3}, {-1.4722194f, 0.0f, 1.4722194f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_sinh) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/sinh.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({-1.0f, 0.0f, 1.0f});
    test_case.add_expected_output<float>({-1.1752012f, 0.f, 1.1752012f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_cosh) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/cosh.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({-1.0f, 0.0f, 1.0f});
    test_case.add_expected_output<float>({1.54308069f, 1.f, 1.54308069f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_sign) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/sign.onnx"));

    Inputs inputs{std::vector<float>{-std::numeric_limits<float>::infinity(),
                                     -3.141592f,
                                     0.0f,
                                     2.71828f,
                                     std::numeric_limits<float>::infinity()}};

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output<float>({-1.0f, -1.0f, 0.0f, 1.0f, 1.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_one_hot_with_axis) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/one_hot_axis.onnx"));

    Inputs inputs{{1.0, 9.0, 2.0, 4.0}, {1.0, 3.0}};
    std::vector<float> expected_output{{1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                        1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0,
                                        1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_one_hot_without_axis) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/one_hot_no_axis.onnx"));

    std::vector<std::vector<std::int64_t>> inputs{{0, 7, 8}, {2, 5}};
    std::vector<std::int64_t> expected_output{5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                              2, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 2, 2, 2};

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_where) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/where.onnx"));

    // conditions tensor - 3x3x3
    auto condition =
        std::vector<int>{{0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0}};

    // 1x3 tensor of "1"
    auto x1 = std::vector<int>{1, 1, 1};
    // 3x1 tensor of "2"
    auto x2 = std::vector<int>{2, 2, 2};

    std::vector<std::vector<int>> inputs;
    inputs.push_back(std::move(condition));
    inputs.push_back(std::move(x1));
    inputs.push_back(std::move(x2));

    // y = 3x3x3
    std::vector<int> expected_output{2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2};

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_erf) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/erf.onnx"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 2>{
        {-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()},
        {-3.141592f, 0.0f},
        {0.5f, 1.0f}}.get_vector());

    const std::vector<float> expected_output =
        test::NDArray<float, 2>{{-1.0f, 1.0f}, {-0.99999112f, 0.0f}, {0.52049988f, 0.84270079f}}.get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_erf_int32) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/erf_int32.onnx"));

    const std::vector<std::vector<int32_t>> inputs{
        {-std::numeric_limits<int32_t>::max(), -1, 0, 1, std::numeric_limits<int32_t>::max()}};

    const std::vector<int32_t> expected_output{-1, -1, 0, 1, 1};

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_shrink_float) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/shrink_float.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({-2.0f, -1.6f, -1.5f, -1.4f, -1.0f, 0.0f, 1.0f, 1.4f, 1.5f, 1.6f, 2.0f});
    test_case.add_expected_output<float>(Shape{11},
                                         {-1.5f, -1.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.1f, 1.5f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_shrink_int) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/shrink_int.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<int>({-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5});
    test_case.add_expected_output<int>(Shape{11}, {-4, -3, -2, -1, 0, 0, 0, 1, 2, 3, 4});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_lp_norm_p1) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/lp_norm_p1.onnx"));

    Shape data_shape{2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    std::iota(std::begin(data), std::end(data), 1);

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(
        data_shape,
        {0.07142857f, 0.125f,  0.16666667f, 0.2f,        0.22727273f, 0.25f,   0.26923078f, 0.2857143f,
         0.3f,        0.3125f, 0.32352942f, 0.33333334f, 0.9285714f,  0.875f,  0.8333333f,  0.8f,
         0.77272725f, 0.75f,   0.7307692f,  0.71428573f, 0.7f,        0.6875f, 0.6764706f,  0.6666667f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_lp_norm_p2) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/lp_norm_p2.onnx"));

    Shape data_shape{2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    std::iota(std::begin(data), std::end(data), 1);

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(
        data_shape,
        {0.0766965f,  0.14142136f, 0.19611613f, 0.24253564f, 0.28216633f, 0.31622776f, 0.34570536f, 0.37139067f,
         0.39391932f, 0.41380295f, 0.4314555f,  0.4472136f,  0.9970545f,  0.98994946f, 0.9805807f,  0.97014254f,
         0.9593655f,  0.9486833f,  0.9383431f,  0.9284767f,  0.91914505f, 0.9103665f,  0.9021342f,  0.8944272f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_lp_norm_default) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/lp_norm_default.onnx"));

    Shape data_shape{2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    std::iota(std::begin(data), std::end(data), 1);

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(
        data_shape,
        {0.18257418f, 0.36514837f, 0.5477225f, 0.73029673f, 0.37904903f, 0.45485884f, 0.5306686f,  0.60647845f,
         0.42616236f, 0.47351375f, 0.5208651f, 0.5682165f,  0.4469492f,  0.48132992f, 0.51571065f, 0.5500913f,
         0.45862272f, 0.48560053f, 0.5125783f, 0.53955615f, 0.46609157f, 0.4882864f,  0.51048124f, 0.5326761f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_lp_norm_default_dynamic) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/lp_norm_default_dynamic.onnx"));

    Shape data_shape{2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    std::iota(std::begin(data), std::end(data), 1);

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(data_shape, data);
    test_case.add_expected_output<float>(
        data_shape,
        {0.18257418f, 0.36514837f, 0.5477225f, 0.73029673f, 0.37904903f, 0.45485884f, 0.5306686f,  0.60647845f,
         0.42616236f, 0.47351375f, 0.5208651f, 0.5682165f,  0.4469492f,  0.48132992f, 0.51571065f, 0.5500913f,
         0.45862272f, 0.48560053f, 0.5125783f, 0.53955615f, 0.46609157f, 0.4882864f,  0.51048124f, 0.5326761f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_instance_normalization) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/instance_norm.onnx"));

    Shape data_shape{1, 2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    std::iota(std::begin(data), std::end(data), 1);

    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>(data);
    test_case.add_input<float>(std::vector<float>{2.134f, 3.256f});
    test_case.add_input<float>(std::vector<float>{0.765f, 1.055f});
    test_case.add_expected_output<float>(
        data_shape,
        {-2.6335807f,  -2.015657f, -1.3977331f, -0.77980936f, -0.16188562f, 0.45603812f, 1.0739619f,  1.6918856f,
         2.3098092f,   2.927733f,  3.5456567f,  4.1635804f,   -4.130463f,   -3.1876516f, -2.2448401f, -1.3020288f,
         -0.35921717f, 0.5835942f, 1.5264057f,  2.469217f,    3.4120288f,   4.35484f,    5.2976513f,  6.240463f});
    const size_t tolerance_bits = 3;
    test_case.run(tolerance_bits);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_instance_normalization_dynamic) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/instance_norm_dynamic.onnx"));

    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input_data{1.f, 2.f, 3.f};
    test_case.add_input<float>(Shape{1, 3, 1, 1}, input_data);
    test_case.add_expected_output<float>(Shape{1, 3, 1, 1},
                                         {0.3341970741748809814, 0.3321160078048706055, 0.3407136797904968262});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_eye_like) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/eye_like.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{3, 4}, {5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f});
    test_case.add_expected_output<float>(Shape{3, 4}, {0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reverse_sequence_0_batch_1) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reverse_sequence_time_0_batch_1.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({0.f, 4.f, 8.f, 12.f, 1.f, 5.f, 9.f, 13.f, 2.f, 6.f, 10.f, 14.f, 3.f, 7.f, 11.f, 15.f});
    test_case.add_input<int>({4, 3, 2, 1});
    test_case.add_expected_output<float>(
        Shape{4, 4},
        {3.f, 6.f, 9.f, 12.f, 2.f, 5.f, 8.f, 13.f, 1.f, 4.f, 10.f, 14.f, 0.f, 7.f, 11.f, 15.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reverse_sequence_1_batch_0) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reverse_sequence_time_1_batch_0.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f});
    test_case.add_input<int>({1, 2, 3, 4});
    test_case.add_expected_output<float>(
        Shape{4, 4},
        {0.f, 1.f, 2.f, 3.f, 5.f, 4.f, 6.f, 7.f, 10.f, 9.f, 8.f, 11.f, 15.f, 14.f, 13.f, 12.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reverse_sequence_incorrect_batch_axis) {
    EXPECT_THROW(onnx_import::import_onnx_model(
                     file_util::path_join(SERIALIZED_ZOO, "onnx/reverse_sequence_incorrect_batch_axis.onnx")),
                 ngraph_error)
        << "ReverseSequence batch_axis attribute can only equal 0 or 1. Value of '2' is not "
           "accepted.";
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reverse_sequence_incorrect_time_axis) {
    EXPECT_THROW(onnx_import::import_onnx_model(
                     file_util::path_join(SERIALIZED_ZOO, "onnx/reverse_sequence_incorrect_time_axis.onnx")),
                 ngraph_error)
        << "ReverseSequence time_axis attribute can only equal 0 or 1. Value of '2' is not "
           "accepted.";
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reverse_sequence_time_and_batch_axis_equal) {
    EXPECT_THROW(onnx_import::import_onnx_model(
                     file_util::path_join(SERIALIZED_ZOO, "onnx/reverse_sequence_time_and_batch_axis_equal.onnx")),
                 ngraph_error)
        << "ReverseSequence 'time_axis' and 'batch_axis' can't be equal.";
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_matmul_float_type) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/matmul_float.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(std::vector<float>{0, 1, 2, 3, 4, 5});
    test_case.add_input<float>(std::vector<float>{0, 1});
    test_case.add_expected_output<float>(Shape{3, 1}, std::vector<float>{1, 3, 5});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_mod_sign) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/mod_sign.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<int32_t>({-4, 7, 5, 4, -7, 8});
    test_case.add_input<int32_t>({2, -3, 8, -2, 3, 5});
    test_case.add_expected_output<int32_t>(Shape{6}, {0, -2, 5, 0, 2, 3});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_mod_sign_i64) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/mod_sign_i64.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<int64_t>({-4, 7, 5, 4, -7, 8});
    test_case.add_input<int64_t>({2, -3, 8, -2, 3, 5});
    test_case.add_expected_output<int64_t>(Shape{6}, {0, -2, 5, 0, 2, 3});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_mod_sign_broadcast) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/mod_sign_broadcast.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<int32_t>({-8, 3, 4, 9, -17, 1});
    test_case.add_input<int32_t>({3});
    test_case.add_expected_output<int32_t>(Shape{6}, {1, 0, 1, 0, 1, 1});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_mod_sign_f32) {
    try {
        const auto function =
            onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/mod_sign_f32.onnx"));
        FAIL() << "Expected exception was not thrown";
    } catch (const ngraph::ngraph_error& e) {
        EXPECT_HAS_SUBSTRING(
            e.what(),
            std::string("If the input type is floating point, then `fmod` attribute must be set to 1."));
    } catch (...) {
        FAIL() << "Expected ngraph_error exception was not thrown";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_mod_sign_fmod) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/mod_sign_fmod.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<int32_t>({-8, 3, 4, 9, -17, 1});
    test_case.add_input<int32_t>({22, -13, 8, -3, 7, 2});
    test_case.add_expected_output<int32_t>(Shape{6}, {-8, 3, 4, 0, -3, 1});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_mod_sign_fmod_broadcast) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/mod_sign_fmod_broadcast.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<int32_t>({-8, 3, 4, 9, -17, 1});
    test_case.add_input<int32_t>({3});
    test_case.add_expected_output<int32_t>(Shape{6}, {-2, 0, 1, 0, -2, 1});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_mod_sign_fmod_f32) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/mod_sign_fmod_f32.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({-4.3, 7.2, 5.0, 4.3, -7.2, 8.0});
    test_case.add_input<float>({2.1, -3.4, 8.0, -2.1, 3.4, 5.0});
    test_case.add_expected_output<float>(Shape{6}, {-0.10000038, 0.39999962, 5., 0.10000038, -0.39999962, 3.});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_mod_incorrect_fmod) {
    try {
        const auto function =
            onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/mod_incorrect_fmod.onnx"));
        FAIL() << "Expected exception was not thrown";
    } catch (const ngraph::ngraph_error& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string("Unsupported value of 'fmod' attribute (should be: 0 or 1)"));
    } catch (...) {
        FAIL() << "Expected ngraph_error exception was not thrown";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_scatterND_param_i64_indices) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/scatter_nd_param_i64_indices.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    test_case.add_input<int64_t>({4, 3, 1, 7});
    test_case.add_input<float>({9.f, 10.f, 11.f, 12.f});
    test_case.add_expected_output<float>(Shape{8}, {1.f, 11.f, 3.f, 10.f, 9.f, 6.f, 7.f, 12.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_scatterND_const_i32_indices) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/scatter_nd_const_i32_indices.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    test_case.add_input<float>({9.f, 10.f, 11.f, 12.f});
    test_case.add_expected_output<float>(Shape{8}, {1.f, 11.f, 3.f, 10.f, 9.f, 6.f, 7.f, 12.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_gather_float_1D) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/gather_float_1D.onnx"));
    auto test_case = test::TestCase(function, s_device);

    // clang-format off
    test_case.add_input<float>(Shape{3},
        {   5, 6, 7 });
    test_case.add_input<int64_t>(Shape{2, 2},
        {   0, 1,
            1, 2    });
    test_case.add_expected_output<float>(Shape{2, 2},
        {   5, 6,
            6, 7    });
    // clang-format on

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_gather_int32_3D_axis_1) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/gather_int32_3D_axis_1.onnx"));
    auto test_case = test::TestCase(function, s_device);

    // clang-format off
    test_case.add_input<int32_t>(Shape{2, 2, 2},
        {   1, 2,
            3, 4,

            5, 6,
            7, 8    });
    test_case.add_input<int32_t>(Shape{4, 1},
        {   0,
            1,
            1,
            0       });
    test_case.add_expected_output<int32_t>(Shape{2, 4, 1, 2},
        {   1, 2,
            3, 4,
            3, 4,
            1, 2,

            5, 6,
            7, 8,
            7, 8,
            5, 6     });
    // clang-format on

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_gather_int8_3D_axis_neg_1) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/gather_int8_3D_axis_neg_1.onnx"));
    auto test_case = test::TestCase(function, s_device);

    // clang-format off
    test_case.add_input<int8_t>(Shape{2, 2, 2},
        {   1, 2,
            3, 4,

            5, 6,
            7, 8            });
    test_case.add_input<int32_t>(Shape{4, 1},
        {   0, 1, 1, 0      });
    test_case.add_expected_output<int8_t>(Shape{2, 2, 4, 1},
        {   1, 2, 2, 1,
            3, 4, 4, 3,

            5, 6, 6, 5,
            7, 8, 8, 7      });
    // clang-format on

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_gather_float_2D_neg_indices) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/gather_float_2D_axis_1.onnx"));
    auto test_case = test::TestCase(function, s_device);

    // clang-format off
    test_case.add_input<float>(Shape{3, 3},
        {   0.0, 0.1, 0.2,
            1.0, 1.1, 1.2,
            2.0, 2.1, 2.2   });
    test_case.add_input<int64_t>(Shape{2, 2},
        {   -1, -2,
            -3, -2      });
    test_case.add_expected_output<float>(Shape{3, 2, 2},
        {
            0.2, 0.1,
            0.0, 0.1,

            1.2, 1.1,
            1.0, 1.1,

            2.2, 2.1,
            2.0, 2.1    });
    // clang-format on

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_gather_elements_float_1D) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/gather_elements_float_1D.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>(Shape{3}, {1, 2, 3});
    test_case.add_input<int64_t>(Shape{1}, {1});
    test_case.add_expected_output<float>(Shape{1}, {2});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_gather_elements_int8_axis_1) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/gather_elements_int8_axis_1.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<int8_t>(Shape{2, 2}, {1, 2, 3, 4});
    test_case.add_input<int32_t>(Shape{2, 2}, {0, 0, 1, 0});
    test_case.add_expected_output<int8_t>(Shape{2, 2}, {1, 1, 4, 3});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_gather_elements_int32_axis_0) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/gather_elements_int32_axis_0.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<int32_t>(Shape{3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    test_case.add_input<int64_t>(Shape{2, 3}, {1, 2, 0, 2, 0, 0});
    test_case.add_expected_output<int32_t>(Shape{2, 3}, {4, 8, 3, 7, 2, 3});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_gather_elements_float_negative_axis) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/gather_elements_float_negative_axis.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>(Shape{2, 2}, {1, 2, 3, 4});
    test_case.add_input<int64_t>(Shape{2, 2}, {1, 1, 1, 0});
    test_case.add_expected_output<float>(Shape{2, 2}, {2, 2, 4, 3});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_gather_elements_float_3D_axis_2) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/gather_elements_float_3D_axis_2.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>(Shape{2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    test_case.add_input<int64_t>(Shape{2, 2, 1}, {0, 1, 0, 1});
    test_case.add_expected_output<float>(Shape{2, 2, 1}, {1, 4, 5, 8});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_gatherND_int32) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/gatherND_int32.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<int32_t>({0, 1, 2, 3});
    test_case.add_input<int64_t>({1, 0});
    test_case.add_expected_output<int32_t>(Shape{2, 2}, {2, 3, 0, 1});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_gatherND_float) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/gatherND_float.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f});
    test_case.add_input<int64_t>({0, 1, 1, 0});
    test_case.add_expected_output<float>(Shape{2, 2}, {2.f, 3.f, 4.f, 5.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_pad_constant) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/pad_constant.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({1.f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f});
    test_case.add_expected_output<float>(Shape{3, 4},
                                         {0.f, 0.f, 1.f, 1.2f, 0.f, 0.f, 2.3f, 3.4f, 0.f, 0.f, 4.5f, 5.7f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_pow_float32_float32) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/pow_float32_float32.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});  // base
    test_case.add_input<float>({3.5f});                // exponent

    test_case.add_expected_output<float>(Shape{1, 4}, {1.f, 11.313708f, 46.765373f, 128.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_pow_float32_int32) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/pow_float32_int32.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});  // base
    test_case.add_input<int>({3});                     // exponent

    test_case.add_expected_output<float>(Shape{1, 4}, {1.f, 8.f, 27.f, 64.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_pow_int32_float32) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/pow_int32_float32.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<int>({1, 2, 3, 4});  // base
    test_case.add_input<float>({3.5f});      // exponent

    test_case.add_expected_output<int>(Shape{1, 4}, {1, 11, 46, 128});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reciprocal) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/reciprocal.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    test_case.add_expected_output<float>(Shape{3, 2}, {1.f, 1 / 2.f, 1 / 3.f, 1 / 4.f, 1 / 5.f, 1 / 6.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_round) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/round.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({0.1f, 0.9f, 1.2f, 1.5f, 1.8f, 2.3f, 2.7f, -1.1f, -1.9f, -2.2f, -2.8f});
    test_case.add_expected_output<float>({0.f, 1.f, 1.f, 2.f, 2.f, 2.f, 3.f, -1.f, -2.f, -2.f, -3.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_round_half_nearest_even) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/round_half_nearest_even.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({0.5f, 2.5f, -1.5f, -2.5f});
    test_case.add_expected_output<float>({0.f, 2.f, -2.f, -2.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_scatter10_import_only) {
    const auto scatter_fn =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/scatter_opset10.onnx"));

    const Shape data_shape{2, 2};

    EXPECT_EQ(scatter_fn->get_output_size(), 1);
    EXPECT_EQ(scatter_fn->get_output_shape(0), data_shape);
    EXPECT_EQ(count_ops_of_type<op::v3::ScatterElementsUpdate>(scatter_fn), 1);
    EXPECT_EQ(count_ops_of_type<op::v0::Constant>(scatter_fn), 4);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_scatter_elements_import_only) {
    const auto scatter_fn =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/scatter_elements_opset11.onnx"));

    const Shape data_shape{1, 5};

    EXPECT_EQ(scatter_fn->get_output_size(), 1);
    EXPECT_EQ(scatter_fn->get_output_shape(0), data_shape);
    EXPECT_EQ(count_ops_of_type<op::v3::ScatterElementsUpdate>(scatter_fn), 1);
    EXPECT_EQ(count_ops_of_type<op::v0::Constant>(scatter_fn), 4);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_upsample6_nearest_infer) {
    // clang-format off
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/upsample6_nearest.onnx"));
    // height_scale: 2.0
    // width_scale: 3.0
    // mode: nearest
    const Shape input_shape          {1, 1, 2, 2};
    const Shape expected_output_shape{1, 1, 4, 6};

    auto test_case = test::TestCase(function, s_device);
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

NGRAPH_TEST(${BACKEND_NAME}, onnx_upsample6_bilinear_infer) {
    // clang-format off
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/upsample6_bilinear.onnx"));
    // height_scale: 2.0
    // width_scale: 3.0
    // mode: bilinear
    const Shape input_shape          {1, 1, 2, 2};
    const Shape expected_output_shape{1, 1, 4, 6};

    auto test_case = test::TestCase(function, s_device);
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

NGRAPH_TEST(${BACKEND_NAME}, onnx_upsample6_dynamic) {
    // clang-format off
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/upsample6_dynamic.onnx"));
    // height_scale: 1.5
    // width_scale: 2.5
    // mode: nearest
    //
    //  X ───╤══> Reshape ──R──> Upsample ──> Y
    //  S ───┘

    auto test_case = test::TestCase(function, s_device);

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

NGRAPH_TEST(${BACKEND_NAME}, onnx_upsample8_nearest_infer) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/upsample8_nearest.onnx"));

    // Input data shape (1, 1, 2, 2)
    // Scales attribute values {1.0, 1.0, 2.0, 3.0}
    // mode: nearest

    const Shape expected_output_shape{1, 1, 4, 6};
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.0, 2.0, 3.0, 4.0});
    test_case.add_expected_output<float>(expected_output_shape,
                                         {1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
                                          3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_upsample8_linear_infer) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/upsample8_linear.onnx"));

    // Input data shape (1, 1, 2, 2)
    // Scales attribute values {1.0, 1.0, 2.0, 2.0}
    // mode: linear

    const Shape expected_output_shape{1, 1, 4, 4};
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.0, 2.0, 3.0, 4.0});
    test_case.add_expected_output<float>(
        expected_output_shape,
        {1.0, 1.5, 2.0, 2.0, 2.0, 2.5, 3.0, 3.0, 3.0, 3.5, 4.0, 4.0, 3.0, 3.5, 4.0, 4.0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_upsample9_scales_const_nearest_infer) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/upsample9_scales_const_nearest.onnx"));

    // Input data shape (1, 1, 2, 2)
    // Input const scales values {1.0, 1.0, 2.0, 3.0}
    // mode: nearest

    const Shape expected_output_shape{1, 1, 4, 6};
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.0, 2.0, 3.0, 4.0});
    test_case.add_expected_output<float>(expected_output_shape,
                                         {1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
                                          3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_upsample9_scales_const_linear_infer) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/upsample9_scales_const_linear.onnx"));

    // Input data shape (1, 1, 2, 2)
    // Input const scales values {1.0, 1.0, 2.0, 2.0}
    // mode: linear

    const Shape expected_output_shape{1, 1, 4, 4};
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.0, 2.0, 3.0, 4.0});
    test_case.add_expected_output<float>(
        expected_output_shape,
        {1.0, 1.5, 2.0, 2.0, 2.0, 2.5, 3.0, 3.0, 3.0, 3.5, 4.0, 4.0, 3.0, 3.5, 4.0, 4.0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_image_scaler) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/image_scaler.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0});
    test_case.add_expected_output<float>(Shape{1, 2, 2, 2}, {12.0, 14.0, 16.0, 18.0, 21.0, 41.0, 61.0, 81.0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_size_op_single) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/size_op_single.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    test_case.add_expected_output<int64_t>(Shape{}, {6});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_size_op_graph_end) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/size_op_graph_end.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.0, 2.0, 3.0, 4.0});
    test_case.add_expected_output<int64_t>(Shape{}, {4});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_size_op_graph_middle) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/size_op_graph_middle.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.0, 2.0, 3.0, 4.0});
    test_case.add_expected_output<float>(Shape{}, {4.0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_size_op_on_input_graph_middle) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/size_op_on_input_graph_middle.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{1, 2, 4, 1, 3}, {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.});
    test_case.add_expected_output<float>(Shape{1, 2, 4, 1, 3},
                                         {24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24.,
                                          24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24.});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_empty_initializers_handling) {
    // int this test the "scales" input of the Resize operator is set to an empty initializer
    // this input should be ignored since the "sizes" optional input is provided
    // and the inference should use the data from the latter
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/empty_initializers_handling.onnx"));

    const Shape expected_output_shape{2, 1, 4, 8};
    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input_data{2.0f, 4.0f, 1.0f, 3.0f, 7.0f, 8.0f, 9.0f, 6.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(
        expected_output_shape,
        {2.0f, 2.5f,  3.0f, 3.5f,  4.0f, 4.0f, 4.0f, 4.0f, 1.5f, 2.0f,  2.5f, 3.0f,  3.5f, 3.5f, 3.5f, 3.5f,
         1.0f, 1.5f,  2.0f, 2.5f,  3.0f, 3.0f, 3.0f, 3.0f, 1.0f, 1.5f,  2.0f, 2.5f,  3.0f, 3.0f, 3.0f, 3.0f,
         7.0f, 7.25f, 7.5f, 7.75f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 7.75f, 7.5f, 7.25f, 7.0f, 7.0f, 7.0f, 7.0f,
         9.0f, 8.25f, 7.5f, 6.75f, 6.0f, 6.0f, 6.0f, 6.0f, 9.0f, 8.25f, 7.5f, 6.75f, 6.0f, 6.0f, 6.0f, 6.0f});

    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_roi_align_f32) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/roi_align_f32.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12., 13., 14.,
                                15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29.,
                                30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40., 41., 42., 43., 44.,
                                45., 46., 47., 48., 49., 50., 51., 52., 53., 54., 55., 56., 57., 58., 59.,
                                60., 61., 62., 63., 64., 65., 66., 67., 68., 69., 70., 71., 72., 73., 74.});

    test_case.add_input<float>(
        {7., 5., 7., 5., -15., -15., -15., -15., -10., 21., -10., 21., 13., 8., 13., 8., -14., 19., -14., 19.});

    test_case.add_input<int32_t>({0, 0, 0, 0, 0});
    test_case.add_expected_output<float>(
        Shape{5, 3, 3, 4},
        {2.95833f, 3.20833f, 3.45833f, 3.70833f, 4.625f,   4.875f,   5.125f,   5.375f,   6.29167f, 6.54167f, 6.79167f,
         7.04167f, 27.9583f, 28.2083f, 28.4583f, 28.7083f, 29.625f,  29.875f,  30.125f,  30.375f,  31.2917f, 31.5417f,
         31.7917f, 32.0417f, 52.9583f, 53.2083f, 53.4583f, 53.7083f, 54.625f,  54.875f,  55.125f,  55.375f,  56.2917f,
         56.5417f, 56.7917f, 57.0417f, 0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,
         0.f,      0.f,      0.f,      0.f,      25.f,     25.f,     25.f,     25.f,     25.f,     25.f,     25.f,
         25.f,     25.f,     25.f,     25.f,     25.f,     50.f,     50.f,     50.f,     50.f,     50.f,     50.f,
         50.f,     50.f,     50.f,     50.f,     50.f,     50.f,     7.39583f, 7.39583f, 7.42708f, 7.64583f, 9.0625f,
         9.0625f,  9.09375f, 9.3125f,  10.7292f, 10.7292f, 10.7604f, 10.9792f, 32.3958f, 32.3958f, 32.4271f, 32.6458f,
         34.0625f, 34.0625f, 34.0938f, 34.3125f, 35.7292f, 35.7292f, 35.7604f, 35.9792f, 57.3958f, 57.3958f, 57.4271f,
         57.6458f, 59.0625f, 59.0625f, 59.0938f, 59.3125f, 60.7292f, 60.7292f, 60.7604f, 60.9792f, 4.27083f, 4.52083f,
         4.77083f, 5.02083f, 5.9375f,  6.1875f,  6.4375f,  6.6875f,  7.60417f, 7.85417f, 8.10417f, 8.35417f, 29.2708f,
         29.5208f, 29.7708f, 30.0208f, 30.9375f, 31.1875f, 31.4375f, 31.6875f, 32.6042f, 32.8542f, 33.1042f, 33.3542f,
         54.2708f, 54.5208f, 54.7708f, 55.0208f, 55.9375f, 56.1875f, 56.4375f, 56.6875f, 57.6042f, 57.8542f, 58.1042f,
         58.3542f, 6.77083f, 6.77083f, 6.77083f, 6.80208f, 8.4375f,  8.4375f,  8.4375f,  8.46875f, 10.1042f, 10.1042f,
         10.1042f, 10.1354f, 31.7708f, 31.7708f, 31.7708f, 31.8021f, 33.4375f, 33.4375f, 33.4375f, 33.4688f, 35.1042f,
         35.1042f, 35.1042f, 35.1354f, 56.7708f, 56.7708f, 56.7708f, 56.8021f, 58.4375f, 58.4375f, 58.4375f, 58.4688f,
         60.1042f, 60.1042f, 60.1042f, 60.1354f});
    test_case.run_with_tolerance_as_fp(1.0e-4f);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_roialign16_avg_out_half_pixel) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/roialign16_avg_out_half_pixel.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(
        {1.1,   2.2,   3.3,   4.4,   5.5,   6.6,   7.7,   8.8,   9.9,   11.,   12.1,  13.2,  14.3,  15.4,  16.5,  17.6,
         18.7,  19.8,  20.9,  22.,   23.1,  24.2,  25.3,  26.4,  27.5,  28.6,  29.7,  30.8,  31.9,  33.,   34.1,  35.2,
         36.3,  37.4,  38.5,  39.6,  40.7,  41.8,  42.9,  44.,   45.1,  46.2,  47.3,  48.4,  49.5,  50.6,  51.7,  52.8,
         53.9,  55.,   56.1,  57.2,  58.3,  59.4,  60.5,  61.6,  62.7,  63.8,  64.9,  66.,   67.1,  68.2,  69.3,  70.4,
         71.5,  72.6,  73.7,  74.8,  75.9,  77.,   78.1,  79.2,  80.3,  81.4,  82.5,  83.6,  84.7,  85.8,  86.9,  88.,
         89.1,  90.2,  91.3,  92.4,  93.5,  94.6,  95.7,  96.8,  97.9,  99.,   100.1, 101.2, 102.3, 103.4, 104.5, 105.6,
         106.7, 107.8, 108.9, 110.,  111.1, 112.2, 113.3, 114.4, 115.5, 116.6, 117.7, 118.8, 119.9, 121.,  122.1, 123.2,
         124.3, 125.4, 126.5, 127.6, 128.7, 129.8, 130.9, 132.,  133.1, 134.2, 135.3, 136.4, 137.5, 138.6, 139.7, 140.8,
         141.9, 143.,  144.1, 145.2, 146.3, 147.4, 148.5, 149.6, 150.7, 151.8, 152.9, 154.,  155.1, 156.2, 157.3, 158.4,
         159.5, 160.6, 161.7, 162.8, 163.9, 165.,  166.1, 167.2, 168.3, 169.4, 170.5, 171.6, 172.7, 173.8, 174.9, 176.,
         177.1, 178.2, 179.3, 180.4, 181.5, 182.6, 183.7, 184.8, 185.9, 187.,  188.1, 189.2, 190.3, 191.4, 192.5, 193.6,
         194.7, 195.8, 196.9, 198.,  199.1, 200.2, 201.3, 202.4, 203.5, 204.6, 205.7, 206.8, 207.9, 209.,  210.1, 211.2,
         212.3, 213.4, 214.5, 215.6, 216.7, 217.8, 218.9, 220.,  221.1, 222.2, 223.3, 224.4, 225.5, 226.6, 227.7, 228.8,
         229.9, 231.,  232.1, 233.2, 234.3, 235.4, 236.5, 237.6});

    test_case.add_input<float>({0, 0, 0.75, 2.2, 1.2, 0.5, 2.8, 1.9, 0, 3, 0, 3});

    test_case.add_input<int64_t>({0, 2, 1});
    test_case.add_expected_output<float>(
        Shape{3, 2, 4, 4},
        {2.145,     2.42,      2.6950002, 2.9700003, 3.96,      4.235,    4.51,      4.7850003, 5.775,     6.05,
         6.325,     6.6000004, 7.59,      7.8650007, 8.14,      8.415001, 41.745003, 42.019997, 42.295,    42.57,
         43.56,     43.835,    44.11,     44.385002, 45.375,    45.65,    45.925003, 46.200005, 47.190002, 47.465004,
         47.74,     48.015,    162.77249, 163.0475,  163.32251, 163.5975, 164.42252, 164.69751, 164.9725,  165.2475,
         166.07251, 166.3475,  166.6225,  166.8975,  167.72249, 167.9975, 168.27249, 168.5475,  202.3725,  202.6475,
         202.9225,  203.19751, 204.02252, 204.2975,  204.57251, 204.8475, 205.6725,  205.94751, 206.2225,  206.4975,
         207.32251, 207.5975,  207.8725,  208.1475,  91.162506, 91.4375,  91.7125,   91.9875,   92.8125,   93.0875,
         93.3625,   93.6375,   94.4625,   94.7375,   95.0125,   95.28749, 96.1125,   96.3875,   96.6625,   96.9375,
         130.76251, 131.0375,  131.3125,  131.5875,  132.4125,  132.6875, 132.9625,  133.2375,  134.0625,  134.33751,
         134.6125,  134.88751, 135.7125,  135.9875,  136.26251, 136.53749});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_roialign16_avg_half_pixel) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/roialign16_avg_half_pixel.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(
        {1.1,   2.2,   3.3,   4.4,   5.5,   6.6,   7.7,   8.8,   9.9,   11.,   12.1,  13.2,  14.3,  15.4,  16.5,  17.6,
         18.7,  19.8,  20.9,  22.,   23.1,  24.2,  25.3,  26.4,  27.5,  28.6,  29.7,  30.8,  31.9,  33.,   34.1,  35.2,
         36.3,  37.4,  38.5,  39.6,  40.7,  41.8,  42.9,  44.,   45.1,  46.2,  47.3,  48.4,  49.5,  50.6,  51.7,  52.8,
         53.9,  55.,   56.1,  57.2,  58.3,  59.4,  60.5,  61.6,  62.7,  63.8,  64.9,  66.,   67.1,  68.2,  69.3,  70.4,
         71.5,  72.6,  73.7,  74.8,  75.9,  77.,   78.1,  79.2,  80.3,  81.4,  82.5,  83.6,  84.7,  85.8,  86.9,  88.,
         89.1,  90.2,  91.3,  92.4,  93.5,  94.6,  95.7,  96.8,  97.9,  99.,   100.1, 101.2, 102.3, 103.4, 104.5, 105.6,
         106.7, 107.8, 108.9, 110.,  111.1, 112.2, 113.3, 114.4, 115.5, 116.6, 117.7, 118.8, 119.9, 121.,  122.1, 123.2,
         124.3, 125.4, 126.5, 127.6, 128.7, 129.8, 130.9, 132.,  133.1, 134.2, 135.3, 136.4, 137.5, 138.6, 139.7, 140.8,
         141.9, 143.,  144.1, 145.2, 146.3, 147.4, 148.5, 149.6, 150.7, 151.8, 152.9, 154.,  155.1, 156.2, 157.3, 158.4,
         159.5, 160.6, 161.7, 162.8, 163.9, 165.,  166.1, 167.2, 168.3, 169.4, 170.5, 171.6, 172.7, 173.8, 174.9, 176.,
         177.1, 178.2, 179.3, 180.4, 181.5, 182.6, 183.7, 184.8, 185.9, 187.,  188.1, 189.2, 190.3, 191.4, 192.5, 193.6,
         194.7, 195.8, 196.9, 198.,  199.1, 200.2, 201.3, 202.4, 203.5, 204.6, 205.7, 206.8, 207.9, 209.,  210.1, 211.2,
         212.3, 213.4, 214.5, 215.6, 216.7, 217.8, 218.9, 220.,  221.1, 222.2, 223.3, 224.4, 225.5, 226.6, 227.7, 228.8,
         229.9, 231.,  232.1, 233.2, 234.3, 235.4, 236.5, 237.6});

    test_case.add_input<float>({0, 0, 0.75, 2.2, 1.2, 0.5, 2.8, 1.9, 0, 3, 0, 3});

    test_case.add_input<int64_t>({0, 2, 1});
    test_case.add_expected_output<float>(
        Shape{3, 2, 4, 4},
        {1.1,       1.1,       1.1,       1.1,       1.1,       1.1,       1.1,       1.1,       2.3375,    2.3375,
         2.3375,    2.3375,    4.1525,    4.1525,    4.1525,    4.1525,    40.7,      40.7,      40.7,      40.7,
         40.7,      40.7,      40.7,      40.7,      41.9375,   41.9375,   41.9375,   41.9375,   43.7525,   43.7525,
         43.7525,   43.7525,   159.72,    159.94,    160.16,    160.38,    159.90562, 160.12563, 160.34563, 160.56563,
         160.9575,  161.1775,  161.3975,  161.61751, 162.1125,  162.3325,  162.55249, 162.77249, 199.32,    199.54001,
         199.76001, 199.97998, 199.50562, 199.72563, 199.94562, 200.16562, 200.5575,  200.7775,  200.9975,  201.2175,
         201.7125,  201.93251, 202.1525,  202.37251, 86.9,      86.9,      86.9,      86.9,      86.9,      86.9,
         86.9,      86.9,      86.9,      86.9,      86.9,      86.9,      86.9,      86.9,      86.9,      86.9,
         126.5,     126.5,     126.5,     126.5,     126.5,     126.5,     126.5,     126.5,     126.5,     126.5,
         126.5,     126.5,     126.5,     126.5,     126.5,     126.5});
    test_case.run_with_tolerance_as_fp(0.01f);
}

NGRAPH_TEST(${BACKEND_NAME}, quant_dequant_pattern) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/quant_dequant_pattern.onnx"));
    auto test_case = test::TestCase(function, s_device);
    // scale == 3.0
    // zero point == 10
    test_case.add_input<float>({9.0, 10.0, 15.0, 20.0, 30.0});
    test_case.add_input<float>({1});
    test_case.add_expected_output<float>(Shape{5}, {9.0, 9.0, 15.0, 21.0, 30.0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, quant_dequant_pattern_axis) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/quant_dequant_pattern_axis.onnx"));
    auto test_case = test::TestCase(function, s_device);
    // axis = 1
    // scale == {2.0, 3.0, 4.0}
    // zero point == {10, 20, 30}
    test_case.add_input<float>({1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 40.0, 50.0, 100.0});
    test_case.add_expected_output<float>(Shape{3, 3}, {0, 3, 4, 10, 21, 32, 40, 51, 100});
    test_case.add_input<float>({1});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_logsoftmax_0D) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/softmax_0D.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({3.141592});
    test_case.add_expected_output<float>({0.0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_logsoftmax_1D) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/logsoftmax_1D.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({-1.0f, 0.0f, 1.0f});
    test_case.add_expected_output<float>(Shape{3}, {-2.4076061, -1.407606, -0.407606});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_logsoftmax13_1D) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/logsoftmax13_1D.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({-1.0f, 0.0f, 1.0f});
    test_case.add_expected_output<float>(Shape{3}, {-2.4076061, -1.407606, -0.407606});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_logsoftmax13_2D) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/logsoftmax13_2D.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({0.0f, 1.0f, 2.0f, 3.0f, 10000, 10001, 10002, 10003});
    test_case.add_expected_output<float>(
        Shape{2, 4},
        {-3.4401896, -2.4401896, -1.4401896, -0.44018966, -3.4401896, -2.4401896, -1.4401896, -0.44018966});
    test_case.run_with_tolerance_as_fp();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_logsoftmax13_2D_reshape) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/logsoftmax13_2D.onnx"));
    InferenceEngine::CNNNetwork net(function);
    InferenceEngine::ICNNNetwork::InputShapes shapes = {};
    InferenceEngine::SizeVector shape = {1, 1, 4000};
    shapes[net.getInputsInfo().begin()->first] = shape;
    EXPECT_NO_THROW(net.reshape(shapes));
    ASSERT_EQ(shape, net.getOutputsInfo().begin()->second->getDims());
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_hard_sigmoid) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/hard_sigmoid.onnx"));

    const auto inf = std::numeric_limits<float>::infinity();
    const auto neg_inf = -std::numeric_limits<float>::infinity();

    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({inf, neg_inf, 0.0f, 1.0f});
    test_case.add_expected_output<float>(Shape{4}, {1.0f, 0.0f, 0.5f, 0.699999988079071f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_mul_v6) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/mul_v6.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({1.0f, 2.0f, 3.0f});
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(Shape{3}, {3.0f, 8.0f, 15.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_mul_v6_broadcast_axis_1) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/mul_v6_broadcast_axis_1.onnx"));
    auto test_case = test::TestCase(function, s_device);

    Shape shape{1, 3, 2, 2};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(
        shape,
        {3.0f, 6.0f, 9.0f, 12.0f, 20.0f, 24.0f, 28.0f, 32.0f, 45.0f, 50.0f, 55.0f, 60.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_mul_v6_broadcast_axes_1_2) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/mul_v6_broadcast_axes_1_2.onnx"));
    auto test_case = test::TestCase(function, s_device);

    Shape shape{1, 3, 2, 2};
    std::vector<float> A(shape_size(shape), -1.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    test_case.add_expected_output<float>(shape,
                                         {-3.f, -3.f, -4.f, -4.f, -5.f, -5.f, -6.f, -6.f, -7.f, -7.f, -8.f, -8.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_mul_v6_broadcast_no_axis) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/mul_v6_broadcast_no_axis.onnx"));
    auto test_case = test::TestCase(function, s_device);

    Shape shape{2, 2};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f});
    test_case.add_expected_output<float>(shape, {3.0f, 6.0f, 9.0f, 12.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_mul_v7) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/mul_v7.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({1.0f, 2.0f, 3.0f});
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(Shape{3}, {3.0f, 8.0f, 15.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_mul_v7_broadcast) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/mul_v7_broadcast.onnx"));
    auto test_case = test::TestCase(function, s_device);

    Shape shape{1, 2, 3};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(shape, {3.0f, 8.0f, 15.0f, 12.0f, 20.0f, 30.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_add_v6_broadcast_axis_1) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/add_v6_broadcast_axis_1.onnx"));
    auto test_case = test::TestCase(function, s_device);

    Shape shape{1, 3, 2, 2};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(
        shape,
        {4.0f, 5.0f, 6.0f, 7.0f, 9.0f, 10.0f, 11.0f, 12.0f, 14.0f, 15.0f, 16.0f, 17.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_add_v6_broadcast_axes_1_2) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/add_v6_broadcast_axes_1_2.onnx"));
    auto test_case = test::TestCase(function, s_device);

    Shape shape{1, 3, 2, 2};
    std::vector<float> A(shape_size(shape), 0.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    test_case.add_expected_output<float>(shape, {3.f, 3.f, 4.f, 4.f, 5.f, 5.f, 6.f, 6.f, 7.f, 7.f, 8.f, 8.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_add_v6_broadcast_no_axis) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/add_v6_broadcast_no_axis.onnx"));
    auto test_case = test::TestCase(function, s_device);

    Shape shape{2, 2};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f});
    test_case.add_expected_output<float>(shape, {4.0f, 5.0f, 6.0f, 7.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_add_v7) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/add_v7.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({1.0f, 2.0f, 3.0f});
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(Shape{3}, {4.0f, 6.0f, 8.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_sub_v6_broadcast_axis_1) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/sub_v6_broadcast_axis_1.onnx"));
    auto test_case = test::TestCase(function, s_device);

    Shape shape{1, 3, 2, 2};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(shape,
                                         {-2.0f, -1.0f, 0.0f, 1.0f, 1.0f, 2.0f, 3.0f, 4.0f, 4.0f, 5.0f, 6.0f, 7.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_sub_v6_broadcast_axes_1_2) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/sub_v6_broadcast_axes_1_2.onnx"));
    auto test_case = test::TestCase(function, s_device);

    Shape shape{1, 3, 2, 2};
    std::vector<float> A(shape_size(shape), 0.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    test_case.add_expected_output<float>(shape,
                                         {-3.f, -3.f, -4.f, -4.f, -5.f, -5.f, -6.f, -6.f, -7.f, -7.f, -8.f, -8.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_sub_v6_broadcast_no_axis) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/sub_v6_broadcast_no_axis.onnx"));
    auto test_case = test::TestCase(function, s_device);

    Shape shape{2, 2};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f});
    test_case.add_expected_output<float>(shape, {-2.0f, -1.0f, 0.0f, 1.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_sub_v7) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/sub_v7.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({1.0f, 2.0f, 3.0f});
    test_case.add_input<float>({3.0f, 8.0f, 7.0f});
    test_case.add_expected_output<float>(Shape{3}, {-2.0f, -6.0f, -4.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_sub_v7_broadcast) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/sub_v7_broadcast.onnx"));
    auto test_case = test::TestCase(function, s_device);

    Shape shape{1, 2, 3};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(shape, {-2.0f, -2.0f, -2.0f, 1.0f, 1.0f, 1.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_div_v6_broadcast_axis_1) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/div_v6_broadcast_axis_1.onnx"));
    auto test_case = test::TestCase(function, s_device);

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

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_div_v6_broadcast_axes_1_2) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/div_v6_broadcast_axes_1_2.onnx"));
    auto test_case = test::TestCase(function, s_device);

    Shape shape{1, 3, 2, 2};
    std::vector<float> A(shape_size(shape), 840.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    test_case.add_expected_output<float>(
        shape,
        {280.f, 280.f, 210.f, 210.f, 168.f, 168.f, 140.f, 140.f, 120.f, 120.f, 105.f, 105.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_div_v6_broadcast_no_axis) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/div_v6_broadcast_no_axis.onnx"));
    auto test_case = test::TestCase(function, s_device);

    Shape shape{2, 2};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1);
    test_case.add_input<float>(A);
    test_case.add_input<float>({2.0f});
    test_case.add_expected_output<float>(shape, {0.5f, 1.0f, 1.5f, 2.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_div_v7) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/div_v7.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({1.0f, 2.0f, 3.0f});
    test_case.add_input<float>({3.0f, 8.0f, 7.0f});
    test_case.add_expected_output<float>(Shape{3}, {0.3333333f, 0.25f, 0.4285714f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_div_v7_broadcast) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/div_v7_broadcast.onnx"));
    auto test_case = test::TestCase(function, s_device);

    Shape shape{1, 2, 3};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(shape, {0.3333333f, 0.5f, 0.6f, 1.3333333f, 1.25f, 1.2f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_dangling_parameter) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/dangling_parameter.onnx"));

    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({-1.0f, 2.0f, -3.0f});
    test_case.add_expected_output<float>(Shape{3}, {1.0f, 2.0f, 3.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_clip_inbounds) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/test_clip_inbounds.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const std::vector<int32_t> data{-1, 0, 1, -9999, 9999};
    test_case.add_input<int32_t>(data);
    test_case.add_expected_output<int32_t>(Shape{data.size()}, data);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_clip_no_min_no_max) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/clip_no_min_no_max.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const std::vector<float> data{-1.6, -0.1, 10., 0., -10., 1.99, 2.015, 3.};

    test_case.add_input<float>(data);

    test_case.add_expected_output<float>(Shape{2, 4}, data);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_clip_no_min_no_max_inf) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/clip_no_min_no_max.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const std::vector<float> data{std::numeric_limits<float>::infinity(),
                                  -std::numeric_limits<float>::infinity(),
                                  static_cast<float>(std::numeric_limits<double>::max()),
                                  std::numeric_limits<float>::min(),
                                  std::numeric_limits<float>::max(),
                                  std::numeric_limits<float>::lowest(),
                                  0,
                                  -1};

    const std::vector<float> expected_output{std::numeric_limits<float>::max(),
                                             std::numeric_limits<float>::lowest(),
                                             std::numeric_limits<float>::max(),
                                             std::numeric_limits<float>::min(),
                                             std::numeric_limits<float>::max(),
                                             std::numeric_limits<float>::lowest(),
                                             0,
                                             -1};

    test_case.add_input<float>(data);

    test_case.add_expected_output<float>(Shape{2, 4}, expected_output);
    test_case.run_with_tolerance_as_fp(0);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_clip_no_min_set_max) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/clip_no_min_set_max.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const std::vector<float> data{-1.6, -0.1, 10., 0., -10., 1.99, 2.015, 3.};
    const std::vector<float> max_val{2.01};
    const std::vector<float> output{-1.6, -0.1, 2.01, 0., -10., 1.99, 2.01, 2.01};

    test_case.add_input<float>(data);
    test_case.add_input<float>(max_val);

    test_case.add_expected_output<float>(Shape{2, 4}, output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_clip_set_min_no_max) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/clip_set_min_no_max.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const std::vector<float> data{-1.6, -0.1, 10., 0., -10., 1.99, 2.015, 3.};
    const std::vector<float> min_val{-1.59};
    const std::vector<float> output{-1.59, -0.1, 10., 0., -1.59, 1.99, 2.015, 3.};

    test_case.add_input<float>(data);
    test_case.add_input<float>(min_val);

    test_case.add_expected_output<float>(Shape{2, 4}, output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_clip_no_min_no_max_int64) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/clip_no_min_no_max_int64.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const std::vector<int64_t> data{INT64_MAX, INT64_MIN, INT32_MAX, INT32_MIN, 0, -1, 1, 0};

    test_case.add_input<int64_t>(data);

    test_case.add_expected_output<int64_t>(Shape{2, 4}, data);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_clip_no_min_set_max_int64) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/clip_no_min_set_max_int64.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const std::vector<int64_t> data{INT64_MAX, INT64_MIN, INT32_MAX, INT32_MIN, 0, -1, 1, 0};
    const std::vector<int64_t> max_val{INT32_MAX};
    const std::vector<int64_t> output{INT32_MAX, INT64_MIN, INT32_MAX, INT32_MIN, 0, -1, 1, 0};

    test_case.add_input<int64_t>(data);
    test_case.add_input<int64_t>(max_val);

    test_case.add_expected_output<int64_t>(Shape{2, 4}, output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_clip_set_min_no_max_initializers) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/clip_set_min_no_max_initializers.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const std::vector<float> data{-1.6, -0.1, 10., 0., -10., 1.99, 2.015, 3.};
    const std::vector<float> output{-1.59, -0.1, 10., 0., -1.59, 1.99, 2.015, 3.};

    test_case.add_input<float>(data);

    test_case.add_expected_output<float>(Shape{2, 4}, output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_clip_set_min_set_max) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/clip_set_min_set_max.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const std::vector<float> data{-1.6, -0.1, 10., 0., -10., 1.99, 2.015, 3.};
    const std::vector<float> min_val{-1.59};
    const std::vector<float> max_val{2.01};
    const std::vector<float> output{-1.59, -0.1, 2.01, 0., -1.59, 1.99, 2.01, 2.01};

    test_case.add_input<float>(data);
    test_case.add_input<float>(min_val);
    test_case.add_input<float>(max_val);

    test_case.add_expected_output<float>(Shape{2, 4}, output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_clip_set_min_set_max_initializers) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/clip_set_min_set_max_initializers.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const std::vector<float> data{-1.6, -0.1, 10., 0., -10., 1.99, 2.015, 3.};
    const std::vector<float> output{-1.59, -0.1, 2.01, 0., -1.59, 1.99, 2.01, 2.01};

    test_case.add_input<float>(data);

    test_case.add_expected_output<float>(Shape{2, 4}, output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_mvn_v6) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/mvn_v6.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({0.8439683,  0.5665144, 0.05836735, 0.02916367, 0.12964272, 0.5060197, 0.79538304,
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

NGRAPH_TEST(${BACKEND_NAME}, onnx_dropout1_no_training_no_return_mask) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dropout1_no_training_no_return_mask.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const std::vector<float> data(3 * 4 * 5, 2.0f);
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(Shape{3, 4, 5}, data);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dropout1_no_training_return_mask) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dropout1_no_training_return_mask.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const std::vector<float> data(3 * 4 * 5, 2.0f);
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(Shape{3, 4, 5}, data);
    test_case.add_expected_output<int32_t>(Shape{3, 4, 5},
                                           std::vector<int32_t>(3 * 4 * 5, 1));  // // bool converted to i32
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dropout7_no_return_mask) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/dropout7_no_return_mask.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const std::vector<float> data(3 * 4 * 5, 2.0f);
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(Shape{3, 4, 5}, data);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dropout12_no_training_no_return_mask) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dropout12_no_training_no_return_mask.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const std::vector<float> data(3 * 4 * 5, 2.0f);
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(Shape{3, 4, 5}, data);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dropout12_no_training_return_mask) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dropout12_no_training_return_mask.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const std::vector<float> data(3 * 4 * 5, 2.0f);
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(Shape{3, 4, 5}, data);
    test_case.add_expected_output<int32_t>(Shape{3, 4, 5},
                                           std::vector<int32_t>(3 * 4 * 5, 1));  // bool converted to i32
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dropout12_no_traning_no_const_rato) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dropout12_no_traning_no_const_rato.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1, 2, 3, 4});
    // test_case.add_input<float>(Shape{}, {0.5}); // ratio input is ignored

    test_case.add_expected_output<float>(Shape{1, 4}, {1., 2., 3., 4.});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dropout12_training_mode) {
    try {
        auto function =
            onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/dropout12_training_mode.onnx"));
        FAIL() << "Expected exception was not thrown";
    } catch (const ngraph::ngraph_error& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string("Training mode is not supported for Dropout op"));
    } catch (...) {
        FAIL() << "Expected ngraph_error exception was not thrown";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dropout12_not_const_training_mode) {
    try {
        auto function = onnx_import::import_onnx_model(
            file_util::path_join(SERIALIZED_ZOO, "onnx/dropout12_not_const_training_mode.onnx"));
        FAIL() << "Expected exception was not thrown";
    } catch (const ngraph::ngraph_error& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string("Non-constant training_mode input is not supported."));
    } catch (...) {
        FAIL() << "Expected ngraph_error exception was not thrown";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_multiple_slices_last_layer) {
    std::vector<float> data(1 * 30 * 320 * 320);
    std::fill(data.begin(), data.end(), 1);

    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/multiple_slices_last_layer.onnx"));
    auto test_case = test::TestCase(function, s_device);
    std::vector<float> o1(1 * 320 * 320 * 21);
    std::fill(o1.begin(), o1.end(), 1);

    std::vector<float> o2(1 * 320 * 320 * 9);
    std::fill(o2.begin(), o2.end(), 1);

    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(Shape{1, 320, 320, 21}, o1);
    test_case.add_expected_output<float>(Shape{1, 320, 320, 9}, o2);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_slice_const_axes_source) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/slice_const_axes_source.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    test_case.add_expected_output<float>(Shape{2, 2}, {2.f, 3.f, 6.f, 7.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_softmax_crossentropy_loss_mean) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/softmax_crossentropy_loss_mean.onnx"));

    auto test_case = test::TestCase(function, s_device);
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

NGRAPH_TEST(${BACKEND_NAME}, onnx_negativelog_likelihood_loss) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/negativelog_likelihood_loss.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({
        0.54881352186203,    0.7151893377304077,   0.6027633547782898, 0.5448831915855408, 0.42365479469299316,
        0.6458941102027893,  0.4375872015953064,   0.891772985458374,  0.9636627435684204, 0.3834415078163147,
        0.7917250394821167,  0.5288949012756348,   0.5680445432662964, 0.9255966544151306, 0.07103605568408966,
        0.08712930232286453, 0.020218396559357643, 0.832619845867157,  0.7781567573547363, 0.8700121641159058,
        0.978618323802948,   0.7991585731506348,   0.4614793658256531, 0.7805292010307312, 0.11827442795038223,
        0.6399210095405579,  0.14335328340530396,  0.9446688890457153, 0.5218483209609985, 0.4146619439125061,
    });
    test_case.add_input<int64_t>({3, 3, 2, 4, 2, 0});
    test_case.add_expected_output<float>(Shape{}, {-0.531306922435760498});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_constant_fill_input_as_shape_default_value) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/constant_fill_input_as_shape_default_value.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<float>(Shape{1, 2, 3}, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_constant_fill_input_as_shape_u8_type) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/constant_fill_input_as_shape_u8_type.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<uint8_t>(Shape{3, 1, 2}, {3, 3, 3, 3, 3, 3});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_constant_fill_extra_shape) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/constant_fill_extra_shape.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<float>(Shape{3, 1, 2, 2, 1}, std::vector<float>(12, 3.0f));
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_constant_fill_shape_attribute) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/constant_fill_shape_attribute.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<int32_t>(Shape{2, 3, 4}, std::vector<int32_t>(24, 5));
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_constant_float_tensor) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/constant_float_tensor.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<float>(Shape{2, 3}, {0.0f, 0.5f, 1.f, 1.5f, 2.f, 2.5f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_constant_bfloat_tensor) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/constant_bfloat_tensor.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<bfloat16>(Shape{2, 3}, {0.f, 5.f, 10.f, 15.f, 20.f, 25.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_constant_float_scalar) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/constant_float_scalar.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<float>(Shape{}, {0.5f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_constant_float_array) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/constant_float_array.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<float>(Shape{3}, {0.5f, 1.f, 1.5f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_constant_integer_scalar) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/constant_integer_scalar.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<std::int64_t>(Shape{}, {1});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_constant_integer_array) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/constant_integer_array.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<std::int64_t>(Shape{3}, {0, 1, 2});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_float_2x2) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/constant_sparse_tensor.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<float>(Shape{2, 2}, {0.f, 5.f, 0.f, 0.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_float_3x4) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/constant_sparse_tensor_float_3x4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<float>(Shape{3, 4}, {1.f, 0.f, 0.f, 8.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 3.f, 0.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_float_3x4_linearized_indices) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/constant_sparse_tensor_float_3x4_linearized_indices.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<float>(Shape{3, 4}, {1.f, 0.f, 0.f, 8.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 3.f, 0.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_int32_3x4) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/constant_sparse_tensor_int32_3x4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<int32_t>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_int64_3x4) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/constant_sparse_tensor_int64_3x4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<int64_t>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_boolean_3x4) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/constant_sparse_tensor_boolean_3x4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<bool>(Shape{3, 4}, {1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_float16_3x4) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/constant_sparse_tensor_float16_3x4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<ngraph::float16>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_double_3x4) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/constant_sparse_tensor_double_3x4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<double>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_int8_3x4) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/constant_sparse_tensor_int8_3x4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<int8_t>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_int16_3x4) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/constant_sparse_tensor_int16_3x4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<int16_t>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_uint8_3x4) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/constant_sparse_tensor_uint8_3x4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<uint8_t>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_uint16_3x4) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/constant_sparse_tensor_uint16_3x4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<uint16_t>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_uint32_3x4) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/constant_sparse_tensor_uint32_3x4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<uint32_t>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_uint64_3x4) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/constant_sparse_tensor_uint64_3x4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<uint64_t>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_bfloat16_3x4) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/constant_sparse_tensor_bfloat16_3x4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<ngraph::bfloat16>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_float_8x17) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/constant_sparse_tensor_float_8x17.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<float>(
        Shape{8, 17},
        {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
         0.f, 0.f, 0.f, 1.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 1.f,
         0.f, 0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0.f,
         0.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 1.f,
         0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0.f,
         0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
         0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_float_2x3x4) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/constant_sparse_tensor_float_2x3x4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<float>(Shape{2, 3, 4}, {1.f, 0.f, 0.f, 8.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 3.f, 0.f,
                                                          0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 3.f, 0.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_float_2x2x3x4) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/constant_sparse_tensor_float_2x2x3x4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<float>(
        Shape{2, 2, 3, 4},
        {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 2.f, 3.f, 0.f, 0.f, 0.f, 0.f,
         0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 8.f, 0.f, 1.f, 2.f, 0.f,
         0.f, 0.f, 3.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 2.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_einsum_sum) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/einsum_sum.onnx"));
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{3, 4},
                               {1.764052345967664,
                                0.4001572083672233,
                                0.9787379841057392,
                                2.240893199201458,
                                1.8675579901499675,
                                -0.977277879876411,
                                0.9500884175255894,
                                -0.1513572082976979,
                                -0.10321885179355784,
                                0.41059850193837233,
                                0.144043571160878,
                                1.454273506962975});
    test_case.add_expected_output<float>(Shape{3}, {5.3838407376420845, 1.689011319501448, 1.9056967282686674});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_float16_tensor_as_int32) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/conv_fp16_W_as_int32.onnx"));

    auto test_case = test::TestCase(function, s_device);
    // clang-format off
    test_case.add_input<ngraph::float16>(Shape{1, 1, 4, 4},
            {   0,  1,  2,  3,
                4,  5,  6,  7,
                8,  9,  10, 11,
                12, 13, 14, 15  });
    /* filters
            [[[[0.25, 0.5, 0.25],
               [0.5,  1.0, 0.5],
               [0.25, 0.5, 0.25]]]] */
    test_case.add_expected_output<ngraph::float16>(Shape{1, 1, 2, 2},
            {   20, 24,
                36, 40  });
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_max_pool_3d) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/max_pool_3d.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<int32_t>(Shape{1, 3, 3}, {-1, 0, 1, 20, -20, 10, 0, 2, 1});
    test_case.add_expected_output<int32_t>(Shape{1, 3, 2}, {0, 1, 20, 10, 2, 2});
    test_case.add_expected_output<int64_t>(Shape{1, 3, 2}, {1, 2, 3, 5, 7, 7});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_max_pool_4d_ceil_mode) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/max_pool_4d_ceil_mode.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<int32_t>(Shape{1, 1, 4, 4}, gen_range<int32_t>(16, 1));
    test_case.add_expected_output<int32_t>(Shape{1, 1, 2, 2}, {11, 12, 15, 16});
    test_case.add_expected_output<int64_t>(Shape{1, 1, 2, 2}, {10, 11, 14, 15});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_max_pool_4d_dilations) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/max_pool_4d_dilations.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<int32_t>(Shape{1, 1, 4, 4}, {9, 10, 11, 12, 1, 2, 3, 4, 16, 14, 15, 13, 5, 6, 8, 7});
    test_case.add_expected_output<int32_t>(Shape{1, 1, 2, 2}, {16, 14, 8, 7});
    test_case.add_expected_output<int64_t>(Shape{1, 1, 2, 2}, {8, 9, 14, 15});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_max_pool_4d_strides) {
    // kernel: 3x3
    // strides: 3, 3
    // explicit pads: 2, 2, 2, 2
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/max_pool_4d_strides.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<int8_t>(Shape{1, 1, 5, 5}, gen_range<int8_t>(25, 1));
    test_case.add_expected_output<int8_t>(Shape{1, 1, 3, 3}, {1, 4, 5, 16, 19, 20, 21, 24, 25});
    test_case.add_expected_output<int64_t>(Shape{1, 1, 3, 3}, {0, 3, 4, 15, 18, 19, 20, 23, 24});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_max_pool_4d_ceil_strides) {
    // kernel: 3x3
    // strides: 2, 2
    // ceil_mode: 1
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/max_pool_4d_ceil_strides.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(
        Shape{1, 1, 4, 4},
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
    test_case.add_expected_output<float>(Shape{1, 1, 2, 2}, {11.0f, 12.0f, 15.0f, 16.0f});
    test_case.add_expected_output<int64_t>(Shape{1, 1, 2, 2}, {10, 11, 14, 15});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_random_uniform) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/random_uniform.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<float>(Shape{2, 2}, {43.45518, 48.67585, 42.227386, 40.86294});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_random_uniform_like) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/random_uniform_like.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{2, 2}, {41, 42, 43, 44});
    test_case.add_expected_output<float>(Shape{2, 2}, {43.45518, 48.67585, 42.227386, 40.86294});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_random_normal) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/random_normal.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<float>(Shape{2, 2}, {13.459274, 41.75028, -19.311913, 131.79282});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_random_normal_like) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/random_normal_like.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{2, 2}, {0, 0, 0, 0});
    test_case.add_expected_output<float>(Shape{2, 2}, {13.459274, 41.75028, -19.311913, 131.79282});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_aten_embedding_bag_packed_sum_2in) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/aten_embedding_sum_packed_2in.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{5, 2}, {-0.2, -0.6, -0.1, -0.4, -1.9, -1.8, -1., 1.5, 0.8, -0.7});
    test_case.add_input<int32_t>(Shape{3, 2}, {0, 2, 1, 2, 3, 4});  // indices

    test_case.add_expected_output<float>(Shape{3, 2}, {-2.1, -2.4, -2., -2.2, -0.19999999, 0.8});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_aten_embedding_bag_packed_sum_3in_offsets_none) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/aten_embedding_sum_packed_3in_offset_none.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{5, 2}, {-0.2, -0.6, -0.1, -0.4, -1.9, -1.8, -1., 1.5, 0.8, -0.7});
    test_case.add_input<int32_t>(Shape{3, 2}, {0, 2, 1, 2, 3, 4});  // indices

    test_case.add_expected_output<float>(Shape{3, 2}, {-2.1, -2.4, -2., -2.2, -0.19999999, 0.8});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_aten_embedding_bag_packed_sum_4in_per_sample_weights) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/aten_embedding_sum_packed_4in_per_sample_weights.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{5, 2}, {-0.2, -0.6, -0.1, -0.4, -1.9, -1.8, -1., 1.5, 0.8, -0.7});
    test_case.add_input<int32_t>(Shape{3, 2}, {0, 2, 1, 2, 3, 4});            // indices
    test_case.add_input<float>(Shape{3, 2}, {0.5, 0.5, 0.5, 0.5, 0.5, 0.5});  // per_sample_weights

    test_case.add_expected_output<float>(Shape{3, 2}, {-1.05, -1.2, -1., -1.1, -0.09999999, 0.4});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_aten_embedding_bag_packed_sum_4in_two_none) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/aten_embedding_sum_packed_4in_two_none.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{5, 2}, {-0.2, -0.6, -0.1, -0.4, -1.9, -1.8, -1., 1.5, 0.8, -0.7});
    test_case.add_input<int32_t>(Shape{3, 2}, {0, 2, 1, 2, 3, 4});  // indices

    test_case.add_expected_output<float>(Shape{3, 2}, {-2.1, -2.4, -2., -2.2, -0.19999999, 0.8});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_aten_embedding_bag_offsets_sum_3in) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/aten_embedding_sum_offset_3in.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{5, 2}, {-0.2, -0.6, -0.1, -0.4, -1.9, -1.8, -1., 1.5, 0.8, -0.7});
    test_case.add_input<int32_t>(Shape{4}, {0, 2, 3, 4});  // indices
    test_case.add_input<int32_t>(Shape{3}, {0, 2, 2});     // offsets

    test_case.add_expected_output<float>(Shape{3, 2}, {-2.1, -2.4, 0, 0, -0.2, 0.8});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_aten_embedding_bag_offsets_sum_4in) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/aten_embedding_sum_offset_4in.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{5, 2}, {-0.2, -0.6, -0.1, -0.4, -1.9, -1.8, -1., 1.5, 0.8, -0.7});
    test_case.add_input<int32_t>(Shape{4}, {0, 2, 3, 4});        // indices
    test_case.add_input<int32_t>(Shape{3}, {0, 2, 2});           // offsets
    test_case.add_input<float>(Shape{4}, {0.5, 0.5, 0.5, 0.5});  // per_sample_weights

    test_case.add_expected_output<float>(Shape{3, 2}, {-1.05, -1.2, 0., 0., -0.09999999, 0.4});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_aten_embedding_bag_many_node_outputs) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/aten_embedding_sum_many_outputs.onnx"));

    // 4 outputs in onnx Node (1 connected and 3 not connected)
    EXPECT_EQ(function->outputs().size(), 1);
    EXPECT_EQ(function->get_results().size(), 1);

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{5, 2}, {-0.2, -0.6, -0.1, -0.4, -1.9, -1.8, -1., 1.5, 0.8, -0.7});
    test_case.add_input<int32_t>(Shape{4}, {0, 2, 3, 4});  // indices
    test_case.add_input<int32_t>(Shape{3}, {0, 2, 2});     // offsets

    test_case.add_expected_output<float>(Shape{3, 2}, {-2.1, -2.4, 0, 0, -0.2, 0.8});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_aten_unsupported_embedding_mode) {
    try {
        const auto function = onnx_import::import_onnx_model(
            file_util::path_join(SERIALIZED_ZOO, "onnx/aten_unsupported_embedding_mode.onnx"));
        FAIL() << "Expected exception was not thrown.";
    } catch (const ngraph::ngraph_error& e) {
        EXPECT_HAS_SUBSTRING(
            e.what(),
            std::string(
                "Unsupported mode, only `0` (sum) is supported as ATen embedding_bag `mode` attribute. Got: 1"));
    } catch (...) {
        FAIL() << "Other exception than expected was thrown.";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_aten_unsupported_operator) {
    try {
        const auto function =
            onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/aten_unsupported_operator.onnx"));
        FAIL() << "Expected exception was not thrown.";
    } catch (const ngraph::ngraph_error& e) {
        EXPECT_HAS_SUBSTRING(
            e.what(),
            std::string(
                "Only `embedding_bag` is supported as ATen `operator` attribute. Got: test_unsupported_operator"));
    } catch (...) {
        FAIL() << "Other exception than expected was thrown.";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_unsqueeze_ai_onnx_domain) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/unsqueeze_ai_onnx_domain.onnx"));

    auto input = test::NDArray<float, 3>({{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                          {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                          {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
                     .get_vector();

    auto expected_output =
        test::NDArray<float, 4>({{{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                  {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                  {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}}})
            .get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input(input);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_unsqueeze_default_domain) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/unsqueeze_default_domain.onnx"));

    auto input = test::NDArray<float, 3>({{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                          {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                          {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
                     .get_vector();

    auto expected_output =
        test::NDArray<float, 4>({{{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                  {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                  {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}}})
            .get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input(input);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_unsqueeze_default_domain_opset13) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/unsqueeze_default_domain_opset13.onnx"));

    auto input = test::NDArray<float, 3>({{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                          {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                          {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
                     .get_vector();
    auto expected_output =
        test::NDArray<float, 4>({{{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                  {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                  {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}}})
            .get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input(input);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_unsqueeze_ai_onnx_domain_opset13) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/unsqueeze_ai_onnx_domain_opset13.onnx"));

    auto input = test::NDArray<float, 3>({{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                          {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                          {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
                     .get_vector();
    auto expected_output =
        test::NDArray<float, 4>({{{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                  {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                  {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}}})
            .get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input(input);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_expand_failsafe_node) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/expand_failsafe_node.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const auto input_data = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
    test_case.add_input<float>(input_data);
    // the target shape is an empty constant so the Expand operation should not modify the input shape
    test_case.add_expected_output<float>(input_data);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_scan15_fib_like) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/scan15_fib_like.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{}, {0});
    test_case.add_input<float>(Shape{}, {1});
    test_case.add_input<float>(Shape{10}, std::vector<float>(10, 1));

    test_case.add_expected_output<float>(Shape{}, {55});
    test_case.add_expected_output<float>(Shape{}, {89});
    test_case.add_expected_output<float>(Shape{10}, {1., 2., 3., 5., 8., 13., 21., 34., 55., 89.});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_scan15_fib_like_out_rev) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/scan15_fib_like_out_rev.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{}, {0});
    test_case.add_input<float>(Shape{}, {1});
    test_case.add_input<float>(Shape{10}, std::vector<float>(10, 1));

    test_case.add_expected_output<float>(Shape{}, {55});
    test_case.add_expected_output<float>(Shape{}, {89});
    test_case.add_expected_output<float>(Shape{10}, {89., 55., 34., 21., 13., 8., 5., 3., 2., 1.});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_scan15_fib_like_input_rev) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/scan15_fib_like_input_rev.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{}, {0});
    test_case.add_input<float>(Shape{}, {1});
    test_case.add_input<float>(Shape{10}, std::vector<float>{0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9});

    test_case.add_expected_output<float>(Shape{}, {0.14897026});
    test_case.add_expected_output<float>(Shape{}, {0.});
    test_case.add_expected_output<float>(
        Shape{10},
        {0.9, 1.52, 1.694, 1.9284, 1.8112, 1.4958401, 0.9921121, 0.49759045, 0.14897026, 0.});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_scan15_fib_like_input_out_rev) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/scan15_fib_like_input_out_rev.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{}, {0});
    test_case.add_input<float>(Shape{}, {1});
    test_case.add_input<float>(Shape{10}, std::vector<float>{0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9});

    test_case.add_expected_output<float>(Shape{}, {0.14897026});
    test_case.add_expected_output<float>(Shape{}, {0.});
    test_case.add_expected_output<float>(
        Shape{10},
        {0., 0.14897026, 0.49759045, 0.9921121, 1.4958401, 1.8112, 1.9284, 1.694, 1.52, 0.9});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_scan15_ND_mixed_ones) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/scan15_ND_mixed.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{1, 3, 2}, {0, 0, 0, 0, 0, 0});
    test_case.add_input<float>(Shape{1, 3, 2}, {1, 1, 1, 1, 1, 1});
    test_case.add_input<float>(Shape{1, 3, 5, 2}, std::vector<float>(30, 1));  // multiply by one
    test_case.add_input<float>(Shape{1, 5, 3, 2}, std::vector<float>(30, 1));  // div by one

    test_case.add_expected_output<float>(Shape{1, 3, 2}, {5., 5., 5., 5., 5., 5.});
    test_case.add_expected_output<float>(Shape{1, 3, 2}, {8., 8., 8., 8., 8., 8.});
    test_case.add_expected_output<float>(Shape{1, 3, 2, 5},
                                         {8., 5., 3., 2., 1., 8., 5., 3., 2., 1., 8., 5., 3., 2., 1.,
                                          8., 5., 3., 2., 1., 8., 5., 3., 2., 1., 8., 5., 3., 2., 1.});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_scan15_ND_mixed_vals) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/scan15_ND_mixed.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{1, 3, 2}, {0, 0, 0, 0, 0, 0});
    test_case.add_input<float>(Shape{1, 3, 2}, {1, 1, 1, 1, 1, 1});
    std::vector<float> sequence_vals{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.,  1.1, 1.2, 1.3, 1.4, 1.5,
                                     1.6, 1.7, 1.8, 1.9, 2.,  2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.};
    test_case.add_input<float>(Shape{1, 3, 5, 2}, sequence_vals);  // multiply factor (reverse)
    test_case.add_input<float>(Shape{1, 5, 3, 2}, sequence_vals);  // div factor

    test_case.add_expected_output<float>(Shape{1, 3, 2},
                                         {2.7327938, 2.1428573, 21.070545, 16.92727, 49.765778, 41.444443});
    test_case.add_expected_output<float>(Shape{1, 3, 2},
                                         {0.40161943, 0.5274726, 16.80789, 14.025973, 59.98805, 50.518517});
    test_case.add_expected_output<float>(
        Shape{1, 3, 2, 5},
        {0.40161943, 2.7327938, 7.3076925, 10.,       9.,       0.5274726, 2.1428573, 4.714286,  6.,        5.,
         16.80789,   21.070545, 20.185184, 13.851851, 6.333333, 14.025973, 16.92727,  15.799998, 10.799999, 5.,
         59.98805,   49.765778, 33.074867, 16.690908, 5.8,      50.518517, 41.444443, 27.444445, 14.,       5.});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_scan15_ND_mixed_vals_neg_axes) {
    // Negative indices for scan_input_axes and scan_output_axes attributes
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/scan15_ND_mixed_neg_axes.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{1, 3, 2}, {0, 0, 0, 0, 0, 0});
    test_case.add_input<float>(Shape{1, 3, 2}, {1, 1, 1, 1, 1, 1});
    std::vector<float> sequence_vals{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.,  1.1, 1.2, 1.3, 1.4, 1.5,
                                     1.6, 1.7, 1.8, 1.9, 2.,  2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.};
    test_case.add_input<float>(Shape{1, 3, 5, 2}, sequence_vals);  // multiply factor (reverse)
    test_case.add_input<float>(Shape{1, 5, 3, 2}, sequence_vals);  // div factor

    test_case.add_expected_output<float>(Shape{1, 3, 2},
                                         {2.7327938, 2.1428573, 21.070545, 16.92727, 49.765778, 41.444443});
    test_case.add_expected_output<float>(Shape{1, 3, 2},
                                         {0.40161943, 0.5274726, 16.80789, 14.025973, 59.98805, 50.518517});
    test_case.add_expected_output<float>(
        Shape{1, 3, 2, 5},
        {0.40161943, 2.7327938, 7.3076925, 10.,       9.,       0.5274726, 2.1428573, 4.714286,  6.,        5.,
         16.80789,   21.070545, 20.185184, 13.851851, 6.333333, 14.025973, 16.92727,  15.799998, 10.799999, 5.,
         59.98805,   49.765778, 33.074867, 16.690908, 5.8,      50.518517, 41.444443, 27.444445, 14.,       5.});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_scan15_dyn_rank_vals) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/scan15_dyn_rank.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{1, 3, 2}, {0, 0, 0, 0, 0, 0});
    test_case.add_input<float>(Shape{1, 3, 2}, {1, 1, 1, 1, 1, 1});
    std::vector<float> sequence_vals{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.,  1.1, 1.2, 1.3, 1.4, 1.5,
                                     1.6, 1.7, 1.8, 1.9, 2.,  2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.};
    test_case.add_input<float>(Shape{1, 3, 5, 2}, sequence_vals);  // multiply factor (reverse)
    test_case.add_input<float>(Shape{1, 5, 3, 2}, sequence_vals);  // div factor

    test_case.add_expected_output<float>(Shape{1, 3, 2},
                                         {2.7327938, 2.1428573, 21.070545, 16.92727, 49.765778, 41.444443});
    test_case.add_expected_output<float>(Shape{1, 3, 2},
                                         {0.40161943, 0.5274726, 16.80789, 14.025973, 59.98805, 50.518517});
    test_case.add_expected_output<float>(
        Shape{1, 3, 2, 5},
        {0.40161943, 2.7327938, 7.3076925, 10.,       9.,       0.5274726, 2.1428573, 4.714286,  6.,        5.,
         16.80789,   21.070545, 20.185184, 13.851851, 6.333333, 14.025973, 16.92727,  15.799998, 10.799999, 5.,
         59.98805,   49.765778, 33.074867, 16.690908, 5.8,      50.518517, 41.444443, 27.444445, 14.,       5.});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_scan15_dyn_rank_vals_neg_axes) {
    // Negative indices for scan_input_axes and scan_output_axes attributes
    try {
        const auto function =
            onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/scan15_dyn_rank_neg_axes.onnx"));
    } catch (const ngraph::ngraph_error& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string("Rank must be static in order to normalize negative axis"));
    } catch (...) {
        FAIL() << "Expected exception was not thrown.";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_scan15_ND_b4_input_rev_vals) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/scan15_ND_b4_input_rev.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{4, 3, 2}, std::vector<float>(24, 0));
    test_case.add_input<float>(Shape{4, 3, 2}, std::vector<float>(24, 1));
    std::vector<float> sequence_vals{
        0.1,  0.2, 0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1.,   1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,
        1.9,  2.,  2.1,  2.2,  2.3,  2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  3.,   3.1,  3.2,  3.3,  3.4,  3.5,  3.6,
        3.7,  3.8, 3.9,  4.,   4.1,  4.2,  4.3,  4.4,  4.5,  4.6,  4.7,  4.8,  4.9,  5.,   5.1,  5.2,  5.3,  5.4,
        5.5,  5.6, 5.7,  5.8,  5.9,  6.,   6.1,  6.2,  6.3,  6.4,  6.5,  6.6,  6.7,  6.8,  6.9,  7.,   7.1,  7.2,
        7.3,  7.4, 7.5,  7.6,  7.7,  7.8,  7.9,  8.,   8.1,  8.2,  8.3,  8.4,  8.5,  8.6,  8.7,  8.8,  8.9,  9.,
        9.1,  9.2, 9.3,  9.4,  9.5,  9.6,  9.7,  9.8,  9.9,  10.,  10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8,
        10.9, 11., 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12.};
    test_case.add_input<float>(Shape{4, 5, 3, 2}, sequence_vals);  // multiply factor (reverse)
    test_case.add_input<float>(Shape{4, 5, 3, 2}, sequence_vals);  // div factor

    test_case.add_expected_output<float>(
        Shape{4, 3, 2},
        {61.210526, 33.2,      23.857145, 19.181818, 16.373913, 14.5,      6.8880844, 6.83,
         6.7754016, 6.7239814, 6.6754713, 6.6296296, 5.9686656, 5.953226,  5.9382715, 5.9237804,
         5.9097314, 5.896105,  5.652082,  5.645059,  5.638186,  5.6314588, 5.624872,  5.618421});
    test_case.add_expected_output<float>(
        Shape{4, 3, 2},
        {6.271278, 6.2461543, 6.2433867, 6.2545457, 6.2744985, 6.3,       6.9531364, 6.970527,
         6.987378, 7.003712,  7.019554,  7.034921,  7.30868,   7.3164845, 7.324116,  7.3315806,
         7.338885, 7.346032,  7.485426,  7.489783,  7.494067,  7.49828,   7.5024257, 7.506502});
    test_case.add_expected_output<float>(
        Shape{5, 4, 3, 2},
        {25.,       13.,       9.,        7.,        5.8,       5.,        1.7741936, 1.75,      1.7272727, 1.7058823,
         1.6857144, 1.6666667, 1.3934426, 1.3870969, 1.3809522, 1.375,     1.3692307, 1.3636364, 1.2637362, 1.2608696,
         1.2580644, 1.2553192, 1.2526315, 1.25,      70.57143,  35.,       23.333334, 17.6,      14.218181, 12.,
         3.6739323, 3.618421,  3.5664334, 3.5176468, 3.471777,  3.4285717, 2.822119,  2.8083491, 2.7950313, 2.7821426,
         2.7696643, 2.757576,  2.543786,  2.5377107, 2.5317693, 2.5259573, 2.520271,  2.514706,  95.57143,  47.999996,
         32.333336, 24.6,      20.01818,  17.,       5.448126,  5.368421,  5.293706,  5.223529,  5.157491,  5.0952387,
         4.215562,  4.195446,  4.1759834, 4.1571426, 4.138895,  4.1212125, 3.8075223, 3.7985802, 3.7898335, 3.7812767,
         3.7729027, 3.764706,  61.210526, 33.2,      23.857145, 19.181818, 16.373913, 14.5,      6.8880844, 6.83,
         6.7754016, 6.7239814, 6.6754713, 6.6296296, 5.9686656, 5.953226,  5.9382715, 5.9237804, 5.9097314, 5.896105,
         5.652082,  5.645059,  5.638186,  5.6314588, 5.624872,  5.618421,  6.271278,  6.2461543, 6.2433867, 6.2545457,
         6.2744985, 6.3,       6.9531364, 6.970527,  6.987378,  7.003712,  7.019554,  7.034921,  7.30868,   7.3164845,
         7.324116,  7.3315806, 7.338885,  7.346032,  7.485426,  7.489783,  7.494067,  7.49828,   7.5024257, 7.506502});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_scan8_ND_b4_ones) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/scan8_ND_b4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{4, 3, 2}, std::vector<float>(24, 0));
    test_case.add_input<float>(Shape{4, 3, 2}, std::vector<float>(24, 1));
    std::vector<float> sequence_vals(120, 1);
    test_case.add_input<float>(Shape{4, 5, 3, 2}, sequence_vals);  // multiply by one
    test_case.add_input<float>(Shape{4, 5, 3, 2}, sequence_vals);  // div by one

    test_case.add_expected_output<float>(Shape{4, 3, 2}, {5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                                          5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.});
    test_case.add_expected_output<float>(Shape{4, 3, 2}, {8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.,
                                                          8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.});
    test_case.add_expected_output<float>(
        Shape{4, 5, 3, 2},
        {1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 5., 5., 5., 5., 5., 5.,
         8., 8., 8., 8., 8., 8., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3.,
         5., 5., 5., 5., 5., 5., 8., 8., 8., 8., 8., 8., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2.,
         3., 3., 3., 3., 3., 3., 5., 5., 5., 5., 5., 5., 8., 8., 8., 8., 8., 8., 1., 1., 1., 1., 1., 1.,
         2., 2., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 5., 5., 5., 5., 5., 5., 8., 8., 8., 8., 8., 8.});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_scan8_ND_b4_input_rev_vals) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/scan8_ND_b4_input_rev.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{4, 3, 2}, std::vector<float>(24, 0));
    test_case.add_input<float>(Shape{4, 3, 2}, std::vector<float>(24, 1));
    std::vector<float> sequence_vals{
        0.1,  0.2, 0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1.,   1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,
        1.9,  2.,  2.1,  2.2,  2.3,  2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  3.,   3.1,  3.2,  3.3,  3.4,  3.5,  3.6,
        3.7,  3.8, 3.9,  4.,   4.1,  4.2,  4.3,  4.4,  4.5,  4.6,  4.7,  4.8,  4.9,  5.,   5.1,  5.2,  5.3,  5.4,
        5.5,  5.6, 5.7,  5.8,  5.9,  6.,   6.1,  6.2,  6.3,  6.4,  6.5,  6.6,  6.7,  6.8,  6.9,  7.,   7.1,  7.2,
        7.3,  7.4, 7.5,  7.6,  7.7,  7.8,  7.9,  8.,   8.1,  8.2,  8.3,  8.4,  8.5,  8.6,  8.7,  8.8,  8.9,  9.,
        9.1,  9.2, 9.3,  9.4,  9.5,  9.6,  9.7,  9.8,  9.9,  10.,  10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8,
        10.9, 11., 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12.};
    test_case.add_input<float>(Shape{4, 5, 3, 2}, sequence_vals);  // multiply factor (reverse)
    test_case.add_input<float>(Shape{4, 5, 3, 2}, sequence_vals);  // div factor

    test_case.add_expected_output<float>(
        Shape{4, 3, 2},
        {61.210526, 33.2,      23.857145, 19.181818, 16.373913, 14.5,      6.8880844, 6.83,
         6.7754016, 6.7239814, 6.6754713, 6.6296296, 5.9686656, 5.953226,  5.9382715, 5.9237804,
         5.9097314, 5.896105,  5.652082,  5.645059,  5.638186,  5.6314588, 5.624872,  5.618421});
    test_case.add_expected_output<float>(
        Shape{4, 3, 2},
        {6.271278, 6.2461543, 6.2433867, 6.2545457, 6.2744985, 6.3,       6.9531364, 6.970527,
         6.987378, 7.003712,  7.019554,  7.034921,  7.30868,   7.3164845, 7.324116,  7.3315806,
         7.338885, 7.346032,  7.485426,  7.489783,  7.494067,  7.49828,   7.5024257, 7.506502});
    test_case.add_expected_output<float>(
        Shape{4, 5, 3, 2},
        {25.,       13.,       9.,        7.,        5.8,       5.,        70.57143,  35.,       23.333334, 17.6,
         14.218181, 12.,       95.57143,  47.999996, 32.333336, 24.6,      20.01818,  17.,       61.210526, 33.2,
         23.857145, 19.181818, 16.373913, 14.5,      6.271278,  6.2461543, 6.2433867, 6.2545457, 6.2744985, 6.3,
         1.7741936, 1.75,      1.7272727, 1.7058823, 1.6857144, 1.6666667, 3.6739323, 3.618421,  3.5664334, 3.5176468,
         3.471777,  3.4285717, 5.448126,  5.368421,  5.293706,  5.223529,  5.157491,  5.0952387, 6.8880844, 6.83,
         6.7754016, 6.7239814, 6.6754713, 6.6296296, 6.9531364, 6.970527,  6.987378,  7.003712,  7.019554,  7.034921,
         1.3934426, 1.3870969, 1.3809522, 1.375,     1.3692307, 1.3636364, 2.822119,  2.8083491, 2.7950313, 2.7821426,
         2.7696643, 2.757576,  4.215562,  4.195446,  4.1759834, 4.1571426, 4.138895,  4.1212125, 5.9686656, 5.953226,
         5.9382715, 5.9237804, 5.9097314, 5.896105,  7.30868,   7.3164845, 7.324116,  7.3315806, 7.338885,  7.346032,
         1.2637362, 1.2608696, 1.2580644, 1.2553192, 1.2526315, 1.25,      2.543786,  2.5377107, 2.5317693, 2.5259573,
         2.520271,  2.514706,  3.8075223, 3.7985802, 3.7898335, 3.7812767, 3.7729027, 3.764706,  5.652082,  5.645059,
         5.638186,  5.6314588, 5.624872,  5.618421,  7.485426,  7.489783,  7.494067,  7.49828,   7.5024257, 7.506502});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_scan8_ND_b4_seq_lens) {
    // ONNX Scan-8 can has optional `sequence_lens` input, the input was removed since ONNX Scan-9
    try {
        const auto function =
            onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/scan8_ND_b4_seq_lens.onnx"));
    } catch (const ngraph::ngraph_error& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string(" ONNX Scan-8 `sequence_lens` input is not supported. "));
    } catch (...) {
        FAIL() << "Expected exception was not thrown.";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_softsign) {
    auto model = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/softsign.onnx"));

    Inputs inputs{std::vector<float>{1.0, 0.1, 20.0, 12.0, -12.0, -0.2, 0.5, 100.0, 0.0, -1.0}};

    std::vector<float>
        output{0.5, 0.09090909, 0.95238096, 0.9230769, -0.9230769, -0.16666666, 0.33333334, 0.990099, 0., -0.5};

    auto test_case = test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_grid_sample) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/grid_sample.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{1, 1, 4, 4}, gen_range<float>(16));
    test_case.add_input<float>(
        Shape{1, 6, 6, 2},
        {-1.0000f, -1.0000f, -0.6000f, -1.0000f, -0.2000f, -1.0000f, 0.2000f,  -1.0000f, 0.6000f,  -1.0000f, 1.0000f,
         -1.0000f, -1.0000f, -0.6000f, -0.6000f, -0.6000f, -0.2000f, -0.6000f, 0.2000f,  -0.6000f, 0.6000f,  -0.6000f,
         1.0000f,  -0.6000f, -1.0000f, -0.2000f, -0.6000f, -0.2000f, -0.2000f, -0.2000f, 0.2000f,  -0.2000f, 0.6000f,
         -0.2000f, 1.0000f,  -0.2000f, -1.0000f, 0.2000f,  -0.6000f, 0.2000f,  -0.2000f, 0.2000f,  0.2000f,  0.2000f,
         0.6000f,  0.2000f,  1.0000f,  0.2000f,  -1.0000f, 0.6000f,  -0.6000f, 0.6000f,  -0.2000f, 0.6000f,  0.2000f,
         0.6000f,  0.6000f,  0.6000f,  1.0000f,  0.6000f,  -1.0000f, 1.0000f,  -0.6000f, 1.0000f,  -0.2000f, 1.0000f,
         0.2000f,  1.0000f,  0.6000f,  1.0000f,  1.0000f,  1.0000});

    test_case.add_expected_output<float>(
        Shape{1, 1, 6, 6},
        {0.0000f,  0.1500f,  0.5500f, 0.9500f, 1.3500f,  0.7500f, 0.6000f, 1.5000f,  2.3000f,
         3.1000f,  3.9000f,  2.1000f, 2.2000f, 4.7000f,  5.5000f, 6.3000f, 7.1000f,  3.7000f,
         3.8000f,  7.9000f,  8.7000f, 9.5000f, 10.3000f, 5.3000f, 5.4000f, 11.1000f, 11.9000f,
         12.7000f, 13.5000f, 6.9000f, 3.0000f, 6.1500f,  6.5500f, 6.9500f, 7.3500f,  3.7500});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_concat_empty_init) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/concat_empty_init.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<int64_t>(Shape{2}, std::vector<int64_t>{1, 2});
    test_case.add_expected_output<int64_t>(Shape{2}, std::vector<int64_t>{1, 2});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_layer_norm) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/layer_norm.onnx"));

    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input = {
        31.,
        245.,
        47.,
        239.,
        -106.,
        167.,
        33.,
        157.,
        59.,
        -193.,
        -103.,
        -246.,
    };

    std::vector<float> bias = {
        43.,
        -83.,
        -92.,
        12.,
    };

    std::vector<float> scale = {
        19.,
        68.,
        57.,
        59.,
    };

    std::vector<float> output = {
        22.538681,
        -13.113842,
        -144.41461,
        69.15499,
        14.064551,
        -19.023893,
        -107.303635,
        62.184105,
        72.52179,
        -125.468506,
        -83.254326,
        -51.877796,
    };

    test_case.add_input<float>(Shape{3, 4}, input);
    test_case.add_input<float>(scale);
    test_case.add_input<float>(bias);
    test_case.add_expected_output<float>(Shape{3, 4}, output);

    test_case.run_with_tolerance_as_fp(1e-5);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_layer_norm_dynamic_4d) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/layer_norm_dynamic_4d.onnx"));

    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input = {
        159., 1.,   214.,  -12.,  -56.,  -165., -38.,  251.,  -226., -201., 113.,  101., -217., 167.,  -199., 230.,
        -13., 94.,  121.,  78.,   139.,  -56.,  -139., -204., -188., 56.,   -165., 59.,  113.,  229.,  -72.,  75.,
        -75., 202., -195., -102., -234., 237.,  210.,  -49.,  182.,  195.,  150.,  140., 108.,  -245., 63.,   -249.,
    };
    std::vector<float> scale = {
        -74.,
        -49.,
        46.,
        56.,
    };
    std::vector<float> bias = {
        119.,
        41.,
        -23.,
        71.,
    };
    std::vector<float> output = {
        6.7305779e+01,  8.5723816e+01,  3.4935467e+01,  1.2462846e+01, 1.4495818e+02,  9.2883873e+01,  -3.3757442e+01,
        1.6303590e+02,  1.9862335e+02,  8.6093529e+01,  2.4633244e+01, 1.2480267e+02,  1.9577968e+02,  -1.3956308e-01,
        -6.6680313e+01, 1.3526292e+02,  2.4104924e+02,  1.7631407e+01, 2.3617962e+01,  7.9902321e+01,  1.9151673e+00,
        3.7579597e+01,  -4.9401482e+01, 1.0627163e+01,  2.0007460e+02, -7.2534118e+00, -6.4377121e+01, 1.2757914e+02,
        1.0059356e+02,  -2.4040894e+01, -9.0688644e+01, 6.5141930e+01, 1.3524843e+02,  -3.9941471e+01, -7.0394051e+01,
        4.8488670e+01,  2.2384190e+02,  -8.4791994e+00, 1.7051153e+01, 4.5034241e+01,  6.8866104e+01,  -2.0495653e+01,
        -5.7229656e+01, 4.4509735e+00,  3.5370048e+01,  8.9188629e+01, 1.6592127e+01,  1.4586085e+01,
    };

    test_case.add_input<float>(Shape{2, 2, 3, 4}, input);
    test_case.add_input<float>(scale);
    test_case.add_input<float>(bias);
    test_case.add_expected_output<float>(Shape{2, 2, 3, 4}, output);

    test_case.run_with_tolerance_as_fp(1e-5);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_layer_norm_dynamic_4d_axis_minus1) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/layer_norm_dynamic_4d_axis_-1.onnx"));

    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input = {
        159., 1.,   214.,  -12.,  -56.,  -165., -38.,  251.,  -226., -201., 113.,  101., -217., 167.,  -199., 230.,
        -13., 94.,  121.,  78.,   139.,  -56.,  -139., -204., -188., 56.,   -165., 59.,  113.,  229.,  -72.,  75.,
        -75., 202., -195., -102., -234., 237.,  210.,  -49.,  182.,  195.,  150.,  140., 108.,  -245., 63.,   -249.,
    };
    std::vector<float> scale = {
        -74.,
        -49.,
        46.,
        56.,
    };
    std::vector<float> bias = {
        119.,
        41.,
        -23.,
        71.,
    };
    std::vector<float> output = {
        6.7305779e+01,  8.5723816e+01,  3.4935467e+01,  1.2462846e+01, 1.4495818e+02,  9.2883873e+01,  -3.3757442e+01,
        1.6303590e+02,  1.9862335e+02,  8.6093529e+01,  2.4633244e+01, 1.2480267e+02,  1.9577968e+02,  -1.3956308e-01,
        -6.6680313e+01, 1.3526292e+02,  2.4104924e+02,  1.7631407e+01, 2.3617962e+01,  7.9902321e+01,  1.9151673e+00,
        3.7579597e+01,  -4.9401482e+01, 1.0627163e+01,  2.0007460e+02, -7.2534118e+00, -6.4377121e+01, 1.2757914e+02,
        1.0059356e+02,  -2.4040894e+01, -9.0688644e+01, 6.5141930e+01, 1.3524843e+02,  -3.9941471e+01, -7.0394051e+01,
        4.8488670e+01,  2.2384190e+02,  -8.4791994e+00, 1.7051153e+01, 4.5034241e+01,  6.8866104e+01,  -2.0495653e+01,
        -5.7229656e+01, 4.4509735e+00,  3.5370048e+01,  8.9188629e+01, 1.6592127e+01,  1.4586085e+01,
    };

    test_case.add_input<float>(Shape{2, 2, 3, 4}, input);
    test_case.add_input<float>(scale);
    test_case.add_input<float>(bias);
    test_case.add_expected_output<float>(Shape{2, 2, 3, 4}, output);

    test_case.run_with_tolerance_as_fp(1e-5);
}
