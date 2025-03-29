// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
#include "common_test_utils/all_close.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ndarray.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_case.hpp"
#include "common_test_utils/test_control.hpp"
#include "common_test_utils/test_tools.hpp"
#include "common_test_utils/type_prop.hpp"
#include "conversion_extension.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/scatter_elements_update.hpp"

using namespace ov;
using namespace ov::frontend::onnx::tests;

static std::string s_manifest = onnx_backend_manifest(MANIFEST);
static std::string s_device = backend_name_to_device("${BACKEND_NAME}");

using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;

// ############################################################################ CORE TESTS
OPENVINO_TEST(${BACKEND_NAME}, onnx_test_test_case) {
    auto model = convert_model("add_abc.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({1});
    test_case.add_input<float>({2});
    test_case.add_input<float>({3});
    test_case.add_expected_output<float>(Shape{1}, {6});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_test_test_case_mutliple_inputs) {
    auto model = convert_model("add_abc.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(Inputs{{1}, {2}, {3}});
    test_case.add_expected_output(Shape{1}, std::vector<float>{6});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_output_names_check) {
    auto model = convert_model("split_equal_parts_default.onnx");

    std::size_t size = model->get_output_size();
    for (std::size_t i{0}; i < size; ++i) {
        std::shared_ptr<Node> node = model->get_output_op(i);
        EXPECT_EQ(node->get_friendly_name(), "output_" + std::to_string(i + 1) + "/sink_port_0");
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_node_names_check) {
    auto model = convert_model("add_abc.onnx");

    // Filter out Add nodes from the model graph
    std::vector<std::shared_ptr<Node>> additions;
    auto ordered_ops = model->get_ordered_ops();
    std::copy_if(ordered_ops.begin(), ordered_ops.end(), std::back_inserter(additions), [](std::shared_ptr<Node> op) {
        return std::string(op->get_type_name()) == "Add";
    });

    EXPECT_EQ(additions.size(), 2);
    EXPECT_EQ(additions.at(0)->get_friendly_name(), "add_node1");
    EXPECT_EQ(additions.at(0)->get_output_tensor(0).get_names(), std::unordered_set<std::string>{"X"});
    EXPECT_EQ(additions.at(1)->get_friendly_name(), "Y");
    EXPECT_EQ(additions.at(1)->get_output_tensor(0).get_names(), std::unordered_set<std::string>{"Y"});
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_duplicated_output_name) {
    auto model = convert_model("duplicated_output_name.onnx");
    EXPECT_EQ(model->get_output_size(), 2);

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(Inputs{{1}, {2}, {3}});
    test_case.add_expected_output(Shape{1}, std::vector<float>{6});
    test_case.add_expected_output(Shape{1}, std::vector<float>{6});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_duplicated_more_output_names) {
    auto model = convert_model("duplicated_more_output_names.onnx");
    EXPECT_EQ(model->get_output_size(), 4);

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(Inputs{{1, 2}, {2}, {3}});
    test_case.add_expected_output(Shape{1}, std::vector<float>{6});
    test_case.add_expected_output(Shape{1}, std::vector<float>{6});
    test_case.add_expected_output(Shape{1}, std::vector<float>{7});
    test_case.add_expected_output(Shape{1}, std::vector<float>{6});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_binary_add_abc) {
    auto model = convert_model("add_abc.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(Inputs{{1}, {2}, {3}});
    test_case.add_expected_output(Shape{1}, std::vector<float>{6});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_bool_const_op) {
    auto model = convert_model("bool_const_op.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output(std::vector<bool>{1, 0, 0, 1});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_bool_init_and) {
    auto model = convert_model("bool_init_and.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output(std::vector<bool>{1});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_bool_input_or) {
    auto model = convert_model("bool_input_or.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input(std::vector<bool>{true, false, true, false});
    test_case.add_input(std::vector<bool>{false, false, true, true});
    test_case.add_expected_output(std::vector<bool>{1, 0, 1, 1});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_bool_init_raw) {
    auto model = convert_model("bool_init_raw.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output(std::vector<bool>{true, false, true});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_int4_const) {
    auto model = convert_model("int4_const.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output(std::vector<int64_t>{4});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_int4_input) {
    const auto model = convert_model("int4_input.onnx");
    auto test_case = test::TestCase(model);
    test_case.add_input<uint8_t>({0xEF, 0x01, 0x70});
    test_case.add_expected_output<int64_t>({5});
    test_case.add_expected_output<uint8_t>({0xEF, 0x01, 0x70});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_uint4_const) {
    auto model = convert_model("uint4_const.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output(std::vector<int64_t>{4});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_uint4_input) {
    const auto model = convert_model("uint4_input.onnx");
    auto test_case = test::TestCase(model);
    test_case.add_input<uint8_t>({0x01, 0xF0});
    test_case.add_expected_output<int64_t>({3});
    test_case.add_expected_output<uint8_t>({0x01, 0xF0});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_add_abc_initializers) {
    auto model = convert_model("add_abc_initializers.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({1, 2, 3, 4});
    test_case.add_expected_output<float>({3, 6, 9, 12});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_import_non_existing_file) {
    try {
        convert_model("i.dont.exist");
    } catch (const std::runtime_error& exc) {
        // asserts that an exception was thrown and that the error message contains the file name
        std::string msg{exc.what()};
        EXPECT_TRUE(msg.find("i.dont.exist") != std::string::npos);
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_unsupported_op) {
    try {
        convert_model("unsupported_op.onnx");
        FAIL() << "Expected ov::Exception";
    } catch (ov::Exception const& err) {
        std::string what{err.what()};
        EXPECT_NE(what.find("OpenVINO does not support"), std::string::npos);
        EXPECT_NE(what.find("FakeOpName"), std::string::npos);
        EXPECT_NE(what.find("AnotherFakeOpName"), std::string::npos);
    } catch (...) {
        FAIL() << "Expected ov::Exception";
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_unknown_domain) {
    // the importer should not throw when it encounters an unknown domain in the model
    EXPECT_NO_THROW(convert_model("unknown_domain.onnx"));
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_op_in_unknown_domain) {
    try {
        convert_model("unknown_domain_add.onnx");

        FAIL() << "The onnx_importer did not throw for unknown domain and op";
    } catch (const ov::Exception& e) {
        const std::string msg = e.what();

        EXPECT_NE(msg.find("unknown.domain.Add"), std::string::npos)
            << "The error message should contain domain and op name: unknown.domain.Add";
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_initializer_wo_input) {
    // This test checks a model which has an initializer, but no input with the same name
    auto model = convert_model("initializer_wo_input.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({0, 1, 2, 3, 4, 5});
    test_case.add_expected_output<float>({0, 2, 6, 12, 20, 30});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_expand_model_dependency_to_created_subgraph) {
    const auto model = convert_model("transformations/greater_or_equal.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{5}, {3.f, 5.f, 3.f, 3.f, 6.f});
    test_case.add_input<float>(Shape{5}, {1.f, 4.f, 3.f, 7.f, 8.f});
    test_case.add_expected_output<int32_t>(Shape{5}, {1, 1, 1, 0, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_expand_model_greater_or_equal_inside_if) {
    const auto model = convert_model("transformations/greater_or_equal_inside_if.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    // case when condition == true and any(x >= y)
    // expected value == x * y
    std::vector<float> x(40, 2);
    std::vector<float> y(40);
    std::iota(y.begin(), y.end(), -20.f);
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

OPENVINO_TEST(${BACKEND_NAME}, onnx_expand_context_dependent_model) {
    auto model = convert_model("transformations/softmax_crossentropy_consumed.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{3, 5},
                               {0.54881352186203f,
                                0.7151893377304077f,
                                0.6027633547782898f,
                                0.5448831915855408f,
                                0.42365479469299316f,
                                0.6458941102027893f,
                                0.4375872015953064f,
                                0.891772985458374f,
                                0.9636627435684204f,
                                0.3834415078163147f,
                                0.7917250394821167f,
                                0.5288949012756348f,
                                0.5680445432662964f,
                                0.9255966544151306f,
                                0.07103605568408966f});
    test_case.add_input<int64_t>(Shape{3}, {1, 4, 3});
    test_case.add_expected_output<int32_t>(Shape{}, {1});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_expand_model_with_initializers) {
    const auto model = convert_model("transformations/celu_with_initializers.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output<float>({0.5, 1.0, 1.5, 2.0});
    test_case.run();
}

// ############################################################################ OPERATOR TESTS
OPENVINO_TEST(${BACKEND_NAME}, onnx_model_addmul_abc) {
    auto model = convert_model("addmul_abc.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({9, 10, 11, 12});
    test_case.add_input<float>({5, 6, 7, 8});
    test_case.add_input<float>({1, 2, 3, 4});
    test_case.add_expected_output<float>(Shape{1, 2, 2}, {46, 62, 80, 100});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_argmin_no_keepdims) {
    auto model = convert_model("argmin_no_keepdims.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({2, 1, 3, 10});
    test_case.add_expected_output<int64_t>(Shape{2}, {1, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_batch_norm_default) {
    // Batch Normalization with default parameters
    auto model = convert_model("batchnorm_default.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({-1.f, 0.f, 1.f, 2.f, 3.f, 4.f});  // data {1, 2, 1, 3}
    test_case.add_input<float>({1.f, 1.5f});                      // scale
    test_case.add_input<float>({0.f, 1.f});                       // bias
    test_case.add_input<float>({0.f, 3.f});                       // mean
    test_case.add_input<float>({1.f, 1.5f});                      // var
    test_case.add_expected_output<float>(Shape{1, 2, 1, 3},
                                         {-0.999995f, 0.f, 0.999995f, -0.22474074f, 1.f, 2.2247407f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_batch_norm_opset1) {
    // Batch Normalization with default parameters
    auto model = convert_model("batchnorm_opset1.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({-1.f, 0.f, 1.f, 2.f, 3.f, 4.f});  // data {1, 2, 1, 3}
    test_case.add_input<float>({1.f, 1.5f});                      // scale
    test_case.add_input<float>({0.f, 1.f});                       // bias
    test_case.add_input<float>({0.f, 3.f});                       // mean
    test_case.add_input<float>({1.f, 1.5f});                      // var
    test_case.add_expected_output<float>(Shape{1, 2, 1, 3},
                                         {-0.999995f, 0.f, 0.999995f, -0.22474074f, 1.f, 2.2247407f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_batch_norm_opset6) {
    // Batch Normalization with default parameters
    auto model = convert_model("batchnorm_opset6.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({-1.f, 0.f, 1.f, 2.f, 3.f, 4.f});  // data {1, 2, 1, 3}
    test_case.add_input<float>({1.f, 1.5f});                      // scale
    test_case.add_input<float>({0.f, 1.f});                       // bias
    test_case.add_input<float>({0.f, 3.f});                       // mean
    test_case.add_input<float>({1.f, 1.5f});                      // var
    test_case.add_expected_output<float>(Shape{1, 2, 1, 3},
                                         {-0.999995f, 0.f, 0.999995f, -0.22474074f, 1.f, 2.2247407f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_batch_norm_opset7) {
    // Batch Normalization with default parameters
    auto model = convert_model("batchnorm_opset7.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({-1.f, 0.f, 1.f, 2.f, 3.f, 4.f});  // data {1, 2, 1, 3}
    test_case.add_input<float>({1.f, 1.5f});                      // scale
    test_case.add_input<float>({0.f, 1.f});                       // bias
    test_case.add_input<float>({0.f, 3.f});                       // mean
    test_case.add_input<float>({1.f, 1.5f});                      // var
    test_case.add_expected_output<float>(Shape{1, 2, 1, 3},
                                         {-0.999995f, 0.f, 0.999995f, -0.22474074f, 1.f, 2.2247407f});
    test_case.run();
}
OPENVINO_TEST(${BACKEND_NAME}, onnx_model_batch_norm_opset9) {
    // Batch Normalization with default parameters
    auto model = convert_model("batchnorm_opset9.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({-1.f, 0.f, 1.f, 2.f, 3.f, 4.f});  // data {1, 2, 1, 3}
    test_case.add_input<float>({1.f, 1.5f});                      // scale
    test_case.add_input<float>({0.f, 1.f});                       // bias
    test_case.add_input<float>({0.f, 3.f});                       // mean
    test_case.add_input<float>({1.f, 1.5f});                      // var
    test_case.add_expected_output<float>(Shape{1, 2, 1, 3},
                                         {-0.999995f, 0.f, 0.999995f, -0.22474074f, 1.f, 2.2247407f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_batch_norm_opset14) {
    // Batch Normalization with default parameters
    auto model = convert_model("batchnorm_opset14.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({-1.f, 0.f, 1.f, 2.f, 3.f, 4.f});  // data {1, 2, 1, 3}
    test_case.add_input<float>({1.f, 1.5f});                      // scale
    test_case.add_input<float>({0.f, 1.f});                       // bias
    test_case.add_input<float>({0.f, 3.f});                       // mean
    test_case.add_input<float>({1.f, 1.5f});                      // var
    test_case.add_expected_output<float>(Shape{1, 2, 1, 3},
                                         {-0.999995f, 0.f, 0.999995f, -0.22474074f, 1.f, 2.2247407f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_batch_norm_opset15) {
    // Batch Normalization with default parameters
    auto model = convert_model("batchnorm_opset15.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({-1.f, 0.f, 1.f, 2.f, 3.f, 4.f});  // data {1, 2, 1, 3}
    test_case.add_input<float>({1.f, 1.5f});                      // scale
    test_case.add_input<float>({0.f, 1.f});                       // bias
    test_case.add_input<float>({0.f, 3.f});                       // mean
    test_case.add_input<float>({1.f, 1.5f});                      // var
    test_case.add_expected_output<float>(Shape{1, 2, 1, 3},
                                         {-0.999995f, 0.f, 0.999995f, -0.22474074f, 1.f, 2.2247407f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_relu) {
    // Simple ReLU test
    auto model = convert_model("relu.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({-1, -2, 0, 1, 2, 3});
    test_case.add_expected_output<float>({0, 0, 0, 1, 2, 3});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_sum_opset1) {
    // Simple Sum test for opset1.
    auto model = convert_model("sum_opset1.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({3.f, 0.f, 2.f});
    test_case.add_input<float>({1.f, 3.f, 4.f});
    test_case.add_input<float>({2.f, 6.f, 6.f});
    test_case.add_expected_output<float>(Shape{3}, {6.f, 9.f, 12.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_sum) {
    // Simple Sum test for opset8.
    auto model = convert_model("sum.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({3.f});
    test_case.add_input<float>({1.f, 3.f, 4.f});
    test_case.add_input<float>({2.f, 6.f, 6.f});
    test_case.add_expected_output<float>(Shape{3}, {6.f, 12.f, 13.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_sum_one_input) {
    auto model = convert_model("sum_one_input.onnx");

    // input data shape (3, )
    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({3.f, 0.f, 2.f});
    test_case.add_expected_output<float>({3.f, 0.f, 2.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_cum_sum_1d) {
    auto model = convert_model("cum_sum_1d.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f});
    test_case.add_expected_output<float>(Shape{3}, {1.f, 3.f, 6.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_cum_sum_2d_axis_input) {
    auto model = convert_model("cum_sum_2d_axis_input.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    test_case.add_expected_output<float>(Shape{2, 3}, {1.f, 3.f, 6.f, 4.f, 9.f, 15.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_cum_sum_2d_dynamic_axis_input) {
    auto model = convert_model("cum_sum_2d_dynamic_axis_input.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    test_case.add_input<std::int32_t>({1});
    test_case.add_expected_output<float>(Shape{2, 3}, {1.f, 3.f, 6.f, 4.f, 9.f, 15.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_cum_sum_2d_axis_input_1d) {
    auto model = convert_model("cum_sum_2d_axis_input_1d.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    test_case.add_expected_output<float>(Shape{2, 3}, {1.f, 3.f, 6.f, 4.f, 9.f, 15.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_cum_sum_2d_dynamic_axis_input_1d) {
    auto model = convert_model("cum_sum_2d_dynamic_axis_input_1d.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    test_case.add_input<std::int64_t>({0});
    test_case.add_expected_output<float>(Shape{2, 3}, {1.f, 2.f, 3.f, 5.f, 7.f, 9.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_cum_sum_3d_exclusive_reverse) {
    auto model = convert_model("cum_sum_3d_exclusive_reverse.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({1.f,  2.f,  3.f,  4.f,  5.f,  6.f,  7.f,  8.f,  9.f,  10.f, 11.f, 12.f,
                                13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f});
    test_case.add_expected_output<float>(Shape{2, 3, 4},
                                         {13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f,
                                          0.f,  0.f,  0.f,  0.f,  0.f,  0.f,  0.f,  0.f,  0.f,  0.f,  0.f,  0.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_min_two_inputs_opset1) {
    auto model = convert_model("min_two_inputs_opset1.onnx");

    // input data shape (3, )
    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({1.f, 2.f, 1.f});
    test_case.add_input<float>({1.f, 4.f, 4.f});
    test_case.add_expected_output<float>({1.f, 2.f, 1.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_min_two_inputs) {
    auto model = convert_model("min_two_inputs.onnx");

    // input data shape (3, )
    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({2.f});
    test_case.add_input<float>({1.f, 4.f, 4.f});
    test_case.add_expected_output<float>({1.f, 2.f, 2.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_max_opset1) {
    auto model = convert_model("max_opset1.onnx");

    // input data shape (3, )
    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({3.f, 2.f, 1.f});
    test_case.add_input<float>({1.f, 4.f, 4.f});
    test_case.add_input<float>({2.f, 5.f, 3.f});

    test_case.add_expected_output<float>({3.f, 5.f, 4.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_max) {
    auto model = convert_model("max.onnx");

    // input data shape (3, )
    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({1.f, 4.f, 4.f});
    test_case.add_input<float>({3.f});
    test_case.add_input<float>({2.f, 5.f, 3.f});

    test_case.add_expected_output<float>({3.f, 5.f, 4.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mean_opset1) {
    auto model = convert_model("mean_opset1.onnx");

    // input data shape (3, )
    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({3.f, 0.f, 2.f});
    test_case.add_input<float>({1.f, 3.f, 4.f});
    test_case.add_input<float>({2.f, 6.f, 6.f});

    test_case.add_expected_output<float>({2.f, 3.f, 4.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mean) {
    auto model = convert_model("mean.onnx");

    // input data shape (3, )
    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({3.f});
    test_case.add_input<float>({1.f, 2.f, 5.f});
    test_case.add_input<float>({2.f, 7.f, 7.f});

    test_case.add_expected_output<float>({2.f, 4.f, 5.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_gemm_abc) {
    auto model = convert_model("gemm_abc.onnx");

    Inputs inputs;
    inputs.emplace_back(
        ov::test::NDArray<float, 2>({{1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12}, {13, 14, 15, 16, 17, 18}})
            .get_vector());

    inputs.emplace_back(ov::test::NDArray<float, 2>({{19, 20, 21, 22},
                                                     {23, 24, 25, 26},
                                                     {27, 28, 29, 30},
                                                     {31, 32, 33, 34},
                                                     {35, 36, 37, 38},
                                                     {39, 40, 41, 42}})
                            .get_vector());

    inputs.emplace_back(ov::test::NDArray<float, 2>({{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}).get_vector());

    auto expected_output =
        ov::test::NDArray<float, 2>({{340, 350.5, 361, 371.5}, {862, 890.5, 919, 947.5}, {1384, 1430.5, 1477, 1523.5}})
            .get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_matmul) {
    auto model = convert_model("matmul.onnx");

    std::vector<std::vector<float>> inputs;

    inputs.emplace_back(ov::test::NDArray<float, 2>({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}).get_vector());

    inputs.emplace_back(
        ov::test::NDArray<float, 2>({{13, 14, 15}, {16, 17, 18}, {19, 20, 21}, {22, 23, 24}}).get_vector());

    auto expected_output =
        ov::test::NDArray<float, 2>({{190, 200, 210}, {470, 496, 522}, {750, 792, 834}}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_softmax_0D) {
    auto model = convert_model("softmax_0D.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output<float>({1.0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_softmax_1D) {
    auto model = convert_model("softmax_1D.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({-1.0, 0.0, 1.0});
    test_case.add_expected_output<float>({0.09003058f, 0.24472848f, 0.66524094f});
    test_case.run();
}
namespace {
// common input for all Softmax 3D test cases (Shape = {3,4,5})
// clang-format off
const std::vector<float> SOFTMAX_INPUT = {
    2.75793882f,  -0.50841322f, 0.82013929f,  -0.62409912f, -0.96136118f,
    0.21004745f,  1.38337255f,  1.19030397f,  2.0940445f,   -0.03551657f,
    -0.78686039f, 1.992782f,    0.04300319f,  -0.29230777f, -0.56797112f,
    -1.26732165f, -0.61935399f, 0.57670432f,  0.92844898f,  2.82469233f,

    0.98721677f,  -0.05100663f, -1.21178917f, -0.17530157f, 1.40051805f,
    -0.13259761f, -1.14313018f, 0.2673723f,   -0.87996154f, 1.29053106f,
    1.55f,        0.8396538f,   1.20729817f,  0.23727845f,  -0.89113606f,
    -1.70909842f, 0.26460363f,  -0.70566808f, 2.383518f,    1.07024615f,

    -1.21722605f, 0.82919357f,  0.55765697f,  0.12657686f,  0.63432172f,
    0.75425957f,  -2.43721014f, -1.24478184f, 2.65316853f,  1.19509542f,
    -0.95523998f, 0.5149006f,   -0.01151649f, 0.68327026f,  -0.4589638f,
    -0.46554745f, 0.21055324f,  0.39266729f,  2.05098086f,  1.83207919f};
}  // namespace
// clang-format on

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_softmax_axis_0) {
    auto model = convert_model("softmax_axis_0.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(SOFTMAX_INPUT);

    // clang-format off
    test_case.add_expected_output<float>(
        Shape{3, 4, 5},
        {0.09683057f, 0.00369363f, 0.01394559f, 0.00329012f, 0.00234823f,
         0.00757665f, 0.02449322f, 0.02019284f, 0.04985249f, 0.00592694f,
         0.00279593f, 0.04505148f, 0.00641108f, 0.00458466f, 0.00348007f,
         0.00172928f, 0.00330577f, 0.01093237f, 0.01554086f, 0.10351497f,

         0.01648154f, 0.00583583f, 0.00182802f, 0.00515374f, 0.02491679f,
         0.00537859f, 0.00195794f, 0.00802367f, 0.00254737f, 0.0223216f,
         0.02893419f, 0.0142204f,  0.02053893f, 0.00778581f, 0.00251907f,
         0.00111174f, 0.00800149f, 0.0030324f,  0.06658917f, 0.0179084f,

         0.00181811f, 0.01407243f, 0.01072611f, 0.0069699f,  0.01158077f,
         0.01305647f, 0.00053677f, 0.0017687f,  0.08719896f, 0.02028982f,
         0.00236265f, 0.01027717f, 0.0060709f,  0.01216173f, 0.00388087f,
         0.00385541f, 0.00758048f, 0.00909469f, 0.04775123f, 0.03836337f});
    // clang-format on

    test_case.run(6);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_softmax_axis_1) {
    auto model = convert_model("softmax_axis_1.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(SOFTMAX_INPUT);

    // clang-format off
    test_case.add_expected_output<float>(
        Shape{3, 4, 5},
        {0.22757064f, 0.00868076f, 0.03277484f, 0.00773243f, 0.0055188f,
         0.0178066f,  0.05756383f, 0.04745709f, 0.11716303f, 0.01392945f,
         0.00657097f, 0.10587974f, 0.01506727f, 0.01077484f, 0.00817884f,
         0.00406413f, 0.00776921f, 0.0256932f,  0.03652405f, 0.24328028f,

         0.06217413f, 0.02201481f, 0.00689594f, 0.01944171f, 0.09399488f,
         0.02028993f, 0.00738604f, 0.03026811f, 0.00960958f, 0.08420492f,
         0.10914991f, 0.05364435f, 0.07748005f, 0.02937079f, 0.0095028f,
         0.00419387f, 0.03018442f, 0.01143929f, 0.2511977f,  0.06755678f,

         0.00587593f, 0.04548053f, 0.0346656f,  0.02252594f, 0.03742775f,
         0.04219705f, 0.00173478f, 0.00571623f, 0.2818174f,  0.06557446f,
         0.00763582f, 0.03321466f, 0.01962049f, 0.03930537f, 0.01254255f,
         0.01246025f, 0.02449929f, 0.02939305f, 0.15432668f, 0.12398617f});
    // clang-format on

    test_case.run(4);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_softmax_axis_1_opset11) {
    auto model = convert_model("softmax_axis_1_opset11.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(SOFTMAX_INPUT);

    // clang-format off
    test_case.add_expected_output<float>(
        Shape{3, 4, 5},
        {0.88890495f, 0.04825497f, 0.27088348f, 0.04490523f, 0.02037154f,
         0.06955369f, 0.31998834f, 0.39223197f, 0.68041159f, 0.05141776f,
         0.02566661f, 0.5885689f,  0.12453075f, 0.06257374f, 0.03019055f,
         0.01587475f, 0.0431878f,  0.21235381f, 0.21210944f, 0.89802015f,

         0.31752626f, 0.19442629f, 0.0546935f,  0.06279221f, 0.36823282f,
         0.10362164f, 0.06523066f, 0.24006419f, 0.03103672f, 0.32987983f,
         0.55743381f, 0.473766f,   0.61451431f, 0.09486084f, 0.03722801f,
         0.02141829f, 0.26657706f, 0.090728f,   0.81131024f, 0.26465935f,

         0.08619648f, 0.43343993f, 0.3877785f,  0.04523505f, 0.15625437f,
         0.61900597f, 0.01653285f, 0.06394322f, 0.56592636f, 0.27376196f,
         0.11201305f, 0.31654337f, 0.21947994f, 0.07893034f, 0.05236297f,
         0.18278451f, 0.23348385f, 0.32879834f, 0.30990825f, 0.5176207f});
    // clang-format on

    test_case.run(4);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_softmax_axis_negative_1_opset11) {
    auto model = convert_model("softmax_axis_negative_1_opset11.onnx");

    auto test_case = ov::test::TestCase(model);
    test_case.add_input<float>(SOFTMAX_INPUT);

    // clang-format off
    test_case.add_expected_output<float>(
        Shape{3, 4, 5},
        {0.80619484f, 0.03075256f, 0.1161086f,  0.027393f,   0.01955098f,
         0.07012683f, 0.22670066f, 0.18689778f, 0.4614171f,  0.05485764f,
         0.04486171f, 0.7228683f,  0.10286818f, 0.07356264f, 0.05583908f,
         0.01280724f, 0.02448298f, 0.08096659f, 0.11509769f, 0.76664555f,

         0.30399805f, 0.10764059f, 0.03371745f, 0.09505949f, 0.4595844f,
         0.13369875f, 0.04866969f, 0.19944906f, 0.0633215f,  0.554861f,
         0.39101103f, 0.19217177f, 0.27755913f, 0.10521588f, 0.03404216f,
         0.01150354f, 0.08279411f, 0.03137731f, 0.6890207f,  0.18530433f,

         0.0402528f,  0.31156224f, 0.23747502f, 0.15431291f, 0.25639707f,
         0.10627912f, 0.00436928f, 0.01439711f, 0.7097961f,  0.16515835f,
         0.06798343f, 0.29571748f, 0.17468554f, 0.34994435f, 0.11166911f,
         0.03615172f, 0.07108136f, 0.08527993f, 0.4477579f,  0.35972902f});
    // clang-format on

    test_case.run(6);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_softmax_axis_negative_1_opset13) {
    auto model = convert_model("softmax_axis_negative_1_opset13.onnx");

    auto test_case = ov::test::TestCase(model);
    test_case.add_input<float>(SOFTMAX_INPUT);

    // clang-format off
    test_case.add_expected_output<float>(
        Shape{3, 4, 5},
        {0.80619484f, 0.03075256f, 0.1161086f,  0.027393f,   0.01955098f,
         0.07012683f, 0.22670066f, 0.18689778f, 0.4614171f,  0.05485764f,
         0.04486171f, 0.7228683f,  0.10286818f, 0.07356264f, 0.05583908f,
         0.01280724f, 0.02448298f, 0.08096659f, 0.11509769f, 0.76664555f,

         0.30399805f, 0.10764059f, 0.03371745f, 0.09505949f, 0.4595844f,
         0.13369875f, 0.04866969f, 0.19944906f, 0.0633215f,  0.554861f,
         0.39101103f, 0.19217177f, 0.27755913f, 0.10521588f, 0.03404216f,
         0.01150354f, 0.08279411f, 0.03137731f, 0.6890207f,  0.18530433f,

         0.0402528f,  0.31156224f, 0.23747502f, 0.15431291f, 0.25639707f,
         0.10627912f, 0.00436928f, 0.01439711f, 0.7097961f,  0.16515835f,
         0.06798343f, 0.29571748f, 0.17468554f, 0.34994435f, 0.11166911f,
         0.03615172f, 0.07108136f, 0.08527993f, 0.4477579f,  0.35972902f});
    // clang-format on

    test_case.run(6);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_sub) {
    auto model = convert_model("sub.onnx");

    Inputs inputs;
    inputs.emplace_back(ov::test::NDArray<float, 3>({{{1, 2, 3}}}).get_vector());

    inputs.emplace_back(ov::test::NDArray<float, 3>({{{4, 5, 7}}}).get_vector());

    auto expected_output = ov::test::NDArray<float, 3>({{{-3, -3, -4}}}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_div) {
    auto model = convert_model("div.onnx");

    Inputs inputs;
    inputs.emplace_back(ov::test::NDArray<float, 3>({{{1, 2, 3}}}).get_vector());
    inputs.emplace_back(ov::test::NDArray<float, 3>({{{1, 4, 12}}}).get_vector());

    auto expected_output = ov::test::NDArray<float, 3>({{{1, 0.5, 0.25}}}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_add_bcast) {
    auto model = convert_model("add_bcast.onnx");

    Inputs inputs;
    inputs.emplace_back(
        ov::test::NDArray<float, 3>({{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                     {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                     {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
            .get_vector());

    inputs.emplace_back(ov::test::NDArray<float, 1>({1, 2, 3, 4, 5}).get_vector());

    auto expected_output =
        ov::test::NDArray<float, 4>({{{{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}},
                                      {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}},
                                      {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}}}})
            .get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_nonmaxsuppression_center_point_box_format) {
    auto model = convert_model("nonmaxsuppression_center_point_box_format.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

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

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_nonmaxsuppression_single_box) {
    auto model = convert_model("nonmaxsuppression_single_box.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input(std::vector<float>({0.0f, 0.0f, 1.0f, 1.0f}));  // boxes
    test_case.add_input(std::vector<float>({0.9f}));                    // scores
    test_case.add_input(std::vector<int64_t>({3}));                     // max_output_boxes_per_class
    test_case.add_input(std::vector<float>({0.5f}));                    // iou_threshold
    test_case.add_input(std::vector<float>({0.0f}));                    // score_threshold

    test_case.add_expected_output<int64_t>(Shape{1, 3}, {0, 0, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_nonmaxsuppression_v9_single_box) {
    auto model = convert_model("nonmaxsuppression_v9_single_box.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input(std::vector<float>({0.0f, 0.0f, 1.0f, 1.0f}));  // boxes
    test_case.add_input(std::vector<float>({0.9f}));                    // scores
    test_case.add_input(std::vector<int64_t>({3}));                     // max_output_boxes_per_class
    test_case.add_input(std::vector<float>({0.5f}));                    // iou_threshold
    test_case.add_input(std::vector<float>({0.0f}));                    // score_threshold

    test_case.add_expected_output<int64_t>(Shape{1, 3}, {0, 0, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_nonmaxsuppression_default_score_threshold) {
    // TEMPLATE plugin has a run-to-run issue with this test, CVS-127743, CVS-122120
    if (std::string("${BACKEND_NAME}") == std::string("INTERPRETER")) {
        GTEST_SKIP();
    }

    auto model = convert_model("nms_default_score_threshold.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input(
        Shape{1, 50, 4},
        std::vector<float>(
            {278.862060546875f,   453.5412902832031f,  295.09234619140625f, 470.2095031738281f,  225.9730682373047f,
             387.33990478515625f, 241.69297790527344f, 403.43377685546875f, 281.3062438964844f,  453.8412170410156f,
             298.6865539550781f,  470.9977111816406f,  216.9517364501953f,  450.6717529296875f,  232.95777893066406f,
             466.14276123046875f, 217.54473876953125f, 449.9130859375f,     233.97265625f,       466.1539306640625f,
             279.0079650878906f,  453.865234375f,      294.8210144042969f,  470.123046875f,      226.5626983642578f,
             388.5235290527344f,  242.2290496826172f,  404.2589416503906f,  216.49752807617188f, 450.7710876464844f,
             233.07443237304688f, 466.7010192871094f,  281.3638000488281f,  454.33892822265625f, 298.5252990722656f,
             471.1678466796875f,  217.3330841064453f,  451.484130859375f,   234.1898651123047f,  466.83148193359375f,
             187.2439727783203f,  466.8524475097656f,  208.7089385986328f,  489.7967224121094f,  257.8833923339844f,
             515.705322265625f,   280.8927917480469f,  539.775146484375f,   226.52525329589844f, 387.7011413574219f,
             241.6272430419922f,  403.7854919433594f,  187.38221740722656f, 466.5717468261719f,  209.05845642089844f,
             489.4494323730469f,  217.56448364257812f, 451.1393737792969f,  233.90216064453125f, 466.1475524902344f,
             279.45611572265625f, 454.00299072265625f, 296.16424560546875f, 471.84521484375f,    279.04486083984375f,
             453.9889221191406f,  295.2816162109375f,  470.4144592285156f,  187.18997192382812f, 466.4650573730469f,
             209.26266479492188f, 488.8149719238281f,  189.04197692871094f, 469.8923034667969f,  208.8195037841797f,
             491.5357971191406f,  216.47879028320312f, 450.1073303222656f,  233.21575927734375f, 466.9475402832031f,
             278.86163330078125f, 454.966552734375f,   296.38958740234375f, 471.9764404296875f,  259.4800720214844f,
             515.1390991210938f,  282.3655090332031f,  539.4806518554688f,  285.031494140625f,   389.0125427246094f,
             302.09747314453125f, 406.9799499511719f,  285.1270446777344f,  389.06890869140625f, 301.2108459472656f,
             405.7711181640625f,  188.17117309570312f, 467.71533203125f,    208.49929809570312f, 490.401611328125f,
             278.93292236328125f, 453.8080139160156f,  295.4295654296875f,  469.9015808105469f,  279.0393371582031f,
             454.2393798828125f,  296.3529357910156f,  471.6363525390625f,  187.29873657226562f, 467.9837951660156f,
             208.29107666015625f, 489.8014221191406f,  187.79478454589844f, 466.6510314941406f,  208.3644561767578f,
             490.2976989746094f,  188.4196014404297f,  468.3448486328125f,  209.06849670410156f, 491.94384765625f,
             281.4726867675781f,  454.0541687011719f,  298.2876892089844f,  470.2845764160156f,  225.8560333251953f,
             387.4819030761719f,  241.4767608642578f,  403.4317321777344f,  280.7021484375f,     455.43206787109375f,
             297.9931640625f,     471.99749755859375f, 226.0373077392578f,  387.4749450683594f,  241.48097229003906f,
             403.4716491699219f,  259.018310546875f,   515.3871459960938f,  281.7872314453125f,  540.0093383789062f,
             217.71246337890625f, 450.4556884765625f,  234.254150390625f,   467.68182373046875f, 257.5479736328125f,
             518.8912353515625f,  280.48260498046875f, 541.3863525390625f,  216.87359619140625f, 450.3395080566406f,
             232.39752197265625f, 465.5039367675781f,  258.2445068359375f,  515.2009887695312f,  280.29803466796875f,
             540.3602905273438f,  217.54478454589844f, 451.3944091796875f,  233.6602020263672f,  467.51971435546875f,
             258.30133056640625f, 515.2357788085938f,  280.1400146484375f,  541.3275756835938f,  217.05136108398438f,
             451.8975524902344f,  232.9573974609375f,  466.9907531738281f,  215.86386108398438f, 450.801025390625f,
             232.117919921875f,   466.3701171875f,     279.01593017578125f, 453.6647644042969f,  296.13372802734375f,
             471.4644470214844f,  280.1851806640625f,  454.41900634765625f, 296.481201171875f,   471.63104248046875f,
             259.1214904785156f,  516.8644409179688f,  281.7276306152344f,  541.0162963867188f,  285.2935485839844f,
             389.03515625f,       302.1134948730469f,  406.89373779296875f, 279.6715393066406f,  455.1846923828125f,
             296.6995544433594f,  471.5782470703125f,  258.1405029296875f,  518.9312744140625f,  281.019287109375f,
             541.5760498046875f,  187.80953979492188f, 466.8480224609375f,  208.54336547851562f, 489.9696044921875f}));
    test_case.add_input(
        Shape{1, 1, 50},
        std::vector<float>(
            {5.485373497009277f,  5.469169616699219f,  5.450349807739258f,  5.446445465087891f, 5.43833065032959f,
             5.407294273376465f,  5.3790669441223145f, 5.3575520515441895f, 5.348986625671387f, 5.309826850891113f,
             5.266261577606201f,  5.230800151824951f,  5.079848766326904f,  5.066829204559326f, 4.913329601287842f,
             4.895563125610352f,  4.8786115646362305f, 4.872953414916992f,  4.825906753540039f, 4.812736511230469f,
             4.761179447174072f,  4.657320022583008f,  4.640903949737549f,  4.63286828994751f,  4.600266933441162f,
             4.599870204925537f,  4.5536088943481445f, 4.521742820739746f,  4.465426445007324f, 4.4556074142456055f,
             4.451722621917725f,  4.416017055511475f,  4.410635471343994f,  4.403003215789795f, 4.387508392333984f,
             4.3634934425354f,    4.362300872802734f,  4.348748683929443f,  4.345107555389404f, 4.32416296005249f,
             4.3132781982421875f, 4.287333965301514f,  4.223401069641113f,  4.220005035400391f, 4.179988861083984f,
             4.099865436553955f,  4.097578048706055f,  4.075544357299805f,  4.0459885597229f}));

    test_case.add_expected_output<int64_t>(Shape{7, 3},
                                           {0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 10, 0, 0, 11, 0, 0, 22});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_log_sum) {
    auto model = convert_model("reduce_log_sum.onnx");

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        ov::test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = ov::test::NDArray<float, 4>({{{{2.77258872f}}}}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_log_sum_18) {
    auto model = convert_model("reduce_log_sum_18.onnx");

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        ov::test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = ov::test::NDArray<float, 4>({{{{2.77258872f}}}}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_log_sum_18_axes_as_input) {
    auto model = convert_model("reduce_log_sum_18_axes_as_input.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(Shape{1, 1, 4, 4}, {2, 1, 4, 2, 3, 1, 3, 2, 4, 2, 4, 2, 2, 2, 1, 4});
    test_case.add_input<int64_t>({3});

    test_case.add_expected_output(Shape{1, 1, 4, 1},
                                  std::vector<float>{2.19722458f, 2.19722458f, 2.48490665f, 2.19722458f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_log_sum_exp) {
    auto model = convert_model("reduce_log_sum_exp.onnx");

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        ov::test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = ov::test::NDArray<float, 4>({{{{3.77258872f}}}}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_log_sum_exp_18) {
    auto model = convert_model("reduce_log_sum_exp_18.onnx");

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        ov::test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}}).get_vector()};

    // output data shape (4)
    auto expected_output = ov::test::NDArray<float, 1>({2.38629f, 2.38629f, 2.38629f, 2.38629f}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run_with_tolerance_as_fp();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_l1) {
    auto model = convert_model("reduce_l1.onnx");

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        ov::test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = ov::test::NDArray<float, 4>({{{{16}}}}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_l1_18) {
    auto model = convert_model("reduce_l1_18.onnx");

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        ov::test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = ov::test::NDArray<float, 4>({{{{16}}}}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_l1_18_axes_as_input) {
    auto model = convert_model("reduce_l1_18_axes_as_input.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(Shape{1, 1, 4, 4}, {2, 1, 4, 2, 3, 1, 3, 2, 4, 2, 4, 2, 2, 2, 1, 4});
    test_case.add_input<int64_t>({3});

    test_case.add_expected_output(Shape{1, 1, 4, 1}, std::vector<float>{9, 9, 12, 9});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_l2) {
    auto model = convert_model("reduce_l2.onnx");

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        ov::test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = ov::test::NDArray<float, 4>({{{{4}}}}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_l2_11) {
    auto model = convert_model("reduce_l2_11.onnx");

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        ov::test::NDArray<float, 4>({{{{3, 3, 3, 3}, {3, 3, 3, 3}, {3, 3, 3, 3}, {3, 3, 3, 3}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = ov::test::NDArray<float, 4>({{{{12}}}}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_l2_13) {
    auto model = convert_model("reduce_l2_13.onnx");

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        ov::test::NDArray<float, 4>({{{{4, 4, 4, 4}, {4, 4, 4, 4}, {4, 4, 4, 4}, {4, 4, 4, 4}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = ov::test::NDArray<float, 4>({{{{16}}}}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_l2_18) {
    auto model = convert_model("reduce_l2_18.onnx");

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        ov::test::NDArray<float, 4>({{{{5, 5, 5, 5}, {5, 5, 5, 5}, {5, 5, 5, 5}, {5, 5, 5, 5}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = ov::test::NDArray<float, 4>({{{{20}}}}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_max) {
    auto model = convert_model("reduce_max.onnx");

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        ov::test::NDArray<float, 4>({{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = ov::test::NDArray<float, 4>({{{{16}}}}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_max_18) {
    // TEMPLATE plugin has an issue with evaluation for u8 type
    if (std::string("${BACKEND_NAME}") == std::string("INTERPRETER")) {
        GTEST_SKIP();
    }

    auto model = convert_model("reduce_max_18.onnx");

    // input data shape (1, 1, 4, 4)
    std::vector<std::vector<uint8_t>> inputs{
        ov::test::NDArray<uint8_t, 4>({{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}}})
            .get_vector()};

    // output data shape (1,)
    auto expected_output = ov::test::NDArray<uint8_t, 1>({13, 14, 15, 16}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_max_invalid_axes) {
    EXPECT_THROW(convert_model("reduce_max_invalid_axes.onnx"), ov::Exception);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_mean) {
    auto model = convert_model("reduce_mean.onnx");

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        ov::test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = ov::test::NDArray<float, 4>({{{{1}}}}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(Shape{}, expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_mean_18) {
    auto model = convert_model("reduce_mean_18.onnx");

    // input data shape (1, 1, 4, 4)
    std::vector<std::vector<uint8_t>> inputs{
        ov::test::NDArray<uint8_t, 4>({{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}}})
            .get_vector()};

    // output data shape (1,)
    auto expected_output = ov::test::NDArray<uint8_t, 1>({7, 8, 9, 10}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_min) {
    auto model = convert_model("reduce_min.onnx");

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        ov::test::NDArray<float, 4>({{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = ov::test::NDArray<float, 4>({{{{1}}}}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_prod) {
    auto model = convert_model("reduce_prod.onnx");

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        ov::test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = ov::test::NDArray<float, 4>({{{{1}}}}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_prod_18) {
    // TEMPLATE plugin has an issue with evaluation for reduceprod, CVS-148827
    if (std::string("${BACKEND_NAME}") == std::string("INTERPRETER")) {
        GTEST_SKIP();
    }

    auto model = convert_model("reduce_prod_18.onnx");

    // input data shape (1, 1, 4, 4)
    std::vector<std::vector<uint8_t>> inputs{
        ov::test::NDArray<uint8_t, 4>({{{{1, 1, 1, 1}, {1, 2, 3, 4}, {1, 1, 1, 1}, {2, 2, 2, 2}}}}).get_vector()};

    // output data shape (4)
    auto expected_output = ov::test::NDArray<uint8_t, 1>({2, 4, 6, 8}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_sum) {
    auto model = convert_model("reduce_sum.onnx");

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        ov::test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = ov::test::NDArray<float, 4>({{{{16}}}}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_dynamic_rank_input) {
    auto model = convert_model("reduce_sum_dynamic_rank_input.onnx");
    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(
        Shape{1, 1, 4, 4},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

    test_case.add_expected_output<float>(Shape{1, 1, 1, 1}, {16.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_square) {
    auto model = convert_model("reduce_sum_square.onnx");

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        ov::test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = ov::test::NDArray<float, 4>({{{{16}}}}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_square_13) {
    auto model = convert_model("reduce_sum_square_13.onnx");

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        ov::test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = ov::test::NDArray<float, 4>({{{{16}}}}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_square_18) {
    auto model = convert_model("reduce_sum_square_18.onnx");

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        ov::test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = ov::test::NDArray<float, 4>({{{{16}}}}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_as_constant) {
    auto model = convert_model("reduce_sum_13_axes_as_constant.onnx");

    Inputs inputs{ov::test::NDArray<float, 4>({{{{1.0f, 1.0f, 1.0f, 1.0f},
                                                 {1.0f, 1.0f, 1.0f, 1.0f},
                                                 {1.0f, 1.0f, 1.0f, 1.0f},
                                                 {1.0f, 1.0f, 1.0f, 1.0f}}}})
                      .get_vector()};

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_expected_output<float>(Shape{1, 1, 1, 1}, {16.0f});

    test_case.add_multiple_inputs(inputs);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_as_constant_single_axis) {
    auto model = convert_model("reduce_sum_13_axes_as_constant_single_axis.onnx");

    Inputs inputs{ov::test::NDArray<float, 3>({{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}}).get_vector()};

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_expected_output<float>(Shape{2, 1, 3}, {5.0f, 7.0f, 9.0f, 17.0f, 19.0f, 21.0f});

    test_case.add_multiple_inputs(inputs);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_as_constant_keepdims_off) {
    auto model = convert_model("reduce_sum_13_axes_as_constant_keepdims_off.onnx");

    // input data shape (1, 1, 4, 4)
    Inputs inputs{ov::test::NDArray<float, 4>({{{{1.0f, 1.0f, 1.0f, 1.0f},
                                                 {1.0f, 1.0f, 1.0f, 1.0f},
                                                 {1.0f, 1.0f, 1.0f, 1.0f},
                                                 {1.0f, 1.0f, 1.0f, 1.0f}}}})
                      .get_vector()};

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_expected_output<float>(Shape{}, {16.0f});

    test_case.add_multiple_inputs(inputs);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_as_input) {
    auto model = convert_model("reduce_sum_13_axes_as_input.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({1.0f, 2.0f, 3.0f, 4.0f});
    test_case.add_input<int64_t>({1});

    test_case.add_expected_output<float>(Shape{2, 1}, {3.0f, 7.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_as_0_dim_input) {
    auto model = convert_model("reduce_sum_13_axes_as_0_dim_input.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

    test_case.add_expected_output<float>(Shape{3, 2, 2},
                                         {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_input_dynamic) {
    auto model = convert_model("reduce_sum_13_input_dynamic.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<int64_t>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

    test_case.add_expected_output<int64_t>(Shape{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {5});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_empty) {
    auto model = convert_model("reduce_sum_13_axes_empty.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

    test_case.add_expected_output<float>(Shape{1, 1, 1, 1}, {16.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_empty_dynamic_rank_input) {
    auto model = convert_model("reduce_sum_13_axes_empty_dynamic_rank_input.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(
        Shape{1, 1, 4, 4},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

    test_case.add_expected_output<float>(Shape{1, 1, 1, 1}, {16.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_empty_with_noop) {
    auto model = convert_model("reduce_sum_13_axes_empty_with_noop.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(
        {1.f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

    test_case.add_expected_output<float>(
        Shape{1, 1, 4, 4},
        {1.f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_empty_without_noop) {
    auto model = convert_model("reduce_sum_13_axes_empty_without_noop.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(
        {1.f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

    test_case.add_expected_output<float>(Shape{1, 1, 1, 1}, {16.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize10_asymertic_last_dim) {
    const auto model = convert_model("resize10_asymertic_last_dim.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
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

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize10_asymertic_dim_in_the_middle) {
    const auto model = convert_model("resize10_asymertic_dim_in_the_middle.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
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

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_empty_constant_as_input) {
    // this model contains a Constant node with an empty underlying tensor
    // this node is connected to the "roi" input of the Resize op but this input should be
    // ignored since the Resize coordinate_transformation_mode is set to asymmetric
    const auto model = convert_model("resize11_empty_constant_as_input.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
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

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize10_down_scales_const_linear) {
    const auto model = convert_model("resize10_down_scales_const_linear.onnx");

    // Input data shape (1, 1, 2, 4)
    // Input const scales values {1.0, 1.0, 0.6, 0.6}
    // mode: linear

    Shape expected_output_shape{1, 1, 1, 2};
    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    test_case.add_expected_output<float>(expected_output_shape, {1.0f, 2.6666665f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize10_down_scales_const_nearest) {
    const auto model = convert_model("resize10_down_scales_const_nearest.onnx");

    // Input data shape (1, 1, 2, 4)
    // Input const scales values {1.0, 1.0, 0.6, 0.6}
    // mode: nearest

    Shape expected_output_shape{1, 1, 1, 2};
    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    test_case.add_expected_output<float>(expected_output_shape, {1.0, 3.0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize10_up_scales_const_linear) {
    const auto model = convert_model("resize10_up_scales_const_linear.onnx");

    // Input data shape (1, 1, 2, 2)
    // Input const scales values {1.0, 1.0, 2.0, 2.0}
    // mode: nearest

    Shape expected_output_shape{1, 1, 4, 4};
    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({1.0, 2.0, 3.0, 4.0});
    test_case.add_expected_output<float>(
        expected_output_shape,
        {1.0, 1.5, 2.0, 2.0, 2.0, 2.5, 3.0, 3.0, 3.0, 3.5, 4.0, 4.0, 3.0, 3.5, 4.0, 4.0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize10_up_scales_const_nearest) {
    const auto model = convert_model("resize10_up_scales_const_nearest.onnx");

    // Input data shape (1, 1, 2, 2)
    // Input const scales values {1.0, 1.0, 2.0, 3.0}
    // mode: linear

    Shape expected_output_shape{1, 1, 4, 6};
    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({1.0, 2.0, 3.0, 4.0});
    test_case.add_expected_output<float>(expected_output_shape,
                                         {1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
                                          3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_down_scales_linear_asymmetric) {
    const auto model = convert_model("resize11_down_scales_linear_asymmetric.onnx");

    const Shape expected_output_shape{1, 1, 1, 2};
    auto test_case = ov::test::TestCase(model, s_device);
    const size_t input_size = 8;
    std::vector<float> input_data(input_size);
    std::iota(std::begin(input_data), std::end(input_data), 1.0f);
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(expected_output_shape, {1.0f, 2.66666651f});

    test_case.run_with_tolerance_as_fp();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_scales_nearest_asymmetric_floor_dynamic_sizes) {
    const auto model = convert_model("resize11_scales_nearest_asymmetric_floor_dynamic_scales.onnx");

    const Shape expected_output_shape{2, 1, 4, 1};
    auto test_case = ov::test::TestCase(model, s_device);
    const std::vector<float> input_data{1.0f, 3.0f, 4.0f, 8.0f, 6.0f, 2.0f, 7.0f, 11.0f};
    test_case.add_input<float>(input_data);
    test_case.add_input<float>(std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});  // roi
    test_case.add_input<float>(std::vector<float>{1.0f, 1.0f, 2.0f, 0.5f});                          // scales
    test_case.add_expected_output<float>(expected_output_shape, {1.0f, 1.0f, 4.0f, 4.0f, 6.0f, 6.0f, 7.0f, 7.0f});

    test_case.run_with_tolerance_as_fp();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_up_scales_linear_asymmetric) {
    const auto model = convert_model("resize11_up_scales_linear_asymmetric.onnx");

    const Shape expected_output_shape{2, 1, 4, 8};
    auto test_case = ov::test::TestCase(model, s_device);
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

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_scales_nearest_asymmetric_floor) {
    const auto model = convert_model("resize11_scales_nearest_asymmetric_floor.onnx");

    const Shape expected_output_shape{2, 1, 4, 1};
    auto test_case = ov::test::TestCase(model, s_device);
    const std::vector<float> input_data{1.0f, 3.0f, 4.0f, 8.0f, 6.0f, 2.0f, 7.0f, 11.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(expected_output_shape, {1.0f, 1.0f, 4.0f, 4.0f, 6.0f, 6.0f, 7.0f, 7.0f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_up_scales_cubic_align_corners) {
    const auto model = convert_model("resize11_up_scales_cubic_align_corners.onnx");

    const Shape expected_output_shape{1, 1, 8, 8};
    auto test_case = ov::test::TestCase(model, s_device);
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

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_up_scales_tf_half_pixel) {
    const auto model = convert_model("resize11_up_scales_tf_half_pixel.onnx");

    const Shape expected_output_shape{1, 1, 8, 8};
    auto test_case = ov::test::TestCase(model, s_device);
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

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_all_attributes_default) {
    const auto model = convert_model("resize11_up_sizes_all_attributes_default.onnx");

    const Shape expected_output_shape{1, 1, 7, 8};
    auto test_case = ov::test::TestCase(model, s_device);
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

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_sizes_nearest_asymmetric_floor) {
    const auto model = convert_model("resize11_sizes_nearest_asymmetric_floor.onnx");

    const Shape expected_output_shape{2, 1, 4, 1};
    auto test_case = ov::test::TestCase(model, s_device);
    std::vector<float> input_data{1.0f, 3.0f, 4.0f, 8.0f, 6.0f, 2.0f, 7.0f, 11.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(expected_output_shape, {1.0f, 1.0f, 4.0f, 4.0f, 6.0f, 6.0f, 7.0f, 7.0f});

    test_case.run_with_tolerance_as_fp();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_linear_asymmetric) {
    const auto model = convert_model("resize11_up_sizes_linear_asymmetric.onnx");

    const Shape expected_output_shape{2, 1, 4, 8};
    auto test_case = ov::test::TestCase(model, s_device);
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

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_down_sizes_cubic_half_pixel) {
    const auto model = convert_model("resize11_down_sizes_cubic_half_pixel.onnx");

    const Shape expected_output_shape{1, 1, 3, 3};
    auto test_case = ov::test::TestCase(model, s_device);
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
        {1.6307871f, 3.0046299f, 4.3784733f, 7.1261587f, 8.5f, 9.873844f, 12.621532f, 13.995373f, 15.369216f});

    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_down_sizes_linear_pytorch_half_pixel) {
    const auto model = convert_model("resize11_down_sizes_linear_pytorch_half_pixel.onnx");

    const Shape expected_output_shape{1, 1, 3, 1};
    auto test_case = ov::test::TestCase(model, s_device);
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

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_cubic_half_pixel) {
    const auto model = convert_model("resize11_up_sizes_cubic_half_pixel.onnx");

    const Shape expected_output_shape{1, 1, 9, 10};
    auto test_case = ov::test::TestCase(model, s_device);
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
        {0.45507922f,  0.64057922f,  0.97157922f,  1.42257922f,  1.90732922f,  2.22332922f,  2.70807922f,  3.15907922f,
         3.49007922f,  3.67557922f,  1.39437963f,  1.57987963f,  1.91087963f,  2.36187963f,  2.84662963f,  3.16262963f,
         3.64737963f,  4.09837963f,  4.42937963f,  4.61487963f,  2.95130693f,  3.13680693f,  3.46780693f,  3.91880693f,
         4.40355693f,  4.71955693f,  5.20430693f,  5.65530693f,  5.98630693f,  6.17180693f,  5.20525069f,  5.39075069f,
         5.72175069f,  6.17275069f,  6.65750069f,  6.97350069f,  7.45825069f,  7.90925069f,  8.24025069f,  8.42575069f,
         6.88975f,     7.07525f,     7.40625f,     7.85725f,     8.342f,       8.658f,       9.14275f,     9.59375f,
         9.92475f,     10.11025f,    8.57424931f,  8.75974931f,  9.09074931f,  9.54174931f,  10.02649931f, 10.34249931f,
         10.82724931f, 11.27824931f, 11.60924931f, 11.79474931f, 10.82819307f, 11.01369307f, 11.34469307f, 11.79569307f,
         12.28044307f, 12.59644307f, 13.08119307f, 13.53219307f, 13.86319307f, 14.04869307f, 12.38512037f, 12.57062037f,
         12.90162037f, 13.35262037f, 13.83737037f, 14.15337037f, 14.63812037f, 15.08912037f, 15.42012037f, 15.60562037f,
         13.32442078f, 13.50992078f, 13.84092078f, 14.29192078f, 14.77667078f, 15.09267078f, 15.57742078f, 16.02842078f,
         16.35942078f, 16.54492078f});
    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_cubic_half_pixel_dynamic_sizes) {
    const auto model = convert_model("resize11_up_sizes_cubic_half_pixel_dynamic_sizes.onnx");

    const Shape expected_output_shape{1, 1, 9, 10};
    auto test_case = ov::test::TestCase(model, s_device);
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
        {0.45507922f,  0.64057922f,  0.97157922f,  1.42257922f,  1.90732922f,  2.22332922f,  2.70807922f,  3.15907922f,
         3.49007922f,  3.67557922f,  1.39437963f,  1.57987963f,  1.91087963f,  2.36187963f,  2.84662963f,  3.16262963f,
         3.64737963f,  4.09837963f,  4.42937963f,  4.61487963f,  2.95130693f,  3.13680693f,  3.46780693f,  3.91880693f,
         4.40355693f,  4.71955693f,  5.20430693f,  5.65530693f,  5.98630693f,  6.17180693f,  5.20525069f,  5.39075069f,
         5.72175069f,  6.17275069f,  6.65750069f,  6.97350069f,  7.45825069f,  7.90925069f,  8.24025069f,  8.42575069f,
         6.88975f,     7.07525f,     7.40625f,     7.85725f,     8.342f,       8.658f,       9.14275f,     9.59375f,
         9.92475f,     10.11025f,    8.57424931f,  8.75974931f,  9.09074931f,  9.54174931f,  10.02649931f, 10.34249931f,
         10.82724931f, 11.27824931f, 11.60924931f, 11.79474931f, 10.82819307f, 11.01369307f, 11.34469307f, 11.79569307f,
         12.28044307f, 12.59644307f, 13.08119307f, 13.53219307f, 13.86319307f, 14.04869307f, 12.38512037f, 12.57062037f,
         12.90162037f, 13.35262037f, 13.83737037f, 14.15337037f, 14.63812037f, 15.08912037f, 15.42012037f, 15.60562037f,
         13.32442078f, 13.50992078f, 13.84092078f, 14.29192078f, 14.77667078f, 15.09267078f, 15.57742078f, 16.02842078f,
         16.35942078f, 16.54492078f});
    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_nearest_round_prefer_floor_half_pixel) {
    const auto model = convert_model("resize11_up_sizes_nearest_round_prefer_floor_half_pixel.onnx");

    const Shape expected_output_shape{1, 1, 7, 8};
    auto test_case = ov::test::TestCase(model, s_device);
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

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_nearest_prefer_ceil_asymmetric) {
    const auto model = convert_model("resize11_up_sizes_nearest_prefer_ceil_asymmetric.onnx");

    const Shape expected_output_shape{1, 1, 8, 8};
    auto test_case = ov::test::TestCase(model, s_device);
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

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_nearest_ceil_half_pixel) {
    const auto model = convert_model("resize11_up_sizes_nearest_ceil_half_pixel.onnx");

    const Shape expected_output_shape{1, 1, 8, 8};
    auto test_case = ov::test::TestCase(model, s_device);
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

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_nearest_floor_align_corners) {
    const auto model = convert_model("resize11_up_sizes_nearest_floor_align_corners.onnx");

    const Shape expected_output_shape{1, 1, 8, 8};
    auto test_case = ov::test::TestCase(model, s_device);
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

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_down_sizes_tf_half_pixel) {
    const auto model = convert_model("resize11_down_sizes_tf_half_pixel.onnx");

    const Shape expected_output_shape{1, 1, 3, 2};
    auto test_case = ov::test::TestCase(model, s_device);
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

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_shape) {
    auto model = convert_model("shape.onnx");

    Inputs inputs;
    inputs.emplace_back(
        ov::test::NDArray<float, 3>({{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                     {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                     {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
            .get_vector());

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output<int64_t>({3, 4, 5});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_shape15_end_1) {
    // shape 15 tests
    auto model = convert_model("shape_opset15_end_1.onnx");

    Inputs inputs;
    inputs.emplace_back(
        ov::test::NDArray<float, 3>({{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                     {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                     {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
            .get_vector());

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output<int64_t>({3});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_shape15_end_negative_1) {
    // shape 15 tests
    auto model = convert_model("shape_opset15_end_negative_1.onnx");

    Inputs inputs;
    inputs.emplace_back(
        ov::test::NDArray<float, 3>({{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                     {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                     {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
            .get_vector());

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output<int64_t>({3, 4});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_shape15_start_1) {
    // shape 15 tests
    auto model = convert_model("shape_opset15_start_1.onnx");

    Inputs inputs;
    inputs.emplace_back(
        ov::test::NDArray<float, 3>({{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                     {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                     {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
            .get_vector());

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output<int64_t>({4, 5});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_shape15_start_1_end_2) {
    // shape 15 tests
    auto model = convert_model("shape_opset15_start_1_end_2.onnx");

    Inputs inputs;
    inputs.emplace_back(
        ov::test::NDArray<float, 3>({{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                     {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                     {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
            .get_vector());

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output<int64_t>({4});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_shape15_start_1_end_negative_1) {
    // shape 15 tests
    auto model = convert_model("shape_opset15_start_1_end_negative_1.onnx");

    Inputs inputs;
    inputs.emplace_back(
        ov::test::NDArray<float, 3>({{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                     {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                     {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
            .get_vector());

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output<int64_t>({4});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_shape15_start_negative_1) {
    // shape 15 tests
    auto model = convert_model("shape_opset15_start_negative_1.onnx");

    Inputs inputs;
    inputs.emplace_back(
        ov::test::NDArray<float, 3>({{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                     {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                     {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
            .get_vector());

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output<int64_t>({5});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_shape15_start_negative_2) {
    // shape 15 tests
    auto model = convert_model("shape_opset15_start_negative_2.onnx");

    Inputs inputs;
    inputs.emplace_back(
        ov::test::NDArray<float, 3>({{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                     {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                     {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
            .get_vector());

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output<int64_t>({4, 5});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_shape15_end_negative_2) {
    // shape 15 tests
    auto model = convert_model("shape_opset15_end_negative_2.onnx");

    Inputs inputs;
    inputs.emplace_back(
        ov::test::NDArray<float, 3>({{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                     {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                     {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
            .get_vector());

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output<int64_t>({3});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_elu) {
    auto model = convert_model("elu.onnx");

    Inputs inputs;
    inputs.emplace_back(
        ov::test::NDArray<float, 3>({{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
                                     {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
                                     {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    auto expected_output =
        ov::test::NDArray<float, 3>(
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

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_leaky_relu) {
    auto model = convert_model("leaky_relu.onnx");

    Inputs inputs;
    inputs.emplace_back(
        ov::test::NDArray<float, 3>({{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
                                     {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
                                     {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    auto expected_output =
        ov::test::NDArray<float, 3>(
            {{{-0.9f, -0.8f, -0.7f, -0.6f, -0.5f}, {-0.4f, -0.3f, -0.2f, -0.1f, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
             {{-0.4f, -0.3f, -0.2f, -0.1f, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
             {{1, 1, 1, 1, 1}, {-0.1f, -0.1f, -0.1f, -0.1f, -0.1f}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_prelu_nd) {
    auto model = convert_model("prelu.onnx");

    Inputs inputs;
    inputs.emplace_back(
        ov::test::NDArray<float, 3>({{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
                                     {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
                                     {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    inputs.emplace_back(
        ov::test::NDArray<float, 3>({{{1, 0, 1, 0, 1}, {0, 1, 0, 1, 0}, {1, 0, 1, 0, 1}, {0, 1, 0, 1, 0}},
                                     {{0, 1, 0, 1, 0}, {1, 0, 1, 0, 1}, {0, 1, 0, 1, 0}, {1, 0, 1, 0, 1}},
                                     {{1, 0, 1, 0, 1}, {0, 1, 0, 1, 0}, {1, 0, 1, 0, 1}, {0, 1, 0, 1, 0}}})
            .get_vector());

    auto expected_output =
        ov::test::NDArray<float, 3>({{{-9, 0, -7, 0, -5}, {0, -3, 0, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
                                     {{0, -3, 0, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
                                     {{1, 1, 1, 1, 1}, {0, -1, 0, -1, 0}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_prelu_batch_nd_elementwise) {
    auto model = convert_model("prelu_batch_nd.onnx");

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
    std::iota(std::begin(slope), std::end(slope), 0.f);
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

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_prelu_1d) {
    auto model = convert_model("prelu_1d.onnx");

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

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_prelu_C_1_1) {
    auto model = convert_model("prelu_c_1_1.onnx");

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

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_selu) {
    auto model = convert_model("selu.onnx");

    Inputs inputs;
    inputs.emplace_back(
        ov::test::NDArray<float, 3>({{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
                                     {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
                                     {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    auto expected_output =
        ov::test::NDArray<float, 3>(
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

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_sigmoid) {
    auto model = convert_model("sigmoid.onnx");

    Inputs inputs;
    inputs.emplace_back(
        ov::test::NDArray<float, 3>({{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
                                     {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
                                     {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    auto expected_output =
        ov::test::NDArray<float, 3>(
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

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_tanh) {
    auto model = convert_model("tanh.onnx");

    Inputs inputs;
    inputs.emplace_back(
        ov::test::NDArray<float, 3>({{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
                                     {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
                                     {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    auto expected_output =
        ov::test::NDArray<float, 3>(
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

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_thresholded_relu) {
    auto model = convert_model("thresholded_relu.onnx");

    Inputs inputs;
    inputs.emplace_back(
        ov::test::NDArray<float, 3>({{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
                                     {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
                                     {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    auto expected_output =
        ov::test::NDArray<float, 3>({{{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 3, 4, 5}, {6, 7, 8, 9, 10}},
                                     {{0, 0, 0, 0, 0}, {0, 0, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
                                     {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}}})
            .get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_matmul_vec_ten3d) {
    auto model = convert_model("matmul_vec_ten3d.onnx");

    Inputs inputs;
    inputs.emplace_back(std::vector<float>{0.f, 1.f});
    inputs.emplace_back(ov::test::NDArray<float, 3>{{{0.f}, {1.f}}, {{2.f}, {3.f}}, {{4.f}, {5.f}}}.get_vector());

    auto expected_output = ov::test::NDArray<float, 2>{{1.f}, {3.f}, {5.f}}.get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_softplus) {
    auto model = convert_model("softplus.onnx");

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
    std::vector<float> output{0.3132616579532623291f,
                              0.6931471824645996094f,
                              1.313261628150939941f,
                              10.0000457763671875f,
                              100.0f,
                              0.0f,
                              1000.0f,
                              0.0f,
                              0.6931471824645996094f,
                              0.6931471824645996094f,
                              0.6931471824645996094f,
                              inf,
                              0.0f};

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_softplus_infinity) {
    auto model = convert_model("softplus.onnx");

    std::vector<float> input(13, std::numeric_limits<float>::infinity());
    std::vector<float> expected_output(13, std::numeric_limits<float>::infinity());

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input(input);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_sum_opset8) {
    auto model = convert_model("sum_opset8.onnx");

    Inputs inputs;
    inputs.emplace_back(std::vector<float>{1.0f, 2.0f, 3.0f});
    inputs.emplace_back(ov::test::NDArray<float, 2>{{10.0f}, {20.0f}, {30.0f}}.get_vector());
    inputs.emplace_back(ov::test::NDArray<float, 3>{{{100.0f}}, {{200.0f}}, {{300.0f}}}.get_vector());

    auto expected_output =
        ov::test::NDArray<float, 3>{{{111.0f, 112.0f, 113.0f}, {121.0f, 122.0f, 123.0f}, {131.0f, 132.0f, 133.0f}},

                                    {{211.0f, 212.0f, 213.0f}, {221.0f, 222.0f, 223.0f}, {231.0f, 232.0f, 233.0f}},

                                    {{311.0f, 312.0f, 313.0f}, {321.0f, 322.0f, 323.0f}, {331.0f, 332.0f, 333.0f}}}
            .get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_argmax_int32) {
    auto model = convert_model("argmax_int32.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<std::int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    test_case.add_expected_output<std::int64_t>({1, 1, 1, 1, 1, 1});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_argmin_int32) {
    auto model = convert_model("argmin_int32.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<std::int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    test_case.add_expected_output<std::int64_t>({0, 0, 0, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_argmax_float) {
    auto model = convert_model("argmax_float.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({4.f, 0.1f, 2.f, 3.f, -3.f, 1.f, -0.9f, 0.f, 1.f, 2.f, 3.f, 0.f});
    test_case.add_expected_output<std::int64_t>({0, 3, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_argmin_float) {
    auto model = convert_model("argmin_float.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({4.f, 0.1f, 2.f, 3.f, -3.f, 1.f, -0.9f, 0.f, 1.f, 2.f, 3.f, 0.f});
    test_case.add_expected_output<std::int64_t>({1, 1, 0, 2});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_argmax_select_last_index) {
    auto model = convert_model("argmax_select_last_index.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{4, 3}, {1.f, 1.f, 1.f, 0.5f, 3.f, 4.f, 0.5f, 1.f, 1.1f, 0.f, 3.f, 0.f});
    test_case.add_expected_output<std::int64_t>(Shape{1, 3}, {0, 3, 1});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_argmin_select_last_index) {
    auto model = convert_model("argmin_select_last_index.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{4, 3}, {1.f, 1.f, 1.f, 2.f, 3.f, 4.f, 2.f, 1.f, 1.1f, 3.f, 3.f, 8.f});
    test_case.add_expected_output<std::int64_t>(Shape{4}, {2, 0, 1, 1});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_top_k) {
    auto model = convert_model("top_k.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    test_case.add_expected_output<float>(Shape{3, 3}, {3, 2, 1, 7, 6, 5, 11, 10, 9});       // values
    test_case.add_expected_output<std::int64_t>(Shape{3, 3}, {3, 2, 1, 3, 2, 1, 3, 2, 1});  // indices
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_top_k_opset_10) {
    auto model = convert_model("top_k_opset_10.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    test_case.add_input<int64_t>({3});

    test_case.add_expected_output<float>(Shape{3, 3}, {3, 2, 1, 7, 6, 5, 11, 10, 9});       // values
    test_case.add_expected_output<std::int64_t>(Shape{3, 3}, {3, 2, 1, 3, 2, 1, 3, 2, 1});  // indices
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_top_k_opset_10_const_k) {
    auto model = convert_model("top_k_opset_10_const_k.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

    test_case.add_expected_output<float>(Shape{3, 3}, {3, 2, 1, 7, 6, 5, 11, 10, 9});       // values
    test_case.add_expected_output<std::int64_t>(Shape{3, 3}, {3, 2, 1, 3, 2, 1, 3, 2, 1});  // indices
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_top_k_opset_11_const_k_smallest) {
    auto model = convert_model("top_k_opset_11_const_k_smallest.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8});

    test_case.add_expected_output<float>(Shape{3, 3}, {0, 1, 2, 4, 5, 6, 8, 9, 10});        // values
    test_case.add_expected_output<std::int64_t>(Shape{3, 3}, {0, 1, 2, 0, 1, 2, 3, 2, 1});  // indices
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_top_k_opset_11_const_k_smallest_negative_axis) {
    auto model = convert_model("top_k_opset_11_const_k_smallest_negative_axis.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8});

    test_case.add_expected_output<float>(Shape{3, 3}, {0, 1, 2, 4, 5, 6, 8, 9, 10});        // values
    test_case.add_expected_output<std::int64_t>(Shape{3, 3}, {0, 1, 2, 0, 1, 2, 3, 2, 1});  // indices
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_top_k_repeating_1D) {
    auto model = convert_model("top_k_repeating_1D.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<int32_t>({1, 1, 2, 0, 2, 100});
    test_case.add_input<int64_t>({5});

    test_case.add_expected_output<int32_t>(Shape{5}, {100, 2, 2, 1, 1});
    test_case.add_expected_output<int64_t>(Shape{5}, {5, 2, 4, 0, 1});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_top_k_repeating) {
    auto model = convert_model("top_k_repeating.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<int32_t>(Shape{3, 6}, {100, 1, 1, 2, 0, 2, 1, 2, 3, 4, 5, 6, 100, 1, 1, 2, 0, 2});
    test_case.add_input<int64_t>({3});

    test_case.add_expected_output<int32_t>(Shape{3, 3}, {100, 2, 2, 6, 5, 4, 7, 2, 2});
    test_case.add_expected_output<int64_t>(Shape{3, 3}, {0, 3, 5, 5, 4, 3, 0, 2, 4});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_top_k_repeating_axis_0) {
    auto model = convert_model("top_k_repeating_axis_0.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<int32_t>(Shape{3, 6}, {100, 1, 1, 2, 0, 2, 1, 2, 3, 4, 5, 6, 7, 1, 2, 0, 2, 1});
    test_case.add_input<int64_t>({2});

    test_case.add_expected_output<int32_t>(Shape{2, 6}, {100, 2, 3, 4, 5, 6, 7, 1, 2, 2, 2, 2});
    test_case.add_expected_output<int64_t>(Shape{2, 6}, {0, 1, 1, 1, 1, 1, 2, 0, 2, 0, 2, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_top_k_repeating_unsorted) {
    auto model = convert_model("top_k_repeating_unsorted.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<int32_t>(Shape{3, 6}, {100, 1, 1, 2, 0, 2, 1, 2, 3, 4, 5, 6, 7, 1, 2, 0, 2, 1});
    test_case.add_input<int64_t>({3});

    test_case.add_expected_output<int32_t>(Shape{3, 3}, {1, 1, 0, 3, 2, 1, 1, 1, 0});
    test_case.add_expected_output<int64_t>(Shape{3, 3}, {2, 1, 4, 2, 1, 0, 5, 1, 3});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_acosh) {
    auto model = convert_model("acosh.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{1, 3}, {1.0f, 2.5f, 4.3f});
    test_case.add_expected_output<float>(Shape{1, 3}, {0.0f, 1.5667993f, 2.13795861f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_asinh) {
    auto model = convert_model("asinh.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{1, 3}, {-1.0f, 0.0f, 1.0f});
    test_case.add_expected_output<float>(Shape{1, 3}, {-0.88137358f, 0.0f, 0.88137358f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_atanh) {
    auto model = convert_model("atanh.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{1, 3}, {-0.9f, 0.0f, 0.9f});
    test_case.add_expected_output<float>(Shape{1, 3}, {-1.4722194f, 0.0f, 1.4722194f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_sinh) {
    auto model = convert_model("sinh.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({-1.0f, 0.0f, 1.0f});
    test_case.add_expected_output<float>({-1.1752012f, 0.f, 1.1752012f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_cosh) {
    auto model = convert_model("cosh.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({-1.0f, 0.0f, 1.0f});
    test_case.add_expected_output<float>({1.54308069f, 1.f, 1.54308069f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_sign) {
    auto model = convert_model("sign.onnx");

    Inputs inputs{std::vector<float>{-std::numeric_limits<float>::infinity(),
                                     -3.141592f,
                                     0.0f,
                                     2.71828f,
                                     std::numeric_limits<float>::infinity()}};

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output<float>({-1.0f, -1.0f, 0.0f, 1.0f, 1.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_one_hot_with_axis) {
    auto model = convert_model("one_hot_axis.onnx");

    Inputs inputs{{1.0f, 9.0f, 2.0f, 4.0f}, {1.0f, 3.0f}};
    std::vector<float> expected_output{{1.0f, 1.0f, 3.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 3.0f,
                                        1.0f, 1.0f, 1.0f, 1.0f, 3.0f, 1.0f, 1.0f, 1.0f, 1.0f, 3.0f,
                                        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}};

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_one_hot_without_axis) {
    auto model = convert_model("one_hot_no_axis.onnx");

    std::vector<std::vector<std::int64_t>> inputs{{0, 7, 8}, {2, 5}};
    std::vector<std::int64_t> expected_output{5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                              2, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 2, 2, 2};

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_where) {
    auto model = convert_model("where.onnx");

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

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_erf) {
    const auto model = convert_model("erf.onnx");

    Inputs inputs;
    inputs.emplace_back(ov::test::NDArray<float, 2>{
        {-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()},
        {-3.141592f, 0.0f},
        {0.5f, 1.0f}}.get_vector());

    const std::vector<float> expected_output =
        ov::test::NDArray<float, 2>{{-1.0f, 1.0f}, {-0.99999112f, 0.0f}, {0.52049988f, 0.84270079f}}.get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_erf_int32) {
    const auto model = convert_model("erf_int32.onnx");

    const std::vector<std::vector<int32_t>> inputs{
        {-std::numeric_limits<int32_t>::max(), -1, 0, 1, std::numeric_limits<int32_t>::max()}};

    const std::vector<int32_t> expected_output{-1, -1, 0, 1, 1};

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_shrink_float) {
    const auto model = convert_model("shrink_float.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({-2.0f, -1.6f, -1.5f, -1.4f, -1.0f, 0.0f, 1.0f, 1.4f, 1.5f, 1.6f, 2.0f});
    test_case.add_expected_output<float>(Shape{11},
                                         {-1.5f, -1.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.1f, 1.5f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_shrink_int) {
    const auto model = convert_model("shrink_int.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<int>({-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5});
    test_case.add_expected_output<int>(Shape{11}, {-4, -3, -2, -1, 0, 0, 0, 1, 2, 3, 4});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_lp_norm_p1) {
    const auto model = convert_model("lp_norm_p1.onnx");

    Shape data_shape{2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    std::iota(std::begin(data), std::end(data), 1.f);

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(
        data_shape,
        {0.07142857f, 0.125f,  0.16666667f, 0.2f,        0.22727273f, 0.25f,   0.26923078f, 0.2857143f,
         0.3f,        0.3125f, 0.32352942f, 0.33333334f, 0.9285714f,  0.875f,  0.8333333f,  0.8f,
         0.77272725f, 0.75f,   0.7307692f,  0.71428573f, 0.7f,        0.6875f, 0.6764706f,  0.6666667f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_lp_norm_p2) {
    const auto model = convert_model("lp_norm_p2.onnx");

    Shape data_shape{2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    std::iota(std::begin(data), std::end(data), 1.f);

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(
        data_shape,
        {0.0766965f,  0.14142136f, 0.19611613f, 0.24253564f, 0.28216633f, 0.31622776f, 0.34570536f, 0.37139067f,
         0.39391932f, 0.41380295f, 0.4314555f,  0.4472136f,  0.9970545f,  0.98994946f, 0.9805807f,  0.97014254f,
         0.9593655f,  0.9486833f,  0.9383431f,  0.9284767f,  0.91914505f, 0.9103665f,  0.9021342f,  0.8944272f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_lp_norm_default) {
    const auto model = convert_model("lp_norm_default.onnx");

    Shape data_shape{2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    std::iota(std::begin(data), std::end(data), 1.f);

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(
        data_shape,
        {0.18257418f, 0.36514837f, 0.5477225f, 0.73029673f, 0.37904903f, 0.45485884f, 0.5306686f,  0.60647845f,
         0.42616236f, 0.47351375f, 0.5208651f, 0.5682165f,  0.4469492f,  0.48132992f, 0.51571065f, 0.5500913f,
         0.45862272f, 0.48560053f, 0.5125783f, 0.53955615f, 0.46609157f, 0.4882864f,  0.51048124f, 0.5326761f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_lp_norm_default_dynamic) {
    const auto model = convert_model("lp_norm_default_dynamic.onnx");

    Shape data_shape{2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    std::iota(std::begin(data), std::end(data), 1.f);

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(data_shape, data);
    test_case.add_expected_output<float>(
        data_shape,
        {0.18257418f, 0.36514837f, 0.5477225f, 0.73029673f, 0.37904903f, 0.45485884f, 0.5306686f,  0.60647845f,
         0.42616236f, 0.47351375f, 0.5208651f, 0.5682165f,  0.4469492f,  0.48132992f, 0.51571065f, 0.5500913f,
         0.45862272f, 0.48560053f, 0.5125783f, 0.53955615f, 0.46609157f, 0.4882864f,  0.51048124f, 0.5326761f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_instance_normalization) {
    const auto model = convert_model("instance_norm.onnx");

    Shape data_shape{1, 2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    std::iota(std::begin(data), std::end(data), 1.f);

    auto test_case = ov::test::TestCase(model, s_device);

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

OPENVINO_TEST(${BACKEND_NAME}, onnx_instance_normalization_dynamic) {
    auto model = convert_model("instance_norm_dynamic.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    std::vector<float> input_data{1.f, 2.f, 3.f};
    test_case.add_input<float>(Shape{1, 3, 1, 1}, input_data);
    test_case.add_expected_output<float>(Shape{1, 3, 1, 1},
                                         {0.3341970741748809814f, 0.3321160078048706055f, 0.3407136797904968262f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_eye_like) {
    const auto model = convert_model("eye_like.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{3, 4}, {5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f});
    test_case.add_expected_output<float>(Shape{3, 4}, {0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reverse_sequence_0_batch_1) {
    const auto model = convert_model("reverse_sequence_time_0_batch_1.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({0.f, 4.f, 8.f, 12.f, 1.f, 5.f, 9.f, 13.f, 2.f, 6.f, 10.f, 14.f, 3.f, 7.f, 11.f, 15.f});
    test_case.add_input<int>({4, 3, 2, 1});
    test_case.add_expected_output<float>(
        Shape{4, 4},
        {3.f, 6.f, 9.f, 12.f, 2.f, 5.f, 8.f, 13.f, 1.f, 4.f, 10.f, 14.f, 0.f, 7.f, 11.f, 15.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reverse_sequence_1_batch_0) {
    const auto model = convert_model("reverse_sequence_time_1_batch_0.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f});
    test_case.add_input<int>({1, 2, 3, 4});
    test_case.add_expected_output<float>(
        Shape{4, 4},
        {0.f, 1.f, 2.f, 3.f, 5.f, 4.f, 6.f, 7.f, 10.f, 9.f, 8.f, 11.f, 15.f, 14.f, 13.f, 12.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reverse_sequence_incorrect_batch_axis) {
    EXPECT_THROW(convert_model("reverse_sequence_incorrect_batch_axis.onnx"), ov::Exception)
        << "ReverseSequence batch_axis attribute can only equal 0 or 1. Value of '2' is not "
           "accepted.";
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reverse_sequence_incorrect_time_axis) {
    EXPECT_THROW(convert_model("reverse_sequence_incorrect_time_axis.onnx"), ov::Exception)
        << "ReverseSequence time_axis attribute can only equal 0 or 1. Value of '2' is not "
           "accepted.";
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reverse_sequence_time_and_batch_axis_equal) {
    EXPECT_THROW(convert_model("reverse_sequence_time_and_batch_axis_equal.onnx"), ov::Exception)
        << "ReverseSequence 'time_axis' and 'batch_axis' can't be equal.";
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_matmul_float_type) {
    auto model = convert_model("matmul_float.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(std::vector<float>{0, 1, 2, 3, 4, 5});
    test_case.add_input<float>(std::vector<float>{0, 1});
    test_case.add_expected_output<float>(Shape{3, 1}, std::vector<float>{1, 3, 5});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mod_sign) {
    const auto model = convert_model("mod_sign.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<int32_t>({-4, 7, 5, 4, -7, 8});
    test_case.add_input<int32_t>({2, -3, 8, -2, 3, 5});
    test_case.add_expected_output<int32_t>(Shape{6}, {0, -2, 5, 0, 2, 3});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mod_sign_i64) {
    const auto model = convert_model("mod_sign_i64.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<int64_t>({-4, 7, 5, 4, -7, 8});
    test_case.add_input<int64_t>({2, -3, 8, -2, 3, 5});
    test_case.add_expected_output<int64_t>(Shape{6}, {0, -2, 5, 0, 2, 3});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mod_sign_broadcast) {
    const auto model = convert_model("mod_sign_broadcast.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<int32_t>({-8, 3, 4, 9, -17, 1});
    test_case.add_input<int32_t>({3});
    test_case.add_expected_output<int32_t>(Shape{6}, {1, 0, 1, 0, 1, 1});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mod_sign_f32) {
    try {
        const auto model = convert_model("mod_sign_f32.onnx");
        FAIL() << "Expected exception was not thrown";
    } catch (const ov::Exception& e) {
        EXPECT_HAS_SUBSTRING(
            e.what(),
            std::string("If the input type is floating point, then `fmod` attribute must be set to 1."));
    } catch (...) {
        FAIL() << "Expected ov::Exception was not thrown";
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mod_sign_fmod) {
    const auto model = convert_model("mod_sign_fmod.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<int32_t>({-8, 3, 4, 9, -17, 1});
    test_case.add_input<int32_t>({22, -13, 8, -3, 7, 2});
    test_case.add_expected_output<int32_t>(Shape{6}, {-8, 3, 4, 0, -3, 1});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mod_sign_fmod_broadcast) {
    const auto model = convert_model("mod_sign_fmod_broadcast.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<int32_t>({-8, 3, 4, 9, -17, 1});
    test_case.add_input<int32_t>({3});
    test_case.add_expected_output<int32_t>(Shape{6}, {-2, 0, 1, 0, -2, 1});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mod_sign_fmod_f32) {
    const auto model = convert_model("mod_sign_fmod_f32.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({-4.3f, 7.2f, 5.0f, 4.3f, -7.2f, 8.0f});
    test_case.add_input<float>({2.1f, -3.4f, 8.0f, -2.1f, 3.4f, 5.0f});
    test_case.add_expected_output<float>(Shape{6}, {-0.10000038f, 0.39999962f, 5.f, 0.10000038f, -0.39999962f, 3.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mod_incorrect_fmod) {
    try {
        const auto model = convert_model("mod_incorrect_fmod.onnx");
        FAIL() << "Expected exception was not thrown";
    } catch (const ov::Exception& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string("Unsupported value of 'fmod' attribute (should be: 0 or 1)"));
    } catch (...) {
        FAIL() << "Expected ov::Exception was not thrown";
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatterND_param_i64_indices) {
    const auto model = convert_model("scatter_nd_param_i64_indices.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    test_case.add_input<int64_t>({4, 3, 1, 7});
    test_case.add_input<float>({9.f, 10.f, 11.f, 12.f});
    test_case.add_expected_output<float>(Shape{8}, {1.f, 11.f, 3.f, 10.f, 9.f, 6.f, 7.f, 12.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatterND_const_i32_indices) {
    const auto model = convert_model("scatter_nd_const_i32_indices.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    test_case.add_input<float>({9.f, 10.f, 11.f, 12.f});
    test_case.add_expected_output<float>(Shape{8}, {1.f, 11.f, 3.f, 10.f, 9.f, 6.f, 7.f, 12.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatterND_opset16_reduction_none) {
    const auto model = convert_model("scatter_nd_opset16_reduction_none.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    test_case.add_input<int64_t>({4, 3, 1, 7});
    test_case.add_input<float>({9.f, 10.f, 11.f, 12.f});
    test_case.add_expected_output<float>(Shape{8}, {1.f, 11.f, 3.f, 10.f, 9.f, 6.f, 7.f, 12.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatterND_opset16_reduction_add) {
    EXPECT_THROW(convert_model("scatter_nd_opset16_reduction_add.onnx"), ov::Exception)
        << "Unsupported type of attribute: `reduction`. Only `none` is supported";
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_gather_float_1D) {
    const auto model = convert_model("gather_float_1D.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

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

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_gather_int32_3D_axis_1) {
    const auto model = convert_model("gather_int32_3D_axis_1.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

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

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_gather_int8_3D_axis_neg_1) {
    const auto model = convert_model("gather_int8_3D_axis_neg_1.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

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

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_gather_float_2D_neg_indices) {
    const auto model = convert_model("gather_float_2D_axis_1.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    // clang-format off
    test_case.add_input<float>(Shape{3, 3},
        {   0.0f, 0.1f, 0.2f,
            1.0f, 1.1f, 1.2f,
            2.0f, 2.1f, 2.2f   });
    test_case.add_input<int64_t>(Shape{2, 2},
        {   -1, -2,
            -3, -2      });
    test_case.add_expected_output<float>(Shape{3, 2, 2},
        {
            0.2f, 0.1f,
            0.0f, 0.1f,

            1.2f, 1.1f,
            1.0f, 1.1f,

            2.2f, 2.1f,
            2.0f, 2.1f    });
    // clang-format on

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_gather_elements_float_1D) {
    const auto model = convert_model("gather_elements_float_1D.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(Shape{3}, {1, 2, 3});
    test_case.add_input<int64_t>(Shape{1}, {1});
    test_case.add_expected_output<float>(Shape{1}, {2});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_gather_elements_int8_axis_1) {
    const auto model = convert_model("gather_elements_int8_axis_1.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<int8_t>(Shape{2, 2}, {1, 2, 3, 4});
    test_case.add_input<int32_t>(Shape{2, 2}, {0, 0, 1, 0});
    test_case.add_expected_output<int8_t>(Shape{2, 2}, {1, 1, 4, 3});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_gather_elements_int32_axis_0) {
    const auto model = convert_model("gather_elements_int32_axis_0.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<int32_t>(Shape{3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    test_case.add_input<int64_t>(Shape{2, 3}, {1, 2, 0, 2, 0, 0});
    test_case.add_expected_output<int32_t>(Shape{2, 3}, {4, 8, 3, 7, 2, 3});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_gather_elements_float_negative_axis) {
    const auto model = convert_model("gather_elements_float_negative_axis.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(Shape{2, 2}, {1, 2, 3, 4});
    test_case.add_input<int64_t>(Shape{2, 2}, {1, 1, 1, 0});
    test_case.add_expected_output<float>(Shape{2, 2}, {2, 2, 4, 3});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_gather_elements_float_3D_axis_2) {
    const auto model = convert_model("gather_elements_float_3D_axis_2.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(Shape{2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    test_case.add_input<int64_t>(Shape{2, 2, 1}, {0, 1, 0, 1});
    test_case.add_expected_output<float>(Shape{2, 2, 1}, {1, 4, 5, 8});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_gatherND_int32) {
    const auto model = convert_model("gatherND_int32.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<int32_t>({0, 1, 2, 3});
    test_case.add_input<int64_t>({1, 0});
    test_case.add_expected_output<int32_t>(Shape{2, 2}, {2, 3, 0, 1});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_gatherND_float) {
    const auto model = convert_model("gatherND_float.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f});
    test_case.add_input<int64_t>({0, 1, 1, 0});
    test_case.add_expected_output<float>(Shape{2, 2}, {2.f, 3.f, 4.f, 5.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_pad_constant) {
    const auto model = convert_model("pad_constant.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({1.f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f});
    test_case.add_expected_output<float>(Shape{3, 4},
                                         {0.f, 0.f, 1.f, 1.2f, 0.f, 0.f, 2.3f, 3.4f, 0.f, 0.f, 4.5f, 5.7f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_pad_non_scalar_values) {
    const auto model = convert_model("pad_non_scalar_values.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({1.f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f});
    test_case.add_expected_output<float>(Shape{3, 4},
                                         {44.f, 44.f, 1.f, 1.2f, 44.f, 44.f, 2.3f, 3.4f, 44.f, 44.f, 4.5f, 5.7f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_pad_optional_constant) {
    const auto model = convert_model("pad_optional_constant.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({1.f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f});
    test_case.add_expected_output<float>(Shape{3, 4},
                                         {0.f, 0.f, 1.f, 1.2f, 0.f, 0.f, 2.3f, 3.4f, 0.f, 0.f, 4.5f, 5.7f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_pad_constant_negative_begin_end) {
    const auto model = convert_model("pad_negative_begin_end.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    test_case.add_input<int64_t>({-1, -1, -1, -1});
    test_case.add_expected_output<int32_t>(Shape{1, 2}, {6, 7});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_pow_float32_float32) {
    const auto model = convert_model("pow_float32_float32.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});  // base
    test_case.add_input<float>({3.5f});                // exponent

    test_case.add_expected_output<float>(Shape{1, 4}, {1.f, 11.313708f, 46.765373f, 128.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_pow_float32_int32) {
    const auto model = convert_model("pow_float32_int32.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});  // base
    test_case.add_input<int>({3});                     // exponent

    test_case.add_expected_output<float>(Shape{1, 4}, {1.f, 8.f, 27.f, 64.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_pow_int32_float32) {
    const auto model = convert_model("pow_int32_float32.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<int>({1, 2, 3, 4});  // base
    test_case.add_input<float>({3.5f});      // exponent

    test_case.add_expected_output<int>(Shape{1, 4}, {1, 11, 46, 128});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reciprocal) {
    const auto model = convert_model("reciprocal.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    test_case.add_expected_output<float>(Shape{3, 2}, {1.f, 1 / 2.f, 1 / 3.f, 1 / 4.f, 1 / 5.f, 1 / 6.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_round) {
    const auto model = convert_model("round.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({0.1f, 0.9f, 1.2f, 1.5f, 1.8f, 2.3f, 2.7f, -1.1f, -1.9f, -2.2f, -2.8f});
    test_case.add_expected_output<float>({0.f, 1.f, 1.f, 2.f, 2.f, 2.f, 3.f, -1.f, -2.f, -2.f, -3.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_round_half_nearest_even) {
    const auto model = convert_model("round_half_nearest_even.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({0.5f, 2.5f, -1.5f, -2.5f});
    test_case.add_expected_output<float>({0.f, 2.f, -2.f, -2.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatter10) {
    const auto scatter_fn = convert_model("scatter_opset10.onnx");

    const Shape data_shape{2, 2};

    EXPECT_EQ(scatter_fn->get_output_size(), 1);
    EXPECT_EQ(scatter_fn->get_output_shape(0), data_shape);
    EXPECT_EQ(count_ops_of_type<op::v12::ScatterElementsUpdate>(scatter_fn), 1);
    EXPECT_EQ(count_ops_of_type<op::v0::Constant>(scatter_fn), 4);

    auto test_case = ov::test::TestCase(scatter_fn, s_device);
    test_case.add_expected_output<float>({12.01f, 3.f, 4.f, 13.99f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatter_elements_opset11) {
    const auto scatter_fn = convert_model("scatter_elements_opset11.onnx");

    const Shape data_shape{1, 5};

    EXPECT_EQ(scatter_fn->get_output_size(), 1);
    EXPECT_EQ(scatter_fn->get_output_shape(0), data_shape);
    EXPECT_EQ(count_ops_of_type<op::v12::ScatterElementsUpdate>(scatter_fn), 1);
    EXPECT_EQ(count_ops_of_type<op::v0::Constant>(scatter_fn), 4);

    auto test_case = ov::test::TestCase(scatter_fn, s_device);
    test_case.add_expected_output<float>({1.f, 1.1f, 3.f, 2.1f, 5.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatter_elements_opset16_reduction_none) {
    const auto scatter_fn = convert_model("scatter_elements_opset16_reduction_none.onnx");

    const Shape data_shape{1, 5};

    EXPECT_EQ(scatter_fn->get_output_size(), 1);
    EXPECT_EQ(scatter_fn->get_output_shape(0), data_shape);
    EXPECT_EQ(count_ops_of_type<op::v12::ScatterElementsUpdate>(scatter_fn), 1);
    EXPECT_EQ(count_ops_of_type<op::v0::Constant>(scatter_fn), 4);

    auto test_case = ov::test::TestCase(scatter_fn, s_device);
    test_case.add_expected_output<float>({1.f, 1.1f, 3.f, 2.1f, 5.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatter_elements_opset16_reduction_add) {
    const auto scatter_fn = convert_model("scatter_elements_opset16_reduction_add.onnx");

    auto test_case = ov::test::TestCase(scatter_fn, s_device);
    test_case.add_expected_output<float>({1.f, 3.1f, 3.f, 6.1f, 5.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatter_elements_default_opset18) {
    const auto scatter_fn = convert_model("scatter_elements_default_opset18.onnx");

    auto test_case = ov::test::TestCase(scatter_fn, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f});  // Shape: (2, 5)
    test_case.add_input<int64_t>({1, 3, 2, 4});                                       // Shape: (2, 2)
    test_case.add_input<float>({11.f, 12.f, 13.f, 14.f});                             // Shape: (2, 2)
    test_case.add_expected_output<float>({1.f, 11.f, 3.f, 12.f, 5.f, 6.f, 7.f, 13.f, 9.f, 14.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatter_elements_none_opset18) {
    const auto scatter_fn = convert_model("scatter_elements_none_opset18.onnx");

    auto test_case = ov::test::TestCase(scatter_fn, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f});  // Shape: (2, 5)
    test_case.add_input<int64_t>({1, 3, 2, 4});                                       // Shape: (2, 2)
    test_case.add_input<float>({11.f, 12.f, 13.f, 14.f});                             // Shape: (2, 2)
    test_case.add_expected_output<float>({1.f, 11.f, 3.f, 12.f, 5.f, 6.f, 7.f, 13.f, 9.f, 14.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatter_elements_none_neg_ind_opset18) {
    const auto scatter_fn = convert_model("scatter_elements_none_opset18.onnx");

    auto test_case = ov::test::TestCase(scatter_fn, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f});               // Shape: (2, 5)
    test_case.add_input<int64_t>({-4, -2, -3, -1});                                                // Shape: (2, 2)
    test_case.add_input<float>({11.f, 12.f, 13.f, 14.f});                                          // Shape: (2, 2)
    test_case.add_expected_output<float>({1.f, 11.f, 3.f, 12.f, 5.f, 6.f, 7.f, 13.f, 9.f, 14.f});  // Shape: (2, 5)
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatter_elements_add_opset18) {
    const auto scatter_fn = convert_model("scatter_elements_add_opset18.onnx");

    auto test_case = ov::test::TestCase(scatter_fn, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f});  // Shape: (2, 5)
    test_case.add_input<int64_t>({1, 3, 2, 4});                                       // Shape: (2, 2)
    test_case.add_input<float>({11.f, 12.f, 13.f, 14.f});                             // Shape: (2, 2)
    test_case.add_expected_output<float>({1.f, 13.f, 3.f, 16.f, 5.f, 6.f, 7.f, 21.f, 9.f, 24.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatter_elements_add_neg_ind_opset18) {
    const auto scatter_fn = convert_model("scatter_elements_add_opset18.onnx");

    auto test_case = ov::test::TestCase(scatter_fn, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f});               // Shape: (2, 5)
    test_case.add_input<int64_t>({-4, -2, -3, -1});                                                // Shape: (2, 2)
    test_case.add_input<float>({11.f, 12.f, 13.f, 14.f});                                          // Shape: (2, 2)
    test_case.add_expected_output<float>({1.f, 13.f, 3.f, 16.f, 5.f, 6.f, 7.f, 21.f, 9.f, 24.f});  // Shape: (2, 5)
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatter_elements_mul_opset18) {
    const auto scatter_fn = convert_model("scatter_elements_mul_opset18.onnx");

    auto test_case = ov::test::TestCase(scatter_fn, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f});  // Shape: (2, 5)
    test_case.add_input<int64_t>({1, 3, 2, 4});                                       // Shape: (2, 2)
    test_case.add_input<float>({11.f, 12.f, 13.f, 14.f});                             // Shape: (2, 2)
    test_case.add_expected_output<float>({1.f, 22.f, 3.f, 48.f, 5.f, 6.f, 7.f, 104.f, 9.f, 140.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatter_elements_min_opset18) {
    const auto scatter_fn = convert_model("scatter_elements_min_opset18.onnx");

    auto test_case = ov::test::TestCase(scatter_fn, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f});  // Shape: (2, 5)
    test_case.add_input<int64_t>({1, 3, 2, 4});                                       // Shape: (2, 2)
    test_case.add_input<float>({0.f, 100.f, -1.f, 200.f});                            // Shape: (2, 2)
    test_case.add_expected_output<float>({1.f, 0.f, 3.f, 4.f, 5.f, 6.f, 7.f, -1.f, 9.f, 10.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatter_elements_max_opset18) {
    const auto scatter_fn = convert_model("scatter_elements_max_opset18.onnx");

    auto test_case = ov::test::TestCase(scatter_fn, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f});  // Shape: (2, 5)
    test_case.add_input<int64_t>({1, 3, 2, 4});                                       // Shape: (2, 2)
    test_case.add_input<float>({0.f, 100.f, -1.f, 200.f});                            // Shape: (2, 2)
    test_case.add_expected_output<float>({1.f, 2.f, 3.f, 100.f, 5.f, 6.f, 7.f, 8.f, 9.f, 200.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_upsample6_nearest_infer) {
    // clang-format off
    const auto model = convert_model("upsample6_nearest.onnx");
    // height_scale: 2.0
    // width_scale: 3.0
    // mode: nearest
    const Shape input_shape          {1, 1, 2, 2};
    const Shape expected_output_shape{1, 1, 4, 6};

    auto test_case = ov::test::TestCase(model, s_device);
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

OPENVINO_TEST(${BACKEND_NAME}, onnx_upsample6_bilinear_infer) {
    // clang-format off
    const auto model = convert_model("upsample6_bilinear.onnx");
    // height_scale: 2.0
    // width_scale: 3.0
    // mode: bilinear
    const Shape input_shape          {1, 1, 2, 2};
    const Shape expected_output_shape{1, 1, 4, 6};

    auto test_case = ov::test::TestCase(model, s_device);
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

OPENVINO_TEST(${BACKEND_NAME}, onnx_upsample6_dynamic) {
    // clang-format off
    const auto model = convert_model("upsample6_dynamic.onnx");
    // height_scale: 1.5
    // width_scale: 2.5
    // mode: nearest
    //
    //  X ───╤══> Reshape ──R──> Upsample ──> Y
    //  S ───┘

    auto test_case = ov::test::TestCase(model, s_device);

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

OPENVINO_TEST(${BACKEND_NAME}, onnx_upsample8_nearest_infer) {
    const auto model = convert_model("upsample8_nearest.onnx");

    // Input data shape (1, 1, 2, 2)
    // Scales attribute values {1.0, 1.0, 2.0, 3.0}
    // mode: nearest

    const Shape expected_output_shape{1, 1, 4, 6};
    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({1.0f, 2.0f, 3.0f, 4.0f});
    test_case.add_expected_output<float>(expected_output_shape,
                                         {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,
                                          3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_upsample8_linear_infer) {
    const auto model = convert_model("upsample8_linear.onnx");

    // Input data shape (1, 1, 2, 2)
    // Scales attribute values {1.0, 1.0, 2.0, 2.0}
    // mode: linear

    const Shape expected_output_shape{1, 1, 4, 4};
    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({1.0f, 2.0f, 3.0f, 4.0f});
    test_case.add_expected_output<float>(
        expected_output_shape,
        {1.0f, 1.5f, 2.0f, 2.0f, 2.0f, 2.5f, 3.0f, 3.0f, 3.0f, 3.5f, 4.0f, 4.0f, 3.0f, 3.5f, 4.0f, 4.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_upsample9_scales_const_nearest_infer) {
    const auto model = convert_model("upsample9_scales_const_nearest.onnx");

    // Input data shape (1, 1, 2, 2)
    // Input const scales values {1.0, 1.0, 2.0, 3.0}
    // mode: nearest

    const Shape expected_output_shape{1, 1, 4, 6};
    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({1.0f, 2.0f, 3.0f, 4.0f});
    test_case.add_expected_output<float>(expected_output_shape,
                                         {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,
                                          3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_upsample9_scales_const_linear_infer) {
    const auto model = convert_model("upsample9_scales_const_linear.onnx");

    // Input data shape (1, 1, 2, 2)
    // Input const scales values {1.0, 1.0, 2.0, 2.0}
    // mode: linear

    const Shape expected_output_shape{1, 1, 4, 4};
    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({1.0f, 2.0f, 3.0f, 4.0f});
    test_case.add_expected_output<float>(
        expected_output_shape,
        {1.0f, 1.5f, 2.0f, 2.0f, 2.0f, 2.5f, 3.0f, 3.0f, 3.0f, 3.5f, 4.0f, 4.0f, 3.0f, 3.5f, 4.0f, 4.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_image_scaler) {
    const auto model = convert_model("image_scaler.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({1.0f, 2.0f, 3.0f, 4.0f, 10.0f, 20.0f, 30.0f, 40.0f});
    test_case.add_expected_output<float>(Shape{1, 2, 2, 2}, {12.0f, 14.0f, 16.0f, 18.0f, 21.0f, 41.0f, 61.0f, 81.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_size_op_single) {
    const auto model = convert_model("size_op_single.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    test_case.add_expected_output<int64_t>(Shape{}, {6});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_size_op_graph_end) {
    const auto model = convert_model("size_op_graph_end.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({1.0f, 2.0f, 3.0f, 4.0f});
    test_case.add_expected_output<int64_t>(Shape{}, {4});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_size_op_graph_middle) {
    const auto model = convert_model("size_op_graph_middle.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({1.0f, 2.0f, 3.0f, 4.0f});
    test_case.add_expected_output<float>(Shape{}, {4.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_size_op_on_input_graph_middle) {
    const auto model = convert_model("size_op_on_input_graph_middle.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{1, 2, 4, 1, 3}, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                                                      0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f});
    test_case.add_expected_output<float>(Shape{1, 2, 4, 1, 3},
                                         {24.f, 24.f, 24.f, 24.f, 24.f, 24.f, 24.f, 24.f, 24.f, 24.f, 24.f, 24.f,
                                          24.f, 24.f, 24.f, 24.f, 24.f, 24.f, 24.f, 24.f, 24.f, 24.f, 24.f, 24.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_empty_initializers_handling) {
    // int this test the "scales" input of the Resize operator is set to an empty initializer
    // this input should be ignored since the "sizes" optional input is provided
    // and the inference should use the data from the latter
    const auto model = convert_model("empty_initializers_handling.onnx");

    const Shape expected_output_shape{2, 1, 4, 8};
    auto test_case = ov::test::TestCase(model, s_device);
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

OPENVINO_TEST(${BACKEND_NAME}, onnx_roi_align_f32) {
    const auto model = convert_model("roi_align_f32.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
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

OPENVINO_TEST(${BACKEND_NAME}, onnx_roialign16_avg_out_half_pixel) {
    const auto model = convert_model("roialign16_avg_out_half_pixel.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(
        {1.1f,   2.2f,   3.3f,   4.4f,   5.5f,   6.6f,   7.7f,   8.8f,   9.9f,   11.f,   12.1f,  13.2f,  14.3f,  15.4f,
         16.5f,  17.6f,  18.7f,  19.8f,  20.9f,  22.f,   23.1f,  24.2f,  25.3f,  26.4f,  27.5f,  28.6f,  29.7f,  30.8f,
         31.9f,  33.f,   34.1f,  35.2f,  36.3f,  37.4f,  38.5f,  39.6f,  40.7f,  41.8f,  42.9f,  44.f,   45.1f,  46.2f,
         47.3f,  48.4f,  49.5f,  50.6f,  51.7f,  52.8f,  53.9f,  55.f,   56.1f,  57.2f,  58.3f,  59.4f,  60.5f,  61.6f,
         62.7f,  63.8f,  64.9f,  66.f,   67.1f,  68.2f,  69.3f,  70.4f,  71.5f,  72.6f,  73.7f,  74.8f,  75.9f,  77.f,
         78.1f,  79.2f,  80.3f,  81.4f,  82.5f,  83.6f,  84.7f,  85.8f,  86.9f,  88.f,   89.1f,  90.2f,  91.3f,  92.4f,
         93.5f,  94.6f,  95.7f,  96.8f,  97.9f,  99.f,   100.1f, 101.2f, 102.3f, 103.4f, 104.5f, 105.6f, 106.7f, 107.8f,
         108.9f, 110.f,  111.1f, 112.2f, 113.3f, 114.4f, 115.5f, 116.6f, 117.7f, 118.8f, 119.9f, 121.f,  122.1f, 123.2f,
         124.3f, 125.4f, 126.5f, 127.6f, 128.7f, 129.8f, 130.9f, 132.f,  133.1f, 134.2f, 135.3f, 136.4f, 137.5f, 138.6f,
         139.7f, 140.8f, 141.9f, 143.f,  144.1f, 145.2f, 146.3f, 147.4f, 148.5f, 149.6f, 150.7f, 151.8f, 152.9f, 154.f,
         155.1f, 156.2f, 157.3f, 158.4f, 159.5f, 160.6f, 161.7f, 162.8f, 163.9f, 165.f,  166.1f, 167.2f, 168.3f, 169.4f,
         170.5f, 171.6f, 172.7f, 173.8f, 174.9f, 176.f,  177.1f, 178.2f, 179.3f, 180.4f, 181.5f, 182.6f, 183.7f, 184.8f,
         185.9f, 187.f,  188.1f, 189.2f, 190.3f, 191.4f, 192.5f, 193.6f, 194.7f, 195.8f, 196.9f, 198.f,  199.1f, 200.2f,
         201.3f, 202.4f, 203.5f, 204.6f, 205.7f, 206.8f, 207.9f, 209.f,  210.1f, 211.2f, 212.3f, 213.4f, 214.5f, 215.6f,
         216.7f, 217.8f, 218.9f, 220.f,  221.1f, 222.2f, 223.3f, 224.4f, 225.5f, 226.6f, 227.7f, 228.8f, 229.9f, 231.f,
         232.1f, 233.2f, 234.3f, 235.4f, 236.5f, 237.6f});

    test_case.add_input<float>({0.f, 0.f, 0.75f, 2.2f, 1.2f, 0.5f, 2.8f, 1.9f, 0.f, 3.f, 0.f, 3.f});

    test_case.add_input<int64_t>({0, 2, 1});
    test_case.add_expected_output<float>(
        Shape{3, 2, 4, 4},
        {2.145f,     2.42f,      2.6950002f, 2.9700003f, 3.96f,      4.235f,     4.51f,      4.7850003f, 5.775f,
         6.05f,      6.325f,     6.6000004f, 7.59f,      7.8650007f, 8.14f,      8.415001f,  41.745003f, 42.019997f,
         42.295f,    42.57f,     43.56f,     43.835f,    44.11f,     44.385002f, 45.375f,    45.65f,     45.925003f,
         46.200005f, 47.190002f, 47.465004f, 47.74f,     48.015f,    162.77249f, 163.0475f,  163.32251f, 163.5975f,
         164.42252f, 164.69751f, 164.9725f,  165.2475f,  166.07251f, 166.3475f,  166.6225f,  166.8975f,  167.72249f,
         167.9975f,  168.27249f, 168.5475f,  202.3725f,  202.6475f,  202.9225f,  203.19751f, 204.02252f, 204.2975f,
         204.57251f, 204.8475f,  205.6725f,  205.94751f, 206.2225f,  206.4975f,  207.32251f, 207.5975f,  207.8725f,
         208.1475f,  91.162506f, 91.4375f,   91.7125f,   91.9875f,   92.8125f,   93.0875f,   93.3625f,   93.6375f,
         94.4625f,   94.7375f,   95.0125f,   95.28749f,  96.1125f,   96.3875f,   96.6625f,   96.9375f,   130.76251f,
         131.0375f,  131.3125f,  131.5875f,  132.4125f,  132.6875f,  132.9625f,  133.2375f,  134.0625f,  134.33751f,
         134.6125f,  134.88751f, 135.7125f,  135.9875f,  136.26251f, 136.53749f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_roialign16_avg_half_pixel) {
    const auto model = convert_model("roialign16_avg_half_pixel.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(
        {1.1f,     2.2f,   3.3f,   4.4f,   5.5f,   6.6f,   7.7f,   8.8f,   9.9f,   11.f,   12.1f,  13.2f,  14.3f,
         15.4f,    16.5f,  17.6f,  18.7f,  19.8f,  20.9f,  22.f,   23.1f,  24.2f,  25.3f,  26.4f,  27.5f,  28.6f,
         29.7f,    30.8f,  31.9f,  33.f,   34.1f,  35.2f,  36.3f,  37.4f,  38.5f,  39.6f,  40.7f,  41.8f,  42.9f,
         44.f,     45.1f,  46.2f,  47.3f,  48.4f,  49.5f,  50.6f,  51.7f,  52.8f,  53.9f,  55.f,   56.1f,  57.2f,
         58.3f,    59.4f,  60.5f,  61.6f,  62.7f,  63.8f,  64.9f,  66.f,   67.1f,  68.2f,  69.3f,  70.4f,  71.5f,
         72.6f,    73.7f,  74.8f,  75.9f,  77.f,   78.1f,  79.2f,  80.3f,  81.4f,  82.5f,  83.6f,  84.7f,  85.8f,
         86.9f,    88.f,   89.1f,  90.2f,  91.3f,  92.4f,  93.5f,  94.6f,  95.7f,  96.8f,  97.9f,  99.f,   100.1f,
         101.2f,   102.3f, 103.4f, 104.5f, 105.6f, 106.7f, 107.8f, 108.9f, 110.f,  111.1f, 112.2f, 113.3f, 114.4f,
         115.5f,   116.6f, 117.7f, 118.8f, 119.9f, 121.f,  122.1f, 123.2f, 124.3f, 125.4f, 126.5f, 127.6f, 128.7f,
         129.8f,   130.9f, 132.f,  133.1f, 134.2f, 135.3f, 136.4f, 137.5f, 138.6f, 139.7f, 140.8f, 141.9f, 143.f,
         144.1f,   145.2f, 146.3f, 147.4f, 148.5f, 149.6f, 150.7f, 151.8f, 152.9f, 154.f,  155.1f, 156.2f, 157.3f,
         158.4f,   159.5f, 160.6f, 161.7f, 162.8f, 163.9f, 165.f,  166.1f, 167.2f, 168.3f, 169.4f, 170.5f, 171.6f,
         172.7f,   173.8f, 174.9f, 176.f,  177.1f, 178.2f, 179.3f, 180.4f, 181.5f, 182.6f, 183.7f, 184.8f, 185.9f,
         187.198f, 188.1f, 189.2f, 190.3f, 191.4f, 192.5f, 193.6f, 194.7f, 195.8f, 196.9f, 198.f,  199.1f, 200.2f,
         201.3f,   202.4f, 203.5f, 204.6f, 205.7f, 206.8f, 207.9f, 209.f,  210.1f, 211.2f, 212.3f, 213.4f, 214.5f,
         215.6f,   216.7f, 217.8f, 218.9f, 220.f,  221.1f, 222.2f, 223.3f, 224.4f, 225.5f, 226.6f, 227.7f, 228.8f,
         229.9f,   231.f,  232.1f, 233.2f, 234.3f, 235.4f, 236.5f, 237.6f});

    test_case.add_input<float>({0.f, 0.f, 0.75f, 2.2f, 1.2f, 0.5f, 2.8f, 1.9f, 0.f, 3.f, 0.f, 3.f});

    test_case.add_input<int64_t>({0, 2, 1});
    test_case.add_expected_output<float>(
        Shape{3, 2, 4, 4},
        {1.1f,       1.1f,       1.1f,       1.1f,       1.1f,       1.1f,       1.1f,       1.1f,       2.3375f,
         2.3375f,    2.3375f,    2.3375f,    4.1525f,    4.1525f,    4.1525f,    4.1525f,    40.7f,      40.7f,
         40.7f,      40.7f,      40.7f,      40.7f,      40.7f,      40.7f,      41.9375f,   41.9375f,   41.9375f,
         41.9375f,   43.7525f,   43.7525f,   43.7525f,   43.7525f,   159.72f,    159.94f,    160.16f,    160.38f,
         159.90562f, 160.12563f, 160.34563f, 160.56563f, 160.9575f,  161.1775f,  161.3975f,  161.61751f, 162.1125f,
         162.3325f,  162.55249f, 162.77249f, 199.32f,    199.54001f, 199.76001f, 199.97998f, 199.50562f, 199.72563f,
         199.94562f, 200.16562f, 200.5575f,  200.7775f,  200.9975f,  201.2175f,  201.7125f,  201.93251f, 202.1525f,
         202.37251f, 86.9f,      86.9f,      86.9f,      86.9f,      86.9f,      86.9f,      86.9f,      86.9f,
         86.9f,      86.9f,      86.9f,      86.9f,      86.9f,      86.9f,      86.9f,      86.9f,      126.5f,
         126.5f,     126.5f,     126.5f,     126.5f,     126.5f,     126.5f,     126.5f,     126.5f,     126.5f,
         126.5f,     126.5f,     126.5f,     126.5f,     126.5f,     126.5f});
    test_case.run_with_tolerance_as_fp(0.01f);
}

OPENVINO_TEST(${BACKEND_NAME}, quant_dequant_pattern) {
    const auto model = convert_model("quant_dequant_pattern.onnx");
    auto test_case = ov::test::TestCase(model, s_device);
    // scale == 3.0
    // zero point == 10
    test_case.add_input<float>({9.0f, 10.0f, 15.0f, 20.0f, 30.0f});
    test_case.add_input<float>({1.f});
    test_case.add_expected_output<float>(Shape{5}, {9.0f, 9.0f, 15.0f, 21.0f, 30.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, quant_dequant_pattern_axis) {
    const auto model = convert_model("quant_dequant_pattern_axis.onnx");
    auto test_case = ov::test::TestCase(model, s_device);
    // axis = 1
    // scale == {2.0, 3.0, 4.0}
    // zero point == {10, 20, 30}
    test_case.add_input<float>({1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 100.0f});
    test_case.add_expected_output<float>(Shape{3, 3}, {0.f, 3.f, 4.f, 10.f, 21.f, 32.f, 40.f, 51.f, 100.f});
    test_case.add_input<float>({1.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_logsoftmax_0D) {
    auto model = convert_model("softmax_0D.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({3.141592f});
    test_case.add_expected_output<float>({0.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_logsoftmax_1D) {
    const auto model = convert_model("logsoftmax_1D.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({-1.0f, 0.0f, 1.0f});
    test_case.add_expected_output<float>(Shape{3}, {-2.4076061f, -1.407606f, -0.407606f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_logsoftmax13_1D) {
    const auto model = convert_model("logsoftmax13_1D.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({-1.0f, 0.0f, 1.0f});
    test_case.add_expected_output<float>(Shape{3}, {-2.4076061f, -1.407606f, -0.407606f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_logsoftmax13_2D) {
    const auto model = convert_model("logsoftmax13_2D.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({0.0f, 1.0f, 2.0f, 3.0f, 10000.f, 10001.f, 10002.f, 10003.f});
    test_case.add_expected_output<float>(
        Shape{2, 4},
        {-3.4401896f, -2.4401896f, -1.4401896f, -0.44018966f, -3.4401896f, -2.4401896f, -1.4401896f, -0.44018966f});
    test_case.run_with_tolerance_as_fp();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_logsoftmax13_2D_reshape) {
    const auto model = convert_model("logsoftmax13_2D.onnx");
    std::map<std::string, ov::PartialShape> shapes = {};
    ov::Shape shape = {1, 1, 4000};
    shapes[model->inputs().begin()->get_any_name()] = shape;
    EXPECT_NO_THROW(model->reshape(shapes));
    ASSERT_EQ(shape, model->outputs().begin()->get_shape());
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_hard_sigmoid) {
    auto model = convert_model("hard_sigmoid.onnx");

    const auto inf = std::numeric_limits<float>::infinity();
    const auto neg_inf = -std::numeric_limits<float>::infinity();

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({inf, neg_inf, 0.0f, 1.0f});
    test_case.add_expected_output<float>(Shape{4}, {1.0f, 0.0f, 0.5f, 0.699999988079071f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mul_v6) {
    const auto model = convert_model("mul_v6.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({1.0f, 2.0f, 3.0f});
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(Shape{3}, {3.0f, 8.0f, 15.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mul_v6_broadcast_axis_1) {
    const auto model = convert_model("mul_v6_broadcast_axis_1.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    Shape shape{1, 3, 2, 2};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(
        shape,
        {3.0f, 6.0f, 9.0f, 12.0f, 20.0f, 24.0f, 28.0f, 32.0f, 45.0f, 50.0f, 55.0f, 60.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mul_v6_broadcast_axes_1_2) {
    const auto model = convert_model("mul_v6_broadcast_axes_1_2.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    Shape shape{1, 3, 2, 2};
    std::vector<float> A(shape_size(shape), -1.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    test_case.add_expected_output<float>(shape,
                                         {-3.f, -3.f, -4.f, -4.f, -5.f, -5.f, -6.f, -6.f, -7.f, -7.f, -8.f, -8.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mul_v6_broadcast_no_axis) {
    const auto model = convert_model("mul_v6_broadcast_no_axis.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    Shape shape{2, 2};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f});
    test_case.add_expected_output<float>(shape, {3.0f, 6.0f, 9.0f, 12.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mul_v7) {
    const auto model = convert_model("mul_v7.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({1.0f, 2.0f, 3.0f});
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(Shape{3}, {3.0f, 8.0f, 15.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mul_v7_broadcast) {
    const auto model = convert_model("mul_v7_broadcast.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    Shape shape{1, 2, 3};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(shape, {3.0f, 8.0f, 15.0f, 12.0f, 20.0f, 30.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_add_v6_broadcast_axis_1) {
    const auto model = convert_model("add_v6_broadcast_axis_1.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    Shape shape{1, 3, 2, 2};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(
        shape,
        {4.0f, 5.0f, 6.0f, 7.0f, 9.0f, 10.0f, 11.0f, 12.0f, 14.0f, 15.0f, 16.0f, 17.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_add_v6_broadcast_axes_1_2) {
    const auto model = convert_model("add_v6_broadcast_axes_1_2.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    Shape shape{1, 3, 2, 2};
    std::vector<float> A(shape_size(shape), 0.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    test_case.add_expected_output<float>(shape, {3.f, 3.f, 4.f, 4.f, 5.f, 5.f, 6.f, 6.f, 7.f, 7.f, 8.f, 8.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_add_v6_broadcast_no_axis) {
    const auto model = convert_model("add_v6_broadcast_no_axis.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    Shape shape{2, 2};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f});
    test_case.add_expected_output<float>(shape, {4.0f, 5.0f, 6.0f, 7.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_add_v7) {
    const auto model = convert_model("add_v7.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({1.0f, 2.0f, 3.0f});
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(Shape{3}, {4.0f, 6.0f, 8.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_sub_v6_broadcast_axis_1) {
    const auto model = convert_model("sub_v6_broadcast_axis_1.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    Shape shape{1, 3, 2, 2};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(shape,
                                         {-2.0f, -1.0f, 0.0f, 1.0f, 1.0f, 2.0f, 3.0f, 4.0f, 4.0f, 5.0f, 6.0f, 7.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_sub_v6_broadcast_axes_1_2) {
    const auto model = convert_model("sub_v6_broadcast_axes_1_2.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    Shape shape{1, 3, 2, 2};
    std::vector<float> A(shape_size(shape), 0.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    test_case.add_expected_output<float>(shape,
                                         {-3.f, -3.f, -4.f, -4.f, -5.f, -5.f, -6.f, -6.f, -7.f, -7.f, -8.f, -8.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_sub_v6_broadcast_no_axis) {
    const auto model = convert_model("sub_v6_broadcast_no_axis.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    Shape shape{2, 2};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f});
    test_case.add_expected_output<float>(shape, {-2.0f, -1.0f, 0.0f, 1.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_sub_v7) {
    const auto model = convert_model("sub_v7.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({1.0f, 2.0f, 3.0f});
    test_case.add_input<float>({3.0f, 8.0f, 7.0f});
    test_case.add_expected_output<float>(Shape{3}, {-2.0f, -6.0f, -4.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_sub_v7_broadcast) {
    const auto model = convert_model("sub_v7_broadcast.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    Shape shape{1, 2, 3};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(shape, {-2.0f, -2.0f, -2.0f, 1.0f, 1.0f, 1.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_div_v6_broadcast_axis_1) {
    const auto model = convert_model("div_v6_broadcast_axis_1.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    Shape shape{1, 3, 2, 2};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(
        shape,
        {0.3333333f, 0.6666666f, 1.0f, 1.333333f, 1.25f, 1.5f, 1.75f, 2.0f, 1.8f, 2.0, 2.2f, 2.4f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_div_v6_broadcast_axes_1_2) {
    const auto model = convert_model("div_v6_broadcast_axes_1_2.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    Shape shape{1, 3, 2, 2};
    std::vector<float> A(shape_size(shape), 840.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    test_case.add_expected_output<float>(
        shape,
        {280.f, 280.f, 210.f, 210.f, 168.f, 168.f, 140.f, 140.f, 120.f, 120.f, 105.f, 105.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_div_v6_broadcast_no_axis) {
    const auto model = convert_model("div_v6_broadcast_no_axis.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    Shape shape{2, 2};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({2.0f});
    test_case.add_expected_output<float>(shape, {0.5f, 1.0f, 1.5f, 2.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_div_v7) {
    const auto model = convert_model("div_v7.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({1.0f, 2.0f, 3.0f});
    test_case.add_input<float>({3.0f, 8.0f, 7.0f});
    test_case.add_expected_output<float>(Shape{3}, {0.3333333f, 0.25f, 0.4285714f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_div_v7_broadcast) {
    const auto model = convert_model("div_v7_broadcast.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    Shape shape{1, 2, 3};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(shape, {0.3333333f, 0.5f, 0.6f, 1.3333333f, 1.25f, 1.2f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_dangling_parameter) {
    auto model = convert_model("dangling_parameter.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({-1.0f, 2.0f, -3.0f});
    test_case.add_expected_output<float>(Shape{3}, {1.0f, 2.0f, 3.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_clip_inbounds) {
    auto model = convert_model("test_clip_inbounds.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    const std::vector<int32_t> data{-1, 0, 1, -9999, 9999};
    test_case.add_input<int32_t>(data);
    test_case.add_expected_output<int32_t>(Shape{data.size()}, data);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_clip_no_min_no_max) {
    auto model = convert_model("clip_no_min_no_max.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    const std::vector<float> data{-1.6f, -0.1f, 10.f, 0.f, -10.f, 1.99f, 2.015f, 3.f};

    test_case.add_input<float>(data);

    test_case.add_expected_output<float>(Shape{2, 4}, data);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_clip_no_min_no_max_inf) {
    auto model = convert_model("clip_no_min_no_max.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    const std::vector<float> data{std::numeric_limits<float>::infinity(),
                                  -std::numeric_limits<float>::infinity(),
                                  static_cast<float>(std::numeric_limits<float>::max()),
                                  std::numeric_limits<float>::min(),
                                  std::numeric_limits<float>::max(),
                                  std::numeric_limits<float>::lowest(),
                                  0.f,
                                  -1.f};

    const std::vector<float> expected_output{std::numeric_limits<float>::max(),
                                             std::numeric_limits<float>::lowest(),
                                             std::numeric_limits<float>::max(),
                                             std::numeric_limits<float>::min(),
                                             std::numeric_limits<float>::max(),
                                             std::numeric_limits<float>::lowest(),
                                             0.f,
                                             -1.f};

    test_case.add_input<float>(data);

    test_case.add_expected_output<float>(Shape{2, 4}, expected_output);
    test_case.run_with_tolerance_as_fp(0.f);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_clip_no_min_set_max) {
    auto model = convert_model("clip_no_min_set_max.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    const std::vector<float> data{-1.6f, -0.1f, 10.f, 0.f, -10.f, 1.99f, 2.015f, 3.f};
    const std::vector<float> max_val{2.01f};
    const std::vector<float> output{-1.6f, -0.1f, 2.01f, 0.f, -10.f, 1.99f, 2.01f, 2.01f};

    test_case.add_input<float>(data);
    test_case.add_input<float>(max_val);

    test_case.add_expected_output<float>(Shape{2, 4}, output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_clip_set_min_no_max) {
    auto model = convert_model("clip_set_min_no_max.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    const std::vector<float> data{-1.6f, -0.1f, 10.f, 0.f, -10.f, 1.99f, 2.015f, 3.f};
    const std::vector<float> min_val{-1.59f};
    const std::vector<float> output{-1.59f, -0.1f, 10.f, 0.f, -1.59f, 1.99f, 2.015f, 3.f};

    test_case.add_input<float>(data);
    test_case.add_input<float>(min_val);

    test_case.add_expected_output<float>(Shape{2, 4}, output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_clip_no_min_no_max_int64) {
    auto model = convert_model("clip_no_min_no_max_int64.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    const std::vector<int64_t> data{INT64_MAX, INT64_MIN, INT32_MAX, INT32_MIN, 0, -1, 1, 0};

    test_case.add_input<int64_t>(data);

    test_case.add_expected_output<int64_t>(Shape{2, 4}, data);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_clip_no_min_set_max_int64) {
    auto model = convert_model("clip_no_min_set_max_int64.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    const std::vector<int64_t> data{INT64_MAX, INT64_MIN, INT32_MAX, INT32_MIN, 0, -1, 1, 0};
    const std::vector<int64_t> max_val{INT32_MAX};
    const std::vector<int64_t> output{INT32_MAX, INT64_MIN, INT32_MAX, INT32_MIN, 0, -1, 1, 0};

    test_case.add_input<int64_t>(data);
    test_case.add_input<int64_t>(max_val);

    test_case.add_expected_output<int64_t>(Shape{2, 4}, output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_clip_set_min_no_max_initializers) {
    auto model = convert_model("clip_set_min_no_max_initializers.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    const std::vector<float> data{-1.6f, -0.1f, 10.f, 0.f, -10.f, 1.99f, 2.015f, 3.f};
    const std::vector<float> output{-1.59f, -0.1f, 10.f, 0.f, -1.59f, 1.99f, 2.015f, 3.f};

    test_case.add_input<float>(data);

    test_case.add_expected_output<float>(Shape{2, 4}, output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_clip_set_min_set_max) {
    auto model = convert_model("clip_set_min_set_max.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    const std::vector<float> data{-1.6f, -0.1f, 10.f, 0.f, -10.f, 1.99f, 2.015f, 3.f};
    const std::vector<float> min_val{-1.59f};
    const std::vector<float> max_val{2.01f};
    const std::vector<float> output{-1.59f, -0.1f, 2.01f, 0.f, -1.59f, 1.99f, 2.01f, 2.01f};

    test_case.add_input<float>(data);
    test_case.add_input<float>(min_val);
    test_case.add_input<float>(max_val);

    test_case.add_expected_output<float>(Shape{2, 4}, output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_clip_set_min_set_max_initializers) {
    auto model = convert_model("clip_set_min_set_max_initializers.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    const std::vector<float> data{-1.6f, -0.1f, 10.f, 0.f, -10.f, 1.99f, 2.015f, 3.f};
    const std::vector<float> output{-1.59f, -0.1f, 2.01f, 0.f, -1.59f, 1.99f, 2.01f, 2.01f};

    test_case.add_input<float>(data);

    test_case.add_expected_output<float>(Shape{2, 4}, output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_mvn_v6) {
    auto model = convert_model("mvn_v6.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({0.8439683f,  0.5665144f, 0.05836735f, 0.02916367f, 0.12964272f, 0.5060197f, 0.79538304f,
                                0.9411346f,  0.9546573f, 0.17730942f, 0.46192095f, 0.26480448f, 0.6746842f, 0.01665257f,
                                0.62473077f, 0.9240844f, 0.9722341f,  0.11965699f, 0.41356155f, 0.9129373f, 0.59330076f,
                                0.81929934f, 0.7862604f, 0.11799799f, 0.69248444f, 0.54119414f, 0.07513223f});
    test_case.add_expected_output<float>(
        Shape{3, 3, 3, 1},
        {1.3546423f,  0.33053496f, -1.5450814f,  -1.2106764f,  -0.8925952f,  0.29888135f, 0.38083088f,
         0.81808794f, 0.85865635f, -1.1060555f,  -0.05552877f, -0.78310335f, 0.83281356f, -1.250282f,
         0.67467856f, 0.7669372f,  0.9113869f,   -1.6463585f,  -0.23402764f, 1.6092131f,  0.42940593f,
         1.2906139f,  1.1860244f,  -0.92945826f, 0.0721334f,   -0.38174f,    -1.7799333f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dropout1_no_training_no_return_mask) {
    auto model = convert_model("dropout1_no_training_no_return_mask.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    const std::vector<float> data(3 * 4 * 5, 2.0f);
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(Shape{3, 4, 5}, data);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dropout1_no_training_return_mask) {
    auto model = convert_model("dropout1_no_training_return_mask.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    const std::vector<float> data(3 * 4 * 5, 2.0f);
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(Shape{3, 4, 5}, data);
    test_case.add_expected_output<int32_t>(Shape{3, 4, 5},
                                           std::vector<int32_t>(3 * 4 * 5, 1));  // // bool converted to i32
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dropout7_no_return_mask) {
    auto model = convert_model("dropout7_no_return_mask.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    const std::vector<float> data(3 * 4 * 5, 2.0f);
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(Shape{3, 4, 5}, data);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dropout12_no_training_no_return_mask) {
    auto model = convert_model("dropout12_no_training_no_return_mask.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    const std::vector<float> data(3 * 4 * 5, 2.0f);
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(Shape{3, 4, 5}, data);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dropout12_no_training_return_mask) {
    auto model = convert_model("dropout12_no_training_return_mask.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    const std::vector<float> data(3 * 4 * 5, 2.0f);
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(Shape{3, 4, 5}, data);
    test_case.add_expected_output<int32_t>(Shape{3, 4, 5},
                                           std::vector<int32_t>(3 * 4 * 5, 1));  // bool converted to i32
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dropout12_no_traning_no_const_rato) {
    auto model = convert_model("dropout12_no_traning_no_const_rato.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({1, 2, 3, 4});
    // test_case.add_input<float>(Shape{}, {0.5}); // ratio input is ignored

    test_case.add_expected_output<float>(Shape{1, 4}, {1., 2., 3., 4.});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dropout12_training_mode) {
    try {
        auto model = convert_model("dropout12_training_mode.onnx");
        FAIL() << "Expected exception was not thrown";
    } catch (const ov::Exception& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string("Training mode is not supported for Dropout op"));
    } catch (...) {
        FAIL() << "Expected ov::Exception was not thrown";
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dropout12_not_const_training_mode) {
    try {
        auto model = convert_model("dropout12_not_const_training_mode.onnx");
        FAIL() << "Expected exception was not thrown";
    } catch (const ov::Exception& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string("Non-constant training_mode input is not supported."));
    } catch (...) {
        FAIL() << "Expected ov::Exception was not thrown";
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_multiple_slices_last_layer) {
    std::vector<float> data(1 * 30 * 320 * 320);
    std::fill(data.begin(), data.end(), 1.f);

    const auto model = convert_model("multiple_slices_last_layer.onnx");
    auto test_case = ov::test::TestCase(model, s_device);
    std::vector<float> o1(1 * 320 * 320 * 21);
    std::fill(o1.begin(), o1.end(), 1.f);

    std::vector<float> o2(1 * 320 * 320 * 9);
    std::fill(o2.begin(), o2.end(), 1.f);

    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(Shape{1, 320, 320, 21}, o1);
    test_case.add_expected_output<float>(Shape{1, 320, 320, 9}, o2);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_slice_const_axes_source) {
    auto model = convert_model("slice_const_axes_source.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    test_case.add_expected_output<float>(Shape{2, 2}, {2.f, 3.f, 6.f, 7.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_softmax_crossentropy_loss_mean) {
    auto model = convert_model("softmax_crossentropy_loss_mean.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({0.54881352186203f,
                                0.7151893377304077f,
                                0.6027633547782898f,
                                0.5448831915855408f,
                                0.42365479469299316f,
                                0.6458941102027893f,
                                0.4375872015953064f,
                                0.891772985458374f,
                                0.9636627435684204f,
                                0.3834415078163147f,
                                0.7917250394821167f,
                                0.5288949012756348f,
                                0.5680445432662964f,
                                0.9255966544151306f,
                                0.07103605568408966f});
    test_case.add_input<int64_t>({1, 4, 3});
    test_case.add_expected_output<float>(Shape{}, {1.561384797096252441f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_negativelog_likelihood_loss) {
    auto model = convert_model("negativelog_likelihood_loss.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({
        0.54881352186203f,    0.7151893377304077f,   0.6027633547782898f, 0.5448831915855408f, 0.42365479469299316f,
        0.6458941102027893f,  0.4375872015953064f,   0.891772985458374f,  0.9636627435684204f, 0.3834415078163147f,
        0.7917250394821167f,  0.5288949012756348f,   0.5680445432662964f, 0.9255966544151306f, 0.07103605568408966f,
        0.08712930232286453f, 0.020218396559357643f, 0.832619845867157f,  0.7781567573547363f, 0.8700121641159058f,
        0.978618323802948f,   0.7991585731506348f,   0.4614793658256531f, 0.7805292010307312f, 0.11827442795038223f,
        0.6399210095405579f,  0.14335328340530396f,  0.9446688890457153f, 0.5218483209609985f, 0.4146619439125061f,
    });
    test_case.add_input<int64_t>({3, 3, 2, 4, 2, 0});
    test_case.add_expected_output<float>(Shape{}, {-0.531306922435760498f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_fill_input_as_shape_default_value) {
    auto model = convert_model("constant_fill_input_as_shape_default_value.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output<float>(Shape{1, 2, 3}, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_fill_input_as_shape_u8_type) {
    auto model = convert_model("constant_fill_input_as_shape_u8_type.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output<uint8_t>(Shape{3, 1, 2}, {3, 3, 3, 3, 3, 3});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_fill_extra_shape) {
    auto model = convert_model("constant_fill_extra_shape.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output<float>(Shape{3, 1, 2, 2, 1}, std::vector<float>(12, 3.0f));
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_fill_shape_attribute) {
    auto model = convert_model("constant_fill_shape_attribute.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output<int32_t>(Shape{2, 3, 4}, std::vector<int32_t>(24, 5));
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_float_tensor) {
    auto model = convert_model("constant_float_tensor.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output<float>(Shape{2, 3}, {0.0f, 0.5f, 1.f, 1.5f, 2.f, 2.5f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_bfloat_tensor) {
    auto model = convert_model("constant_bfloat_tensor.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output<bfloat16>(Shape{2, 3}, {0.f, 5.f, 10.f, 15.f, 20.f, 25.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_float_scalar) {
    auto model = convert_model("constant_float_scalar.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output<float>(Shape{}, {0.5f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_float_array) {
    auto model = convert_model("constant_float_array.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output<float>(Shape{3}, {0.5f, 1.f, 1.5f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_integer_scalar) {
    auto model = convert_model("constant_integer_scalar.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output<std::int64_t>(Shape{}, {1});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_integer_array) {
    auto model = convert_model("constant_integer_array.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output<std::int64_t>(Shape{3}, {0, 1, 2});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_float_2x2) {
    auto model = convert_model("constant_sparse_tensor.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output<float>(Shape{2, 2}, {0.f, 5.f, 0.f, 0.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_float_3x4) {
    auto model = convert_model("constant_sparse_tensor_float_3x4.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output<float>(Shape{3, 4}, {1.f, 0.f, 0.f, 8.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 3.f, 0.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_float_3x4_linearized_indices) {
    auto model = convert_model("constant_sparse_tensor_float_3x4_linearized_indices.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output<float>(Shape{3, 4}, {1.f, 0.f, 0.f, 8.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 3.f, 0.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_int32_3x4) {
    auto model = convert_model("constant_sparse_tensor_int32_3x4.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output<int32_t>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_int64_3x4) {
    auto model = convert_model("constant_sparse_tensor_int64_3x4.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output<int64_t>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_boolean_3x4) {
    auto model = convert_model("constant_sparse_tensor_boolean_3x4.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output<bool>(Shape{3, 4}, {1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_float16_3x4) {
    auto model = convert_model("constant_sparse_tensor_float16_3x4.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output<ov::float16>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_double_3x4) {
    auto model = convert_model("constant_sparse_tensor_double_3x4.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output<double>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_int8_3x4) {
    auto model = convert_model("constant_sparse_tensor_int8_3x4.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output<int8_t>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_int16_3x4) {
    auto model = convert_model("constant_sparse_tensor_int16_3x4.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output<int16_t>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_uint8_3x4) {
    auto model = convert_model("constant_sparse_tensor_uint8_3x4.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output<uint8_t>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_uint16_3x4) {
    auto model = convert_model("constant_sparse_tensor_uint16_3x4.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output<uint16_t>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_uint32_3x4) {
    auto model = convert_model("constant_sparse_tensor_uint32_3x4.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output<uint32_t>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_uint64_3x4) {
    auto model = convert_model("constant_sparse_tensor_uint64_3x4.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output<uint64_t>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_bfloat16_3x4) {
    auto model = convert_model("constant_sparse_tensor_bfloat16_3x4.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output<ov::bfloat16>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_float_8x17) {
    auto model = convert_model("constant_sparse_tensor_float_8x17.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
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

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_float_2x3x4) {
    auto model = convert_model("constant_sparse_tensor_float_2x3x4.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output<float>(Shape{2, 3, 4}, {1.f, 0.f, 0.f, 8.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 3.f, 0.f,
                                                          0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 3.f, 0.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_float_2x2x3x4) {
    auto model = convert_model("constant_sparse_tensor_float_2x2x3x4.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output<float>(
        Shape{2, 2, 3, 4},
        {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 2.f, 3.f, 0.f, 0.f, 0.f, 0.f,
         0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 8.f, 0.f, 1.f, 2.f, 0.f,
         0.f, 0.f, 3.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 2.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_einsum_sum) {
    auto model = convert_model("einsum_sum.onnx");
    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{3, 4},
                               {1.764052345967664f,
                                0.4001572083672233f,
                                0.9787379841057392f,
                                2.240893199201458f,
                                1.8675579901499675f,
                                -0.977277879876411f,
                                0.9500884175255894f,
                                -0.1513572082976979f,
                                -0.10321885179355784f,
                                0.41059850193837233f,
                                0.144043571160878f,
                                1.454273506962975f});
    test_case.add_expected_output<float>(Shape{3}, {5.3838407376420845f, 1.689011319501448f, 1.9056967282686674f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_float16_tensor_as_int32) {
    const auto model = convert_model("conv_fp16_W_as_int32.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    // clang-format off
    test_case.add_input<ov::float16>(Shape{1, 1, 4, 4},
            {   0,  1,  2,  3,
                4,  5,  6,  7,
                8,  9,  10, 11,
                12, 13, 14, 15  });
    /* filters
            [[[[0.25, 0.5, 0.25],
               [0.5,  1.0, 0.5],
               [0.25, 0.5, 0.25]]]] */
    test_case.add_expected_output<ov::float16>(Shape{1, 1, 2, 2},
            {   20, 24,
                36, 40  });
    // clang-format on
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_max_pool_3d) {
    const auto model = convert_model("max_pool_3d.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<int32_t>(Shape{1, 3, 3}, {-1, 0, 1, 20, -20, 10, 0, 2, 1});
    test_case.add_expected_output<int32_t>(Shape{1, 3, 2}, {0, 1, 20, 10, 2, 2});
    test_case.add_expected_output<int64_t>(Shape{1, 3, 2}, {1, 2, 3, 5, 7, 7});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_max_pool_4d_ceil_mode) {
    const auto model = convert_model("max_pool_4d_ceil_mode.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<int32_t>(Shape{1, 1, 4, 4}, gen_range<int32_t>(16, 1));
    test_case.add_expected_output<int32_t>(Shape{1, 1, 2, 2}, {11, 12, 15, 16});
    test_case.add_expected_output<int64_t>(Shape{1, 1, 2, 2}, {10, 11, 14, 15});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_max_pool_4d_dilations) {
    const auto model = convert_model("max_pool_4d_dilations.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<int32_t>(Shape{1, 1, 4, 4}, {9, 10, 11, 12, 1, 2, 3, 4, 16, 14, 15, 13, 5, 6, 8, 7});
    test_case.add_expected_output<int32_t>(Shape{1, 1, 2, 2}, {16, 14, 8, 7});
    test_case.add_expected_output<int64_t>(Shape{1, 1, 2, 2}, {8, 9, 14, 15});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_max_pool_4d_strides) {
    // kernel: 3x3
    // strides: 3, 3
    // explicit pads: 2, 2, 2, 2
    const auto model = convert_model("max_pool_4d_strides.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<int8_t>(Shape{1, 1, 5, 5}, gen_range<int8_t>(25, 1));
    test_case.add_expected_output<int8_t>(Shape{1, 1, 3, 3}, {1, 4, 5, 16, 19, 20, 21, 24, 25});
    test_case.add_expected_output<int64_t>(Shape{1, 1, 3, 3}, {0, 3, 4, 15, 18, 19, 20, 23, 24});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_max_pool_4d_ceil_strides) {
    // kernel: 3x3
    // strides: 2, 2
    // ceil_mode: 1
    const auto model = convert_model("max_pool_4d_ceil_strides.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(
        Shape{1, 1, 4, 4},
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
    test_case.add_expected_output<float>(Shape{1, 1, 2, 2}, {11.0f, 12.0f, 15.0f, 16.0f});
    test_case.add_expected_output<int64_t>(Shape{1, 1, 2, 2}, {10, 11, 14, 15});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_random_uniform) {
    const auto model = convert_model("random_uniform.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    if (std::string("${BACKEND_NAME}") == std::string("IE_GPU")) {
        test_case.add_expected_output<float>(Shape{2, 2}, {40.96875f, 43.4375f, 49.4375f, 45.46875f});
    } else {
        test_case.add_expected_output<float>(Shape{2, 2}, {43.70129f, 45.26042f, 43.48503f, 46.43743f});
    }

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_random_uniform_like) {
    const auto model = convert_model("random_uniform_like.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{2, 2}, {41, 42, 43, 44});

    if (std::string("${BACKEND_NAME}") == std::string("IE_GPU")) {
        test_case.add_expected_output<float>(Shape{2, 2}, {40.96875f, 43.4375f, 49.4375f, 45.46875f});
    } else {
        test_case.add_expected_output<float>(Shape{2, 2}, {43.70129f, 45.26042f, 43.48503f, 46.43743f});
    }

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_random_normal) {
    const auto model = convert_model("random_normal.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    if (std::string("${BACKEND_NAME}") == std::string("IE_GPU")) {
        test_case.add_expected_output<float>(Shape{2, 2}, {77.351875f, 74.047821f, -5.996780f, 13.922290f});
    } else {
        test_case.add_expected_output<float>(Shape{2, 2}, {30.357481f, 72.41268f, 12.999034f, 70.04985f});
    }

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_random_normal_like) {
    const auto model = convert_model("random_normal_like.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{2, 2}, {0, 0, 0, 0});

    if (std::string("${BACKEND_NAME}") == std::string("IE_GPU")) {
        test_case.add_expected_output<float>(Shape{2, 2}, {77.351875f, 74.047821f, -5.996780f, 13.922290f});
    } else {
        test_case.add_expected_output<float>(Shape{2, 2}, {30.357481f, 72.41268f, 12.999034f, 70.04985f});
    }

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_aten_embedding_bag_packed_sum_2fin) {
    const auto model = convert_model("aten_embedding_sum_packed_2in.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{5, 2}, {-0.2f, -0.6f, -0.1f, -0.4f, -1.9f, -1.8f, -1.f, 1.5f, 0.8f, -0.7f});
    test_case.add_input<int32_t>(Shape{3, 2}, {0, 2, 1, 2, 3, 4});  // indices

    test_case.add_expected_output<float>(Shape{3, 2}, {-2.1f, -2.4f, -2.f, -2.2f, -0.19999999f, 0.8f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_aten_embedding_bag_packed_sum_3fin_offsets_none) {
    const auto model = convert_model("aten_embedding_sum_packed_3in_offset_none.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{5, 2}, {-0.2f, -0.6f, -0.1f, -0.4f, -1.9f, -1.8f, -1.f, 1.5f, 0.8f, -0.7f});
    test_case.add_input<int32_t>(Shape{3, 2}, {0, 2, 1, 2, 3, 4});  // indices

    test_case.add_expected_output<float>(Shape{3, 2}, {-2.1f, -2.4f, -2.f, -2.2f, -0.19999999f, 0.8f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_aten_embedding_bag_packed_sum_4fin_per_sample_weights) {
    const auto model = convert_model("aten_embedding_sum_packed_4in_per_sample_weights.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{5, 2}, {-0.2f, -0.6f, -0.1f, -0.4f, -1.9f, -1.8f, -1.f, 1.5f, 0.8f, -0.7f});
    test_case.add_input<int32_t>(Shape{3, 2}, {0, 2, 1, 2, 3, 4});                  // indices
    test_case.add_input<float>(Shape{3, 2}, {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f});  // per_sample_weights

    test_case.add_expected_output<float>(Shape{3, 2}, {-1.05f, -1.2f, -1.f, -1.1f, -0.09999999f, 0.4f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_aten_embedding_bag_packed_sum_4in_two_none) {
    const auto model = convert_model("aten_embedding_sum_packed_4in_two_none.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{5, 2}, {-0.2f, -0.6f, -0.1f, -0.4f, -1.9f, -1.8f, -1.f, 1.5f, 0.8f, -0.7f});
    test_case.add_input<int32_t>(Shape{3, 2}, {0, 2, 1, 2, 3, 4});  // indices

    test_case.add_expected_output<float>(Shape{3, 2}, {-2.1f, -2.4f, -2.f, -2.2f, -0.19999999f, 0.8f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_aten_embedding_bag_offsets_sum_3in) {
    const auto model = convert_model("aten_embedding_sum_offset_3in.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{5, 2}, {-0.2f, -0.6f, -0.1f, -0.4f, -1.9f, -1.8f, -1.f, 1.5f, 0.8f, -0.7f});
    test_case.add_input<int32_t>(Shape{4}, {0, 2, 3, 4});  // indices
    test_case.add_input<int32_t>(Shape{3}, {0, 2, 2});     // offsets

    test_case.add_expected_output<float>(Shape{3, 2}, {-2.1f, -2.4f, 0.f, 0.f, -0.2f, 0.8f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_aten_embedding_bag_offsets_sum_4in) {
    const auto model = convert_model("aten_embedding_sum_offset_4in.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{5, 2}, {-0.2f, -0.6f, -0.1f, -0.4f, -1.9f, -1.8f, -1.f, 1.5f, 0.8f, -0.7f});
    test_case.add_input<int32_t>(Shape{4}, {0, 2, 3, 4});            // indices
    test_case.add_input<int32_t>(Shape{3}, {0, 2, 2});               // offsets
    test_case.add_input<float>(Shape{4}, {0.5f, 0.5f, 0.5f, 0.5f});  // per_sample_weights

    test_case.add_expected_output<float>(Shape{3, 2}, {-1.05f, -1.2f, 0.f, 0.f, -0.09999999f, 0.4f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_aten_embedding_bag_many_node_outputs) {
    const auto model = convert_model("aten_embedding_sum_many_outputs.onnx");

    // 4 outputs in onnx Node (1 connected and 3 not connected)
    EXPECT_EQ(model->outputs().size(), 1);
    EXPECT_EQ(model->get_results().size(), 1);

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{5, 2}, {-0.2f, -0.6f, -0.1f, -0.4f, -1.9f, -1.8f, -1.f, 1.5f, 0.8f, -0.7f});
    test_case.add_input<int32_t>(Shape{4}, {0, 2, 3, 4});  // indices
    test_case.add_input<int32_t>(Shape{3}, {0, 2, 2});     // offsets

    test_case.add_expected_output<float>(Shape{3, 2}, {-2.1f, -2.4f, 0.f, 0.f, -0.2f, 0.8f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_aten_unsupported_embedding_mode) {
    try {
        const auto model = convert_model("aten_unsupported_embedding_mode.onnx");
        FAIL() << "Expected exception was not thrown.";
    } catch (const ov::Exception& e) {
        EXPECT_HAS_SUBSTRING(
            e.what(),
            std::string(
                "Unsupported mode, only `0` (sum) is supported as ATen embedding_bag `mode` attribute. Got: 1"));
    } catch (...) {
        FAIL() << "Other exception than expected was thrown.";
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_aten_unsupported_operator) {
    try {
        const auto model = convert_model("aten_unsupported_operator.onnx");
        FAIL() << "Expected exception was not thrown.";
    } catch (const ov::Exception& e) {
        EXPECT_HAS_SUBSTRING(
            e.what(),
            std::string(
                "Only `embedding_bag` is supported as ATen `operator` attribute. Got: test_unsupported_operator"));
    } catch (...) {
        FAIL() << "Other exception than expected was thrown.";
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_unsqueeze_ai_onnx_domain) {
    auto model = convert_model("unsqueeze_ai_onnx_domain.onnx");

    auto input = ov::test::NDArray<float, 3>({{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                              {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                              {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
                     .get_vector();

    auto expected_output =
        ov::test::NDArray<float, 4>({{{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                      {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                      {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}}})
            .get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input(input);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_unsqueeze_default_domain) {
    auto model = convert_model("unsqueeze_default_domain.onnx");

    auto input = ov::test::NDArray<float, 3>({{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                              {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                              {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
                     .get_vector();

    auto expected_output =
        ov::test::NDArray<float, 4>({{{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                      {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                      {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}}})
            .get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input(input);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_unsqueeze_default_domain_opset13) {
    auto model = convert_model("unsqueeze_default_domain_opset13.onnx");

    auto input = ov::test::NDArray<float, 3>({{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                              {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                              {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
                     .get_vector();
    auto expected_output =
        ov::test::NDArray<float, 4>({{{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                      {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                      {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}}})
            .get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input(input);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_unsqueeze_ai_onnx_domain_opset13) {
    auto model = convert_model("unsqueeze_ai_onnx_domain_opset13.onnx");

    auto input = ov::test::NDArray<float, 3>({{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                              {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                              {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
                     .get_vector();
    auto expected_output =
        ov::test::NDArray<float, 4>({{{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                      {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                      {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}}})
            .get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input(input);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_expand_failsafe_node) {
    const auto model = convert_model("expand_failsafe_node.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    const auto input_data = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
    test_case.add_input<float>(input_data);
    // the target shape is an empty constant so the Expand operation should not modify the input shape
    test_case.add_expected_output<float>(input_data);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_scan15_fib_like) {
    const auto model = convert_model("scan15_fib_like.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{}, {0});
    test_case.add_input<float>(Shape{}, {1});
    test_case.add_input<float>(Shape{10}, std::vector<float>(10, 1));

    test_case.add_expected_output<float>(Shape{}, {55});
    test_case.add_expected_output<float>(Shape{}, {89});
    test_case.add_expected_output<float>(Shape{10}, {1., 2., 3., 5., 8., 13., 21., 34., 55., 89.});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_scan15_fib_like_out_rev) {
    const auto model = convert_model("scan15_fib_like_out_rev.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{}, {0});
    test_case.add_input<float>(Shape{}, {1});
    test_case.add_input<float>(Shape{10}, std::vector<float>(10, 1));

    test_case.add_expected_output<float>(Shape{}, {55});
    test_case.add_expected_output<float>(Shape{}, {89});
    test_case.add_expected_output<float>(Shape{10}, {89., 55., 34., 21., 13., 8., 5., 3., 2., 1.});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_scan15_fib_like_input_rev) {
    const auto model = convert_model("scan15_fib_like_input_rev.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{}, {0});
    test_case.add_input<float>(Shape{}, {1});
    test_case.add_input<float>(Shape{10},
                               std::vector<float>{0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f});

    test_case.add_expected_output<float>(Shape{}, {0.14897026f});
    test_case.add_expected_output<float>(Shape{}, {0.f});
    test_case.add_expected_output<float>(
        Shape{10},
        {0.9f, 1.52f, 1.694f, 1.9284f, 1.8112f, 1.4958401f, 0.9921121f, 0.49759045f, 0.14897026f, 0.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_scan15_fib_like_input_out_rev) {
    const auto model = convert_model("scan15_fib_like_input_out_rev.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{}, {0});
    test_case.add_input<float>(Shape{}, {1});
    test_case.add_input<float>(Shape{10},
                               std::vector<float>{0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f});

    test_case.add_expected_output<float>(Shape{}, {0.14897026f});
    test_case.add_expected_output<float>(Shape{}, {0.});
    test_case.add_expected_output<float>(
        Shape{10},
        {0.f, 0.14897026f, 0.49759045f, 0.9921121f, 1.4958401f, 1.8112f, 1.9284f, 1.694f, 1.52f, 0.9f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_scan15_ND_mixed_ones) {
    const auto model = convert_model("scan15_ND_mixed.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
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

OPENVINO_TEST(${BACKEND_NAME}, onnx_scan15f_ND_mixed_vals) {
    const auto model = convert_model("scan15_ND_mixed.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{1, 3, 2}, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f});
    test_case.add_input<float>(Shape{1, 3, 2}, {1.f, 1.f, 1.f, 1.f, 1.f, 1.f});
    std::vector<float> sequence_vals{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.f,
                                     1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.f,
                                     2.1f, 2.2f, 2.3f, 2.4f, 2.5f, 2.6f, 2.7f, 2.8f, 2.9f, 3.f};
    test_case.add_input<float>(Shape{1, 3, 5, 2}, sequence_vals);  // multiply factor (reverse)
    test_case.add_input<float>(Shape{1, 5, 3, 2}, sequence_vals);  // div factor

    test_case.add_expected_output<float>(Shape{1, 3, 2},
                                         {2.7327938f, 2.1428573f, 21.070545f, 16.92727f, 49.765778f, 41.444443f});
    test_case.add_expected_output<float>(Shape{1, 3, 2},
                                         {0.40161943f, 0.5274726f, 16.80789f, 14.025973f, 59.98805f, 50.518517f});
    test_case.add_expected_output<float>(
        Shape{1, 3, 2, 5},
        {0.40161943f, 2.7327938f, 7.3076925f, 10.f,       9.f,        0.5274726f, 2.1428573f, 4.714286f,
         6.f,         5.f,        16.80789f,  21.070545f, 20.185184f, 13.851851f, 6.333333f,  14.025973f,
         16.92727f,   15.799998f, 10.799999f, 5.f,        59.98805f,  49.765778f, 33.074867f, 16.690908f,
         5.8f,        50.518517f, 41.444443f, 27.444445f, 14.f,       5.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_scan15_ND_mixed_vals_neg_axes) {
    // Negative indices for scan_input_axes and scan_output_axes attributes
    const auto model = convert_model("scan15_ND_mixed_neg_axes.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{1, 3, 2}, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f});
    test_case.add_input<float>(Shape{1, 3, 2}, {1.f, 1.f, 1.f, 1.f, 1.f, 1.f});
    std::vector<float> sequence_vals{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.f,
                                     1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.f,
                                     2.1f, 2.2f, 2.3f, 2.4f, 2.5f, 2.6f, 2.7f, 2.8f, 2.9f, 3.f};
    test_case.add_input<float>(Shape{1, 3, 5, 2}, sequence_vals);  // multiply factor (reverse)
    test_case.add_input<float>(Shape{1, 5, 3, 2}, sequence_vals);  // div factor

    test_case.add_expected_output<float>(Shape{1, 3, 2},
                                         {2.7327938f, 2.1428573f, 21.070545f, 16.92727f, 49.765778f, 41.444443f});
    test_case.add_expected_output<float>(Shape{1, 3, 2},
                                         {0.40161943f, 0.5274726f, 16.80789f, 14.025973f, 59.98805f, 50.518517f});
    test_case.add_expected_output<float>(
        Shape{1, 3, 2, 5},
        {0.40161943f, 2.7327938f, 7.3076925f, 10.f,       9.f,        0.5274726f, 2.1428573f, 4.714286f,
         6.f,         5.f,        16.80789f,  21.070545f, 20.185184f, 13.851851f, 6.333333f,  14.025973f,
         16.92727f,   15.799998f, 10.799999f, 5.f,        59.98805f,  49.765778f, 33.074867f, 16.690908f,
         5.8f,        50.518517f, 41.444443f, 27.444445f, 14.f,       5.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_scan15_dyn_rank_vals) {
    const auto model = convert_model("scan15_dyn_rank.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{1, 3, 2}, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f});
    test_case.add_input<float>(Shape{1, 3, 2}, {1.f, 1.f, 1.f, 1.f, 1.f, 1.f});
    std::vector<float> sequence_vals{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.f,
                                     1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.f,
                                     2.1f, 2.2f, 2.3f, 2.4f, 2.5f, 2.6f, 2.7f, 2.8f, 2.9f, 3.f};
    test_case.add_input<float>(Shape{1, 3, 5, 2}, sequence_vals);  // multiply factor (reverse)
    test_case.add_input<float>(Shape{1, 5, 3, 2}, sequence_vals);  // div factor

    test_case.add_expected_output<float>(Shape{1, 3, 2},
                                         {2.7327938f, 2.1428573f, 21.070545f, 16.92727f, 49.765778f, 41.444443f});
    test_case.add_expected_output<float>(Shape{1, 3, 2},
                                         {0.40161943f, 0.5274726f, 16.80789f, 14.025973f, 59.98805f, 50.518517f});
    test_case.add_expected_output<float>(
        Shape{1, 3, 2, 5},
        {0.40161943f, 2.7327938f, 7.3076925f, 10.f,       9.f,        0.5274726f, 2.1428573f, 4.714286f,
         6.f,         5.f,        16.80789f,  21.070545f, 20.185184f, 13.851851f, 6.333333f,  14.025973f,
         16.92727f,   15.799998f, 10.799999f, 5.f,        59.98805f,  49.765778f, 33.074867f, 16.690908f,
         5.8f,        50.518517f, 41.444443f, 27.444445f, 14.f,       5.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_scan15_dyn_rank_vals_neg_axes) {
    // Negative indices for scan_input_axes and scan_output_axes attributes
    try {
        const auto model = convert_model("scan15_dyn_rank_neg_axes.onnx");
    } catch (const ov::Exception& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string("Rank must be static in order to normalize negative axis"));
    } catch (...) {
        FAIL() << "Expected exception was not thrown.";
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_scan15_ND_b4_input_rev_vals) {
    const auto model = convert_model("scan15_ND_b4_input_rev.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{4, 3, 2}, std::vector<float>(24, 0.f));
    test_case.add_input<float>(Shape{4, 3, 2}, std::vector<float>(24, 1.f));
    std::vector<float> sequence_vals{
        0.1f,  0.2f,  0.3f,  0.4f,  0.5f, 0.6f,  0.7f,  0.8f,  0.9f,  1.f,   1.1f,  1.2f,  1.3f,  1.4f,  1.5f,
        1.6f,  1.7f,  1.8f,  1.9f,  2.f,  2.1f,  2.2f,  2.3f,  2.4f,  2.5f,  2.6f,  2.7f,  2.8f,  2.9f,  3.f,
        3.1f,  3.2f,  3.3f,  3.4f,  3.5f, 3.6f,  3.7f,  3.8f,  3.9f,  4.f,   4.1f,  4.2f,  4.3f,  4.4f,  4.5f,
        4.6f,  4.7f,  4.8f,  4.9f,  5.f,  5.1f,  5.2f,  5.3f,  5.4f,  5.5f,  5.6f,  5.7f,  5.8f,  5.9f,  6.f,
        6.1f,  6.2f,  6.3f,  6.4f,  6.5f, 6.6f,  6.7f,  6.8f,  6.9f,  7.f,   7.1f,  7.2f,  7.3f,  7.4f,  7.5f,
        7.6f,  7.7f,  7.8f,  7.9f,  8.f,  8.1f,  8.2f,  8.3f,  8.4f,  8.5f,  8.6f,  8.7f,  8.8f,  8.9f,  9.f,
        9.1f,  9.2f,  9.3f,  9.4f,  9.5f, 9.6f,  9.7f,  9.8f,  9.9f,  10.f,  10.1f, 10.2f, 10.3f, 10.4f, 10.5f,
        10.6f, 10.7f, 10.8f, 10.9f, 11.f, 11.1f, 11.2f, 11.3f, 11.4f, 11.5f, 11.6f, 11.7f, 11.8f, 11.9f, 12.f};
    test_case.add_input<float>(Shape{4, 5, 3, 2}, sequence_vals);  // multiply factor (areverse)
    test_case.add_input<float>(Shape{4, 5, 3, 2}, sequence_vals);  // div factor

    test_case.add_expected_output<float>(
        Shape{4, 3, 2},
        {61.210526f, 33.2f,      23.857145f, 19.181818f, 16.373913f, 14.5f,      6.8880844f, 6.83f,
         6.7754016f, 6.7239814f, 6.6754713f, 6.6296296f, 5.9686656f, 5.953226f,  5.9382715f, 5.9237804f,
         5.9097314f, 5.896105f,  5.652082f,  5.645059f,  5.638186f,  5.6314588f, 5.624872f,  5.618421f});
    test_case.add_expected_output<float>(
        Shape{4, 3, 2},
        {6.271278f, 6.2461543f, 6.2433867f, 6.2545457f, 6.2744985f, 6.3f,       6.9531364f, 6.970527f,
         6.987378f, 7.003712f,  7.019554f,  7.034921f,  7.30868f,   7.3164845f, 7.324116f,  7.3315806f,
         7.338885f, 7.346032f,  7.485426f,  7.489783f,  7.494067f,  7.49828f,   7.5024257f, 7.506502f});
    test_case.add_expected_output<float>(
        Shape{5, 4, 3, 2},
        {25.f,       13.f,       9.f,        7.f,        5.8f,       5.f,        1.7741936f, 1.75f,      1.7272727f,
         1.7058823f, 1.6857144f, 1.6666667f, 1.3934426f, 1.3870969f, 1.3809522f, 1.375f,     1.3692307f, 1.3636364f,
         1.2637362f, 1.2608696f, 1.2580644f, 1.2553192f, 1.2526315f, 1.25f,      70.57143f,  35.f,       23.333334f,
         17.6f,      14.218181f, 12.f,       3.6739323f, 3.618421f,  3.5664334f, 3.5176468f, 3.471777f,  3.4285717f,
         2.822119f,  2.8083491f, 2.7950313f, 2.7821426f, 2.7696643f, 2.757576f,  2.543786f,  2.5377107f, 2.5317693f,
         2.5259573f, 2.520271f,  2.514706f,  95.57143f,  47.999996f, 32.333336f, 24.6f,      20.01818f,  17.f,
         5.448126f,  5.368421f,  5.293706f,  5.223529f,  5.157491f,  5.0952387f, 4.215562f,  4.195446f,  4.1759834f,
         4.1571426f, 4.138895f,  4.1212125f, 3.8075223f, 3.7985802f, 3.7898335f, 3.7812767f, 3.7729027f, 3.764706f,
         61.210526f, 33.2f,      23.857145f, 19.181818f, 16.373913f, 14.5f,      6.8880844f, 6.83f,      6.7754016f,
         6.7239814f, 6.6754713f, 6.6296296f, 5.9686656f, 5.953226f,  5.9382715f, 5.9237804f, 5.9097314f, 5.896105f,
         5.652082f,  5.645059f,  5.638186f,  5.6314588f, 5.624872f,  5.618421f,  6.271278f,  6.2461543f, 6.2433867f,
         6.2545457f, 6.2744985f, 6.3f,       6.9531364f, 6.970527f,  6.987378f,  7.003712f,  7.019554f,  7.034921f,
         7.30868f,   7.3164845f, 7.324116f,  7.3315806f, 7.338885f,  7.346032f,  7.485426f,  7.489783f,  7.494067f,
         7.49828f,   7.5024257f, 7.506502f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_scan8_ND_b4_ones) {
    const auto model = convert_model("scan8_ND_b4.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
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

OPENVINO_TEST(${BACKEND_NAME}, onnx_scan8_ND_b4_input_rev_vals) {
    const auto model = convert_model("scan8_ND_b4_input_rev.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{4, 3, 2}, std::vector<float>(24, 0.f));
    test_case.add_input<float>(Shape{4, 3, 2}, std::vector<float>(24, 1.f));
    std::vector<float> sequence_vals{
        0.1f,  0.2f,  0.3f,  0.4f,  0.5f, 0.6f,  0.7f,  0.8f,  0.9f,  1.f,   1.1f,  1.2f,  1.3f,  1.4f,  1.5f,
        1.6f,  1.7f,  1.8f,  1.9f,  2.f,  2.1f,  2.2f,  2.3f,  2.4f,  2.5f,  2.6f,  2.7f,  2.8f,  2.9f,  3.f,
        3.1f,  3.2f,  3.3f,  3.4f,  3.5f, 3.6f,  3.7f,  3.8f,  3.9f,  4.f,   4.1f,  4.2f,  4.3f,  4.4f,  4.5f,
        4.6f,  4.7f,  4.8f,  4.9f,  5.f,  5.1f,  5.2f,  5.3f,  5.4f,  5.5f,  5.6f,  5.7f,  5.8f,  5.9f,  6.f,
        6.1f,  6.2f,  6.3f,  6.4f,  6.5f, 6.6f,  6.7f,  6.8f,  6.9f,  7.f,   7.1f,  7.2f,  7.3f,  7.4f,  7.5f,
        7.6f,  7.7f,  7.8f,  7.9f,  8.f,  8.1f,  8.2f,  8.3f,  8.4f,  8.5f,  8.6f,  8.7f,  8.8f,  8.9f,  9.f,
        9.1f,  9.2f,  9.3f,  9.4f,  9.5f, 9.6f,  9.7f,  9.8f,  9.9f,  10.f,  10.1f, 10.2f, 10.3f, 10.4f, 10.5f,
        10.6f, 10.7f, 10.8f, 10.9f, 11.f, 11.1f, 11.2f, 11.3f, 11.4f, 11.5f, 11.6f, 11.7f, 11.8f, 11.9f, 12.f};
    test_case.add_input<float>(Shape{4, 5, 3, 2}, sequence_vals);  // multiply factor (reverse)
    test_case.add_input<float>(Shape{4, 5, 3, 2}, sequence_vals);  // div factor

    test_case.add_expected_output<float>(
        Shape{4, 3, 2},
        {61.210526f, 33.2f,      23.857145f, 19.181818f, 16.373913f, 14.5f,      6.8880844f, 6.83f,
         6.7754016f, 6.7239814f, 6.6754713f, 6.6296296f, 5.9686656f, 5.953226f,  5.9382715f, 5.9237804f,
         5.9097314f, 5.896105f,  5.652082f,  5.645059f,  5.638186f,  5.6314588f, 5.624872f,  5.618421f});
    test_case.add_expected_output<float>(
        Shape{4, 3, 2},
        {6.271278f, 6.2461543f, 6.2433867f, 6.2545457f, 6.2744985f, 6.3f,       6.9531364f, 6.970527f,
         6.987378f, 7.003712f,  7.019554f,  7.034921f,  7.30868f,   7.3164845f, 7.324116f,  7.3315806f,
         7.338885f, 7.346032f,  7.485426f,  7.489783f,  7.494067f,  7.49828f,   7.5024257f, 7.506502f});
    test_case.add_expected_output<float>(
        Shape{4, 5, 3, 2},
        {25.f,       13.f,       9.f,        7.f,        5.8f,       5.f,        70.57143f,  35.f,       23.333334f,
         17.6f,      14.218181f, 12.f,       95.57143f,  47.999996f, 32.333336f, 24.6f,      20.01818f,  17.f,
         61.210526f, 33.2f,      23.857145f, 19.181818f, 16.373913f, 14.5f,      6.271278f,  6.2461543f, 6.2433867f,
         6.2545457f, 6.2744985f, 6.3f,       1.7741936f, 1.75f,      1.7272727f, 1.7058823f, 1.6857144f, 1.6666667f,
         3.6739323f, 3.618421f,  3.5664334f, 3.5176468f, 3.471777f,  3.4285717f, 5.448126f,  5.368421f,  5.293706f,
         5.223529f,  5.157491f,  5.0952387f, 6.8880844f, 6.83f,      6.7754016f, 6.7239814f, 6.6754713f, 6.6296296f,
         6.9531364f, 6.970527f,  6.987378f,  7.003712f,  7.019554f,  7.034921f,  1.3934426f, 1.3870969f, 1.3809522f,
         1.375f,     1.3692307f, 1.3636364f, 2.822119f,  2.8083491f, 2.7950313f, 2.7821426f, 2.7696643f, 2.757576f,
         4.215562f,  4.195446f,  4.1759834f, 4.1571426f, 4.138895f,  4.1212125f, 5.9686656f, 5.953226f,  5.9382715f,
         5.9237804f, 5.9097314f, 5.896105f,  7.30868f,   7.3164845f, 7.324116f,  7.3315806f, 7.338885f,  7.346032f,
         1.2637362f, 1.2608696f, 1.2580644f, 1.2553192f, 1.2526315f, 1.25f,      2.543786f,  2.5377107f, 2.5317693f,
         2.5259573f, 2.520271f,  2.514706f,  3.8075223f, 3.7985802f, 3.7898335f, 3.7812767f, 3.7729027f, 3.764706f,
         5.652082f,  5.645059f,  5.638186f,  5.6314588f, 5.624872f,  5.618421f,  7.485426f,  7.489783f,  7.494067f,
         7.49828f,   7.5024257f, 7.506502f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_scan8_ND_b4_seq_lens) {
    // ONNX Scan-8 can has optional `sequence_lens` input, the input was removed since ONNX Scan-9
    try {
        const auto model = convert_model("scan8_ND_b4_seq_lens.onnx");
    } catch (const ov::Exception& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string(" ONNX Scan-8 `sequence_lens` input is not supported. "));
    } catch (...) {
        FAIL() << "Expected exception was not thrown.";
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_softsign) {
    auto model = convert_model("softsign.onnx");

    Inputs inputs{std::vector<float>{1.0f, 0.1f, 20.0f, 12.0f, -12.0f, -0.2f, 0.5f, 100.0f, 0.0f, -1.0f}};

    std::vector<float> output{0.5f,
                              0.09090909f,
                              0.95238096f,
                              0.9230769f,
                              -0.9230769f,
                              -0.16666666f,
                              0.33333334f,
                              0.990099f,
                              0.f,
                              -0.5f};

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_grid_sample) {
    const auto model = convert_model("grid_sample.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
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

OPENVINO_TEST(${BACKEND_NAME}, onnx_concat_empty_init) {
    const auto model = convert_model("concat_empty_init.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<int64_t>(Shape{2}, std::vector<int64_t>{1, 2});
    test_case.add_expected_output<int64_t>(Shape{2}, std::vector<int64_t>{1, 2});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_trilu_basic) {
    const auto model = convert_model("trilu_basic.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    // clang-format off
    test_case.add_input<float>(Shape{5, 5},
        std::vector<float>{ 1,  2,  3,  4,  5,
                            6,  7,  8,  9, 10,
                           11, 12, 13, 14, 15,
                           16, 17, 18, 19, 20,
                           21, 22, 23, 24, 25});
    test_case.add_expected_output<float>(Shape{5, 5},
        std::vector<float>{ 1,  0,  0,  0,  0,
                            6,  7,  0,  0,  0,
                           11, 12, 13,  0,  0,
                           16, 17, 18, 19,  0,
                           21, 22, 23, 24, 25});
    // clang-format on
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_trilu_lower) {
    const auto model = convert_model("trilu_lower.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    // clang-format off
    test_case.add_input<float>(Shape{4, 5},
        std::vector<float>{ 1,  2,  3,  4,  5,
                            6,  7,  8,  9, 10,
                           11, 12, 13, 14, 15,
                           16, 17, 18, 19, 20});
    test_case.add_input<int64_t>(Shape{}, {0}); // k
    test_case.add_expected_output<float>(Shape{4, 5},
        std::vector<float>{ 1,  0,  0,  0,  0,
                            6,  7,  0,  0,  0,
                           11, 12, 13,  0,  0,
                           16, 17, 18, 19,  0});
    test_case.run();

    test_case.add_input<float>(Shape{4, 5},
        std::vector<float>{ 1,  2,  3,  4,  5,
                            6,  7,  8,  9, 10,
                           11, 12, 13, 14, 15,
                           16, 17, 18, 19, 20});
    test_case.add_input<int64_t>(Shape{}, {2}); // k
    test_case.add_expected_output<float>(Shape{4, 5},
        std::vector<float>{ 1,  2,  3,  0,  0,
                            6,  7,  8,  9,  0,
                           11, 12, 13, 14, 15,
                           16, 17, 18, 19, 20});
    test_case.run();

    test_case.add_input<float>(Shape{4, 5},
        std::vector<float>{ 1,  2,  3,  4,  5,
                            6,  7,  8,  9, 10,
                           11, 12, 13, 14, 15,
                           16, 17, 18, 19, 20});
    test_case.add_input<int64_t>(Shape{}, {-2}); // k
    test_case.add_expected_output<float>(Shape{4, 5},
        std::vector<float>{ 0,  0,  0,  0,  0,
                            0,  0,  0,  0,  0,
                           11,  0,  0,  0,  0,
                           16, 17,  0,  0,  0});
    test_case.run();

    // clang-format on
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_trilu_upper) {
    const auto model = convert_model("trilu_upper.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    // clang-format off

    test_case.add_input<float>(Shape{5, 4},
        std::vector<float>{ 1,  2,  3,  4,
                            5,  6,  7,  8,
                            9, 10, 11, 12,
                           13, 14, 15, 16,
                           17, 18, 19, 20});
    test_case.add_input<int64_t>(Shape{}, {0}); // k
    test_case.add_expected_output<float>(Shape{5, 4},
        std::vector<float>{ 1,  2,  3,  4,
                            0,  6,  7,  8,
                            0,  0, 11, 12,
                            0,  0,  0, 16,
                            0,  0,  0,  0});
    test_case.run();

    test_case.add_input<float>(Shape{5, 4},
        std::vector<float>{ 1,  2,  3,  4,
                            5,  6,  7,  8,
                            9, 10, 11, 12,
                           13, 14, 15, 16,
                           17, 18, 19, 20});
    test_case.add_input<int64_t>(Shape{}, {1}); // k
    test_case.add_expected_output<float>(Shape{5, 4},
        std::vector<float>{ 0,  2,  3,  4,
                            0,  0,  7,  8,
                            0,  0,  0, 12,
                            0,  0,  0,  0,
                            0,  0,  0,  0});
    test_case.run();

    test_case.add_input<float>(Shape{5, 4},
        std::vector<float>{ 1,  2,  3,  4,
                            5,  6,  7,  8,
                            9, 10, 11, 12,
                           13, 14, 15, 16,
                           17, 18, 19, 20});
    test_case.add_input<int64_t>(Shape{}, {-1}); // k
    test_case.add_expected_output<float>(Shape{5, 4},
        std::vector<float>{ 1,  2,  3,  4,
                            5,  6,  7,  8,
                            0, 10, 11, 12,
                            0,  0, 15, 16,
                            0,  0,  0, 20});
    test_case.run();

    // clang-format on
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_trilu_upper_3d) {
    const auto model = convert_model("trilu_upper_3d.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    // clang-format off

    test_case.add_input<float>(Shape{2, 5, 4},
        std::vector<float>{ 1,  2,  3,  4,
                            5,  6,  7,  8,
                            9, 10, 11, 12,
                           13, 14, 15, 16,
                           17, 18, 19, 20,

                           21, 22, 23, 24,
                           25, 26, 27, 28,
                           29, 30, 31, 32,
                           33, 34, 35, 36,
                           37, 38, 39, 40});
    test_case.add_input<int64_t>(Shape{}, {0}); // k
    test_case.add_expected_output<float>(Shape{2, 5, 4},
        std::vector<float>{ 1,  2,  3,  4,
                            0,  6,  7,  8,
                            0,  0, 11, 12,
                            0,  0,  0, 16,
                            0,  0,  0,  0,

                           21, 22, 23, 24,
                            0, 26, 27, 28,
                            0,  0, 31, 32,
                            0,  0,  0, 36,
                            0,  0,  0,  0});
    test_case.run();

    test_case.add_input<float>(Shape{2, 5, 4},
        std::vector<float>{ 1,  2,  3,  4,
                            5,  6,  7,  8,
                            9, 10, 11, 12,
                           13, 14, 15, 16,
                           17, 18, 19, 20,

                           21, 22, 23, 24,
                           25, 26, 27, 28,
                           29, 30, 31, 32,
                           33, 34, 35, 36,
                           37, 38, 39, 40});
    test_case.add_input<int64_t>(Shape{}, {2}); // k
    test_case.add_expected_output<float>(Shape{2, 5, 4},
        std::vector<float>{ 0,  0,  3,  4,
                            0,  0,  0,  8,
                            0,  0,  0,  0,
                            0,  0,  0,  0,
                            0,  0,  0,  0,

                            0,  0, 23, 24,
                            0,  0,  0, 28,
                            0,  0,  0,  0,
                            0,  0,  0,  0,
                            0,  0,  0,  0});
    test_case.run();

    test_case.add_input<float>(Shape{2, 5, 4},
        std::vector<float>{ 1,  2,  3,  4,
                            5,  6,  7,  8,
                            9, 10, 11, 12,
                           13, 14, 15, 16,
                           17, 18, 19, 20,

                           21, 22, 23, 24,
                           25, 26, 27, 28,
                           29, 30, 31, 32,
                           33, 34, 35, 36,
                           37, 38, 39, 40});
    test_case.add_input<int64_t>(Shape{}, {-2}); // k
    test_case.add_expected_output<float>(Shape{2, 5, 4},
        std::vector<float>{ 1,  2,  3,  4,
                            5,  6,  7,  8,
                            9, 10, 11, 12,
                            0, 14, 15, 16,
                            0,  0, 19, 20,

                           21, 22, 23, 24,
                           25, 26, 27, 28,
                           29, 30, 31, 32,
                            0, 34, 35, 36,
                            0,  0, 39, 40});
    test_case.run();

    // clang-format on
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_trilu_lower_4d) {
    const auto model = convert_model("trilu_lower_4d.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    // clang-format off

    test_case.add_input<float>(Shape{2, 1, 4, 5},
        std::vector<float>{ 1,  2,  3,  4,  5,
                            6,  7,  8,  9, 10,
                           11, 12, 13, 14, 15,
                           16, 17, 18, 19, 20,

                           21, 22, 23, 24, 25,
                           26, 27, 28, 29, 30,
                           31, 32, 33, 34, 35,
                           36, 37, 38, 39, 40});
    test_case.add_input<int64_t>(Shape{}, {0}); // k
    test_case.add_expected_output<float>(Shape{2, 1, 4, 5},
        std::vector<float>{ 1,  0,  0,  0,  0,
                            6,  7,  0,  0,  0,
                           11, 12, 13,  0,  0,
                           16, 17, 18, 19,  0,

                           21,  0,  0,  0,  0,
                           26, 27,  0,  0,  0,
                           31, 32, 33,  0,  0,
                           36, 37, 38, 39,  0});
    test_case.run();

    test_case.add_input<float>(Shape{2, 1, 4, 5},
        std::vector<float>{ 1,  2,  3,  4,  5,
                            6,  7,  8,  9, 10,
                           11, 12, 13, 14, 15,
                           16, 17, 18, 19, 20,

                           21, 22, 23, 24, 25,
                           26, 27, 28, 29, 30,
                           31, 32, 33, 34, 35,
                           36, 37, 38, 39, 40});
    test_case.add_input<int64_t>(Shape{}, {1}); // k
    test_case.add_expected_output<float>(Shape{2, 1, 4, 5},
        std::vector<float>{ 1,  2,  0,  0,  0,
                            6,  7,  8,  0,  0,
                           11, 12, 13, 14,  0,
                           16, 17, 18, 19, 20,

                           21, 22,  0,  0,  0,
                           26, 27, 28,  0,  0,
                           31, 32, 33, 34,  0,
                           36, 37, 38, 39, 40});
    test_case.run();

    test_case.add_input<float>(Shape{2, 1, 4, 5},
        std::vector<float>{ 1,  2,  3,  4,  5,
                            6,  7,  8,  9, 10,
                           11, 12, 13, 14, 15,
                           16, 17, 18, 19, 20,

                           21, 22, 23, 24, 25,
                           26, 27, 28, 29, 30,
                           31, 32, 33, 34, 35,
                           36, 37, 38, 39, 40});
    test_case.add_input<int64_t>(Shape{}, {-1}); // k
    test_case.add_expected_output<float>(Shape{2, 1, 4, 5},
        std::vector<float>{ 0,  0,  0,  0,  0,
                            6,  0,  0,  0,  0,
                           11, 12,  0,  0,  0,
                           16, 17, 18,  0,  0,

                            0,  0,  0,  0,  0,
                           26,  0,  0,  0,  0,
                           31, 32,  0,  0,  0,
                           36, 37, 38,  0,  0});
    test_case.run();

    // clang-format on
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_trilu_dynamic_shapes) {
    const auto model = convert_model("dynamic_shapes/trilu_lower.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    // clang-format off

    test_case.add_input<float>(Shape{2, 1, 4, 5},
        std::vector<float>{ 1,  2,  3,  4,  5,
                            6,  7,  8,  9, 10,
                           11, 12, 13, 14, 15,
                           16, 17, 18, 19, 20,

                           21, 22, 23, 24, 25,
                           26, 27, 28, 29, 30,
                           31, 32, 33, 34, 35,
                           36, 37, 38, 39, 40});
    test_case.add_input<int64_t>(Shape{}, {1}); // k
    test_case.add_expected_output<float>(Shape{2, 1, 4, 5},
        std::vector<float>{ 1,  2,  0,  0,  0,
                            6,  7,  8,  0,  0,
                           11, 12, 13, 14,  0,
                           16, 17, 18, 19, 20,

                           21, 22,  0,  0,  0,
                           26, 27, 28,  0,  0,
                           31, 32, 33, 34,  0,
                           36, 37, 38, 39, 40});
    test_case.run();

    // clang-format on
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_is_finite) {
    const auto model = convert_model("is_finite.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    // clang-format off

    test_case.add_input<float>(Shape{1, 2, 3}, {std::nanf(""), std::numeric_limits<float>::infinity(), -0.6000f, -1.0000f, std::nanf(""), -1.0000f});

    test_case.add_expected_output<bool>(
        Shape{1, 2, 3},
        {false, false, true, true, false, true});

    test_case.run();

    // clang-format on
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_is_inf_default) {
    const auto model = convert_model("is_inf.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    // clang-format off

    test_case.add_input<float>(
        Shape{2, 2, 2},
        std::vector<float>{ std::numeric_limits<float>::infinity(), 0.0000f,
                            std::numeric_limits<float>::max(), -0.5000f,
                            -std::numeric_limits<float>::infinity(), 1.0000f,
                            std::numeric_limits<float>::min(), std::nanf("")});
    test_case.add_expected_output<bool>(
        Shape{2, 2, 2},
        std::vector<bool>{true, false,
                          false, false,
                          true, false,
                          false, false});
    test_case.run();

    // clang-format on
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_is_inf_negative_only) {
    const auto model = convert_model("is_inf_negative.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    // clang-format off

    test_case.add_input<float>(
        Shape{2, 2, 2},
        std::vector<float>{ std::numeric_limits<float>::infinity(), 0.0000f,
                            std::numeric_limits<float>::max(), -0.5000f,
                            -std::numeric_limits<float>::infinity(), 1.0000f,
                            std::numeric_limits<float>::min(), std::nanf("")});
    test_case.add_expected_output<bool>(
        Shape{2, 2, 2},
        std::vector<bool>{false, false,
                          false, false,
                          true, false,
                          false, false});
    test_case.run();

    // clang-format on
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_is_inf_positive_only) {
    const auto model = convert_model("is_inf_positive.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    // clang-format off

    test_case.add_input<float>(
        Shape{2, 2, 2},
        std::vector<float>{ std::numeric_limits<float>::infinity(), 0.0000f,
                            std::numeric_limits<float>::max(), -0.5000f,
                            -std::numeric_limits<float>::infinity(), 1.0000f,
                            std::numeric_limits<float>::min(), std::nanf("")});
    test_case.add_expected_output<bool>(
        Shape{2, 2, 2},
        std::vector<bool>{true, false,
                          false, false,
                          false, false,
                          false, false});
    test_case.run();

    // clang-format on
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_is_inf_detect_none) {
    const auto model = convert_model("is_inf_none.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    // clang-format off

    test_case.add_input<float>(
        Shape{2, 2, 2},
        std::vector<float>{ std::numeric_limits<float>::infinity(), 0.0000f,
                            std::numeric_limits<float>::max(), -0.5000f,
                            -std::numeric_limits<float>::infinity(), 1.0000f,
                            std::numeric_limits<float>::min(), std::nanf("")});
    test_case.add_expected_output<bool>(
        Shape{2, 2, 2},
        std::vector<bool>{false, false,
                          false, false,
                          false, false,
                          false, false});
    test_case.run();

    // clang-format on
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_is_nan) {
    const auto model = convert_model("is_nan.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    // clang-format off

    test_case.add_input<float>(Shape{1, 2, 3}, {std::nanf(""), std::nanf(""), -0.6000f, -1.0000f, std::nanf(""), -1.0000f});

    test_case.add_expected_output<bool>(
        Shape{1, 2, 3},
        {true, true, false, false, true, false});

    test_case.run();

    // clang-format on
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_squeeze_default_domain_opset13) {
    auto model = convert_model("squeeze_default_domain_opset13.onnx");

    auto input = ov::test::NDArray<float, 3>({{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
                     .get_vector();
    auto expected_output =
        ov::test::NDArray<float, 2>({{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input(input);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_of_shape_empty_init) {
    auto model = convert_model("constant_of_shape_empty_init.onnx");
    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output<int32_t>(Shape{}, {1});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_of_shape_null_node) {
    auto model = convert_model("constant_of_shape_null_node.onnx");
    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_expected_output<int32_t>(Shape{}, {1});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, castlike_float16_to_uint32) {
    auto model = convert_model("castlike_float16_to_uint32.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<ov::float16>(Shape{1, 1, 2, 2}, std::vector<ov::float16>{1.5f, 2.3f, 3.f, 4.f});
    test_case.add_input<uint32_t>(Shape{4}, {1, 2, 3, 4});
    test_case.add_expected_output<uint32_t>(std::vector<uint32_t>{1, 2, 3, 4});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, castlike_float16_to_int64) {
    auto model = convert_model("castlike_float16_to_int64.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<ov::float16>(Shape{1, 1, 2, 2}, std::vector<ov::float16>{1.5f, 2.3f, 3.f, 4.f});
    test_case.add_input<int64_t>(Shape{4}, {1, 2, 3, 4});
    test_case.add_expected_output<int64_t>(std::vector<int64_t>{1, 2, 3, 4});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, DISABLED_castlike_int8_to_uint16) {
    auto model = convert_model("castlike_int8_to_uint16.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<int8_t>(Shape{1, 1, 2, 2}, std::vector<int8_t>{-1, -2, 3, 4});
    test_case.add_input<uint16_t>(Shape{4}, {1, 2, 3, 4});
    test_case.add_expected_output<uint16_t>(std::vector<uint16_t>{65535, 65534, 3, 4});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, castlike_float64_to_int64) {
    auto model = convert_model("castlike_float64_to_int64.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<double>(Shape{1, 1, 2, 2}, std::vector<double>{1.5, 2.3, 3, 4});
    test_case.add_input<int64_t>(Shape{4}, {1, 2, 3, 4});
    test_case.add_expected_output<int64_t>(std::vector<int64_t>{1, 2, 3, 4});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, castlike_int8_to_float16) {
    auto model = convert_model("castlike_int8_to_float16.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<int8_t>(Shape{1, 1, 2, 2}, std::vector<int8_t>{-127, -2, 3, 4});
    test_case.add_input<ov::float16>(Shape{4}, {1, 2, 3, 4});
    test_case.add_expected_output<ov::float16>(std::vector<ov::float16>{-127.0, -2.0, 3.0, 4.0});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, castlike_int32_to_float) {
    auto model = convert_model("castlike_int32_to_float64.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<int32_t>(Shape{1, 1, 2, 2}, std::vector<int32_t>{-1, 2, 3, 4});
    test_case.add_input<float>(Shape{4}, {1, 2, 3, 4});
    test_case.add_expected_output<float>(std::vector<float>{-1.0f, 2.0f, 3.0f, 4.0f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, castlike_float64_to_int32) {
    auto model = convert_model("castlike_float64_to_int32.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(Shape{1, 1, 2, 2}, std::vector<float>{-107374.9876543f, -2.2f, 3.3f, 4.4f});
    test_case.add_input<int32_t>(Shape{4}, {1, 2, 3, 4});
    test_case.add_expected_output<int32_t>(std::vector<int32_t>{-107374, -2, 3, 4});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, DISABLED_castlike_float32_to_bfloat16) {
    auto model = convert_model("castlike_float32_to_bfloat16.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(
        Shape{3, 4},
        std::vector<float>{121.5f, 122.7f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.8f, 9.f, 10.f, 11.f, 12.f});
    test_case.add_input<bfloat16>(Shape{3, 4},
                                  {1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f, 9.5f, 10.5f, 11.5f, 12.5f});
    test_case.add_expected_output<bfloat16>(
        std::vector<bfloat16>{121.5f, 122.7f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.8f, 9.f, 10.f, 11.f, 12.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, DISABLED_castlike_bfloat16_to_float32) {
    auto model = convert_model("castlike_bfloat16_to_float32.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<bfloat16>(
        Shape{3, 4},
        std::vector<bfloat16>{121.5f, 122.7f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.8f, 9.f, 10.f, 11.f, 12.f});
    test_case.add_input<float>(Shape{3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    test_case.add_expected_output<float>(
        std::vector<float>{121.5f, 122.7f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.75f, 9.f, 10.f, 11.f, 12.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_unique_3d_default_attributes) {
    auto model = convert_model("unique_3d_default_attributes.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<int32_t>({9, 12, 3, 121, 5, 4, 10, 9});
    test_case.add_expected_output<int32_t>(Shape{7}, {3, 4, 5, 9, 10, 12, 121});
    test_case.add_expected_output<int64_t>(Shape{7}, {2, 5, 4, 0, 6, 1, 3});
    test_case.add_expected_output<int64_t>(Shape{8}, {3, 5, 0, 6, 2, 1, 4, 3});
    test_case.add_expected_output<int64_t>(Shape{7}, {1, 1, 1, 2, 1, 1, 1});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_unique_1d_no_duplicates) {
    auto model = convert_model("unique_1d_no_duplicates.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<int32_t>({5, 4, 3, 2, 1});
    test_case.add_expected_output<int32_t>(Shape{5}, {5, 4, 3, 2, 1});
    test_case.add_expected_output<int64_t>(Shape{5}, {0, 1, 2, 3, 4});
    test_case.add_expected_output<int64_t>(Shape{5}, {0, 1, 2, 3, 4});
    test_case.add_expected_output<int64_t>(Shape{5}, {1, 1, 1, 1, 1});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_unique_1d_no_duplicates_sorted) {
    auto model = convert_model("unique_1d_no_duplicates_sorted.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<int32_t>({5, 4, 3, 2, 1});
    test_case.add_expected_output<int32_t>(Shape{5}, {1, 2, 3, 4, 5});
    test_case.add_expected_output<int64_t>(Shape{5}, {4, 3, 2, 1, 0});
    test_case.add_expected_output<int64_t>(Shape{5}, {4, 3, 2, 1, 0});
    test_case.add_expected_output<int64_t>(Shape{5}, {1, 1, 1, 1, 1});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_unique_3d_with_duplicates_and_axis) {
    auto model = convert_model("unique_3d_with_duplicates_and_axis.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<int32_t>({1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6});
    test_case.add_expected_output<int32_t>(Shape{1, 2, 3}, {1, 2, 3, 4, 5, 6});
    test_case.add_expected_output<int64_t>(Shape{1}, {0});
    test_case.add_expected_output<int64_t>(Shape{2}, {0, 0});
    test_case.add_expected_output<int64_t>(Shape{1}, {2});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_unique_3d_with_duplicates_and_axis_2) {
    auto model = convert_model("unique_3d_with_duplicates_and_axis_2.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<int32_t>({-1, 2, -1, 5, -3, 5, 7, -8, 7, 4, 4, 4});
    test_case.add_expected_output<int32_t>(Shape{2, 2, 2}, {-1, 2, 5, -3, 7, -8, 4, 4});
    test_case.add_expected_output<int64_t>(Shape{2}, {0, 1});
    test_case.add_expected_output<int64_t>(Shape{3}, {0, 1, 0});
    test_case.add_expected_output<int64_t>(Shape{2}, {2, 1});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_blackmanwindow_periodic) {
    auto model = convert_model("blackmanwindow_periodic.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<int64_t>({10});
    test_case.add_expected_output<float>(Shape{10},
                                         {-0.000000014901161f,
                                          0.040212844f,
                                          0.20077012f,
                                          0.50978714f,
                                          0.8492299f,
                                          0.99999994f,
                                          0.84922975f,
                                          0.5097869f,
                                          0.20077008f,
                                          0.040212862f});

    // GPU has an accuracy drop, need to use different tolerance
    if (std::string("${BACKEND_NAME}") != std::string("IE_GPU")) {
        test_case.run_with_tolerance_as_fp();
    } else {
        test_case.run_with_tolerance_as_fp(0.01f);
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_blackmanwindow_symmetric) {
    auto model = convert_model("blackmanwindow_symmetric.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<int64_t>({10});
    test_case.add_expected_output<float>(Shape{10},
                                         {-0.00000001f,
                                          0.05086961f,
                                          0.25800052f,
                                          0.63000000f,
                                          0.95112991f,
                                          0.95112979f,
                                          0.62999994f,
                                          0.25800028f,
                                          0.05086958f,
                                          -0.00000001f});

    // GPU has an accuracy drop, need to use different tolerance
    if (std::string("${BACKEND_NAME}") != std::string("IE_GPU")) {
        test_case.run_with_tolerance_as_fp();
    } else {
        test_case.run_with_tolerance_as_fp(0.01f);
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_hammingwindow_periodic) {
    auto model = convert_model("hammingwindow_periodic.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<int64_t>({10});
    test_case.add_expected_output<float>(Shape{10},
                                         {0.08695650f,
                                          0.17414439f,
                                          0.40240526f,
                                          0.68455124f,
                                          0.91281211f,
                                          1.00000000f,
                                          0.91281211f,
                                          0.68455112f,
                                          0.40240520f,
                                          0.17414442f});

    // GPU has an accuracy drop, need to use different tolerance
    if (std::string("${BACKEND_NAME}") != std::string("IE_GPU")) {
        test_case.run_with_tolerance_as_fp();
    } else {
        test_case.run_with_tolerance_as_fp(0.01f);
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_hammingwindow_symmetric) {
    auto model = convert_model("hammingwindow_symmetric.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<int64_t>({10});
    test_case.add_expected_output<float>(Shape{10},
                                         {0.08695650f,
                                          0.19376230f,
                                          0.46420413f,
                                          0.77173913f,
                                          0.97246838f,
                                          0.97246838f,
                                          0.77173907f,
                                          0.46420389f,
                                          0.19376221f,
                                          0.08695650f});

    // GPU has an accuracy drop, need to use different tolerance
    if (std::string("${BACKEND_NAME}") != std::string("IE_GPU")) {
        test_case.run_with_tolerance_as_fp();
    } else {
        test_case.run_with_tolerance_as_fp(0.01f);
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_hannwindow_periodic) {
    auto model = convert_model("hannwindow_periodic.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<int64_t>({10});
    test_case.add_expected_output<float>(Shape{10},
                                         {0.00000000f,
                                          0.09549150f,
                                          0.34549153f,
                                          0.65450853f,
                                          0.90450847f,
                                          1.00000000f,
                                          0.90450847f,
                                          0.65450835f,
                                          0.34549144f,
                                          0.09549153f});

    // GPU has an accuracy drop, need to use different tolerance
    if (std::string("${BACKEND_NAME}") != std::string("IE_GPU")) {
        test_case.run_with_tolerance_as_fp();
    } else {
        test_case.run_with_tolerance_as_fp(0.01f);
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_hannwindow_symmetric) {
    auto model = convert_model("hannwindow_symmetric.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<int64_t>({10});
    test_case.add_expected_output<float>(Shape{10},
                                         {0.00000000f,
                                          0.11697778f,
                                          0.41317594f,
                                          0.75000000f,
                                          0.96984637f,
                                          0.96984625f,
                                          0.74999994f,
                                          0.41317570f,
                                          0.11697769f,
                                          0.00000000f});

    // GPU has an accuracy drop, need to use different tolerance
    if (std::string("${BACKEND_NAME}") != std::string("IE_GPU")) {
        test_case.run_with_tolerance_as_fp();
    } else {
        test_case.run_with_tolerance_as_fp(0.01f);
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_group_normalization_3grp_default_eps) {
    auto model = convert_model("group_normalization_3grp.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(
        {-0.2261407f, -1.8793484f,  -0.37692875f, 0.8860143f,   0.05993791f,  -0.7634332f,  0.61080337f,  0.09776749f,
         0.5835062f,  -0.32338685f, -0.23485906f, -0.04752525f, 2.4905143f,   -0.11199934f, -0.20539412f, -2.4455426f,
         -0.5437323f, 0.51794696f,  -0.44127423f, 0.09666952f,  -0.09539367f, -1.962784f,   0.25065672f,  1.5909688f,
         0.927671f,   -0.46812922f, 0.2925484f,   -1.1766007f,  0.7675745f,   -0.94145614f, 1.1552521f,   1.6375796f,
         0.0198675f,  -0.45938072f, 0.43037328f,  0.37999842f,  -0.45021877f, -0.84925014f, 1.6790043f,   -1.0172538f,
         0.0493111f,  -0.53391f,    -0.08101435f, 0.14738432f,  -0.58910686f, 0.51673824f,  -1.7001126f,  -1.888597f});
    test_case.add_input<float>({2.4556813f, 0.12371606f, 1.5681714f});
    test_case.add_input<float>({0.79260737f, -0.74518913f, 1.370796f});

    test_case.add_expected_output<float>(
        Shape{2, 6, 2, 2},
        {0.70938545f,  -4.3442307f,  0.24844825f,  4.109082f,   1.5838864f,   -0.93303996f, 3.267802f,    1.6995258f,
         -0.6843487f,  -0.7732928f,  -0.76461035f, -0.7462375f, -0.49731785f, -0.75256085f, -0.7617206f,  -0.9814244f,
         0.5922366f,   2.3495553f,   0.76182777f,  1.652246f,   1.3343381f,   -1.7566144f,  1.9071295f,   4.1256485f,
         2.4563973f,   -1.0979934f,  0.8390641f,   -2.9021082f, 2.0487132f,   -2.3033152f,  3.03593f,     4.2641716f,
         -0.73710674f, -0.80988204f, -0.6747702f,  -0.6824198f, -0.8084908f,  -0.86908495f, -0.48516175f, -0.8945968f,
         2.4475086f,   1.3245938f,   2.1965842f,   2.6363354f,  1.2183195f,   3.3474774f,   -0.92077446f, -1.2836761f});

    test_case.run_with_tolerance_as_fp(0.000001f);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_group_normalization_3grp_custom_eps) {
    auto model = convert_model("group_normalization_custom_eps.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(
        {1.8079232f,  -0.2892469f,  2.0915377f,   -1.8837914f, 0.25869793f,  0.80542284f,  2.9528935f,  0.16081251f,
         0.10507602f, -1.7271832f,  -1.0217364f,  -1.1528395f, -0.69146425f, -2.4292548f,  -0.4232518f, 0.33357796f,
         -1.4946569f, -0.08947915f, -0.7962127f,  1.3765403f,  -0.1947846f,  0.30173305f,  0.08576944f, 0.8134404f,
         0.62960416f, -1.0745901f,  -0.27037576f, -0.3607608f, 0.14347585f,  1.4590056f,   -1.1309915f, 0.88850766f,
         0.5367185f,  -0.7771955f,  0.81048864f,  0.45839247f, 1.0398412f,   -0.21019235f, -1.037122f,  -0.36852306f,
         2.7608335f,  0.3126114f,   0.336343f,    0.76919895f, 0.58595645f,  0.71894723f,  -1.2922621f, -0.542859f});
    test_case.add_input<float>({-0.05215209f, -0.5643389f, -0.6959881f});
    test_case.add_input<float>({1.4327786f, 0.01641126f, -1.471873f});

    test_case.add_expected_output<float>(
        Shape{2, 6, 2, 2},
        {1.3937842f,   1.4702199f,  1.3834473f,   1.5283363f,   1.4502488f,   1.4303224f,  1.3520534f,   1.4538165f,
         -0.628196f,   0.5758153f,  0.11225323f,  0.19840352f,  -0.10477467f, 1.0371594f,  -0.281022f,   -0.77834874f,
         -0.22489226f, -1.3969909f, -0.8074844f,  -2.6198394f,  -1.3091526f,  -1.7233121f, -1.5431708f,  -2.1501417f,
         1.3968898f,   1.4998344f,  1.4512546f,   1.4567144f,   1.4262552f,   1.3467885f,  1.5032414f,   1.3812504f,
         -0.36344206f, 0.6759755f,  -0.58001745f, -0.30147952f, -0.7614548f,  0.22742787f, 0.8815994f,   0.35268092f,
         -2.9372354f,  -1.3806448f, -1.3957335f,  -1.6709452f,  -1.5544388f,  -1.6389949f, -0.36025894f, -0.83673286f});

    test_case.run_with_tolerance_as_fp(0.000001f);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_group_normalization_2grp_custom_eps) {
    auto model = convert_model("group_normalization_2grp.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({-0.424049f, 1.7215315f,  1.429421f,   0.52730036f,  2.0628972f,  -0.15856522f,
                                2.274094f,  -1.9989003f, -1.7827071f, -0.87104136f, -1.2995626f, 0.16800839f,
                                0.5934625f, 1.553442f,   -0.5482905f, 0.6079124f,   0.3598974f,  -0.15221423f,
                                1.1135519f, -1.2533926f, -1.019778f,  -1.9142767f,  -1.2984604f, 0.5587884f});
    test_case.add_input<float>({-1.4678609f, -1.8223071f});
    test_case.add_input<float>({1.1155374f, -0.6101201f});

    test_case.add_expected_output<float>(
        Shape{1, 4, 2, 3},
        {1.694167f,   -0.51719165f, -0.21612573f, 0.71365166f, -0.86902285f, 1.4205441f, -1.0866947f, 3.3172996f,
         3.0944781f,  2.154863f,    2.5965219f,   1.0839586f,  -1.8562672f,  -3.540983f, 0.14745194f, -1.8816261f,
         -1.4463723f, -0.547642f,   -2.768998f,   1.3848708f,  0.97488886f,  2.5446892f, 1.4639623f,  -1.7954159f});

    test_case.run_with_tolerance_as_fp(0.000001f);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mm_nms_rotated) {
    auto model = convert_model("mm_nms_rotated.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input(Shape{1, 4, 5},
                        std::vector<float>({23.0f, 10.5f, 4.0f, 15.0f, 2.5f,  11.0f, 15.0f, 4.0f, 2.0f, 0.7854f,
                                            20.0f, 4.5f,  4.0f, 3.0f,  -5.3f, 8.0f,  11.5f, 4.0f, 3.0f, -0.5236f}));
    test_case.add_input(Shape{1, 1, 4}, std::vector<float>({0.6f, 0.8f, 0.5f, 0.7f}));
    test_case.add_expected_output<int64_t>(Shape{4, 3}, {0, 0, 1, 0, 0, 3, 0, 0, 0, 0, 0, 2});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_less_or_equal) {
    auto model = convert_model("less_or_equal.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{5}, {1., 2., 3., 4., 5.});
    test_case.add_input<float>(Shape{5}, {3., 3., 3., 3., 3.});
    test_case.add_expected_output<bool>(Shape{5}, {true, true, true, false, false});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_less_or_equal_broadcast) {
    auto model = convert_model("less_or_equal_broadcast.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{5}, {1., 2., 3., 4., 5.});
    test_case.add_input<float>(Shape{1}, {3.});
    test_case.add_expected_output<bool>(Shape{5}, {true, true, true, false, false});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_greater_or_equal_int) {
    auto model = convert_model("greater_or_equal_int.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<int64_t>(Shape{2}, {10, 20});
    test_case.add_input<int64_t>(Shape{2}, {15, 15});
    test_case.add_expected_output<bool>(Shape{2}, {false, true});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_greater_or_equal_float) {
    auto model = convert_model("greater_or_equal_float.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(Shape{2}, {12.03513f, 22.03513f});
    test_case.add_input<float>(Shape{2}, {5.84916f, 22.03513f});
    test_case.add_expected_output<bool>(Shape{2}, {true, true});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_bitwise_and) {
    auto model = convert_model("bitwise_and.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<int>(Shape{5}, {1, 2, 3, 4, 5});
    test_case.add_input<int>(Shape{5}, {5, 5, 5, 5, 5});
    test_case.add_expected_output<int>(Shape{5}, {1, 0, 1, 4, 5});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_bitwise_and_broadcast_condition) {
    auto model = convert_model("bitwise_and_broadcast_condition.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<int>(Shape{5}, {1, 2, 3, 4, 5});
    test_case.add_input<int>(Shape{1}, {4});
    test_case.add_expected_output<int>(Shape{5}, {0, 0, 0, 4, 4});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_bitwise_or) {
    auto model = convert_model("bitwise_or.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<int>(Shape{5}, {1, 2, 3, 4, 5});
    test_case.add_input<int>(Shape{5}, {5, 5, 5, 5, 5});
    test_case.add_expected_output<int>(Shape{5}, {5, 7, 7, 5, 5});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_bitwise_or_broadcast_condition) {
    auto model = convert_model("bitwise_or_broadcast_condition.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<int>(Shape{5}, {1, 2, 3, 4, 5});
    test_case.add_input<int>(Shape{1}, {4});
    test_case.add_expected_output<int>(Shape{5}, {5, 6, 7, 4, 5});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_max_roi_pool_region_divisible_by_output_shape) {
    auto model = convert_model("max_roi_pool_divisible.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{1, 1, 5, 5}, {1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12., 13.,
                                                   14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.});
    test_case.add_input<float>({0, 0, 0, 3, 3});
    test_case.add_expected_output<float>(Shape{1, 1, 2, 2}, {7., 9., 17., 19.});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_max_roi_pool_region_not_divisible_by_output_shape) {
    auto model = convert_model("max_roi_pool_non_divisible.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{1, 1, 5, 5}, {1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12., 13.,
                                                   14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.});
    test_case.add_input<float>({0, 0, 0, 4, 4});
    test_case.add_expected_output<float>(Shape{1, 1, 2, 2}, {13., 15., 23., 25.});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_max_roi_pool_with_spatial_scale) {
    auto model = convert_model("max_roi_pool_spatial_scale.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{1, 1, 5, 5}, {1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12., 13.,
                                                   14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.});
    test_case.add_input<float>({0, 0, 0, 2, 2});
    test_case.add_expected_output<float>(Shape{1, 1, 2, 2}, {1., 2., 6., 7.});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_bitwise_xor) {
    auto model = convert_model("bitwise_xor.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<int>(Shape{5}, {1, 2, 3, 4, 5});
    test_case.add_input<int>(Shape{5}, {5, 5, 5, 5, 5});
    test_case.add_expected_output<int>(Shape{5}, {4, 7, 6, 1, 0});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_bitwise_xor_broadcast_condition) {
    auto model = convert_model("bitwise_xor_broadcast_condition.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<int>(Shape{5}, {1, 2, 3, 4, 5});
    test_case.add_input<int>(Shape{1}, {4});
    test_case.add_expected_output<int>(Shape{5}, {5, 6, 7, 0, 1});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_bitwise_not) {
    auto model = convert_model("bitwise_not.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<int64_t>(Shape{5}, {5, 10, 200, 35, 1});
    test_case.add_expected_output<int64_t>(Shape{5}, {-6, -11, -201, -36, -2});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_celu_float) {
    auto model = convert_model("celu_float.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{2}, {-45.f, 22.98f});
    test_case.add_expected_output<float>(Shape{2}, {-1.f, 22.98f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_celu_float_alpha) {
    auto model = convert_model("celu_float_alpha.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{4}, {-5.f, -4.25f, -10.f, 7.3f});
    test_case.add_expected_output<float>(Shape{4}, {-2.43337319f, -2.27243678f, -2.89297802f, 7.3f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_gelu_float) {
    auto model = convert_model("gelu_float.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{2}, {-16.13f, 24.33f});
    test_case.add_expected_output<float>(Shape{2}, {0.0f, 24.33f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_gelu_float_none) {
    auto model = convert_model("gelu_float_none.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{2}, {-16.13f, 24.33f});
    test_case.add_expected_output<float>(Shape{2}, {0.0f, 24.33f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_gelu_float_tanh) {
    auto model = convert_model("gelu_float_tanh.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{2}, {-0.5f, 24.33f});
    test_case.add_expected_output<float>(Shape{2}, {-0.15428598f, 24.f});
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mish_activation) {
    auto model = convert_model("mish.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({1.8079f, -0.2892f, 2.0915f,  12.5101f, -1.8837f, 0.2586f, 2.9528f,  0.001f,
                                6.0296f, -1.0745f, -0.2703f, 1.319f,   -3.3607f, 0.1434f, -8.4590f, 0.0f,
                                2.7608f, 0.3126f,  0.3f,     3.0f,     7.6919f,  0.5859f, -11.992f, -37.8f});

    test_case.add_expected_output<float>({1.737521f,  -0.146684f, 2.041557f,  12.5101f,   -0.264820f, 0.176079f,
                                          2.938304f,  0.0006f,    6.029531f,  -0.306873f, -0.138725f, 1.206575f,
                                          -0.114629f, 0.092553f,  -0.001792f, 0.0f,       2.741334f,  0.217909f,
                                          0.208001f,  2.986535f,  7.691896f,  0.453058f,  -0.000074f, 0.0f});

    test_case.run_with_tolerance_as_fp(0.000001f);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mmdeploy_roi_align_rotated) {
    float eps = 0.0001f;

    if (std::string("${BACKEND_NAME}") == std::string("IE_GPU")) {
        eps = 0.01f;
    }

    auto model = convert_model("mmdeploy_roi_align_rotated.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{1, 1, 5, 5}, {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,
                                                   10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
                                                   19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f});

    test_case.add_input<float>(Shape{1, 6}, {0.0f, 3.0f, 1.0f, 4.0f, 2.0f, 1.0471975512f});

    test_case.add_expected_output<float>(
        Shape{1, 1, 5, 2},
        {5.1271f, 1.2473f, 6.1773f, 2.9598f, 7.2275f, 3.2300f, 8.2777f, 3.7458f, 9.3279f, 4.4060f});

    test_case.run_with_tolerance_as_fp(eps);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_min_18) {
    // Credit: This function is a modified version of PR 23475's test function.
    // TEMPLATE plugin has an issue with evaluation for u8 type
    if (std::string("${BACKEND_NAME}") == std::string("INTERPRETER")) {
        GTEST_SKIP();
    }

    auto model = convert_model("reduce_min_18.onnx");

    // input data shape (1, 1, 4, 4)
    std::vector<std::vector<uint8_t>> inputs{
        ov::test::NDArray<uint8_t, 4>({{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}}})
            .get_vector()};

    // output data shape (1,)
    auto expected_output = ov::test::NDArray<uint8_t, 1>({1, 2, 3, 4}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_min_20_boolean) {
    // Credit: This function is a modified version of PR 23475's test function.
    // TEMPLATE plugin has an issue with evaluation for u8 type
    if (std::string("${BACKEND_NAME}") == std::string("INTERPRETER")) {
        GTEST_SKIP();
    }

    auto model = convert_model("reduce_min_20.onnx");

    // input data shape (1, 1, 4, 4)
    std::vector<std::vector<bool>> inputs{ov::test::NDArray<bool, 4>({{{{true, true, false, false},
                                                                        {true, false, false, false},
                                                                        {true, false, false, false},
                                                                        {true, true, false, false}}}})
                                              .get_vector()};

    // output data shape (1,)
    auto expected_output = ov::test::NDArray<uint8_t, 1>({1, 0, 0, 0}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_string_input) {
    const auto model = convert_model("string_input.onnx");
    auto test_case = test::TestCase(model);
    test_case.add_input<std::string>({"strinpt1", "strinpt2"});
    test_case.add_expected_output<int64_t>({2});
    test_case.add_expected_output<std::string>({"strinpt1", "strinpt2"});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_string_constant) {
    const auto model = convert_model("string_constant.onnx");
    auto test_case = test::TestCase(model);
    test_case.add_expected_output<int64_t>({2});
    test_case.add_expected_output<std::string>({"string1", "string2"});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_multinomial_7) {
    auto model = convert_model("multinomial.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    auto expected_shape = Shape{3, 5};
    EXPECT_EQ(model->get_output_shape(0), expected_shape);

    std::vector<float> input_values = {0.1f, 0.2f, 0.7f, 0.2f, 0.4f, 0.4f, 1.0f, 0.0f, 0.0f};
    test_case.add_input<float>(ov::Shape{3, 3}, input_values);

    // Values are collected for seed 1.23
    if (std::string("${BACKEND_NAME}") == std::string("INTERPRETER")) {
        test_case.add_expected_output<int32_t>(Shape{3, 5}, {0, 2, 0, 1, 1, 2, 1, 2, 1, 2, 0, 1, 0, 0, 0});
    } else if (std::string("${BACKEND_NAME}") == std::string("IE_CPU")) {
        test_case.add_expected_output<int32_t>(Shape{3, 5}, {2, 2, 2, 2, 0, 2, 2, 2, 0, 0, 0, 1, 0, 1, 0});
    } else if (std::string("${BACKEND_NAME}") == std::string("IE_GPU")) {
        test_case.add_expected_output<int32_t>(Shape{3, 5}, {1, 0, 0, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0});
    } else {
        GTEST_FAIL();
    }

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_float8e5m2_input) {
    const auto model = convert_model("float8e5m2_input.onnx");
    auto test_case = test::TestCase(model);
    test_case.add_input<ov::float8_e5m2>({1.0f, 0.0f, -1.0f, NAN, -INFINITY, INFINITY});
    test_case.add_expected_output<int64_t>({6});
    test_case.add_expected_output<ov::float8_e5m2>({1.0f, 0.0f, -1.0f, NAN, -INFINITY, INFINITY});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_float8e5m2_constant) {
    const auto model = convert_model("float8e5m2_constant.onnx");
    auto test_case = test::TestCase(model);
    test_case.add_expected_output<int64_t>({6});
    test_case.add_expected_output<ov::float8_e5m2>({-1.0f, 0.0f, 1.0f, NAN, INFINITY, -INFINITY});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_float8e4m3fn_input) {
    const auto model = convert_model("float8e4m3fn_input.onnx");
    auto test_case = test::TestCase(model);
    test_case.add_input<ov::float8_e4m3>({1.0f, 0.0f, -1.0f, NAN, -256.f, 256.f});
    test_case.add_expected_output<int64_t>({6});
    // Float8e4m3(fn) doesn't have infinity/-infinity values
    test_case.add_expected_output<ov::float8_e4m3>({1.0f, 0.0f, -1.0f, NAN, -256.f, 256.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_float8e4m3fn_constant) {
    const auto model = convert_model("float8e4m3fn_constant.onnx");
    auto test_case = test::TestCase(model);
    test_case.add_expected_output<int64_t>({6});
    // Float8e4m3(fn) doesn't have infinity/-infinity values
    test_case.add_expected_output<ov::float8_e4m3>({-1.0f, 0.0f, 1.0f, NAN, 256.f, -256.f});

    test_case.run();
}