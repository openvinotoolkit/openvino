// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif
#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include <functional>
#include <iterator>
#include <numeric>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_case.hpp"
#include "common_test_utils/test_control.hpp"
#include "common_test_utils/test_tools.hpp"
#include "common_test_utils/type_prop.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"
#include "openvino/op/constant.hpp"

using namespace ov;
using namespace ov::frontend::onnx::tests;

static std::string s_manifest = onnx_backend_manifest("${MANIFEST}");
static std::string s_device = backend_name_to_device("${BACKEND_NAME}");

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_onnx_dynamic_dims_to_ov_dynamic_dims) {
    // the model represents a linear model A * x + B
    // where all 3 operands are model inputs (no initializers)
    const auto model = convert_model("dynamic_shapes/ab_plus_c.onnx");

    const auto& graph_inputs = model->get_parameters();
    EXPECT_EQ(graph_inputs.size(), 3);

    // all inputs in the model have a 2D partial shape {?, 2}
    for (const auto& input : graph_inputs) {
        const auto& input_ps = input->get_partial_shape();
        EXPECT_TRUE(input_ps.is_dynamic());

        ASSERT_TRUE(input_ps.rank().is_static());
        EXPECT_EQ(input_ps.rank().get_length(), 2);

        EXPECT_TRUE(input_ps[0].is_dynamic());
        ASSERT_TRUE(input_ps[1].is_static());
        EXPECT_EQ(input_ps[1].get_length(), 2);
    }

    const auto& graph_outputs = model->get_results();
    EXPECT_EQ(graph_outputs.size(), 1);

    const auto out = *(graph_outputs.cbegin());
    const auto& out_ps = out->get_output_partial_shape(0);
    ASSERT_TRUE(out_ps.rank().is_static());
    EXPECT_EQ(out_ps.rank().get_length(), 2);

    EXPECT_TRUE(out_ps[0].is_dynamic());
    ASSERT_TRUE(out_ps[1].is_static());
    EXPECT_EQ(out_ps[1].get_length(), 2);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_ab_plus_c_inference) {
    const auto model = convert_model("dynamic_shapes/ab_plus_c.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    struct ExpectedValuesGenerator {
        int64_t i = 1;
        int64_t operator()() {
            const auto ret = i * i + i;
            ++i;
            return ret;
        }
    };

    const size_t NUM_BATCHES_TO_TEST = 5;

    for (size_t batch = 1; batch <= NUM_BATCHES_TO_TEST; ++batch) {
        const Shape shape{batch, 2};
        const auto elems_in_tensor = shape_size(shape);

        std::vector<int64_t> input_values(elems_in_tensor);
        std::iota(input_values.begin(), input_values.end(), 1);

        test_case.add_input<int64_t>(shape, input_values);
        test_case.add_input<int64_t>(shape, input_values);
        test_case.add_input<int64_t>(shape, input_values);

        std::vector<int64_t> expected_values(elems_in_tensor);
        std::generate(expected_values.begin(), expected_values.end(), ExpectedValuesGenerator{});
        test_case.add_expected_output<int64_t>(shape, expected_values);

        test_case.run();
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_scalar_initializers_shape_check) {
    // initializers defined witout the "dims" field should produce Constants with an empty Shape
    // initializers with "dims: 0" should be have the same way (Shape{} not Shape{0})
    const auto model = convert_model("dynamic_shapes/scalar_initializers.onnx");

    for (auto ng_node : model->get_ordered_ops()) {
        if (as_type_ptr<op::v0::Constant>(ng_node)) {
            EXPECT_EQ(ng_node->get_shape(), Shape{});
        }
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_dynamic_rank_input_check) {
    // the model contains a single Add operation that takes a fully dynamic input and a scalar
    const auto model = convert_model("dynamic_shapes/a_plus_b_dyn_rank.onnx");

    const auto& graph_inputs = model->get_parameters();
    ASSERT_EQ(graph_inputs.size(), 2);

    const auto dyn_rank_input = graph_inputs[0];
    const auto scalar_input = graph_inputs[1];

    EXPECT_TRUE(dyn_rank_input->get_partial_shape().rank().is_dynamic());

    ASSERT_TRUE(scalar_input->get_partial_shape().is_static());
    EXPECT_EQ(scalar_input->get_partial_shape().to_shape(), Shape{});

    const auto& graph_outputs = model->get_results();
    EXPECT_EQ(graph_outputs.size(), 1);

    const auto out = *(graph_outputs.cbegin());
    EXPECT_TRUE(out->get_output_partial_shape(0).rank().is_dynamic());
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_dynamic_rank_input_inference) {
    // the model contains a single Add operation that takes a fully dynamic input and a scalar
    const auto model = convert_model("dynamic_shapes/a_plus_b_dyn_rank.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    const size_t RANKS_TO_TEST = 3;
    const int64_t SCALAR_INPUT_VAL = 5;

    for (size_t r = 0; r <= RANKS_TO_TEST; ++r) {
        const Shape shape(r, 2);
        const auto elems_in_tensor = shape_size(shape);

        std::vector<int64_t> input_values(elems_in_tensor);
        std::iota(input_values.begin(), input_values.end(), 1);

        test_case.add_input<int64_t>(shape, input_values);
        test_case.add_input<int64_t>(Shape{}, {SCALAR_INPUT_VAL});

        std::vector<int64_t> expected_values(elems_in_tensor);
        std::iota(expected_values.begin(), expected_values.end(), SCALAR_INPUT_VAL + 1);
        test_case.add_expected_output<int64_t>(shape, expected_values);

        test_case.run();
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_model_acosh_1_3) {
    auto model = convert_model("dynamic_shapes/acosh_dyn_shape.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{1, 3}, {1.0f, 2.5f, 4.3f});
    test_case.add_expected_output<float>(Shape{1, 3}, {0.0f, 1.5667993f, 2.1379586f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_model_acosh_3_2) {
    auto model = convert_model("dynamic_shapes/acosh_dyn_shape.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{3, 2}, {1.0f, 2.5f, 4.3f, 1.0f, 2.5f, 4.3f});
    test_case.add_expected_output<float>(Shape{3, 2}, {0.0f, 1.5667993f, 2.1379586f, 0.0f, 1.5667993f, 2.1379586f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_model_asinh_1_3) {
    auto model = convert_model("dynamic_shapes/asinh_dyn_shape.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{1, 3}, {-1.5f, 0.0f, 1.5f});
    test_case.add_expected_output<float>(Shape{1, 3}, {-1.1947632f, 0.0f, 1.1947632f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_model_asinh_3_2) {
    auto model = convert_model("dynamic_shapes/asinh_dyn_shape.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{3, 2}, {-1.5f, 0.0f, 1.5f, -1.5f, 0.0f, 1.5f});
    test_case.add_expected_output<float>(Shape{3, 2}, {-1.1947632f, 0.0f, 1.1947632f, -1.1947632f, 0.0f, 1.1947632f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_model_atanh_1_3) {
    auto model = convert_model("dynamic_shapes/atanh_dyn_shape.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{1, 3}, {-0.9f, 0.0f, 0.9f});
    test_case.add_expected_output<float>(Shape{1, 3}, {-1.47221948f, 0.0f, 1.47221948f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_model_atanh_3_2) {
    auto model = convert_model("dynamic_shapes/atanh_dyn_shape.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{3, 2}, {-0.9f, 0.0f, 0.9f, -0.9f, 0.0f, 0.9f});
    test_case.add_expected_output<float>(Shape{3, 2},
                                         {-1.47221948f, 0.0f, 1.47221948f, -1.47221948f, 0.0f, 1.47221948f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_model_conv_with_dynamic_batch) {
    const auto model = convert_model("dynamic_shapes/conv_with_dynamic_batch.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    const auto data_shape = Shape{1, 3, 7, 7};
    const auto filters_shape = Shape{10, 3, 2, 2};
    const auto data_elems = shape_size(data_shape);
    const auto filters_elems = shape_size(filters_shape);

    test_case.add_input<int64_t>(data_shape, std::vector<int64_t>(data_elems, 1));
    test_case.add_input<int64_t>(filters_shape, std::vector<int64_t>(filters_elems, 1));
    test_case.add_input<int64_t>(Shape{10}, std::vector<int64_t>(10, 1));

    const auto expected_out_shape = Shape{1, 10, 6, 6};
    const std::vector<int64_t> expected_values(shape_size(expected_out_shape), 13);
    test_case.add_expected_output<int64_t>(expected_out_shape, expected_values);

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_model_conv_with_dynamic_bias) {
    const auto model = convert_model("dynamic_shapes/conv_with_dynamic_bias.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    const auto data_shape = Shape{1, 3, 7, 7};
    const auto filters_shape = Shape{10, 3, 2, 2};
    const auto data_elems = shape_size(data_shape);
    const auto filters_elems = shape_size(filters_shape);

    test_case.add_input(data_shape, std::vector<int64_t>(data_elems, 1));
    test_case.add_input(filters_shape, std::vector<int64_t>(filters_elems, 1));
    test_case.add_input(Shape{10}, std::vector<int64_t>(10, 1));

    const auto expected_out_shape = Shape{1, 10, 6, 6};
    const std::vector<int64_t> expected_values(shape_size(expected_out_shape), 13);
    test_case.add_expected_output(expected_out_shape, expected_values);

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_avg_pool_dyn_shape) {
    const auto model = convert_model("dynamic_shapes/average_pool_2d_dyn.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    const Shape shape{1, 1, 4, 4};
    const auto elems_in_tensor = shape_size(shape);
    std::vector<float> input_values(elems_in_tensor);
    std::iota(input_values.begin(), input_values.end(), 0.f);

    test_case.add_input<float>(shape, input_values);

    std::vector<float> expected_values{2.5f, 4.5f, 10.5f, 12.5f};
    test_case.add_expected_output<float>(Shape{1, 1, 2, 2}, expected_values);

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_max_pool_dyn_shape) {
    const auto model = convert_model("dynamic_shapes/max_pool_2d_dyn.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    const Shape shape{1, 1, 4, 4};
    const auto elems_in_tensor = shape_size(shape);
    std::vector<float> input_values(elems_in_tensor);
    std::iota(input_values.begin(), input_values.end(), 0.f);

    test_case.add_input<float>(shape, input_values);

    std::vector<float> expected_values{0.f, 2.f, 3.f, 8.f, 10.f, 11.f, 12.f, 14.f, 15.f};
    test_case.add_expected_output<float>(Shape{1, 1, 3, 3}, expected_values);

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_max_pool_with_indices_output) {
    const auto model = convert_model("dynamic_shapes/max_pool_with_indices_output.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    const Shape shape{1, 1, 5, 5};
    std::vector<float> input_values(shape_size(shape));
    std::iota(input_values.begin(), input_values.end(), 1.f);

    test_case.add_input<float>(shape, input_values);

    std::vector<float> expected_values{13.f, 14.f, 15.f, 15.f, 15.f, 18.f, 19.f, 20.f, 20.f, 20.f, 23.f, 24.f, 25.f,
                                       25.f, 25.f, 23.f, 24.f, 25.f, 25.f, 25.f, 23.f, 24.f, 25.f, 25.f, 25.f};
    test_case.add_expected_output<float>(Shape{1, 1, 5, 5}, expected_values);

    std::vector<int64_t> expected_indices{12, 13, 14, 14, 14, 17, 18, 19, 19, 19, 22, 23, 24,
                                          24, 24, 22, 23, 24, 24, 24, 22, 23, 24, 24, 24};
    test_case.add_expected_output<int64_t>(Shape{1, 1, 5, 5}, expected_indices);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_global_avg_pool_dyn_shape) {
    const auto model = convert_model("dynamic_shapes/global_average_pool_dyn.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    const Shape shape{1, 3, 5, 5};
    const auto elems_in_tensor = shape_size(shape);
    std::vector<float> input_values(elems_in_tensor);
    std::iota(input_values.begin(), input_values.end(), 0.f);

    test_case.add_input<float>(shape, input_values);

    std::vector<float> expected_values{12.f, 37.f, 62.f};
    test_case.add_expected_output<float>(Shape{1, 3, 1, 1}, expected_values);

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_global_max_pool_dyn_shape) {
    const auto model = convert_model("dynamic_shapes/global_max_pool_dyn.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    const Shape shape{1, 3, 5, 5};
    const auto elems_in_tensor = shape_size(shape);
    std::vector<float> input_values(elems_in_tensor);
    std::iota(input_values.begin(), input_values.end(), 0.f);

    test_case.add_input<float>(shape, input_values);

    std::vector<float> expected_values{24.f, 49.f, 74.f};
    test_case.add_expected_output<float>(Shape{1, 3, 1, 1}, expected_values);

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_arg_max_dyn_shape) {
    const auto model = convert_model("dynamic_shapes/argmax_dyn.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    const Shape shape{3, 2, 2};
    const auto elems_in_tensor = shape_size(shape);
    std::vector<int32_t> input_values(elems_in_tensor);
    std::iota(input_values.begin(), input_values.end(), 1);

    test_case.add_input<int32_t>(shape, input_values);

    std::vector<int64_t> expected_values{1, 1, 1, 1, 1, 1};
    test_case.add_expected_output<int64_t>(Shape{3, 1, 2}, expected_values);

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_arg_min_no_keep_dims_dyn_shape) {
    const auto model = convert_model("dynamic_shapes/argmin_no_keep_dims_dyn.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    const Shape shape{3, 2, 2};
    const auto elems_in_tensor = shape_size(shape);
    std::vector<int32_t> input_values(elems_in_tensor);
    std::iota(input_values.begin(), input_values.end(), 1);

    test_case.add_input<int32_t>(shape, input_values);

    std::vector<int64_t> expected_values{0, 0, 0, 0};
    test_case.add_expected_output<int64_t>(Shape{2, 2}, expected_values);

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_constant_of_shape_float_zeros) {
    auto model = convert_model("dynamic_shapes/constant_of_shape_float_zeros.onnx");

    std::vector<float> expected_values(24, 0);

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<int64_t>(Shape{3}, std::vector<int64_t>{2, 3, 4});
    test_case.add_expected_output<float>(Shape{2, 3, 4}, expected_values);

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_constant_of_shape_int_ones) {
    auto model = convert_model("dynamic_shapes/constant_of_shape_int_ones.onnx");

    std::vector<int32_t> expected_values(6, 1);

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<int64_t>(Shape{2}, std::vector<int64_t>{2, 3});
    test_case.add_expected_output<int32_t>(Shape{2, 3}, expected_values);

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_expand_1_dyn_shape) {
    const auto model = convert_model("dynamic_shapes/expand_dyn.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(Shape{3, 1}, std::vector<float>{1.f, 2.f, 3.f});
    test_case.add_input<int64_t>(Shape{3}, std::vector<int64_t>{2, 1, 6});

    std::vector<float> expected_values{1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f,
                                       3.f, 3.f, 3.f, 3.f, 3.f, 3.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
                                       2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 3.f, 3.f, 3.f, 3.f, 3.f, 3.f};
    test_case.add_expected_output<float>(Shape{2, 3, 6}, expected_values);

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_expand_2_dyn_shape) {
    const auto model = convert_model("dynamic_shapes/expand_dyn.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(Shape{3, 1}, std::vector<float>{1.f, 2.f, 3.f});
    test_case.add_input<int64_t>(Shape{3}, std::vector<int64_t>{2, 3, 4});

    std::vector<float> expected_values{1.f, 1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 2.f, 3.f, 3.f, 3.f, 3.f,
                                       1.f, 1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 2.f, 3.f, 3.f, 3.f, 3.f};
    test_case.add_expected_output<float>(Shape{2, 3, 4}, expected_values);

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_expand_3_dyn_shape) {
    const auto model = convert_model("dynamic_shapes/expand_dyn.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(Shape{2, 1}, std::vector<float>{4.f, 5.f});
    test_case.add_input<int64_t>(Shape{2}, std::vector<int64_t>{2, 4});

    std::vector<float> expected_values{4.f, 4.f, 4.f, 4.f, 5.f, 5.f, 5.f, 5.f};
    test_case.add_expected_output<float>(Shape{2, 4}, expected_values);

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_expand_4_dyn_shape) {
    const auto model = convert_model("dynamic_shapes/expand_dyn.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(Shape{1, 3, 1}, std::vector<float>{7.f, 8.f, 9.f});
    test_case.add_input<int64_t>(Shape{2}, std::vector<int64_t>{3, 1});

    std::vector<float> expected_values{7.f, 8.f, 9.f};
    test_case.add_expected_output<float>(Shape{1, 3, 1}, expected_values);

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_expand_5_dyn_shape) {
    const auto model = convert_model("dynamic_shapes/expand_dyn.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(Shape{1, 4, 1}, std::vector<float>{7.f, 8.f, 9.f, 10.f});
    test_case.add_input<int64_t>(Shape{2}, std::vector<int64_t>{1, 4});

    std::vector<float>
        expected_values{7.f, 7.f, 7.f, 7.f, 8.f, 8.f, 8.f, 8.f, 9.f, 9.f, 9.f, 9.f, 10.f, 10.f, 10.f, 10.f};
    test_case.add_expected_output<float>(Shape{1, 4, 4}, expected_values);

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_expand_6_dyn_shape) {
    const auto model = convert_model("dynamic_shapes/expand_dyn.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(Shape{1, 3, 1}, std::vector<float>{7.f, 8.f, 9.f});
    test_case.add_input<int64_t>(Shape{3}, std::vector<int64_t>{3, 1, 3});

    std::vector<float> expected_values{7.f, 7.f, 7.f, 8.f, 8.f, 8.f, 9.f, 9.f, 9.f, 7.f, 7.f, 7.f, 8.f, 8.f,
                                       8.f, 9.f, 9.f, 9.f, 7.f, 7.f, 7.f, 8.f, 8.f, 8.f, 9.f, 9.f, 9.f};
    test_case.add_expected_output<float>(Shape{3, 3, 3}, expected_values);

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_expand_uint16_dyn_shape) {
    const auto model = convert_model("dynamic_shapes/expand_uint16_dyn.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<uint16_t>(Shape{1, 2, 1}, std::vector<uint16_t>{1, 2});
    test_case.add_input<int64_t>(Shape{4}, std::vector<int64_t>{2, 2, 1, 2});

    std::vector<uint16_t> expected_values{1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2};
    test_case.add_expected_output<uint16_t>(Shape{2, 2, 2, 2}, expected_values);

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_model_tile) {
    auto model = convert_model("tile.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<std::int16_t>({0, 1, 2, 3, 4, 5});  // input
    test_case.add_input<std::int64_t>({2, 1});              // repeats
    test_case.add_expected_output<std::int16_t>(Shape{4, 3}, {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_model_tile_static) {
    auto model = convert_model("tile_static.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<std::int16_t>({0, 1, 2, 3, 4, 5});  // input
    test_case.add_expected_output<std::int16_t>(Shape{4, 6}, {0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5,
                                                              0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_model_convtranspose_dyn_data) {
    auto ct_fn = convert_model("convtranspose_dyn_data.onnx");

    auto test_case = ov::test::TestCase(ct_fn);

    // data
    test_case.add_input<float>(
        Shape{1, 2, 3, 3},
        {0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f});

    // filters
    test_case.add_input<float>({0.f,  0.2f, 0.4f, 0.6f, 0.8f, 1.f,  1.2f, 1.4f, 1.6f, 1.8f, 2.f,  2.2f,
                                2.4f, 2.6f, 2.8f, 3.f,  3.2f, 3.4f, 3.6f, 3.8f, 4.f,  4.2f, 4.4f, 4.6f,
                                4.8f, 5.f,  5.2f, 5.4f, 5.6f, 5.8f, 6.f,  6.2f, 6.4f, 6.6f, 6.8f, 7.f});

    // bias
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});

    // output
    test_case.add_expected_output<float>(Shape{1, 4, 2, 2},
                                         {1.4000001f,
                                          1.52f,
                                          2.6799998f,
                                          2.6799998f,
                                          5.1000004f,
                                          4.6800003f,
                                          10.16f,
                                          8.539999f,
                                          30.939999f,
                                          22.96f,
                                          53.28f,
                                          38.7f,
                                          44.36f,
                                          32.6f,
                                          75.340004f,
                                          54.28f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_model_convtranspose_dyn_filters) {
    auto ct_fn = convert_model("convtranspose_dyn_filters.onnx");

    auto test_case = ov::test::TestCase(ct_fn);

    // data
    test_case.add_input<float>(
        {0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f});

    // filters
    test_case.add_input<float>(
        Shape{2, 2, 3, 3},
        {0.f,  0.2f, 0.4f, 0.6f, 0.8f, 1.f,  1.2f, 1.4f, 1.6f, 1.8f, 2.f,  2.2f, 2.4f, 2.6f, 2.8f, 3.f,  3.2f, 3.4f,
         3.6f, 3.8f, 4.f,  4.2f, 4.4f, 4.6f, 4.8f, 5.f,  5.2f, 5.4f, 5.6f, 5.8f, 6.f,  6.2f, 6.4f, 6.6f, 6.8f, 7.f});

    // bias
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});

    // output
    test_case.add_expected_output<float>(Shape{1, 4, 2, 2},
                                         {1.4000001f,
                                          1.52f,
                                          2.6799998f,
                                          2.6799998f,
                                          5.1000004f,
                                          4.6800003f,
                                          10.16f,
                                          8.539999f,
                                          30.939999f,
                                          22.96f,
                                          53.28f,
                                          38.7f,
                                          44.36f,
                                          32.6f,
                                          75.340004f,
                                          54.28f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_transpose) {
    const auto model = convert_model("dynamic_shapes/transpose.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    Shape shape{2, 2, 4, 3};
    const auto elems_in_tensor = shape_size(shape);

    std::vector<float> input_values(elems_in_tensor);
    std::iota(std::begin(input_values), std::end(input_values), 1.f);

    test_case.add_input<float>(shape, input_values);

    std::vector<float> expected_values{1.f,  25.f, 13.f, 37.f, 4.f,  28.f, 16.f, 40.f, 7.f,  31.f, 19.f, 43.f,
                                       10.f, 34.f, 22.f, 46.f, 2.f,  26.f, 14.f, 38.f, 5.f,  29.f, 17.f, 41.f,
                                       8.f,  32.f, 20.f, 44.f, 11.f, 35.f, 23.f, 47.f, 3.f,  27.f, 15.f, 39.f,
                                       6.f,  30.f, 18.f, 42.f, 9.f,  33.f, 21.f, 45.f, 12.f, 36.f, 24.f, 48.f};
    Shape expected_shape{3, 4, 2, 2};
    test_case.add_expected_output<float>(expected_shape, expected_values);

    test_case.run();
}

namespace {
Shape get_flattened_shape(const Shape& in_shape, size_t axis) {
    size_t first_dim_size = std::accumulate(begin(in_shape),
                                            next(begin(in_shape), axis),
                                            static_cast<size_t>(1),
                                            std::multiplies<size_t>());
    size_t last_dim_size =
        std::accumulate(next(begin(in_shape), axis), end(in_shape), static_cast<size_t>(1), std::multiplies<size_t>());
    return Shape{first_dim_size, last_dim_size};
}
}  // namespace

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_flatten_axis_0) {
    const auto model = convert_model("dynamic_shapes/flatten_dyn_shape_axis0.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    const size_t RANKS_TO_TEST = 4;
    const size_t AXIS = 0;

    for (size_t r = 0; r <= RANKS_TO_TEST; ++r) {
        const Shape shape(r, 2);
        const auto elems_in_tensor = shape_size(shape);

        std::vector<float> input_values(elems_in_tensor);
        std::iota(input_values.begin(), input_values.end(), 1.f);

        test_case.add_input<float>(shape, input_values);

        std::vector<float> expected_values(input_values.begin(), input_values.end());
        const Shape expected_shape(get_flattened_shape(shape, AXIS));
        test_case.add_expected_output<float>(expected_shape, expected_values);

        test_case.run();
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_flatten_axis) {
    const auto model = convert_model("dynamic_shapes/flatten_dyn_shape_axis.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    const size_t RANKS_TO_TEST = 4;
    const size_t AXIS = 3;

    for (size_t r = AXIS; r <= RANKS_TO_TEST + AXIS; ++r) {
        const Shape shape(r, 2);
        const auto elems_in_tensor = shape_size(shape);

        std::vector<float> input_values(elems_in_tensor);
        std::iota(input_values.begin(), input_values.end(), 1.f);

        test_case.add_input<float>(shape, input_values);

        std::vector<float> expected_values(input_values.begin(), input_values.end());
        const Shape expected_shape(get_flattened_shape(shape, AXIS));
        test_case.add_expected_output<float>(expected_shape, expected_values);

        test_case.run();
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_flatten_neg_axis) {
    const auto model = convert_model("dynamic_shapes/flatten_dyn_shape_neg_axis.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    const size_t RANKS_TO_TEST = 4;
    const int64_t AXIS = -3;

    for (size_t r = -AXIS; r <= RANKS_TO_TEST + -AXIS; ++r) {
        const Shape shape(r, 2);
        const auto elems_in_tensor = shape_size(shape);

        std::vector<float> input_values(elems_in_tensor);
        std::iota(input_values.begin(), input_values.end(), 1.f);

        test_case.add_input<float>(shape, input_values);

        std::vector<float> expected_values(input_values.begin(), input_values.end());
        const Shape expected_shape(get_flattened_shape(shape, r + AXIS));
        test_case.add_expected_output<float>(expected_shape, expected_values);

        test_case.run();
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_model_flatten) {
    auto model = convert_model("flatten.onnx");

    std::vector<float> data{1, 2, 3, 4, 5, 6, 7, 8};
    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{1, 2, 2, 2}, data);
    test_case.add_expected_output<float>(Shape{1, 8}, data);

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_model_global_lp_dynamic_hw) {
    auto model = convert_model("global_lp_pool_dynamic_hw.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<int64_t>(Shape{1, 2, 3, 4},
                                 {1, 0, -4, 0, 2, 1, -6, 1, 0, 0, 0, 0, -7, 1, -1, 0, -1, 8, 0, 10, 9, 0, 0, 5});
    test_case.add_expected_output(Shape{1, 2, 1, 1}, std::vector<int64_t>{6, 8});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_slice_10_2d_input) {
    auto model = convert_model("dynamic_shapes/slice_2d_input.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    test_case.add_input<int64_t>({1, 0});
    test_case.add_input<int64_t>({2, 3});
    test_case.add_input<int64_t>({1, 2});
    test_case.add_expected_output<float>(Shape{1, 2}, {5, 7});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_slice_10_default_steps) {
    auto model = convert_model("dynamic_shapes/slice_default_steps.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({1, 2, 3, 4, 5, 6, 7, 8});
    test_case.add_input<int64_t>({1, 0});
    test_case.add_input<int64_t>({2, 3});
    test_case.add_expected_output<float>(Shape{1, 3}, {5, 6, 7});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_slice_10_slice_2d_default_steps_dyn_begin_end) {
    auto model = convert_model("dynamic_shapes/slice_2d_default_steps_dyn_begin_end.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({1, 2, 3, 4});
    test_case.add_input<int64_t>({2}, {1, 1});
    test_case.add_input<int64_t>({2}, {2, 2});
    test_case.add_expected_output<float>(Shape{1, 1}, {4});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_slice_10_clamp_neg_ends) {
    auto model = convert_model("dynamic_shapes/slice_default_steps.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    test_case.add_input<int64_t>({0, 1});
    test_case.add_input<int64_t>({-1, 1000});
    test_case.add_expected_output<float>(Shape{1, 3}, {2, 3, 4});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_slice_10_3d_input) {
    auto model = convert_model("dynamic_shapes/slice_3d_input.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    const Shape input_shape{3, 4, 1};
    std::vector<float> input_values(shape_size(input_shape));
    std::iota(input_values.begin(), input_values.end(), 0.f);
    test_case.add_input<float>(input_values);
    test_case.add_input<int64_t>({0, 0});
    test_case.add_input<int64_t>({2, 3});
    test_case.add_input<int64_t>({1, 1});
    test_case.add_expected_output<float>(Shape{2, 3, 1}, {0, 1, 2, 4, 5, 6});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_slice_10_3d_input_neg_axes) {
    auto model = convert_model("dynamic_shapes/slice_3d_input_neg_axes.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    const Shape input_shape{3, 4, 1};
    std::vector<float> input_values(shape_size(input_shape));
    std::iota(input_values.begin(), input_values.end(), 0.f);
    test_case.add_input<float>(input_values);
    test_case.add_input<int64_t>({0, 0});
    test_case.add_input<int64_t>({2, 3});
    test_case.add_input<int64_t>({1, 1});
    test_case.add_expected_output<float>(Shape{2, 3, 1}, {0, 1, 2, 4, 5, 6});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_slice_10_3d_input_12_axes) {
    auto model = convert_model("dynamic_shapes/slice_3d_input_12_axes.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    const Shape input_shape{4, 3, 2};
    std::vector<float> input_values(shape_size(input_shape));
    std::iota(input_values.begin(), input_values.end(), 0.f);
    test_case.add_input<float>(input_values);
    test_case.add_input<int64_t>({0, 0});
    test_case.add_input<int64_t>({2, 1});
    test_case.add_expected_output<float>(Shape{4, 2, 1}, {0, 2, 6, 8, 12, 14, 18, 20});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_slice_10_3d_input_20_axes) {
    auto model = convert_model("dynamic_shapes/slice_3d_input_20_axes.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    const Shape input_shape{4, 3, 2};
    std::vector<float> input_values(shape_size(input_shape));
    std::iota(input_values.begin(), input_values.end(), 0.f);
    test_case.add_input<float>(input_shape, input_values);
    test_case.add_input<int64_t>({0, 1});
    test_case.add_input<int64_t>({1, 3});
    test_case.add_input<int64_t>({1, 1});
    test_case.add_expected_output<float>(Shape{2, 3, 1}, {6, 8, 10, 12, 14, 16});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_slice_10_4d_input_23_axes) {
    auto model = convert_model("dynamic_shapes/slice_4d_input_23_axes.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    const Shape input_shape{2, 2, 2, 2};
    std::vector<float> input_values(shape_size(input_shape));
    std::iota(input_values.begin(), input_values.end(), 0.f);
    test_case.add_input<float>(input_values);
    test_case.add_input<int64_t>({0, 0});
    test_case.add_input<int64_t>({1, 1});
    test_case.add_expected_output<float>(Shape{2, 2, 1, 1}, {0, 4, 8, 12});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_slice_10_4d_input_0231_axes_ends_max) {
    auto model = convert_model("dynamic_shapes/slice_4d_input_0231_axes_ends_max.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    const Shape input_shape{2, 2, 2, 2};
    std::vector<float> input_values(shape_size(input_shape));
    std::iota(input_values.begin(), input_values.end(), 0.f);
    test_case.add_input<float>(input_values);
    test_case.add_input<int64_t>({0, 1, 1, 0});
    test_case.add_input<int64_t>({std::numeric_limits<int64_t>::max(),
                                  std::numeric_limits<int64_t>::max(),
                                  std::numeric_limits<int64_t>::max(),
                                  std::numeric_limits<int64_t>::max()});
    test_case.add_expected_output<float>(Shape{2, 2, 1, 1}, {3, 7, 11, 15});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_slice_10_4d_input_2103_axes_ends_max) {
    auto model = convert_model("dynamic_shapes/slice_4d_input_2103_axes.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    const Shape input_shape{2, 2, 2, 5};
    std::vector<float> input_values(shape_size(input_shape));
    std::iota(input_values.begin(), input_values.end(), 0.f);
    test_case.add_input<float>(input_values);
    test_case.add_input<int64_t>({1, 0, 0, 1});
    test_case.add_input<int64_t>({2,
                                  std::numeric_limits<int64_t>::max(),
                                  std::numeric_limits<int64_t>::max(),
                                  std::numeric_limits<int64_t>::max()});
    test_case.add_input<int64_t>({1, 1, 1, 2});
    test_case.add_expected_output<float>(Shape{2, 2, 1, 2}, {6, 8, 16, 18, 26, 28, 36, 38});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_slice_10_4d_input_23_axes_21_steps) {
    auto model = convert_model("dynamic_shapes/slice_4d_input_23_axes_21_steps.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    const Shape input_shape{2, 2, 6, 2};
    std::vector<float> input_values(shape_size(input_shape));
    std::iota(input_values.begin(), input_values.end(), 0.f);
    test_case.add_input<float>(input_values);
    test_case.add_input<int64_t>({0, 1});
    test_case.add_input<int64_t>({5, 2});
    test_case.add_expected_output<float>(Shape{2, 2, 3, 1}, {1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_slice_10_default_axes) {
    auto model = convert_model("dynamic_shapes/slice_default_axes.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    const Shape input_shape{4, 3, 2};
    std::vector<float> input_values(shape_size(input_shape));
    std::iota(input_values.begin(), input_values.end(), 0.f);
    test_case.add_input<float>(input_values);
    test_case.add_input<int64_t>({1, 1, 1});
    test_case.add_input<int64_t>({2, 2, 2});
    test_case.add_expected_output<float>(Shape{1, 1, 1}, {9});
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_slice_10_the_same_output_same) {
    auto model = convert_model("dynamic_shapes/slice_2d_the_same_out_shape.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    test_case.add_input<float>(std::vector<float>{1.0f, 1.0f});
    test_case.add_expected_output<float>(Shape{3, 2}, {2.0f, 1.0f, 4.0f, 3.0f, 6.0f, 5.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_model_hardmax) {
    auto model = convert_model("hardmax.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(
        {-2.02458119f, 0.00126542f,  -0.58045743f, -0.75186814f, 0.9406899f,   -0.513188f,   0.85887463f,
         1.61444086f,  0.23801147f,  -0.26816885f, 0.6597208f,   1.43889519f,  0.28798895f,  1.44769952f,
         -1.99466756f, 0.41386644f,  0.69389555f,  1.46118255f,  -1.67628606f, 1.49697552f,

         0.06337166f,  -1.15740783f, 0.8792142f,   -0.95352717f, -1.87895792f, -0.74066102f, -0.27131459f,
         0.2219685f,   0.31831001f,  0.52495901f,  0.60283089f,  0.60397976f,  0.92401468f,  0.29565101f,
         -1.14443776f, -1.07399045f, -0.92266259f, 0.24017731f,  -0.30105675f, 1.18513269f,

         0.55494542f,  1.12119279f,  -0.43156474f, 0.15101668f,  -1.460439f,   0.96375129f,  1.10411785f,
         -0.30272771f, -0.48855848f, 0.12103213f,  -0.71388492f, 1.38398178f,  0.21924434f,  0.93105052f,
         -0.21074303f, 0.48213503f,  -1.37810638f, 8.99060285f,  0.54794592f,  -0.46820172f});

    // values for hardmax with axis==2
    test_case.add_expected_output<float>(Shape{3, 4, 5}, {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                                          0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,

                                                          0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                                                          0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,

                                                          0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                                                          0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_model_softmax_axis_2) {
    auto model = convert_model("softmax_axis_2.onnx");

    const std::vector<float> input = {
        2.75793882f,  -0.50841322f, 0.82013929f,  -0.62409912f, -0.96136118f, 0.21004745f,  1.38337255f,
        1.19030397f,  2.0940445f,   -0.03551657f, -0.78686039f, 1.992782f,    0.04300319f,  -0.29230777f,
        -0.56797112f, -1.26732165f, -0.61935399f, 0.57670432f,  0.92844898f,  2.82469233f,

        0.98721677f,  -0.05100663f, -1.21178917f, -0.17530157f, 1.40051805f,  -0.13259761f, -1.14313018f,
        0.2673723f,   -0.87996154f, 1.29053106f,  1.55f,        0.8396538f,   1.20729817f,  0.23727845f,
        -0.89113606f, -1.70909842f, 0.26460363f,  -0.70566808f, 2.383518f,    1.07024615f,

        -1.21722605f, 0.82919357f,  0.55765697f,  0.12657686f,  0.63432172f,  0.75425957f,  -2.43721014f,
        -1.24478184f, 2.65316853f,  1.19509542f,  -0.95523998f, 0.5149006f,   -0.01151649f, 0.68327026f,
        -0.4589638f,  -0.46554745f, 0.21055324f,  0.39266729f,  2.05098086f,  1.83207919f};

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(input);

    test_case.add_expected_output<float>(
        Shape{3, 4, 5},
        {0.80619486f, 0.03075257f, 0.1161086f,  0.027393f,   0.01955098f, 0.07012682f, 0.22670066f,
         0.18689779f, 0.4614171f,  0.05485763f, 0.04486172f, 0.72286838f, 0.10286818f, 0.07356265f,
         0.05583908f, 0.01280724f, 0.02448298f, 0.08096658f, 0.11509768f, 0.76664552f,

         0.30399806f, 0.1076406f,  0.03371745f, 0.0950595f,  0.4595844f,  0.13369873f, 0.04866969f,
         0.19944906f, 0.06332151f, 0.55486101f, 0.39101105f, 0.19217177f, 0.27755913f, 0.10521588f,
         0.03404216f, 0.01150354f, 0.08279411f, 0.03137732f, 0.68902071f, 0.18530432f,

         0.0402528f,  0.31156222f, 0.23747503f, 0.1543129f,  0.25639705f, 0.10627912f, 0.00436928f,
         0.01439711f, 0.70979614f, 0.16515835f, 0.06798343f, 0.2957175f,  0.17468555f, 0.34994439f,
         0.11166912f, 0.03615172f, 0.07108136f, 0.08527994f, 0.44775794f, 0.35972905f});

    test_case.run(3);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_range_positive_step) {
    const auto model = convert_model("range.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({1.f});
    test_case.add_input<float>({10.f});
    test_case.add_input<float>({2.f});
    test_case.add_expected_output<float>(Shape{5}, {1.f, 3.f, 5.f, 7.f, 9.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_range_negative_step) {
    const auto model = convert_model("range.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({10.f});
    test_case.add_input<float>({1.f});
    test_case.add_input<float>({-2.f});
    test_case.add_expected_output<float>(Shape{5}, {10.f, 8.f, 6.f, 4.f, 2.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_instance_normalization_dyn_shape) {
    const auto model = convert_model("dynamic_shapes/instance_norm_dyn_shape.onnx");

    Shape data_shape{1, 2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    std::iota(std::begin(data), std::end(data), 1.f);

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(data_shape, data);
    test_case.add_input<float>(Shape{2}, std::vector<float>{2.134f, 3.256f});
    test_case.add_input<float>(Shape{2}, std::vector<float>{0.765f, 1.055f});
    test_case.add_expected_output<float>(
        data_shape,
        {-2.6335807f,  -2.015657f, -1.3977331f, -0.77980936f, -0.16188562f, 0.45603812f, 1.0739619f,  1.6918856f,
         2.3098092f,   2.927733f,  3.5456567f,  4.1635804f,   -4.130463f,   -3.1876516f, -2.2448401f, -1.3020288f,
         -0.35921717f, 0.5835942f, 1.5264057f,  2.469217f,    3.4120288f,   4.35484f,    5.2976513f,  6.240463f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_instance_normalization_dyn_shape2) {
    const auto model = convert_model("dynamic_shapes/instance_norm_dyn_shape2.onnx");

    Shape data_shape{1, 2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    std::iota(std::begin(data), std::end(data), 1.f);

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(data_shape, data);
    test_case.add_input<float>(std::vector<float>{2.134f, 3.256f});
    test_case.add_input<float>(std::vector<float>{0.765f, 1.055f});
    test_case.add_expected_output<float>(
        data_shape,
        {-2.6335807f,  -2.015657f, -1.3977331f, -0.77980936f, -0.16188562f, 0.45603812f, 1.0739619f,  1.6918856f,
         2.3098092f,   2.927733f,  3.5456567f,  4.1635804f,   -4.130463f,   -3.1876516f, -2.2448401f, -1.3020288f,
         -0.35921717f, 0.5835942f, 1.5264057f,  2.469217f,    3.4120288f,   4.35484f,    5.2976513f,  6.240463f});
    test_case.run();
}

// OPENVINO_TEST(${BACKEND_NAME}, onnx_upsample9_scales_input_nearest_infer)
// {
//     const auto model = convert_model(
//
//         "upsample9_scales_input_nearest.onnx");
//
//     // Input data shape (1, 1, 2, 2)
//     // mode: nearest
//
//     Shape expected_output_shape{1, 1, 4, 6};
//     auto test_case = ov::test::TestCase(model, s_device);
//     test_case.add_input<float>({1.0, 2.0, 3.0, 4.0});
//     test_case.add_input<float>({1.0, 1.0, 2.0, 3.0});
//     test_case.add_expected_output<float>(
//         expected_output_shape, {1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
//                                 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0});
//     test_case.run();
// }

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_slice_1_2d_input) {
    auto model = convert_model("dynamic_shapes/slice_2d_input_opset1.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    test_case.add_expected_output<float>(Shape{1, 4}, {5, 6, 7, 8});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_slice_1_clamp_neg_ends) {
    auto model = convert_model("dynamic_shapes/slice_2d_clamp_neg_ends_opset1.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    test_case.add_expected_output<float>(Shape{1, 3}, {2, 3, 4});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_slice_1_3d_input_21_axes_ends_max) {
    auto model = convert_model("dynamic_shapes/slice_3d_input_21_axes_ends_max_opset1.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    const Shape input_shape{1, 2, 3, 4};
    std::vector<float> input_values(shape_size(input_shape));
    std::iota(input_values.begin(), input_values.end(), 0.f);
    test_case.add_input<float>(input_shape, input_values);
    test_case.add_expected_output<float>(Shape{1, 1, 3, 3}, {13, 14, 15, 17, 18, 19, 21, 22, 23});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dyn_shapes_reduce_max_dynamic_input_rank_negative_axis) {
    // the ReduceMax node has a fully dynamic input and the reduction axis is -1
    auto model = convert_model("dynamic_shapes/reduce_max_dynamic_input_rank_negative_axis.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{2, 4}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    test_case.add_expected_output<float>(Shape{2, 1}, {4, 8});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_size_dyn_op) {
    const auto model = convert_model("dynamic_shapes/size_op_dyn.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    test_case.add_expected_output<int64_t>(Shape{}, {6});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_max_pool_dyn_rank_without_default_attrs) {
    auto model = convert_model("dynamic_shapes/max_pool_dyn_rank_without_default_attrs.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    Shape input_shape{1, 1, 4, 4};
    std::vector<float> input(shape_size(input_shape));
    std::iota(input.begin(), input.end(), 0.f);
    test_case.add_input<float>(input_shape, input);
    test_case.add_expected_output<float>(Shape{1, 1, 3, 3}, {5, 6, 7, 9, 10, 11, 13, 14, 15});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_depth_to_space_dynamic_input) {
    auto model = convert_model("dynamic_shapes/depth_to_space.onnx");

    std::vector<float> input(32);
    std::iota(input.begin(), input.end(), 0.f);

    std::vector<float> expected_output{0.f,  8.f,  1.f,  9.f,  16.f, 24.f, 17.f, 25.f, 2.f,  10.f, 3.f,
                                       11.f, 18.f, 26.f, 19.f, 27.f, 4.f,  12.f, 5.f,  13.f, 20.f, 28.f,
                                       21.f, 29.f, 6.f,  14.f, 7.f,  15.f, 22.f, 30.f, 23.f, 31.f};

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input(Shape{1, 8, 2, 2}, input);
    test_case.add_expected_output(Shape{1, 2, 4, 4}, expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_space_to_depth_dynamic_input) {
    auto model = convert_model("dynamic_shapes/space_to_depth.onnx");

    std::vector<float> input(32);
    std::iota(input.begin(), input.end(), 0.f);

    std::vector<float> expected_output{
        0.f, 2.f, 8.f,  10.f, 16.f, 18.f, 24.f, 26.f, 1.f, 3.f, 9.f,  11.f, 17.f, 19.f, 25.f, 27.f,
        4.f, 6.f, 12.f, 14.f, 20.f, 22.f, 28.f, 30.f, 5.f, 7.f, 13.f, 15.f, 21.f, 23.f, 29.f, 31.f,
    };

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input(Shape{1, 2, 4, 4}, input);
    test_case.add_expected_output(Shape{1, 8, 2, 2}, expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_eye_like_dyn_shape) {
    const auto model = convert_model("dynamic_shapes/eye_like_dyn_shape.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{3, 4}, {5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f});
    test_case.add_expected_output<float>(Shape{3, 4}, {0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_eye_like_dyn_rank) {
    const auto model = convert_model("dynamic_shapes/eye_like_dyn_rank.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{3, 4}, {5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f});
    test_case.add_expected_output<float>(Shape{3, 4}, {0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f});

    test_case.run();
}
