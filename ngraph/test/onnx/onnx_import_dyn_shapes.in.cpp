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

#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "ngraph/frontend/onnx_import/default_opset.hpp"
#include "ngraph/frontend/onnx_import/onnx.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"
#include "util/type_prop.hpp"

using namespace ngraph;
using namespace ngraph::onnx_import;
using namespace ngraph::test;

static std::string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_onnx_dynamic_dims_to_ngraph_dynamic_dims)
{
    // the model represents a linear function A * x + B
    // where all 3 operands are model inputs (no initializers)
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dynamic_shapes/ab_plus_c.prototxt"));

    const auto& graph_inputs = function->get_parameters();
    EXPECT_EQ(graph_inputs.size(), 3);

    // all inputs in the model have a 2D partial shape {?, 2}
    for (const auto& input : graph_inputs)
    {
        const auto& input_ps = input->get_partial_shape();
        EXPECT_TRUE(input_ps.is_dynamic());

        ASSERT_TRUE(input_ps.rank().is_static());
        EXPECT_EQ(input_ps.rank().get_length(), 2);

        EXPECT_TRUE(input_ps[0].is_dynamic());
        ASSERT_TRUE(input_ps[1].is_static());
        EXPECT_EQ(input_ps[1].get_length(), 2);
    }

    const auto& graph_outputs = function->get_results();
    EXPECT_EQ(graph_outputs.size(), 1);

    const auto out = *(graph_outputs.cbegin());
    const auto& out_ps = out->get_output_partial_shape(0);
    ASSERT_TRUE(out_ps.rank().is_static());
    EXPECT_EQ(out_ps.rank().get_length(), 2);

    EXPECT_TRUE(out_ps[0].is_dynamic());
    ASSERT_TRUE(out_ps[1].is_static());
    EXPECT_EQ(out_ps[1].get_length(), 2);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_ab_plus_c_inference)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dynamic_shapes/ab_plus_c.prototxt"));

    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);

    struct ExpectedValuesGenerator
    {
        int64_t i = 1;
        int64_t operator()()
        {
            const auto ret = i * i + i;
            ++i;
            return ret;
        }
    };

    const size_t NUM_BATCHES_TO_TEST = 5;

    for (size_t batch = 1; batch <= NUM_BATCHES_TO_TEST; ++batch)
    {
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

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_scalar_initializers_shape_check)
{
    // initializers defined witout the "dims" field should produce Constants with an empty Shape
    // initializers with "dims: 0" should be have the same way (Shape{} not Shape{0})
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dynamic_shapes/scalar_initializers.prototxt"));

    for (auto ng_node : function->get_ordered_ops())
    {
        if (as_type_ptr<default_opset::Constant>(ng_node))
        {
            EXPECT_EQ(ng_node->get_shape(), Shape{});
        }
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_dynamic_rank_input_check)
{
    // the model contains a single Add operation that takes a fully dynamic input and a scalar
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dynamic_shapes/a_plus_b_dyn_rank.prototxt"));

    const auto& graph_inputs = function->get_parameters();
    ASSERT_EQ(graph_inputs.size(), 2);

    const auto dyn_rank_input = graph_inputs[0];
    const auto scalar_input = graph_inputs[1];

    EXPECT_TRUE(dyn_rank_input->get_partial_shape().rank().is_dynamic());

    ASSERT_TRUE(scalar_input->get_partial_shape().is_static());
    EXPECT_EQ(scalar_input->get_partial_shape().to_shape(), Shape{});

    const auto& graph_outputs = function->get_results();
    EXPECT_EQ(graph_outputs.size(), 1);

    const auto out = *(graph_outputs.cbegin());
    EXPECT_TRUE(out->get_output_partial_shape(0).rank().is_dynamic());
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_dynamic_rank_input_inference)
{
    // the model contains a single Add operation that takes a fully dynamic input and a scalar
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dynamic_shapes/a_plus_b_dyn_rank.prototxt"));

    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);

    const size_t RANKS_TO_TEST = 3;
    const int64_t SCALAR_INPUT_VAL = 5;

    for (size_t r = 0; r <= RANKS_TO_TEST; ++r)
    {
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

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_model_acosh_1_3)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dynamic_shapes/acosh_dyn_shape.prototxt"));

    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);
    test_case.add_input<float>(Shape{1, 3}, {1.0f, 2.5f, 4.3f});
    test_case.add_expected_output<float>(Shape{1, 3}, {0.0f, 1.5667993f, 2.1379586f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_model_acosh_3_2)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dynamic_shapes/acosh_dyn_shape.prototxt"));

    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);
    test_case.add_input<float>(Shape{3, 2}, {1.0f, 2.5f, 4.3f, 1.0f, 2.5f, 4.3f});
    test_case.add_expected_output<float>(
        Shape{3, 2}, {0.0f, 1.5667993f, 2.1379586f, 0.0f, 1.5667993f, 2.1379586f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_model_asinh_1_3)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dynamic_shapes/asinh_dyn_shape.prototxt"));

    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);
    test_case.add_input<float>(Shape{1, 3}, {-1.5f, 0.0f, 1.5f});
    test_case.add_expected_output<float>(Shape{1, 3}, {-1.1947632f, 0.0f, 1.1947632f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_model_asinh_3_2)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dynamic_shapes/asinh_dyn_shape.prototxt"));

    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);
    test_case.add_input<float>(Shape{3, 2}, {-1.5f, 0.0f, 1.5f, -1.5f, 0.0f, 1.5f});
    test_case.add_expected_output<float>(
        Shape{3, 2}, {-1.1947632f, 0.0f, 1.1947632f, -1.1947632, 0.0f, 1.1947632f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_model_atanh_1_3)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dynamic_shapes/atanh_dyn_shape.prototxt"));

    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);
    test_case.add_input<float>(Shape{1, 3}, {-0.9f, 0.0f, 0.9f});
    test_case.add_expected_output<float>(Shape{1, 3}, {-1.47221948f, 0.0f, 1.47221948f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_model_atanh_3_2)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dynamic_shapes/atanh_dyn_shape.prototxt"));

    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);
    test_case.add_input<float>(Shape{3, 2}, {-0.9f, 0.0f, 0.9f, -0.9f, 0.0f, 0.9f});
    test_case.add_expected_output<float>(
        Shape{3, 2}, {-1.47221948f, 0.0f, 1.47221948f, -1.47221948f, 0.0f, 1.47221948f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_model_conv_with_dynamic_batch)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/dynamic_shapes/conv_with_dynamic_batch.prototxt"));

    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);

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

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_avg_pool_dyn_shape)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dynamic_shapes/average_pool_2d_dyn.prototxt"));

    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);

    const Shape shape{1, 1, 4, 4};
    const auto elems_in_tensor = shape_size(shape);
    std::vector<float> input_values(elems_in_tensor);
    std::iota(input_values.begin(), input_values.end(), 0.f);

    test_case.add_input<float>(shape, input_values);

    std::vector<float> expected_values{2.5f, 4.5f, 10.5f, 12.5f};
    test_case.add_expected_output<float>(Shape{1, 1, 2, 2}, expected_values);

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_max_pool_dyn_shape)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dynamic_shapes/max_pool_2d_dyn.prototxt"));

    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);

    const Shape shape{1, 1, 4, 4};
    const auto elems_in_tensor = shape_size(shape);
    std::vector<float> input_values(elems_in_tensor);
    std::iota(input_values.begin(), input_values.end(), 0.f);

    test_case.add_input<float>(shape, input_values);

    std::vector<float> expected_values{0.f, 2.f, 3.f, 8.f, 10.f, 11.f, 12.f, 14.f, 15.f};
    test_case.add_expected_output<float>(Shape{1, 1, 3, 3}, expected_values);

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_global_avg_pool_dyn_shape)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/dynamic_shapes/global_average_pool_dyn.prototxt"));

    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);

    const Shape shape{1, 3, 5, 5};
    const auto elems_in_tensor = shape_size(shape);
    std::vector<float> input_values(elems_in_tensor);
    std::iota(input_values.begin(), input_values.end(), 0.f);

    test_case.add_input<float>(shape, input_values);

    std::vector<float> expected_values{12.f, 37.f, 62.f};
    test_case.add_expected_output<float>(Shape{1, 3, 1, 1}, expected_values);

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_global_max_pool_dyn_shape)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dynamic_shapes/global_max_pool_dyn.prototxt"));

    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);

    const Shape shape{1, 3, 5, 5};
    const auto elems_in_tensor = shape_size(shape);
    std::vector<float> input_values(elems_in_tensor);
    std::iota(input_values.begin(), input_values.end(), 0.f);

    test_case.add_input<float>(shape, input_values);

    std::vector<float> expected_values{24.f, 49.f, 74.f};
    test_case.add_expected_output<float>(Shape{1, 3, 1, 1}, expected_values);

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_arg_max_dyn_shape)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dynamic_shapes/argmax_dyn.prototxt"));

    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);

    const Shape shape{3, 2, 2};
    const auto elems_in_tensor = shape_size(shape);
    std::vector<int32_t> input_values(elems_in_tensor);
    std::iota(input_values.begin(), input_values.end(), 1);

    test_case.add_input<int32_t>(shape, input_values);

    std::vector<int64_t> expected_values{1, 1, 1, 1, 1, 1};
    test_case.add_expected_output<int64_t>(Shape{3, 1, 2}, expected_values);

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_arg_min_no_keep_dims_dyn_shape)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/dynamic_shapes/argmin_no_keep_dims_dyn.prototxt"));

    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);

    const Shape shape{3, 2, 2};
    const auto elems_in_tensor = shape_size(shape);
    std::vector<int32_t> input_values(elems_in_tensor);
    std::iota(input_values.begin(), input_values.end(), 1);

    test_case.add_input<int32_t>(shape, input_values);

    std::vector<int64_t> expected_values{0, 0, 0, 0};
    test_case.add_expected_output<int64_t>(Shape{2, 2}, expected_values);

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_constant_of_shape_float_zeros)
{
    auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/dynamic_shapes/constant_of_shape_float_zeros.prototxt"));

    std::vector<float> expected_values(24, 0);

    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);

    test_case.add_input<int64_t>(Shape{3}, std::vector<int64_t>{2, 3, 4});
    test_case.add_expected_output<float>(Shape{2, 3, 4}, expected_values);

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_constant_of_shape_int_ones)
{
    auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/dynamic_shapes/constant_of_shape_int_ones.prototxt"));

    std::vector<int32_t> expected_values(6, 1);

    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);

    test_case.add_input<int64_t>(Shape{2}, std::vector<int64_t>{2, 3});
    test_case.add_expected_output<int32_t>(Shape{2, 3}, expected_values);

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_expand_1_dyn_shape)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dynamic_shapes/expand_dyn.prototxt"));

    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);

    test_case.add_input<float>(Shape{3, 1}, std::vector<float>{1.f, 2.f, 3.f});
    test_case.add_input<int64_t>(Shape{3}, std::vector<int64_t>{2, 1, 6});

    std::vector<float> expected_values{1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f,
                                       3.f, 3.f, 3.f, 3.f, 3.f, 3.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
                                       2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 3.f, 3.f, 3.f, 3.f, 3.f, 3.f};
    test_case.add_expected_output<float>(Shape{2, 3, 6}, expected_values);

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_expand_2_dyn_shape)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dynamic_shapes/expand_dyn.prototxt"));

    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);

    test_case.add_input<float>(Shape{3, 1}, std::vector<float>{1.f, 2.f, 3.f});
    test_case.add_input<int64_t>(Shape{3}, std::vector<int64_t>{2, 3, 4});

    std::vector<float> expected_values{1.f, 1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 2.f, 3.f, 3.f, 3.f, 3.f,
                                       1.f, 1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 2.f, 3.f, 3.f, 3.f, 3.f};
    test_case.add_expected_output<float>(Shape{2, 3, 4}, expected_values);

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_expand_3_dyn_shape)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dynamic_shapes/expand_dyn.prototxt"));

    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);

    test_case.add_input<float>(Shape{2, 1}, std::vector<float>{4.f, 5.f});
    test_case.add_input<int64_t>(Shape{2}, std::vector<int64_t>{2, 4});

    std::vector<float> expected_values{4.f, 4.f, 4.f, 4.f, 5.f, 5.f, 5.f, 5.f};
    test_case.add_expected_output<float>(Shape{2, 4}, expected_values);

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_expand_4_dyn_shape)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dynamic_shapes/expand_dyn.prototxt"));

    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);

    test_case.add_input<float>(Shape{1, 3, 1}, std::vector<float>{7.f, 8.f, 9.f});
    test_case.add_input<int64_t>(Shape{2}, std::vector<int64_t>{3, 1});

    std::vector<float> expected_values{7.f, 8.f, 9.f};
    test_case.add_expected_output<float>(Shape{1, 3, 1}, expected_values);

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_expand_5_dyn_shape)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dynamic_shapes/expand_dyn.prototxt"));

    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);

    test_case.add_input<float>(Shape{1, 4, 1}, std::vector<float>{7.f, 8.f, 9.f, 10.f});
    test_case.add_input<int64_t>(Shape{2}, std::vector<int64_t>{1, 4});

    std::vector<float> expected_values{
        7.f, 7.f, 7.f, 7.f, 8.f, 8.f, 8.f, 8.f, 9.f, 9.f, 9.f, 9.f, 10.f, 10.f, 10.f, 10.f};
    test_case.add_expected_output<float>(Shape{1, 4, 4}, expected_values);

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_expand_6_dyn_shape)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dynamic_shapes/expand_dyn.prototxt"));

    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);

    test_case.add_input<float>(Shape{1, 3, 1}, std::vector<float>{7.f, 8.f, 9.f});
    test_case.add_input<int64_t>(Shape{3}, std::vector<int64_t>{3, 1, 3});

    std::vector<float> expected_values{7.f, 7.f, 7.f, 8.f, 8.f, 8.f, 9.f, 9.f, 9.f,
                                       7.f, 7.f, 7.f, 8.f, 8.f, 8.f, 9.f, 9.f, 9.f,
                                       7.f, 7.f, 7.f, 8.f, 8.f, 8.f, 9.f, 9.f, 9.f};
    test_case.add_expected_output<float>(Shape{3, 3, 3}, expected_values);

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_expand_uint16_dyn_shape)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dynamic_shapes/expand_uint16_dyn.prototxt"));

    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);

    test_case.add_input<uint16_t>(Shape{1, 2, 1}, std::vector<uint16_t>{1, 2});
    test_case.add_input<int64_t>(Shape{4}, std::vector<int64_t>{2, 2, 1, 2});

    std::vector<uint16_t> expected_values{1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2};
    test_case.add_expected_output<uint16_t>(Shape{2, 2, 2, 2}, expected_values);

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_model_tile)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/tile.prototxt"));

    auto test_case =
        ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);
    test_case.add_input<std::int16_t>({0, 1, 2, 3, 4, 5}); // input
    test_case.add_input<std::int16_t>({2, 1});             // repeats
    test_case.add_expected_output<std::int16_t>(Shape{4, 3}, {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_model_tile_static)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/tile_static.prototxt"));

    auto test_case =
        ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);
    test_case.add_input<std::int16_t>({0, 1, 2, 3, 4, 5}); // input
    test_case.add_expected_output<std::int16_t>(
        Shape{4, 6}, {0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_model_convtranspose_dyn_data)
{
    auto ct_fn = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/convtranspose_dyn_data.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(ct_fn, "${BACKEND_NAME}", BackendMode::DYNAMIC);

    // data
    test_case.add_input<float>(Shape{1, 2, 3, 3},
                               {0.f,
                                0.1f,
                                0.2f,
                                0.3f,
                                0.4f,
                                0.5f,
                                0.6f,
                                0.7f,
                                0.8f,
                                0.9f,
                                1.f,
                                1.1f,
                                1.2f,
                                1.3f,
                                1.4f,
                                1.5f,
                                1.6f,
                                1.7f});

    // filters
    test_case.add_input<float>({0.f,  0.2f, 0.4f, 0.6f, 0.8f, 1.f,  1.2f, 1.4f, 1.6f,
                                1.8f, 2.f,  2.2f, 2.4f, 2.6f, 2.8f, 3.f,  3.2f, 3.4f,
                                3.6f, 3.8f, 4.f,  4.2f, 4.4f, 4.6f, 4.8f, 5.f,  5.2f,
                                5.4f, 5.6f, 5.8f, 6.f,  6.2f, 6.4f, 6.6f, 6.8f, 7.f});

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

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_model_convtranspose_dyn_filters)
{
    auto ct_fn = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/convtranspose_dyn_filters.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(ct_fn, "${BACKEND_NAME}", BackendMode::DYNAMIC);

    // data
    test_case.add_input<float>({0.f,
                                0.1f,
                                0.2f,
                                0.3f,
                                0.4f,
                                0.5f,
                                0.6f,
                                0.7f,
                                0.8f,
                                0.9f,
                                1.f,
                                1.1f,
                                1.2f,
                                1.3f,
                                1.4f,
                                1.5f,
                                1.6f,
                                1.7f});

    // filters
    test_case.add_input<float>(
        Shape{2, 2, 3, 3}, {0.f,  0.2f, 0.4f, 0.6f, 0.8f, 1.f,  1.2f, 1.4f, 1.6f, 1.8f, 2.f,  2.2f,
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

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_transpose)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dynamic_shapes/transpose.prototxt"));
    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);

    Shape shape{2, 2, 4, 3};
    const auto elems_in_tensor = shape_size(shape);

    std::vector<float> input_values(elems_in_tensor);
    std::iota(std::begin(input_values), std::end(input_values), 1);

    test_case.add_input<float>(shape, input_values);

    std::vector<float> expected_values{1.f,  25.f, 13.f, 37.f, 4.f,  28.f, 16.f, 40.f, 7.f,  31.f,
                                       19.f, 43.f, 10.f, 34.f, 22.f, 46.f, 2.f,  26.f, 14.f, 38.f,
                                       5.f,  29.f, 17.f, 41.f, 8.f,  32.f, 20.f, 44.f, 11.f, 35.f,
                                       23.f, 47.f, 3.f,  27.f, 15.f, 39.f, 6.f,  30.f, 18.f, 42.f,
                                       9.f,  33.f, 21.f, 45.f, 12.f, 36.f, 24.f, 48.f};
    Shape expected_shape{3, 4, 2, 2};
    test_case.add_expected_output<float>(expected_shape, expected_values);

    test_case.run();
}

namespace
{
    Shape get_flattened_shape(const Shape& in_shape, size_t axis)
    {
        size_t first_dim_size = std::accumulate(
            begin(in_shape), next(begin(in_shape), axis), 1UL, std::multiplies<size_t>());
        size_t last_dim_size = std::accumulate(
            next(begin(in_shape), axis), end(in_shape), 1UL, std::multiplies<size_t>());
        return Shape{first_dim_size, last_dim_size};
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_flatten_axis_0)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/dynamic_shapes/flatten_dyn_shape_axis0.prototxt"));
    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);

    const size_t RANKS_TO_TEST = 4;
    const size_t AXIS = 0;

    for (size_t r = 0; r <= RANKS_TO_TEST; ++r)
    {
        const Shape shape(r, 2);
        const auto elems_in_tensor = shape_size(shape);

        std::vector<float> input_values(elems_in_tensor);
        std::iota(input_values.begin(), input_values.end(), 1);

        test_case.add_input<float>(shape, input_values);

        std::vector<float> expected_values(input_values.begin(), input_values.end());
        const Shape expected_shape(get_flattened_shape(shape, AXIS));
        test_case.add_expected_output<float>(expected_shape, expected_values);

        test_case.run();
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_flatten_axis)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/dynamic_shapes/flatten_dyn_shape_axis.prototxt"));
    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);

    const size_t RANKS_TO_TEST = 4;
    const size_t AXIS = 3;

    for (size_t r = AXIS; r <= RANKS_TO_TEST + AXIS; ++r)
    {
        const Shape shape(r, 2);
        const auto elems_in_tensor = shape_size(shape);

        std::vector<float> input_values(elems_in_tensor);
        std::iota(input_values.begin(), input_values.end(), 1);

        test_case.add_input<float>(shape, input_values);

        std::vector<float> expected_values(input_values.begin(), input_values.end());
        const Shape expected_shape(get_flattened_shape(shape, AXIS));
        test_case.add_expected_output<float>(expected_shape, expected_values);

        test_case.run();
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_flatten_neg_axis)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/dynamic_shapes/flatten_dyn_shape_neg_axis.prototxt"));
    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);

    const size_t RANKS_TO_TEST = 4;
    const int64_t AXIS = -3;

    for (size_t r = -AXIS; r <= RANKS_TO_TEST + -AXIS; ++r)
    {
        const Shape shape(r, 2);
        const auto elems_in_tensor = shape_size(shape);

        std::vector<float> input_values(elems_in_tensor);
        std::iota(input_values.begin(), input_values.end(), 1);

        test_case.add_input<float>(shape, input_values);

        std::vector<float> expected_values(input_values.begin(), input_values.end());
        const Shape expected_shape(get_flattened_shape(shape, r + AXIS));
        test_case.add_expected_output<float>(expected_shape, expected_values);

        test_case.run();
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_model_global_lp_dynamic_hw)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/global_lp_pool_dynamic_hw.prototxt"));

    auto test_case =
        ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);
    test_case.add_input<int64_t>(Shape{1, 2, 3, 4}, {1,  0, -4, 0, 2,  1, -6, 1,  0, 0, 0, 0,
                                                     -7, 1, -1, 0, -1, 8, 0,  10, 9, 0, 0, 5});
    test_case.add_expected_output(Shape{1, 2, 1, 1}, std::vector<int64_t>{6, 8});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_slice_10_2d_input)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dynamic_shapes/slice_2d_input.prototxt"));

    auto test_case =
        ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}", test::BackendMode::DYNAMIC);
    test_case.add_input<float>(std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    test_case.add_input<int64_t>({1, 0});
    test_case.add_input<int64_t>({2, 3});
    test_case.add_input<int64_t>({1, 2});
    test_case.add_expected_output<float>(Shape{1, 2}, {5, 7});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_slice_10_default_steps)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dynamic_shapes/slice_default_steps.prototxt"));

    auto test_case =
        ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}", test::BackendMode::DYNAMIC);
    test_case.add_input<float>({1, 2, 3, 4, 5, 6, 7, 8});
    test_case.add_input<int64_t>({1, 0});
    test_case.add_input<int64_t>({2, 3});
    test_case.add_expected_output<float>(Shape{1, 3}, {5, 6, 7});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_slice_10_slice_2d_default_steps_dyn_begin_end)
{
    auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/dynamic_shapes/slice_2d_default_steps_dyn_begin_end.prototxt"));

    auto test_case =
        ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}", test::BackendMode::DYNAMIC);
    test_case.add_input<float>({1, 2, 3, 4});
    test_case.add_input<int64_t>({2}, {1, 1});
    test_case.add_input<int64_t>({2}, {2, 2});
    test_case.add_expected_output<float>(Shape{1, 1}, {4});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_slice_10_clamp_neg_ends)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dynamic_shapes/slice_default_steps.prototxt"));

    auto test_case =
        ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}", test::BackendMode::DYNAMIC);
    test_case.add_input<float>(std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    test_case.add_input<int64_t>({0, 1});
    test_case.add_input<int64_t>({-1, 1000});
    test_case.add_expected_output<float>(Shape{1, 3}, {2, 3, 4});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_slice_10_3d_input)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dynamic_shapes/slice_3d_input.prototxt"));

    auto test_case =
        ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}", test::BackendMode::DYNAMIC);

    const Shape input_shape{3, 4, 1};
    std::vector<float> input_values(shape_size(input_shape));
    std::iota(input_values.begin(), input_values.end(), 0);
    test_case.add_input<float>(input_values);
    test_case.add_input<int64_t>({0, 0});
    test_case.add_input<int64_t>({2, 3});
    test_case.add_input<int64_t>({1, 1});
    test_case.add_expected_output<float>(Shape{2, 3, 1}, {0, 1, 2, 4, 5, 6});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_slice_10_3d_input_neg_axes)
{
    auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/dynamic_shapes/slice_3d_input_neg_axes.prototxt"));

    auto test_case =
        ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}", test::BackendMode::DYNAMIC);

    const Shape input_shape{3, 4, 1};
    std::vector<float> input_values(shape_size(input_shape));
    std::iota(input_values.begin(), input_values.end(), 0);
    test_case.add_input<float>(input_values);
    test_case.add_input<int64_t>({0, 0});
    test_case.add_input<int64_t>({2, 3});
    test_case.add_input<int64_t>({1, 1});
    test_case.add_expected_output<float>(Shape{2, 3, 1}, {0, 1, 2, 4, 5, 6});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_slice_10_3d_input_12_axes)
{
    auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/dynamic_shapes/slice_3d_input_12_axes.prototxt"));

    auto test_case =
        ngraph::test::NgraphTestCase(function, "INTERPRETER", test::BackendMode::DYNAMIC);

    const Shape input_shape{4, 3, 2};
    std::vector<float> input_values(shape_size(input_shape));
    std::iota(input_values.begin(), input_values.end(), 0);
    test_case.add_input<float>(input_values);
    test_case.add_input<int64_t>({0, 0});
    test_case.add_input<int64_t>({2, 1});
    test_case.add_expected_output<float>(Shape{4, 2, 1}, {0, 2, 6, 8, 12, 14, 18, 20});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_slice_10_3d_input_20_axes)
{
    auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/dynamic_shapes/slice_3d_input_20_axes.prototxt"));

    auto test_case =
        ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}", test::BackendMode::DYNAMIC);

    const Shape input_shape{4, 3, 2};
    std::vector<float> input_values(shape_size(input_shape));
    std::iota(input_values.begin(), input_values.end(), 0);
    test_case.add_input<float>(input_shape, input_values);
    test_case.add_input<int64_t>({0, 1});
    test_case.add_input<int64_t>({1, 3});
    test_case.add_input<int64_t>({1, 1});
    test_case.add_expected_output<float>(Shape{2, 3, 1}, {6, 8, 10, 12, 14, 16});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_slice_10_4d_input_23_axes)
{
    auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/dynamic_shapes/slice_4d_input_23_axes.prototxt"));

    auto test_case =
        ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}", test::BackendMode::DYNAMIC);

    const Shape input_shape{2, 2, 2, 2};
    std::vector<float> input_values(shape_size(input_shape));
    std::iota(input_values.begin(), input_values.end(), 0);
    test_case.add_input<float>(input_values);
    test_case.add_input<int64_t>({0, 0});
    test_case.add_input<int64_t>({1, 1});
    test_case.add_expected_output<float>(Shape{2, 2, 1, 1}, {0, 4, 8, 12});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_slice_10_4d_input_0231_axes_ends_max)
{
    auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/dynamic_shapes/slice_4d_input_0231_axes_ends_max.prototxt"));

    auto test_case =
        ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}", test::BackendMode::DYNAMIC);

    const Shape input_shape{2, 2, 2, 2};
    std::vector<float> input_values(shape_size(input_shape));
    std::iota(input_values.begin(), input_values.end(), 0);
    test_case.add_input<float>(input_values);
    test_case.add_input<int64_t>({0, 1, 1, 0});
    test_case.add_input<int64_t>({std::numeric_limits<int64_t>::max(),
                                  std::numeric_limits<int64_t>::max(),
                                  std::numeric_limits<int64_t>::max(),
                                  std::numeric_limits<int64_t>::max()});
    test_case.add_expected_output<float>(Shape{2, 2, 1, 1}, {3, 7, 11, 15});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_slice_10_4d_input_2103_axes_ends_max)
{
    auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/dynamic_shapes/slice_4d_input_2103_axes.prototxt"));

    auto test_case =
        ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}", test::BackendMode::DYNAMIC);

    const Shape input_shape{2, 2, 2, 5};
    std::vector<float> input_values(shape_size(input_shape));
    std::iota(input_values.begin(), input_values.end(), 0);
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

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_slice_10_4d_input_23_axes_21_steps)
{
    auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/dynamic_shapes/slice_4d_input_23_axes_21_steps.prototxt"));

    auto test_case =
        ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}", test::BackendMode::DYNAMIC);

    const Shape input_shape{2, 2, 6, 2};
    std::vector<float> input_values(shape_size(input_shape));
    std::iota(input_values.begin(), input_values.end(), 0);
    test_case.add_input<float>(input_values);
    test_case.add_input<int64_t>({0, 1});
    test_case.add_input<int64_t>({5, 2});
    test_case.add_expected_output<float>(Shape{2, 2, 3, 1},
                                         {1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_dyn_shapes_slice_10_default_axes)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dynamic_shapes/slice_default_axes.prototxt"));

    auto test_case =
        ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}", test::BackendMode::DYNAMIC);

    const Shape input_shape{4, 3, 2};
    std::vector<float> input_values(shape_size(input_shape));
    std::iota(input_values.begin(), input_values.end(), 0);
    test_case.add_input<float>(input_values);
    test_case.add_input<int64_t>({1, 1, 1});
    test_case.add_input<int64_t>({2, 2, 2});
    test_case.add_expected_output<float>(Shape{1, 1, 1}, {9});
}
