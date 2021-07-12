// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <string>

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif
// clang-format on

#include "gtest/gtest.h"
#include "ngraph/check.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/engine/test_engines.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});


NGRAPH_TEST(${BACKEND_NAME}, hardsigmoid)
{
    const Shape shape{2, 7};
    const float alpha_f = 0.125f;
    const float beta_f = 0.642f;
    const auto A = make_shared<op::Parameter>(element::f32, shape);
    const auto alpha = op::Constant::create<float>(A->get_element_type(), Shape{}, {alpha_f});
    const auto beta = op::Constant::create<float>(A->get_element_type(), Shape{}, {beta_f});
    auto hardsigmoid = make_shared<op::HardSigmoid>(A, alpha, beta);
    auto f = make_shared<Function>(NodeVector{hardsigmoid}, ParameterVector{A});
    vector<float> input{-1.f,
                        0.f,
                        1.f,
                        -100.f,
                        100.f,
                        -3.1234567f,
                        5.876543f,
                        7.13245364f,
                        numeric_limits<float>::max(),
                        numeric_limits<float>::lowest(),
                        numeric_limits<float>::min(),
                        numeric_limits<float>::infinity(),
                        numeric_limits<float>::min() / 16.f,
                        -numeric_limits<float>::min() / 16.f};

    // Prepare expected output data
    auto impl = [alpha_f, beta_f](float val) { return min(max(alpha_f * val + beta_f, 0.f), 1.f); };
    vector<float> expected_output;
    transform(begin(input), end(input), back_inserter(expected_output), impl);

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(input);
    test_case.add_expected_output<float>(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, mvn_mean_normalization)
{
    Shape data_shape{1, 2, 5};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);

    auto mvn_func = make_shared<op::MVN>(data, true, false);
    auto function = make_shared<Function>(NodeVector{mvn_func}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(function);
    // data
    vector<float> data_vector(shape_size(data_shape));
    iota(begin(data_vector), end(data_vector), 0);
    test_case.add_input<float>(data_vector);

    // expected result
    test_case.add_expected_output<float>(
        data_shape, vector<float>{-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, mvn_mean_normalization_split_channels)
{
    Shape data_shape{1, 2, 5, 1};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);

    auto mvn_func = make_shared<op::MVN>(data, false, false);
    auto function = make_shared<Function>(NodeVector{mvn_func}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(function);
    // data
    vector<float> data_vector(shape_size(data_shape));
    iota(begin(data_vector), end(data_vector), 0);
    test_case.add_input<float>(data_vector);

    // expected result
    test_case.add_expected_output<float>({1, 2, 5, 1},
                                         vector<float>{-2, -1, 0, 1, 2, -2, -1, 0, 1, 2});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, mvn_mean_variance_normalization)
{
    Shape data_shape{1, 2, 5};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);

    auto mvn_func = make_shared<op::MVN>(data);
    auto function = make_shared<Function>(NodeVector{mvn_func}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(function);
    // data
    vector<float> data_vector(shape_size(data_shape));
    iota(begin(data_vector), end(data_vector), 0);
    test_case.add_input<float>(data_vector);

    // expected result
    test_case.add_expected_output<float>(data_shape,
                                         vector<float>{-1.566698903055826,
                                                       -1.2185435912656424,
                                                       -0.87038827947545883,
                                                       -0.52223296768527527,
                                                       -0.17407765589509178,
                                                       0.17407765589509178,
                                                       0.52223296768527527,
                                                       0.87038827947545883,
                                                       1.2185435912656424,
                                                       1.566698903055826});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, mvn_mean_variance_normalization_split_channels)
{
    Shape data_shape{1, 2, 5};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);

    auto mvn_func = make_shared<op::MVN>(data, false);
    auto function = make_shared<Function>(NodeVector{mvn_func}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(function);
    // data
    vector<float> data_vector(shape_size(data_shape));
    iota(begin(data_vector), end(data_vector), 0);
    test_case.add_input<float>(data_vector);

    // expected result
    test_case.add_expected_output<float>(data_shape,
                                         vector<float>{-1.4142135613730948,
                                                       -0.70710678068654742,
                                                       0.000000000000000,
                                                       0.70710678068654742,
                                                       1.4142135613730948,
                                                       -1.4142135613730948,
                                                       -0.70710678068654742,
                                                       0.000000000000000,
                                                       0.70710678068654742,
                                                       1.4142135613730948});

    test_case.run();
}
NGRAPH_TEST(${BACKEND_NAME}, mvn_mean_variance_normalization_shared_across_channel_batch_size_2)
{
    Shape data_shape{2, 2, 5};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);

    auto mvn_func = make_shared<op::MVN>(data, true);
    auto function = make_shared<Function>(NodeVector{mvn_func}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(function);
    // data
    vector<float> data_vector(shape_size(data_shape));
    iota(begin(data_vector), end(data_vector), 0);
    test_case.add_input<float>(data_vector);

    // expected result
    test_case.add_expected_output<float>(
        data_shape,
        {-1.5666989f, -1.2185436f, -0.8703883f, -0.5222329f, -0.1740777f, 0.1740777f,  0.5222329f,
         0.8703883f,  1.2185436f,  1.5666989f,  -1.5666989f, -1.2185436f, -0.8703883f, -0.5222329f,
         -0.1740777f, 0.1740777f,  0.5222329f,  0.8703883f,  1.2185436f,  1.5666989f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, mvn_mean_variance_normalization_not_shared_across_channel_batch_size_2)
{
    Shape data_shape{2, 2, 5};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);

    auto mvn_func = make_shared<op::MVN>(data, false);
    auto function = make_shared<Function>(NodeVector{mvn_func}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(function);
    // data
    vector<float> data_vector(shape_size(data_shape));
    iota(begin(data_vector), end(data_vector), 0);
    test_case.add_input<float>(data_vector);

    // expected result
    test_case.add_expected_output<float>(
        data_shape,
        {-1.4142135f, -0.7071068f, 0.0000000f,  0.7071068f,  1.4142135f,  -1.4142135f, -0.7071068f,
         0.0000000f,  0.7071068f,  1.4142135f,  -1.4142135f, -0.7071068f, 0.0000000f,  0.7071068f,
         1.4142135f,  -1.4142135f, -0.7071068f, 0.0000000f,  0.7071068f,  1.4142135f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, grn_4d)
{
    const Shape data_shape{1, 2, 3, 4};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);
    float bias{1e-6f};

    const auto grn = make_shared<op::GRN>(data, bias);
    const auto function = make_shared<Function>(NodeVector{grn}, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(
        data_shape, {0.0766965f,  0.14142136f, 0.19611613f, 0.24253564f, 0.28216633f, 0.31622776f,
                     0.34570536f, 0.37139067f, 0.39391932f, 0.41380295f, 0.4314555f,  0.4472136f,
                     0.9970545f,  0.98994946f, 0.9805807f,  0.97014254f, 0.9593655f,  0.9486833f,
                     0.9383431f,  0.9284767f,  0.91914505f, 0.9103665f,  0.9021342f,  0.8944272f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, DISABLED_grn_2d_with_bias)
{
    const Shape data_shape{3, 4};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);
    float bias{2.25f};

    const auto grn = make_shared<op::GRN>(data, bias);
    const auto function = make_shared<Function>(NodeVector{grn}, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(data_shape,
                                         {0.5547002f,
                                          0.8f,
                                          0.8944272f,
                                          0.9363292f,
                                          0.95782626f,
                                          0.9701425f,
                                          0.9778024f,
                                          0.98287225f,
                                          0.9863939f,
                                          0.9889363f,
                                          0.9908301f,
                                          0.99227786f});
    test_case.run();
}

// TODO: Issue: 37534
NGRAPH_TEST(${BACKEND_NAME}, DISABLED_squared_difference)
{
    const auto x1 = make_shared<op::Parameter>(element::f32, Shape{2, 2});
    const auto x2 = make_shared<op::Parameter>(element::f32, Shape{2, 2});

    auto tested_op = make_shared<op::SquaredDifference>(x1, x2);
    auto function = make_shared<Function>(tested_op, ParameterVector{x1, x2});

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({1.0, 16.0, 0.0, 1.234567});
    test_case.add_input<float>({1.0, 8.0, -3.0, 3.456789});

    test_case.add_expected_output<float>(Shape{2, 2}, {0.0, 64.0, 9.0, 4.938270617284});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, DISABLED_squared_difference_broadcast)
{
    const auto x1 = make_shared<op::Parameter>(element::i32, Shape{2, 2});
    const auto x2 = make_shared<op::Parameter>(element::i32, Shape{});

    auto tested_op = make_shared<op::SquaredDifference>(x1, x2);
    auto function = make_shared<Function>(tested_op, ParameterVector{x1, x2});

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<int32_t>({1, 1, 1, 1});
    test_case.add_input<int32_t>({1});

    test_case.add_expected_output<int32_t>(Shape{2, 2}, {0, 0, 0, 0});
    test_case.run();
}

// TODO: Issue: 37511
NGRAPH_TEST(${BACKEND_NAME}, DISABLED_fake_quantize)
{
    const Shape data_shape{1, 2, 3, 4};
    const size_t levels = 4;
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto input_low = make_shared<op::Parameter>(element::f32, Shape{});
    const auto input_high = make_shared<op::Parameter>(element::f32, Shape{});
    const auto output_low = make_shared<op::Parameter>(element::f32, Shape{});
    const auto output_high = make_shared<op::Parameter>(element::f32, Shape{});

    const auto quantize =
        make_shared<op::FakeQuantize>(data, input_low, input_high, output_low, output_high, levels);
    const auto function = make_shared<Function>(
        NodeVector{quantize},
        ParameterVector{data, input_low, input_high, output_low, output_high});
    auto test_case = test::TestCase<TestEngine>(function);

    const size_t n_elements = shape_size(data_shape);
    vector<float> input_data(n_elements);
    iota(begin(input_data), end(input_data), 0);

    test_case.add_input<float>(input_data);
    // input_low
    test_case.add_input<float>({0.0f});
    // input_high
    test_case.add_input<float>({23.f});
    // output_low
    test_case.add_input<float>({2.f});
    // output_high
    test_case.add_input<float>({16.f});

    // expected result
    test_case.add_expected_output<float>(
        data_shape,
        vector<float>{2.f,          2.f,          2.f,          2.f,          6.6666669f,
                      6.6666669f,   6.6666669f,   6.6666669f,   6.6666669f,   6.6666669f,
                      6.6666669f,   6.6666669f,   11.33333301f, 11.33333301f, 11.33333301f,
                      11.33333301f, 11.33333301f, 11.33333301f, 11.33333301f, 11.33333301f,
                      16.f,         16.f,         16.f,         16.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, DISABLED_fake_quantize_with_clip)
{
    const Shape data_shape{1, 2, 3, 4};
    const size_t levels = 5;
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto input_low = make_shared<op::Parameter>(element::f32, Shape{});
    const auto input_high = make_shared<op::Parameter>(element::f32, Shape{});
    const auto output_low = make_shared<op::Parameter>(element::f32, Shape{});
    const auto output_high = make_shared<op::Parameter>(element::f32, Shape{});

    const auto quantize =
        make_shared<op::FakeQuantize>(data, input_low, input_high, output_low, output_high, levels);
    const auto function = make_shared<Function>(
        NodeVector{quantize},
        ParameterVector{data, input_low, input_high, output_low, output_high});
    auto test_case = test::TestCase<TestEngine>(function);

    const size_t n_elements = shape_size(data_shape);
    vector<float> input_data(n_elements);
    iota(begin(input_data), end(input_data), 0);

    test_case.add_input<float>(input_data);
    // input_low
    test_case.add_input<float>({3.f});
    // input_high
    test_case.add_input<float>({17.f});
    // output_low
    test_case.add_input<float>({2.f});
    // output_high
    test_case.add_input<float>({16.f});

    // expected result
    test_case.add_expected_output<float>(
        data_shape,
        vector<float>{2.f,   2.f,   2.f,   2.f,   2.f,  5.5f, 5.5f, 5.5f, 5.5f, 9.f,  9.f,  9.f,
                      12.5f, 12.5f, 12.5f, 12.5f, 16.f, 16.f, 16.f, 16.f, 16.f, 16.f, 16.f, 16.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, DISABLED_fake_quantize_with_clip_across_channels)
{
    Shape data_shape{1, 2, 5, 5};
    size_t levels = 5;
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto input_low = make_shared<op::Parameter>(element::f32, Shape{2, 1, 1});
    auto input_high = make_shared<op::Parameter>(element::f32, Shape{2, 1, 1});
    auto output_low = make_shared<op::Parameter>(element::f32, Shape{2, 1, 1});
    auto output_high = make_shared<op::Parameter>(element::f32, Shape{2, 1, 1});

    auto quantize =
        make_shared<op::FakeQuantize>(data, input_low, input_high, output_low, output_high, levels);
    auto function = make_shared<Function>(
        NodeVector{quantize},
        ParameterVector{data, input_low, input_high, output_low, output_high});
    auto test_case = test::TestCase<TestEngine>(function);

    size_t n_elements = shape_size(data_shape);
    vector<float> input_data(n_elements);
    iota(begin(input_data), end(input_data), 0);

    test_case.add_input<float>(input_data);
    // input_low
    test_case.add_input<float>(vector<float>{5.f, 30.f});
    // input_high
    test_case.add_input<float>(vector<float>{10.f, 40.f});
    // output_low
    test_case.add_input<float>(vector<float>{0.f, 50.f});
    // output_high
    test_case.add_input<float>(vector<float>{20.f, 70.f});

    // expected result
    test_case.add_expected_output<float>(
        data_shape,
        vector<float>{0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  5.0f,  10.0f, 10.0f, 15.0f,
                      20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f,
                      20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f,
                      50.0f, 50.0f, 55.0f, 55.0f, 60.0f, 60.0f, 60.0f, 65.0f, 65.0f, 70.0f,
                      70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, DISABLED_fake_quantize_pdpd)
{
    Shape data_shape{1, 2, 5, 5};
    size_t levels = 5;
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto input_low = make_shared<op::Parameter>(element::f32, Shape{2});
    auto input_high = make_shared<op::Parameter>(element::f32, Shape{2});
    auto output_low = make_shared<op::Parameter>(element::f32, Shape{2});
    auto output_high = make_shared<op::Parameter>(element::f32, Shape{2});

    auto quantize =
        make_shared<op::FakeQuantize>(data,
                                      input_low,
                                      input_high,
                                      output_low,
                                      output_high,
                                      levels,
                                      op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD, 1));
    auto function = make_shared<Function>(
        NodeVector{quantize},
        ParameterVector{data, input_low, input_high, output_low, output_high});
    auto test_case = test::TestCase<TestEngine>(function);

    size_t n_elements = shape_size(data_shape);
    vector<float> input_data(n_elements);
    iota(begin(input_data), end(input_data), 0);

    test_case.add_input<float>(input_data);
    // input_low
    test_case.add_input<float>(vector<float>{5.f, 30.f});
    // input_high
    test_case.add_input<float>(vector<float>{10.f, 40.f});
    // output_low
    test_case.add_input<float>(vector<float>{0.f, 50.f});
    // output_high
    test_case.add_input<float>(vector<float>{20.f, 70.f});

    // expected result
    test_case.add_expected_output<float>(
        data_shape,
        vector<float>{0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  5.0f,  10.0f, 10.0f, 15.0f,
                      20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f,
                      20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f,
                      50.0f, 50.0f, 55.0f, 55.0f, 60.0f, 60.0f, 60.0f, 65.0f, 65.0f, 70.0f,
                      70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, depth_to_space_space_to_depth_block_first)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    Shape dts_input_shape{2, 32, 2, 4, 2, 4};
    size_t block_size = 2;

    auto dts_input = make_shared<op::Parameter>(element::f32, dts_input_shape);
    auto depth_to_space = make_shared<op::DepthToSpace>(
        dts_input, op::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, block_size);
    auto dts_func = make_shared<Function>(NodeVector{depth_to_space}, ParameterVector{dts_input});

    auto dts_input_tensor = backend->create_tensor(element::f32, dts_input_shape);
    const auto data_size = shape_size(dts_input_shape);
    vector<float> data(data_size);
    std::iota(data.begin(), data.end(), 0);
    copy_data(dts_input_tensor, data);
    const auto dts_output_shape = depth_to_space->get_output_shape(0);
    auto dts_output_tensor = backend->create_tensor(element::f32, dts_output_shape);
    auto handle = backend->compile(dts_func);
    handle->call_with_validate({dts_output_tensor}, {dts_input_tensor});
    auto dts_result = read_vector<float>(dts_output_tensor);

    // use depth_to_space output as space_to_depth input
    auto std_input = make_shared<op::Parameter>(element::f32, dts_output_shape);
    auto space_to_depth = make_shared<op::SpaceToDepth>(
        std_input, op::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, block_size);
    auto std_func = make_shared<Function>(NodeVector{space_to_depth}, ParameterVector{std_input});

    auto std_input_tensor = backend->create_tensor(element::f32, dts_output_shape);
    copy_data(std_input_tensor, dts_result);
    auto std_output_tensor = backend->create_tensor(element::f32, dts_input_shape);
    handle = backend->compile(std_func);
    handle->call_with_validate({std_output_tensor}, {std_input_tensor});
    auto std_result = read_vector<float>(std_output_tensor);

    // expected output of space_to_depth is input of depth_to_space
    ASSERT_EQ(dts_input_shape, space_to_depth->get_output_shape(0));
    EXPECT_TRUE(test::all_close_f(std_result, data, data_size));
}

NGRAPH_TEST(${BACKEND_NAME}, depth_to_space_space_to_depth_depth_first)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    Shape dts_input_shape{2, 32, 2, 4, 2, 4};
    size_t block_size = 2;

    auto dts_input = make_shared<op::Parameter>(element::f32, dts_input_shape);
    auto depth_to_space = make_shared<op::DepthToSpace>(
        dts_input, op::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, block_size);
    auto dts_func = make_shared<Function>(NodeVector{depth_to_space}, ParameterVector{dts_input});

    auto dts_input_tensor = backend->create_tensor(element::f32, dts_input_shape);
    const auto data_size = shape_size(dts_input_shape);
    vector<float> data(data_size);
    std::iota(data.begin(), data.end(), 0);
    copy_data(dts_input_tensor, data);
    const auto dts_output_shape = depth_to_space->get_output_shape(0);
    auto dts_output_tensor = backend->create_tensor(element::f32, dts_output_shape);
    auto handle = backend->compile(dts_func);
    handle->call_with_validate({dts_output_tensor}, {dts_input_tensor});
    auto dts_result = read_vector<float>(dts_output_tensor);

    // use depth_to_space output as space_to_depth input
    auto std_input = make_shared<op::Parameter>(element::f32, dts_output_shape);
    auto space_to_depth = make_shared<op::SpaceToDepth>(
        std_input, op::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST, block_size);
    auto std_func = make_shared<Function>(NodeVector{space_to_depth}, ParameterVector{std_input});

    auto std_input_tensor = backend->create_tensor(element::f32, dts_output_shape);
    copy_data(std_input_tensor, dts_result);
    auto std_output_tensor = backend->create_tensor(element::f32, dts_input_shape);
    handle = backend->compile(std_func);
    handle->call_with_validate({std_output_tensor}, {std_input_tensor});
    auto std_result = read_vector<float>(std_output_tensor);

    // expected output of space_to_depth is input of depth_to_space
    ASSERT_EQ(dts_input_shape, space_to_depth->get_output_shape(0));
    EXPECT_TRUE(test::all_close_f(std_result, data, data_size));
}
