// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <string>
#include <vector>

#include "ngraph/op/parameter.hpp"
#include "ngraph/output_vector.hpp"
#include "ngraph/shape.hpp"

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif

#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, DISABLED_smoke_fake_quantize_f32)
{
    const Shape data_shape{1, 30, 1024, 10240};
    const size_t levels = 256;
    const auto data = std::make_shared<op::Parameter>(element::f32, data_shape);
    const auto input_low = op::Constant::create(element::f32, Shape{}, {0.f});
    const auto input_high = op::Constant::create(element::f32, Shape{}, {23.f});
    const auto output_low = op::Constant::create(element::f32, Shape{}, {2.f});
    const auto output_high = op::Constant::create(element::f32, Shape{}, {16.f});

    const auto quantize = std::make_shared<op::FakeQuantize>(
        data, input_low, input_high, output_low, output_high, levels);
    const auto function = std::make_shared<Function>(NodeVector{quantize}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(function);

    const size_t n_elements = shape_size(data_shape);
    std::vector<float> input_data(n_elements);
    iota(begin(input_data), end(input_data), 0);

    test_case.add_input<float>(input_data);

    // expected result
    test_case.add_expected_output<float>(data_shape, input_data);

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, DISABLED_smoke_fake_quantize_with_clip_across_channels_f32)
{
    Shape data_shape{1, 3, 1024, 102400};
    size_t levels = 256;
    auto data = std::make_shared<op::Parameter>(element::f32, data_shape);
    auto input_low = op::Constant::create(element::f32, Shape{3, 1, 1}, {5.f, 30.f, 10.f});
    auto input_high = op::Constant::create(element::f32, Shape{3, 1, 1}, {10.f, 40.f, 20.f});
    auto output_low = op::Constant::create(element::f32, Shape{3, 1, 1}, {0.f, 50.f, 70.f});
    auto output_high = op::Constant::create(element::f32, Shape{3, 1, 1}, {20.f, 70.f, 120.f});

    auto quantize = std::make_shared<op::FakeQuantize>(
        data, input_low, input_high, output_low, output_high, levels);
    auto function = std::make_shared<Function>(NodeVector{quantize}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(function);

    size_t n_elements = shape_size(data_shape);
    std::vector<float> input_data(n_elements);
    iota(begin(input_data), end(input_data), 0);

    test_case.add_input<float>(input_data);
    // expected result
    test_case.add_expected_output<float>(data_shape, input_data);

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, DISABLED_smoke_fake_quantize_i16)
{
    const Shape data_shape{1, 3, 1024, 1024};
    const size_t levels = 256;
    const auto data = std::make_shared<op::Parameter>(element::i16, data_shape);
    const auto input_low = op::Constant::create(element::i16, Shape{}, {0});
    const auto input_high = op::Constant::create(element::i16, Shape{}, {23});
    const auto output_low = op::Constant::create(element::i16, Shape{}, {2});
    const auto output_high = op::Constant::create(element::i16, Shape{}, {16});

    const auto quantize = std::make_shared<op::FakeQuantize>(
        data, input_low, input_high, output_low, output_high, levels);
    const auto function = std::make_shared<Function>(NodeVector{quantize}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(function);

    const size_t n_elements = shape_size(data_shape);
    std::vector<int16_t> input_data(n_elements);
    iota(begin(input_data), end(input_data), 0);

    test_case.add_input<int16_t>(input_data);

    // expected result
    test_case.add_expected_output<int16_t>(data_shape, input_data);

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, DISABLED_smoke_fake_quantize_with_clip_across_channels_i16)
{
    Shape data_shape{1, 3, 1024, 1024};
    size_t levels = 256;
    auto data = std::make_shared<op::Parameter>(element::i16, data_shape);
    auto input_low = op::Constant::create(element::i16, Shape{3, 1, 1}, {5, 30, 10});
    auto input_high = op::Constant::create(element::i16, Shape{3, 1, 1}, {10, 40, 20});
    auto output_low = op::Constant::create(element::i16, Shape{3, 1, 1}, {0, 50, 70});
    auto output_high = op::Constant::create(element::i16, Shape{3, 1, 1}, {20, 70, 120});

    auto quantize = std::make_shared<op::FakeQuantize>(
        data, input_low, input_high, output_low, output_high, levels);
    auto function = std::make_shared<Function>(NodeVector{quantize}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(function);

    size_t n_elements = shape_size(data_shape);
    std::vector<int16_t> input_data(n_elements);
    iota(begin(input_data), end(input_data), 0);

    test_case.add_input<int16_t>(input_data);
    // expected result
    test_case.add_expected_output<int16_t>(data_shape, input_data);

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, fake_quantize)
{
    const Shape data_shape{1, 2, 3, 4};
    const size_t levels = 4;
    const auto data = std::make_shared<op::Parameter>(element::f32, data_shape);
    const auto input_low = op::Constant::create(element::f32, Shape{}, {0.f});
    const auto input_high = op::Constant::create(element::f32, Shape{}, {23.f});
    const auto output_low = op::Constant::create(element::f32, Shape{}, {2.f});
    const auto output_high = op::Constant::create(element::f32, Shape{}, {16.f});

    const auto quantize = std::make_shared<op::FakeQuantize>(
        data, input_low, input_high, output_low, output_high, levels);
    const auto function = std::make_shared<Function>(NodeVector{quantize}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(function);

    const size_t n_elements = shape_size(data_shape);
    std::vector<float> input_data(n_elements);
    iota(begin(input_data), end(input_data), 0);

    test_case.add_input<float>(input_data);

    // expected result
    test_case.add_expected_output<float>(
        data_shape,
        std::vector<float>{2.f,          2.f,          2.f,          2.f,          6.6666669f,
                           6.6666669f,   6.6666669f,   6.6666669f,   6.6666669f,   6.6666669f,
                           6.6666669f,   6.6666669f,   11.33333301f, 11.33333301f, 11.33333301f,
                           11.33333301f, 11.33333301f, 11.33333301f, 11.33333301f, 11.33333301f,
                           16.f,         16.f,         16.f,         16.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, fake_quantize_with_clip)
{
    const Shape data_shape{1, 2, 3, 4};
    const size_t levels = 5;
    const auto data = std::make_shared<op::Parameter>(element::f32, data_shape);
    const auto input_low = op::Constant::create(element::f32, Shape{}, {3.f});
    const auto input_high = op::Constant::create(element::f32, Shape{}, {17.f});
    const auto output_low = op::Constant::create(element::f32, Shape{}, {2.f});
    const auto output_high = op::Constant::create(element::f32, Shape{}, {16.f});

    const auto quantize = std::make_shared<op::FakeQuantize>(
        data, input_low, input_high, output_low, output_high, levels);
    const auto function = std::make_shared<Function>(NodeVector{quantize}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(function);

    const size_t n_elements = shape_size(data_shape);
    std::vector<float> input_data(n_elements);
    iota(begin(input_data), end(input_data), 0);

    test_case.add_input<float>(input_data);

    // expected result
    test_case.add_expected_output<float>(
        data_shape, std::vector<float>{2.f,  2.f,  2.f,  2.f,  2.f,   5.5f,  5.5f,  5.5f,
                                       5.5f, 9.f,  9.f,  9.f,  12.5f, 12.5f, 12.5f, 12.5f,
                                       16.f, 16.f, 16.f, 16.f, 16.f,  16.f,  16.f,  16.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, fake_quantize_with_clip_across_channels)
{
    Shape data_shape{1, 2, 5, 5};
    size_t levels = 5;
    auto data = std::make_shared<op::Parameter>(element::f32, data_shape);
    auto input_low = op::Constant::create(element::f32, Shape{2, 1, 1}, {5.f, 30.f});
    auto input_high = op::Constant::create(element::f32, Shape{2, 1, 1}, {10.f, 40.f});
    auto output_low = op::Constant::create(element::f32, Shape{2, 1, 1}, {0.f, 50.f});
    auto output_high = op::Constant::create(element::f32, Shape{2, 1, 1}, {20.f, 70.f});

    auto quantize = std::make_shared<op::FakeQuantize>(
        data, input_low, input_high, output_low, output_high, levels);
    auto function = std::make_shared<Function>(NodeVector{quantize}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(function);

    size_t n_elements = shape_size(data_shape);
    std::vector<float> input_data(n_elements);
    iota(begin(input_data), end(input_data), 0);

    test_case.add_input<float>(input_data);

    // expected result
    test_case.add_expected_output<float>(
        data_shape,
        std::vector<float>{0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  5.0f,  10.0f, 10.0f, 15.0f,
                           20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f,
                           20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f,
                           50.0f, 50.0f, 55.0f, 55.0f, 60.0f, 60.0f, 60.0f, 65.0f, 65.0f, 70.0f,
                           70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f});

    test_case.run();
}


NGRAPH_TEST(${BACKEND_NAME}, fake_quantize_pdpd)
{
    Shape data_shape{1, 2, 5, 5};
    size_t levels = 5;
    const auto broadcast = op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD, 1);
    auto data = std::make_shared<op::Parameter>(element::f32, data_shape);
    auto input_low = op::Constant::create(element::f32, Shape{2}, {5.f, 30.f});
    auto input_high = op::Constant::create(element::f32, Shape{2}, {10.f, 40.f});
    auto output_low = op::Constant::create(element::f32, Shape{2}, {0.f, 50.f});
    auto output_high = op::Constant::create(element::f32, Shape{2}, {20.f, 70.f});

    auto quantize = std::make_shared<op::FakeQuantize>(
        data, input_low, input_high, output_low, output_high, levels, broadcast);
    auto function = std::make_shared<Function>(NodeVector{quantize}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(function);

    size_t n_elements = shape_size(data_shape);
    std::vector<float> input_data(n_elements);
    iota(begin(input_data), end(input_data), 0);

    test_case.add_input<float>(input_data);

    // expected result
    test_case.add_expected_output<float>(
        data_shape,
        std::vector<float>{0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  5.0f,  10.0f, 10.0f, 15.0f,
                           20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f,
                           20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f,
                           50.0f, 50.0f, 55.0f, 55.0f, 60.0f, 60.0f, 60.0f, 65.0f, 65.0f, 70.0f,
                           70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, fake_quantize_pdpd_default_axis)
{
    Shape data_shape{1, 2, 5, 5};
    size_t levels = 5;
    const auto broadcast = op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD, -1);
    auto data = std::make_shared<op::Parameter>(element::f32, data_shape);
    auto input_low = op::Constant::create(element::f32, Shape{2,1,1}, {5.f, 30.f});
    auto input_high = op::Constant::create(element::f32, Shape{2,1,1}, {10.f, 40.f});
    auto output_low = op::Constant::create(element::f32, Shape{2,1,1}, {0.f, 50.f});
    auto output_high = op::Constant::create(element::f32, Shape{2,1,1}, {20.f, 70.f});

    auto quantize = std::make_shared<op::FakeQuantize>(
        data, input_low, input_high, output_low, output_high, levels, broadcast);
    auto function = std::make_shared<Function>(NodeVector{quantize}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(function);

    size_t n_elements = shape_size(data_shape);
    std::vector<float> input_data(n_elements);
    iota(begin(input_data), end(input_data), 0);

    test_case.add_input<float>(input_data);

    // expected result
    test_case.add_expected_output<float>(
        data_shape,
        std::vector<float>{0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  5.0f,  10.0f, 10.0f, 15.0f,
                           20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f,
                           20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f,
                           50.0f, 50.0f, 55.0f, 55.0f, 60.0f, 60.0f, 60.0f, 65.0f, 65.0f, 70.0f,
                           70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f});

    test_case.run();
}
