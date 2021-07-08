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

namespace
{
    template <typename T>
    std::vector<T> iota_vector(size_t size, T first_value = {})
    {
        std::vector<T> d(size);
        std::iota(begin(d), end(d), first_value);
        return d;
    }
} // namespace

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

    test_case.add_input<float>(iota_vector<float>(shape_size(data_shape)));

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

    test_case.add_input<float>(iota_vector<float>(shape_size(data_shape)));

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

    test_case.add_input<float>(iota_vector<float>(shape_size(data_shape)));

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
    auto input_low = op::Constant::create(element::f32, Shape{2, 1, 1, 1}, {5.f, 30.f});
    auto input_high = op::Constant::create(element::f32, Shape{2, 1, 1, 1}, {10.f, 40.f});
    auto output_low = op::Constant::create(element::f32, Shape{2, 1, 1, 1}, {0.f, 50.f});
    auto output_high = op::Constant::create(element::f32, Shape{2, 1, 1, 1}, {20.f, 70.f});

    auto quantize = std::make_shared<op::FakeQuantize>(
        data, input_low, input_high, output_low, output_high, levels, broadcast);
    auto function = std::make_shared<Function>(NodeVector{quantize}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>(iota_vector<float>(shape_size(data_shape)));

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
    auto input_low = op::Constant::create(element::f32, Shape{2, 1, 1}, {5.f, 30.f});
    auto input_high = op::Constant::create(element::f32, Shape{2, 1, 1}, {10.f, 40.f});
    auto output_low = op::Constant::create(element::f32, Shape{2, 1, 1}, {0.f, 50.f});
    auto output_high = op::Constant::create(element::f32, Shape{2, 1, 1}, {20.f, 70.f});

    auto quantize = std::make_shared<op::FakeQuantize>(
        data, input_low, input_high, output_low, output_high, levels, broadcast);
    auto function = std::make_shared<Function>(NodeVector{quantize}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>(iota_vector<float>(shape_size(data_shape)));

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
