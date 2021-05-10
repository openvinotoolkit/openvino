// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/reference/convert.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "runtime/backend.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/engine/test_engines.hpp"
#include "util/ndarray.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});
namespace
{
    template <typename T_IN, typename T_OUT>
    void ConvertTest(const std::vector<T_IN>& input,
                     const Shape& input_shape,
                     const ngraph::element::Type& input_type,
                     const std::vector<T_OUT>& expected_output,
                     const ngraph::element::Type& expected_output_type)
    {
        const auto in = make_shared<op::Parameter>(input_type, input_shape);
        const auto convert = make_shared<op::Convert>(in, expected_output_type);
        const auto f = make_shared<Function>(NodeVector{convert}, ParameterVector{in});

        auto test_case = test::TestCase<TestEngine>(f);
        test_case.add_input(input);
        test_case.add_expected_output(expected_output);

        test_case.run();
    }
} // namespace

// destination: f32
NGRAPH_TEST(${BACKEND_NAME}, convert_i32_to_f32)
{
    const std::vector<int32_t> input{281, 2, 3, 4};
    const Shape input_shape{2, 2};
    const element::Type input_type = ngraph::element::i32;

    const std::vector<float> expected_output{281.0f, 2.0f, 3.0f, 4.0f};
    const element::Type expected_output_type = ngraph::element::f32;

    ConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u16_to_f32)
{
    const std::vector<uint16_t> input{1, 2, 3, 4};
    const Shape input_shape{2, 2};
    const element::Type input_type = ngraph::element::u16;

    const std::vector<float> expected_output{1.0f, 2.0f, 3.0f, 4.0f};
    const element::Type expected_output_type = ngraph::element::f32;

    ConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_bf16_to_f32)
{
    const std::vector<bfloat16> input{
        0.5, 1.5, 0.5, 2.5, 1.5, 0.5, 3.5, 2.5, 0.5, 0.5, 2.5, 0.5, 0.5, 0.5, 1.5};
    const Shape input_shape{1, 1, 3, 5};
    const element::Type input_type = ngraph::element::bf16;

    const std::vector<float> expected_output(std::begin(input), std::end(input));
    const element::Type expected_output_type = ngraph::element::f32;

    ConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_f16_to_f32)
{
    const std::vector<float16> input{-20.5, -15, -10.5, -0.5, 0, 0.5, 10.5, 15, 20.5};
    const Shape input_shape{3, 3};
    const element::Type input_type = ngraph::element::f16;

    const std::vector<float> expected_output{-20.5, -15, -10.5, -0.5, 0, 0.5, 10.5, 15, 20.5};
    const element::Type expected_output_type = ngraph::element::f32;

    ConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

// destination: boolean
NGRAPH_TEST(${BACKEND_NAME}, convert_i32_to_boolean)
{
    const int32_t lowest = std::numeric_limits<int32_t>::lowest();
    const int32_t max = std::numeric_limits<int32_t>::max();

    const std::vector<int32_t> input{0, 12, 23, 0, lowest, max};
    const Shape input_shape{2, 3};
    const element::Type input_type = ngraph::element::i32;

    const std::vector<char> expected_output{0, 1, 1, 0, 1, 1};
    const element::Type expected_output_type = ngraph::element::boolean;

    ConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_f32_to_boolean)
{
    const float lowest = std::numeric_limits<float>::lowest();
    const float max = std::numeric_limits<float>::max();
    const float min = std::numeric_limits<float>::min();
    const float pos_inf = std::numeric_limits<float>::infinity();
    const float neg_inf = -std::numeric_limits<float>::infinity();

    const std::vector<float> input{0.f, 1.5745f, 0.12352f, 0.f, lowest, max, min, pos_inf, neg_inf};
    const Shape input_shape{3, 3};
    const element::Type input_type = ngraph::element::f32;

    const std::vector<char> expected_output{0, 1, 1, 0, 1, 1, 1, 1, 1};
    const element::Type expected_output_type = ngraph::element::boolean;

    ConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

// destination: bf16
NGRAPH_TEST(${BACKEND_NAME}, convert_f32_to_bf16)
{
    const std::vector<float> input{
        0.5f, 1.5f, 0.5f, 2.5f, 1.5f, 0.5f, 3.5f, 2.5f, 0.5f, 0.5f, 2.5f, 0.5f, 0.5f, 0.5f, 1.5f};
    const Shape input_shape{1, 1, 3, 5};
    const element::Type input_type = ngraph::element::f32;

    const std::vector<bfloat16> expected_output(std::begin(input), std::end(input));
    const element::Type expected_output_type = ngraph::element::bf16;

    ConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

// destination: f16
NGRAPH_TEST(${BACKEND_NAME}, convert_u8_to_f16)
{
    const std::vector<uint8_t> input{0, 10, 15, 20, 43, 56, 78, 99, 102, 130, 142};
    const Shape input_shape{11};
    const element::Type input_type = ngraph::element::u8;

    const std::vector<float16> expected_output{0, 10, 15, 20, 43, 56, 78, 99, 102, 130, 142};
    const element::Type expected_output_type = ngraph::element::f16;

    ConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}
