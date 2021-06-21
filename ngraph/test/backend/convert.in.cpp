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
    std::shared_ptr<Function> CreateFunction(const Shape& input_shape,
                                             const element::Type& input_type,
                                             const element::Type& expected_output_type)
    {
        const auto in = make_shared<op::Parameter>(input_type, input_shape);
        const auto convert = make_shared<op::Convert>(in, expected_output_type);
        return make_shared<Function>(NodeVector{convert}, ParameterVector{in});
    }

    template <typename T_IN, typename T_OUT>
    void ConvertTest(const std::vector<T_IN>& input,
                     const Shape& input_shape,
                     const ngraph::element::Type& input_type,
                     const std::vector<T_OUT>& expected_output,
                     const ngraph::element::Type& expected_output_type)
    {
        const auto f = CreateFunction(input_shape, input_type, expected_output_type);
        auto test_case = test::TestCase<TestEngine>(f);
        test_case.add_input(input);
        test_case.add_expected_output(expected_output);

        test_case.run();
    }

    // TestCase doesn't support LP types
    template <typename T_IN, typename T_OUT>
    void LPConvertTest(const std::vector<T_IN>& input,
                       const Shape& input_shape,
                       const ngraph::element::Type& input_type,
                       const std::vector<T_OUT>& expected_output,
                       const ngraph::element::Type& expected_output_type)
    {
        const auto f = CreateFunction(input_shape, input_type, expected_output_type);
        auto backend = runtime::Backend::create("${BACKEND_NAME}");
        auto input_tesnor = backend->create_tensor(input_type, input_shape);
        copy_data(input_tesnor, input);
        auto output = backend->create_tensor(expected_output_type, input_shape);
        auto handle = backend->compile(f);
        handle->call_with_validate({output}, {input_tesnor});

        std::vector<T_OUT> result(expected_output.size());
        output->read(result.data(), result.size() * sizeof(uint8_t));
        EXPECT_TRUE(test::all_close(expected_output, result));
    }
} // namespace

// destination: boolean
NGRAPH_TEST(${BACKEND_NAME}, convert_u8_to_boolean)
{
    const uint8_t lowest = std::numeric_limits<uint8_t>::lowest();
    const uint8_t max = std::numeric_limits<uint8_t>::max();

    const std::vector<uint8_t> input{0, 12, 23, 0, lowest, max};
    const Shape input_shape{2, 3};
    const element::Type input_type = ngraph::element::u8;

    const std::vector<char> expected_output{0, 1, 1, 0, 0, 1};
    const element::Type expected_output_type = ngraph::element::boolean;

    ConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_i32_to_boolean)
{
    const int32_t lowest = std::numeric_limits<int32_t>::lowest();
    const int32_t max = std::numeric_limits<int32_t>::max();

    const std::vector<int32_t> input{0, -12, 23, 0, lowest, max};
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

// destination: f32
NGRAPH_TEST(${BACKEND_NAME}, convert_i4_to_f32)
{
    const std::vector<uint8_t> input{0xFE, 0xF2};
    const Shape input_shape{2, 2};
    const element::Type input_type = ngraph::element::i4;

    const std::vector<float> expected_output{-1.0f, -2.0f, -1.0f, 2.0f};
    const element::Type expected_output_type = ngraph::element::f32;

    {
        const auto f = CreateFunction(input_shape, input_type, expected_output_type);
        auto backend = runtime::Backend::create("${BACKEND_NAME}");
        auto input_tesnor = backend->create_tensor(input_type, input_shape);
        copy_data(input_tesnor, input);
        auto output = backend->create_tensor(expected_output_type, input_shape);
        auto handle = backend->compile(f);
        handle->call_with_validate({output}, {input_tesnor});

        std::vector<float> result(expected_output.size());
        output->read(result.data(), result.size() * sizeof(float));
        EXPECT_TRUE(test::all_close_f(expected_output, result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, convert_i8_to_f32)
{
    const std::vector<int8_t> input{-127, -0, 0, 127};
    const Shape input_shape{2, 2};
    const element::Type input_type = ngraph::element::i8;

    const std::vector<float> expected_output{-127.0f, -0.0f, 0.0f, 127.0f};
    const element::Type expected_output_type = ngraph::element::f32;

    ConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_i16_to_f32)
{
    const std::vector<int16_t> input{-32000, -0, 0, 32000};
    const Shape input_shape{2, 2};
    const element::Type input_type = ngraph::element::i16;

    const std::vector<float> expected_output{-32000.0f, -0.0f, 0.0f, 32000.0f};
    const element::Type expected_output_type = ngraph::element::f32;

    ConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_i32_to_f32)
{
    const std::vector<int32_t> input{-64000, -0, 0, 64000};
    const Shape input_shape{2, 2};
    const element::Type input_type = ngraph::element::i32;

    const std::vector<float> expected_output{-64000.0f, -0.0f, 0.0f, 64000.0f};
    const element::Type expected_output_type = ngraph::element::f32;

    ConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_i64_to_f32)
{
    const std::vector<int64_t> input{-64000, -0, 0, 64000};
    const Shape input_shape{2, 2};
    const element::Type input_type = ngraph::element::i64;

    const std::vector<float> expected_output{-64000.0f, -0.0f, 0.0f, 64000.0f};
    const element::Type expected_output_type = ngraph::element::f32;

    ConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u1_to_f32)
{
    const std::vector<uint8_t> input{0xA0};
    const Shape input_shape{2, 2};
    const element::Type input_type = ngraph::element::u1;

    const std::vector<float> expected_output{1.0f, 0.0f, 1.0f, 0.0f};
    const element::Type expected_output_type = ngraph::element::f32;

    {
        const auto f = CreateFunction(input_shape, input_type, expected_output_type);
        auto backend = runtime::Backend::create("${BACKEND_NAME}");
        auto input_tesnor = backend->create_tensor(input_type, input_shape);
        copy_data(input_tesnor, input);
        auto output = backend->create_tensor(expected_output_type, input_shape);
        auto handle = backend->compile(f);
        handle->call_with_validate({output}, {input_tesnor});

        std::vector<float> result(expected_output.size());
        output->read(result.data(), result.size() * sizeof(float));
        EXPECT_TRUE(test::all_close_f(expected_output, result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u4_to_f32)
{
    const std::vector<uint8_t> input{0xFB, 0x0A};
    const Shape input_shape{2, 2};
    const element::Type input_type = ngraph::element::u4;

    const std::vector<float> expected_output{15.0f, 11.0f, 0.0f, 10.0f};
    const element::Type expected_output_type = ngraph::element::f32;

    {
        const auto f = CreateFunction(input_shape, input_type, expected_output_type);
        auto backend = runtime::Backend::create("${BACKEND_NAME}");
        auto input_tesnor = backend->create_tensor(input_type, input_shape);
        copy_data(input_tesnor, input);
        auto output = backend->create_tensor(expected_output_type, input_shape);
        auto handle = backend->compile(f);
        handle->call_with_validate({output}, {input_tesnor});

        std::vector<float> result(expected_output.size());
        output->read(result.data(), result.size() * sizeof(float));
        EXPECT_TRUE(test::all_close_f(expected_output, result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u8_to_f32)
{
    const std::vector<uint8_t> input{255, 128, 32, 0};
    const Shape input_shape{2, 2};
    const element::Type input_type = ngraph::element::u8;

    const std::vector<float> expected_output{255.0f, 128.0f, 32.0f, 0.0f};
    const element::Type expected_output_type = ngraph::element::f32;

    ConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u16_to_f32)
{
    const std::vector<uint16_t> input{64000, 32000, 128, 0};
    const Shape input_shape{2, 2};
    const element::Type input_type = ngraph::element::u16;

    const std::vector<float> expected_output{64000.0f, 32000.0f, 128.0f, 0.0f};
    const element::Type expected_output_type = ngraph::element::f32;

    ConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u32_to_f32)
{
    const std::vector<uint32_t> input{4000000, 2000000, 128, 0};
    const Shape input_shape{2, 2};
    const element::Type input_type = ngraph::element::u32;

    const std::vector<float> expected_output{4000000.0f, 2000000.0f, 128.0f, 0.0f};
    const element::Type expected_output_type = ngraph::element::f32;

    ConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u64_to_f32)
{
    const std::vector<uint64_t> input{4000000, 2000000, 128, 0};
    const Shape input_shape{2, 2};
    const element::Type input_type = ngraph::element::u64;

    const std::vector<float> expected_output{4000000.0f, 2000000.0f, 128.0f, 0.0f};
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

NGRAPH_TEST(${BACKEND_NAME}, convert_f32_to_f32)
{
    const std::vector<float> input{-20.5, -15, -10.5, -0.5, 0, 0.5, 10.5, 15, 20.5};
    const Shape input_shape{3, 3};
    const element::Type input_type = ngraph::element::f32;

    const std::vector<float> expected_output{-20.5, -15, -10.5, -0.5, 0, 0.5, 10.5, 15, 20.5};
    const element::Type expected_output_type = ngraph::element::f32;

    ConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

// destination: i4
NGRAPH_TEST(${BACKEND_NAME}, convert_u1_to_i4)
{
    const std::vector<uint8_t> input{0xA0};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::u1;

    const std::vector<uint8_t> expected_output{0x10, 0x10};
    const element::Type expected_output_type = ngraph::element::i4;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u4_to_i4)
{
    const std::vector<uint8_t> input{0x12, 0x03};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::u4;

    const std::vector<uint8_t> expected_output{0x12, 0x03};
    const element::Type expected_output_type = ngraph::element::i4;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u8_to_i4)
{
    const std::vector<uint8_t> input{1, 2, 0, 3};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::u8;

    const std::vector<uint8_t> expected_output{0x12, 0x03};
    const element::Type expected_output_type = ngraph::element::i4;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u16_to_i4)
{
    const std::vector<uint16_t> input{1, 2, 0, 3};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::u16;

    const std::vector<uint8_t> expected_output{0x12, 0x03};
    const element::Type expected_output_type = ngraph::element::i4;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u32_to_i4)
{
    const std::vector<uint32_t> input{1, 2, 0, 3};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::u32;

    const std::vector<uint8_t> expected_output{0x12, 0x03};
    const element::Type expected_output_type = ngraph::element::i4;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u64_to_i4)
{
    const std::vector<uint64_t> input{1, 2, 0, 3};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::u64;

    const std::vector<uint8_t> expected_output{0x12, 0x03};
    const element::Type expected_output_type = ngraph::element::i4;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_i8_to_i4)
{
    const std::vector<int8_t> input{-1, -2, 2, 3};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::i8;

    const std::vector<uint8_t> expected_output{0xFE, 0x23};
    const element::Type expected_output_type = ngraph::element::i4;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_i16_to_i4)
{
    const std::vector<int16_t> input{-1, -2, 2, 3};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::i16;

    const std::vector<uint8_t> expected_output{0xFE, 0x23};
    const element::Type expected_output_type = ngraph::element::i4;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_i32_to_i4)
{
    const std::vector<int32_t> input{-1, -2, 2, 3};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::i32;

    const std::vector<uint8_t> expected_output{0xFE, 0x23};
    const element::Type expected_output_type = ngraph::element::i4;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_i64_to_i4)
{
    const std::vector<int64_t> input{-1, -2, 2, 3};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::i64;

    const std::vector<uint8_t> expected_output{0xFE, 0x23};
    const element::Type expected_output_type = ngraph::element::i4;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_f16_to_i4)
{
    const std::vector<float16> input{-1, -2, 0, 3};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::f16;

    const std::vector<uint8_t> expected_output{0xFE, 0x03};
    const element::Type expected_output_type = ngraph::element::i4;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_bf16_to_i4)
{
    const std::vector<bfloat16> input{-1, -2, 0, 3};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::bf16;

    const std::vector<uint8_t> expected_output{0xFE, 0x03};
    const element::Type expected_output_type = ngraph::element::i4;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_f32_to_i4)
{
    const std::vector<float> input{-1, -2, 0, 3};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::f32;

    const std::vector<uint8_t> expected_output{0xFE, 0x03};
    const element::Type expected_output_type = ngraph::element::i4;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

// destination: i8
NGRAPH_TEST(${BACKEND_NAME}, convert_u1_to_i8)
{
    const std::vector<uint8_t> input{0x81};
    const Shape input_shape{8};
    const element::Type input_type = ngraph::element::u1;

    const std::vector<int8_t> expected_output{1, 0, 0, 0, 0, 0, 0, 1};
    const element::Type expected_output_type = ngraph::element::i8;

    {
        const auto f = CreateFunction(input_shape, input_type, expected_output_type);
        auto backend = runtime::Backend::create("${BACKEND_NAME}");
        auto input_tesnor = backend->create_tensor(input_type, input_shape);
        copy_data(input_tesnor, input);
        auto output = backend->create_tensor(expected_output_type, input_shape);
        auto handle = backend->compile(f);
        handle->call_with_validate({output}, {input_tesnor});

        std::vector<int8_t> result(expected_output.size());
        output->read(result.data(), result.size() * sizeof(int8_t));
        EXPECT_TRUE(test::all_close(expected_output, result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u4_to_i8)
{
    const std::vector<uint8_t> input{0x21, 0x43};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::u4;

    const std::vector<int8_t> expected_output{2, 1, 4, 3};
    const element::Type expected_output_type = ngraph::element::i8;

    {
        const auto f = CreateFunction(input_shape, input_type, expected_output_type);
        auto backend = runtime::Backend::create("${BACKEND_NAME}");
        auto input_tesnor = backend->create_tensor(input_type, input_shape);
        copy_data(input_tesnor, input);
        auto output = backend->create_tensor(expected_output_type, input_shape);
        auto handle = backend->compile(f);
        handle->call_with_validate({output}, {input_tesnor});

        std::vector<int8_t> result(expected_output.size());
        output->read(result.data(), result.size() * sizeof(int8_t));
        EXPECT_TRUE(test::all_close(expected_output, result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, convert_i4_to_i8)
{
    const std::vector<uint8_t> input{0xFE, 0x43};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::i4;

    const std::vector<int8_t> expected_output{-1, -2, 4, 3};
    const element::Type expected_output_type = ngraph::element::i8;

    {
        const auto f = CreateFunction(input_shape, input_type, expected_output_type);
        auto backend = runtime::Backend::create("${BACKEND_NAME}");
        auto input_tesnor = backend->create_tensor(input_type, input_shape);
        copy_data(input_tesnor, input);
        auto output = backend->create_tensor(expected_output_type, input_shape);
        auto handle = backend->compile(f);
        handle->call_with_validate({output}, {input_tesnor});

        std::vector<int8_t> result(expected_output.size());
        output->read(result.data(), result.size() * sizeof(int8_t));
        EXPECT_TRUE(test::all_close(expected_output, result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u8_to_i8)
{
    const std::vector<uint8_t> input{0, 10, 15, 20, 43, 56, 78, 99, 102, 110, 128};
    const Shape input_shape{11};
    const element::Type input_type = ngraph::element::u8;

    const std::vector<int8_t> expected_output{0, 10, 15, 20, 43, 56, 78, 99, 102, 110, 127};
    const element::Type expected_output_type = ngraph::element::i8;

    ConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

// destination: i16
NGRAPH_TEST(${BACKEND_NAME}, convert_u1_to_i16)
{
    const std::vector<uint8_t> input{0x81};
    const Shape input_shape{8};
    const element::Type input_type = ngraph::element::u1;

    const std::vector<int16_t> expected_output{1, 0, 0, 0, 0, 0, 0, 1};
    const element::Type expected_output_type = ngraph::element::i16;

    {
        const auto f = CreateFunction(input_shape, input_type, expected_output_type);
        auto backend = runtime::Backend::create("${BACKEND_NAME}");
        auto input_tesnor = backend->create_tensor(input_type, input_shape);
        copy_data(input_tesnor, input);
        auto output = backend->create_tensor(expected_output_type, input_shape);
        auto handle = backend->compile(f);
        handle->call_with_validate({output}, {input_tesnor});

        std::vector<int16_t> result(expected_output.size());
        output->read(result.data(), result.size() * sizeof(int16_t));
        EXPECT_TRUE(test::all_close(expected_output, result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u4_to_i16)
{
    const std::vector<uint8_t> input{0x21, 0x43};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::u4;

    const std::vector<int16_t> expected_output{2, 1, 4, 3};
    const element::Type expected_output_type = ngraph::element::i16;

    {
        const auto f = CreateFunction(input_shape, input_type, expected_output_type);
        auto backend = runtime::Backend::create("${BACKEND_NAME}");
        auto input_tesnor = backend->create_tensor(input_type, input_shape);
        copy_data(input_tesnor, input);
        auto output = backend->create_tensor(expected_output_type, input_shape);
        auto handle = backend->compile(f);
        handle->call_with_validate({output}, {input_tesnor});

        std::vector<int16_t> result(expected_output.size());
        output->read(result.data(), result.size() * sizeof(int16_t));
        EXPECT_TRUE(test::all_close(expected_output, result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, convert_i4_to_i16)
{
    const std::vector<uint8_t> input{0xFE, 0x43};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::i4;

    const std::vector<int16_t> expected_output{-1, -2, 4, 3};
    const element::Type expected_output_type = ngraph::element::i16;

    {
        const auto f = CreateFunction(input_shape, input_type, expected_output_type);
        auto backend = runtime::Backend::create("${BACKEND_NAME}");
        auto input_tesnor = backend->create_tensor(input_type, input_shape);
        copy_data(input_tesnor, input);
        auto output = backend->create_tensor(expected_output_type, input_shape);
        auto handle = backend->compile(f);
        handle->call_with_validate({output}, {input_tesnor});

        std::vector<int16_t> result(expected_output.size());
        output->read(result.data(), result.size() * sizeof(int16_t));
        EXPECT_TRUE(test::all_close(expected_output, result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u8_to_i16)
{
    const std::vector<uint8_t> input{0, 10, 15, 20, 43, 56, 78, 99, 102, 130, 142};
    const Shape input_shape{11};
    const element::Type input_type = ngraph::element::u8;

    const std::vector<int16_t> expected_output{0, 10, 15, 20, 43, 56, 78, 99, 102, 130, 142};
    const element::Type expected_output_type = ngraph::element::i16;

    ConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

// destination: i32
NGRAPH_TEST(${BACKEND_NAME}, convert_u1_to_i32)
{
    const std::vector<uint8_t> input{0x81};
    const Shape input_shape{8};
    const element::Type input_type = ngraph::element::u1;

    const std::vector<int32_t> expected_output{1, 0, 0, 0, 0, 0, 0, 1};
    const element::Type expected_output_type = ngraph::element::i32;

    {
        const auto f = CreateFunction(input_shape, input_type, expected_output_type);
        auto backend = runtime::Backend::create("${BACKEND_NAME}");
        auto input_tesnor = backend->create_tensor(input_type, input_shape);
        copy_data(input_tesnor, input);
        auto output = backend->create_tensor(expected_output_type, input_shape);
        auto handle = backend->compile(f);
        handle->call_with_validate({output}, {input_tesnor});

        std::vector<int32_t> result(expected_output.size());
        output->read(result.data(), result.size() * sizeof(int32_t));
        EXPECT_TRUE(test::all_close(expected_output, result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u4_to_i32)
{
    const std::vector<uint8_t> input{0x21, 0x43};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::u4;

    const std::vector<int32_t> expected_output{2, 1, 4, 3};
    const element::Type expected_output_type = ngraph::element::i32;

    {
        const auto f = CreateFunction(input_shape, input_type, expected_output_type);
        auto backend = runtime::Backend::create("${BACKEND_NAME}");
        auto input_tesnor = backend->create_tensor(input_type, input_shape);
        copy_data(input_tesnor, input);
        auto output = backend->create_tensor(expected_output_type, input_shape);
        auto handle = backend->compile(f);
        handle->call_with_validate({output}, {input_tesnor});

        std::vector<int32_t> result(expected_output.size());
        output->read(result.data(), result.size() * sizeof(int32_t));
        EXPECT_TRUE(test::all_close(expected_output, result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, convert_i4_to_i32)
{
    const std::vector<uint8_t> input{0xFE, 0x43};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::i4;

    const std::vector<int32_t> expected_output{-1, -2, 4, 3};
    const element::Type expected_output_type = ngraph::element::i32;

    {
        const auto f = CreateFunction(input_shape, input_type, expected_output_type);
        auto backend = runtime::Backend::create("${BACKEND_NAME}");
        auto input_tesnor = backend->create_tensor(input_type, input_shape);
        copy_data(input_tesnor, input);
        auto output = backend->create_tensor(expected_output_type, input_shape);
        auto handle = backend->compile(f);
        handle->call_with_validate({output}, {input_tesnor});

        std::vector<int32_t> result(expected_output.size());
        output->read(result.data(), result.size() * sizeof(int32_t));
        EXPECT_TRUE(test::all_close(expected_output, result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u8_to_i32)
{
    const std::vector<uint8_t> input{0, 10, 15, 20, 43, 56, 78, 99, 102, 130, 142};
    const Shape input_shape{11};
    const element::Type input_type = ngraph::element::u8;

    const std::vector<int32_t> expected_output{0, 10, 15, 20, 43, 56, 78, 99, 102, 130, 142};
    const element::Type expected_output_type = ngraph::element::i32;

    ConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

// destination: i64
NGRAPH_TEST(${BACKEND_NAME}, convert_u1_to_i64)
{
    const std::vector<uint8_t> input{0x81};
    const Shape input_shape{8};
    const element::Type input_type = ngraph::element::u1;

    const std::vector<int64_t> expected_output{1, 0, 0, 0, 0, 0, 0, 1};
    const element::Type expected_output_type = ngraph::element::i64;

    {
        const auto f = CreateFunction(input_shape, input_type, expected_output_type);
        auto backend = runtime::Backend::create("${BACKEND_NAME}");
        auto input_tesnor = backend->create_tensor(input_type, input_shape);
        copy_data(input_tesnor, input);
        auto output = backend->create_tensor(expected_output_type, input_shape);
        auto handle = backend->compile(f);
        handle->call_with_validate({output}, {input_tesnor});

        std::vector<int64_t> result(expected_output.size());
        output->read(result.data(), result.size() * sizeof(int64_t));
        EXPECT_TRUE(test::all_close(expected_output, result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u4_to_i64)
{
    const std::vector<uint8_t> input{0x21, 0x43};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::u4;

    const std::vector<int64_t> expected_output{2, 1, 4, 3};
    const element::Type expected_output_type = ngraph::element::i64;

    {
        const auto f = CreateFunction(input_shape, input_type, expected_output_type);
        auto backend = runtime::Backend::create("${BACKEND_NAME}");
        auto input_tesnor = backend->create_tensor(input_type, input_shape);
        copy_data(input_tesnor, input);
        auto output = backend->create_tensor(expected_output_type, input_shape);
        auto handle = backend->compile(f);
        handle->call_with_validate({output}, {input_tesnor});

        std::vector<int64_t> result(expected_output.size());
        output->read(result.data(), result.size() * sizeof(int64_t));
        EXPECT_TRUE(test::all_close(expected_output, result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, convert_i4_to_i64)
{
    const std::vector<uint8_t> input{0xFE, 0x43};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::i4;

    const std::vector<int64_t> expected_output{-1, -2, 4, 3};
    const element::Type expected_output_type = ngraph::element::i64;

    {
        const auto f = CreateFunction(input_shape, input_type, expected_output_type);
        auto backend = runtime::Backend::create("${BACKEND_NAME}");
        auto input_tesnor = backend->create_tensor(input_type, input_shape);
        copy_data(input_tesnor, input);
        auto output = backend->create_tensor(expected_output_type, input_shape);
        auto handle = backend->compile(f);
        handle->call_with_validate({output}, {input_tesnor});

        std::vector<int64_t> result(expected_output.size());
        output->read(result.data(), result.size() * sizeof(int64_t));
        EXPECT_TRUE(test::all_close(expected_output, result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u8_to_i64)
{
    const std::vector<uint8_t> input{0, 10, 15, 20, 43, 56, 78, 99, 102, 130, 142};
    const Shape input_shape{11};
    const element::Type input_type = ngraph::element::u8;

    const std::vector<int64_t> expected_output{0, 10, 15, 20, 43, 56, 78, 99, 102, 130, 142};
    const element::Type expected_output_type = ngraph::element::i64;

    ConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

// destination: u1
NGRAPH_TEST(${BACKEND_NAME}, convert_u1_to_u1)
{
    const std::vector<uint8_t> input{0xF0};
    const Shape input_shape{8};
    const element::Type input_type = ngraph::element::u1;

    const std::vector<uint8_t> expected_output{0xF0};
    const element::Type expected_output_type = ngraph::element::u1;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u4_to_u1)
{
    const std::vector<uint8_t> input{0x10, 0x01, 0x00, 0x00};
    const Shape input_shape{8};
    const element::Type input_type = ngraph::element::u4;

    const std::vector<uint8_t> expected_output{0x90};
    const element::Type expected_output_type = ngraph::element::u1;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u8_to_u1)
{
    const std::vector<uint8_t> input{1, 0, 1, 0, 0, 0, 0, 1};
    const Shape input_shape{8};
    const element::Type input_type = ngraph::element::u8;

    const std::vector<uint8_t> expected_output{0xA1};
    const element::Type expected_output_type = ngraph::element::u1;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u16_to_u1)
{
    const std::vector<uint16_t> input{1, 0, 1, 0, 0, 0, 0, 1};
    const Shape input_shape{8};
    const element::Type input_type = ngraph::element::u16;

    const std::vector<uint8_t> expected_output{0xA1};
    const element::Type expected_output_type = ngraph::element::u1;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u32_to_u1)
{
    const std::vector<uint32_t> input{1, 0, 1, 0, 0, 0, 0, 1};
    const Shape input_shape{8};
    const element::Type input_type = ngraph::element::u32;

    const std::vector<uint8_t> expected_output{0xA1};
    const element::Type expected_output_type = ngraph::element::u1;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u64_to_u1)
{
    const std::vector<uint64_t> input{1, 0, 1, 0, 0, 0, 0, 1};
    const Shape input_shape{8};
    const element::Type input_type = ngraph::element::u64;

    const std::vector<uint8_t> expected_output{0xA1};
    const element::Type expected_output_type = ngraph::element::u1;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_i4_to_u1)
{
    const std::vector<uint8_t> input{0x10, 0x01, 0x00, 0x00};
    const Shape input_shape{8};
    const element::Type input_type = ngraph::element::u4;

    const std::vector<uint8_t> expected_output{0x90};
    const element::Type expected_output_type = ngraph::element::u1;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_i8_to_u1)
{
    const std::vector<int8_t> input{1, 0, 1, 0, 0, 0, 0, 1};
    const Shape input_shape{8};
    const element::Type input_type = ngraph::element::i8;

    const std::vector<uint8_t> expected_output{0xA1};
    const element::Type expected_output_type = ngraph::element::u1;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_i16_to_u1)
{
    const std::vector<int16_t> input{1, 0, 1, 0, 0, 0, 0, 1};
    const Shape input_shape{8};
    const element::Type input_type = ngraph::element::i16;

    const std::vector<uint8_t> expected_output{0xA1};
    const element::Type expected_output_type = ngraph::element::u1;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_i32_to_u1)
{
    const std::vector<int32_t> input{1, 0, 1, 0, 0, 0, 0, 1};
    const Shape input_shape{8};
    const element::Type input_type = ngraph::element::i32;

    const std::vector<uint8_t> expected_output{0xA1};
    const element::Type expected_output_type = ngraph::element::u1;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_i64_to_u1)
{
    const std::vector<int64_t> input{1, 0, 1, 0, 0, 0, 0, 1};
    const Shape input_shape{8};
    const element::Type input_type = ngraph::element::i64;

    const std::vector<uint8_t> expected_output{0xA1};
    const element::Type expected_output_type = ngraph::element::u1;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_f16_to_u1)
{
    const std::vector<float16> input{1, 0, 1, 0, 0, 0, 0, 1};
    const Shape input_shape{8};
    const element::Type input_type = ngraph::element::f16;

    const std::vector<uint8_t> expected_output{0xA1};
    const element::Type expected_output_type = ngraph::element::u1;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_bf16_to_u1)
{
    const std::vector<bfloat16> input{1, 0, 1, 0, 0, 0, 0, 1};
    const Shape input_shape{8};
    const element::Type input_type = ngraph::element::bf16;

    const std::vector<uint8_t> expected_output{0xA1};
    const element::Type expected_output_type = ngraph::element::u1;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_f32_to_u1)
{
    const std::vector<float> input{1, 0, 1, 0, 0, 0, 0, 1};
    const Shape input_shape{8};
    const element::Type input_type = ngraph::element::f32;

    const std::vector<uint8_t> expected_output{0xA1};
    const element::Type expected_output_type = ngraph::element::u1;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

// destination: u4
NGRAPH_TEST(${BACKEND_NAME}, convert_u1_to_u4)
{
    const std::vector<uint8_t> input{0xF0};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::u1;

    const std::vector<uint8_t> expected_output{0x11, 0x11};
    const element::Type expected_output_type = ngraph::element::u4;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u4_to_u4)
{
    const std::vector<uint8_t> input{0x22, 0x33};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::u4;

    const std::vector<uint8_t> expected_output{0x22, 0x33};
    const element::Type expected_output_type = ngraph::element::u4;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u8_to_u4)
{
    const std::vector<uint8_t> input{7, 0, 1, 15};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::u8;

    const std::vector<uint8_t> expected_output{0x70, 0x1F};
    const element::Type expected_output_type = ngraph::element::u4;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u16_to_u4)
{
    const std::vector<uint16_t> input{7, 0, 1, 15};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::u16;

    const std::vector<uint8_t> expected_output{0x70, 0x1F};
    const element::Type expected_output_type = ngraph::element::u4;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u32_to_u4)
{
    const std::vector<uint32_t> input{7, 0, 1, 15};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::u32;

    const std::vector<uint8_t> expected_output{0x70, 0x1F};
    const element::Type expected_output_type = ngraph::element::u4;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u64_to_u4)
{
    const std::vector<uint64_t> input{7, 0, 1, 15};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::u64;

    const std::vector<uint8_t> expected_output{0x70, 0x1F};
    const element::Type expected_output_type = ngraph::element::u4;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_i4_to_u4)
{
    const std::vector<uint8_t> input{0x22, 0x33};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::i4;

    const std::vector<uint8_t> expected_output{0x22, 0x33};
    const element::Type expected_output_type = ngraph::element::u4;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_i8_to_u4)
{
    const std::vector<int8_t> input{7, 0, 1, 15};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::i8;

    const std::vector<uint8_t> expected_output{0x70, 0x1F};
    const element::Type expected_output_type = ngraph::element::u4;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_i16_to_u4)
{
    const std::vector<int16_t> input{7, 0, 1, 15};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::i16;

    const std::vector<uint8_t> expected_output{0x70, 0x1F};
    const element::Type expected_output_type = ngraph::element::u4;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_i32_to_u4)
{
    const std::vector<int32_t> input{7, 0, 1, 15};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::i32;

    const std::vector<uint8_t> expected_output{0x70, 0x1F};
    const element::Type expected_output_type = ngraph::element::u4;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_i64_to_u4)
{
    const std::vector<int64_t> input{7, 0, 1, 15};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::i64;

    const std::vector<uint8_t> expected_output{0x70, 0x1F};
    const element::Type expected_output_type = ngraph::element::u4;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_f16_to_u4)
{
    const std::vector<float16> input{7, 0, 1, 15};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::f16;

    const std::vector<uint8_t> expected_output{0x70, 0x1F};
    const element::Type expected_output_type = ngraph::element::u4;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_bf16_to_u4)
{
    const std::vector<bfloat16> input{7, 0, 1, 15};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::bf16;

    const std::vector<uint8_t> expected_output{0x70, 0x1F};
    const element::Type expected_output_type = ngraph::element::u4;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_f32_to_u4)
{
    const std::vector<float> input{7, 0, 1, 15};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::f32;

    const std::vector<uint8_t> expected_output{0x70, 0x1F};
    const element::Type expected_output_type = ngraph::element::u4;

    LPConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

// destination: u8
NGRAPH_TEST(${BACKEND_NAME}, convert_u1_to_u8)
{
    const std::vector<uint8_t> input{0x81};
    const Shape input_shape{8};
    const element::Type input_type = ngraph::element::u1;

    const std::vector<uint8_t> expected_output{1, 0, 0, 0, 0, 0, 0, 1};
    const element::Type expected_output_type = ngraph::element::u8;

    {
        const auto f = CreateFunction(input_shape, input_type, expected_output_type);
        auto backend = runtime::Backend::create("${BACKEND_NAME}");
        auto input_tesnor = backend->create_tensor(input_type, input_shape);
        copy_data(input_tesnor, input);
        auto output = backend->create_tensor(expected_output_type, input_shape);
        auto handle = backend->compile(f);
        handle->call_with_validate({output}, {input_tesnor});

        std::vector<uint8_t> result(expected_output.size());
        output->read(result.data(), result.size() * sizeof(uint8_t));
        EXPECT_TRUE(test::all_close(expected_output, result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u4_to_u8)
{
    const std::vector<uint8_t> input{0x21, 0x43};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::u4;

    const std::vector<uint8_t> expected_output{2, 1, 4, 3};
    const element::Type expected_output_type = ngraph::element::u8;

    {
        const auto f = CreateFunction(input_shape, input_type, expected_output_type);
        auto backend = runtime::Backend::create("${BACKEND_NAME}");
        auto input_tesnor = backend->create_tensor(input_type, input_shape);
        copy_data(input_tesnor, input);
        auto output = backend->create_tensor(expected_output_type, input_shape);
        auto handle = backend->compile(f);
        handle->call_with_validate({output}, {input_tesnor});

        std::vector<uint8_t> result(expected_output.size());
        output->read(result.data(), result.size() * sizeof(uint8_t));
        EXPECT_TRUE(test::all_close(expected_output, result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, convert_i4_to_u8)
{
    const std::vector<uint8_t> input{0x12, 0x43};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::i4;

    const std::vector<uint8_t> expected_output{1, 2, 4, 3};
    const element::Type expected_output_type = ngraph::element::u8;

    {
        const auto f = CreateFunction(input_shape, input_type, expected_output_type);
        auto backend = runtime::Backend::create("${BACKEND_NAME}");
        auto input_tesnor = backend->create_tensor(input_type, input_shape);
        copy_data(input_tesnor, input);
        auto output = backend->create_tensor(expected_output_type, input_shape);
        auto handle = backend->compile(f);
        handle->call_with_validate({output}, {input_tesnor});

        std::vector<uint8_t> result(expected_output.size());
        output->read(result.data(), result.size() * sizeof(uint8_t));
        EXPECT_TRUE(test::all_close(expected_output, result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u8_to_u8)
{
    const std::vector<uint8_t> input{0, 10, 15, 20, 43, 56, 78, 99, 102, 110, 127};
    const Shape input_shape{11};
    const element::Type input_type = ngraph::element::u8;

    const std::vector<uint8_t> expected_output{0, 10, 15, 20, 43, 56, 78, 99, 102, 110, 127};
    const element::Type expected_output_type = ngraph::element::u8;

    ConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

// destination: u16
NGRAPH_TEST(${BACKEND_NAME}, convert_u1_to_u16)
{
    const std::vector<uint8_t> input{0x81};
    const Shape input_shape{8};
    const element::Type input_type = ngraph::element::u1;

    const std::vector<uint16_t> expected_output{1, 0, 0, 0, 0, 0, 0, 1};
    const element::Type expected_output_type = ngraph::element::u16;

    {
        const auto f = CreateFunction(input_shape, input_type, expected_output_type);
        auto backend = runtime::Backend::create("${BACKEND_NAME}");
        auto input_tesnor = backend->create_tensor(input_type, input_shape);
        copy_data(input_tesnor, input);
        auto output = backend->create_tensor(expected_output_type, input_shape);
        auto handle = backend->compile(f);
        handle->call_with_validate({output}, {input_tesnor});

        std::vector<uint16_t> result(expected_output.size());
        output->read(result.data(), result.size() * sizeof(uint16_t));
        EXPECT_TRUE(test::all_close(expected_output, result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u4_to_u16)
{
    const std::vector<uint8_t> input{0x21, 0x43};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::u4;

    const std::vector<uint16_t> expected_output{2, 1, 4, 3};
    const element::Type expected_output_type = ngraph::element::u16;

    {
        const auto f = CreateFunction(input_shape, input_type, expected_output_type);
        auto backend = runtime::Backend::create("${BACKEND_NAME}");
        auto input_tesnor = backend->create_tensor(input_type, input_shape);
        copy_data(input_tesnor, input);
        auto output = backend->create_tensor(expected_output_type, input_shape);
        auto handle = backend->compile(f);
        handle->call_with_validate({output}, {input_tesnor});

        std::vector<uint16_t> result(expected_output.size());
        output->read(result.data(), result.size() * sizeof(uint16_t));
        EXPECT_TRUE(test::all_close(expected_output, result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, convert_i4_to_u16)
{
    const std::vector<uint8_t> input{0x12, 0x43};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::i4;

    const std::vector<uint16_t> expected_output{1, 2, 4, 3};
    const element::Type expected_output_type = ngraph::element::u16;

    {
        const auto f = CreateFunction(input_shape, input_type, expected_output_type);
        auto backend = runtime::Backend::create("${BACKEND_NAME}");
        auto input_tesnor = backend->create_tensor(input_type, input_shape);
        copy_data(input_tesnor, input);
        auto output = backend->create_tensor(expected_output_type, input_shape);
        auto handle = backend->compile(f);
        handle->call_with_validate({output}, {input_tesnor});

        std::vector<uint16_t> result(expected_output.size());
        output->read(result.data(), result.size() * sizeof(uint16_t));
        EXPECT_TRUE(test::all_close(expected_output, result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u8_to_u16)
{
    const std::vector<uint8_t> input{0, 10, 15, 20, 43, 56, 78, 99, 102, 110, 127};
    const Shape input_shape{11};
    const element::Type input_type = ngraph::element::u8;

    const std::vector<uint16_t> expected_output{0, 10, 15, 20, 43, 56, 78, 99, 102, 110, 127};
    const element::Type expected_output_type = ngraph::element::u16;

    ConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

// destination: u32
NGRAPH_TEST(${BACKEND_NAME}, convert_u1_to_u32)
{
    const std::vector<uint8_t> input{0x81};
    const Shape input_shape{8};
    const element::Type input_type = ngraph::element::u1;

    const std::vector<uint32_t> expected_output{1, 0, 0, 0, 0, 0, 0, 1};
    const element::Type expected_output_type = ngraph::element::u32;

    {
        const auto f = CreateFunction(input_shape, input_type, expected_output_type);
        auto backend = runtime::Backend::create("${BACKEND_NAME}");
        auto input_tesnor = backend->create_tensor(input_type, input_shape);
        copy_data(input_tesnor, input);
        auto output = backend->create_tensor(expected_output_type, input_shape);
        auto handle = backend->compile(f);
        handle->call_with_validate({output}, {input_tesnor});

        std::vector<uint32_t> result(expected_output.size());
        output->read(result.data(), result.size() * sizeof(uint32_t));
        EXPECT_TRUE(test::all_close(expected_output, result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u4_to_u32)
{
    const std::vector<uint8_t> input{0x21, 0x43};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::u4;

    const std::vector<uint32_t> expected_output{2, 1, 4, 3};
    const element::Type expected_output_type = ngraph::element::u32;

    {
        const auto f = CreateFunction(input_shape, input_type, expected_output_type);
        auto backend = runtime::Backend::create("${BACKEND_NAME}");
        auto input_tesnor = backend->create_tensor(input_type, input_shape);
        copy_data(input_tesnor, input);
        auto output = backend->create_tensor(expected_output_type, input_shape);
        auto handle = backend->compile(f);
        handle->call_with_validate({output}, {input_tesnor});

        std::vector<uint32_t> result(expected_output.size());
        output->read(result.data(), result.size() * sizeof(uint32_t));
        EXPECT_TRUE(test::all_close(expected_output, result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, convert_i4_to_u32)
{
    const std::vector<uint8_t> input{0x12, 0x43};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::i4;

    const std::vector<uint32_t> expected_output{1, 2, 4, 3};
    const element::Type expected_output_type = ngraph::element::u32;

    {
        const auto f = CreateFunction(input_shape, input_type, expected_output_type);
        auto backend = runtime::Backend::create("${BACKEND_NAME}");
        auto input_tesnor = backend->create_tensor(input_type, input_shape);
        copy_data(input_tesnor, input);
        auto output = backend->create_tensor(expected_output_type, input_shape);
        auto handle = backend->compile(f);
        handle->call_with_validate({output}, {input_tesnor});

        std::vector<uint32_t> result(expected_output.size());
        output->read(result.data(), result.size() * sizeof(uint32_t));
        EXPECT_TRUE(test::all_close(expected_output, result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u8_to_u32)
{
    const std::vector<uint8_t> input{0, 10, 15, 20, 43, 56, 78, 99, 102, 110, 127};
    const Shape input_shape{11};
    const element::Type input_type = ngraph::element::u8;

    const std::vector<uint32_t> expected_output{0, 10, 15, 20, 43, 56, 78, 99, 102, 110, 127};
    const element::Type expected_output_type = ngraph::element::u32;

    ConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

// destination: u64
NGRAPH_TEST(${BACKEND_NAME}, convert_u1_to_u64)
{
    const std::vector<uint8_t> input{0x81};
    const Shape input_shape{8};
    const element::Type input_type = ngraph::element::u1;

    const std::vector<uint64_t> expected_output{1, 0, 0, 0, 0, 0, 0, 1};
    const element::Type expected_output_type = ngraph::element::u64;

    {
        const auto f = CreateFunction(input_shape, input_type, expected_output_type);
        auto backend = runtime::Backend::create("${BACKEND_NAME}");
        auto input_tesnor = backend->create_tensor(input_type, input_shape);
        copy_data(input_tesnor, input);
        auto output = backend->create_tensor(expected_output_type, input_shape);
        auto handle = backend->compile(f);
        handle->call_with_validate({output}, {input_tesnor});

        std::vector<uint64_t> result(expected_output.size());
        output->read(result.data(), result.size() * sizeof(uint64_t));
        EXPECT_TRUE(test::all_close(expected_output, result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u4_to_u64)
{
    const std::vector<uint8_t> input{0x21, 0x43};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::u4;

    const std::vector<uint64_t> expected_output{2, 1, 4, 3};
    const element::Type expected_output_type = ngraph::element::u64;

    {
        const auto f = CreateFunction(input_shape, input_type, expected_output_type);
        auto backend = runtime::Backend::create("${BACKEND_NAME}");
        auto input_tesnor = backend->create_tensor(input_type, input_shape);
        copy_data(input_tesnor, input);
        auto output = backend->create_tensor(expected_output_type, input_shape);
        auto handle = backend->compile(f);
        handle->call_with_validate({output}, {input_tesnor});

        std::vector<uint64_t> result(expected_output.size());
        output->read(result.data(), result.size() * sizeof(uint64_t));
        EXPECT_TRUE(test::all_close(expected_output, result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, convert_i4_to_u64)
{
    const std::vector<uint8_t> input{0x12, 0x43};
    const Shape input_shape{4};
    const element::Type input_type = ngraph::element::i4;

    const std::vector<uint64_t> expected_output{1, 2, 4, 3};
    const element::Type expected_output_type = ngraph::element::u64;

    {
        const auto f = CreateFunction(input_shape, input_type, expected_output_type);
        auto backend = runtime::Backend::create("${BACKEND_NAME}");
        auto input_tesnor = backend->create_tensor(input_type, input_shape);
        copy_data(input_tesnor, input);
        auto output = backend->create_tensor(expected_output_type, input_shape);
        auto handle = backend->compile(f);
        handle->call_with_validate({output}, {input_tesnor});

        std::vector<uint64_t> result(expected_output.size());
        output->read(result.data(), result.size() * sizeof(uint64_t));
        EXPECT_TRUE(test::all_close(expected_output, result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, convert_u8_to_u64)
{
    const std::vector<uint8_t> input{0, 10, 15, 20, 43, 56, 78, 99, 102, 110, 127};
    const Shape input_shape{11};
    const element::Type input_type = ngraph::element::u8;

    const std::vector<uint64_t> expected_output{0, 10, 15, 20, 43, 56, 78, 99, 102, 110, 127};
    const element::Type expected_output_type = ngraph::element::u64;

    ConvertTest(input, input_shape, input_type, expected_output, expected_output_type);
}

// jit tests
NGRAPH_TEST(${BACKEND_NAME}, convert_float32_int8)
{
    std::vector<float> f32vec = {-100.5, -20.5, -15, -10.5, -0.5, 0, 0.5, 10.5, 15, 20.5, 100.5};
    std::vector<int8_t> result(f32vec.size());
    std::vector<int8_t> i8vec(std::begin(f32vec), std::end(f32vec));
    runtime::reference::convert(f32vec.data(), result.data(), f32vec.size());
    EXPECT_EQ(result, i8vec);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_fp16_int8)
{
    std::vector<float> f32vec = {-100.5, -20.5, -15, -10.5, -0.5, 0, 0.5, 10.5, 15, 20.5, 100.5};
    std::vector<float16> f16vec(std::begin(f32vec), std::end(f32vec));
    std::vector<int8_t> i8vec(std::begin(f16vec), std::end(f16vec));
    std::vector<int8_t> result(i8vec.size());
    runtime::reference::convert(f16vec.data(), result.data(), f16vec.size());
    EXPECT_EQ(result, i8vec);
}
