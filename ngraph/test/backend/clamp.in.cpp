//*****************************************************************************
// Copyright 2021 Intel Corporation
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

#include "util/unary_test.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, clamp_integral)
{
    Shape in_shape{6};
    float min = 0.4; // ceiled to 1
    float max = 5.6; // floored to 5

    test::make_unary_test<TestEngine, op::Clamp, element::i32>(in_shape, min, max)
        .test({-1, 3, -10, 20, 6, 2}, {1, 3, 1, 5, 5, 2});
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_integral_negative)
{
    Shape in_shape{6};
    float min = -5.6; // ceiled to -5
    float max = -0.4; // floored to -1

    test::make_unary_test<TestEngine, op::Clamp, element::i32>(in_shape, min, max)
        .test({-6, 1, -2, 0, -1, 2}, {-5, -1, -2, -1, -1, -1});
}

constexpr bool is_et_float_point(ngraph::element::Type_t et)
{
    return et == ngraph::element::Type_t::bf16 || et == ngraph::element::Type_t::f16 ||
           et == ngraph::element::Type_t::f32 || et == ngraph::element::Type_t::f64;
}

template <ngraph::element::Type_t et,
          typename ctype = ngraph::fundamental_type_for<et>,
          typename std::enable_if<!is_et_float_point(et), bool>::type = true>
void clamp_typed_test()
{
    auto sshape = Shape{4, 2};
    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<double>::infinity();
    auto ninf = -numeric_limits<double>::infinity();

    if (et == element::Type_t::u8 || et == element::Type_t::u16 || et == element::Type_t::u32)
    {
        // TODO: Fix CPU DEX / MLIR correctness bug: using signed comparison for unsigned ints
        max = (static_cast<ctype>(1) << (numeric_limits<ctype>::digits - 1)) - 1;
        pinf = static_cast<double>(max);
    }
    if (et == element::Type_t::u64)
    {
        // TODO: Fix CPU DEX / MLIR correctness bug: using signed comparison for unsigned ints
        max = (static_cast<ctype>(1) << (32 - 1)) - 1;
        pinf = static_cast<double>(max);
    }

    test::Data<et> input{min, max, 9, 10, 11, 19, 20, 21};

    // static shape
    test::make_unary_test<TestEngine, op::Clamp, et>(sshape, 10.0, 20.0)
        .test(input, {10, 20, 10, 10, 11, 19, 20, 20});
    test::make_unary_test<TestEngine, op::Clamp, et>(sshape, 10.0, pinf)
        .test(input, {10, max, 10, 10, 11, 19, 20, 21});
    test::make_unary_test<TestEngine, op::Clamp, et>(sshape, ninf, 20.0)
        .test(input, {min, 20, 9, 10, 11, 19, 20, 20});
}

template <ngraph::element::Type_t et,
          typename ctype = ngraph::fundamental_type_for<et>,
          typename std::enable_if<is_et_float_point(et), bool>::type = true>
void clamp_typed_test()
{
    auto sshape = Shape{5, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<float>::infinity();
    auto ninf = -numeric_limits<float>::infinity();

    test::Data<et> input{
        min, max, ninf, pinf, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.000001};

    // static shape
    test::make_unary_test<TestEngine, op::Clamp, et>(sshape, 0.2, 0.6)
        .test({-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
              {0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.6});

    test::make_unary_test<TestEngine, op::Clamp, et>(sshape, 10.0, 20.0)
        .test(input, {10.0, 20.0, 10.0, 20.0, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.0});

    test::make_unary_test<TestEngine, op::Clamp, et>(sshape, 10.0, pinf)
        .test(input, {10.0, max, 10.0, pinf, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.000001});

    test::make_unary_test<TestEngine, op::Clamp, et>(sshape, ninf, 20.0)
        .test(input, {min, 20.0, ninf, 20.0, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.0});
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_int8)
{
    clamp_typed_test<element::Type_t::i8>();
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_int16)
{
    clamp_typed_test<element::Type_t::i16>();
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_int32)
{
    clamp_typed_test<element::Type_t::i32>();
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_int64)
{
    clamp_typed_test<element::Type_t::i64>();
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_uint8)
{
    clamp_typed_test<element::Type_t::u8>();
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_uint16)
{
    clamp_typed_test<element::Type_t::u16>();
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_uint32)
{
    clamp_typed_test<element::Type_t::u32>();
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_uint64)
{
    clamp_typed_test<element::Type_t::u64>();
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_float)
{
    clamp_typed_test<element::Type_t::f32>();
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_float16)
{
    clamp_typed_test<element::Type_t::f16>();
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_bfloat16)
{
    clamp_typed_test<element::Type_t::bf16>();
}
