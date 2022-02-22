//*****************************************************************************
// Copyright (C) 2018-2022 Intel Corporation
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

#include "engines_util/test_case.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

namespace {
template <typename T>
void clamp_test(const element::Type& type,
                const PartialShape& dynamic_shape,
                const Shape& static_shape,
                const std::vector<T>& input,
                double min,
                double max,
                const std::vector<T>& output) {
    auto data = make_shared<op::Parameter>(type, dynamic_shape);
    auto clamp = make_shared<op::Clamp>(data, min, max);
    auto function = make_shared<Function>(clamp, ParameterVector{data});

    auto test_case = test::TestCase(function);
    test_case.template add_input<T>(static_shape, input);
    test_case.template add_expected_output<T>(static_shape, output);
    return test_case.run();
}
}  // namespace

NGRAPH_TEST(op_eval, clamp_float_dynamic) {
    auto type = element::f32;
    typedef float ctype;

    auto sshape = Shape{5, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<float>::infinity();
    auto ninf = -numeric_limits<float>::infinity();

    vector<ctype> input{min, max, ninf, pinf, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.000001};

    clamp_test<ctype>(type,
                      dshape,
                      sshape,
                      {-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
                      0.2,
                      0.6,
                      {0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.6});

    clamp_test<ctype>(type,
                      dshape,
                      sshape,
                      input,
                      10.0,
                      20.0,
                      {10.0, 20.0, 10.0, 20.0, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.0});

    clamp_test<ctype>(type,
                      dshape,
                      sshape,
                      input,
                      10.0,
                      pinf,
                      {10.0, max, 10.0, pinf, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.000001});

    clamp_test<ctype>(type,
                      dshape,
                      sshape,
                      input,
                      ninf,
                      20.0,
                      {min, 20.0, ninf, 20.0, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.0});
}

NGRAPH_TEST(op_eval, clamp_int8_dynamic) {
    auto type = element::i8;
    typedef int8_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<double>::infinity();
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    clamp_test<ctype>(type, dshape, sshape, input, 10.0, 20.0, {10, 20, 10, 10, 11, 19, 20, 20});
    clamp_test<ctype>(type, dshape, sshape, input, 10.0, pinf, {10, max, 10, 10, 11, 19, 20, 21});
    clamp_test<ctype>(type, dshape, sshape, input, ninf, 20.0, {min, 20, 9, 10, 11, 19, 20, 20});
}

NGRAPH_TEST(op_eval, clamp_int16_dynamic) {
    auto type = element::i16;
    typedef int16_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<double>::infinity();
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    clamp_test<ctype>(type, dshape, sshape, input, 10.0, 20.0, {10, 20, 10, 10, 11, 19, 20, 20});
    clamp_test<ctype>(type, dshape, sshape, input, 10.0, pinf, {10, max, 10, 10, 11, 19, 20, 21});
    clamp_test<ctype>(type, dshape, sshape, input, ninf, 20.0, {min, 20, 9, 10, 11, 19, 20, 20});
}

NGRAPH_TEST(op_eval, clamp_int32_dynamic) {
    auto type = element::i32;
    typedef int32_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<double>::infinity();
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    clamp_test<ctype>(type, dshape, sshape, input, 10.0, 20.0, {10, 20, 10, 10, 11, 19, 20, 20});
    clamp_test<ctype>(type, dshape, sshape, input, 10.0, pinf, {10, max, 10, 10, 11, 19, 20, 21});
    clamp_test<ctype>(type, dshape, sshape, input, ninf, 20.0, {min, 20, 9, 10, 11, 19, 20, 20});
}

NGRAPH_TEST(op_eval, clamp_int64_dynamic) {
    auto type = element::i64;
    typedef int64_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<double>::infinity();
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    clamp_test<ctype>(type, dshape, sshape, input, 10.0, 20.0, {10, 20, 10, 10, 11, 19, 20, 20});
    clamp_test<ctype>(type, dshape, sshape, input, 10.0, pinf, {10, max, 10, 10, 11, 19, 20, 21});
    clamp_test<ctype>(type, dshape, sshape, input, ninf, 20.0, {min, 20, 9, 10, 11, 19, 20, 20});
}

NGRAPH_TEST(op_eval, clamp_uint8_dynamic) {
    auto type = element::u8;
    typedef uint8_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    // TODO: Fix CPU DEX / MLIR correctness bug: using signed comparison for unsigned ints
    // auto max = numeric_limits<ctype>::max();
    // auto pinf = numeric_limits<double>::infinity();
    ctype max = (static_cast<ctype>(1) << (numeric_limits<ctype>::digits - 1)) - 1;
    auto pinf = static_cast<double>(max);
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    clamp_test<ctype>(type, dshape, sshape, input, 10.0, 20.0, {10, 20, 10, 10, 11, 19, 20, 20});
    clamp_test<ctype>(type, dshape, sshape, input, 10.0, pinf, {10, max, 10, 10, 11, 19, 20, 21});
    clamp_test<ctype>(type, dshape, sshape, input, ninf, 20.0, {min, 20, 9, 10, 11, 19, 20, 20});
}

NGRAPH_TEST(op_eval, clamp_uint16_dynamic) {
    auto type = element::u16;
    typedef uint16_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    // TODO: Fix CPU DEX / MLIR correctness bug: using signed comparison for unsigned ints
    // auto max = numeric_limits<ctype>::max();
    // auto pinf = numeric_limits<double>::infinity();
    ctype max = (static_cast<ctype>(1) << (numeric_limits<ctype>::digits - 1)) - 1;
    auto pinf = static_cast<double>(max);
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    // dynamic shape
    clamp_test<ctype>(type, dshape, sshape, input, 10.0, 20.0, {10, 20, 10, 10, 11, 19, 20, 20});
    clamp_test<ctype>(type, dshape, sshape, input, 10.0, pinf, {10, max, 10, 10, 11, 19, 20, 21});
    clamp_test<ctype>(type, dshape, sshape, input, ninf, 20.0, {min, 20, 9, 10, 11, 19, 20, 20});
}

NGRAPH_TEST(op_eval, clamp_uint32_dynamic) {
    auto type = element::u32;
    typedef uint32_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    // TODO: Fix CPU DEX / MLIR correctness bug: using signed comparison for unsigned ints
    // auto max = numeric_limits<ctype>::max();
    // auto pinf = numeric_limits<double>::infinity();
    ctype max = (static_cast<ctype>(1) << (numeric_limits<ctype>::digits - 1)) - 1;
    auto pinf = static_cast<double>(max);
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    clamp_test<ctype>(type, dshape, sshape, input, 10.0, 20.0, {10, 20, 10, 10, 11, 19, 20, 20});
    clamp_test<ctype>(type, dshape, sshape, input, 10.0, pinf, {10, max, 10, 10, 11, 19, 20, 21});
    clamp_test<ctype>(type, dshape, sshape, input, ninf, 20.0, {min, 20, 9, 10, 11, 19, 20, 20});
}

NGRAPH_TEST(op_eval, clamp_uint64_dynamic) {
    auto type = element::u64;
    typedef uint64_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    // TODO: Fix CPU DEX / MLIR correctness bug: using signed comparison for unsigned ints
    // auto max = numeric_limits<ctype>::max();
    // auto pinf = numeric_limits<double>::infinity();
    ctype max = (static_cast<ctype>(1) << (32 - 1)) - 1;
    auto pinf = static_cast<double>(max);
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    clamp_test<ctype>(type, dshape, sshape, input, 10.0, 20.0, {10, 20, 10, 10, 11, 19, 20, 20});
    clamp_test<ctype>(type, dshape, sshape, input, 10.0, pinf, {10, max, 10, 10, 11, 19, 20, 21});
    clamp_test<ctype>(type, dshape, sshape, input, ninf, 20.0, {min, 20, 9, 10, 11, 19, 20, 20});
}

NGRAPH_TEST(op_eval, clamp_float16_dynamic) {
    auto type = element::f16;
    typedef float16 ctype;

    auto sshape = Shape{5, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<float>::infinity();
    auto ninf = -numeric_limits<float>::infinity();

    vector<ctype> input{min, max, ninf, pinf, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.000001};

    clamp_test<ctype>(type,
                      dshape,
                      sshape,
                      {-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
                      0.2,
                      0.6,
                      {0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.6});

    clamp_test<ctype>(type,
                      dshape,
                      sshape,
                      input,
                      10.0,
                      20.0,
                      {10.0, 20.0, 10.0, 20.0, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.0});

    clamp_test<ctype>(type,
                      dshape,
                      sshape,
                      input,
                      10.0,
                      pinf,
                      {10.0, max, 10.0, pinf, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.000001});

    clamp_test<ctype>(type,
                      dshape,
                      sshape,
                      input,
                      ninf,
                      20.0,
                      {min, 20.0, ninf, 20.0, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.0});
}

NGRAPH_TEST(op_eval, clamp_bfloat16_dynamic) {
    auto type = element::bf16;
    typedef bfloat16 ctype;

    auto sshape = Shape{5, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<float>::infinity();
    auto ninf = -numeric_limits<float>::infinity();

    vector<ctype> input{min, max, ninf, pinf, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.000001};

    clamp_test<ctype>(type,
                      dshape,
                      sshape,
                      {-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
                      0.2,
                      0.6,
                      {0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.6});

    clamp_test<ctype>(type,
                      dshape,
                      sshape,
                      input,
                      10.0,
                      20.0,
                      {10.0, 20.0, 10.0, 20.0, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.0});

    clamp_test<ctype>(type,
                      dshape,
                      sshape,
                      input,
                      10.0,
                      pinf,
                      {10.0, max, 10.0, pinf, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.000001});

    clamp_test<ctype>(type,
                      dshape,
                      sshape,
                      input,
                      ninf,
                      20.0,
                      {min, 20.0, ninf, 20.0, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.0});
}
