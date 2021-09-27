// Copyright (C) 2018-2021 Intel Corporation
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

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, max_unpool_2d) {
    Shape shape{1, 1, 7, 7};
    const Strides& strides{2, 2};
    const Shape& pads_begin{0, 0};
    const Shape& pads_end{0, 0};
    const Shape& kernel{2, 2};
    const op::RoundingType rounding_type = op::RoundingType::FLOOR;
    const op::PadType pad_type = op::PadType::NOTSET;

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto maxPool = make_shared<op::v1::MaxPool>(A, strides, pads_begin, pads_end, kernel, rounding_type, pad_type);
    auto relu = std::make_shared<op::Relu>(maxPool);
    auto maxUnpool = make_shared<op::v8::MaxUnpool>(A, maxPool, relu, A);
    auto f = make_shared<Function>(maxUnpool, ParameterVector{A});

    std::vector<float> a;
    for (int i = 1; i < 50; ++i)
        a.push_back(i);
    std::vector<float> result{0, 0, 0, 0, 0, 0, 0,
                              0, 9, 0, 11, 0, 13 ,0,
                              0, 0, 0, 0, 0, 0, 0,
                              0, 23, 0, 25, 0, 27, 0,
                              0, 0, 0, 0, 0, 0, 0,
                              0, 37, 0, 39, 0, 41, 0,
                              0, 0, 0, 0, 0, 0, 0};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({a});
    test_case.add_expected_output<float>(shape, result);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, max_unpool_2d_output_smaller_then_input) {
    Shape shape{1, 1, 6, 6};
    Shape pooling_shape{1, 1, 4, 6};
    const Strides& strides{2, 2};
    const Shape& pads_begin{0, 0};
    const Shape& pads_end{0, 0};
    const Shape& kernel{2, 2};
    const op::RoundingType rounding_type = op::RoundingType::FLOOR;
    const op::PadType pad_type = op::PadType::NOTSET;

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, pooling_shape);
    auto maxPool = make_shared<op::v1::MaxPool>(A, strides, pads_begin, pads_end, kernel, rounding_type, pad_type);
    auto relu = std::make_shared<op::Relu>(maxPool);
    auto maxUnpool = make_shared<op::v8::MaxUnpool>(A, maxPool, relu, B);
    auto f = make_shared<Function>(maxUnpool, ParameterVector{A, B});

    std::vector<float> a;
    for (int i = 1; i < 37; ++i)
        a.push_back(i);
    std::vector<float> b(24, 0);
    std::vector<float> result{0, 0, 0, 0, 0, 0,
                              0, 8, 0, 10, 0, 12,
                              0, 0, 0, 0, 0, 0,
                              0, 20, 0, 22, 0, 24};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(pooling_shape, result);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, max_unpool_2d_reshape_output) {
    Shape shape{1, 1, 6, 7};
    Shape pooling_shape{1, 1, 7, 6};
    const Strides& strides{2, 2};
    const Shape& pads_begin{0, 0};
    const Shape& pads_end{0, 0};
    const Shape& kernel{2, 2};
    const op::RoundingType rounding_type = op::RoundingType::FLOOR;
    const op::PadType pad_type = op::PadType::NOTSET;

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, pooling_shape);
    auto maxPool = make_shared<op::v1::MaxPool>(A, strides, pads_begin, pads_end, kernel, rounding_type, pad_type);
    auto relu = std::make_shared<op::Relu>(maxPool);
    auto maxUnpool = make_shared<op::v8::MaxUnpool>(A, maxPool, relu, B);
    auto f = make_shared<Function>(maxUnpool, ParameterVector{A, B});

    std::vector<float> a;
    for (int i = 1; i < 43; ++i)
        a.push_back(i);
    std::vector<float> b(42, 0);
    std::vector<float> result{0, 0, 0, 0, 0, 0,
                              0, 0, 9, 0, 11, 0,
                              13, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 23, 0,
                              25, 0, 27, 0, 0, 0,
                              0, 0, 0, 0, 0, 0,
                              37, 0, 39, 0, 41, 0};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(pooling_shape, result);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, max_unpool_2d_kernel_3) {
    Shape shape{1, 1, 7, 9};
    const Strides& strides{2, 2};
    const Shape& pads_begin{0, 0};
    const Shape& pads_end{0, 0};
    const Shape& kernel{3, 3};
    const op::RoundingType rounding_type = op::RoundingType::FLOOR;
    const op::PadType pad_type = op::PadType::NOTSET;

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto maxPool = make_shared<op::v1::MaxPool>(A, strides, pads_begin, pads_end, kernel, rounding_type, pad_type);
    auto relu = std::make_shared<op::Relu>(maxPool);
    auto maxUnpool = make_shared<op::v8::MaxUnpool>(A, maxPool, relu, A);
    auto f = make_shared<Function>(maxUnpool, ParameterVector{A});

    std::vector<float> a;
    for (int i = 1; i < 64; ++i)
        a.push_back(i);
    std::vector<float> result{0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 21, 0, 23, 0, 25, 0, 27,
                              0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 39, 0, 41, 0, 43, 0, 45,
                              0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 57, 0, 59, 0, 61, 0, 63};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({a});
    test_case.add_expected_output<float>(shape, result);
    test_case.run();
}
