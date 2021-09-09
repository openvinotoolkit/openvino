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

NGRAPH_TEST(${BACKEND_NAME}, max_unpool_2d_floor) {
    Shape shape{1, 1, 3, 3};
    const Strides& strides{1, 1};
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

    std::vector<float> a{1, -2, -3, 4, -5, -6, 7, 8, 9};
    std::vector<float> result{0, 0, 0, 4, 0, 0, 0, 8, 9};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({a});
    test_case.add_expected_output<float>(shape, result);
    test_case.run();
}
