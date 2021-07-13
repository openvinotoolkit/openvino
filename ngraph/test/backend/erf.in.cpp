// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>

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

NGRAPH_TEST(${BACKEND_NAME}, erf_1D)
{
    Shape shape{8};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Erf>(A), ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    vector<float> in_vec{-4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    vector<float> out_vec{in_vec};

    std::transform(
        out_vec.begin(), out_vec.end(), out_vec.begin(), [](float x) -> float { return erf(x); });

    test_case.add_input<float>(in_vec);
    test_case.add_expected_output<float>(shape, out_vec);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, erf_4D)
{
    Shape shape{2, 2, 2, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Erf>(A), ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    vector<float> in_vec{-8.8f, -5.4f, -3.1f, -2.5f, 0, 1.7f, 3.2f, 4.9f};
    vector<float> out_vec{in_vec};

    std::transform(
        out_vec.begin(), out_vec.end(), out_vec.begin(), [](float x) -> float { return erf(x); });

    test_case.add_input<float>(in_vec);
    test_case.add_expected_output<float>(shape, out_vec);
    test_case.run();
}
