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

NGRAPH_TEST(${BACKEND_NAME}, elu)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{3, 2});
    auto elu = make_shared<op::Elu>(A, 0.5f);
    auto function = make_shared<Function>(NodeVector{elu}, ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input(vector<float>{-2.f, 3.f, -2.f, 1.f, -1.f, 0.f});
    test_case.add_expected_output(
        vector<float>{-0.432332358f, 3.f, -0.432332358f, 1.f, -0.316060279f, 0.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, elu_negative_alpha)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{3, 2});
    auto elu = make_shared<op::Elu>(A, -1.f);
    auto function = make_shared<Function>(NodeVector{elu}, ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input(vector<float>{-2.f, 3.f, -2.f, 1.f, -1.f, 0.f});
    test_case.add_expected_output(
        vector<float>{0.864664717f, 3.f, 0.864664717f, 1.f, 0.632120559f, 0.f});
    test_case.run();
}