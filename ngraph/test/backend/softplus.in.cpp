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

NGRAPH_TEST(${BACKEND_NAME}, softplus)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{4});
    auto softplus = make_shared<op::v4::SoftPlus>(A);
    auto function = make_shared<Function>(NodeVector{softplus}, ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input(vector<float>{-1.0, 0.0, 1.0, 20.0});
    test_case.add_expected_output(
        vector<float>{0.31326166, 0.69314718, 1.3132616, 20.0});
    test_case.run();
}
