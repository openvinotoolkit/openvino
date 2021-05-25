// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "ngraph/op/result.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, result)
{
    Shape shape_a{2, 2};
    auto a = std::make_shared<op::Parameter>(element::f32, shape_a);
    auto f = std::make_shared<Function>(std::make_shared<op::v0::Result>(a), ParameterVector{a});

    std::vector<float> a_values{1, 2, 3, 5};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(a_values);
    test_case.add_expected_output<float>(shape_a, a_values);
    test_case.run();
}
