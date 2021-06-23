// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backend/unary_test.hpp"

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, abs)
{
    test_unary<TestEngine, element::f32>(unary_func<op::Abs>(),
                                         {1, -2, 0, -4.75f},
                                         {1, 2, 0, 4.75f},
                                         Shape{2, 2},
                                         Shape{2, 2},
                                         MIN_FLOAT_TOLERANCE_BITS);
}
