// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/op/prior_box.hpp"

#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, prior_box)
{
    op::PriorBoxAttrs attrs;
    attrs.min_size = {2.0f};
    attrs.aspect_ratio = {1.5f};
    attrs.scale_all_sizes = false;

    Shape layer_shape_shape{2};
    Shape image_shape_shape{2};
    vector<int64_t> layer_shape{2, 2};
    vector<int64_t> image_shape{10, 10};

    auto LS = op::Constant::create(element::i64, layer_shape_shape, layer_shape);
    auto IS = op::Constant::create(element::i64, image_shape_shape, image_shape);
    auto f = make_shared<Function>(make_shared<op::PriorBox>(LS, IS, attrs), ParameterVector{});
    const auto exp_shape = Shape{2, 32};
    vector<float> out{-0.75, -0.75, 1.25, 1.25, -0.974745, -0.566497,  1.47474, 1.0665,
                      -0.25, -0.75, 1.75, 1.25, -0.474745, -0.566497,  1.97474, 1.0665,
                      -0.75, -0.25, 1.25, 1.75, -0.974745, -0.0664966, 1.47474, 1.5665,
                      -0.25, -0.25, 1.75, 1.75, -0.474745, -0.0664966, 1.97474, 1.5665,
                      0.1,   0.1,   0.1,  0.1,  0.1,       0.1,        0.1,     0.1,
                      0.1,   0.1,   0.1,  0.1,  0.1,       0.1,        0.1,     0.1,
                      0.1,   0.1,   0.1,  0.1,  0.1,       0.1,        0.1,     0.1,
                      0.1,   0.1,   0.1,  0.1,  0.1,       0.1,        0.1,     0.1};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_expected_output<float>(exp_shape, out);
    test_case.run_with_tolerance_as_fp(1.0e-5f);
}
