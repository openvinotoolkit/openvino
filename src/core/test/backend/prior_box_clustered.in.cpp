// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/op/prior_box_clustered.hpp"

#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, prior_box_clustered)
{
    op::PriorBoxClusteredAttrs attrs;
    attrs.widths = {3.0f};
    attrs.heights = {3.0f};
    attrs.clip = true;

    Shape layer_shape_shape{2};
    Shape image_shape_shape{2};
    vector<int64_t> layer_shape{2, 2};
    vector<int64_t> image_shape{10, 10};

    auto LS = op::Constant::create(element::i64, layer_shape_shape, layer_shape);
    auto IS = op::Constant::create(element::i64, image_shape_shape, image_shape);
    auto f = make_shared<Function>(make_shared<op::PriorBoxClustered>(LS, IS, attrs), ParameterVector{});
    const auto exp_shape = Shape{2, 16};
    vector<float> out{0,    0,        0.15f, 0.15f,    0.34999f, 0,        0.64999f, 0.15f,
                      0,    0.34999f, 0.15f, 0.64999f, 0.34999f, 0.34999f, 0.64999f, 0.64999f,
                      0.1f, 0.1f,     0.1f,  0.1f,     0.1f,     0.1f,     0.1f,     0.1f,
                      0.1f, 0.1f,     0.1f,  0.1f,     0.1f,     0.1f,     0.1f,     0.1f};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_expected_output<float>(exp_shape, out);
    test_case.run_with_tolerance_as_fp(1.0e-5f);
}

NGRAPH_TEST(${BACKEND_NAME}, prior_box_clustered_non_default_variances)
{
    op::PriorBoxClusteredAttrs attrs;
    attrs.widths = {3.0f};
    attrs.heights = {3.0f};
    attrs.clip = true;
    attrs.variances = {0.1f, 0.2f, 0.3f, 0.4f};

    Shape layer_shape_shape{2};
    Shape image_shape_shape{2};
    vector<int64_t> layer_shape{2, 2};
    vector<int64_t> image_shape{10, 10};

    auto LS = op::Constant::create(element::i64, layer_shape_shape, layer_shape);
    auto IS = op::Constant::create(element::i64, image_shape_shape, image_shape);
    auto f = make_shared<Function>(make_shared<op::PriorBoxClustered>(LS, IS, attrs), ParameterVector{});
    const auto exp_shape = Shape{2, 16};
    vector<float> out{0,    0,        0.15f, 0.15f,    0.34999f, 0,        0.64999f, 0.15f,
                      0,    0.34999f, 0.15f, 0.64999f, 0.34999f, 0.34999f, 0.64999f, 0.64999f,
                      0.1f, 0.2f,     0.3f,  0.4f,     0.1f,     0.2f,     0.3f,     0.4f,
                      0.1f, 0.2f,     0.3f,  0.4f,     0.1f,     0.2f,     0.3f,     0.4f};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_expected_output<float>(exp_shape, out);
    test_case.run_with_tolerance_as_fp(1.0e-5f);
}
