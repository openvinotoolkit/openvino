// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "engines_util/test_case.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(op_eval, bucketize_empty_buckets) {
    Shape data_shape{1, 1, 3};
    Shape bucket_shape{0};

    const auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto buckets = make_shared<op::Parameter>(element::f32, bucket_shape);
    const auto bucketize = make_shared<op::v3::Bucketize>(data, buckets);
    const auto f = make_shared<Function>(bucketize, ParameterVector{data, buckets});

    vector<float> data_vect = {8.f, 1.f, 2.f};
    vector<float> buckets_vect;
    vector<int> expected_vect = {0, 0, 0};

    auto test_case = test::TestCase(f);
    test_case.add_input<float>(data_shape, data_vect);
    test_case.add_input<float>(bucket_shape, buckets_vect);
    test_case.add_expected_output<int>(data_shape, expected_vect);
    test_case.run();
}
