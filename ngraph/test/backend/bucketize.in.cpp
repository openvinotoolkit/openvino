//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, bucketize_right_edge)
{
    Shape data_shape{10, 1};
    Shape bucket_shape{4};

    const auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto buckets = make_shared<op::Parameter>(element::i32, bucket_shape);
    const auto bucketize = make_shared<op::v3::Bucketize>(data, buckets, element::i32, true);
    const auto f = make_shared<Function>(bucketize, ParameterVector{data, buckets});

    vector<float> data_vect = {8.f, 1.f, 2.f, 1.1f, 8.f, 10.f, 1.f, 10.2f, 0.f, 20.f};
    vector<int32_t> buckets_vect = {1, 4, 10, 20};
    vector<int32_t> expected_vect = {2, 0, 1, 1, 2, 2, 0, 3, 0, 3};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(data_shape, data_vect);
    test_case.add_input<int32_t>(bucket_shape, buckets_vect);
    test_case.add_expected_output<int32_t>(data_shape, expected_vect);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, bucketize_left_edge)
{
    Shape data_shape{1, 1, 10};
    Shape bucket_shape{4};

    const auto data = make_shared<op::Parameter>(element::i32, data_shape);
    const auto buckets = make_shared<op::Parameter>(element::f32, bucket_shape);
    const auto bucketize = make_shared<op::v3::Bucketize>(data, buckets, element::i32, false);
    const auto f = make_shared<Function>(bucketize, ParameterVector{data, buckets});

    vector<int32_t> data_vect = {8, 1, 2, 1, 8, 5, 1, 5, 0, 20};
    vector<float> buckets_vect = {1.f, 4.f, 10.f, 20.f};
    vector<int32_t> expected_vect = {2, 1, 1, 1, 2, 2, 1, 2, 0, 4};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<int32_t>(data_shape, data_vect);
    test_case.add_input<float>(bucket_shape, buckets_vect);
    test_case.add_expected_output<int32_t>(data_shape, expected_vect);
    test_case.run();
}
