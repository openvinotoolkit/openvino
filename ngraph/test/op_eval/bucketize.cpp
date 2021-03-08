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
#include "util/engine/interpreter_engine.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(op_eval, bucketize_empty_buckets)
{
    Shape data_shape{1, 1, 3};
    Shape bucket_shape{0};

    const auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto buckets = make_shared<op::Parameter>(element::f32, bucket_shape);
    const auto bucketize = make_shared<op::v3::Bucketize>(data, buckets);
    const auto f = make_shared<Function>(bucketize, ParameterVector{data, buckets});

    vector<float> data_vect = {8.f, 1.f, 2.f};
    vector<float> buckets_vect;
    vector<int> expected_vect = {0, 0, 0};

    auto test_case = test::TestCase<ngraph::test::INTERPRETER_Engine>(f);
    test_case.add_input<float>(data_shape, data_vect);
    test_case.add_input<float>(bucket_shape, buckets_vect);
    test_case.add_expected_output<int>(data_shape, expected_vect);
    test_case.run();
}
