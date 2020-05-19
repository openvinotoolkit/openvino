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

#include "ngraph/file_util.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/serializer.hpp"

using namespace ngraph;
using namespace std;

TEST(core, function_size)
{
    const string m1 = file_util::path_join(SERIALIZED_ZOO, "mxnet/mnist_mlp_forward.json");
    const string m2 = file_util::path_join(SERIALIZED_ZOO, "mxnet/10_bucket_LSTM.json");

    auto f1 = deserialize(m1);
    auto f2 = deserialize(m2);
    auto s1 = f1->get_graph_size();
    auto s2 = f2->get_graph_size();
    EXPECT_GT(s2, s1);
}
