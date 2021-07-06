// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "runtime/backend.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, tile_3d_small_data_rank)
{
    Shape shape_a{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_re{3};
    auto repeats = make_shared<op::Constant>(element::i64, shape_re, vector<int>{2, 2, 1});
    Shape shape_r{2, 2, 3};

    auto tile = make_shared<op::v0::Tile>(A, repeats);

    auto f = make_shared<Function>(tile, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3});

    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(vector<float>{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3},
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, tile_3d_few_repeats)
{
    Shape shape_a{2, 1, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_re{2};
    auto repeats = make_shared<op::Constant>(element::i64, shape_re, vector<int>{2, 1});
    Shape shape_r{2, 2, 3};

    auto tile = make_shared<op::v0::Tile>(A, repeats);

    auto f = make_shared<Function>(tile, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});

    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(vector<float>{1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6},
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}
