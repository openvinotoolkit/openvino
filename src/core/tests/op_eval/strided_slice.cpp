// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/strided_slice.hpp"

#include <numeric>
#include <vector>

#include "engines_util/execute_tools.hpp"
#include "gtest/gtest.h"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"
#include "util/test_tools.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

OPENVINO_SUPPRESS_DEPRECATED_START

TEST(op_eval, strided_slice1) {
    auto A_shape = Shape{3, 2, 3};
    auto A = make_shared<op::Parameter>(element::i64, A_shape);
    auto begin = make_shared<op::Parameter>(element::i64, Shape{3});
    auto end = make_shared<op::Parameter>(element::i64, Shape{3});
    auto strides = make_shared<op::Parameter>(element::i64, Shape{3});
    auto r = make_shared<op::v1::StridedSlice>(A,
                                               begin,
                                               end,
                                               strides,
                                               std::vector<int64_t>(3, 0),
                                               std::vector<int64_t>(3, 0),
                                               std::vector<int64_t>(3, 0),
                                               std::vector<int64_t>(3, 0),
                                               std::vector<int64_t>(3, 0));
    auto f = make_shared<Function>(r, ParameterVector{A, begin, end, strides});

    std::vector<int64_t> A_vec(3 * 2 * 3);
    std::iota(A_vec.begin(), A_vec.end(), 0);
    std::vector<std::vector<int64_t>> begin_vecs{{1, 0, 0}, {1, 0, 0}, {2, 0, 0}};
    std::vector<std::vector<int64_t>> end_vecs{{2, 1, 3}, {2, 2, 3}, {3, 2, 3}};
    std::vector<std::vector<int64_t>> strides_vecs{{1, 1, 1}, {1, 1, 1}, {1, 1, 2}};

    std::vector<std::vector<int64_t>> expected_results{{6, 7, 8}, {6, 7, 8, 9, 10, 11}, {12, 14, 15, 17}};
    std::vector<Shape> expected_shape{Shape{1, 1, 3}, Shape{1, 2, 3}, Shape{1, 2, 2}};

    for (size_t i = 0; i < begin_vecs.size(); ++i) {
        auto result = make_shared<HostTensor>();
        ASSERT_TRUE(f->evaluate({result},
                                {make_host_tensor<element::Type_t::i64>(A_shape, A_vec),
                                 make_host_tensor<element::Type_t::i64>(Shape{3}, begin_vecs[i]),
                                 make_host_tensor<element::Type_t::i64>(Shape{3}, end_vecs[i]),
                                 make_host_tensor<element::Type_t::i64>(Shape{3}, strides_vecs[i])}));
        EXPECT_EQ(result->get_element_type(), element::i64);
        EXPECT_EQ(result->get_shape(), expected_shape[i]);
        EXPECT_EQ(read_vector<int64_t>(result), expected_results[i]);
    }
}

// A Shape{3, 2, 3}
// [[[ 0  1  2]
//   [ 3  4  5]]
//  [[ 6  7  8]
//   [ 9 10 11]]
//  [[12 13 14]
//   [15 16 17]]]

// A[1:, :, :]
// result Shape{2, 2, 3}
// [[[ 6  7  8]
//   [ 9 10 11]]
//  [[12 13 14]
//   [15 16 17]]]
TEST(op_eval, strided_slice2) {
    auto A_shape = Shape{3, 2, 3};
    auto A = make_shared<op::Parameter>(element::i64, A_shape);
    auto begin = make_shared<op::Parameter>(element::i64, Shape{3});
    auto end = make_shared<op::Parameter>(element::i64, Shape{3});
    auto strides = make_shared<op::Parameter>(element::i64, Shape{3});

    std::vector<int64_t> begin_vec{1, 0, 0};
    std::vector<int64_t> end_vec{0, 0, 0};
    std::vector<int64_t> strides_vec{1, 1, 1};
    std::vector<int64_t> begin_mask{0, 1, 1};
    std::vector<int64_t> end_mask{1, 1, 1};

    auto r = make_shared<op::v1::StridedSlice>(A,
                                               begin,
                                               end,
                                               strides,
                                               begin_mask,
                                               end_mask,
                                               std::vector<int64_t>(3, 0),
                                               std::vector<int64_t>(3, 0),
                                               std::vector<int64_t>(3, 0));
    auto f = make_shared<Function>(r, ParameterVector{A, begin, end, strides});

    std::vector<int64_t> A_vec(3 * 2 * 3);
    std::iota(A_vec.begin(), A_vec.end(), 0);

    std::vector<int64_t> expected{6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
    Shape expected_shape{2, 2, 3};

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(f->evaluate({result},
                            {make_host_tensor<element::Type_t::i64>(A_shape, A_vec),
                             make_host_tensor<element::Type_t::i64>(Shape{3}, begin_vec),
                             make_host_tensor<element::Type_t::i64>(Shape{3}, end_vec),
                             make_host_tensor<element::Type_t::i64>(Shape{3}, strides_vec)}));
    EXPECT_EQ(result->get_element_type(), element::i64);
    EXPECT_EQ(result->get_shape(), expected_shape);
    EXPECT_EQ(read_vector<int64_t>(result), expected);
}

// A Shape{3, 2, 3}
// A[:2, 1:, ::2]
// result Shape{2, 1, 2}
// [[[3 5]]
//  [[9 11]]]
TEST(op_eval, strided_slice3) {
    auto A_shape = Shape{3, 2, 3};
    auto A = make_shared<op::Parameter>(element::i64, A_shape);
    auto begin = make_shared<op::Parameter>(element::i64, Shape{3});
    auto end = make_shared<op::Parameter>(element::i64, Shape{3});
    auto strides = make_shared<op::Parameter>(element::i64, Shape{3});

    std::vector<int64_t> begin_vec{0, 1, 0};
    std::vector<int64_t> end_vec{2, 0, 0};
    std::vector<int64_t> strides_vec{1, 1, 2};
    std::vector<int64_t> begin_mask{1, 0, 1};
    std::vector<int64_t> end_mask{0, 1, 1};

    auto r = make_shared<op::v1::StridedSlice>(A,
                                               begin,
                                               end,
                                               strides,
                                               begin_mask,
                                               end_mask,
                                               std::vector<int64_t>(3, 0),
                                               std::vector<int64_t>(3, 0),
                                               std::vector<int64_t>(3, 0));
    auto f = make_shared<Function>(r, ParameterVector{A, begin, end, strides});

    std::vector<int64_t> A_vec(3 * 2 * 3);
    std::iota(A_vec.begin(), A_vec.end(), 0);

    std::vector<int64_t> expected{3, 5, 9, 11};
    Shape expected_shape{2, 1, 2};

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(f->evaluate({result},
                            {make_host_tensor<element::Type_t::i64>(A_shape, A_vec),
                             make_host_tensor<element::Type_t::i64>(Shape{3}, begin_vec),
                             make_host_tensor<element::Type_t::i64>(Shape{3}, end_vec),
                             make_host_tensor<element::Type_t::i64>(Shape{3}, strides_vec)}));
    EXPECT_EQ(result->get_element_type(), element::i64);
    EXPECT_EQ(result->get_shape(), expected_shape);
    EXPECT_EQ(read_vector<int64_t>(result), expected);
}

// A Shape{3, 2, 3}
// A[0:1, :, ::-1]
// result Shape{1, 2, 3}
// [[[2 1 0]
//   [5 4 3]]]
TEST(op_eval, strided_slice_reverse) {
    auto A_shape = Shape{3, 2, 3};
    auto A = make_shared<op::Parameter>(element::i64, A_shape);
    auto begin = make_shared<op::Parameter>(element::i64, Shape{3});
    auto end = make_shared<op::Parameter>(element::i64, Shape{3});
    auto strides = make_shared<op::Parameter>(element::i64, Shape{3});

    std::vector<int64_t> begin_vec{0, 0, 0};
    std::vector<int64_t> end_vec{1, 0, 0};
    std::vector<int64_t> strides_vec{1, 1, -1};
    std::vector<int64_t> begin_mask{0, 1, 1};
    std::vector<int64_t> end_mask{0, 1, 1};

    auto r = make_shared<op::v1::StridedSlice>(A,
                                               begin,
                                               end,
                                               strides,
                                               begin_mask,
                                               end_mask,
                                               std::vector<int64_t>(3, 0),
                                               std::vector<int64_t>(3, 0),
                                               std::vector<int64_t>(3, 0));
    auto f = make_shared<Function>(r, ParameterVector{A, begin, end, strides});

    std::vector<int64_t> A_vec(3 * 2 * 3);
    std::iota(A_vec.begin(), A_vec.end(), 0);

    std::vector<int64_t> expected{2, 1, 0, 5, 4, 3};
    Shape expected_shape{1, 2, 3};

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(f->evaluate({result},
                            {make_host_tensor<element::Type_t::i64>(A_shape, A_vec),
                             make_host_tensor<element::Type_t::i64>(Shape{3}, begin_vec),
                             make_host_tensor<element::Type_t::i64>(Shape{3}, end_vec),
                             make_host_tensor<element::Type_t::i64>(Shape{3}, strides_vec)}));
    EXPECT_EQ(result->get_element_type(), element::i64);
    EXPECT_EQ(result->get_shape(), expected_shape);
    EXPECT_EQ(read_vector<int64_t>(result), expected);
}
