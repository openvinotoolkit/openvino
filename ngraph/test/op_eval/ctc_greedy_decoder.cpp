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
#include "ngraph/op/ctc_greedy_decoder.hpp"
#include <vector>
#include "gtest/gtest.h"
#include "ngraph/runtime/host_tensor.hpp"
#include "util/all_close_f.hpp"

using namespace std;
using namespace ngraph;

TEST(op_eval, ctc_greedy_decoder_single_batch)
{
    const int T = 3;
    const int N = 1;
    const int C = 2;
    const auto data_shape = Shape{T, N, C};
    const auto masks_shape = Shape{T, N};

    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto masks = make_shared<op::Parameter>(element::f32, masks_shape);
    auto decoder = make_shared<op::CTCGreedyDecoder>(data, masks, false);
    auto f = make_shared<Function>(decoder, ParameterVector{data, masks});

    std::vector<float> data_vec{0.1f, 0.2f, 0.4f, 0.3f, 0.5f, 0.6f};
    std::vector<float> masks_vec{1.0f, 1.0f, 1.0f};
    std::vector<float> expected_vec{1.0f, 0.0f, 1.0f};

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(f->evaluate({result},
                            {make_host_tensor<element::Type_t::f32>(data_shape, data_vec),
                             make_host_tensor<element::Type_t::f32>(masks_shape, masks_vec)}));
    const auto expected_shape = Shape{N, T, 1, 1};
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), expected_shape);
    ASSERT_TRUE(test::all_close_f(read_vector<float>(result), expected_vec, 6, 0.001));
}

TEST(op_eval, ctc_greedy_decoder_multiple_batches)
{
    const int T = 3;
    const int N = 2;
    const int C = 2;
    const auto data_shape = Shape{T, N, C};
    const auto masks_shape = Shape{T, N};

    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto masks = make_shared<op::Parameter>(element::f32, masks_shape);
    auto decoder = make_shared<op::CTCGreedyDecoder>(data, masks, false);
    auto f = make_shared<Function>(decoder, ParameterVector{data, masks});

    std::vector<float> data_vec{
        0.1f, 0.2f, 0.15f, 0.25f, 0.4f, 0.3f, 0.45f, 0.35f, 0.5f, 0.6f, 0.55f, 0.65f};

    std::vector<float> masks_vec{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    std::vector<float> expected_vec{1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(f->evaluate({result},
                            {make_host_tensor<element::Type_t::f32>(data_shape, data_vec),
                             make_host_tensor<element::Type_t::f32>(masks_shape, masks_vec)}));
    const auto expected_shape = Shape{N, T, 1, 1};
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), expected_shape);
    ASSERT_TRUE(test::all_close_f(read_vector<float>(result), expected_vec, 6, 0.001));
}

TEST(op_eval, ctc_greedy_decoder_single_batch_short_sequence)
{
    const int T = 3;
    const int N = 1;
    const int C = 2;
    const auto data_shape = Shape{T, N, C};
    const auto masks_shape = Shape{T, N};

    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto masks = make_shared<op::Parameter>(element::f32, masks_shape);
    auto decoder = make_shared<op::CTCGreedyDecoder>(data, masks, false);
    auto f = make_shared<Function>(decoder, ParameterVector{data, masks});

    std::vector<float> data_vec{0.1f, 0.2f, 0.4f, 0.3f, 0.5f, 0.6f};
    std::vector<float> masks_vec{1.0f, 1.0f, 0.0f};
    std::vector<float> expected_vec{1.0f, 0.0f, -1.0f};

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(f->evaluate({result},
                            {make_host_tensor<element::Type_t::f32>(data_shape, data_vec),
                             make_host_tensor<element::Type_t::f32>(masks_shape, masks_vec)}));
    const auto expected_shape = Shape{N, T, 1, 1};
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), expected_shape);
    ASSERT_TRUE(test::all_close_f(read_vector<float>(result), expected_vec, 6, 0.001));
}

TEST(op_eval, ctc_greedy_decoder_merge)
{
    const int T = 3;
    const int N = 1;
    const int C = 2;
    const auto data_shape = Shape{T, N, C};
    const auto masks_shape = Shape{T, N};

    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto masks = make_shared<op::Parameter>(element::f32, masks_shape);
    auto decoder = make_shared<op::CTCGreedyDecoder>(data, masks, true);
    auto f = make_shared<Function>(decoder, ParameterVector{data, masks});

    std::vector<float> data_vec{0.1f, 0.2f, 0.3f, 0.4f, 0.6f, 0.5f};
    std::vector<float> masks_vec{1.0f, 1.0f, 1.0f};
    std::vector<float> expected_vec{1.0f, 0.0f, -1.0f};

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(f->evaluate({result},
                            {make_host_tensor<element::Type_t::f32>(data_shape, data_vec),
                             make_host_tensor<element::Type_t::f32>(masks_shape, masks_vec)}));
    const auto expected_shape = Shape{N, T, 1, 1};
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), expected_shape);
    ASSERT_TRUE(test::all_close_f(read_vector<float>(result), expected_vec, 6, 0.001));
}

TEST(op_eval, ctc_greedy_decoder_single_no_merge)
{
    const int T = 3;
    const int N = 1;
    const int C = 2;
    const auto data_shape = Shape{T, N, C};
    const auto masks_shape = Shape{T, N};

    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto masks = make_shared<op::Parameter>(element::f32, masks_shape);
    auto decoder = make_shared<op::CTCGreedyDecoder>(data, masks, false);
    auto f = make_shared<Function>(decoder, ParameterVector{data, masks});

    std::vector<float> data_vec{0.1f, 0.2f, 0.3f, 0.4f, 0.6f, 0.5f};
    std::vector<float> masks_vec{1.0f, 1.0f, 1.0f};
    std::vector<float> expected_vec{1.0f, 1.0f, 0.0f};

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(f->evaluate({result},
                            {make_host_tensor<element::Type_t::f32>(data_shape, data_vec),
                             make_host_tensor<element::Type_t::f32>(masks_shape, masks_vec)}));
    const auto expected_shape = Shape{N, T, 1, 1};
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), expected_shape);
    ASSERT_TRUE(test::all_close_f(read_vector<float>(result), expected_vec, 6, 0.001));
}