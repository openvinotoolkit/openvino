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
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, batch_to_space_output_shape_2D)
{
    auto data = make_shared<op::Parameter>(element::Type_t::f32, Shape{10, 26});
    auto block_shape =
        make_shared<op::Constant>(element::Type_t::i64, Shape{2}, vector<int64_t>{1, 5});
    auto pads_begin =
        make_shared<op::Constant>(element::Type_t::i64, Shape{2}, vector<int64_t>{0, 2});
    auto pads_end =
        make_shared<op::Constant>(element::Type_t::i64, Shape{2}, vector<int64_t>{0, 0});

    auto batch_to_space =
        make_shared<op::v1::BatchToSpace>(data, block_shape, pads_begin, pads_end);

    ASSERT_EQ(batch_to_space->get_element_type(), element::Type_t::f32);
    ASSERT_EQ(batch_to_space->get_shape(), (Shape{10 / 5, 26 * 5 - 2}));
}

TEST(type_prop, batch_to_space_output_shape_4D)
{
    auto data = make_shared<op::Parameter>(element::Type_t::f32, Shape{100, 7, 13, 3});
    auto block_shape =
        make_shared<op::Constant>(element::Type_t::i64, Shape{4}, vector<int64_t>{1, 10, 5, 1});
    auto pads_begin =
        make_shared<op::Constant>(element::Type_t::i64, Shape{4}, vector<int64_t>{0, 3, 1, 0});
    auto pads_end =
        make_shared<op::Constant>(element::Type_t::i64, Shape{4}, vector<int64_t>{0, 3, 0, 0});

    auto batch_to_space =
        make_shared<op::v1::BatchToSpace>(data, block_shape, pads_begin, pads_end);

    ASSERT_EQ(batch_to_space->get_element_type(), element::Type_t::f32);
    ASSERT_EQ(batch_to_space->get_shape(), (Shape{100 / (10 * 5), 7 * 10 - 3 - 3, 13 * 5 - 1, 3}));
}

TEST(type_prop, batch_to_space_output_shape_5D)
{
    auto data = make_shared<op::Parameter>(element::Type_t::f32, Shape{960, 6, 13, 128, 16});
    auto block_shape =
        make_shared<op::Constant>(element::Type_t::i32, Shape{5}, vector<int64_t>{1, 6, 5, 1, 16});
    auto pads_begin =
        make_shared<op::Constant>(element::Type_t::i32, Shape{5}, vector<int64_t>{0, 2, 0, 0, 0});
    auto pads_end =
        make_shared<op::Constant>(element::Type_t::i32, Shape{5}, vector<int64_t>{0, 2, 1, 0, 0});

    auto batch_to_space =
        make_shared<op::v1::BatchToSpace>(data, block_shape, pads_begin, pads_end);

    ASSERT_EQ(batch_to_space->get_element_type(), element::Type_t::f32);
    ASSERT_EQ(batch_to_space->get_shape(),
              (Shape{960 / (6 * 5 * 16), 6 * 6 - 2 - 2, 13 * 5 - 1, 128, 16 * 16}));
}

TEST(type_prop, batch_to_space_and_space_to_batch)
{
    auto data = make_shared<op::Parameter>(element::Type_t::f32, Shape{4800, 9, 11, 2});
    auto block_shape =
        make_shared<op::Constant>(element::Type_t::i64, Shape{4}, vector<int64_t>{1, 12, 100, 2});
    auto pads_begin =
        make_shared<op::Constant>(element::Type_t::i64, Shape{4}, vector<int64_t>{0, 3, 38, 1});
    auto pads_end =
        make_shared<op::Constant>(element::Type_t::i64, Shape{4}, vector<int64_t>{0, 5, 38, 0});

    auto batch_to_space =
        make_shared<op::v1::BatchToSpace>(data, block_shape, pads_begin, pads_end);

    ASSERT_EQ(batch_to_space->get_element_type(), element::Type_t::f32);
    ASSERT_EQ(batch_to_space->get_shape(),
              (Shape{4800 / (12 * 100 * 2), 9 * 12 - 3 - 5, 11 * 100 - 38 - 38, 2 * 2 - 1}));

    auto space_to_batch =
        make_shared<op::v1::SpaceToBatch>(batch_to_space, block_shape, pads_begin, pads_end);
    ASSERT_EQ(space_to_batch->get_element_type(), element::Type_t::f32);
    ASSERT_EQ(space_to_batch->get_shape(), (Shape{4800, 9, 11, 2}));
}
