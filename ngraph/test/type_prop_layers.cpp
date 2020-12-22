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
#include "ngraph/op/ctc_greedy_decoder.hpp"
#include "ngraph/op/interpolate.hpp"
#include "ngraph/op/prior_box.hpp"
#include "ngraph/op/prior_box_clustered.hpp"
#include "ngraph/op/region_yolo.hpp"
#include "ngraph/op/reorg_yolo.hpp"
#include "ngraph/op/roi_pooling.hpp"

#include <memory>
using namespace std;
using namespace ngraph;

TEST(type_prop_layers, ctc_greedy_decoder)
{
    auto input = make_shared<op::Parameter>(element::f32, Shape{88, 2, 48});
    auto seq_len = make_shared<op::Parameter>(element::f32, Shape{88, 2});
    auto op = make_shared<op::CTCGreedyDecoder>(input, seq_len, false);
    ASSERT_EQ(op->get_shape(), (Shape{2, 88, 1, 1}));
}

TEST(type_prop_layers, interpolate)
{
    auto image = make_shared<op::Parameter>(element::f32, Shape{2, 2, 33, 65});
    auto dyn_output_shape = make_shared<op::Parameter>(element::i64, Shape{2});
    auto output_shape = op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {15, 30});

    op::v0::InterpolateAttrs attrs;
    attrs.axes = {2, 3};
    attrs.mode = "nearest";
    attrs.align_corners = true;
    attrs.antialias = false;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};
    auto op = make_shared<op::v0::Interpolate>(image, output_shape, attrs);
    ASSERT_EQ(op->get_shape(), (Shape{2, 2, 15, 30}));

    EXPECT_TRUE(make_shared<op::v0::Interpolate>(image, dyn_output_shape, attrs)
                    ->get_output_partial_shape(0)
                    .same_scheme(PartialShape{2, 2, Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop_layers, prior_box1)
{
    op::PriorBoxAttrs attrs;
    attrs.min_size = {2.0f, 3.0f};
    attrs.aspect_ratio = {1.5f, 2.0f, 2.5f};
    attrs.scale_all_sizes = false;

    auto layer_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, {32, 32});
    auto image_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, {300, 300});
    auto pb = make_shared<op::PriorBox>(layer_shape, image_shape, attrs);
    ASSERT_EQ(pb->get_shape(), (Shape{2, 20480}));
}

TEST(type_prop_layers, prior_box2)
{
    op::PriorBoxAttrs attrs;
    attrs.min_size = {2.0f, 3.0f};
    attrs.aspect_ratio = {1.5f, 2.0f, 2.5f};
    attrs.flip = true;
    attrs.scale_all_sizes = false;

    auto layer_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, {32, 32});
    auto image_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, {300, 300});
    auto pb = make_shared<op::PriorBox>(layer_shape, image_shape, attrs);
    ASSERT_EQ(pb->get_shape(), (Shape{2, 32768}));
}

TEST(type_prop_layers, prior_box3)
{
    op::PriorBoxAttrs attrs;
    attrs.min_size = {256.0f};
    attrs.max_size = {315.0f};
    attrs.aspect_ratio = {2.0f};
    attrs.flip = true;

    auto layer_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, {1, 1});
    auto image_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, {300, 300});
    auto pb = make_shared<op::PriorBox>(layer_shape, image_shape, attrs);
    ASSERT_EQ(pb->get_shape(), (Shape{2, 16}));
}

TEST(type_prop_layers, prior_box_clustered)
{
    op::PriorBoxClusteredAttrs attrs;
    attrs.widths = {4.0f, 2.0f, 3.2f};
    attrs.heights = {1.0f, 2.0f, 1.1f};

    auto layer_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, {19, 19});
    auto image_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, {300, 300});
    auto pbc = make_shared<op::PriorBoxClustered>(layer_shape, image_shape, attrs);
    // Output shape - 4 * 19 * 19 * 3 (attrs.widths.size())
    ASSERT_EQ(pbc->get_shape(), (Shape{2, 4332}));
}

TEST(type_prop_layers, region_yolo1)
{
    auto inputs = make_shared<op::Parameter>(element::f32, Shape{1, 125, 13, 13});
    auto op = make_shared<op::RegionYolo>(inputs, 0, 0, 0, true, std::vector<int64_t>{}, 0, 1);
    ASSERT_EQ(op->get_shape(), (Shape{1 * 125, 13, 13}));
}

TEST(type_prop_layers, region_yolo2)
{
    auto inputs = make_shared<op::Parameter>(element::f32, Shape{1, 125, 13, 13});
    auto op = make_shared<op::RegionYolo>(inputs, 0, 0, 0, true, std::vector<int64_t>{}, 0, 2);
    ASSERT_EQ(op->get_shape(), (Shape{1 * 125 * 13, 13}));
}

TEST(type_prop_layers, region_yolo3)
{
    auto inputs = make_shared<op::Parameter>(element::f32, Shape{1, 125, 13, 13});
    auto op =
        make_shared<op::RegionYolo>(inputs, 4, 80, 1, false, std::vector<int64_t>{6, 7, 8}, 0, -1);
    ASSERT_EQ(op->get_shape(), (Shape{1, (80 + 4 + 1) * 3, 13, 13}));
}

TEST(type_prop_layers, reorg_yolo)
{
    auto inputs = make_shared<op::Parameter>(element::f32, Shape{2, 24, 34, 62});
    auto op = make_shared<op::ReorgYolo>(inputs, Strides{2});
    ASSERT_EQ(op->get_shape(), (Shape{2, 96, 17, 31}));
}

TEST(type_prop_layers, roi_pooling)
{
    auto inputs = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});
    auto coords = make_shared<op::Parameter>(element::f32, Shape{150, 5});
    auto op = make_shared<op::ROIPooling>(inputs, coords, Shape{6, 6}, 0.0625, "max");
    ASSERT_EQ(op->get_shape(), (Shape{150, 3, 6, 6}));
}
