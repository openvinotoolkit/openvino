//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include <vector>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace ngraph;

using ExperimentalProposals = op::v6::ExperimentalDetectronGenerateProposalsSingleImage;
using Attrs = op::v6::ExperimentalDetectronGenerateProposalsSingleImage::Attributes;

TEST(type_prop, detectron_proposals)
{
    Attrs attrs;
    attrs.min_size = 0.0f;
    attrs.nms_threshold = 0.699999988079071f;
    attrs.post_nms_count = 1000;
    attrs.pre_nms_count = 1000;

    auto im_info = std::make_shared<op::Parameter>(element::f32, Shape{3});
    auto anchors = std::make_shared<op::Parameter>(element::f32, Shape{201600, 4});
    auto deltas = std::make_shared<op::Parameter>(element::f32, Shape{12, 200, 336});
    auto scores = std::make_shared<op::Parameter>(element::f32, Shape{3, 200, 336});

    auto proposals =
        std::make_shared<ExperimentalProposals>(im_info, anchors, deltas, scores, attrs);

    ASSERT_EQ(proposals->get_output_element_type(0), element::f32);
    ASSERT_EQ(proposals->get_output_element_type(1), element::f32);
    EXPECT_EQ(proposals->get_output_shape(0), (Shape{attrs.post_nms_count, 4}));
    EXPECT_EQ(proposals->get_output_shape(1), (Shape{attrs.post_nms_count}));
}
