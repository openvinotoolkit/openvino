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

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace ngraph;

using ExperimentalTopKROIs = op::v6::ExperimentalDetectronTopKROIs;

TEST(type_prop, detectron_topk_rois)
{
    auto input_rois = std::make_shared<op::Parameter>(element::f32, Shape{5000, 4});
    auto rois_probs = std::make_shared<op::Parameter>(element::f32, Shape{5000});
    size_t max_rois = 1000;
    auto top_rois = std::make_shared<ExperimentalTopKROIs>(input_rois, rois_probs, max_rois);

    ASSERT_EQ(roi->get_output_element_type(0), element::f32);
    EXPECT_EQ(roi->get_output_shape(0), (Shape{1000, 4}));
}