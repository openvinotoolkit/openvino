//*****************************************************************************
// Copyright 2020 Intel Corporation
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

using Attrs = op::v6::ExperimentalDetectronROIFeatureExtractor::Attributes;
using ExperimentalROI = op::v6::ExperimentalDetectronROIFeatureExtractor;

TEST(type_prop, detectron_roi_feature_extractor)
{
    Attrs attrs;
    attrs.aligned = false;
    attrs.distribute_rois_between_levels = 1;
    attrs.output_size = 14;
    attrs.preserve_rois_order = 1;
    attrs.sampling_ratio = 2;
    attrs.pyramid_scales = {4, 8, 16, 32};
    attrs.image_id = 0;

    auto input = std::make_shared<op::Parameter>(element::f32, Shape{1000, 4});
    auto pyramid_layer0 = std::make_shared<op::Parameter>(element::f32, Shape{1, 256, 200, 336});
    auto pyramid_layer1 = std::make_shared<op::Parameter>(element::f32, Shape{1, 256, 100, 168});
    auto pyramid_layer2 = std::make_shared<op::Parameter>(element::f32, Shape{1, 256, 50, 84});
    auto pyramid_layer3 = std::make_shared<op::Parameter>(element::f32, Shape{1, 256, 25, 42});

    auto inputs = NodeVector{input, pyramid_layer0, pyramid_layer1, pyramid_layer2, pyramid_layer3};
    auto roi = std::make_shared<ExperimentalROI>(inputs, attrs)

    ASSERT_EQ(detection->get_output_element_type(0), element::f32);
    EXPECT_EQ(detection->get_output_shape(0), (Shape{1000, 256, 14, 14}));
}
