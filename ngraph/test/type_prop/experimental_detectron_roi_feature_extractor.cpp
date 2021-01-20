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

using Attrs = op::v6::ExperimentalDetectronROIFeatureExtractor::Attributes;
using ExperimentalROI = op::v6::ExperimentalDetectronROIFeatureExtractor;

TEST(type_prop, detectron_roi_feature_extractor)
{
    Attrs attrs;
    attrs.aligned = false;
    attrs.output_size = 14;
    attrs.sampling_ratio = 2;
    attrs.pyramid_scales = {4, 8, 16, 32};

    auto input = std::make_shared<op::Parameter>(element::f32, Shape{1000, 4});
    auto pyramid_layer0 = std::make_shared<op::Parameter>(element::f32, Shape{1, 256, 200, 336});
    auto pyramid_layer1 = std::make_shared<op::Parameter>(element::f32, Shape{1, 256, 100, 168});
    auto pyramid_layer2 = std::make_shared<op::Parameter>(element::f32, Shape{1, 256, 50, 84});
    auto pyramid_layer3 = std::make_shared<op::Parameter>(element::f32, Shape{1, 256, 25, 42});

    auto roi = std::make_shared<ExperimentalROI>(
        NodeVector{input, pyramid_layer0, pyramid_layer1, pyramid_layer2, pyramid_layer3}, attrs);

    ASSERT_EQ(roi->get_output_element_type(0), element::f32);
    EXPECT_EQ(roi->get_output_shape(0), (Shape{1000, 256, 14, 14}));
}

TEST(type_prop, detectron_roi_feature_extractor_dynamic)
{
    Attrs attrs;
    attrs.aligned = false;
    attrs.output_size = 14;
    attrs.sampling_ratio = 2;
    attrs.pyramid_scales = {4, 8, 16, 32};

    struct Shapes
    {
        PartialShape input_shape;
        Dimension channels;
    };

    const auto dyn_dim = Dimension::dynamic();

    std::vector<Shapes> shapes = {
        {{1000, 4}, dyn_dim}, {{dyn_dim, 4}, 256}, {{dyn_dim, 4}, dyn_dim}};
    for (const auto& s : shapes)
    {
        auto layer0_shape = PartialShape{1, s.channels, 200, 336};
        auto layer1_shape = PartialShape{1, s.channels, 100, 168};
        auto layer2_shape = PartialShape{1, s.channels, 50, 84};
        auto layer3_shape = PartialShape{1, s.channels, 25, 42};

        auto ref_out_shape = PartialShape{s.input_shape[0], s.channels, 14, 14};

        auto input = std::make_shared<op::Parameter>(element::f32, s.input_shape);
        auto pyramid_layer0 = std::make_shared<op::Parameter>(element::f32, layer0_shape);
        auto pyramid_layer1 = std::make_shared<op::Parameter>(element::f32, layer1_shape);
        auto pyramid_layer2 = std::make_shared<op::Parameter>(element::f32, layer2_shape);
        auto pyramid_layer3 = std::make_shared<op::Parameter>(element::f32, layer3_shape);

        auto roi = std::make_shared<ExperimentalROI>(
            NodeVector{input, pyramid_layer0, pyramid_layer1, pyramid_layer2, pyramid_layer3},
            attrs);

        ASSERT_EQ(roi->get_output_element_type(0), element::f32);
        ASSERT_TRUE(roi->get_output_partial_shape(0).same_scheme(ref_out_shape));
    }
}

TEST(type_prop, detectron_roi_feature_extractor_intervals)
{
    Attrs attrs;
    attrs.aligned = false;
    attrs.output_size = 14;
    attrs.sampling_ratio = 2;
    attrs.pyramid_scales = {4, 8, 16, 32};

    struct Shapes
    {
        PartialShape input_shape;
        Dimension channels[4];
    };

    const auto dyn_dim = Dimension::dynamic();

    std::vector<Shapes> shapes = {
        {{1000, 4}, {Dimension(0, 128), Dimension(0, 256), Dimension(0, 64), Dimension(0, 33)}},
        {{1000, 4}, {Dimension(0, 128), Dimension(0, 256), Dimension(0, 64), Dimension(33)}},
        {{1000, 4}, {Dimension(0, 128), Dimension(0, 256), Dimension(64), Dimension(0, 72)}},
        {{1000, 4}, {Dimension(0, 128), Dimension(0, 256), Dimension(64), Dimension(64)}},
        {{1000, 4}, {Dimension(0, 512), Dimension(256), Dimension(0, 640), Dimension(0, 330)}},
        {{1000, 4}, {Dimension(0, 512), Dimension(256), Dimension(0, 640), Dimension(256)}},
        {{1000, 4}, {Dimension(0, 512), Dimension(256), Dimension(256), Dimension(0, 720)}},
        {{1000, 4}, {Dimension(0, 380), Dimension(256), Dimension(256), Dimension(256)}},
        {{1000, 4}, {Dimension(128), Dimension(0, 256), Dimension(0, 640), Dimension(0, 330)}},
        {{1000, 4}, {Dimension(128), Dimension(0, 256), Dimension(0, 640), Dimension(128)}},
        {{1000, 4}, {Dimension(128), Dimension(0, 256), Dimension(128), Dimension(0, 720)}},
        {{1000, 4}, {Dimension(128), Dimension(0, 256), Dimension(128), Dimension(128)}},
        {{1000, 4}, {Dimension(256), Dimension(256), Dimension(0, 640), Dimension(0, 330)}},
        {{1000, 4}, {Dimension(256), Dimension(256), Dimension(0, 640), Dimension(256)}},
        {{1000, 4}, {Dimension(256), Dimension(256), Dimension(256), Dimension(0, 330)}},
        {{1000, 4}, {Dimension(256), Dimension(256), Dimension(256), Dimension(256)}},
        {{dyn_dim, 4}, {Dimension(0, 128), Dimension(0, 256), Dimension(0, 64), Dimension(0, 33)}},
        {{dyn_dim, 4}, {Dimension(0, 128), Dimension(0, 256), Dimension(0, 64), Dimension(33)}},
        {{dyn_dim, 4}, {Dimension(0, 128), Dimension(0, 256), Dimension(64), Dimension(0, 72)}},
        {{dyn_dim, 4}, {Dimension(0, 128), Dimension(0, 256), Dimension(64), Dimension(64)}},
        {{dyn_dim, 4}, {Dimension(0, 512), Dimension(256), Dimension(0, 640), Dimension(0, 330)}},
        {{dyn_dim, 4}, {Dimension(0, 512), Dimension(256), Dimension(0, 640), Dimension(256)}},
        {{dyn_dim, 4}, {Dimension(0, 512), Dimension(256), Dimension(256), Dimension(0, 720)}},
        {{dyn_dim, 4}, {Dimension(0, 380), Dimension(256), Dimension(256), Dimension(256)}},
        {{dyn_dim, 4}, {Dimension(128), Dimension(0, 256), Dimension(0, 640), Dimension(0, 330)}},
        {{dyn_dim, 4}, {Dimension(128), Dimension(0, 256), Dimension(0, 640), Dimension(128)}},
        {{dyn_dim, 4}, {Dimension(128), Dimension(0, 256), Dimension(128), Dimension(0, 720)}},
        {{dyn_dim, 4}, {Dimension(128), Dimension(0, 256), Dimension(128), Dimension(128)}},
        {{dyn_dim, 4}, {Dimension(256), Dimension(256), Dimension(0, 640), Dimension(0, 330)}},
        {{dyn_dim, 4}, {Dimension(256), Dimension(256), Dimension(0, 640), Dimension(256)}},
        {{dyn_dim, 4}, {Dimension(256), Dimension(256), Dimension(256), Dimension(0, 330)}},
        {{dyn_dim, 4}, {Dimension(256), Dimension(256), Dimension(256), Dimension(256)}}};

    for (const auto& s : shapes)
    {
        auto layer0_channels = s.channels[0];
        auto layer1_channels = s.channels[1];
        auto layer2_channels = s.channels[2];
        auto layer3_channels = s.channels[3];

        auto layer0_shape = PartialShape{1, layer0_channels, 200, 336};
        auto layer1_shape = PartialShape{1, layer1_channels, 100, 168};
        auto layer2_shape = PartialShape{1, layer2_channels, 50, 84};
        auto layer3_shape = PartialShape{1, layer3_channels, 25, 42};

        auto expected_channels =
            layer0_channels & layer1_channels & layer2_channels & layer3_channels;

        auto ref_out_shape = PartialShape{s.input_shape[0], expected_channels, 14, 14};

        auto input = std::make_shared<op::Parameter>(element::f32, s.input_shape);
        auto pyramid_layer0 = std::make_shared<op::Parameter>(element::f32, layer0_shape);
        auto pyramid_layer1 = std::make_shared<op::Parameter>(element::f32, layer1_shape);
        auto pyramid_layer2 = std::make_shared<op::Parameter>(element::f32, layer2_shape);
        auto pyramid_layer3 = std::make_shared<op::Parameter>(element::f32, layer3_shape);

        auto roi = std::make_shared<ExperimentalROI>(
            NodeVector{input, pyramid_layer0, pyramid_layer1, pyramid_layer2, pyramid_layer3},
            attrs);

        ASSERT_EQ(roi->get_output_element_type(0), element::f32);
        ASSERT_TRUE(roi->get_output_partial_shape(0).same_scheme(ref_out_shape));
    }
}
