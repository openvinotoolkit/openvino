// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

struct ROIFeatureIntervalsTestParams
{
    PartialShape input_shape;
    Dimension channels[4];
    Dimension first_dims[4];
};

struct ROIFeatureIntervalsTest : ::testing::TestWithParam<ROIFeatureIntervalsTestParams>
{
};

TEST_P(ROIFeatureIntervalsTest, detectron_roi_feature_extractor_intervals_1)
{
    auto params = GetParam();

    Attrs attrs;
    attrs.aligned = false;
    attrs.output_size = 14;
    attrs.sampling_ratio = 2;
    attrs.pyramid_scales = {4, 8, 16, 32};

    auto layer0_channels = params.channels[0];
    auto layer1_channels = params.channels[1];
    auto layer2_channels = params.channels[2];
    auto layer3_channels = params.channels[3];

    auto layer0_shape = PartialShape{params.first_dims[0], layer0_channels, 200, 336};
    auto layer1_shape = PartialShape{params.first_dims[1], layer1_channels, 100, 168};
    auto layer2_shape = PartialShape{params.first_dims[2], layer2_channels, 50, 84};
    auto layer3_shape = PartialShape{params.first_dims[3], layer3_channels, 25, 42};

    auto expected_channels = layer0_channels & layer1_channels & layer2_channels & layer3_channels;

    auto ref_out_shape = PartialShape{params.input_shape[0], expected_channels, 14, 14};

    auto input = std::make_shared<op::Parameter>(element::f32, params.input_shape);
    auto pyramid_layer0 = std::make_shared<op::Parameter>(element::f32, layer0_shape);
    auto pyramid_layer1 = std::make_shared<op::Parameter>(element::f32, layer1_shape);
    auto pyramid_layer2 = std::make_shared<op::Parameter>(element::f32, layer2_shape);
    auto pyramid_layer3 = std::make_shared<op::Parameter>(element::f32, layer3_shape);

    auto roi = std::make_shared<ExperimentalROI>(
        NodeVector{input, pyramid_layer0, pyramid_layer1, pyramid_layer2, pyramid_layer3}, attrs);

    ASSERT_EQ(roi->get_output_element_type(0), element::f32);
    ASSERT_TRUE(roi->get_output_partial_shape(0).same_scheme(ref_out_shape));
}

INSTANTIATE_TEST_SUITE_P(
    type_prop,
    ROIFeatureIntervalsTest,
    ::testing::Values(
        ROIFeatureIntervalsTestParams{
            {1000, Dimension(0, 5)},
            {Dimension(0, 128), Dimension(0, 256), Dimension(0, 64), Dimension(0, 33)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {1000, Dimension(0, 5)},
            {Dimension(0, 128), Dimension(0, 256), Dimension(0, 64), Dimension(33)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {1000, Dimension(2, 5)},
            {Dimension(0, 128), Dimension(0, 256), Dimension(64), Dimension(0, 72)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {1000, Dimension(2, 5)},
            {Dimension(0, 128), Dimension(0, 256), Dimension(64), Dimension(64)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {1000, Dimension(0, 5)},
            {Dimension(0, 512), Dimension(256), Dimension(0, 640), Dimension(0, 330)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {1000, Dimension(0, 5)},
            {Dimension(0, 512), Dimension(256), Dimension(0, 640), Dimension(256)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {1000, Dimension(2, 4)},
            {Dimension(0, 512), Dimension(256), Dimension(256), Dimension(0, 720)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {1000, Dimension(2, 4)},
            {Dimension(0, 380), Dimension(256), Dimension(256), Dimension(256)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {1000, Dimension(3, 4)},
            {Dimension(0, 380), Dimension(256), Dimension(256), Dimension(256)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {1000, Dimension(3, 4)},
            {Dimension(128), Dimension(0, 256), Dimension(0, 640), Dimension(0, 330)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {1000, Dimension(0, 6)},
            {Dimension(128), Dimension(0, 256), Dimension(0, 640), Dimension(128)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {1000, Dimension(0, 6)},
            {Dimension(128), Dimension(0, 256), Dimension(128), Dimension(0, 720)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {1000, Dimension(3, 7)},
            {Dimension(128), Dimension(0, 256), Dimension(128), Dimension(128)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {1000, Dimension(4, 6)},
            {Dimension(256), Dimension(256), Dimension(0, 640), Dimension(0, 330)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {1000, Dimension(4, 6)},
            {Dimension(256), Dimension(256), Dimension(0, 640), Dimension(256)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {1000, Dimension(2, 8)},
            {Dimension(256), Dimension(256), Dimension(256), Dimension(0, 330)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {1000, Dimension(2, 8)},
            {Dimension(256), Dimension(256), Dimension(256), Dimension(256)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {Dimension::dynamic(), Dimension(0, 4)},
            {Dimension(0, 128), Dimension(0, 256), Dimension(0, 64), Dimension(0, 33)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {Dimension::dynamic(), Dimension(0, 4)},
            {Dimension(0, 128), Dimension(0, 256), Dimension(0, 64), Dimension(33)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {Dimension::dynamic(), Dimension(1, 4)},
            {Dimension(0, 128), Dimension(0, 256), Dimension(64), Dimension(0, 72)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {Dimension::dynamic(), Dimension(1, 4)},
            {Dimension(0, 128), Dimension(0, 256), Dimension(64), Dimension(64)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {Dimension::dynamic(), Dimension(2, 4)},
            {Dimension(0, 512), Dimension(256), Dimension(0, 640), Dimension(0, 330)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {Dimension::dynamic(), Dimension(2, 4)},
            {Dimension(0, 512), Dimension(256), Dimension(0, 640), Dimension(256)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {Dimension::dynamic(), Dimension(3, 5)},
            {Dimension(0, 512), Dimension(256), Dimension(256), Dimension(0, 720)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {Dimension::dynamic(), Dimension(3, 5)},
            {Dimension(0, 380), Dimension(256), Dimension(256), Dimension(256)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {Dimension::dynamic(), Dimension(4, 6)},
            {Dimension(128), Dimension(0, 256), Dimension(0, 640), Dimension(0, 330)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {Dimension::dynamic(), Dimension(4, 6)},
            {Dimension(128), Dimension(0, 256), Dimension(0, 640), Dimension(128)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {Dimension::dynamic(), Dimension(3, 8)},
            {Dimension(128), Dimension(0, 256), Dimension(128), Dimension(0, 720)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {Dimension::dynamic(), Dimension(3, 8)},
            {Dimension(128), Dimension(0, 256), Dimension(128), Dimension(128)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {Dimension::dynamic(), Dimension(4, 11)},
            {Dimension(256), Dimension(256), Dimension(0, 640), Dimension(0, 330)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {Dimension::dynamic(), Dimension(4, 11)},
            {Dimension(256), Dimension(256), Dimension(0, 640), Dimension(256)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {Dimension::dynamic(), Dimension(2, 16)},
            {Dimension(256), Dimension(256), Dimension(256), Dimension(0, 330)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
        ROIFeatureIntervalsTestParams{
            {Dimension::dynamic(), Dimension(2, 16)},
            {Dimension(256), Dimension(256), Dimension(256), Dimension(256)},
            {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}}),
    PrintToDummyParamName());

struct ROIFeatureIntervalsSameFirstDimsTestParams
{
    PartialShape input_shape;
    Dimension channels[4];
};

struct ROIFeatureIntervalsSameFirstDimsTest
    : ::testing::TestWithParam<ROIFeatureIntervalsSameFirstDimsTestParams>
{
};

TEST_P(ROIFeatureIntervalsSameFirstDimsTest, detectron_roi_feature_extractor_intervals_1)
{
    auto params = GetParam();

    Attrs attrs;
    attrs.aligned = false;
    attrs.output_size = 14;
    attrs.sampling_ratio = 2;
    attrs.pyramid_scales = {4, 8, 16, 32};

    auto layer0_channels = params.channels[0];
    auto layer1_channels = params.channels[1];
    auto layer2_channels = params.channels[2];
    auto layer3_channels = params.channels[3];

    auto layer0_shape = PartialShape{1, layer0_channels, 200, 336};
    auto layer1_shape = PartialShape{1, layer1_channels, 100, 168};
    auto layer2_shape = PartialShape{1, layer2_channels, 50, 84};
    auto layer3_shape = PartialShape{1, layer3_channels, 25, 42};

    auto expected_channels = layer0_channels & layer1_channels & layer2_channels & layer3_channels;

    auto ref_out_shape = PartialShape{params.input_shape[0], expected_channels, 14, 14};

    auto input = std::make_shared<op::Parameter>(element::f32, params.input_shape);
    auto pyramid_layer0 = std::make_shared<op::Parameter>(element::f32, layer0_shape);
    auto pyramid_layer1 = std::make_shared<op::Parameter>(element::f32, layer1_shape);
    auto pyramid_layer2 = std::make_shared<op::Parameter>(element::f32, layer2_shape);
    auto pyramid_layer3 = std::make_shared<op::Parameter>(element::f32, layer3_shape);

    auto roi = std::make_shared<ExperimentalROI>(
        NodeVector{input, pyramid_layer0, pyramid_layer1, pyramid_layer2, pyramid_layer3}, attrs);

    ASSERT_EQ(roi->get_output_element_type(0), element::f32);
    ASSERT_TRUE(roi->get_output_partial_shape(0).same_scheme(ref_out_shape));
}

INSTANTIATE_TEST_SUITE_P(
    type_prop,
    ROIFeatureIntervalsSameFirstDimsTest,
    ::testing::Values(
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension(1000), Dimension(4)},
            {Dimension(0, 128), Dimension(0, 256), Dimension(0, 64), Dimension(0, 33)}},
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension(1000), Dimension(4)},
            {Dimension(0, 128), Dimension(0, 256), Dimension(0, 64), Dimension(33)}},
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension(1000), Dimension(4)},
            {Dimension(0, 128), Dimension(0, 256), Dimension(64), Dimension(0, 72)}},
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension(1000), Dimension(4)},
            {Dimension(0, 128), Dimension(0, 256), Dimension(64), Dimension(64)}},
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension(1000), Dimension(4)},
            {Dimension(0, 512), Dimension(256), Dimension(0, 640), Dimension(0, 330)}},
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension(1000), Dimension(4)},
            {Dimension(0, 512), Dimension(256), Dimension(0, 640), Dimension(256)}},
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension(1000), Dimension(4)},
            {Dimension(0, 512), Dimension(256), Dimension(256), Dimension(0, 720)}},
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension(1000), Dimension(4)},
            {Dimension(0, 380), Dimension(256), Dimension(256), Dimension(256)}},
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension(1000), Dimension(4)},
            {Dimension(128), Dimension(0, 256), Dimension(0, 640), Dimension(0, 330)}},
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension(1000), Dimension(4)},
            {Dimension(128), Dimension(0, 256), Dimension(0, 640), Dimension(128)}},
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension(1000), Dimension(4)},
            {Dimension(128), Dimension(0, 256), Dimension(128), Dimension(0, 720)}},
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension(1000), Dimension(4)},
            {Dimension(128), Dimension(0, 256), Dimension(128), Dimension(128)}},
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension(1000), Dimension(4)},
            {Dimension(256), Dimension(256), Dimension(0, 640), Dimension(0, 330)}},
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension(1000), Dimension(4)},
            {Dimension(256), Dimension(256), Dimension(0, 640), Dimension(256)}},
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension(1000), Dimension(4)},
            {Dimension(256), Dimension(256), Dimension(256), Dimension(0, 330)}},
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension(1000), Dimension(4)},
            {Dimension(256), Dimension(256), Dimension(256), Dimension(256)}},
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension::dynamic(), Dimension(4)},
            {Dimension(0, 128), Dimension(0, 256), Dimension(0, 64), Dimension(0, 33)}},
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension::dynamic(), Dimension(4)},
            {Dimension(0, 128), Dimension(0, 256), Dimension(0, 64), Dimension(33)}},
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension::dynamic(), Dimension(4)},
            {Dimension(0, 128), Dimension(0, 256), Dimension(64), Dimension(0, 72)}},
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension::dynamic(), Dimension(4)},
            {Dimension(0, 128), Dimension(0, 256), Dimension(64), Dimension(64)}},
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension::dynamic(), Dimension(4)},
            {Dimension(0, 512), Dimension(256), Dimension(0, 640), Dimension(0, 330)}},
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension::dynamic(), Dimension(4)},
            {Dimension(0, 512), Dimension(256), Dimension(0, 640), Dimension(256)}},
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension::dynamic(), Dimension(4)},
            {Dimension(0, 512), Dimension(256), Dimension(256), Dimension(0, 720)}},
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension::dynamic(), Dimension(4)},
            {Dimension(0, 380), Dimension(256), Dimension(256), Dimension(256)}},
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension::dynamic(), Dimension(4)},
            {Dimension(128), Dimension(0, 256), Dimension(0, 640), Dimension(0, 330)}},
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension::dynamic(), Dimension(4)},
            {Dimension(128), Dimension(0, 256), Dimension(0, 640), Dimension(128)}},
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension::dynamic(), Dimension(4)},
            {Dimension(128), Dimension(0, 256), Dimension(128), Dimension(0, 720)}},
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension::dynamic(), Dimension(4)},
            {Dimension(128), Dimension(0, 256), Dimension(128), Dimension(128)}},
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension::dynamic(), Dimension(4)},
            {Dimension(256), Dimension(256), Dimension(0, 640), Dimension(0, 330)}},
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension::dynamic(), Dimension(4)},
            {Dimension(256), Dimension(256), Dimension(0, 640), Dimension(256)}},
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension::dynamic(), Dimension(4)},
            {Dimension(256), Dimension(256), Dimension(256), Dimension(0, 330)}},
        ROIFeatureIntervalsSameFirstDimsTestParams{
            {Dimension::dynamic(), Dimension(4)},
            {Dimension(256), Dimension(256), Dimension(256), Dimension(256)}}),
    PrintToDummyParamName());
