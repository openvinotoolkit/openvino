// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include <array>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/opsets/opset11.hpp"

using namespace ov;
using namespace ov::opset11;
using namespace testing;

class TypePropExperimentalDetectronROIFeatureExtractorV6Test
    : public TypePropOpTest<op::v6::ExperimentalDetectronROIFeatureExtractor> {
protected:
    using Attrs = op::v6::ExperimentalDetectronROIFeatureExtractor::Attributes;

    static Attrs make_attrs(int64_t out_size) {
        return {out_size, 2, {4, 8, 16, 32}, false};
    };
};

using Attrs = op::v6::ExperimentalDetectronROIFeatureExtractor::Attributes;
using ExperimentalROI = op::v6::ExperimentalDetectronROIFeatureExtractor;

TEST_F(TypePropExperimentalDetectronROIFeatureExtractorV6Test, default_ctor) {
    const auto rois = std::make_shared<Parameter>(element::f16, Shape{1000, 4});
    const auto pyramid_layer0 = std::make_shared<Parameter>(element::f16, PartialShape{1, 256, 200, 336});
    const auto pyramid_layer1 = std::make_shared<Parameter>(element::f16, PartialShape{1, 256, 100, 168});
    const auto pyramid_layer2 = std::make_shared<Parameter>(element::f16, PartialShape{1, 256, 50, 84});
    const auto pyramid_layer3 = std::make_shared<Parameter>(element::f16, PartialShape{1, 256, 25, 42});

    const auto op = make_op();
    op->set_arguments(OutputVector{rois, pyramid_layer0, pyramid_layer1, pyramid_layer2, pyramid_layer3});
    op->set_attrs({21, 2, {1, 2, 4, 8}, true});
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_input_size(), 5);
    EXPECT_EQ(op->get_output_size(), 2);
    EXPECT_THAT(op->outputs(), Each(Property("Element type", &Output<Node>::get_element_type, element::f16)));
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("ROIs feat shape", &Output<Node>::get_shape, Shape({1000, 256, 21, 21})),
                            Property("ROIs order shape", &Output<Node>::get_shape, Shape({1000, 4}))));
}

TEST_F(TypePropExperimentalDetectronROIFeatureExtractorV6Test, static_shapes) {
    const auto rois = std::make_shared<Parameter>(element::f32, Shape{1000, 4});
    const auto pyramid_layer0 = std::make_shared<Parameter>(element::f32, Shape{1, 256, 200, 336});
    const auto pyramid_layer1 = std::make_shared<Parameter>(element::f32, Shape{1, 256, 100, 168});
    const auto pyramid_layer2 = std::make_shared<Parameter>(element::f32, Shape{1, 256, 50, 84});
    const auto pyramid_layer3 = std::make_shared<Parameter>(element::f32, Shape{1, 256, 25, 42});

    const auto op =
        make_op(NodeVector{rois, pyramid_layer0, pyramid_layer1, pyramid_layer2, pyramid_layer3}, make_attrs(14));

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_shape(0), (Shape{1000, 256, 14, 14}));
}

TEST_F(TypePropExperimentalDetectronROIFeatureExtractorV6Test, dims_and_labels_propagation_all_inputs_labeled) {
    auto in_shape = PartialShape{{100, 200}, 4};
    auto l0_shape = PartialShape{1, {2, 20}, 10, 10};
    auto l1_shape = PartialShape{1, {0, 20}, 32, 32};
    auto l2_shape = PartialShape{1, {1, 10}, 16, 16};

    auto in_symbols = set_shape_symbols(in_shape);
    auto l0_symbols = set_shape_symbols(l0_shape);
    set_shape_symbols(l1_shape);
    set_shape_symbols(l2_shape);

    const auto rois = std::make_shared<Parameter>(element::f64, in_shape);
    const auto pyramid_layer0 = std::make_shared<Parameter>(element::f64, l0_shape);
    const auto pyramid_layer1 = std::make_shared<Parameter>(element::f64, l1_shape);
    const auto pyramid_layer2 = std::make_shared<Parameter>(element::f64, l2_shape);

    const auto op = make_op(OutputVector{rois, pyramid_layer0, pyramid_layer1, pyramid_layer2}, make_attrs(7));

    EXPECT_THAT(op->outputs(), Each(Property("Element type", &Output<Node>::get_element_type, element::f64)));
    EXPECT_THAT(
        op->outputs(),
        ElementsAre(
            Property("ROIs feat shape",
                     &Output<Node>::get_partial_shape,
                     AllOf(PartialShape({{100, 200}, {2, 10}, 7, 7}),
                           ResultOf(get_shape_symbols, ElementsAre(in_symbols[0], l0_symbols[1], nullptr, nullptr)))),
            Property("ROIs order shape",
                     &Output<Node>::get_partial_shape,
                     AllOf(PartialShape({{100, 200}, 4}),
                           ResultOf(get_shape_symbols, ElementsAre(in_symbols[0], in_symbols[1]))))));
}

TEST_F(TypePropExperimentalDetectronROIFeatureExtractorV6Test, dims_and_labels_propagation_not_all_inputs_labeled) {
    auto in_shape = PartialShape{{100, 200}, 4};
    auto l0_shape = PartialShape{1, 5, 10, 10};
    auto l1_shape = PartialShape{1, {0, 20}, 32, 32};
    auto l2_shape = PartialShape{1, {1, 10}, 16, 16};

    auto in_symbol = set_shape_symbols(in_shape);
    auto l0_symbol = set_shape_symbols(l0_shape);
    set_shape_symbols(l1_shape);

    const auto rois = std::make_shared<Parameter>(element::bf16, in_shape);
    const auto pyramid_layer0 = std::make_shared<Parameter>(element::bf16, l0_shape);
    const auto pyramid_layer1 = std::make_shared<Parameter>(element::bf16, l1_shape);
    const auto pyramid_layer2 = std::make_shared<Parameter>(element::bf16, l2_shape);

    const auto op = make_op(NodeVector{rois, pyramid_layer0, pyramid_layer1, pyramid_layer2}, make_attrs(7));

    EXPECT_THAT(op->outputs(), Each(Property("Element type", &Output<Node>::get_element_type, element::bf16)));
    EXPECT_THAT(
        op->outputs(),
        ElementsAre(
            Property("ROIs feat shape",
                     &Output<Node>::get_partial_shape,
                     AllOf(PartialShape({{100, 200}, 5, 7, 7}),
                           ResultOf(get_shape_symbols, ElementsAre(in_symbol[0], l0_symbol[1], nullptr, nullptr)))),
            Property("ROIs order shape",
                     &Output<Node>::get_partial_shape,
                     AllOf(PartialShape({{100, 200}, 4}),
                           ResultOf(get_shape_symbols, ElementsAre(in_symbol[0], in_symbol[1]))))));
}

TEST_F(TypePropExperimentalDetectronROIFeatureExtractorV6Test, all_inputs_dynamic_rank) {
    const auto rois = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());
    const auto pyramid_layer0 = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());
    const auto pyramid_layer1 = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());

    const auto op = make_op(OutputVector{rois, pyramid_layer0, pyramid_layer1}, make_attrs(7));

    EXPECT_THAT(op->outputs(), Each(Property("Element type", &Output<Node>::get_element_type, element::f16)));
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("ROIs feat shape",
                                     &Output<Node>::get_partial_shape,
                                     AllOf(PartialShape({-1, -1, 7, 7}), ResultOf(get_shape_symbols, Each(nullptr)))),
                            Property("ROIs order shape",
                                     &Output<Node>::get_partial_shape,
                                     AllOf(PartialShape({-1, 4}), ResultOf(get_shape_symbols, Each(nullptr))))));
}

TEST_F(TypePropExperimentalDetectronROIFeatureExtractorV6Test, all_inputs_static_rank_but_dynamic_dims) {
    const auto rois = std::make_shared<Parameter>(element::f16, PartialShape::dynamic(2));
    const auto pyramid_layer0 = std::make_shared<Parameter>(element::f16, PartialShape::dynamic(4));
    const auto pyramid_layer1 = std::make_shared<Parameter>(element::f16, PartialShape::dynamic(4));
    const auto pyramid_layer2 = std::make_shared<Parameter>(element::f16, PartialShape::dynamic(4));
    const auto pyramid_layer3 = std::make_shared<Parameter>(element::f16, PartialShape::dynamic(4));
    const auto pyramid_layer4 = std::make_shared<Parameter>(element::f16, PartialShape::dynamic(4));

    const auto op =
        make_op(OutputVector{rois, pyramid_layer0, pyramid_layer1, pyramid_layer2, pyramid_layer3, pyramid_layer4},
                make_attrs(7));

    EXPECT_THAT(op->outputs(), Each(Property("Element type", &Output<Node>::get_element_type, element::f16)));
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("ROIs feat shape",
                                     &Output<Node>::get_partial_shape,
                                     AllOf(PartialShape({-1, -1, 7, 7}), ResultOf(get_shape_symbols, Each(nullptr)))),
                            Property("ROIs order shape",
                                     &Output<Node>::get_partial_shape,
                                     AllOf(PartialShape({-1, 4}), ResultOf(get_shape_symbols, Each(nullptr))))));
}

TEST_F(TypePropExperimentalDetectronROIFeatureExtractorV6Test, input_not_floating_point) {
    const auto bad_param = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto ok_param = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{bad_param, ok_param, ok_param}, make_attrs(7)),
                    NodeValidationFailure,
                    HasSubstr("Input[0] type 'i32' is not floating point or not same as others inputs"));

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{ok_param, bad_param, ok_param, ok_param}, make_attrs(7)),
                    NodeValidationFailure,
                    HasSubstr("Input[1] type 'i32' is not floating point or not same as others inputs"));

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{ok_param, ok_param, bad_param}, make_attrs(7)),
                    NodeValidationFailure,
                    HasSubstr("Input[2] type 'i32' is not floating point or not same as others inputs"));

    OV_EXPECT_THROW(
        std::ignore = make_op(OutputVector{ok_param, ok_param, ok_param, ok_param, bad_param}, make_attrs(7)),
        NodeValidationFailure,
        HasSubstr("Input[4] type 'i32' is not floating point or not same as others inputs"));
}

TEST_F(TypePropExperimentalDetectronROIFeatureExtractorV6Test, input_mixed_floating_point_type) {
    const auto f32_param = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto f16_param = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{f32_param, f16_param, f16_param, f16_param}, make_attrs(100)),
                    NodeValidationFailure,
                    HasSubstr("Input[1] type 'f16' is not floating point or not same as others inputs"));

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{f16_param, f32_param, f16_param, f16_param}, make_attrs(100)),
                    NodeValidationFailure,
                    HasSubstr("Input[1] type 'f32' is not floating point or not same as others inputs"));

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{f16_param, f16_param, f32_param}, make_attrs(100)),
                    NodeValidationFailure,
                    HasSubstr("Input[2] type 'f32' is not floating point or not same as others inputs"));

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{f16_param, f16_param, f16_param, f32_param}, make_attrs(100)),
                    NodeValidationFailure,
                    HasSubstr("Input[3] type 'f32' is not floating point or not same as others inputs"));
}

TEST_F(TypePropExperimentalDetectronROIFeatureExtractorV6Test, rois_not_2d) {
    const auto layer = std::make_shared<Parameter>(element::f16, PartialShape::dynamic(4));

    OV_EXPECT_THROW(
        std::ignore =
            make_op(OutputVector{std::make_shared<Parameter>(element::f16, PartialShape{20}), layer}, make_attrs(7)),
        NodeValidationFailure,
        HasSubstr("Input rois rank must be equal to 2"));

    OV_EXPECT_THROW(std::ignore = make_op(
                        OutputVector{std::make_shared<Parameter>(element::f16, PartialShape{20, 4, 1}), layer, layer},
                        make_attrs(10)),
                    NodeValidationFailure,
                    HasSubstr("Input rois rank must be equal to 2"));
}

TEST_F(TypePropExperimentalDetectronROIFeatureExtractorV6Test, rois_2nd_dim_not_compatible) {
    const auto layer = std::make_shared<Parameter>(element::f16, PartialShape::dynamic(4));

    OV_EXPECT_THROW(
        std::ignore = make_op(OutputVector{std::make_shared<Parameter>(element::f16, PartialShape{20, {0, 3}}), layer},
                              make_attrs(7)),
        NodeValidationFailure,
        HasSubstr("The last dimension of the 'input_rois' input must be equal to 4"));

    OV_EXPECT_THROW(
        std::ignore =
            make_op(OutputVector{std::make_shared<Parameter>(element::f16, PartialShape{20, {5, -1}}), layer, layer},
                    make_attrs(10)),
        NodeValidationFailure,
        HasSubstr("The last dimension of the 'input_rois' input must be equal to 4"));
}

TEST_F(TypePropExperimentalDetectronROIFeatureExtractorV6Test, layers_not_4d) {
    const auto rois = std::make_shared<Parameter>(element::f16, PartialShape::dynamic(2));
    const auto layer = std::make_shared<Parameter>(element::f16, PartialShape::dynamic(4));

    OV_EXPECT_THROW(
        std::ignore = make_op(OutputVector{rois, std::make_shared<Parameter>(element::f16, PartialShape::dynamic(3))},
                              make_attrs(10)),
        NodeValidationFailure,
        HasSubstr("Rank of each element of the pyramid must be equal to 4"));

    OV_EXPECT_THROW(
        std::ignore = make_op(OutputVector{rois, std::make_shared<Parameter>(element::f16, PartialShape::dynamic(5))},
                              make_attrs(10)),
        NodeValidationFailure,
        HasSubstr("Rank of each element of the pyramid must be equal to 4"));

    OV_EXPECT_THROW(std::ignore = make_op(
                        OutputVector{rois, layer, std::make_shared<Parameter>(element::f16, PartialShape::dynamic(3))},
                        make_attrs(10)),
                    NodeValidationFailure,
                    HasSubstr("Rank of each element of the pyramid must be equal to 4"));

    OV_EXPECT_THROW(
        std::ignore = make_op(
            OutputVector{rois, layer, std::make_shared<Parameter>(element::f16, PartialShape::dynamic(3)), layer},
            make_attrs(10)),
        NodeValidationFailure,
        HasSubstr("Rank of each element of the pyramid must be equal to 4"));
}

TEST_F(TypePropExperimentalDetectronROIFeatureExtractorV6Test, layers_1st_dim_not_compatible_1) {
    const auto rois = std::make_shared<Parameter>(element::f16, PartialShape::dynamic(2));
    const auto layer = std::make_shared<Parameter>(element::f16, PartialShape::dynamic(4));

    OV_EXPECT_THROW(
        std::ignore = make_op(OutputVector{rois, std::make_shared<Parameter>(element::f16, PartialShape{2, 5, 16, 16})},
                              make_attrs(10)),
        NodeValidationFailure,
        HasSubstr("The first dimension of each pyramid element must be equal to 1"));

    OV_EXPECT_THROW(
        std::ignore = make_op(
            OutputVector{rois, layer, std::make_shared<Parameter>(element::f16, PartialShape{{2, -1}, -1, -1, -1})},
            make_attrs(10)),
        NodeValidationFailure,
        HasSubstr("The first dimension of each pyramid element must be equal to 1"));

    OV_EXPECT_THROW(
        std::ignore = make_op(OutputVector{rois,
                                           layer,
                                           layer,
                                           std::make_shared<Parameter>(element::f16, PartialShape{5, -1, -1, -1}),
                                           layer},
                              make_attrs(10)),
        NodeValidationFailure,
        HasSubstr("The first dimension of each pyramid element must be equal to 1"));
}

TEST_F(TypePropExperimentalDetectronROIFeatureExtractorV6Test, num_channels_not_same_on_all_layers) {
    const auto rois = std::make_shared<Parameter>(element::f16, PartialShape::dynamic(2));
    const auto layer = std::make_shared<Parameter>(element::f16, PartialShape::dynamic(4));

    OV_EXPECT_THROW(
        std::ignore = make_op(OutputVector{rois,
                                           std::make_shared<Parameter>(element::f16, PartialShape{1, {0, 3}, -1, -1}),
                                           std::make_shared<Parameter>(element::f16, PartialShape{1, {14, 5}, -1, -1})},
                              make_attrs(10)),
        NodeValidationFailure,
        HasSubstr("The number of channels must be the same for all layers of the pyramid"));

    OV_EXPECT_THROW(
        std::ignore = make_op(OutputVector{rois,
                                           std::make_shared<Parameter>(element::f16, PartialShape{1, {2, 3}, -1, -1}),
                                           layer,
                                           std::make_shared<Parameter>(element::f16, PartialShape{1, {4, 5}, -1, -1}),
                                           layer},
                              make_attrs(10)),
        NodeValidationFailure,
        HasSubstr("The number of channels must be the same for all layers of the pyramid"));

    OV_EXPECT_THROW(
        std::ignore = make_op(OutputVector{rois,
                                           std::make_shared<Parameter>(element::f16, PartialShape{1, {2, 3}, -1, -1}),
                                           std::make_shared<Parameter>(element::f16, PartialShape{1, {4, 5}, -1, -1}),
                                           layer},
                              make_attrs(10)),
        NodeValidationFailure,
        HasSubstr("The number of channels must be the same for all layers of the pyramid"));
}

using ROIFeatureIntervalsTestParams = std::tuple<PartialShape, std::array<Dimension, 4>, std::array<Dimension, 4>>;

class ROIFeatureIntervalsTest : public TypePropExperimentalDetectronROIFeatureExtractorV6Test,
                                public WithParamInterface<ROIFeatureIntervalsTestParams> {
protected:
    void SetUp() override {
        std::tie(input_shape, channels, first_dims) = GetParam();

        exp_rois_feat_shape = PartialShape{input_shape[0],
                                           channels[0] & channels[1] & channels[2] & channels[3],
                                           exp_out_size,
                                           exp_out_size};
    }

    PartialShape input_shape, exp_rois_feat_shape;
    std::array<Dimension, 4> channels, first_dims;
    int64_t exp_out_size = 14;
};

INSTANTIATE_TEST_SUITE_P(
    type_prop,
    ROIFeatureIntervalsTest,
    Values(ROIFeatureIntervalsTestParams{{1000, Dimension(0, 5)},
                                         {Dimension(0, 128), Dimension(0, 256), Dimension(0, 64), Dimension(0, 33)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{1000, Dimension(0, 5)},
                                         {Dimension(0, 128), Dimension(0, 256), Dimension(0, 64), Dimension(33)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{1000, Dimension(2, 5)},
                                         {Dimension(0, 128), Dimension(0, 256), Dimension(64), Dimension(0, 72)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{1000, Dimension(2, 5)},
                                         {Dimension(0, 128), Dimension(0, 256), Dimension(64), Dimension(64)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{1000, Dimension(0, 5)},
                                         {Dimension(0, 512), Dimension(256), Dimension(0, 640), Dimension(0, 330)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{1000, Dimension(0, 5)},
                                         {Dimension(0, 512), Dimension(256), Dimension(0, 640), Dimension(256)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{1000, Dimension(2, 4)},
                                         {Dimension(0, 512), Dimension(256), Dimension(256), Dimension(0, 720)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{1000, Dimension(2, 4)},
                                         {Dimension(0, 380), Dimension(256), Dimension(256), Dimension(256)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{1000, Dimension(3, 4)},
                                         {Dimension(0, 380), Dimension(256), Dimension(256), Dimension(256)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{1000, Dimension(3, 4)},
                                         {Dimension(128), Dimension(0, 256), Dimension(0, 640), Dimension(0, 330)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{1000, Dimension(0, 6)},
                                         {Dimension(128), Dimension(0, 256), Dimension(0, 640), Dimension(128)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{1000, Dimension(0, 6)},
                                         {Dimension(128), Dimension(0, 256), Dimension(128), Dimension(0, 720)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{1000, Dimension(3, 7)},
                                         {Dimension(128), Dimension(0, 256), Dimension(128), Dimension(128)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{1000, Dimension(4, 6)},
                                         {Dimension(256), Dimension(256), Dimension(0, 640), Dimension(0, 330)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{1000, Dimension(4, 6)},
                                         {Dimension(256), Dimension(256), Dimension(0, 640), Dimension(256)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{1000, Dimension(2, 8)},
                                         {Dimension(256), Dimension(256), Dimension(256), Dimension(0, 330)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{1000, Dimension(2, 8)},
                                         {Dimension(256), Dimension(256), Dimension(256), Dimension(256)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{Dimension::dynamic(), Dimension(0, 4)},
                                         {Dimension(0, 128), Dimension(0, 256), Dimension(0, 64), Dimension(0, 33)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{Dimension::dynamic(), Dimension(0, 4)},
                                         {Dimension(0, 128), Dimension(0, 256), Dimension(0, 64), Dimension(33)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{Dimension::dynamic(), Dimension(1, 4)},
                                         {Dimension(0, 128), Dimension(0, 256), Dimension(64), Dimension(0, 72)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{Dimension::dynamic(), Dimension(1, 4)},
                                         {Dimension(0, 128), Dimension(0, 256), Dimension(64), Dimension(64)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{Dimension::dynamic(), Dimension(2, 4)},
                                         {Dimension(0, 512), Dimension(256), Dimension(0, 640), Dimension(0, 330)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{Dimension::dynamic(), Dimension(2, 4)},
                                         {Dimension(0, 512), Dimension(256), Dimension(0, 640), Dimension(256)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{Dimension::dynamic(), Dimension(3, 5)},
                                         {Dimension(0, 512), Dimension(256), Dimension(256), Dimension(0, 720)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{Dimension::dynamic(), Dimension(3, 5)},
                                         {Dimension(0, 380), Dimension(256), Dimension(256), Dimension(256)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{Dimension::dynamic(), Dimension(4, 6)},
                                         {Dimension(128), Dimension(0, 256), Dimension(0, 640), Dimension(0, 330)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{Dimension::dynamic(), Dimension(4, 6)},
                                         {Dimension(128), Dimension(0, 256), Dimension(0, 640), Dimension(128)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{Dimension::dynamic(), Dimension(3, 8)},
                                         {Dimension(128), Dimension(0, 256), Dimension(128), Dimension(0, 720)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{Dimension::dynamic(), Dimension(3, 8)},
                                         {Dimension(128), Dimension(0, 256), Dimension(128), Dimension(128)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{Dimension::dynamic(), Dimension(4, 11)},
                                         {Dimension(256), Dimension(256), Dimension(0, 640), Dimension(0, 330)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{Dimension::dynamic(), Dimension(4, 11)},
                                         {Dimension(256), Dimension(256), Dimension(0, 640), Dimension(256)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{Dimension::dynamic(), Dimension(2, 16)},
                                         {Dimension(256), Dimension(256), Dimension(256), Dimension(0, 330)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}},
           ROIFeatureIntervalsTestParams{{Dimension::dynamic(), Dimension(2, 16)},
                                         {Dimension(256), Dimension(256), Dimension(256), Dimension(256)},
                                         {Dimension(0, 2), Dimension(1, 3), Dimension(0, 5), Dimension(1, 2)}}),
    PrintToStringParamName());

TEST_P(ROIFeatureIntervalsTest, interval_shape_inference) {
    auto layer0_shape = PartialShape{first_dims[0], channels[0], 200, 336};
    auto layer1_shape = PartialShape{first_dims[1], channels[1], 100, 168};
    auto layer2_shape = PartialShape{first_dims[2], channels[2], 50, 84};
    auto layer3_shape = PartialShape{first_dims[3], channels[3], 25, 42};

    auto rois = std::make_shared<Parameter>(element::f32, input_shape);
    auto pyramid_layer0 = std::make_shared<Parameter>(element::f32, layer0_shape);
    auto pyramid_layer1 = std::make_shared<Parameter>(element::f32, layer1_shape);
    auto pyramid_layer2 = std::make_shared<Parameter>(element::f32, layer2_shape);
    auto pyramid_layer3 = std::make_shared<Parameter>(element::f32, layer3_shape);

    auto op = make_op(NodeVector{rois, pyramid_layer0, pyramid_layer1, pyramid_layer2, pyramid_layer3},
                      make_attrs(exp_out_size));

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), exp_rois_feat_shape);
}

TEST_P(ROIFeatureIntervalsTest, interval_shape_inference_layers_1st_dim_static) {
    auto layer0_shape = PartialShape{1, channels[0], 200, 336};
    auto layer1_shape = PartialShape{1, channels[1], 100, 168};
    auto layer2_shape = PartialShape{1, channels[2], 50, 84};
    auto layer3_shape = PartialShape{1, channels[3], 25, 42};

    auto rois = std::make_shared<Parameter>(element::f32, input_shape);
    auto pyramid_layer0 = std::make_shared<Parameter>(element::f32, layer0_shape);
    auto pyramid_layer1 = std::make_shared<Parameter>(element::f32, layer1_shape);
    auto pyramid_layer2 = std::make_shared<Parameter>(element::f32, layer2_shape);
    auto pyramid_layer3 = std::make_shared<Parameter>(element::f32, layer3_shape);

    auto op = make_op(NodeVector{rois, pyramid_layer0, pyramid_layer1, pyramid_layer2, pyramid_layer3},
                      make_attrs(exp_out_size));

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), exp_rois_feat_shape);
}
