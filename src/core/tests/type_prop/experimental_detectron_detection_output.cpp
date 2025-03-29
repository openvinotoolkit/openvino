// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/opsets/opset11.hpp"

using namespace ov;
using namespace ov::opset11;
using namespace testing;

class TypePropExperimentalDetectronDetectionOutputV6Test
    : public TypePropOpTest<op::v6::ExperimentalDetectronDetectionOutput> {
protected:
    using Attrs = op::v6::ExperimentalDetectronDetectionOutput::Attributes;

    static Attrs make_attrs(size_t max_detection, int64_t num_classes) {
        return {.05f, .5f, 4.1352f, num_classes, 20, max_detection, false, {10.0f, 10.0f, 5.0f, 5.0f}};
    }

    size_t exp_detection = 25;
    int64_t num_classes = 81;
};

TEST_F(TypePropExperimentalDetectronDetectionOutputV6Test, default_ctor) {
    constexpr size_t max_detection = 100;

    const auto rois = std::make_shared<Parameter>(element::f32, Shape{1000, 4});
    const auto deltas = std::make_shared<Parameter>(element::f32, Shape{1000, 324});
    const auto scores = std::make_shared<Parameter>(element::f32, Shape{1000, 81});
    const auto im_info = std::make_shared<Parameter>(element::f32, Shape{1, 3});

    const auto op = make_op();
    op->set_arguments(OutputVector{rois, deltas, scores, im_info});
    op->set_attrs(make_attrs(max_detection, num_classes));
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_input_size(), 4);
    EXPECT_EQ(op->get_output_size(), 3);
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Boxes type", &Output<Node>::get_element_type, element::f32),
                            Property("Classes type", &Output<Node>::get_element_type, element::i32),
                            Property("Scores types", &Output<Node>::get_element_type, element::f32)));
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Boxes shape", &Output<Node>::get_shape, Shape({max_detection, 4})),
                            Property("Classes shape", &Output<Node>::get_shape, Shape({max_detection})),
                            Property("Scores shape", &Output<Node>::get_shape, Shape({max_detection}))));
}

TEST_F(TypePropExperimentalDetectronDetectionOutputV6Test, basic_shape_inference) {
    const auto rois = std::make_shared<Parameter>(element::f64, Shape{1000, 4});
    const auto deltas = std::make_shared<Parameter>(element::f64, Shape{1000, 324});
    const auto scores = std::make_shared<Parameter>(element::f64, Shape{1000, 81});
    const auto im_info = std::make_shared<Parameter>(element::f64, Shape{1, 3});

    const auto op = make_op(rois, deltas, scores, im_info, make_attrs(exp_detection, num_classes));

    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Boxes type", &Output<Node>::get_element_type, element::f64),
                            Property("Classes type", &Output<Node>::get_element_type, element::i32),
                            Property("Scores types", &Output<Node>::get_element_type, element::f64)));
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Boxes shape", &Output<Node>::get_shape, Shape({exp_detection, 4})),
                            Property("Classes shape", &Output<Node>::get_shape, Shape({exp_detection})),
                            Property("Scores shape", &Output<Node>::get_shape, Shape({exp_detection}))));
}

TEST_F(TypePropExperimentalDetectronDetectionOutputV6Test, all_input_got_dynamic_type) {
    const auto rois = std::make_shared<Parameter>(element::dynamic, Shape{1000, 4});
    const auto deltas = std::make_shared<Parameter>(element::dynamic, Shape{1000, 324});
    const auto scores = std::make_shared<Parameter>(element::dynamic, Shape{1000, 81});
    const auto im_info = std::make_shared<Parameter>(element::dynamic, Shape{1, 3});

    const auto op = make_op(rois, deltas, scores, im_info, make_attrs(exp_detection, num_classes));

    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Boxes type", &Output<Node>::get_element_type, element::dynamic),
                            Property("Classes type", &Output<Node>::get_element_type, element::i32),
                            Property("Scores types", &Output<Node>::get_element_type, element::dynamic)));
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Boxes shape", &Output<Node>::get_shape, Shape({exp_detection, 4})),
                            Property("Classes shape", &Output<Node>::get_shape, Shape({exp_detection})),
                            Property("Scores shape", &Output<Node>::get_shape, Shape({exp_detection}))));
}

TEST_F(TypePropExperimentalDetectronDetectionOutputV6Test, some_input_got_dynamic_type) {
    const auto rois = std::make_shared<Parameter>(element::dynamic, Shape{1000, 4});
    const auto deltas = std::make_shared<Parameter>(element::f64, Shape{1000, 324});
    const auto scores = std::make_shared<Parameter>(element::dynamic, Shape{1000, 81});
    const auto im_info = std::make_shared<Parameter>(element::f64, Shape{1, 3});

    const auto op = make_op(rois, deltas, scores, im_info, make_attrs(exp_detection, num_classes));

    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Boxes type", &Output<Node>::get_element_type, element::f64),
                            Property("Classes type", &Output<Node>::get_element_type, element::i32),
                            Property("Scores types", &Output<Node>::get_element_type, element::f64)));
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Boxes shape", &Output<Node>::get_shape, Shape({exp_detection, 4})),
                            Property("Classes shape", &Output<Node>::get_shape, Shape({exp_detection})),
                            Property("Scores shape", &Output<Node>::get_shape, Shape({exp_detection}))));
}

TEST_F(TypePropExperimentalDetectronDetectionOutputV6Test, interval_shapes) {
    auto rois_shape = PartialShape{{1, 201600}, {1, 4}};
    auto deltas_shape = PartialShape{{1, 12}, {1, 500}};
    auto scores_shape = PartialShape{{1, 3}, {10, 90}};
    auto im_info_shape = PartialShape{1, {1, 4}};
    set_shape_symbols(rois_shape);
    set_shape_symbols(deltas_shape);
    set_shape_symbols(scores_shape);
    set_shape_symbols(im_info_shape);

    const auto rois = std::make_shared<Parameter>(element::f16, rois_shape);
    const auto deltas = std::make_shared<Parameter>(element::f16, deltas_shape);
    const auto scores = std::make_shared<Parameter>(element::f16, scores_shape);
    const auto im_info = std::make_shared<Parameter>(element::f16, im_info_shape);

    const auto op = make_op(rois, deltas, scores, im_info, make_attrs(25, num_classes));

    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Boxes type", &Output<Node>::get_element_type, element::f16),
                            Property("Classes type", &Output<Node>::get_element_type, element::i32),
                            Property("Scores types", &Output<Node>::get_element_type, element::f16)));
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Boxes shape",
                                     &Output<Node>::get_partial_shape,
                                     AllOf(PartialShape({25, 4}), ResultOf(get_shape_symbols, Each(nullptr)))),
                            Property("Classes shape",
                                     &Output<Node>::get_partial_shape,
                                     AllOf(PartialShape({25}), ResultOf(get_shape_symbols, Each(nullptr)))),
                            Property("Scores shape",
                                     &Output<Node>::get_partial_shape,
                                     AllOf(PartialShape({25}), ResultOf(get_shape_symbols, Each(nullptr))))));
}

TEST_F(TypePropExperimentalDetectronDetectionOutputV6Test, all_inputs_dynamic_rank) {
    const auto rois = std::make_shared<Parameter>(element::bf16, PartialShape::dynamic());
    const auto deltas = std::make_shared<Parameter>(element::bf16, PartialShape::dynamic());
    const auto scores = std::make_shared<Parameter>(element::bf16, PartialShape::dynamic());
    const auto im_info = std::make_shared<Parameter>(element::bf16, PartialShape::dynamic());

    const auto op = make_op(rois, deltas, scores, im_info, make_attrs(25, num_classes));

    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Boxes type", &Output<Node>::get_element_type, element::bf16),
                            Property("Classes type", &Output<Node>::get_element_type, element::i32),
                            Property("Scores types", &Output<Node>::get_element_type, element::bf16)));
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Boxes shape",
                                     &Output<Node>::get_partial_shape,
                                     AllOf(PartialShape({25, 4}), ResultOf(get_shape_symbols, Each(nullptr)))),
                            Property("Classes shape",
                                     &Output<Node>::get_partial_shape,
                                     AllOf(PartialShape({25}), ResultOf(get_shape_symbols, Each(nullptr)))),
                            Property("Scores shape",
                                     &Output<Node>::get_partial_shape,
                                     AllOf(PartialShape({25}), ResultOf(get_shape_symbols, Each(nullptr))))));
}

TEST_F(TypePropExperimentalDetectronDetectionOutputV6Test, input_not_floating_point) {
    const auto bad_param = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto ok_param = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());

    OV_EXPECT_THROW(std::ignore = make_op(bad_param, ok_param, ok_param, ok_param, make_attrs(25, num_classes)),
                    NodeValidationFailure,
                    HasSubstr("Input[0] type 'i32' is not floating point or not same as others inputs"));

    OV_EXPECT_THROW(std::ignore = make_op(ok_param, bad_param, ok_param, ok_param, make_attrs(25, num_classes)),
                    NodeValidationFailure,
                    HasSubstr("Input[1] type 'i32' is not floating point or not same as others inputs"));

    OV_EXPECT_THROW(std::ignore = make_op(ok_param, ok_param, bad_param, ok_param, make_attrs(25, num_classes)),
                    NodeValidationFailure,
                    HasSubstr("Input[2] type 'i32' is not floating point or not same as others inputs"));

    OV_EXPECT_THROW(std::ignore = make_op(ok_param, ok_param, ok_param, bad_param, make_attrs(25, num_classes)),
                    NodeValidationFailure,
                    HasSubstr("Input[3] type 'i32' is not floating point or not same as others inputs"));
}

TEST_F(TypePropExperimentalDetectronDetectionOutputV6Test, input_mixed_floating_point_type) {
    const auto f32_param = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto f16_param = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());

    OV_EXPECT_THROW(std::ignore = make_op(f32_param, f16_param, f16_param, f16_param, make_attrs(100, 20)),
                    NodeValidationFailure,
                    HasSubstr("Input[1] type 'f16' is not floating point or not same as others inputs"));

    OV_EXPECT_THROW(std::ignore = make_op(f16_param, f32_param, f16_param, f16_param, make_attrs(100, 20)),
                    NodeValidationFailure,
                    HasSubstr("Input[1] type 'f32' is not floating point or not same as others inputs"));

    OV_EXPECT_THROW(std::ignore = make_op(f16_param, f16_param, f32_param, f16_param, make_attrs(100, 20)),
                    NodeValidationFailure,
                    HasSubstr("Input[2] type 'f32' is not floating point or not same as others inputs"));

    OV_EXPECT_THROW(std::ignore = make_op(f16_param, f16_param, f16_param, f32_param, make_attrs(100, 20)),
                    NodeValidationFailure,
                    HasSubstr("Input[3] type 'f32' is not floating point or not same as others inputs"));
}

TEST_F(TypePropExperimentalDetectronDetectionOutputV6Test, rois_shape_not_2d) {
    const auto deltas = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto scores = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto im_info = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    OV_EXPECT_THROW(std::ignore = make_op(std::make_shared<Parameter>(element::f32, PartialShape{2}),
                                          deltas,
                                          scores,
                                          im_info,
                                          make_attrs(25, num_classes)),
                    NodeValidationFailure,
                    HasSubstr("Input rois rank must be equal to 2."));

    OV_EXPECT_THROW(std::ignore = make_op(std::make_shared<Parameter>(element::f32, PartialShape{2, 3, 1}),
                                          deltas,
                                          scores,
                                          im_info,
                                          make_attrs(25, num_classes)),
                    NodeValidationFailure,
                    HasSubstr("Input rois rank must be equal to 2."));
}

TEST_F(TypePropExperimentalDetectronDetectionOutputV6Test, inputs_deltas_not_2d) {
    const auto rois = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto scores = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto im_info = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    OV_EXPECT_THROW(std::ignore = make_op(rois,
                                          std::make_shared<Parameter>(element::f32, PartialShape{2}),
                                          scores,
                                          im_info,
                                          make_attrs(25, num_classes)),
                    NodeValidationFailure,
                    HasSubstr("Input deltas rank must be equal to 2."));

    OV_EXPECT_THROW(std::ignore = make_op(rois,
                                          std::make_shared<Parameter>(element::f32, PartialShape{2, 3, 1}),
                                          scores,
                                          im_info,
                                          make_attrs(25, num_classes)),
                    NodeValidationFailure,
                    HasSubstr("Input deltas rank must be equal to 2."));
}

TEST_F(TypePropExperimentalDetectronDetectionOutputV6Test, scores_not_2d) {
    const auto rois = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto deltas = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto im_info = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    OV_EXPECT_THROW(std::ignore = make_op(rois,
                                          deltas,
                                          std::make_shared<Parameter>(element::f32, PartialShape{2}),
                                          im_info,
                                          make_attrs(25, num_classes)),
                    NodeValidationFailure,
                    HasSubstr("Input scores rank must be equal to 2."));

    OV_EXPECT_THROW(std::ignore = make_op(rois,
                                          deltas,
                                          std::make_shared<Parameter>(element::f32, PartialShape{2, 3, 1}),
                                          im_info,
                                          make_attrs(25, num_classes)),
                    NodeValidationFailure,
                    HasSubstr("Input scores rank must be equal to 2."));
}

TEST_F(TypePropExperimentalDetectronDetectionOutputV6Test, im_infos_not_2d) {
    const auto rois = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto deltas = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto scores = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    OV_EXPECT_THROW(std::ignore = make_op(rois,
                                          deltas,
                                          scores,
                                          std::make_shared<Parameter>(element::f32, PartialShape{2}),
                                          make_attrs(25, num_classes)),
                    NodeValidationFailure,
                    HasSubstr("Input image info shape must be compatible with [1,3]"));

    OV_EXPECT_THROW(std::ignore = make_op(rois,
                                          deltas,
                                          scores,
                                          std::make_shared<Parameter>(element::f32, PartialShape{1, 3, 1}),
                                          make_attrs(25, num_classes)),
                    NodeValidationFailure,
                    HasSubstr("Input image info shape must be compatible with [1,3]"));

    OV_EXPECT_THROW(std::ignore = make_op(rois,
                                          deltas,
                                          scores,
                                          std::make_shared<Parameter>(element::f32, PartialShape{2, 3}),
                                          make_attrs(25, num_classes)),
                    NodeValidationFailure,
                    HasSubstr("Input image info shape must be compatible with [1,3]"));
}

TEST_F(TypePropExperimentalDetectronDetectionOutputV6Test, rois_2nd_dim_not_compatible_4) {
    const auto deltas = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto scores = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto im_info = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    OV_EXPECT_THROW(std::ignore = make_op(std::make_shared<Parameter>(element::f32, PartialShape{-1, {0, 3}}),
                                          deltas,
                                          scores,
                                          im_info,
                                          make_attrs(25, num_classes)),
                    NodeValidationFailure,
                    HasSubstr("The last dimension of the 'input_rois' input must be compatible with 4."));

    OV_EXPECT_THROW(std::ignore = make_op(std::make_shared<Parameter>(element::f32, PartialShape{-1, {5, -1}}),
                                          deltas,
                                          scores,
                                          im_info,
                                          make_attrs(25, num_classes)),
                    NodeValidationFailure,
                    HasSubstr("The last dimension of the 'input_rois' input must be compatible with 4."));
}

TEST_F(TypePropExperimentalDetectronDetectionOutputV6Test, deltas_2nd_dim_not_compatible_with_num_classes) {
    const auto rois = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto scores = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto im_info = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    OV_EXPECT_THROW(
        std::ignore = make_op(rois,
                              std::make_shared<Parameter>(element::f32, PartialShape{-1, {0, 4 * num_classes - 1}}),
                              scores,
                              im_info,
                              make_attrs(25, num_classes)),
        NodeValidationFailure,
        HasSubstr("The last dimension of the 'input_deltas' input be compatible with the value of the attribute "
                  "'num_classes"));

    OV_EXPECT_THROW(
        std::ignore = make_op(rois,
                              std::make_shared<Parameter>(element::f32, PartialShape{-1, {4 * num_classes + 1, -1}}),
                              scores,
                              im_info,
                              make_attrs(25, num_classes)),
        NodeValidationFailure,
        HasSubstr("The last dimension of the 'input_deltas' input be compatible with the value of the attribute "
                  "'num_classes"));
}

TEST_F(TypePropExperimentalDetectronDetectionOutputV6Test, scores_2nd_dim_not_compatible_with_num_classes) {
    const auto rois = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto deltas = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto im_info = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    OV_EXPECT_THROW(
        std::ignore = make_op(rois,
                              deltas,
                              std::make_shared<Parameter>(element::f32, PartialShape{-1, {0, num_classes - 1}}),
                              im_info,
                              make_attrs(25, num_classes)),
        NodeValidationFailure,
        HasSubstr("The last dimension of the 'input_scores' input must be compatible withthe value of the attribute "
                  "'num_classes'"));

    OV_EXPECT_THROW(
        std::ignore = make_op(rois,
                              deltas,
                              std::make_shared<Parameter>(element::f32, PartialShape{-1, {num_classes + 1, -1}}),
                              im_info,
                              make_attrs(25, num_classes)),
        NodeValidationFailure,
        HasSubstr("The last dimension of the 'input_scores' input must be compatible withthe value of the attribute "
                  "'num_classes'"));
}

TEST_F(TypePropExperimentalDetectronDetectionOutputV6Test, 1st_dim_rois_and_scores_compatible_with_num_batches) {
    const auto scores = std::make_shared<Parameter>(element::f32, PartialShape{{5, 10}, -1});
    const auto im_info = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    OV_EXPECT_THROW(
        std::ignore = make_op(std::make_shared<Parameter>(element::f32, PartialShape{{2, 6}, -1}),
                              std::make_shared<Parameter>(element::f32, PartialShape{{9, 15}, -1}),
                              scores,
                              im_info,
                              make_attrs(25, num_classes)),
        NodeValidationFailure,
        HasSubstr("The first dimension of inputs 'input_rois', 'input_deltas', 'input_scores' must be the compatible"));

    OV_EXPECT_THROW(
        std::ignore = make_op(std::make_shared<Parameter>(element::f32, PartialShape{{2, 6}, -1}),
                              std::make_shared<Parameter>(element::f32, PartialShape{{2, 4}, -1}),
                              scores,
                              im_info,
                              make_attrs(25, num_classes)),
        NodeValidationFailure,
        HasSubstr("The first dimension of inputs 'input_rois', 'input_deltas', 'input_scores' must be the compatible"));

    OV_EXPECT_THROW(
        std::ignore = make_op(std::make_shared<Parameter>(element::f32, PartialShape{{11, 12}, -1}),
                              std::make_shared<Parameter>(element::f32, PartialShape{{9, 15}, -1}),
                              scores,
                              im_info,
                              make_attrs(25, num_classes)),
        NodeValidationFailure,
        HasSubstr("The first dimension of inputs 'input_rois', 'input_deltas', 'input_scores' must be the compatible"));
}

using DetectronGenerateProposalsParams = std::tuple<PartialShape, PartialShape, PartialShape, PartialShape, int64_t>;

class ExperimentalDetectronDetectionOutputV6Test : public TypePropExperimentalDetectronDetectionOutputV6Test,
                                                   public WithParamInterface<DetectronGenerateProposalsParams> {
protected:
    void SetUp() override {
        std::tie(rois_shape, deltas_shape, scores_shape, im_info_shape, num_classes) = GetParam();
    }

    PartialShape rois_shape, deltas_shape, scores_shape, im_info_shape;
};

INSTANTIATE_TEST_SUITE_P(
    type_prop,
    ExperimentalDetectronDetectionOutputV6Test,
    Values(DetectronGenerateProposalsParams({2, 4}, {2, -1}, {2, -1}, {1, 3}, 2),
           DetectronGenerateProposalsParams({{{0, 4}, 4}}, {{2, 5}, {1, 15}}, {3, 3}, {1, 3}, 3),
           DetectronGenerateProposalsParams({{{0, 4}, 4}}, {{2, 5}, 12}, {3, 3}, {1, 3}, 3),
           DetectronGenerateProposalsParams({{{0, 4}, 4}}, {{2, 5}, 12}, {3, {1, 3}}, {1, 3}, 3),
           DetectronGenerateProposalsParams(PartialShape::dynamic(), {5, 40}, {5, -1}, {1, -1}, 10),
           DetectronGenerateProposalsParams({5, 4}, PartialShape::dynamic(), {-1, 10}, {-1, -1}, 10),
           DetectronGenerateProposalsParams({5, 4}, {-1, 40}, PartialShape::dynamic(), {-1, 3}, 10),
           DetectronGenerateProposalsParams({5, 4}, {5, {20, 50}}, {-1, 10}, PartialShape::dynamic(), 10),
           DetectronGenerateProposalsParams(PartialShape::dynamic(2),
                                            PartialShape::dynamic(2),
                                            PartialShape::dynamic(2),
                                            PartialShape::dynamic(2),
                                            100)),
    PrintToStringParamName());

TEST_P(ExperimentalDetectronDetectionOutputV6Test, static_rank_shape_inference) {
    const auto rois = std::make_shared<Parameter>(element::bf16, rois_shape);
    const auto deltas = std::make_shared<Parameter>(element::bf16, deltas_shape);
    const auto scores = std::make_shared<Parameter>(element::bf16, scores_shape);
    const auto im_info = std::make_shared<Parameter>(element::bf16, im_info_shape);

    const auto op = make_op(rois, deltas, scores, im_info, make_attrs(25, num_classes));

    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Boxes type", &Output<Node>::get_element_type, element::bf16),
                            Property("Classes type", &Output<Node>::get_element_type, element::i32),
                            Property("Scores types", &Output<Node>::get_element_type, element::bf16)));
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Boxes shape",
                                     &Output<Node>::get_partial_shape,
                                     AllOf(PartialShape({25, 4}), ResultOf(get_shape_symbols, Each(nullptr)))),
                            Property("Classes shape",
                                     &Output<Node>::get_partial_shape,
                                     AllOf(PartialShape({25}), ResultOf(get_shape_symbols, Each(nullptr)))),
                            Property("Scores shape",
                                     &Output<Node>::get_partial_shape,
                                     AllOf(PartialShape({25}), ResultOf(get_shape_symbols, Each(nullptr))))));
}
