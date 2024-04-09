// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include <vector>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/opsets/opset11.hpp"

using namespace ov;
using namespace ov::opset11;
using namespace testing;

class TypePropExperimentalDetectronGenerateProposalsSingleImageV6Test
    : public TypePropOpTest<op::v6::ExperimentalDetectronGenerateProposalsSingleImage> {
protected:
    using Attrs = op::v6::ExperimentalDetectronGenerateProposalsSingleImage::Attributes;

    static Attrs make_attrs(int64_t post_nms_count) {
        return {0.0f, 0.0f, post_nms_count, 0};
    }
};

TEST_F(TypePropExperimentalDetectronGenerateProposalsSingleImageV6Test, default_ctor) {
    const auto im_info = std::make_shared<Parameter>(element::f32, Shape{3});
    const auto anchors = std::make_shared<Parameter>(element::f32, Shape{201600, 4});
    const auto deltas = std::make_shared<Parameter>(element::f32, Shape{12, 200, 336});
    const auto scores = std::make_shared<Parameter>(element::f32, Shape{3, 200, 336});

    const auto op = make_op();
    op->set_arguments(OutputVector{im_info, anchors, deltas, scores});
    op->set_attrs({1.1f, 20.3f, 20, 0});
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_input_size(), 4);
    EXPECT_EQ(op->get_output_size(), 2);
    EXPECT_FLOAT_EQ(op->get_attrs().min_size, 1.1f);
    EXPECT_FLOAT_EQ(op->get_attrs().nms_threshold, 20.3f);
    EXPECT_EQ(op->get_attrs().post_nms_count, 20);
    EXPECT_EQ(op->get_attrs().pre_nms_count, 0);
    EXPECT_THAT(op->outputs(), Each(Property("Element type", &Output<Node>::get_element_type, element::f32)));
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("ROIs shape", &Output<Node>::get_shape, Shape({20, 4})),
                            Property("ROIs Score shape", &Output<Node>::get_shape, Shape({20}))));
}

TEST_F(TypePropExperimentalDetectronGenerateProposalsSingleImageV6Test, basic_shape_inference) {
    const auto im_info = std::make_shared<Parameter>(element::f64, Shape{3});
    const auto anchors = std::make_shared<Parameter>(element::f64, Shape{201600, 4});
    const auto deltas = std::make_shared<Parameter>(element::f64, Shape{12, 200, 336});
    const auto scores = std::make_shared<Parameter>(element::f64, Shape{3, 200, 336});

    const auto op = make_op(im_info, anchors, deltas, scores, make_attrs(132));

    EXPECT_THAT(op->outputs(), Each(Property("Element type", &Output<Node>::get_element_type, element::f64)));
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("ROIs shape", &Output<Node>::get_shape, Shape({132, 4})),
                            Property("ROIs Score shape", &Output<Node>::get_shape, Shape({132}))));
}

TEST_F(TypePropExperimentalDetectronGenerateProposalsSingleImageV6Test, all_input_got_dynamic_type) {
    const auto im_info = std::make_shared<Parameter>(element::dynamic, Shape{3});
    const auto anchors = std::make_shared<Parameter>(element::dynamic, Shape{201600, 4});
    const auto deltas = std::make_shared<Parameter>(element::dynamic, Shape{12, 200, 336});
    const auto scores = std::make_shared<Parameter>(element::dynamic, Shape{3, 200, 336});

    const auto op = make_op(im_info, anchors, deltas, scores, make_attrs(132));

    EXPECT_THAT(op->outputs(), Each(Property("Element type", &Output<Node>::get_element_type, element::dynamic)));
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("ROIs shape", &Output<Node>::get_shape, Shape({132, 4})),
                            Property("ROIs Score shape", &Output<Node>::get_shape, Shape({132}))));
}

TEST_F(TypePropExperimentalDetectronGenerateProposalsSingleImageV6Test, some_input_got_dynamic_type) {
    const auto im_info = std::make_shared<Parameter>(element::dynamic, Shape{3});
    const auto anchors = std::make_shared<Parameter>(element::f64, Shape{201600, 4});
    const auto deltas = std::make_shared<Parameter>(element::f64, Shape{12, 200, 336});
    const auto scores = std::make_shared<Parameter>(element::dynamic, Shape{3, 200, 336});

    const auto op = make_op(im_info, anchors, deltas, scores, make_attrs(132));

    EXPECT_THAT(op->outputs(), Each(Property("Element type", &Output<Node>::get_element_type, element::f64)));
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("ROIs shape", &Output<Node>::get_shape, Shape({132, 4})),
                            Property("ROIs Score shape", &Output<Node>::get_shape, Shape({132}))));
}

TEST_F(TypePropExperimentalDetectronGenerateProposalsSingleImageV6Test, interval_shapes) {
    auto im_info_shape = PartialShape{{2, 4}};
    auto anchors_shape = PartialShape{{1, 201600}, {1, 4}};
    auto deltas_shape = PartialShape{{1, 12}, {1, 200}, {336, -1}};
    auto scores_shape = PartialShape{{1, 3}, {100, 200}, {2, 336}};
    set_shape_symbols(im_info_shape);
    set_shape_symbols(anchors_shape);
    set_shape_symbols(deltas_shape);
    set_shape_symbols(scores_shape);

    const auto im_info = std::make_shared<Parameter>(element::f16, im_info_shape);
    const auto anchors = std::make_shared<Parameter>(element::f16, anchors_shape);
    const auto deltas = std::make_shared<Parameter>(element::f16, deltas_shape);
    const auto scores = std::make_shared<Parameter>(element::f16, scores_shape);

    const auto op = make_op(im_info, anchors, deltas, scores, make_attrs(44));

    EXPECT_THAT(op->outputs(), Each(Property("Element type", &Output<Node>::get_element_type, element::f16)));
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("ROIs shape",
                                     &Output<Node>::get_partial_shape,
                                     AllOf(PartialShape({44, 4}), ResultOf(get_shape_symbols, Each(nullptr)))),
                            Property("ROIs Score shape",
                                     &Output<Node>::get_partial_shape,
                                     AllOf(PartialShape({44}), ResultOf(get_shape_symbols, Each(nullptr))))));
}

TEST_F(TypePropExperimentalDetectronGenerateProposalsSingleImageV6Test, all_inputs_dynamic_rank) {
    const auto im_info = std::make_shared<Parameter>(element::bf16, PartialShape::dynamic());
    const auto anchors = std::make_shared<Parameter>(element::bf16, PartialShape::dynamic());
    const auto deltas = std::make_shared<Parameter>(element::bf16, PartialShape::dynamic());
    const auto scores = std::make_shared<Parameter>(element::bf16, PartialShape::dynamic());

    const auto op = make_op(im_info, anchors, deltas, scores, make_attrs(100));

    EXPECT_THAT(op->outputs(), Each(Property("Element type", &Output<Node>::get_element_type, element::bf16)));
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("ROIs shape",
                                     &Output<Node>::get_partial_shape,
                                     AllOf(PartialShape({100, 4}), ResultOf(get_shape_symbols, Each(nullptr)))),
                            Property("ROIs Score shape",
                                     &Output<Node>::get_partial_shape,
                                     AllOf(PartialShape({100}), ResultOf(get_shape_symbols, Each(nullptr))))));
}

TEST_F(TypePropExperimentalDetectronGenerateProposalsSingleImageV6Test, input_not_floating_point) {
    const auto bad_param = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto ok_param = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());

    OV_EXPECT_THROW(auto op = make_op(bad_param, ok_param, ok_param, ok_param, make_attrs(100)),
                    NodeValidationFailure,
                    HasSubstr("Input[0] type 'i32' is not floating point or not same as others inputs"));

    OV_EXPECT_THROW(auto op = make_op(ok_param, bad_param, ok_param, ok_param, make_attrs(100)),
                    NodeValidationFailure,
                    HasSubstr("Input[1] type 'i32' is not floating point or not same as others inputs"));

    OV_EXPECT_THROW(auto op = make_op(ok_param, ok_param, bad_param, ok_param, make_attrs(100)),
                    NodeValidationFailure,
                    HasSubstr("Input[2] type 'i32' is not floating point or not same as others inputs"));

    OV_EXPECT_THROW(auto op = make_op(ok_param, ok_param, ok_param, bad_param, make_attrs(100)),
                    NodeValidationFailure,
                    HasSubstr("Input[3] type 'i32' is not floating point or not same as others inputs"));
}

TEST_F(TypePropExperimentalDetectronGenerateProposalsSingleImageV6Test, input_mixed_floating_point_type) {
    const auto f32_param = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto f16_param = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());

    OV_EXPECT_THROW(auto op = make_op(f32_param, f16_param, f16_param, f16_param, make_attrs(100)),
                    NodeValidationFailure,
                    HasSubstr("Input[1] type 'f16' is not floating point or not same as others inputs"));

    OV_EXPECT_THROW(auto op = make_op(f16_param, f32_param, f16_param, f16_param, make_attrs(100)),
                    NodeValidationFailure,
                    HasSubstr("Input[1] type 'f32' is not floating point or not same as others inputs"));

    OV_EXPECT_THROW(auto op = make_op(f16_param, f16_param, f32_param, f16_param, make_attrs(100)),
                    NodeValidationFailure,
                    HasSubstr("Input[2] type 'f32' is not floating point or not same as others inputs"));

    OV_EXPECT_THROW(auto op = make_op(f16_param, f16_param, f16_param, f32_param, make_attrs(100)),
                    NodeValidationFailure,
                    HasSubstr("Input[3] type 'f32' is not floating point or not same as others inputs"));
}

TEST_F(TypePropExperimentalDetectronGenerateProposalsSingleImageV6Test, im_infos_not_1d) {
    const auto anchors = std::make_shared<Parameter>(element::f32, PartialShape{10, 5});
    const auto deltas = std::make_shared<Parameter>(element::f32, PartialShape{5, 50, 20});
    const auto scores = std::make_shared<Parameter>(element::f32, PartialShape{15, 50, 20});

    OV_EXPECT_THROW(auto op = make_op(std::make_shared<Parameter>(element::f32, PartialShape{}),
                                      anchors,
                                      deltas,
                                      scores,
                                      make_attrs(100)),
                    NodeValidationFailure,
                    HasSubstr("The 'input_im_info' input is expected to be a 1D"));

    OV_EXPECT_THROW(auto op = make_op(std::make_shared<Parameter>(element::f32, PartialShape{2, 3}),
                                      anchors,
                                      deltas,
                                      scores,
                                      make_attrs(100)),
                    NodeValidationFailure,
                    HasSubstr("The 'input_im_info' input is expected to be a 1D"));
}

TEST_F(TypePropExperimentalDetectronGenerateProposalsSingleImageV6Test, anchors_not_2d) {
    const auto im_info = std::make_shared<Parameter>(element::f32, PartialShape{3});
    const auto deltas = std::make_shared<Parameter>(element::f32, PartialShape{5, 50, 20});
    const auto scores = std::make_shared<Parameter>(element::f32, PartialShape{15, 50, 20});

    OV_EXPECT_THROW(auto op = make_op(im_info,
                                      std::make_shared<Parameter>(element::f32, PartialShape{1}),
                                      deltas,
                                      scores,
                                      make_attrs(100)),
                    NodeValidationFailure,
                    HasSubstr("The 'input_anchors' input is expected to be a 2D"));

    OV_EXPECT_THROW(auto op = make_op(im_info,
                                      std::make_shared<Parameter>(element::f32, PartialShape{2, 3, 1}),
                                      deltas,
                                      scores,
                                      make_attrs(100)),
                    NodeValidationFailure,
                    HasSubstr("The 'input_anchors' input is expected to be a 2D"));
}

TEST_F(TypePropExperimentalDetectronGenerateProposalsSingleImageV6Test, deltas_not_3d) {
    const auto im_info = std::make_shared<Parameter>(element::f32, PartialShape{3});
    const auto anchors = std::make_shared<Parameter>(element::f32, PartialShape{5, 4});
    const auto scores = std::make_shared<Parameter>(element::f32, PartialShape{15, 50, 20});

    OV_EXPECT_THROW(auto op = make_op(im_info,
                                      anchors,
                                      std::make_shared<Parameter>(element::f32, PartialShape{1}),
                                      scores,
                                      make_attrs(100)),
                    NodeValidationFailure,
                    HasSubstr("The 'input_deltas' input is expected to be a 3D"));

    OV_EXPECT_THROW(auto op = make_op(im_info,
                                      anchors,
                                      std::make_shared<Parameter>(element::f32, PartialShape{2, 50, 20, 3}),
                                      scores,
                                      make_attrs(100)),
                    NodeValidationFailure,
                    HasSubstr("The 'input_deltas' input is expected to be a 3D"));
}

TEST_F(TypePropExperimentalDetectronGenerateProposalsSingleImageV6Test, scores_not_3d) {
    const auto im_info = std::make_shared<Parameter>(element::f32, PartialShape{3});
    const auto anchors = std::make_shared<Parameter>(element::f32, PartialShape{5, 4});
    const auto deltas = std::make_shared<Parameter>(element::f32, PartialShape{15, 50, 20});

    OV_EXPECT_THROW(auto op = make_op(im_info,
                                      anchors,
                                      deltas,
                                      std::make_shared<Parameter>(element::f32, PartialShape{1}),
                                      make_attrs(100)),
                    NodeValidationFailure,
                    HasSubstr("The 'input_scores' input is expected to be a 3D"));

    OV_EXPECT_THROW(auto op = make_op(im_info,
                                      anchors,
                                      deltas,
                                      std::make_shared<Parameter>(element::f32, PartialShape{15, 50, 20, 11}),
                                      make_attrs(100)),
                    NodeValidationFailure,
                    HasSubstr("The 'input_scores' input is expected to be a 3D"));
}

TEST_F(TypePropExperimentalDetectronGenerateProposalsSingleImageV6Test, im_info_dim0_not_compatible) {
    const auto anchors = std::make_shared<Parameter>(element::f32, PartialShape{5, 4});
    const auto deltas = std::make_shared<Parameter>(element::f32, PartialShape{15, 50, 20});
    const auto scores = std::make_shared<Parameter>(element::f32, PartialShape{15, 50, 20});

    OV_EXPECT_THROW(auto op = make_op(std::make_shared<Parameter>(element::f32, PartialShape{{0, 2}}),
                                      anchors,
                                      deltas,
                                      scores,
                                      make_attrs(100)),
                    NodeValidationFailure,
                    HasSubstr("The 'input_im_info' shape is expected to be a compatible with [3]"));

    OV_EXPECT_THROW(auto op = make_op(std::make_shared<Parameter>(element::f32, PartialShape{{4, -1}}),
                                      anchors,
                                      deltas,
                                      scores,
                                      make_attrs(100)),
                    NodeValidationFailure,
                    HasSubstr("The 'input_im_info' shape is expected to be a compatible with [3]"));
}

TEST_F(TypePropExperimentalDetectronGenerateProposalsSingleImageV6Test, anchors_dim1_not_compatible) {
    const auto im_info = std::make_shared<Parameter>(element::f32, PartialShape{3});
    const auto deltas = std::make_shared<Parameter>(element::f32, PartialShape{15, 50, 20});
    const auto scores = std::make_shared<Parameter>(element::f32, PartialShape{15, 50, 20});

    OV_EXPECT_THROW(auto op = make_op(im_info,
                                      std::make_shared<Parameter>(element::f32, PartialShape{10, {0, 3}}),
                                      deltas,
                                      scores,
                                      make_attrs(100)),
                    NodeValidationFailure,
                    HasSubstr("The second dimension of 'input_anchors' should be compatible with 4"));

    OV_EXPECT_THROW(auto op = make_op(im_info,
                                      std::make_shared<Parameter>(element::f32, PartialShape{10, {5, -1}}),
                                      deltas,
                                      scores,
                                      make_attrs(100)),
                    NodeValidationFailure,
                    HasSubstr("The second dimension of 'input_anchors' should be compatible with 4"));
}

TEST_F(TypePropExperimentalDetectronGenerateProposalsSingleImageV6Test, heights_deltas_and_scores_not_compatible) {
    const auto im_info = std::make_shared<Parameter>(element::f32, PartialShape{3});
    const auto anchors = std::make_shared<Parameter>(element::f32, PartialShape{5, 4});
    const auto deltas = std::make_shared<Parameter>(element::f32, PartialShape{15, {0, 50}, 20});
    const auto scores = std::make_shared<Parameter>(element::f32, PartialShape{15, {51, -1}, 20});

    OV_EXPECT_THROW(auto op = make_op(im_info, anchors, deltas, scores, make_attrs(100)),
                    NodeValidationFailure,
                    HasSubstr("Heights for inputs 'input_deltas' and 'input_scores' should be equal"));
}

TEST_F(TypePropExperimentalDetectronGenerateProposalsSingleImageV6Test, widths_deltas_and_scores_not_compatible) {
    const auto im_info = std::make_shared<Parameter>(element::f32, PartialShape{3});
    const auto anchors = std::make_shared<Parameter>(element::f32, PartialShape{5, 4});
    const auto deltas = std::make_shared<Parameter>(element::f32, PartialShape{15, {0, 100}, {0, 20}});
    const auto scores = std::make_shared<Parameter>(element::f32, PartialShape{15, {51, 60}, {21, -1}});

    OV_EXPECT_THROW(auto op = make_op(im_info, anchors, deltas, scores, make_attrs(100)),
                    NodeValidationFailure,
                    HasSubstr("Width for inputs 'input_deltas' and 'input_scores' should be equal"));
}

using DetectronGenerateProposalsParams = std::tuple<PartialShape, PartialShape, PartialShape, PartialShape, int64_t>;

class ExperimentalDetectronGenerateProposalsSingleImageV6Test
    : public TypePropExperimentalDetectronGenerateProposalsSingleImageV6Test,
      public WithParamInterface<DetectronGenerateProposalsParams> {
protected:
    void SetUp() override {
        std::tie(im_info_shape, anchors_shape, deltas_shape, scores_shape, post_nms_count) = GetParam();
    }

    PartialShape im_info_shape, anchors_shape, deltas_shape, scores_shape;
    int64_t post_nms_count;
};

INSTANTIATE_TEST_SUITE_P(
    type_prop,
    ExperimentalDetectronGenerateProposalsSingleImageV6Test,
    Values(DetectronGenerateProposalsParams({3}, {2, -1}, {-1, 10, 20}, {10, 10, 20}, 10),
           DetectronGenerateProposalsParams({{1, 4}}, {-1, {4, 5}}, {2, -1, 20}, {10, 10, 20}, 20),
           DetectronGenerateProposalsParams({-1}, {{100, 200}, 4}, {2, {0, 20}, -1}, {{10, 12}, 10, 20}, 30),
           DetectronGenerateProposalsParams({3}, {10, 4}, {2, {0, 20}, 100}, {{10, 12}, {10, 20}, {20, -1}}, 40),
           DetectronGenerateProposalsParams(PartialShape::dynamic(), {100, 4}, {5, 33, 55}, {21, 33, 55}, 100),
           DetectronGenerateProposalsParams({3}, PartialShape::dynamic(), {2, 33, 55}, {5, 33, 55}, 200),
           DetectronGenerateProposalsParams({3}, {100, 4}, PartialShape::dynamic(), {5, 33, 55}, 300),
           DetectronGenerateProposalsParams({3}, {100, 4}, PartialShape::dynamic(), {5, 33, 55}, 400),
           DetectronGenerateProposalsParams({3}, {100, 4}, {5, 33, 55}, PartialShape::dynamic(), 500),
           DetectronGenerateProposalsParams(PartialShape::dynamic(1),
                                            PartialShape::dynamic(2),
                                            PartialShape::dynamic(3),
                                            PartialShape::dynamic(3),
                                            100)),
    PrintToStringParamName());

TEST_P(ExperimentalDetectronGenerateProposalsSingleImageV6Test, static_rank_shape_inference) {
    const auto im_info = std::make_shared<Parameter>(element::f32, im_info_shape);
    const auto anchors = std::make_shared<Parameter>(element::f32, anchors_shape);
    const auto deltas = std::make_shared<Parameter>(element::f32, deltas_shape);
    const auto scores = std::make_shared<Parameter>(element::f32, scores_shape);

    const auto op = make_op(im_info, anchors, deltas, scores, make_attrs(post_nms_count));

    EXPECT_THAT(op->outputs(), Each(Property("Element type", &Output<Node>::get_element_type, element::f32)));
    EXPECT_THAT(
        op->outputs(),
        ElementsAre(Property("ROIs shape", &Output<Node>::get_partial_shape, PartialShape({post_nms_count, 4})),
                    Property("ROIs Score shape", &Output<Node>::get_partial_shape, PartialShape({post_nms_count}))));
}
