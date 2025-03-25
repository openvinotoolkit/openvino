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

class TypePropExperimentalDetectronPriorGridGeneratorV6Test
    : public TypePropOpTest<op::v6::ExperimentalDetectronPriorGridGenerator> {
protected:
    using Attrs = op::v6::ExperimentalDetectronPriorGridGenerator::Attributes;

    static Attrs make_attrs(bool flatten) {
        return {flatten, 0, 0, 4.0f, 4.0f};
    };
};

TEST_F(TypePropExperimentalDetectronPriorGridGeneratorV6Test, default_ctor_no_flatten) {
    const auto priors = std::make_shared<Parameter>(element::f32, Shape{3, 4});
    const auto feature_map = std::make_shared<Parameter>(element::f32, Shape{1, 3, 200, 336});
    const auto im_data = std::make_shared<Parameter>(element::f32, Shape{1, 3, 800, 1344});

    const auto op = make_op();
    op->set_arguments(OutputVector{priors, feature_map, im_data});
    op->set_attrs(make_attrs(false));
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({200, 336, 3, 4}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
}

TEST_F(TypePropExperimentalDetectronPriorGridGeneratorV6Test, static_shape_flatten) {
    auto priors_shape = PartialShape{3, 4};
    auto feat_map_shape = PartialShape{1, 4, 6, 10};
    auto im_data_shape = PartialShape{1, 4, 128, 128};
    set_shape_symbols(priors_shape);
    set_shape_symbols(feat_map_shape);
    set_shape_symbols(im_data_shape);

    const auto priors = std::make_shared<Parameter>(element::f64, priors_shape);
    const auto feature_map = std::make_shared<Parameter>(element::f64, feat_map_shape);
    const auto im_data = std::make_shared<Parameter>(element::f64, im_data_shape);

    const auto op = make_op(priors, feature_map, im_data, make_attrs(true));

    EXPECT_EQ(op->get_output_element_type(0), element::f64);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({180, 4}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
}

TEST_F(TypePropExperimentalDetectronPriorGridGeneratorV6Test, static_shape_without_flatten) {
    auto priors_shape = PartialShape{3, 4};
    auto feat_map_shape = PartialShape{1, 4, 6, 10};
    auto im_data_shape = PartialShape{1, 4, 128, 128};
    auto priors_symbols = set_shape_symbols(priors_shape);
    auto feat_symbols = set_shape_symbols(feat_map_shape);
    set_shape_symbols(im_data_shape);

    const auto priors = std::make_shared<Parameter>(element::f16, priors_shape);
    const auto feature_map = std::make_shared<Parameter>(element::f16, feat_map_shape);
    const auto im_data = std::make_shared<Parameter>(element::f16, im_data_shape);

    const auto op = make_op(priors, feature_map, im_data, make_attrs(false));

    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({6, 10, 3, 4}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(feat_symbols[2], feat_symbols[3], priors_symbols[0], nullptr));
}

TEST_F(TypePropExperimentalDetectronPriorGridGeneratorV6Test, interval_shapes_flatten) {
    auto priors_shape = PartialShape{{2, 5}, {0, 4}};
    auto feat_map_shape = PartialShape{1, {3, 4}, {5, 10}, {9, 10}};
    auto im_data_shape = PartialShape{1, {2, 4}, {1, 128}, {1, 128}};
    set_shape_symbols(priors_shape);
    set_shape_symbols(feat_map_shape);
    set_shape_symbols(im_data_shape);

    const auto priors = std::make_shared<Parameter>(element::bf16, priors_shape);
    const auto feature_map = std::make_shared<Parameter>(element::bf16, feat_map_shape);
    const auto im_data = std::make_shared<Parameter>(element::bf16, im_data_shape);

    const auto op = make_op(priors, feature_map, im_data, make_attrs(true));

    EXPECT_EQ(op->get_output_element_type(0), element::bf16);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({{90, 500}, 4}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
}

TEST_F(TypePropExperimentalDetectronPriorGridGeneratorV6Test, interval_shapes_no_flatten) {
    auto priors_shape = PartialShape{{2, 5}, {0, 4}};
    auto feat_map_shape = PartialShape{1, {3, 4}, {5, 10}, {9, 10}};
    auto im_data_shape = PartialShape{1, {2, 4}, {1, 128}, {1, 128}};
    auto priors_symbols = set_shape_symbols(priors_shape);
    auto feat_symbols = set_shape_symbols(feat_map_shape);
    set_shape_symbols(im_data_shape);

    const auto priors = std::make_shared<Parameter>(element::bf16, priors_shape);
    const auto feature_map = std::make_shared<Parameter>(element::bf16, feat_map_shape);
    const auto im_data = std::make_shared<Parameter>(element::bf16, im_data_shape);

    const auto op = make_op(priors, feature_map, im_data, make_attrs(false));

    EXPECT_EQ(op->get_output_element_type(0), element::bf16);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({{5, 10}, {9, 10}, {2, 5}, 4}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(feat_symbols[2], feat_symbols[3], priors_symbols[0], nullptr));
}

TEST_F(TypePropExperimentalDetectronPriorGridGeneratorV6Test, all_inputs_dynamic_rank_flatten) {
    const auto priors = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());
    const auto feature_map = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());
    const auto im_data = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());

    const auto op = make_op(priors, feature_map, im_data, make_attrs(true));

    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({-1, 4}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
}

TEST_F(TypePropExperimentalDetectronPriorGridGeneratorV6Test, all_inputs_dynamic_rank_no_flatten) {
    const auto priors = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());
    const auto feature_map = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());
    const auto im_data = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());

    const auto op = make_op(priors, feature_map, im_data, make_attrs(false));

    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({-1, -1, -1, 4}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
}

TEST_F(TypePropExperimentalDetectronPriorGridGeneratorV6Test, all_input_got_dynamic_type) {
    const auto priors = std::make_shared<Parameter>(element::dynamic, Shape{2, 4});
    const auto feature_map = std::make_shared<Parameter>(element::dynamic, Shape{1, 4, 5, 5});
    const auto im_data = std::make_shared<Parameter>(element::dynamic, Shape{1, 4, 500, 500});

    const auto op = make_op(priors, feature_map, im_data, make_attrs(false));

    EXPECT_EQ(op->get_output_element_type(0), element::dynamic);
    EXPECT_EQ(op->get_output_shape(0), Shape({5, 5, 2, 4}));
}

TEST_F(TypePropExperimentalDetectronPriorGridGeneratorV6Test, some_input_got_dynamic_type) {
    const auto priors = std::make_shared<Parameter>(element::dynamic, Shape{2, 4});
    const auto feature_map = std::make_shared<Parameter>(element::f32, Shape{1, 4, 5, 5});
    const auto im_data = std::make_shared<Parameter>(element::dynamic, Shape{1, 4, 500, 500});

    const auto op = make_op(priors, feature_map, im_data, make_attrs(false));

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_shape(0), Shape({5, 5, 2, 4}));
}

TEST_F(TypePropExperimentalDetectronPriorGridGeneratorV6Test, input_not_floating_point) {
    const auto bad_param = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto ok_param = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());

    OV_EXPECT_THROW(std::ignore = make_op(bad_param, ok_param, ok_param, make_attrs(false)),
                    NodeValidationFailure,
                    HasSubstr("Input[0] type 'i32' is not floating point or not same as others inputs"));

    OV_EXPECT_THROW(std::ignore = make_op(ok_param, bad_param, ok_param, make_attrs(false)),
                    NodeValidationFailure,
                    HasSubstr("Input[1] type 'i32' is not floating point or not same as others inputs"));

    OV_EXPECT_THROW(std::ignore = make_op(ok_param, ok_param, bad_param, make_attrs(true)),
                    NodeValidationFailure,
                    HasSubstr("Input[2] type 'i32' is not floating point or not same as others inputs"));
}

TEST_F(TypePropExperimentalDetectronPriorGridGeneratorV6Test, input_mixed_floating_point_type) {
    const auto f32_param = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto f16_param = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());

    OV_EXPECT_THROW(std::ignore = make_op(f32_param, f16_param, f16_param, make_attrs(true)),
                    NodeValidationFailure,
                    HasSubstr("Input[1] type 'f16' is not floating point or not same as others inputs"));

    OV_EXPECT_THROW(std::ignore = make_op(f16_param, f32_param, f16_param, make_attrs(true)),
                    NodeValidationFailure,
                    HasSubstr("Input[1] type 'f32' is not floating point or not same as others inputs"));

    OV_EXPECT_THROW(std::ignore = make_op(f16_param, f16_param, f32_param, make_attrs(true)),
                    NodeValidationFailure,
                    HasSubstr("Input[2] type 'f32' is not floating point or not same as others inputs"));
}

TEST_F(TypePropExperimentalDetectronPriorGridGeneratorV6Test, priors_not_2d) {
    const auto feature_map = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto im_data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    OV_EXPECT_THROW(std::ignore = make_op(std::make_shared<Parameter>(element::f32, PartialShape{2}),
                                          feature_map,
                                          im_data,
                                          make_attrs(false)),
                    NodeValidationFailure,
                    HasSubstr("Priors rank must be equal to 2."));

    OV_EXPECT_THROW(std::ignore = make_op(std::make_shared<Parameter>(element::f32, PartialShape{2, 3, 1}),
                                          feature_map,
                                          im_data,
                                          make_attrs(false)),
                    NodeValidationFailure,
                    HasSubstr("Priors rank must be equal to 2."));
}

TEST_F(TypePropExperimentalDetectronPriorGridGeneratorV6Test, feature_map_not_4d) {
    const auto priors = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto im_data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    OV_EXPECT_THROW(std::ignore = make_op(priors,
                                          std::make_shared<Parameter>(element::f32, PartialShape::dynamic(2)),
                                          im_data,
                                          make_attrs(false)),
                    NodeValidationFailure,
                    HasSubstr("Feature_map rank must be equal to 4"));

    OV_EXPECT_THROW(std::ignore = make_op(priors,
                                          std::make_shared<Parameter>(element::f32, PartialShape::dynamic(5)),
                                          im_data,
                                          make_attrs(false)),
                    NodeValidationFailure,
                    HasSubstr("Feature_map rank must be equal to 4"));
}

TEST_F(TypePropExperimentalDetectronPriorGridGeneratorV6Test, im_data_not_4d) {
    const auto priors = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto feature_map = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    OV_EXPECT_THROW(std::ignore = make_op(priors,
                                          feature_map,
                                          std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3)),
                                          make_attrs(false)),
                    NodeValidationFailure,
                    HasSubstr("Im_data rank must be equal to 4"));

    OV_EXPECT_THROW(std::ignore = make_op(priors,
                                          feature_map,
                                          std::make_shared<Parameter>(element::f32, PartialShape::dynamic(15)),
                                          make_attrs(false)),
                    NodeValidationFailure,
                    HasSubstr("Im_data rank must be equal to 4"));
}

TEST_F(TypePropExperimentalDetectronPriorGridGeneratorV6Test, priors_2nd_dim_not_compatible) {
    const auto feature_map = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto im_data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    OV_EXPECT_THROW(std::ignore = make_op(std::make_shared<Parameter>(element::f32, PartialShape{-1, {0, 3}}),
                                          feature_map,
                                          im_data,
                                          make_attrs(false)),
                    NodeValidationFailure,
                    HasSubstr("The last dimension of the 'priors' input must be equal to 4"));

    OV_EXPECT_THROW(std::ignore = make_op(std::make_shared<Parameter>(element::f32, PartialShape{-1, {5, -1}}),
                                          feature_map,
                                          im_data,
                                          make_attrs(false)),
                    NodeValidationFailure,
                    HasSubstr("The last dimension of the 'priors' input must be equal to 4"));
}

TEST_F(TypePropExperimentalDetectronPriorGridGeneratorV6Test, not_compatible_1st_dim_of_feature_map_and_im_data) {
    const auto priors = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    OV_EXPECT_THROW(
        std::ignore = make_op(priors,
                              std::make_shared<Parameter>(element::f32, PartialShape{3, {0, 3}, -1, -1}),
                              std::make_shared<Parameter>(element::f32, PartialShape{{0, 2}, {4, 0}, -1, -1}),
                              make_attrs(false)),
        NodeValidationFailure,
        HasSubstr("The first dimension of both 'feature_map' and 'im_data' must match"));

    OV_EXPECT_THROW(
        std::ignore = make_op(priors,
                              std::make_shared<Parameter>(element::f32, PartialShape{{0, 1}, 4, -1, -1}),
                              std::make_shared<Parameter>(element::f32, PartialShape{{2, -1}, {2, 10}, -1, -1}),
                              make_attrs(false)),
        NodeValidationFailure,
        HasSubstr("The first dimension of both 'feature_map' and 'im_data' must match"));
}

using DetectronPriorGridGenerator = std::tuple<PartialShape, PartialShape, PartialShape, PartialShape, bool>;

class ExperimentalDetectronPriorGridGeneratorV6Test : public TypePropExperimentalDetectronPriorGridGeneratorV6Test,
                                                      public WithParamInterface<DetectronPriorGridGenerator> {
protected:
    void SetUp() override {
        std::tie(priors_shape, feat_map_shape, im_data_shape, exp_shape, is_flatten) = GetParam();
    }

    PartialShape priors_shape, feat_map_shape, im_data_shape, exp_shape;
    bool is_flatten;
};

INSTANTIATE_TEST_SUITE_P(
    type_prop,
    ExperimentalDetectronPriorGridGeneratorV6Test,
    Values(DetectronPriorGridGenerator(PartialShape::dynamic(), {1, -1, 5, 3}, {1, 3, 100, 100}, {5, 3, -1, 4}, false),
           DetectronPriorGridGenerator(PartialShape::dynamic(2), {1, -1, 5, 3}, {1, 3, 100, 100}, {5, 3, -1, 4}, false),
           DetectronPriorGridGenerator({2, 4}, {1, -1, 5, 3}, {1, 3, 100, 100}, {5, 3, 2, 4}, false),
           DetectronPriorGridGenerator({{2, 4}, {2, 5}}, {1, 3, -1, 3}, {1, -1, 100, 100}, {-1, 3, {2, 4}, 4}, false),
           DetectronPriorGridGenerator({2, 4}, PartialShape::dynamic(), {1, 5, -1, -1}, {-1, -1, 2, 4}, false),
           DetectronPriorGridGenerator({2, 4}, PartialShape::dynamic(4), {1, 5, -1, -1}, {-1, -1, 2, 4}, false),
           DetectronPriorGridGenerator({2, 4}, PartialShape{-1, -1, 6, 6}, {1, 5, -1, -1}, {6, 6, 2, 4}, false),
           DetectronPriorGridGenerator({2, 4}, PartialShape{1, {0, 5}, 7, 6}, {1, 5, -1, -1}, {7, 6, 2, 4}, false),
           DetectronPriorGridGenerator({2, 4}, PartialShape{1, 3, 7, 6}, PartialShape::dynamic(), {7, 6, 2, 4}, false),
           DetectronPriorGridGenerator({2, 4}, PartialShape{1, 3, 7, 6}, PartialShape::dynamic(4), {7, 6, 2, 4}, false),
           DetectronPriorGridGenerator(PartialShape::dynamic(2),
                                       PartialShape::dynamic(4),
                                       PartialShape::dynamic(4),
                                       PartialShape{-1, -1, -1, 4},
                                       false),
           // flatten on
           DetectronPriorGridGenerator(PartialShape::dynamic(), {1, -1, 5, 3}, {1, 3, 100, 100}, {-1, 4}, true),
           DetectronPriorGridGenerator(PartialShape::dynamic(2), {1, -1, 5, 3}, {1, 3, 100, 100}, {-1, 4}, true),
           DetectronPriorGridGenerator({{2, 4}, 4}, {1, -1, 5, 3}, {1, 3, 100, 100}, {{30, 60}, 4}, true),
           DetectronPriorGridGenerator({{2, 4}, {2, 5}}, {1, 3, -1, 3}, {1, -1, 100, 100}, {-1, 4}, true),
           DetectronPriorGridGenerator({2, 4}, PartialShape::dynamic(), {1, 5, -1, -1}, {-1, 4}, true),
           DetectronPriorGridGenerator({2, 4}, PartialShape::dynamic(4), {1, 5, -1, -1}, {-1, 4}, true),
           DetectronPriorGridGenerator({2, 4}, PartialShape{-1, -1, 6, 6}, {1, 5, -1, -1}, {72, 4}, true),
           DetectronPriorGridGenerator({2, 4}, PartialShape{1, {0, 5}, {6, 7}, 6}, {1, 5, -1, -1}, {{72, 84}, 4}, true),
           DetectronPriorGridGenerator({2, 4}, PartialShape{1, 3, 7, 6}, PartialShape::dynamic(), {84, 4}, true),
           DetectronPriorGridGenerator({2, 4}, PartialShape{1, 3, 7, 6}, PartialShape::dynamic(4), {84, 4}, true),
           DetectronPriorGridGenerator(PartialShape::dynamic(2),
                                       PartialShape::dynamic(4),
                                       PartialShape::dynamic(4),
                                       PartialShape{-1, 4},
                                       true)),
    PrintToStringParamName());

TEST_P(ExperimentalDetectronPriorGridGeneratorV6Test, shape_inference) {
    const auto priors = std::make_shared<Parameter>(element::bf16, priors_shape);
    const auto feat_map = std::make_shared<Parameter>(element::bf16, feat_map_shape);
    const auto im_data = std::make_shared<Parameter>(element::bf16, im_data_shape);

    const auto op = make_op(priors, feat_map, im_data, make_attrs(is_flatten));

    EXPECT_EQ(op->get_output_partial_shape(0), exp_shape);
}
