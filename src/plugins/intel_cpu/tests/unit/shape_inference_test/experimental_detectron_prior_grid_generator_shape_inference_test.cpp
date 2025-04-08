// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class ExperimentalDetectronPriorGridGeneratorV6StaticShapeInferenceTest
    : public OpStaticShapeInferenceTest<op::v6::ExperimentalDetectronPriorGridGenerator> {
protected:
    using Attrs = typename op_type::Attributes;

    void SetUp() override {
        output_shapes.resize(1);
    }

    static Attrs make_attrs(bool flatten) {
        return {flatten, 0, 0, 4.0f, 4.0f};
    }
};

TEST_F(ExperimentalDetectronPriorGridGeneratorV6StaticShapeInferenceTest, default_ctor) {
    op = make_op();
    op->set_attrs({true, 0, 0, 5.0f, 5.0f});

    input_shapes = StaticShapeVector{{3, 4}, {1, 5, 7, 2}, {1, 5, 50, 50}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes, StaticShapeVector({{42, 4}}));
}

TEST_F(ExperimentalDetectronPriorGridGeneratorV6StaticShapeInferenceTest, inputs_dynamic_rank) {
    const auto priors = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto feat_map = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto im_data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    op = make_op(priors, feat_map, im_data, make_attrs(false));

    input_shapes = StaticShapeVector{{10, 4}, {1, 2, 4, 5}, {1, 2, 100, 100}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({4, 5, 10, 4}));
}

TEST_F(ExperimentalDetectronPriorGridGeneratorV6StaticShapeInferenceTest, inputs_static_rank) {
    const auto priors = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto feat_map = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto im_data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    op = make_op(priors, feat_map, im_data, make_attrs(true));

    input_shapes = StaticShapeVector{{10, 4}, {1, 2, 4, 5}, {1, 2, 100, 100}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({200, 4}));
}

TEST_F(ExperimentalDetectronPriorGridGeneratorV6StaticShapeInferenceTest, feat_map_wrong_rank) {
    const auto priors = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto feat_map = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto im_data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    op = make_op(priors, feat_map, im_data, make_attrs(true));

    input_shapes = StaticShapeVector{{10, 4}, {1, 2, 4, 5, 1}, {1, 2, 100, 100}};
    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Feature_map rank must be equal to 4"));
}

TEST_F(ExperimentalDetectronPriorGridGeneratorV6StaticShapeInferenceTest, priors_2nd_dim_not_compatible) {
    const auto priors = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto feat_map = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto im_data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    op = make_op(priors, feat_map, im_data, make_attrs(true));

    input_shapes = StaticShapeVector{{10, 5}, {1, 2, 4, 5}, {1, 2, 100, 100}};
    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("The last dimension of the 'priors' input must be equal to 4"));
}
