// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class ExperimentalDetectronROIFeatureExtractorV6StaticShapeInferenceTest
    : public OpStaticShapeInferenceTest<op::v6::ExperimentalDetectronROIFeatureExtractor> {
protected:
    void SetUp() override {
        output_shapes.resize(2);
    }

    static op_type::Attributes make_attrs(int64_t out_size) {
        return {out_size, 2, {4, 8, 16, 32}, false};
    }
};

TEST_F(ExperimentalDetectronROIFeatureExtractorV6StaticShapeInferenceTest, default_ctor) {
    op = make_op();
    op->set_attrs(make_attrs(16));

    input_shapes = StaticShapeVector{{1000, 4}, {1, 5, 8, 8}, {1, 5, 16, 16}, {1, 5, 64, 64}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_THAT(output_shapes, ElementsAre(StaticShape{1000, 5, 16, 16}, StaticShape{1000, 4}));
}

TEST_F(ExperimentalDetectronROIFeatureExtractorV6StaticShapeInferenceTest, inputs_dynamic_rank) {
    const auto rois = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic());
    const auto layer_0 = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic());
    const auto layer_1 = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic());
    op = make_op(OutputVector{rois, layer_0, layer_1}, make_attrs(100));

    input_shapes = StaticShapeVector{{25, 4}, {1, 2, 100, 100}, {1, 2, 20, 300}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_THAT(output_shapes, ElementsAre(StaticShape{25, 2, 100, 100}, StaticShape{25, 4}));
}

TEST_F(ExperimentalDetectronROIFeatureExtractorV6StaticShapeInferenceTest, inputs_static_rank) {
    const auto rois = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(2));
    const auto layer_0 = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(4));
    const auto layer_1 = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(4));
    const auto layer_2 = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(4));
    const auto layer_3 = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(4));
    op = make_op(OutputVector{rois, layer_0, layer_1, layer_2, layer_3}, make_attrs(15));

    input_shapes = StaticShapeVector{{25, 4}, {1, 2, 100, 100}, {1, 2, 20, 300}, {1, 2, 30, 30}, {1, 2, 200, 50}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_THAT(output_shapes, ElementsAre(StaticShape{25, 2, 15, 15}, StaticShape{25, 4}));
}

TEST_F(ExperimentalDetectronROIFeatureExtractorV6StaticShapeInferenceTest, rois_wrong_rank) {
    const auto rois = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic());
    const auto layer_0 = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(4));
    const auto layer_1 = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(4));
    const auto layer_2 = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(4));
    op = make_op(OutputVector{rois, layer_0, layer_1, layer_2}, make_attrs(15));

    input_shapes = StaticShapeVector{{25, 4, 1}, {1, 2, 20, 300}, {1, 2, 30, 30}, {1, 2, 200, 50}};
    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Input rois rank must be equal to 2"));
}

TEST_F(ExperimentalDetectronROIFeatureExtractorV6StaticShapeInferenceTest, layers_num_channels_not_same) {
    const auto rois = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(2));
    const auto layer_0 = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(4));
    const auto layer_1 = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic());
    const auto layer_2 = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic());
    op = make_op(OutputVector{rois, layer_0, layer_1, layer_2}, make_attrs(15));

    input_shapes = StaticShapeVector{{25, 4}, {1, 2, 20, 300}, {1, 2, 30, 30}, {1, 3, 200, 50}};
    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("The number of channels must be the same for all layers of the pyramid"));
}
