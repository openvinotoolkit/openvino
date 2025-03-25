// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

template <typename T>
class StaticShapeROIAlignTest : public OpStaticShapeInferenceTest<T> {
protected:
    void SetUp() override {
        this->output_shapes.resize(1);
    }
};

TYPED_TEST_SUITE_P(StaticShapeROIAlignTest);

TYPED_TEST_P(StaticShapeROIAlignTest, default_ctor_no_args) {
    this->op = this->make_op();
    this->op->set_pooled_h(2);
    this->op->set_pooled_w(2);

    this->input_shapes = StaticShapeVector{{2, 3, 5, 5}, {7, 4}, {7}};
    this->output_shapes = shape_inference(this->op.get(), this->input_shapes);

    EXPECT_EQ(this->output_shapes.size(), 1);
    EXPECT_EQ(this->output_shapes[0], (StaticShape{7, 3, 2, 2}));
}

TYPED_TEST_P(StaticShapeROIAlignTest, all_inputs_dynamic_rank) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic());
    const auto rois = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic());
    const auto batch_indices = std::make_shared<op::v0::Parameter>(element::i8, PartialShape::dynamic());

    this->op = this->make_op(data, rois, batch_indices, 2, 2, 2, 1.0f, TypeParam::PoolingMode::AVG);

    this->input_shapes = StaticShapeVector{{2, 3, 5, 5}, {10, 4}, {10}};
    this->output_shapes = shape_inference(this->op.get(), this->input_shapes);

    EXPECT_EQ(this->output_shapes.size(), 1);
    EXPECT_EQ(this->output_shapes[0], (StaticShape{10, 3, 2, 2}));
}

TYPED_TEST_P(StaticShapeROIAlignTest, all_inputs_static_rank) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(4));
    const auto rois = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(2));
    const auto batch_indices = std::make_shared<op::v0::Parameter>(element::i8, PartialShape::dynamic(1));

    this->op = this->make_op(data, rois, batch_indices, 2, 2, 2, 1.0f, TypeParam::PoolingMode::AVG);

    this->input_shapes = StaticShapeVector{{2, 8, 5, 5}, {10, 4}, {10}};
    this->output_shapes = shape_inference(this->op.get(), this->input_shapes);

    EXPECT_EQ(this->output_shapes.size(), 1);
    EXPECT_EQ(this->output_shapes[0], (StaticShape{10, 8, 2, 2}));
}

TYPED_TEST_P(StaticShapeROIAlignTest, incompatible_input_rank) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic());
    const auto rois = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(2));
    const auto batch_indices = std::make_shared<op::v0::Parameter>(element::i8, PartialShape::dynamic(1));

    this->op = this->make_op(data, rois, batch_indices, 2, 2, 2, 1.0f, TypeParam::PoolingMode::AVG);

    this->input_shapes = StaticShapeVector{{2, 8, 5}, {10, 3}, {10}};
    OV_EXPECT_THROW(shape_inference(this->op.get(), this->input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Expected a 4D tensor for the input data"));
}

TYPED_TEST_P(StaticShapeROIAlignTest, incompatible_rois_rank) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(4));
    const auto rois = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic());
    const auto batch_indices = std::make_shared<op::v0::Parameter>(element::i8, PartialShape::dynamic(1));

    this->op = this->make_op(data, rois, batch_indices, 2, 2, 2, 1.0f, TypeParam::PoolingMode::AVG);

    this->input_shapes = StaticShapeVector{{2, 8, 5, 5}, {10, 3, 1}, {10}};
    OV_EXPECT_THROW(shape_inference(this->op.get(), this->input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Expected a 2D tensor for the ROIs input"));
}

TYPED_TEST_P(StaticShapeROIAlignTest, incompatible_batch_indicies_rank) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(4));
    const auto rois = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(2));
    const auto batch_indices = std::make_shared<op::v0::Parameter>(element::i8, PartialShape::dynamic());

    this->op = this->make_op(data, rois, batch_indices, 2, 2, 2, 1.0f, TypeParam::PoolingMode::AVG);
    this->input_shapes = StaticShapeVector{{2, 8, 5, 5}, {10, 3}, {}};
    OV_EXPECT_THROW(shape_inference(this->op.get(), this->input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Expected a 1D tensor for the batch indices input."));
}

TYPED_TEST_P(StaticShapeROIAlignTest, invalid_rois_2nd_dim) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(4));
    const auto rois = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(2));
    const auto batch_indices = std::make_shared<op::v0::Parameter>(element::i8, PartialShape::dynamic(1));

    this->op = this->make_op(data, rois, batch_indices, 2, 2, 2, 1.0f, TypeParam::PoolingMode::AVG);

    this->input_shapes = StaticShapeVector{{2, 8, 5, 5}, {10, 3}, {10}};
    OV_EXPECT_THROW(shape_inference(this->op.get(), this->input_shapes),
                    NodeValidationFailure,
                    HasSubstr("op dimension is expected to be equal to 4"));
}

REGISTER_TYPED_TEST_SUITE_P(StaticShapeROIAlignTest,
                            default_ctor_no_args,
                            all_inputs_dynamic_rank,
                            all_inputs_static_rank,
                            incompatible_input_rank,
                            incompatible_rois_rank,
                            incompatible_batch_indicies_rank,
                            invalid_rois_2nd_dim);

using ROIAlignTypes = Types<op::v3::ROIAlign, op::v9::ROIAlign>;
INSTANTIATE_TYPED_TEST_SUITE_P(shape_infer, StaticShapeROIAlignTest, ROIAlignTypes);
