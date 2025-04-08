// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

template <class TOp>
class ProposalTest : public OpStaticShapeInferenceTest<TOp> {
protected:
    using Attrs = typename TOp::Attributes;

    static Attrs make_attrs(size_t post_nms_count) {
        Attrs attrs;
        attrs.post_nms_topn = post_nms_count;
        return attrs;
    }

    static size_t exp_out_size() {
        if (std::is_same<op::v0::Proposal, TOp>::value) {
            return 1;
        } else if (std::is_same<op::v4::Proposal, TOp>::value) {
            return 2;
        } else {
            return 0;
        }
    }
};

TYPED_TEST_SUITE_P(ProposalTest);

TYPED_TEST_P(ProposalTest, default_ctor) {
    this->op = this->make_op();
    this->op->set_attrs(this->make_attrs(10));

    this->input_shapes = StaticShapeVector{{2, 3, 10, 10}, {2, 6, 10, 10}, {3}};
    this->output_shapes = shape_inference(this->op.get(), this->input_shapes);

    EXPECT_EQ(this->output_shapes.size(), this->exp_out_size());
    EXPECT_EQ(this->output_shapes.front(), StaticShape({20, 5}));
}

TYPED_TEST_P(ProposalTest, all_inputs_dynamic_rank) {
    const auto class_probs = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto class_bbox_deltas = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto image_shape = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());

    this->op = this->make_op(class_probs, class_bbox_deltas, image_shape, this->make_attrs(4));

    this->input_shapes = StaticShapeVector{{2, 3, 10, 10}, {2, 6, 10, 10}, {3}};
    this->output_shapes = shape_inference(this->op.get(), this->input_shapes);

    EXPECT_EQ(this->output_shapes.size(), this->exp_out_size());
    EXPECT_EQ(this->output_shapes[0], StaticShape({8, 5}));
}

TYPED_TEST_P(ProposalTest, all_inputs_static_rank) {
    const auto class_probs = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto class_bbox_deltas = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto image_shape = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(1));

    this->op = this->make_op(class_probs, class_bbox_deltas, image_shape, this->make_attrs(5));

    this->input_shapes = StaticShapeVector{{3, 4, 10, 10}, {3, 8, 10, 10}, {4}};
    this->output_shapes = shape_inference(this->op.get(), this->input_shapes);

    EXPECT_EQ(this->output_shapes.size(), this->exp_out_size());
    EXPECT_EQ(this->output_shapes[0], StaticShape({15, 5}));
}

TYPED_TEST_P(ProposalTest, batch_size_not_compatible) {
    const auto class_probs = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto class_bbox_deltas = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto image_shape = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(1));

    this->op = this->make_op(class_probs, class_bbox_deltas, image_shape, this->make_attrs(5));

    this->input_shapes = StaticShapeVector{{3, 4, 10, 10}, {4, 8, 10, 10}, {3}};
    OV_EXPECT_THROW(shape_inference(this->op.get(), this->input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Batch size inconsistent between class_probs"));
}

TYPED_TEST_P(ProposalTest, image_shape_input_not_compatible_shape) {
    const auto class_probs = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto class_bbox_deltas = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto image_shape = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());

    this->op = this->make_op(class_probs, class_bbox_deltas, image_shape, this->make_attrs(5));

    this->input_shapes = StaticShapeVector{{3, 4, 10, 10}, {3, 8, 10, 10}, {5}};
    OV_EXPECT_THROW(shape_inference(this->op.get(), this->input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Image_shape must be 1-D tensor and has got 3 or 4 elements"));
}

REGISTER_TYPED_TEST_SUITE_P(ProposalTest,
                            default_ctor,
                            all_inputs_dynamic_rank,
                            all_inputs_static_rank,
                            batch_size_not_compatible,
                            image_shape_input_not_compatible_shape);

using ProposalVersions = Types<op::v0::Proposal, op::v4::Proposal>;
INSTANTIATE_TYPED_TEST_SUITE_P(shape_inference, ProposalTest, ProposalVersions);
