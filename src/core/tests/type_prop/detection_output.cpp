// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/detection_output.hpp"

#include <memory>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

// ------------------------------ V0 ------------------------------
std::shared_ptr<op::DetectionOutput> create_detection_output(const PartialShape& box_logits_shape,
                                                             const PartialShape& class_preds_shape,
                                                             const PartialShape& proposals_shape,
                                                             const PartialShape& aux_class_preds_shape,
                                                             const PartialShape& aux_box_preds_shape,
                                                             const op::DetectionOutputAttrs& attrs,
                                                             element::Type input_type,
                                                             element::Type proposals_type) {
    auto box_logits = make_shared<op::Parameter>(input_type, box_logits_shape);
    auto class_preds = make_shared<op::Parameter>(input_type, class_preds_shape);
    auto proposals = make_shared<op::Parameter>(proposals_type, proposals_shape);
    auto aux_class_preds = make_shared<op::Parameter>(input_type, aux_class_preds_shape);
    auto aux_box_preds = make_shared<op::Parameter>(input_type, aux_box_preds_shape);
    return make_shared<op::DetectionOutput>(box_logits, class_preds, proposals, aux_class_preds, aux_box_preds, attrs);
}

TEST(type_prop_layers, detection_output) {
    op::DetectionOutputAttrs attrs;
    attrs.keep_top_k = {200};
    attrs.num_classes = 2;
    attrs.normalized = true;
    auto op = create_detection_output(Shape{4, 20},
                                      Shape{4, 10},
                                      Shape{4, 2, 20},
                                      Shape{4, 10},
                                      Shape{4, 20},
                                      attrs,
                                      element::f32,
                                      element::f32);
    ASSERT_EQ(op->get_shape(), (Shape{1, 1, 800, 7}));
    ASSERT_EQ(op->get_element_type(), element::f32);
}

TEST(type_prop_layers, detection_output_f16) {
    op::DetectionOutputAttrs attrs;
    attrs.keep_top_k = {200};
    attrs.num_classes = 2;
    attrs.normalized = true;
    auto op = create_detection_output(Shape{4, 20},
                                      Shape{4, 10},
                                      Shape{4, 2, 20},
                                      Shape{4, 10},
                                      Shape{4, 20},
                                      attrs,
                                      element::f16,
                                      element::f16);
    ASSERT_EQ(op->get_shape(), (Shape{1, 1, 800, 7}));
    ASSERT_EQ(op->get_element_type(), element::f16);
}

TEST(type_prop_layers, detection_f16_with_proposals_f32) {
    op::DetectionOutputAttrs attrs;
    attrs.keep_top_k = {200};
    attrs.num_classes = 2;
    attrs.normalized = true;
    auto op = create_detection_output(Shape{4, 20},
                                      Shape{4, 10},
                                      Shape{4, 2, 20},
                                      Shape{4, 10},
                                      Shape{4, 20},
                                      attrs,
                                      element::f16,
                                      element::f32);
    ASSERT_EQ(op->get_shape(), (Shape{1, 1, 800, 7}));
    ASSERT_EQ(op->get_element_type(), element::f16);
}

TEST(type_prop_layers, detection_output_not_normalized) {
    op::DetectionOutputAttrs attrs;
    attrs.keep_top_k = {200};
    attrs.num_classes = 2;
    attrs.normalized = false;
    auto op = create_detection_output(Shape{4, 20},
                                      Shape{4, 10},
                                      Shape{4, 2, 25},
                                      Shape{4, 10},
                                      Shape{4, 20},
                                      attrs,
                                      element::f32,
                                      element::f32);
    ASSERT_EQ(op->get_shape(), (Shape{1, 1, 800, 7}));
    ASSERT_EQ(op->get_element_type(), element::f32);
}

TEST(type_prop_layers, detection_output_negative_keep_top_k) {
    op::DetectionOutputAttrs attrs;
    attrs.keep_top_k = {-1};
    attrs.top_k = -1;
    attrs.normalized = true;
    attrs.num_classes = 2;
    auto op = create_detection_output(Shape{4, 20},
                                      Shape{4, 10},
                                      Shape{4, 2, 20},
                                      Shape{4, 10},
                                      Shape{4, 20},
                                      attrs,
                                      element::f32,
                                      element::f32);
    ASSERT_EQ(op->get_shape(), (Shape{1, 1, 40, 7}));
    ASSERT_EQ(op->get_element_type(), element::f32);
}

TEST(type_prop_layers, detection_output_no_share_location) {
    op::DetectionOutputAttrs attrs;
    attrs.keep_top_k = {-1};
    attrs.top_k = -1;
    attrs.normalized = true;
    attrs.num_classes = 2;
    attrs.share_location = false;
    auto op = create_detection_output(Shape{4, 40},
                                      Shape{4, 10},
                                      Shape{4, 2, 20},
                                      Shape{4, 10},
                                      Shape{4, 40},
                                      attrs,
                                      element::f32,
                                      element::f32);
    ASSERT_EQ(op->get_shape(), (Shape{1, 1, 40, 7}));
    ASSERT_EQ(op->get_element_type(), element::f32);
}

TEST(type_prop_layers, detection_output_calculated_num_prior_boxes) {
    op::DetectionOutputAttrs attrs;
    attrs.keep_top_k = {-1};
    attrs.top_k = -1;
    attrs.normalized = true;
    attrs.num_classes = 2;
    attrs.share_location = false;
    auto op = create_detection_output(PartialShape{4, -1},
                                      PartialShape::dynamic(),
                                      PartialShape::dynamic(),
                                      PartialShape{-1, 20},
                                      PartialShape::dynamic(),
                                      attrs,
                                      element::f32,
                                      element::f32);
    ASSERT_EQ(op->get_shape(), (Shape{1, 1, 80, 7}));
    ASSERT_EQ(op->get_element_type(), element::f32);
}

TEST(type_prop_layers, detection_output_top_k) {
    op::DetectionOutputAttrs attrs;
    attrs.keep_top_k = {-1};
    attrs.top_k = 7;
    attrs.normalized = true;
    attrs.num_classes = 2;
    auto op = create_detection_output(Shape{4, 20},
                                      Shape{4, 10},
                                      Shape{4, 2, 20},
                                      Shape{4, 10},
                                      Shape{4, 20},
                                      attrs,
                                      element::f32,
                                      element::f32);
    ASSERT_EQ(op->get_shape(), (Shape{1, 1, 56, 7}));
    ASSERT_EQ(op->get_element_type(), element::f32);
}

TEST(type_prop_layers, detection_output_all_dynamic_shapes) {
    PartialShape dyn_shape = PartialShape::dynamic();
    op::DetectionOutputAttrs attrs;
    attrs.keep_top_k = {-1};
    attrs.num_classes = 1;
    auto op = create_detection_output(dyn_shape,
                                      dyn_shape,
                                      dyn_shape,
                                      dyn_shape,
                                      dyn_shape,
                                      attrs,
                                      element::f32,
                                      element::f32);
    ASSERT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 1, Dimension::dynamic(), 7}));
    ASSERT_EQ(op->get_element_type(), element::f32);
}

TEST(type_prop_layers, detection_output_dynamic_batch) {
    op::DetectionOutputAttrs attrs;
    attrs.keep_top_k = {200};
    attrs.num_classes = 2;
    attrs.normalized = true;
    auto op = create_detection_output(PartialShape{Dimension::dynamic(), 20},
                                      PartialShape{Dimension::dynamic(), 10},
                                      PartialShape{Dimension::dynamic(), 2, 20},
                                      PartialShape{Dimension::dynamic(), 10},
                                      PartialShape{Dimension::dynamic(), 20},
                                      attrs,
                                      element::f32,
                                      element::f32);
    ASSERT_EQ(op->get_output_partial_shape(0), (PartialShape{{1, 1, Dimension::dynamic(), 7}}));
    ASSERT_EQ(op->get_element_type(), element::f32);
}

void detection_output_invalid_data_type_test(element::Type box_logits_et,
                                             element::Type class_preds_et,
                                             element::Type proposals_et,
                                             element::Type aux_class_preds_et,
                                             element::Type aux_box_preds_et,
                                             const std::string& expected_msg) {
    try {
        auto box_logits = make_shared<op::Parameter>(box_logits_et, Shape{4, 20});
        auto class_preds = make_shared<op::Parameter>(class_preds_et, Shape{4, 10});
        auto proposals = make_shared<op::Parameter>(proposals_et, Shape{4, 2, 20});
        auto aux_class_preds = make_shared<op::Parameter>(aux_class_preds_et, Shape{4, 10});
        auto aux_box_preds = make_shared<op::Parameter>(aux_box_preds_et, Shape{4, 20});
        op::DetectionOutputAttrs attrs;
        attrs.keep_top_k = {200};
        attrs.num_classes = 2;
        attrs.normalized = true;
        auto op =
            make_shared<op::DetectionOutput>(box_logits, class_preds, proposals, aux_class_preds, aux_box_preds, attrs);
        FAIL() << "Exception expected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), expected_msg);
    } catch (...) {
        FAIL() << "Unknown exception was thrown";
    }
}

TEST(type_prop_layers, detection_output_invalid_data_type) {
    detection_output_invalid_data_type_test(element::i32,
                                            element::f32,
                                            element::f32,
                                            element::f32,
                                            element::f32,
                                            "Box logits' data type must be floating point. Got i32");
    detection_output_invalid_data_type_test(
        element::f32,
        element::i32,
        element::f32,
        element::f32,
        element::f32,
        "Class predictions' data type must be the same as box logits type (f32). Got i32");
    detection_output_invalid_data_type_test(element::f32,
                                            element::f32,
                                            element::i32,
                                            element::f32,
                                            element::f32,
                                            "Proposals' data type must be floating point. Got i32");
    detection_output_invalid_data_type_test(element::f32,
                                            element::f32,
                                            element::f32,
                                            element::i32,
                                            element::f32,
                                            "Additional class predictions' data type must be the "
                                            "same as class predictions data type (f32). Got i32");
    detection_output_invalid_data_type_test(element::f32,
                                            element::f32,
                                            element::f32,
                                            element::f32,
                                            element::i32,
                                            "Additional box predictions' data type must be the "
                                            "same as box logits data type (f32). Got i32");
}

TEST(type_prop_layers, detection_output_mismatched_batch_size) {
    {
        try {
            op::DetectionOutputAttrs attrs;
            attrs.keep_top_k = {200};
            attrs.num_classes = 2;
            attrs.normalized = true;
            auto op = create_detection_output(Shape{4, 20},
                                              Shape{5, 10},
                                              Shape{4, 2, 20},
                                              Shape{4, 10},
                                              Shape{4, 20},
                                              attrs,
                                              element::f32,
                                              element::f32);
            FAIL() << "Exception expected";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(),
                                 std::string("Class predictions' first dimension is not compatible with batch size."));
        } catch (...) {
            FAIL() << "Unknown exception was thrown";
        }
    }
    {
        try {
            op::DetectionOutputAttrs attrs;
            attrs.keep_top_k = {200};
            attrs.num_classes = 2;
            attrs.normalized = true;
            auto op = create_detection_output(Shape{4, 20},
                                              Shape{4, 10},
                                              Shape{5, 2, 20},
                                              Shape{4, 10},
                                              Shape{4, 20},
                                              attrs,
                                              element::f32,
                                              element::f32);
            FAIL() << "Exception expected";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(),
                                 std::string("Proposals' first dimension is must be equal to "
                                             "either batch size (4) or 1. Got: 5."));
        } catch (...) {
            FAIL() << "Unknown exception was thrown";
        }
    }
}

TEST(type_prop_layers, detection_output_invalid_ranks) {
    {
        try {
            op::DetectionOutputAttrs attrs;
            attrs.keep_top_k = {200};
            attrs.num_classes = 2;
            attrs.normalized = true;
            auto op = create_detection_output(Shape{4, 20, 1},
                                              Shape{4, 10},
                                              Shape{4, 2, 20},
                                              Shape{4, 10},
                                              Shape{4, 20},
                                              attrs,
                                              element::f32,
                                              element::f32);
            FAIL() << "Exception expected";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), std::string("Box logits rank must be 2. Got 3"));
        } catch (...) {
            FAIL() << "Unknown exception was thrown";
        }
    }
    {
        try {
            op::DetectionOutputAttrs attrs;
            attrs.keep_top_k = {200};
            attrs.num_classes = 2;
            attrs.normalized = true;
            auto op = create_detection_output(Shape{4, 20},
                                              Shape{4, 10, 1},
                                              Shape{4, 2, 20},
                                              Shape{4, 10},
                                              Shape{4, 20},
                                              attrs,
                                              element::f32,
                                              element::f32);
            FAIL() << "Exception expected";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), std::string("Class predictions rank must be 2. Got 3"));
        } catch (...) {
            FAIL() << "Unknown exception was thrown";
        }
    }
    {
        try {
            op::DetectionOutputAttrs attrs;
            attrs.keep_top_k = {200};
            attrs.num_classes = 2;
            attrs.normalized = true;
            auto op = create_detection_output(Shape{4, 20},
                                              Shape{4, 10},
                                              Shape{4, 2},
                                              Shape{4, 10},
                                              Shape{4, 20},
                                              attrs,
                                              element::f32,
                                              element::f32);
            FAIL() << "Exception expected";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), std::string("Proposals rank must be 3. Got 2"));
        } catch (...) {
            FAIL() << "Unknown exception was thrown";
        }
    }
}

TEST(type_prop_layers, detection_output_invalid_box_logits_shape) {
    // share_location = true
    {
        try {
            op::DetectionOutputAttrs attrs;
            attrs.keep_top_k = {-1};
            attrs.num_classes = 3;
            attrs.share_location = true;
            attrs.variance_encoded_in_target = false;
            attrs.normalized = true;
            auto op = create_detection_output(Shape{4, 13},
                                              Shape{4, 9},
                                              Shape{4, 2, 12},
                                              Shape{4, 6},
                                              Shape{4, 12},
                                              attrs,
                                              element::f32,
                                              element::f32);
            FAIL() << "Exception expected";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(
                error.what(),
                std::string("Box logits' second dimension must be a multiply of num_loc_classes * 4 (4)"));
        } catch (...) {
            FAIL() << "Unknown exception was thrown";
        }
    }
    // share_location = false
    {
        try {
            op::DetectionOutputAttrs attrs;
            attrs.keep_top_k = {-1};
            attrs.num_classes = 3;
            attrs.share_location = false;
            attrs.variance_encoded_in_target = false;
            attrs.normalized = true;
            auto op = create_detection_output(Shape{4, 37},
                                              Shape{4, 9},
                                              Shape{4, 2, 12},
                                              Shape{4, 6},
                                              Shape{4, 12},
                                              attrs,
                                              element::f32,
                                              element::f32);
            FAIL() << "Exception expected";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(
                error.what(),
                std::string("Box logits' second dimension must be a multiply of num_loc_classes * 4 (12)"));
        } catch (...) {
            FAIL() << "Unknown exception was thrown";
        }
    }
}

TEST(type_prop_layers, detection_output_invalid_class_preds_shape) {
    try {
        op::DetectionOutputAttrs attrs;
        attrs.keep_top_k = {-1};
        attrs.num_classes = 3;
        auto op = create_detection_output(Shape{4, 12},
                                          Shape{4, 10},
                                          Shape{4, 2, 12},
                                          Shape{4, 6},
                                          Shape{4, 12},
                                          attrs,
                                          element::f32,
                                          element::f32);
        FAIL() << "Exception expected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Class predictions' second dimension must be equal to "
                                         "num_prior_boxes * num_classes (9). Current value is: 10."));
    } catch (...) {
        FAIL() << "Unknown exception was thrown";
    }
}

TEST(type_prop_layers, detection_output_invalid_proposals_shape) {
    // variance_encoded_in_target = false
    {
        try {
            op::DetectionOutputAttrs attrs;
            attrs.keep_top_k = {-1};
            attrs.num_classes = 3;
            attrs.share_location = true;
            attrs.variance_encoded_in_target = false;
            attrs.normalized = true;
            auto op = create_detection_output(Shape{4, 12},
                                              Shape{4, 9},
                                              Shape{4, 1, 12},
                                              Shape{4, 6},
                                              Shape{4, 12},
                                              attrs,
                                              element::f32,
                                              element::f32);
            FAIL() << "Exception expected";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(
                error.what(),
                std::string("Proposals' second dimension is mismatched. Current value is: 1, expected: 2"));
        } catch (...) {
            FAIL() << "Unknown exception was thrown";
        }
    }
    // variance_encoded_in_target = true
    {
        try {
            op::DetectionOutputAttrs attrs;
            attrs.keep_top_k = {-1};
            attrs.num_classes = 3;
            attrs.share_location = true;
            attrs.variance_encoded_in_target = true;
            attrs.normalized = true;
            auto op = create_detection_output(Shape{4, 12},
                                              Shape{4, 9},
                                              Shape{4, 2, 12},
                                              Shape{4, 6},
                                              Shape{4, 12},
                                              attrs,
                                              element::f32,
                                              element::f32);
            FAIL() << "Exception expected";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(
                error.what(),
                std::string("Proposals' second dimension is mismatched. Current value is: 2, expected: 1"));
        } catch (...) {
            FAIL() << "Unknown exception was thrown";
        }
    }
    // normalized = false
    {
        try {
            op::DetectionOutputAttrs attrs;
            attrs.keep_top_k = {-1};
            attrs.num_classes = 3;
            attrs.share_location = true;
            attrs.variance_encoded_in_target = false;
            attrs.normalized = false;
            auto op = create_detection_output(Shape{4, 12},
                                              Shape{4, 9},
                                              Shape{4, 2, 16},
                                              Shape{4, 6},
                                              Shape{4, 12},
                                              attrs,
                                              element::f32,
                                              element::f32);
            FAIL() << "Exception expected";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(),
                                 std::string("Proposals' third dimension must be equal to num_prior_boxes * "
                                             "prior_box_size (15). Current value is: 16."));
        } catch (...) {
            FAIL() << "Unknown exception was thrown";
        }
    }
    // normalized = true
    {
        try {
            op::DetectionOutputAttrs attrs;
            attrs.keep_top_k = {-1};
            attrs.num_classes = 3;
            attrs.share_location = true;
            attrs.variance_encoded_in_target = false;
            attrs.normalized = true;
            auto op = create_detection_output(Shape{4, 12},
                                              Shape{4, 9},
                                              Shape{4, 2, 13},
                                              Shape{4, 6},
                                              Shape{4, 12},
                                              attrs,
                                              element::f32,
                                              element::f32);
            FAIL() << "Exception expected";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(),
                                 std::string("Proposals' third dimension must be equal to num_prior_boxes * "
                                             "prior_box_size (12). Current value is: 13."));
        } catch (...) {
            FAIL() << "Unknown exception was thrown";
        }
    }
}

TEST(type_prop_layers, detection_output_invalid_aux_class_preds) {
    // invalid batch size
    {
        try {
            op::DetectionOutputAttrs attrs;
            attrs.keep_top_k = {-1};
            attrs.num_classes = 3;
            attrs.share_location = true;
            attrs.variance_encoded_in_target = false;
            attrs.normalized = true;
            auto op = create_detection_output(Shape{4, 12},
                                              Shape{4, 9},
                                              Shape{4, 2, 12},
                                              Shape{5, 6},
                                              Shape{4, 12},
                                              attrs,
                                              element::f32,
                                              element::f32);
            FAIL() << "Exception expected";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(),
                                 std::string("Additional class predictions' first dimension must "
                                             "be compatible with batch size."));
        } catch (...) {
            FAIL() << "Unknown exception was thrown";
        }
    }
    // invalid 2nd dimension
    {
        try {
            op::DetectionOutputAttrs attrs;
            attrs.keep_top_k = {-1};
            attrs.num_classes = 3;
            attrs.share_location = true;
            attrs.variance_encoded_in_target = false;
            attrs.normalized = true;
            auto op = create_detection_output(Shape{4, 12},
                                              Shape{4, 9},
                                              Shape{4, 2, 12},
                                              Shape{4, 7},
                                              Shape{4, 12},
                                              attrs,
                                              element::f32,
                                              element::f32);
            FAIL() << "Exception expected";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(
                error.what(),
                std::string("Additional class predictions' second dimension must "
                            "be compatible with num_prior_boxes * 2. Current value is: 7, expected: 6."));
        } catch (...) {
            FAIL() << "Unknown exception was thrown";
        }
    }
}

TEST(type_prop_layers, detection_output_invalid_aux_box_preds) {
    // invalid batch size
    {
        try {
            op::DetectionOutputAttrs attrs;
            attrs.keep_top_k = {-1};
            attrs.num_classes = 3;
            attrs.share_location = true;
            attrs.variance_encoded_in_target = false;
            attrs.normalized = true;
            auto op = create_detection_output(Shape{4, 12},
                                              Shape{4, 9},
                                              Shape{4, 2, 12},
                                              Shape{4, 6},
                                              Shape{5, 12},
                                              attrs,
                                              element::f32,
                                              element::f32);
            FAIL() << "Exception expected";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(
                error.what(),
                std::string("Additional box predictions' shape must be compatible with box logits shape."));
        } catch (...) {
            FAIL() << "Unknown exception was thrown";
        }
    }
    // invalid 2nd dimension
    {
        try {
            op::DetectionOutputAttrs attrs;
            attrs.keep_top_k = {-1};
            attrs.num_classes = 3;
            attrs.share_location = true;
            attrs.variance_encoded_in_target = false;
            attrs.normalized = true;
            auto op = create_detection_output(Shape{4, 12},
                                              Shape{4, 9},
                                              Shape{4, 2, 12},
                                              Shape{4, 6},
                                              Shape{4, 22},
                                              attrs,
                                              element::f32,
                                              element::f32);
            FAIL() << "Exception expected";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(
                error.what(),
                std::string("Additional box predictions' shape must be compatible with box logits shape."));
        } catch (...) {
            FAIL() << "Unknown exception was thrown";
        }
    }
}

// ------------------------------ V8 ------------------------------
namespace {
std::shared_ptr<op::v8::DetectionOutput> create_detection_output_v8(const PartialShape& box_logits_shape,
                                                                    const PartialShape& class_preds_shape,
                                                                    const PartialShape& proposals_shape,
                                                                    const op::v8::DetectionOutput::Attributes& attrs,
                                                                    element::Type input_type) {
    auto box_logits = make_shared<op::Parameter>(input_type, box_logits_shape);
    auto class_preds = make_shared<op::Parameter>(input_type, class_preds_shape);
    auto proposals = make_shared<op::Parameter>(input_type, proposals_shape);
    return make_shared<op::v8::DetectionOutput>(box_logits, class_preds, proposals, attrs);
}

std::shared_ptr<op::v8::DetectionOutput> create_detection_output2_v8(const PartialShape& box_logits_shape,
                                                                     const PartialShape& class_preds_shape,
                                                                     const PartialShape& proposals_shape,
                                                                     const PartialShape& aux_class_preds_shape,
                                                                     const PartialShape& aux_box_preds_shape,
                                                                     const op::v8::DetectionOutput::Attributes& attrs,
                                                                     element::Type input_type) {
    auto box_logits = make_shared<op::Parameter>(input_type, box_logits_shape);
    auto class_preds = make_shared<op::Parameter>(input_type, class_preds_shape);
    auto proposals = make_shared<op::Parameter>(input_type, proposals_shape);
    auto aux_class_preds = make_shared<op::Parameter>(input_type, aux_class_preds_shape);
    auto aux_box_preds = make_shared<op::Parameter>(input_type, aux_box_preds_shape);
    return make_shared<op::v8::DetectionOutput>(box_logits,
                                                class_preds,
                                                proposals,
                                                aux_class_preds,
                                                aux_box_preds,
                                                attrs);
}

PartialShape compute_reference_output_shape(const std::vector<int32_t>& keep_top_k,
                                            int top_k,
                                            const Dimension& deduced_N,
                                            const Dimension& deduced_num_classes,
                                            const Dimension& deduced_num_prior_boxes) {
    if (keep_top_k.size() > 0 && keep_top_k[0] > 0) {
        return PartialShape({1, 1, deduced_N * keep_top_k[0], 7});
    } else if (top_k > 0) {
        return PartialShape({1, 1, deduced_N * top_k * deduced_num_classes, 7});
    } else {
        return PartialShape({1, 1, deduced_N * deduced_num_classes * deduced_num_prior_boxes, 7});
    }
}

std::vector<op::v8::DetectionOutput::Attributes> create_attributes_vector() {
    // initialize attributes affecting shape inference
    // others remain by default
    std::vector<op::v8::DetectionOutput::Attributes> result;
    for (int keep_top_k : {10, -1}) {
        for (int top_k : {5, -1}) {
            for (bool variance_encoded_in_target : {true, false}) {
                for (bool share_location : {true, false}) {
                    for (bool normalized : {true, false}) {
                        op::v8::DetectionOutput::Attributes attrs;
                        attrs.top_k = top_k;
                        attrs.keep_top_k = {keep_top_k};
                        attrs.variance_encoded_in_target = variance_encoded_in_target;
                        attrs.share_location = share_location;
                        attrs.normalized = normalized;
                        result.push_back(attrs);
                    }
                }
            }
        }
    }
    return result;
}
}  // namespace

TEST(type_prop_layers, detection_outputv8_all_static) {
    // this case covers deducing a number of classes value
    // since this value is not saved in attributes
    op::v8::DetectionOutput::Attributes attrs;

    // initialize attributes affecting shape inference
    // others remain by default
    Dimension N = 5;
    Dimension num_prior_boxes = 100;
    Dimension priors_batch_size = N;
    Dimension num_classes = 23;

    auto attrs_vector = create_attributes_vector();
    for (const auto& attrs : attrs_vector) {
        Dimension num_loc_classes = attrs.share_location ? 1 : num_classes;
        Dimension prior_box_size = attrs.normalized ? 4 : 5;

        PartialShape box_logits_shape = {N, num_prior_boxes * num_loc_classes * 4};
        PartialShape class_preds_shape = {N, num_prior_boxes * num_classes};
        PartialShape proposals_shape = {priors_batch_size,
                                        attrs.variance_encoded_in_target ? 1 : 2,
                                        num_prior_boxes * prior_box_size};
        PartialShape output_shape_reference =
            compute_reference_output_shape(attrs.keep_top_k, attrs.top_k, N, num_classes, num_prior_boxes);

        auto op = create_detection_output_v8(box_logits_shape, class_preds_shape, proposals_shape, attrs, element::f32);
        ASSERT_EQ(op->get_output_partial_shape(0), output_shape_reference);
        ASSERT_EQ(op->get_element_type(), element::f32);
    }
}

TEST(type_prop_layers, detection_outputv8_dynamic) {
    op::v8::DetectionOutput::Attributes attrs;

    // initialize attributes affecting shape inference
    // others remain by default
    Dimension N = 13;
    Dimension num_prior_boxes = 33;
    Dimension priors_batch_size = 1;
    Dimension num_classes = 10;

    auto attrs_vector = create_attributes_vector();
    for (const auto& attrs : attrs_vector) {
        Dimension prior_box_size = attrs.normalized ? 4 : 5;

        PartialShape box_logits_shape = {N, Dimension::dynamic()};
        PartialShape class_preds_shape = {Dimension::dynamic(), num_prior_boxes * num_classes};
        PartialShape proposals_shape = {priors_batch_size,
                                        attrs.variance_encoded_in_target ? 1 : 2,
                                        num_prior_boxes * prior_box_size};
        PartialShape output_shape_reference =
            compute_reference_output_shape(attrs.keep_top_k, attrs.top_k, N, num_classes, num_prior_boxes);

        auto op = create_detection_output_v8(box_logits_shape, class_preds_shape, proposals_shape, attrs, element::f32);
        ASSERT_EQ(op->get_output_partial_shape(0), output_shape_reference);
        ASSERT_EQ(op->get_element_type(), element::f32);
    }
}

TEST(type_prop_layers, detection_outputv8_num_classes_not_deduced) {
    op::v8::DetectionOutput::Attributes attrs;

    // initialize attributes affecting shape inference
    // others remain by default
    Dimension N = 13;
    Dimension num_prior_boxes = 33;
    Dimension priors_batch_size = 1;

    auto attrs_vector = create_attributes_vector();
    for (const auto& attrs : attrs_vector) {
        Dimension prior_box_size = attrs.normalized ? 4 : 5;

        PartialShape box_logits_shape = {N, Dimension::dynamic()};
        PartialShape class_preds_shape = {N, Dimension::dynamic()};
        PartialShape proposals_shape = {priors_batch_size,
                                        attrs.variance_encoded_in_target ? 1 : 2,
                                        num_prior_boxes * prior_box_size};
        PartialShape output_shape_reference =
            compute_reference_output_shape(attrs.keep_top_k, attrs.top_k, N, Dimension::dynamic(), num_prior_boxes);

        auto op = create_detection_output_v8(box_logits_shape, class_preds_shape, proposals_shape, attrs, element::f32);
        ASSERT_EQ(op->get_output_partial_shape(0), output_shape_reference);
        ASSERT_EQ(op->get_element_type(), element::f32);
    }
}

TEST(type_prop_layers, detection_outputv8_num_classes_no_deduction) {
    // In this case a number of classes and a number of prior boxes are not deduced
    op::v8::DetectionOutput::Attributes attrs;

    // initialize attributes affecting shape inference
    // others remain by default
    Dimension N = 3;
    Dimension priors_batch_size = N;

    auto attrs_vector = create_attributes_vector();
    for (const auto& attrs : attrs_vector) {
        PartialShape box_logits_shape = {N, Dimension::dynamic()};
        PartialShape class_preds_shape = {N, Dimension::dynamic()};
        PartialShape proposals_shape = {priors_batch_size,
                                        attrs.variance_encoded_in_target ? 1 : 2,
                                        Dimension::dynamic()};
        PartialShape output_shape_reference = compute_reference_output_shape(attrs.keep_top_k,
                                                                             attrs.top_k,
                                                                             N,
                                                                             Dimension::dynamic(),
                                                                             Dimension::dynamic());

        auto op = create_detection_output_v8(box_logits_shape, class_preds_shape, proposals_shape, attrs, element::f32);
        ASSERT_EQ(op->get_output_partial_shape(0), output_shape_reference);
        ASSERT_EQ(op->get_element_type(), element::f32);
    }
}

TEST(type_prop_layers, detection_outputv8_dynamic2) {
    // In this case a number of prior boxes is deduced using additional input
    // and after that a number of classes is deduced using the second input shape
    op::v8::DetectionOutput::Attributes attrs;

    // initialize attributes affecting shape inference
    // others remain by default
    Dimension N = 13;
    Dimension num_prior_boxes = 33;
    Dimension priors_batch_size = 1;
    Dimension num_classes = 10;

    auto attrs_vector = create_attributes_vector();
    for (const auto& attrs : attrs_vector) {
        PartialShape box_logits_shape = {N, Dimension::dynamic()};
        PartialShape class_preds_shape = {Dimension::dynamic(), num_prior_boxes * num_classes};
        PartialShape proposals_shape = {priors_batch_size,
                                        attrs.variance_encoded_in_target ? 1 : 2,
                                        Dimension::dynamic()};
        PartialShape ad_class_preds_shape = {N, num_prior_boxes * 2};
        PartialShape ad_box_preds_shape = {N, Dimension::dynamic()};
        PartialShape output_shape_reference =
            compute_reference_output_shape(attrs.keep_top_k, attrs.top_k, N, num_classes, num_prior_boxes);

        auto op = create_detection_output2_v8(box_logits_shape,
                                              class_preds_shape,
                                              proposals_shape,
                                              ad_class_preds_shape,
                                              ad_box_preds_shape,
                                              attrs,
                                              element::f32);
        ASSERT_EQ(op->get_output_partial_shape(0), output_shape_reference);
        ASSERT_EQ(op->get_element_type(), element::f32);
    }
}
