// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/opsets/opset10.hpp"

using namespace std;
using namespace ov;
using namespace ov::opset10;
using namespace testing;

class TypePropCTCLossV4Test : public TypePropOpTest<op::v4::CTCLoss> {};

TEST_F(TypePropCTCLossV4Test, with_blank_index) {
    // create inputs
    auto logits_shape = PartialShape{10, 120, 28};
    auto symbols = set_shape_symbols(logits_shape);

    auto logits = make_shared<Parameter>(element::f32, logits_shape);
    auto logit_length = make_shared<Parameter>(element::i32, Shape{10});
    auto labels = make_shared<Parameter>(element::i32, Shape{10, 120});
    auto label_length = make_shared<Parameter>(element::i32, Shape{10});
    auto blank_index = make_shared<Parameter>(element::i32, Shape{});

    // create CTCLoss node
    auto ctc_loss = make_op(logits, logit_length, labels, label_length, blank_index);

    // check type and shape infer
    EXPECT_EQ(ctc_loss->get_element_type(), element::f32);
    EXPECT_EQ(ctc_loss->get_output_partial_shape(0), PartialShape({10}));
    EXPECT_THAT(get_shape_symbols(ctc_loss->get_output_partial_shape(0)), ElementsAre(symbols[0]));
}

TEST_F(TypePropCTCLossV4Test, no_blank_index) {
    // create inputs
    auto logits = make_shared<Parameter>(element::f32, Shape{10, 120, 28});
    auto logit_length = make_shared<Parameter>(element::i32, Shape{10});

    auto labels_shape = PartialShape{10, 120};
    auto symbols = set_shape_symbols(labels_shape);
    auto labels = make_shared<Parameter>(element::i32, labels_shape);
    auto label_length = make_shared<Parameter>(element::i32, Shape{10});

    // create CTCLoss node
    auto ctc_loss = make_op(logits, logit_length, labels, label_length);

    // check type and shape infer
    EXPECT_EQ(ctc_loss->get_element_type(), element::f32);
    EXPECT_EQ(ctc_loss->get_output_partial_shape(0), PartialShape({10}));
    EXPECT_THAT(get_shape_symbols(ctc_loss->get_output_partial_shape(0)), ElementsAre(symbols[0]));
}

TEST_F(TypePropCTCLossV4Test, output_type_f64) {
    // create inputs
    auto logits = make_shared<Parameter>(element::f64, Shape{10, 120, 28});
    auto logit_length = make_shared<Parameter>(element::i32, Shape{10});
    auto labels = make_shared<Parameter>(element::i32, Shape{10, 120});

    auto label_len_shape = PartialShape{10};
    auto symbols = set_shape_symbols(label_len_shape);
    auto label_length = make_shared<Parameter>(element::i32, label_len_shape);
    auto blank_index = make_shared<Parameter>(element::i32, Shape{});

    // create CTCLoss node
    auto ctc_loss = make_op(logits, logit_length, labels, label_length, blank_index);

    // check type and shape infer
    EXPECT_EQ(ctc_loss->get_element_type(), element::f64);
    EXPECT_EQ(ctc_loss->get_output_partial_shape(0), PartialShape({10}));
    EXPECT_THAT(get_shape_symbols(ctc_loss->get_output_partial_shape(0)), ElementsAre(symbols[0]));
}

TEST_F(TypePropCTCLossV4Test, non_default_parameters) {
    // create inputs
    auto logits = make_shared<Parameter>(element::f64, Shape{10, 120, 28});
    auto logit_length = make_shared<Parameter>(element::i32, Shape{10});
    auto labels = make_shared<Parameter>(element::i32, Shape{10, 120});
    auto label_length = make_shared<Parameter>(element::i32, Shape{10});
    auto blank_index = make_shared<Parameter>(element::i32, Shape{});

    // create CTCLoss node
    auto ctc_loss = make_op(logits, logit_length, labels, label_length, blank_index, true, false, false);

    // check type and shape infer
    EXPECT_EQ(ctc_loss->get_element_type(), element::f64);
    EXPECT_EQ(ctc_loss->get_output_partial_shape(0), PartialShape({10}));
}

TEST_F(TypePropCTCLossV4Test, dynamic_input) {
    // create inputs
    auto logits = make_shared<Parameter>(element::f32, PartialShape{Dimension::dynamic(), 120, 28});
    auto logit_length = make_shared<Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    auto labels = make_shared<Parameter>(element::i32, PartialShape{Dimension::dynamic(), 120});
    auto label_length = make_shared<Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    auto blank_index = make_shared<Parameter>(element::i32, Shape{});

    // create CTCLoss node
    auto ctc_loss = make_op(logits, logit_length, labels, label_length, blank_index);

    // check type and shape infer
    EXPECT_EQ(ctc_loss->get_element_type(), element::f32);
    EXPECT_EQ(ctc_loss->get_output_partial_shape(0), PartialShape::dynamic(1));
}

TEST_F(TypePropCTCLossV4Test, partly_dynamic_input) {
    // create inputs
    auto logits_shape = PartialShape{{2, 20}, {100, 130}, 28};
    auto logits_len_shape = PartialShape{{5, 10}};
    auto labels_shape = PartialShape{-1, 120};
    auto symbols = set_shape_symbols(logits_shape);
    set_shape_symbols(logits_len_shape);
    set_shape_symbols(labels_shape);

    auto logits = make_shared<Parameter>(element::f32, logits_shape);
    auto logit_length = make_shared<Parameter>(element::i32, logits_len_shape);
    auto labels = make_shared<Parameter>(element::i32, labels_shape);
    auto label_length = make_shared<Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    auto blank_index = make_shared<Parameter>(element::i32, Shape{});

    // create CTCLoss node
    auto ctc_loss = make_op(logits, logit_length, labels, label_length, blank_index);

    // check type and shape infer
    EXPECT_EQ(ctc_loss->get_element_type(), element::f32);
    EXPECT_EQ(ctc_loss->get_output_partial_shape(0), PartialShape({{5, 10}}));
    EXPECT_THAT(get_shape_symbols(ctc_loss->get_output_partial_shape(0)), ElementsAre(symbols[0]));
}

TEST_F(TypePropCTCLossV4Test, fail_inputs_dim) {
    // create inputs
    auto logits = make_shared<Parameter>(element::f32, Shape{10, 120, 40, 28});
    auto logit_length = make_shared<Parameter>(element::i32, Shape{10});
    auto labels = make_shared<Parameter>(element::i32, Shape{10, 120});
    auto label_length = make_shared<Parameter>(element::i32, Shape{10});
    auto blank_index = make_shared<Parameter>(element::i32, Shape{});

    OV_EXPECT_THROW(auto op = make_op(logits, logit_length, labels, label_length, blank_index),
                    NodeValidationFailure,
                    HasSubstr("Expected a 3D tensor for logits."));
}

TEST_F(TypePropCTCLossV4Test, fail_logit_length_dim) {
    // create inputs
    auto logits = make_shared<Parameter>(element::f32, Shape{10, 120, 28});
    auto logit_length = make_shared<Parameter>(element::i32, Shape{10, 20});
    auto labels = make_shared<Parameter>(element::i32, Shape{10, 120});
    auto label_length = make_shared<Parameter>(element::i32, Shape{10});
    auto blank_index = make_shared<Parameter>(element::i32, Shape{});

    OV_EXPECT_THROW(auto op = make_op(logits, logit_length, labels, label_length, blank_index),
                    NodeValidationFailure,
                    HasSubstr("Expected a 1D tensor for logit length."));
}

TEST_F(TypePropCTCLossV4Test, fail_labels_dim) {
    // create inputs
    auto logits = make_shared<Parameter>(element::f32, Shape{10, 120, 28});
    auto logit_length = make_shared<Parameter>(element::i32, Shape{10});
    auto labels = make_shared<Parameter>(element::i32, Shape{10});
    auto label_length = make_shared<Parameter>(element::i32, Shape{10});
    auto blank_index = make_shared<Parameter>(element::i32, Shape{});

    OV_EXPECT_THROW(auto op = make_op(logits, logit_length, labels, label_length, blank_index),
                    NodeValidationFailure,
                    HasSubstr("Expected a 2D tensor for labels."));
}

TEST_F(TypePropCTCLossV4Test, fail_label_length_dim) {
    // create inputs
    auto logits = make_shared<Parameter>(element::f32, Shape{10, 120, 28});
    auto logit_length = make_shared<Parameter>(element::i32, Shape{10});
    auto labels = make_shared<Parameter>(element::i32, Shape{10, 120});
    auto label_length = make_shared<Parameter>(element::i32, Shape{10, 40});
    auto blank_index = make_shared<Parameter>(element::i32, Shape{});

    OV_EXPECT_THROW(auto op = make_op(logits, logit_length, labels, label_length, blank_index),
                    NodeValidationFailure,
                    HasSubstr("Expected a 1D tensor for label length."));
}

TEST_F(TypePropCTCLossV4Test, fail_blank_index_dim) {
    // create inputs
    auto logits = make_shared<Parameter>(element::f32, Shape{10, 120, 28});
    auto logit_length = make_shared<Parameter>(element::i32, Shape{10});
    auto labels = make_shared<Parameter>(element::i32, Shape{10, 120});
    auto label_length = make_shared<Parameter>(element::i32, Shape{10});
    auto blank_index = make_shared<Parameter>(element::i32, Shape{4});

    OV_EXPECT_THROW(auto op = make_op(logits, logit_length, labels, label_length, blank_index),
                    NodeValidationFailure,
                    HasSubstr("Expected a scalar for blank index."));
}

TEST_F(TypePropCTCLossV4Test, fail_batch_dim_mismatch) {
    // create inputs
    auto logits = make_shared<Parameter>(element::f32, Shape{10, 120, 28});
    auto logit_length = make_shared<Parameter>(element::i32, Shape{10});
    auto labels = make_shared<Parameter>(element::i32, Shape{10, 120});
    auto label_length = make_shared<Parameter>(element::i32, Shape{40});
    auto blank_index = make_shared<Parameter>(element::i32, Shape{});

    OV_EXPECT_THROW(auto op = make_op(logits, logit_length, labels, label_length, blank_index),
                    NodeValidationFailure,
                    HasSubstr("The first dimension of label length must be equal to the first dimension of the logits, "
                              "the logit length and labels."));
}

TEST_F(TypePropCTCLossV4Test, fail_time_dim_mismatch) {
    // create inputs
    auto logits = make_shared<Parameter>(element::f32, Shape{10, 120, 28});
    auto logit_length = make_shared<Parameter>(element::i32, Shape{10});
    auto labels = make_shared<Parameter>(element::i32, Shape{10, 130});
    auto label_length = make_shared<Parameter>(element::i32, Shape{40});
    auto blank_index = make_shared<Parameter>(element::i32, Shape{});

    OV_EXPECT_THROW(auto op = make_op(logits, logit_length, labels, label_length, blank_index),
                    NodeValidationFailure,
                    HasSubstr("The second dimension of labels must be equal to the second dimension of logits."));
}

TEST_F(TypePropCTCLossV4Test, default_ctor) {
    // create inputs
    auto logits_shape = PartialShape{{2, 20}, {100, 130}, 28};
    auto logits_len_shape = PartialShape{{5, 10}};
    auto labels_shape = PartialShape{-1, 120};
    auto symbols = set_shape_symbols(logits_shape);
    set_shape_symbols(logits_len_shape);
    set_shape_symbols(labels_shape);

    auto logits = make_shared<Parameter>(element::f32, logits_shape);
    auto logit_length = make_shared<Parameter>(element::i32, logits_len_shape);
    auto labels = make_shared<Parameter>(element::i32, labels_shape);
    auto label_length = make_shared<Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    auto blank_index = make_shared<Parameter>(element::i32, Shape{});

    // create CTCLoss node
    auto ctc_loss = make_op();
    ctc_loss->set_arguments(OutputVector{logits, logit_length, labels, label_length, blank_index});
    ctc_loss->validate_and_infer_types();

    // check type and shape infer
    EXPECT_EQ(ctc_loss->get_element_type(), element::f32);
    EXPECT_EQ(ctc_loss->get_output_partial_shape(0), PartialShape({{5, 10}}));
    EXPECT_THAT(get_shape_symbols(ctc_loss->get_output_partial_shape(0)), ElementsAre(symbols[0]));
}
