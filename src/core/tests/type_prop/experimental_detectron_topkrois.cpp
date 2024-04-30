// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/opsets/opset11.hpp"

using namespace ov;
using namespace ov::opset11;
using namespace testing;

class TypePropExperimentalDetectronTopKROIsV6 : public TypePropOpTest<op::v6::ExperimentalDetectronTopKROIs> {};

TEST_F(TypePropExperimentalDetectronTopKROIsV6, default_ctor) {
    const auto input_rois = std::make_shared<Parameter>(element::f32, Shape{500, 4});
    const auto rois_probs = std::make_shared<Parameter>(element::f32, Shape{500});

    const auto op = make_op();
    op->set_arguments(OutputVector{input_rois, rois_probs});
    op->set_max_rois(20);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_input_size(), 2);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_max_rois(), 20);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({20, 4}));
}

TEST_F(TypePropExperimentalDetectronTopKROIsV6, simple_shape_infer) {
    const auto input_rois = std::make_shared<Parameter>(element::f32, Shape{5000, 4});
    const auto rois_probs = std::make_shared<Parameter>(element::f32, Shape{5000});
    const auto op = make_op(input_rois, rois_probs, 1000);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_shape(0), Shape({1000, 4}));
}

TEST_F(TypePropExperimentalDetectronTopKROIsV6, dynamic_rank_rois_and_probabilities) {
    const auto input_rois = std::make_shared<Parameter>(element::f64, PartialShape::dynamic());
    const auto rois_probs = std::make_shared<Parameter>(element::f64, PartialShape::dynamic());
    const auto op = make_op(input_rois, rois_probs, 1000);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f64);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({1000, 4}));
}

TEST_F(TypePropExperimentalDetectronTopKROIsV6, static_rank_rois_and_probabilities) {
    const auto input_rois = std::make_shared<Parameter>(element::f64, PartialShape::dynamic(2));
    const auto rois_probs = std::make_shared<Parameter>(element::f64, PartialShape::dynamic(1));
    const auto op = make_op(input_rois, rois_probs, 1000);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f64);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({1000, 4}));
}

TEST_F(TypePropExperimentalDetectronTopKROIsV6, interval_num_of_rois_and_dynamic_probabilities) {
    auto input_rois_shape = PartialShape{{20, 30}, 4};
    auto rois_prop_shape = PartialShape{-1};
    set_shape_symbols(input_rois_shape);
    set_shape_symbols(rois_prop_shape);

    const auto input_rois = std::make_shared<Parameter>(element::dynamic, input_rois_shape);
    const auto rois_probs = std::make_shared<Parameter>(element::dynamic, rois_prop_shape);
    const auto op = make_op(input_rois, rois_probs, 20);

    EXPECT_EQ(op->get_output_element_type(0), element::dynamic);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({20, 4}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), ElementsAre(nullptr, nullptr));
}

TEST_F(TypePropExperimentalDetectronTopKROIsV6, interval_num_of_rois_and_probabilities) {
    auto input_rois_shape = PartialShape{{20, 30}, 4};
    auto rois_prop_shape = PartialShape{{10, 35}};
    set_shape_symbols(input_rois_shape);
    set_shape_symbols(rois_prop_shape);

    const auto input_rois = std::make_shared<Parameter>(element::dynamic, input_rois_shape);
    const auto rois_probs = std::make_shared<Parameter>(element::dynamic, rois_prop_shape);
    const auto op = make_op(input_rois, rois_probs, 20);

    EXPECT_EQ(op->get_output_element_type(0), element::dynamic);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({20, 4}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), ElementsAre(nullptr, nullptr));
}

TEST_F(TypePropExperimentalDetectronTopKROIsV6, dynamic_num_rois_and_interval_probabilities) {
    auto input_rois_shape = PartialShape{-1, 4};
    auto rois_prop_shape = PartialShape{{10, 15}};
    set_shape_symbols(input_rois_shape);
    set_shape_symbols(rois_prop_shape);

    const auto input_rois = std::make_shared<Parameter>(element::dynamic, input_rois_shape);
    const auto rois_probs = std::make_shared<Parameter>(element::dynamic, rois_prop_shape);
    const auto op = make_op(input_rois, rois_probs, 200);

    EXPECT_EQ(op->get_output_element_type(0), element::dynamic);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({200, 4}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), ElementsAre(nullptr, nullptr));
}

TEST_F(TypePropExperimentalDetectronTopKROIsV6, element_type_of_inputs_are_not_same) {
    const auto input_rois = std::make_shared<Parameter>(element::f32, Shape{100, 4});
    const auto rois_probs = std::make_shared<Parameter>(element::f64, Shape{100});

    OV_EXPECT_THROW(auto op = make_op(input_rois, rois_probs, 200),
                    NodeValidationFailure,
                    HasSubstr("ROIs and probabilities of ROIs must same floating-point type"));
}

TEST_F(TypePropExperimentalDetectronTopKROIsV6, element_type_of_inputs_rois_is_not_floating_point) {
    const auto input_rois = std::make_shared<Parameter>(element::u32, Shape{100, 4});
    const auto rois_probs = std::make_shared<Parameter>(element::dynamic, Shape{100});

    OV_EXPECT_THROW(auto op = make_op(input_rois, rois_probs, 200),
                    NodeValidationFailure,
                    HasSubstr("ROIs and probabilities of ROIs must same floating-point type"));
}

TEST_F(TypePropExperimentalDetectronTopKROIsV6, element_type_of_probabilities_is_not_floating_point) {
    const auto input_rois = std::make_shared<Parameter>(element::dynamic, Shape{100, 4});
    const auto rois_probs = std::make_shared<Parameter>(element::i64, Shape{100});

    OV_EXPECT_THROW(auto op = make_op(input_rois, rois_probs, 200),
                    NodeValidationFailure,
                    HasSubstr("ROIs and probabilities of ROIs must same floating-point type"));
}

TEST_F(TypePropExperimentalDetectronTopKROIsV6, input_rois_not_2d) {
    const auto rois_probs = std::make_shared<Parameter>(element::f32, Shape{100});

    OV_EXPECT_THROW(auto op = make_op(std::make_shared<Parameter>(element::f32, Shape{100}), rois_probs, 200),
                    NodeValidationFailure,
                    HasSubstr("The 'input_rois' input is expected to be a 2D. Got: 1"));

    OV_EXPECT_THROW(auto op = make_op(std::make_shared<Parameter>(element::f32, Shape{100, 4, 2}), rois_probs, 200),
                    NodeValidationFailure,
                    HasSubstr("The 'input_rois' input is expected to be a 2D. Got: 3"));
}

TEST_F(TypePropExperimentalDetectronTopKROIsV6, input_rois_second_dim_is_not_4) {
    const auto rois_probs = std::make_shared<Parameter>(element::f32, Shape{100});

    OV_EXPECT_THROW(auto op = make_op(std::make_shared<Parameter>(element::f32, PartialShape{100, 3}), rois_probs, 200),
                    NodeValidationFailure,
                    HasSubstr("The second dimension of 'input_rois' should be 4. Got: 3"));

    OV_EXPECT_THROW(auto op = make_op(std::make_shared<Parameter>(element::f32, PartialShape{100, 5}), rois_probs, 200),
                    NodeValidationFailure,
                    HasSubstr("The second dimension of 'input_rois' should be 4. Got: 5"));

    OV_EXPECT_THROW(
        auto op = make_op(std::make_shared<Parameter>(element::f32, PartialShape{100, {5, 6}}), rois_probs, 200),
        NodeValidationFailure,
        HasSubstr("The second dimension of 'input_rois' should be 4. Got: 5..6"));

    OV_EXPECT_THROW(
        auto op = make_op(std::make_shared<Parameter>(element::f32, PartialShape{100, {1, 3}}), rois_probs, 200),
        NodeValidationFailure,
        HasSubstr("The second dimension of 'input_rois' should be 4. Got: 1..3"));
}

TEST_F(TypePropExperimentalDetectronTopKROIsV6, rois_probabilities_not_1d) {
    const auto input_rois = std::make_shared<Parameter>(element::f32, Shape{200, 4});

    OV_EXPECT_THROW(auto op = make_op(input_rois, std::make_shared<Parameter>(element::f32, Shape{}), 200),
                    NodeValidationFailure,
                    HasSubstr("The 'rois_probs' input is expected to be a 1D. Got: 0"));

    OV_EXPECT_THROW(auto op = make_op(input_rois, std::make_shared<Parameter>(element::f32, Shape{200, 2}), 200),
                    NodeValidationFailure,
                    HasSubstr("The 'rois_probs' input is expected to be a 1D. Got: 2"));
}

TEST_F(TypePropExperimentalDetectronTopKROIsV6, number_of_rois_not_compatible_with_rois_probabilities) {
    OV_EXPECT_THROW(auto op = make_op(std::make_shared<Parameter>(element::f32, PartialShape{200, 4}),
                                      std::make_shared<Parameter>(element::f32, PartialShape{100}),
                                      10),
                    NodeValidationFailure,
                    HasSubstr("Number of rois and number of probabilities should be equal."));

    OV_EXPECT_THROW(auto op = make_op(std::make_shared<Parameter>(element::f32, PartialShape{{10, 20}, 4}),
                                      std::make_shared<Parameter>(element::f32, PartialShape{{21, 25}}),
                                      10),
                    NodeValidationFailure,
                    HasSubstr("Number of rois and number of probabilities should be equal."));

    OV_EXPECT_THROW(auto op = make_op(std::make_shared<Parameter>(element::f32, PartialShape{{10, 20}, 4}),
                                      std::make_shared<Parameter>(element::f32, PartialShape{5}),
                                      10),
                    NodeValidationFailure,
                    HasSubstr("Number of rois and number of probabilities should be equal."));
}
