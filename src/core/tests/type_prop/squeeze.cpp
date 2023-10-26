// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/squeeze.hpp"

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "sequnce_generator.hpp"

using namespace std;
using namespace ov;
using namespace testing;

TEST(type_prop, squeeze_axes_invalid_value) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto axes_node = make_shared<ov::op::v0::Constant>(element::u64, Shape{2}, vector<int64_t>{0, 2});

    OV_EXPECT_THROW(auto s = make_shared<op::v0::Squeeze>(param, axes_node),
                    NodeValidationFailure,
                    HasSubstr("provided axis value is invalid. Only axes of size 1 may be removed."));
}

TEST(type_prop, squeeze_single_input) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, -1, 3, 4});
    auto s = make_shared<op::v0::Squeeze>(param);
    EXPECT_EQ(s->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, squeeze_axes_invalid_rank) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto axes_node = make_shared<ov::op::v0::Constant>(element::i32, Shape{2, 1}, vector<int32_t>{0, 2});

    OV_EXPECT_THROW(auto s = make_shared<op::v0::Squeeze>(param, axes_node),
                    NodeValidationFailure,
                    HasSubstr("Second input (axes) should not be of rank higher than 1."));
}

TEST(type_prop, squeeze_incorrect_negative_axes) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4, 1, 4, 1, 8});
    auto axes_node = make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, vector<int64_t>{-6, -10});

    OV_EXPECT_THROW(auto s = make_shared<op::v0::Squeeze>(param, axes_node),
                    ov::Exception,
                    HasSubstr("Parameter axis -10 out of the tensor rank range"));
}

TEST(type_prop, squeeze_data_static_param_axes_1D_single_elem_static_shape_no_squeezable_dims) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{2, 2, 4});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{1});

    OV_EXPECT_THROW(auto s = make_shared<op::v0::Squeeze>(param, axes_node),
                    NodeValidationFailure,
                    HasSubstr("doesn't contain squeezable dimension"));
}

TEST(type_prop, squeeze_data_static_param_axes_1D_two_elem_static_shape_squeezable_dims_two) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{1, 2, 1, 4});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{2});
    const auto squeeze = std::make_shared<op::v0::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, squeeze_data_static_param_axes_1D_two_elem_static_shape_squeezable_dims_one) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{2, 1, 4});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{2});
    const auto squeeze = std::make_shared<op::v0::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, squeeze_data_static_param_axes_1D_single_elem_static_shape_squeezable_dims_one) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{2, 1, 4});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{1});
    const auto squeeze = std::make_shared<op::v0::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic(2));
}

TEST(type_prop, squeeze_data_static_param_axes_scalar_static_shape_squeezable_dims_one) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{2, 1, 4});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{});
    const auto squeeze = std::make_shared<op::v0::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic(2));
}

TEST(type_prop, squeeze_data_scalar_param_axes_1D_single_elem_static_shape) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{1});
    const auto squeeze = std::make_shared<op::v0::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, squeeze_data_dynamic_param_axes_1D_two_elem_static_shape_squeezable_dims_equal) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{-1, {2, 8}, {1, 3}, {4, -1}});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{2});
    const auto squeeze = std::make_shared<op::v0::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, squeeze_data_static_param_axes_1D_two_elem_static_shape_squeezable_dims_more) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{1, 2, 1, 3, 1});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{2});
    const auto squeeze = std::make_shared<op::v0::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, squeeze_data_static_param_axes_1D_single_elem_static_shape_squeezable_dims_more) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{1, 2, 1, 3, 1});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{1});
    const auto squeeze = std::make_shared<op::v0::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic(4));
}

TEST(type_prop, squeeze_data_static_param_axes_scalar_static_shape_squeezable_dims_more) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{1, 2, 1, 3, 1});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{});
    const auto squeeze = std::make_shared<op::v0::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic(4));
}

TEST(type_prop, squeeze_data_dynamic_param_axes_1D_two_elem_static_shape_squeezable_dims_more) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{-1, {2, 8}, {1, 3}, {4, -1}});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{2});
    const auto squeeze = std::make_shared<op::v0::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, squeeze_data_dynamic_param_axes_1D_single_elem_static_shape_squeezable_dims_more) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{-1, {2, 8}, {1, 3}, {4, -1}});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{1});
    const auto squeeze = std::make_shared<op::v0::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic(3));
}

TEST(type_prop, squeeze_data_dynamic_param_axes_scalar_static_shape_squeezable_dims_more) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{-1, {2, 8}, {1, 3}, {4, -1}});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{});
    const auto squeeze = std::make_shared<op::v0::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic(3));
}

TEST(type_prop, squeeze_data_dyamic_param_axes_1D_two_elem_static_shape_squeezable_dims_one) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{2, -1, 4});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{2});
    const auto squeeze = std::make_shared<op::v0::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, squeeze_data_dynamic_param_axes_1D_three_elem_static_shape_squeezable_dims_two) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{-1, {2, 8}, {1, 3}, {4, -1}});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{3});
    const auto squeeze = std::make_shared<op::v0::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, squeeze_data_dynamic_param_axes_1D_single_elem_static_shape_squeezable_dims_less) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{-1, {2, 8}, {1, 3}, {4, -1}});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{1});
    const auto squeeze = std::make_shared<op::v0::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic(3));
}

using SqueezeTypePropTestParam = std::tuple<PartialShape,          // Input shape
                                            std::vector<int64_t>,  // Squeeze axis
                                            PartialShape           // Expected shape
                                            >;

class SqueezeTest : public WithParamInterface<SqueezeTypePropTestParam>, public UnSqueezeFixture {
protected:
    void SetUp() override {
        std::tie(p_shape, axes, exp_shape) = GetParam();
        UnSqueezeFixture::SetUp();
    }

    std::pair<ov::TensorLabel, ov::TensorLabel> make_in_exp_labels() const {
        ov::TensorLabel in_labels;
        std::generate_n(std::back_inserter(in_labels), p_shape.size(), ov::SeqGen<ov::label_t>(1));

        std::set<int64_t> axes_to_remove;
        if (axes.empty()) {
            for (auto dim = p_shape.begin(); dim != p_shape.end(); ++dim) {
                if (dim->get_max_length() == 1 || exp_shape.rank().is_dynamic()) {
                    axes_to_remove.insert(std::distance(p_shape.begin(), dim));
                }
            }
        } else {
            for (const auto& axis : axes) {
                axes_to_remove.insert(axis < 0 ? axis + p_shape.size() : axis);
            }
        }

        auto rm_iter = axes_to_remove.begin();
        int64_t rm_idx = 0;
        auto exp_labels = in_labels;
        exp_labels.erase(std::remove_if(exp_labels.begin(),
                                        exp_labels.end(),
                                        [&](ov::label_t& label) {
                                            if ((rm_iter != axes_to_remove.end()) && (*rm_iter == rm_idx++)) {
                                                return ++rm_iter, true;
                                            } else {
                                                return false;
                                            }
                                        }),
                         exp_labels.end());

        return {in_labels, exp_labels};
    }

    std::vector<int64_t> axes;
};

const auto static_partial_shapes_test_values =
    Values(std::make_tuple(PartialShape{1}, std::vector<int64_t>{0}, PartialShape{}),
           std::make_tuple(PartialShape{}, std::vector<int64_t>{0}, PartialShape{}),
           std::make_tuple(PartialShape{1, 2}, std::vector<int64_t>{0}, PartialShape{2}),
           std::make_tuple(PartialShape{1, 2}, std::vector<int64_t>{-2}, PartialShape{2}),
           std::make_tuple(PartialShape{1, 2, 1}, std::vector<int64_t>{0}, PartialShape{2, 1}),
           std::make_tuple(PartialShape{1, 2}, std::vector<int64_t>{-2, -2}, PartialShape{2}),
           std::make_tuple(PartialShape{1, 4, 1, 4, 1, 8}, std::vector<int64_t>{0, 2}, PartialShape{4, 4, 1, 8}),
           std::make_tuple(PartialShape{1, 4, 1, 4, 1, 8}, std::vector<int64_t>{-6, -4}, PartialShape{4, 4, 1, 8}));

const auto empty_axes_test_values =
    Values(std::make_tuple(PartialShape{1, 4, 1, 4, 1, 8}, std::vector<int64_t>{}, PartialShape{4, 4, 8}),
           std::make_tuple(PartialShape{Dimension(2, 5), Dimension(3, 4), 6},
                           std::vector<int64_t>{},
                           PartialShape{Dimension(2, 5), Dimension(3, 4), 6}),
           std::make_tuple(PartialShape::dynamic(6), std::vector<int64_t>{}, PartialShape::dynamic()),
           std::make_tuple(PartialShape{Dimension(0, 1)}, std::vector<int64_t>{}, PartialShape::dynamic()),
           std::make_tuple(PartialShape{Dimension::dynamic(), 1, Dimension::dynamic()},
                           std::vector<int64_t>{},
                           PartialShape::dynamic()),
           std::make_tuple(PartialShape::dynamic(), std::vector<int64_t>{}, PartialShape::dynamic()));

INSTANTIATE_TEST_SUITE_P(
    type_prop_shrink_dynamic_shape,
    SqueezeTest,
    Values(std::make_tuple(PartialShape::dynamic(6), std::vector<int64_t>{0, 2}, PartialShape::dynamic(4)),
           std::make_tuple(PartialShape{Dimension::dynamic(), 1, Dimension::dynamic()},
                           std::vector<int64_t>{0, 2},
                           PartialShape{1}),
           std::make_tuple(PartialShape::dynamic(), std::vector<int64_t>{0, 2}, PartialShape::dynamic())),
    PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(type_prop_shrink_shape,
                         SqueezeTest,
                         static_partial_shapes_test_values,
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(type_prop_shrink_shape_default_axes,
                         SqueezeTest,
                         empty_axes_test_values,
                         PrintToStringParamName());

TEST_P(SqueezeTest, partial_shape_dimension_propagation_const_axis_i32) {
    const auto axes_node = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{axes.size()}, axes);
    const auto squeeze = std::make_shared<op::v0::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), exp_shape);
}

TEST_P(SqueezeTest, partial_shape_dimension_propagation_parameter_axes_no_data) {
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{Shape{axes.size()}});
    const auto squeeze = std::make_shared<op::v0::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_TRUE(squeeze->get_output_partial_shape(0).compatible(exp_shape));
}

TEST_P(SqueezeTest, partial_shape_dimension_propagation_dynamic_axes) {
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape::dynamic());
    const auto squeeze = std::make_shared<op::v0::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST_P(SqueezeTest, labels_propagation) {
    if (p_shape.rank().is_dynamic()) {
        GTEST_SKIP() << "No dimension to set label";
    }
    ov::TensorLabel in_labels, exp_labels;
    std::tie(in_labels, exp_labels) = make_in_exp_labels();

    set_shape_labels(p_shape, in_labels);
    param = make_shared<ov::op::v0::Parameter>(element::f32, p_shape);

    const auto axes_node = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{axes.size()}, axes);
    const auto squeeze = std::make_shared<op::v0::Squeeze>(param, axes_node);

    EXPECT_EQ(get_shape_labels(squeeze->get_output_partial_shape(0)), exp_labels);
}

using SqueezeShapeTests = SqueezeTest;

INSTANTIATE_TEST_SUITE_P(type_prop_shrink_shape_no_axes,
                         SqueezeShapeTests,
                         static_partial_shapes_test_values,
                         PrintToStringParamName());

TEST_P(SqueezeShapeTests, shape_dimension_propagation_const_axis_i64) {
    param = std::make_shared<ov::op::v0::Parameter>(element::f64, p_shape.to_shape());
    const auto axes_node = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{axes.size()}, axes);
    const auto squeeze = std::make_shared<op::v0::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f64);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), exp_shape.to_shape());
}

using SqueezeNoAxesTest = SqueezeTest;

INSTANTIATE_TEST_SUITE_P(type_prop_shrink_shape_no_axes,
                         SqueezeNoAxesTest,
                         empty_axes_test_values,
                         PrintToStringParamName());

TEST_P(SqueezeNoAxesTest, partial_shape_dimension_propagation_no_axes) {
    const auto squeeze = std::make_shared<op::v0::Squeeze>(param);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), exp_shape);
}

using SqueezeScalarAxisTest = SqueezeTest;

INSTANTIATE_TEST_SUITE_P(
    type_prop_shrink_shape_no_axes,
    SqueezeScalarAxisTest,
    Values(std::make_tuple(PartialShape{1, 2}, std::vector<int64_t>{0}, PartialShape{2}),
           std::make_tuple(PartialShape{3, 1, 2}, std::vector<int64_t>{1}, PartialShape{3, 2}),
           std::make_tuple(PartialShape{3, 1, 2, 1, 1, 5}, std::vector<int64_t>{4}, PartialShape{3, 1, 2, 1, 5})),
    PrintToStringParamName());

TEST_P(SqueezeScalarAxisTest, axis_value_as_vector) {
    const auto axes_node = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, axes);
    const auto squeeze = std::make_shared<op::v0::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), exp_shape);
}

TEST_P(SqueezeScalarAxisTest, axis_value_as_integer) {
    const auto axes_node = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, axes.front());
    const auto squeeze = std::make_shared<op::v0::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), exp_shape);
}

using SqueezeBoundTest = UnSqueezeBoundTest;

INSTANTIATE_TEST_SUITE_P(
    type_prop_bounds_propagate,
    SqueezeBoundTest,
    Values(std::make_tuple(PartialShape::dynamic(6), PartialShape::dynamic(1)),
           std::make_tuple(PartialShape{Dimension(-1)}, PartialShape{Dimension(-1)}),
           std::make_tuple(PartialShape{Dimension::dynamic(), 8}, PartialShape{Dimension::dynamic()}),
           std::make_tuple(PartialShape{Dimension(4, 8), Dimension::dynamic()}, PartialShape{Dimension(4, 8)}),
           std::make_tuple(PartialShape{Dimension(20, -1), Dimension::dynamic()}, PartialShape::dynamic(1)),
           std::make_tuple(PartialShape{Dimension(-1, 5), Dimension::dynamic()}, PartialShape{Dimension(-1, 5)}),
           std::make_tuple(PartialShape{15}, PartialShape{15}),
           std::make_tuple(PartialShape{2, 6}, PartialShape{2})),
    PrintToStringParamName());

/**
 * \brief Check label and dynamic value propagation.
 *
 * Test use evaluate label, lower/upper.
 */
TEST_P(SqueezeBoundTest, propagate_label_and_dynamic_value) {
    PartialShape labeled_shape = PartialShape{p_shape};

    std::generate_n(std::back_inserter(in_labels), labeled_shape.size(), ov::SeqGen<ov::label_t>(1));
    set_shape_labels(labeled_shape, in_labels);

    constexpr auto et = element::i64;
    const auto labeled_param = std::make_shared<ov::op::v0::Parameter>(et, labeled_shape);
    const auto labeled_shape_of = std::make_shared<op::v0::ShapeOf>(labeled_param);

    const auto zero = std::vector<int64_t>{0};
    const auto axis = std::make_shared<op::v0::Constant>(et, Shape{}, zero);
    const auto indices = std::make_shared<op::v0::Constant>(et, Shape{}, zero);
    const auto gather = std::make_shared<op::v7::Gather>(labeled_shape_of, indices, axis);
    const auto axis_1 = std::make_shared<op::v0::Constant>(et, Shape{2}, std::vector<int64_t>{0, 1});
    const auto unsqueeze = std::make_shared<op::v0::Unsqueeze>(gather, axis_1);
    const auto squeeze = std::make_shared<op::v0::Squeeze>(unsqueeze, axis);

    const auto bc = std::make_shared<op::v3::Broadcast>(param, squeeze);

    EXPECT_EQ(bc->get_output_partial_shape(0), exp_shape);
    const auto labels = get_shape_labels(bc->get_output_partial_shape(0));
    EXPECT_THAT(labels, ElementsAre(in_labels.front()));
}
