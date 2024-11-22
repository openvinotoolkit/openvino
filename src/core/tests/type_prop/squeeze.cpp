// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/squeeze.hpp"

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/broadcast.hpp"
#include "sequence_generator.hpp"

using namespace std;
using namespace ov;
using namespace testing;

namespace {

template <typename TSqueeze>
class SqueezelOperator : public TypePropOpTest<TSqueeze> {};

using SqueezeTypes = ::testing::Types<op::v0::Squeeze, op::v15::Squeeze>;

TYPED_TEST_SUITE(SqueezelOperator, SqueezeTypes);

TYPED_TEST(SqueezelOperator, squeeze_axes_invalid_value) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto axes_node = make_shared<ov::op::v0::Constant>(element::u64, Shape{2}, vector<int64_t>{0, 2});
    const auto squeeze = this->make_op(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), (PartialShape{2, 3, 4}));
}

TYPED_TEST(SqueezelOperator, squeeze_single_input) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, -1, 3, 4});
    const auto squeeze = this->make_op(param);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic());
}

TYPED_TEST(SqueezelOperator, squeeze_axes_invalid_rank) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto axes_node = make_shared<ov::op::v0::Constant>(element::i32, Shape{2, 1}, vector<int32_t>{0, 2});

    OV_EXPECT_THROW(const auto squeeze = this->make_op(param, axes_node),
                    NodeValidationFailure,
                    HasSubstr("Second input (axes) should not be of rank higher than 1."));
}

TYPED_TEST(SqueezelOperator, squeeze_incorrect_negative_axes) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4, 1, 4, 1, 8});
    auto axes_node = make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, vector<int64_t>{-6, -10});

    OV_EXPECT_THROW(const auto squeeze = this->make_op(param, axes_node),
                    ov::Exception,
                    HasSubstr("Axis -10 out of the tensor rank range"));
}

TYPED_TEST(SqueezelOperator, squeeze_data_static_param_axes_1D_single_elem_static_shape_no_squeezable_dims) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{2, 2, 4});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{1});
    const auto squeeze = this->make_op(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), (PartialShape{2, 2, 4}));
}

TYPED_TEST(SqueezelOperator, squeeze_data_static_param_axes_1D_two_elem_static_shape_squeezable_dims_two) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{1, 2, 1, 4});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{2});
    const auto squeeze = this->make_op(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic());
}

TYPED_TEST(SqueezelOperator, squeeze_data_static_param_axes_1D_two_elem_static_shape_squeezable_dims_one) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{2, 1, 4});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{2});
    const auto squeeze = this->make_op(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(TypePropSqueezelOperatorV0, squeeze_data_static_param_axes_1D_single_elem_static_shape_squeezable_dims_one) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{2, 1, 4});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{1});
    const auto squeeze = std::make_shared<ov::op::v0::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic(2));
}

TEST(TypePropSqueezelOperatorV15, squeeze_data_static_param_axes_1D_single_elem_static_shape_squeezable_dims_one) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{2, 1, 4});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{1});
    const auto squeeze = std::make_shared<ov::op::v15::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(TypePropSqueezelOperatorV0, squeeze_data_static_param_axes_scalar_static_shape_squeezable_dims_one) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{2, 1, 4});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{});
    const auto squeeze = std::make_shared<ov::op::v0::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic(2));
}

TEST(TypePropSqueezelOperatorV15, squeeze_data_static_param_axes_scalar_static_shape_squeezable_dims_one) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{2, 1, 4});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{});
    const auto squeeze = std::make_shared<ov::op::v15::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic());
}

TYPED_TEST(SqueezelOperator, squeeze_data_scalar_param_axes_1D_single_elem_static_shape) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{1});
    const auto squeeze = this->make_op(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic());
}

TYPED_TEST(SqueezelOperator, squeeze_data_dynamic_param_axes_1D_two_elem_static_shape_squeezable_dims_equal) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{-1, {2, 8}, {1, 3}, {4, -1}});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{2});
    const auto squeeze = this->make_op(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic());
}

TYPED_TEST(SqueezelOperator, squeeze_data_static_param_axes_1D_two_elem_static_shape_squeezable_dims_more) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{1, 2, 1, 3, 1});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{2});
    const auto squeeze = this->make_op(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(TypePropSqueezelOperatorV0, squeeze_data_static_param_axes_1D_single_elem_static_shape_squeezable_dims_more) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{1, 2, 1, 3, 1});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{1});
    const auto squeeze = std::make_shared<ov::op::v0::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic(4));
}

TEST(TypePropSqueezelOperatorV15, squeeze_data_static_param_axes_1D_single_elem_static_shape_squeezable_dims_more) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{1, 2, 1, 3, 1});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{1});
    const auto squeeze = std::make_shared<ov::op::v15::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(TypePropSqueezelOperatorV0, squeeze_data_static_param_axes_scalar_static_shape_squeezable_dims_more) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{1, 2, 1, 3, 1});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{});
    const auto squeeze = std::make_shared<ov::op::v0::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic(4));
}

TEST(TypePropSqueezelOperatorV15, squeeze_data_static_param_axes_scalar_static_shape_squeezable_dims_more) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{1, 2, 1, 3, 1});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{});
    const auto squeeze = std::make_shared<ov::op::v15::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic());
}

TYPED_TEST(SqueezelOperator, squeeze_data_dynamic_param_axes_1D_two_elem_static_shape_squeezable_dims_more) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{-1, {2, 8}, {1, 3}, {4, -1}});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{2});
    const auto squeeze = this->make_op(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(TypePropSqueezelOperatorV0, squeeze_data_dynamic_param_axes_1D_single_elem_static_shape_squeezable_dims_more) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{-1, {2, 8}, {1, 3}, {4, -1}});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{1});
    const auto squeeze = std::make_shared<ov::op::v0::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic(3));
}

TEST(TypePropSqueezelOperatorV15, squeeze_data_dynamic_param_axes_1D_single_elem_static_shape_squeezable_dims_more) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{-1, {2, 8}, {1, 3}, {4, -1}});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{1});
    const auto squeeze = std::make_shared<ov::op::v15::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(TypePropSqueezelOperatorV0, squeeze_data_dynamic_param_axes_scalar_static_shape_squeezable_dims_more) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{-1, {2, 8}, {1, 3}, {4, -1}});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{});
    const auto squeeze = std::make_shared<ov::op::v0::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic(3));
}

TEST(TypePropSqueezelOperatorV15, squeeze_data_dynamic_param_axes_scalar_static_shape_squeezable_dims_more) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{-1, {2, 8}, {1, 3}, {4, -1}});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{});
    const auto squeeze = std::make_shared<ov::op::v15::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic());
}

TYPED_TEST(SqueezelOperator, squeeze_data_dyamic_param_axes_1D_two_elem_static_shape_squeezable_dims_one) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{2, -1, 4});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{2});
    const auto squeeze = this->make_op(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic());
}

TYPED_TEST(SqueezelOperator, squeeze_data_dynamic_param_axes_1D_three_elem_static_shape_squeezable_dims_two) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{-1, {2, 8}, {1, 3}, {4, -1}});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{3});
    const auto squeeze = this->make_op(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(TypePropSqueezelOperatorV0, squeeze_data_dynamic_param_axes_1D_single_elem_static_shape_squeezable_dims_less) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{-1, {2, 8}, {1, 3}, {4, -1}});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{1});
    const auto squeeze = std::make_shared<ov::op::v0::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic(3));
}

TEST(TypePropSqueezelOperatorV15, squeeze_data_dynamic_param_axes_1D_single_elem_static_shape_squeezable_dims_less) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, PartialShape{-1, {2, 8}, {1, 3}, {4, -1}});
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{1});
    const auto squeeze = std::make_shared<ov::op::v15::Squeeze>(param, axes_node);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic());
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

    std::pair<ov::TensorSymbol, ov::TensorSymbol> make_in_exp_symbols() const {
        ov::TensorSymbol in_symbols;
        for (size_t i = 0; i < p_shape.size(); ++i)
            in_symbols.push_back(std::make_shared<Symbol>());

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
        auto exp_symbols = in_symbols;
        exp_symbols.erase(std::remove_if(exp_symbols.begin(),
                                         exp_symbols.end(),
                                         [&](shared_ptr<Symbol> symbol) {
                                             if ((rm_iter != axes_to_remove.end()) && (*rm_iter == rm_idx++)) {
                                                 return ++rm_iter, true;
                                             } else {
                                                 return false;
                                             }
                                         }),
                          exp_symbols.end());

        return {in_symbols, exp_symbols};
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
    {
        const auto squeeze = std::make_shared<op::v0::Squeeze>(param, axes_node);
        EXPECT_EQ(squeeze->get_element_type(), element::f32);
        EXPECT_EQ(squeeze->get_output_partial_shape(0), exp_shape);
    }
    {
        const auto squeeze = std::make_shared<op::v15::Squeeze>(param, axes_node);
        EXPECT_EQ(squeeze->get_element_type(), element::f32);
        EXPECT_EQ(squeeze->get_output_partial_shape(0), exp_shape);
    }
}

TEST_P(SqueezeTest, partial_shape_dimension_propagation_parameter_axes_no_data) {
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape{Shape{axes.size()}});
    {
        const auto squeeze = std::make_shared<op::v0::Squeeze>(param, axes_node);
        EXPECT_EQ(squeeze->get_element_type(), element::f32);
        EXPECT_TRUE(squeeze->get_output_partial_shape(0).compatible(exp_shape));
    }
    {
        const auto squeeze = std::make_shared<op::v15::Squeeze>(param, axes_node);
        EXPECT_EQ(squeeze->get_element_type(), element::f32);
        EXPECT_TRUE(squeeze->get_output_partial_shape(0).compatible(exp_shape));
    }
}

TEST_P(SqueezeTest, partial_shape_dimension_propagation_dynamic_axes) {
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u64, PartialShape::dynamic());
    {
        const auto squeeze = std::make_shared<op::v0::Squeeze>(param, axes_node);
        EXPECT_EQ(squeeze->get_element_type(), element::f32);
        EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic());
    }
    {
        const auto squeeze = std::make_shared<op::v15::Squeeze>(param, axes_node);
        EXPECT_EQ(squeeze->get_element_type(), element::f32);
        EXPECT_EQ(squeeze->get_output_partial_shape(0), PartialShape::dynamic());
    }
}

TEST_P(SqueezeTest, symbols_propagation) {
    if (p_shape.rank().is_dynamic()) {
        GTEST_SKIP() << "No dimension to set symbol";
    }
    ov::TensorSymbol in_symbols, exp_symbols;
    std::tie(in_symbols, exp_symbols) = make_in_exp_symbols();

    set_shape_symbols(p_shape, in_symbols);
    param = make_shared<ov::op::v0::Parameter>(element::f32, p_shape);

    const auto axes_node = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{axes.size()}, axes);
    {
        const auto squeeze = std::make_shared<op::v0::Squeeze>(param, axes_node);
        EXPECT_EQ(get_shape_symbols(squeeze->get_output_partial_shape(0)), exp_symbols);
    }
    {
        const auto squeeze = std::make_shared<op::v15::Squeeze>(param, axes_node);
        EXPECT_EQ(get_shape_symbols(squeeze->get_output_partial_shape(0)), exp_symbols);
    }
}

using SqueezeShapeTests = SqueezeTest;

INSTANTIATE_TEST_SUITE_P(type_prop_shrink_shape_no_axes,
                         SqueezeShapeTests,
                         static_partial_shapes_test_values,
                         PrintToStringParamName());

TEST_P(SqueezeShapeTests, shape_dimension_propagation_const_axis_i64) {
    param = std::make_shared<ov::op::v0::Parameter>(element::f64, p_shape.to_shape());
    const auto axes_node = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{axes.size()}, axes);
    {
        const auto squeeze = std::make_shared<op::v0::Squeeze>(param, axes_node);
        EXPECT_EQ(squeeze->get_element_type(), element::f64);
        EXPECT_EQ(squeeze->get_output_partial_shape(0), exp_shape.to_shape());
    }
    {
        const auto squeeze = std::make_shared<op::v15::Squeeze>(param, axes_node);
        EXPECT_EQ(squeeze->get_element_type(), element::f64);
        EXPECT_EQ(squeeze->get_output_partial_shape(0), exp_shape.to_shape());
    }
}

using SqueezeNoAxesTest = SqueezeTest;

INSTANTIATE_TEST_SUITE_P(type_prop_shrink_shape_no_axes,
                         SqueezeNoAxesTest,
                         empty_axes_test_values,
                         PrintToStringParamName());

TEST_P(SqueezeNoAxesTest, partial_shape_dimension_propagation_no_axes) {
    {
        const auto squeeze = std::make_shared<op::v0::Squeeze>(param);
        EXPECT_EQ(squeeze->get_element_type(), element::f32);
        EXPECT_EQ(squeeze->get_output_partial_shape(0), exp_shape);
    }
    {
        const auto squeeze = std::make_shared<op::v15::Squeeze>(param);
        EXPECT_EQ(squeeze->get_element_type(), element::f32);
        EXPECT_EQ(squeeze->get_output_partial_shape(0), exp_shape);
    }
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
    {
        const auto squeeze = std::make_shared<op::v0::Squeeze>(param, axes_node);
        EXPECT_EQ(squeeze->get_element_type(), element::f32);
        EXPECT_EQ(squeeze->get_output_partial_shape(0), exp_shape);
    }
    {
        const auto squeeze = std::make_shared<op::v15::Squeeze>(param, axes_node);
        EXPECT_EQ(squeeze->get_element_type(), element::f32);
        EXPECT_EQ(squeeze->get_output_partial_shape(0), exp_shape);
    }
}

TEST_P(SqueezeScalarAxisTest, axis_value_as_integer) {
    const auto axes_node = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, axes.front());
    {
        const auto squeeze = std::make_shared<op::v0::Squeeze>(param, axes_node);
        EXPECT_EQ(squeeze->get_element_type(), element::f32);
        EXPECT_EQ(squeeze->get_output_partial_shape(0), exp_shape);
    }
    {
        const auto squeeze = std::make_shared<op::v15::Squeeze>(param, axes_node);
        EXPECT_EQ(squeeze->get_element_type(), element::f32);
        EXPECT_EQ(squeeze->get_output_partial_shape(0), exp_shape);
    }
}

using SqueezeBoundTest = UnSqueezeBoundTest;

const auto test_values_in =
    Values(std::make_tuple(PartialShape::dynamic(6), PartialShape::dynamic(1)),
           std::make_tuple(PartialShape{Dimension(-1)}, PartialShape{Dimension(-1)}),
           std::make_tuple(PartialShape{Dimension::dynamic(), 8}, PartialShape{Dimension::dynamic()}),
           std::make_tuple(PartialShape{Dimension(4, 8), Dimension::dynamic()}, PartialShape{Dimension(4, 8)}),
           std::make_tuple(PartialShape{Dimension(20, -1), Dimension::dynamic()}, PartialShape{{20, -1}}),
           std::make_tuple(PartialShape{Dimension(-1, 5), Dimension::dynamic()}, PartialShape{Dimension(-1, 5)}),
           std::make_tuple(PartialShape{15}, PartialShape{15}),
           std::make_tuple(PartialShape{2, 6}, PartialShape{2}));

INSTANTIATE_TEST_SUITE_P(type_prop_bounds_propagate, SqueezeBoundTest, test_values_in, PrintToStringParamName());

/**
 * \brief Check symbol and dynamic value propagation.
 *
 * Test use evaluate symbol, lower/upper.
 */
TEST_P(SqueezeBoundTest, propagate_symbol_and_dynamic_value_squeeze_v0) {
    PartialShape symboled_shape = PartialShape{p_shape};

    in_symbols = set_shape_symbols(symboled_shape);

    const auto squeeze = create_squeeze<op::v0::Squeeze>(symboled_shape);
    const auto bc = std::make_shared<op::v3::Broadcast>(param, squeeze);

    EXPECT_EQ(bc->get_output_partial_shape(0), exp_shape);
    const auto symbols = get_shape_symbols(bc->get_output_partial_shape(0));
    EXPECT_THAT(symbols, ElementsAre(in_symbols.front()));
}

/**
 * \brief Check symbol and dynamic value propagation.
 *
 * Test use evaluate symbol, lower/upper.
 */
TEST_P(SqueezeBoundTest, propagate_symbol_and_dynamic_value_squeeze_v15) {
    PartialShape symboled_shape = PartialShape{p_shape};

    in_symbols = set_shape_symbols(symboled_shape);

    const auto squeeze = create_squeeze<op::v15::Squeeze>(symboled_shape);
    const auto bc = std::make_shared<op::v3::Broadcast>(param, squeeze);

    EXPECT_EQ(bc->get_output_partial_shape(0), exp_shape);
    const auto symbols = get_shape_symbols(bc->get_output_partial_shape(0));
    EXPECT_THAT(symbols, ElementsAre(in_symbols.front()));
}

using SqueezeAxesDynamicRankTestParam = decltype(std::tuple_cat(SqueezeTypePropTestParam{}, std::make_tuple(false)));
class SqueezeAxesDynamicRank : public ::testing::TestWithParam<SqueezeAxesDynamicRankTestParam> {
protected:
    ov::PartialShape p_shape{}, exp_shape{};
    std::vector<int64_t> axes{};
    bool allow_axis_skip{};
};

INSTANTIATE_TEST_SUITE_P(
    SqueezeAxesDynamicRankTests,
    SqueezeAxesDynamicRank,
    ::testing::Values(
        std::make_tuple(PartialShape{1, 2, -1, 4}, std::vector<int64_t>{}, PartialShape::dynamic(), false),
        std::make_tuple(PartialShape{1, 2, -1, 4}, std::vector<int64_t>{}, PartialShape::dynamic(), true),

        std::make_tuple(PartialShape{1, 2, -1, 4}, std::vector<int64_t>{0}, PartialShape{2, -1, 4}, false),
        std::make_tuple(PartialShape{1, 2, -1, 4}, std::vector<int64_t>{0}, PartialShape{2, -1, 4}, true),

        std::make_tuple(PartialShape{1, 2, -1, 4}, std::vector<int64_t>{2}, PartialShape{1, 2, 4}, false),
        std::make_tuple(PartialShape{1, 2, -1, 4}, std::vector<int64_t>{2}, PartialShape::dynamic(), true),

        std::make_tuple(PartialShape{1, 2, -1, 4}, std::vector<int64_t>{0, 2}, PartialShape{2, 4}, false),
        std::make_tuple(PartialShape{1, 2, -1, 4}, std::vector<int64_t>{0, 2}, PartialShape::dynamic(), true),

        std::make_tuple(PartialShape{1, 2, -1, 4}, std::vector<int64_t>{1}, PartialShape{1, 2, -1, 4}, false),
        std::make_tuple(PartialShape{1, 2, -1, 4}, std::vector<int64_t>{1}, PartialShape{1, 2, -1, 4}, true),

        std::make_tuple(PartialShape{2, 4}, std::vector<int64_t>{1}, PartialShape{2, 4}, false),
        std::make_tuple(PartialShape{2, 4}, std::vector<int64_t>{1}, PartialShape{2, 4}, true),

        std::make_tuple(PartialShape{2, {3, 5}}, std::vector<int64_t>{}, PartialShape{2, {3, 5}}, false),
        std::make_tuple(PartialShape{2, {3, 5}}, std::vector<int64_t>{}, PartialShape{2, {3, 5}}, true),

        std::make_tuple(PartialShape{1, 2, -1}, std::vector<int64_t>{0, 1}, PartialShape{2, -1}, false),
        std::make_tuple(PartialShape{1, 2, -1}, std::vector<int64_t>{0, 1}, PartialShape{2, -1}, true),

        std::make_tuple(PartialShape{1, 2, -1}, std::vector<int64_t>{1}, PartialShape{1, 2, -1}, false),
        std::make_tuple(PartialShape{1, 2, -1}, std::vector<int64_t>{1}, PartialShape{1, 2, -1}, true),

        std::make_tuple(PartialShape{1, 1, -1}, std::vector<int64_t>{0, 1}, PartialShape{-1}, false),
        std::make_tuple(PartialShape{1, 1, -1}, std::vector<int64_t>{0, 1}, PartialShape{-1}, true),

        std::make_tuple(PartialShape{1, 1, -1}, std::vector<int64_t>{1}, PartialShape{1, -1}, false),
        std::make_tuple(PartialShape{1, 1, -1}, std::vector<int64_t>{1}, PartialShape{1, -1}, true),

        std::make_tuple(PartialShape{1, 2, 3}, std::vector<int64_t>{}, PartialShape{2, 3}, false),
        std::make_tuple(PartialShape{1, 2, 3}, std::vector<int64_t>{}, PartialShape{2, 3}, true)));

TEST_P(SqueezeAxesDynamicRank, squeeze_axes_dynamic_rank_param) {
    const auto& params = GetParam();
    p_shape = std::get<0>(params);
    axes = std::get<1>(params);
    exp_shape = std::get<2>(params);
    allow_axis_skip = std::get<3>(params);

    auto param = make_shared<ov::op::v0::Parameter>(element::f32, p_shape);
    auto axes_node = make_shared<ov::op::v0::Constant>(element::u64, Shape{axes.size()}, axes);
    const auto squeeze = std::make_shared<op::v15::Squeeze>(param, axes_node, allow_axis_skip);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), exp_shape);
    EXPECT_EQ(squeeze->get_allow_axis_skip(), allow_axis_skip);
}

TEST(SqueezeDynamicAxis, squeeze_dynamic_non_const_single_axis) {
    auto p_shape = PartialShape{1, 2, -1, 4};
    auto exp_shape = PartialShape::dynamic();
    auto allow_axis_skip = true;

    auto param = make_shared<ov::op::v0::Parameter>(element::f32, p_shape);
    auto axes_node = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1});
    const auto squeeze = std::make_shared<op::v15::Squeeze>(param, axes_node, allow_axis_skip);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), exp_shape);
    EXPECT_EQ(squeeze->get_allow_axis_skip(), allow_axis_skip);
}

TEST(SqueezeDynamicAxis, squeeze_dynamic_non_const_axes) {
    auto p_shape = PartialShape{1, 2, -1, 4};
    auto exp_shape = PartialShape::dynamic();
    auto allow_axis_skip = true;

    auto param = make_shared<ov::op::v0::Parameter>(element::f32, p_shape);
    auto axes_node = make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto squeeze = std::make_shared<op::v15::Squeeze>(param, axes_node, allow_axis_skip);

    EXPECT_EQ(squeeze->get_element_type(), element::f32);
    EXPECT_EQ(squeeze->get_output_partial_shape(0), exp_shape);
    EXPECT_EQ(squeeze->get_allow_axis_skip(), allow_axis_skip);
}

}  // namespace
