// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/transpose.hpp"

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/shape_of.hpp"
#include "sequence_generator.hpp"

using namespace std;
using namespace ov;
using namespace testing;
using namespace ov::op;

TEST(type_prop, transpose_arg_static_input_order_static_ok) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto input_order = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});

    auto r = make_shared<op::v1::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_EQ(r->get_output_partial_shape(0), PartialShape::dynamic(4));
}

TEST(type_prop, transpose_arg_static_input_order_constant_ok) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto input_order = ov::op::v0::Constant::create(element::i64, Shape{4}, vector<int64_t>{2, 1, 0, 3});

    auto r = make_shared<op::v1::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_EQ(r->get_output_partial_shape(0), (PartialShape{6, 4, 2, 8}));
}

TEST(type_prop, transpose_arg_static_input_order_constant_invalid_perm) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto input_order = ov::op::v0::Constant::create(element::i64, Shape{4}, vector<int64_t>{2, 9, 0, 3});

    try {
        auto r = make_shared<op::v1::Transpose>(arg, input_order);
        FAIL() << "Did not detect invalid permutation";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Permutation AxisVector{2, 9, 0, 3} is not valid for input shape"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, transpose_with_not_unique_order) {
    const auto order = std::vector<size_t>{1, 0, 1};
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4, 300});
    auto input_order = make_shared<ov::op::v0::Constant>(element::i64, Shape{order.size()}, order);

    try {
        auto r = make_shared<op::v1::Transpose>(arg, input_order);
        FAIL() << "Did not detect invalid permutation";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Permutation AxisVector{1, 0, 1} is not valid for input shape"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, transpose_arg_rank_static_dynamic_input_order_static_ok) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32,
                                                  PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto input_order = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});

    auto r = make_shared<op::v1::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_EQ(r->get_output_partial_shape(0), PartialShape::dynamic(4));
}

TEST(type_prop, transpose_arg_static_input_order_rank_static_dynamic_ok) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto input_order = make_shared<ov::op::v0::Parameter>(element::i64, PartialShape{Dimension::dynamic()});

    auto r = make_shared<op::v1::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_EQ(r->get_output_partial_shape(0), PartialShape::dynamic(4));
}

TEST(type_prop, transpose_arg_rank_static_dynamic_input_order_rank_static_dynamic_ok) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32,
                                                  PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto input_order = make_shared<ov::op::v0::Parameter>(element::i64, PartialShape{Dimension::dynamic()});

    auto r = make_shared<op::v1::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_EQ(r->get_output_partial_shape(0), PartialShape::dynamic(4));
}

TEST(type_prop, transpose_arg_rank_dynamic_input_order_rank_static_dynamic_ok) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto input_order = make_shared<ov::op::v0::Parameter>(element::i64, PartialShape{Dimension::dynamic()});

    auto r = make_shared<op::v1::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_EQ(r->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, transpose_arg_rank_dynamic_input_order_rank_dynamic_ok) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto input_order = make_shared<ov::op::v0::Parameter>(element::i64, PartialShape::dynamic());

    auto r = make_shared<op::v1::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_EQ(r->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, transpose_arg_rank_static_dynamic_input_order_rank_dynamic_ok) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32,
                                                  PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto input_order = make_shared<ov::op::v0::Parameter>(element::i64, PartialShape::dynamic());

    auto r = make_shared<op::v1::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_EQ(r->get_output_partial_shape(0), PartialShape::dynamic(4));
}

TEST(type_prop, transpose_arg_rank_dynamic_input_order_const_ok) {
    const auto axes_order = std::vector<int64_t>{1, 3, 0, 2};
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto input_order = ov::op::v0::Constant::create(element::i64, Shape{axes_order.size()}, axes_order);

    auto r = make_shared<op::v1::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_EQ(r->get_output_partial_shape(0), PartialShape::dynamic(axes_order.size()));
}

TEST(type_prop, transpose_dynamic_interval_input_data) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension(4, 6), Dimension(2, 3), 8});
    auto input_order = make_shared<ov::op::v0::Parameter>(element::i64, Shape{3});

    auto r = make_shared<op::v1::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_EQ(r->get_output_partial_shape(0), PartialShape::dynamic(3));
}

TEST(type_prop, transpose_arg_static_input_order_static_input_order_not_vector) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{2, 4, 6, 8});
    auto input_order = make_shared<ov::op::v0::Parameter>(element::i64, PartialShape{2, 2});

    try {
        auto r = make_shared<op::v1::Transpose>(arg, input_order);
        FAIL() << "Did not detect input order not vector";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input order must be a vector."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, transpose_arg_static_input_order_rank_static_dynamic_input_order_not_vector) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{2, 4, 6, 8});
    auto input_order = make_shared<ov::op::v0::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});

    try {
        auto r = make_shared<op::v1::Transpose>(arg, input_order);
        FAIL() << "Did not detect input order not vector";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input order must be a vector."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, transpose_arg_static_input_order_static_input_order_wrong_size) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{2, 4, 6, 8});
    auto input_order = make_shared<ov::op::v0::Parameter>(element::i64, PartialShape{5});

    try {
        auto r = make_shared<op::v1::Transpose>(arg, input_order);
        FAIL() << "Did not detect input order wrong size";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input order must have shape [n], where n is the rank of arg."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, transpose_arg_rank_static_dynamic_input_order_static_input_order_not_vector) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32,
                                                  PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto input_order = make_shared<ov::op::v0::Parameter>(element::i64, PartialShape{2, 2});

    try {
        auto r = make_shared<op::v1::Transpose>(arg, input_order);
        FAIL() << "Did not detect input order not vector";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input order must be a vector."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, transpose_arg_rank_static_dynamic_input_order_rank_static_dynamic_input_order_not_vector) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32,
                                                  PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto input_order = make_shared<ov::op::v0::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});

    try {
        auto r = make_shared<op::v1::Transpose>(arg, input_order);
        FAIL() << "Did not detect input order not vector";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input order must be a vector."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, transpose_arg_rank_dynamic_input_order_rank_static_dynamic_input_order_not_vector) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto input_order = make_shared<ov::op::v0::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});

    try {
        auto r = make_shared<op::v1::Transpose>(arg, input_order);
        FAIL() << "Did not detect input order not vector";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input order must be a vector."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, transpose_input_order_et_dynamic_ok) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto input_order = make_shared<ov::op::v0::Parameter>(element::dynamic, Shape{4});

    auto r = make_shared<op::v1::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_EQ(r->get_output_partial_shape(0), PartialShape::dynamic(4));
}

TEST(type_prop, transpose_input_order_et_wrong) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto input_order = make_shared<ov::op::v0::Parameter>(element::boolean, Shape{4});

    try {
        auto r = make_shared<op::v1::Transpose>(arg, input_order);
        FAIL() << "Did not detect input element type not i64";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input order must have an integral number element type."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, transpose_with_empty_order) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 300});
    auto input_order = make_shared<ov::op::v0::Constant>(element::i64, Shape({0}), std::vector<size_t>());

    auto r = make_shared<op::v1::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape({300, 1})));
    EXPECT_EQ(r->get_output_partial_shape(0), (PartialShape{300, 1}));
}

/** \brief Transpose with order as parameter shape dimensions. */
TEST(type_prop, transpose_order_as_parameter_shape) {
    const auto arg = make_shared<v0::Parameter>(element::f32, PartialShape{Dimension(2, 8), Dimension(4, 16), 6});

    const auto param = make_shared<v0::Parameter>(element::i64, PartialShape{2, 0, 1});
    const auto shape_of = make_shared<v3::ShapeOf>(param);
    // order after gather [1, 2, 0]
    const auto gather = make_shared<v1::Gather>(shape_of,
                                                ov::op::v0::Constant::create(element::i64, {3}, {2, 0, 1}),
                                                ov::op::v0::Constant::create(element::i64, {}, {0}));

    const auto r = make_shared<v1::Transpose>(arg, gather);

    EXPECT_EQ(r->get_output_element_type(v1::Transpose::ARG_T), element::f32);
    EXPECT_EQ(r->get_output_partial_shape(v1::Transpose::ARG_T), PartialShape({Dimension(4, 16), 6, Dimension(2, 8)}));
}

/** \brief Transpose with order as paramater shape dimensions after multiple transformations. */
TEST(type_prop, transpose_order_as_parameter_shape_after_transformation) {
    const auto arg = make_shared<v0::Parameter>(element::f32, PartialShape{Dimension(2, 8), Dimension(4, 16), 6});

    const auto param = make_shared<v0::Parameter>(element::i64, PartialShape{8, 20, 1});
    const auto shape_of = make_shared<v3::ShapeOf>(param);
    const auto cast_fp = make_shared<op::v0::Convert>(shape_of, element::f32);
    const auto mul = make_shared<v1::Multiply>(cast_fp, ov::op::v0::Constant::create(element::f32, {3}, {-2, 1, -2}));
    const auto div = make_shared<v1::Divide>(mul, ov::op::v0::Constant::create(element::f32, {3}, {-10, 41, -1}));
    // order after convert [1, 0, 2]
    const auto cast_int = make_shared<op::v0::Convert>(div, element::i32);
    // order after gather [2, 1, 0]
    const auto gather = make_shared<v1::Gather>(cast_int,
                                                ov::op::v0::Constant::create(element::i32, {3}, {2, 0, 1}),
                                                ov::op::v0::Constant::create(element::i32, {}, {0}));

    const auto r = make_shared<v1::Transpose>(arg, gather);

    EXPECT_EQ(r->get_output_element_type(v1::Transpose::ARG_T), element::f32);
    EXPECT_EQ(r->get_output_partial_shape(v1::Transpose::ARG_T), PartialShape({6, Dimension(4, 16), Dimension(2, 8)}));
}

/**
 * \brief Transpose when order is dimensions from parameter shape.
 *
 * One dimension is dynamic, transposed output shape cannot be deduced and will  be dynamic.
 */
TEST(type_prop, transpose_when_order_is_shape_of_dynamic_partial_shape) {
    const auto arg =
        make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension(2, 8), Dimension(4, 16), 6});

    const auto param = make_shared<ov::op::v0::Parameter>(element::i64, PartialShape{0, 2, Dimension(1, 2)});
    const auto shape_of = make_shared<v3::ShapeOf>(param);

    const auto r = make_shared<v1::Transpose>(arg, shape_of);

    EXPECT_EQ(r->get_output_element_type(v1::Transpose::ARG_T), element::f32);
    EXPECT_EQ(r->get_output_partial_shape(v1::Transpose::ARG_T), PartialShape::dynamic(3));
}

using transpose_prop_params = tuple<vector<int64_t>,  // transpose order
                                    PartialShape,     // Input partial shape
                                    PartialShape      // Expected partial shape
                                    >;

// Test pre-defined constants.
static constexpr auto exp_type = element::f32;
static const auto interval_dim_1 = Dimension(3, 5);
static const auto interval_dim_2 = Dimension(1, 8);

/** \brief Parametrize fixture to test transpose property. */
class TransposeTest : public TestWithParam<transpose_prop_params> {
protected:
    PartialShape input_p_shape, exp_p_shape;
    vector<int64_t> transpose_order;

    void SetUp() override {
        std::tie(transpose_order, input_p_shape, exp_p_shape) = GetParam();
    }

    ov::TensorSymbol make_seq_symbols(const size_t count) {
        ov::TensorSymbol symbols;
        for (size_t i = 0; i < count; ++i)
            symbols.push_back(std::make_shared<ov::Symbol>());
        return symbols;
    }

    ov::TensorSymbol make_seq_symbols_by_order(ov::TensorSymbol symbols, const vector<int64_t> order) {
        ov::TensorSymbol new_symbols;
        for (const auto& i : order)
            new_symbols.push_back(symbols[i]);
        return new_symbols;
    }
};

INSTANTIATE_TEST_SUITE_P(
    type_prop,
    TransposeTest,
    Values(make_tuple(vector<int64_t>{2, 0, 1}, PartialShape{2, interval_dim_2, 4}, PartialShape{4, 2, interval_dim_2}),
           make_tuple(vector<int64_t>{0, 2, 1},
                      PartialShape{interval_dim_1, interval_dim_2, 4},
                      PartialShape{interval_dim_1, 4, interval_dim_2}),
           make_tuple(vector<int64_t>{1, 2, 3, 0},
                      PartialShape{interval_dim_1, 2, 3, 4},
                      PartialShape{2, 3, 4, interval_dim_1}),
           make_tuple(vector<int64_t>{3, 0, 2, 1},
                      PartialShape{interval_dim_1, 2, interval_dim_2, 4},
                      PartialShape{4, interval_dim_1, interval_dim_2, 2}),
           make_tuple(vector<int64_t>{1, 0, 3, 2},
                      PartialShape{interval_dim_1, interval_dim_2, interval_dim_2, interval_dim_1},
                      PartialShape{interval_dim_2, interval_dim_1, interval_dim_1, interval_dim_2})),
    PrintToStringParamName());

TEST_P(TransposeTest, use_default_ctor) {
    const auto input = make_shared<ov::op::v0::Parameter>(exp_type, input_p_shape);
    const auto order = ov::op::v0::Constant::create(element::i64, Shape{transpose_order.size()}, transpose_order);

    const auto output = make_shared<op::v1::Transpose>();
    output->set_arguments(NodeVector{input, order});
    output->validate_and_infer_types();

    EXPECT_EQ(output->get_output_element_type(op::v1::Transpose::ARG_T), exp_type);
    EXPECT_EQ(output->get_output_partial_shape(op::v1::Transpose::ARG_T), exp_p_shape);
}

/**
 * \brief Test interval dimension propagate in transpose.
 *
 * The interval dimensions should be moved accordingly to transpose order.
 */
TEST_P(TransposeTest, propagate_interval_shape) {
    const auto input = make_shared<ov::op::v0::Parameter>(exp_type, input_p_shape);
    const auto order = ov::op::v0::Constant::create(element::i64, Shape{transpose_order.size()}, transpose_order);

    const auto output = make_shared<op::v1::Transpose>(input, order);

    EXPECT_EQ(output->get_output_element_type(op::v1::Transpose::ARG_T), exp_type);
    EXPECT_EQ(output->get_output_partial_shape(op::v1::Transpose::ARG_T), exp_p_shape);
}

/**
 * \brief Check symbols propagation for all dimensions.
 *
 * The symbols should be moved accordingly to transpose order.
 */
TEST_P(TransposeTest, propagate_symbols) {
    const auto symbols = make_seq_symbols(transpose_order.size());
    const auto exp_symbols = make_seq_symbols_by_order(symbols, transpose_order);

    set_shape_symbols(input_p_shape, symbols);

    const auto input = make_shared<ov::op::v0::Parameter>(exp_type, input_p_shape);
    const auto order = ov::op::v0::Constant::create(element::i64, Shape{transpose_order.size()}, transpose_order);
    const auto output = make_shared<op::v1::Transpose>(input, order);

    EXPECT_EQ(get_shape_symbols(output->get_output_partial_shape(op::v1::Transpose::ARG_T)), exp_symbols);
}
