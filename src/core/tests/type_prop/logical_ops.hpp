// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"

template <typename T, ov::element::Type_t ELEMENT_TYPE>
class LogicalOperatorType {
public:
    using op_type = T;
    static constexpr ov::element::Type_t element_type = ELEMENT_TYPE;
};

template <typename T>
class LogicalOperatorTypeProp : public TypePropOpTest<typename T::op_type> {
protected:
    size_t exp_logical_op_output_size{1};
};

class LogicalOperatorTypeName {
public:
    template <typename T>
    static std::string GetName(int) {
        using OP_Type = typename T::op_type;
        const ov::Node::type_info_t typeinfo = OP_Type::get_type_info_static();
        return typeinfo.name;
    }
};

TYPED_TEST_SUITE_P(LogicalOperatorTypeProp);

namespace {
template <typename T>
void incorrect_init(const ov::element::Type& type,
                    const std::string& err,
                    const ov::Shape& shape1 = {1, 3, 6},
                    const ov::Shape& shape2 = {1, 3, 6}) {
    auto input1 = std::make_shared<ov::op::v0::Parameter>(type, shape1);
    auto input2 = std::make_shared<ov::op::v0::Parameter>(type, shape2);
    try {
        auto op = std::make_shared<T>(input1, input2);
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), err);
    }
}
}  // namespace

TYPED_TEST_P(LogicalOperatorTypeProp, incorrect_type_f32) {
    using OP_Type = typename TypeParam::op_type;
    incorrect_init<OP_Type>(ov::element::f32,
                            "Operands for logical operators must have boolean element type but have element type f32");
}

TYPED_TEST_P(LogicalOperatorTypeProp, incorrect_type_f64) {
    using OP_Type = typename TypeParam::op_type;
    incorrect_init<OP_Type>(ov::element::f64,
                            "Operands for logical operators must have boolean element type but have element type f64");
}

TYPED_TEST_P(LogicalOperatorTypeProp, incorrect_type_i32) {
    using OP_Type = typename TypeParam::op_type;
    incorrect_init<OP_Type>(ov::element::i32,
                            "Operands for logical operators must have boolean element type but have element type i32");
}

TYPED_TEST_P(LogicalOperatorTypeProp, incorrect_type_i64) {
    using OP_Type = typename TypeParam::op_type;
    incorrect_init<OP_Type>(ov::element::i64,
                            "Operands for logical operators must have boolean element type but have element type i64");
}

TYPED_TEST_P(LogicalOperatorTypeProp, incorrect_type_u32) {
    using OP_Type = typename TypeParam::op_type;
    incorrect_init<OP_Type>(ov::element::u32,
                            "Operands for logical operators must have boolean element type but have element type u32");
}

TYPED_TEST_P(LogicalOperatorTypeProp, incorrect_type_u64) {
    using OP_Type = typename TypeParam::op_type;
    incorrect_init<OP_Type>(ov::element::u64,
                            "Operands for logical operators must have boolean element type but have element type u64");
}

TYPED_TEST_P(LogicalOperatorTypeProp, incorrect_shape) {
    using OP_Type = typename TypeParam::op_type;
    incorrect_init<OP_Type>(ov::element::boolean,
                            "Argument shapes are inconsistent",
                            ov::Shape{1, 3, 6},
                            ov::Shape{1, 2, 3});
}

TYPED_TEST_P(LogicalOperatorTypeProp, inputs_have_different_types) {
    using namespace ov;
    const auto a = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape{1, 1, 6});
    const auto b = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 3, 1});

    OV_EXPECT_THROW(const auto logical_op = this->make_op(a, b),
                    NodeValidationFailure,
                    testing::HasSubstr("Arguments do not have the same element type"));
}

TYPED_TEST_P(LogicalOperatorTypeProp, inputs_have_inconsistent_shapes) {
    using namespace ov;
    const auto a = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape{1, 1, 6});
    const auto b = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape{1, 3, 3});

    OV_EXPECT_THROW(const auto logical_op = this->make_op(a, b),
                    NodeValidationFailure,
                    testing::HasSubstr("Argument shapes are inconsistent"));
}

TYPED_TEST_P(LogicalOperatorTypeProp, shape_broadcast) {
    using namespace ov;
    const auto exp_dtype = TypeParam::element_type;

    const auto a = std::make_shared<op::v0::Parameter>(element::boolean, Shape{1, 1, 6});
    const auto b = std::make_shared<op::v0::Parameter>(element::boolean, Shape{1, 3, 1});

    const auto logical_op = this->make_op(a, b);

    EXPECT_EQ(logical_op->get_element_type(), exp_dtype);
    EXPECT_EQ(logical_op->get_output_size(), this->exp_logical_op_output_size);
    EXPECT_EQ(logical_op->get_shape(), Shape({1, 3, 6}));
}

TYPED_TEST_P(LogicalOperatorTypeProp, partial_shape_no_broadcast) {
    using namespace ov;
    using namespace testing;

    auto shape_a = PartialShape{1, {2, 4}, {2, 5}, 4, -1};
    auto shape_b = PartialShape{1, 3, {1, 6}, 4, {-1, 5}};

    auto A = std::make_shared<ov::Symbol>(), B = std::make_shared<ov::Symbol>(), C = std::make_shared<ov::Symbol>(),
         D = std::make_shared<ov::Symbol>();
    auto E = std::make_shared<ov::Symbol>(), F = std::make_shared<ov::Symbol>();

    set_shape_symbols(shape_a, ov::TensorSymbol{nullptr, A, B, nullptr, C});
    set_shape_symbols(shape_b, ov::TensorSymbol{E, F, nullptr, nullptr, nullptr});
    const auto exp_shape = PartialShape{1, 3, {2, 5}, 4, {-1, 5}};

    const auto a = std::make_shared<op::v0::Parameter>(element::boolean, shape_a);
    const auto b = std::make_shared<op::v0::Parameter>(element::boolean, shape_b);

    EXPECT_THAT(this->make_op(a, b, "NONE")->get_output_partial_shape(0),
                AllOf(Eq(exp_shape), ResultOf(get_shape_symbols, ElementsAre(E, A, B, nullptr, C))));

    EXPECT_THAT(this->make_op(b, a, "NONE")->get_output_partial_shape(0),
                AllOf(Eq(exp_shape), ResultOf(get_shape_symbols, ElementsAre(E, F, B, nullptr, C))));
}

TYPED_TEST_P(LogicalOperatorTypeProp, partial_shape_numpy_broadcast) {
    using namespace ov;
    using namespace testing;

    auto shape_a = PartialShape{1, {2, 4}, {2, 5}, 4, -1};
    auto shape_b = PartialShape{1, 3, {1, 6}, 4};

    auto A = std::make_shared<ov::Symbol>(), B = std::make_shared<ov::Symbol>(), C = std::make_shared<ov::Symbol>(),
         D = std::make_shared<ov::Symbol>();
    auto E = std::make_shared<ov::Symbol>(), F = std::make_shared<ov::Symbol>(), G = std::make_shared<ov::Symbol>();
    set_shape_symbols(shape_a, ov::TensorSymbol{nullptr, A, B, C, D});
    set_shape_symbols(shape_b, ov::TensorSymbol{E, F, nullptr, G});
    const auto exp_shape = PartialShape{1, {2, 4}, 3, 4, 4};

    const auto a = std::make_shared<op::v0::Parameter>(element::boolean, shape_a);
    const auto b = std::make_shared<op::v0::Parameter>(element::boolean, shape_b);

    EXPECT_THAT(this->make_op(a, b, "NUMPY")->get_output_partial_shape(0),
                AllOf(Eq(exp_shape), ResultOf(get_shape_symbols, ElementsAre(nullptr, A, B, C, G))));

    EXPECT_THAT(this->make_op(b, a, "NUMPY")->get_output_partial_shape(0),
                AllOf(Eq(exp_shape), ResultOf(get_shape_symbols, ElementsAre(nullptr, A, F, C, G))));
}

TYPED_TEST_P(LogicalOperatorTypeProp, default_ctor) {
    using namespace ov;

    const auto op = this->make_op();
    const auto a = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape{1, {2, 4}, {2, 5}, 4, -1});
    const auto b = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape{1, 3, {1, 6}, 4});

    op->set_arguments(NodeVector{a, b});
    op->set_autob("NUMPY");
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_autob(), op::AutoBroadcastSpec("NUMPY"));
    EXPECT_EQ(op->get_element_type(), element::boolean);
    EXPECT_EQ(op->get_output_size(), this->exp_logical_op_output_size);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({1, {2, 4}, 3, 4, 4}));
}

REGISTER_TYPED_TEST_SUITE_P(LogicalOperatorTypeProp,
                            shape_broadcast,
                            partial_shape_no_broadcast,
                            partial_shape_numpy_broadcast,
                            incorrect_type_f32,
                            incorrect_type_f64,
                            incorrect_type_i32,
                            incorrect_type_i64,
                            incorrect_type_u32,
                            incorrect_type_u64,
                            incorrect_shape,
                            inputs_have_different_types,
                            inputs_have_inconsistent_shapes,
                            default_ctor);
