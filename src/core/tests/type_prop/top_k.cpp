// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/opsets/opset11.hpp"
#include "topk_shape_inference.hpp"

using namespace ov;
using namespace ov::opset11;
using namespace testing;
template <typename T>
class topk_type_prop : public TypePropOpTest<T> {
protected:
    PartialShapes make_broadcast_shapes_of_topk_outs(T* topk) {
        PartialShapes bcs_outputs;
        for (size_t i = 0; i < topk->get_output_size(); ++i) {
            auto bc = std::make_shared<Broadcast>(std::make_shared<Parameter>(element::i64, PartialShape{1}),
                                                  topk->output(i),
                                                  "BIDIRECTIONAL");
            bcs_outputs.push_back(bc->get_output_partial_shape(0));
        }

        return bcs_outputs;
    }

    element::Type exp_default_idx_type{element::i32};
};

template <typename T>
using topk_type_prop_with_evaluate = topk_type_prop<T>;

TYPED_TEST_SUITE_P(topk_type_prop);
TYPED_TEST_SUITE_P(topk_type_prop_with_evaluate);

TYPED_TEST_P(topk_type_prop, default_ctor) {
    constexpr int64_t exp_axis = -2;
    constexpr auto exp_idx_type = element::i64;
    constexpr auto exp_data_type = element::f32;

    const auto data = std::make_shared<Parameter>(exp_data_type, Shape{1, 2, 3, 4});
    const auto k = Constant::create(element::i64, Shape{}, {2});

    const auto op = this->make_op();
    op->set_arguments(OutputVector{data, k});
    op->set_axis(exp_axis);
    op->set_index_element_type(exp_idx_type);
    op->set_mode(op::TopKMode::MIN);
    op->set_sort_type(op::TopKSortType::SORT_INDICES);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_provided_axis(), exp_axis);
    EXPECT_EQ(op->get_axis(), 2);
    EXPECT_EQ(op->get_input_size(), 2);
    EXPECT_EQ(op->get_output_size(), 2);
    EXPECT_EQ(op->get_mode(), op::TopKMode::MIN);
    EXPECT_EQ(op->get_sort_type(), op::TopKSortType::SORT_INDICES);
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Value type", &Output<Node>::get_element_type, exp_data_type),
                            Property("Index type", &Output<Node>::get_element_type, exp_idx_type)));
    EXPECT_THAT(op->outputs(), Each(Property("Shape", &Output<Node>::get_shape, Shape({1, 2, 2, 4}))));
}

TYPED_TEST_P(topk_type_prop, default_ctor_no_arguments) {
    constexpr int64_t exp_axis = 3;
    const auto data_shape = PartialShape{1, {3, 4}, 4, {2, 6}};
    int64_t k = 3;

    const auto op = this->make_op();
    op->set_axis(data_shape.rank(), exp_axis);
    op->set_mode(op::TopKMode::MIN);
    op->set_sort_type(op::TopKSortType::SORT_INDICES);

    const auto constant_map = std::unordered_map<size_t, ov::Tensor>{{1, {element::i64, Shape{}, &k}}};

    const auto outputs = shape_infer(op.get(), PartialShapes{data_shape, {}}, ov::make_tensor_accessor(constant_map));

    EXPECT_EQ(op->get_provided_axis(), exp_axis);
    EXPECT_EQ(op->get_axis(), exp_axis);
    EXPECT_EQ(op->get_input_size(), 0);
    EXPECT_EQ(op->get_output_size(), 0);
    EXPECT_EQ(op->get_mode(), op::TopKMode::MIN);
    EXPECT_EQ(op->get_sort_type(), op::TopKSortType::SORT_INDICES);
    EXPECT_THAT(op->outputs(),
                Each(Property("Partial shape", &Output<Node>::get_partial_shape, PartialShape({1, {3, 4}, 4, 3}))));
}

TYPED_TEST_P(topk_type_prop, negative_axis_support) {
    constexpr int64_t exp_axis = -1;
    constexpr auto exp_data_type = element::f32;
    constexpr auto exp_idx_type = element::i64;

    auto data_shape = PartialShape{1, 2, 3, 4};
    auto symbols = set_shape_symbols(data_shape);
    const auto data = std::make_shared<Parameter>(exp_data_type, data_shape);
    const auto k = Constant::create(exp_idx_type, Shape{}, {2});

    const auto op = this->make_op(data, k, exp_axis, "max", "value", exp_idx_type);

    EXPECT_EQ(op->get_provided_axis(), exp_axis);
    EXPECT_EQ(op->get_axis(), 3);
    EXPECT_EQ(op->get_input_size(), 2);
    EXPECT_EQ(op->get_output_size(), 2);
    EXPECT_EQ(op->get_mode(), op::TopKMode::MAX);
    EXPECT_EQ(op->get_sort_type(), op::TopKSortType::SORT_VALUES);
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Value type", &Output<Node>::get_element_type, exp_data_type),
                            Property("Index type", &Output<Node>::get_element_type, exp_idx_type)));
    EXPECT_THAT(op->outputs(), Each(Property("Shape", &Output<Node>::get_shape, Shape({1, 2, 3, 2}))));
    EXPECT_THAT(op->outputs(),
                Each(Property(&Output<Node>::get_partial_shape,
                              ResultOf(get_shape_symbols, ElementsAre(symbols[0], symbols[1], symbols[2], nullptr)))));
}

TYPED_TEST_P(topk_type_prop, default_index_element_type) {
    constexpr auto exp_data_type = element::f32;

    const auto data = std::make_shared<Parameter>(exp_data_type, Shape{1, 2, 3, 4});
    const auto k = Constant::create(element::i64, Shape{}, {3});
    {
        // k > dimension
        const auto op = this->make_op(data, k, 0, "max", "value");

        EXPECT_THAT(op->outputs(),
                    ElementsAre(Property("Value type", &Output<Node>::get_element_type, exp_data_type),
                                Property("Index type", &Output<Node>::get_element_type, this->exp_default_idx_type)));
        EXPECT_THAT(op->outputs(), Each(Property("Shape", &Output<Node>::get_shape, Shape({3, 2, 3, 4}))));
    }
    {
        // k < dimension
        const auto op = this->make_op(data, k, 3, "max", "value");

        EXPECT_THAT(op->outputs(),
                    ElementsAre(Property("Value type", &Output<Node>::get_element_type, exp_data_type),
                                Property("Index type", &Output<Node>::get_element_type, this->exp_default_idx_type)));
        EXPECT_THAT(op->outputs(), Each(Property("Shape", &Output<Node>::get_shape, Shape({1, 2, 3, 3}))));
    }
}

TYPED_TEST_P(topk_type_prop, k_is_negative) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{-1, {-1, 2}});
    const auto k = Constant::create(element::i64, Shape{}, {-1});

    OV_EXPECT_THROW(const auto op = this->make_op(data, k, 0, "max", "value"),
                    NodeValidationFailure,
                    HasSubstr("The value of 'K' must be greater or equal to zero."));
}

TYPED_TEST_P(topk_type_prop, k_for_dynamic_dimension) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{-1, {-1, 2}});
    const auto k = Constant::create(element::i64, Shape{}, {5});
    const auto op = this->make_op(data, k, 0, "max", "value");

    EXPECT_THAT(op->outputs(),
                Each(Property("Partial Shape", &Output<Node>::get_partial_shape, PartialShape({5, {-1, 2}}))));
}

TYPED_TEST_P(topk_type_prop, k_for_interval_dimension) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{{2, 12}, {-1, 2}});
    const auto k = Constant::create(element::i64, Shape{}, {6});
    const auto op = this->make_op(data, k, 0, "max", "value");

    EXPECT_THAT(op->outputs(),
                Each(Property("Partial Shape", &Output<Node>::get_partial_shape, PartialShape({6, {-1, 2}}))));
}

TYPED_TEST_P(topk_type_prop, k_is_unknown_for_static_dimension) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{2, 10});
    const auto k = std::make_shared<Parameter>(element::i32, PartialShape({}));
    const auto op = this->make_op(data, k, 1, "max", "value");

    EXPECT_THAT(op->outputs(),
                Each(Property("Partial Shape", &Output<Node>::get_partial_shape, PartialShape({2, {0, 10}}))));
}

TYPED_TEST_P(topk_type_prop, k_is_unknown_for_dynamic_dimension) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{-1, {-1, 2}});
    const auto k = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto op = this->make_op(data, k, 0, "max", "value");

    EXPECT_THAT(op->outputs(),
                Each(Property("Partial Shape", &Output<Node>::get_partial_shape, PartialShape({-1, {-1, 2}}))));
}

TYPED_TEST_P(topk_type_prop, k_is_unknown_for_interval_dimension) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{{2, 100}, {-1, 2}});
    const auto k = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto op = this->make_op(data, k, 0, "max", "value");

    EXPECT_THAT(op->outputs(),
                Each(Property("Partial Shape", &Output<Node>::get_partial_shape, PartialShape({{0, 100}, {-1, 2}}))));
}

TYPED_TEST_P(topk_type_prop, k_is_unknown_for_interval_with_no_upper_bound_dimension) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{{2, -1}, {-1, 2}});
    const auto k = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto op = this->make_op(data, k, 0, "max", "value");

    EXPECT_THAT(op->outputs(),
                Each(Property("Partial Shape", &Output<Node>::get_partial_shape, PartialShape({-1, {-1, 2}}))));
}

TYPED_TEST_P(topk_type_prop, data_and_k_shapes_are_dynamic) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto k = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto op = this->make_op(data, k, 1, "max", "value");

    EXPECT_THAT(op->outputs(),
                Each(Property("Partial Shape", &Output<Node>::get_partial_shape, PartialShape::dynamic())));
}

TYPED_TEST_P(topk_type_prop_with_evaluate, propagate_symbol_and_not_interval_value_max) {
    auto p_shape = PartialShape{5, 6, 4, 3, 8};
    set_shape_symbols(p_shape);

    constexpr auto et = element::i64;
    const auto symboled_param = std::make_shared<Parameter>(et, p_shape);
    const auto symboled_shape_of = std::make_shared<ShapeOf>(symboled_param);

    const auto k = Constant::create(et, Shape{}, {3});
    const auto op = this->make_op(symboled_shape_of, k, 0, "max", "index", element::i32);

    const auto bc_shapes = this->make_broadcast_shapes_of_topk_outs(op.get());

    EXPECT_THAT(bc_shapes, ElementsAre(PartialShape({5, 6, 8}), PartialShape({0, 1, 4})));
    EXPECT_THAT(bc_shapes, Each(ResultOf(get_shape_symbols, Each(nullptr))));
}

TYPED_TEST_P(topk_type_prop_with_evaluate, propagate_symbol_and_not_interval_value_min) {
    auto p_shape = PartialShape{5, 6, 3, 4, 8};
    set_shape_symbols(p_shape);

    constexpr auto et = element::i64;
    const auto symboled_param = std::make_shared<Parameter>(et, p_shape);
    const auto symboled_shape_of = std::make_shared<ShapeOf>(symboled_param);

    const auto k = Constant::create(et, Shape{}, {3});
    const auto op = this->make_op(symboled_shape_of, k, 0, "min", "index", element::i32);

    const auto bc_shapes = this->make_broadcast_shapes_of_topk_outs(op.get());

    EXPECT_THAT(bc_shapes, ElementsAre(PartialShape({5, 3, 4}), PartialShape({0, 2, 3})));
    EXPECT_THAT(bc_shapes, Each(ResultOf(get_shape_symbols, Each(nullptr))));
}

TYPED_TEST_P(topk_type_prop, preserve_partial_values_and_symbols_k_is_interval) {
    auto k_dim = Dimension{10, 20};
    auto shape = PartialShape{k_dim};
    auto k_symbol = std::make_shared<Symbol>();
    k_dim.set_symbol(k_symbol);

    const auto p_k = std::make_shared<Parameter>(element::i64, shape);
    const auto shape_of_k = std::make_shared<ShapeOf>(p_k);
    const auto k = std::make_shared<Squeeze>(shape_of_k, Constant::create(element::i64, Shape{}, {0}));

    auto data_shape = PartialShape{{2, 5}, {12, 18}, {2, 30}, {30, 40}, {-1, 15}, {15, -1}};
    auto symbols = set_shape_symbols(data_shape);
    const auto data = std::make_shared<Parameter>(element::f32, data_shape);

    {
        // dim{2,5} k{10,20} -> {2,5}
        const auto op = this->make_op(data, k, 0, "max", "value");
        EXPECT_THAT(op->get_output_partial_shape(0),
                    AllOf(PartialShape({{2, 5}, {12, 18}, {2, 30}, {30, 40}, {-1, 15}, {15, -1}}),
                          ResultOf(get_shape_symbols,
                                   ElementsAre(nullptr, symbols[1], symbols[2], symbols[3], symbols[4], symbols[5]))));
    }
    {
        // dim{12,18} k{10,20} -> {10,18}
        const auto op = this->make_op(data, k, 1, "max", "value");
        EXPECT_THAT(op->get_output_partial_shape(0),
                    AllOf(PartialShape({{2, 5}, {10, 18}, {2, 30}, {30, 40}, {-1, 15}, {15, -1}}),
                          ResultOf(get_shape_symbols,
                                   ElementsAre(symbols[0], nullptr, symbols[2], symbols[3], symbols[4], symbols[5]))));
    }
    {
        // dim{2, 30} k{10,20} -> {2,20}
        const auto op = this->make_op(data, k, 2, "max", "value");
        EXPECT_THAT(op->get_output_partial_shape(0),
                    AllOf(PartialShape({{2, 5}, {12, 18}, {2, 20}, {30, 40}, {-1, 15}, {15, -1}}),
                          ResultOf(get_shape_symbols,
                                   ElementsAre(symbols[0], symbols[1], nullptr, symbols[3], symbols[4], symbols[5]))));
    }
    {
        // dim{30,40} k{10,20} -> {10,20}  (should use k upper bounds??)
        const auto op = this->make_op(data, k, 3, "max", "value");
        EXPECT_THAT(op->get_output_partial_shape(0),
                    AllOf(PartialShape({{2, 5}, {12, 18}, {2, 30}, {10, 20}, {-1, 15}, {15, -1}}),
                          ResultOf(get_shape_symbols,
                                   ElementsAre(symbols[0], symbols[1], symbols[2], nullptr, symbols[4], symbols[5]))));
    }
    {
        // dim{-inf,15} k{10,20} -> {0,15}
        const auto op = this->make_op(data, k, 4, "max", "value");
        EXPECT_THAT(op->get_output_partial_shape(0),
                    AllOf(PartialShape({{2, 5}, {12, 18}, {2, 30}, {30, 40}, {0, 15}, {15, -1}}),
                          ResultOf(get_shape_symbols,
                                   ElementsAre(symbols[0], symbols[1], symbols[2], symbols[3], nullptr, symbols[5]))));
    }
    {
        // dim{15,inf} k{10,20} -> {10,inf}
        const auto op = this->make_op(data, k, 5, "max", "value");
        EXPECT_THAT(op->get_output_partial_shape(0),
                    AllOf(PartialShape({{2, 5}, {12, 18}, {2, 30}, {30, 40}, {-1, 15}, {10, -1}}),
                          ResultOf(get_shape_symbols,
                                   ElementsAre(symbols[0], symbols[1], symbols[2], symbols[3], symbols[4], nullptr))));
    }
}

TYPED_TEST_P(topk_type_prop, preserve_partial_values_and_symbols_k_is_interval_with_no_upper_bound) {
    auto shape = PartialShape{{4, -1}};
    auto k_symbols = set_shape_symbols(shape);

    const auto p_k = std::make_shared<Parameter>(element::i64, shape);
    const auto shape_of_k = std::make_shared<ShapeOf>(p_k);
    // Squeeze make scalar of interval value {4,inf}
    const auto k = std::make_shared<Squeeze>(shape_of_k, Constant::create(element::i64, Shape{}, {0}));

    auto data_shape = PartialShape{5, {2, 8}, {2, 100}};
    auto symbols = set_shape_symbols(data_shape);
    const auto data = std::make_shared<Parameter>(element::f32, data_shape);

    {
        // dim{5} k{4,inf} -> {4,5}
        const auto op = this->make_op(data, k, 0, "max", "value");
        EXPECT_THAT(op->get_output_partial_shape(0),
                    AllOf(PartialShape({{4, 5}, {2, 8}, {2, 100}}),
                          ResultOf(get_shape_symbols, ElementsAre(nullptr, symbols[1], symbols[2]))));
    }
    {
        // dim{2,8} k{4,inf} -> {2,8}
        const auto op = this->make_op(data, k, 1, "max", "value");
        EXPECT_THAT(op->get_output_partial_shape(0),
                    AllOf(PartialShape({5, {2, 8}, {2, 100}}),
                          ResultOf(get_shape_symbols, ElementsAre(symbols[0], nullptr, symbols[2]))));
    }
    {
        // dim{2,100} k{4,inf} -> {2,100}
        const auto op = this->make_op(data, k, 2, "max", "value");
        EXPECT_THAT(op->get_output_partial_shape(0),
                    AllOf(PartialShape({5, {2, 8}, {2, 100}}),
                          ResultOf(get_shape_symbols, ElementsAre(symbols[0], symbols[1], nullptr))));
    }
}

TYPED_TEST_P(topk_type_prop, negative_axis_dynamic_rank) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto k = Constant::create(element::i64, Shape{}, {2});
    const int64_t axis = -2;
    const auto op = this->make_op(data, k, axis, "max", "value");

    OV_EXPECT_THROW(op->get_axis(), NodeValidationFailure, HasSubstr("Normalized axis of TopK is unknown"));
}

TYPED_TEST_P(topk_type_prop, incorrect_index_element_type) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto k = Constant::create(element::i64, Shape{}, {2});
    const int64_t axis = -2;

    OV_EXPECT_THROW(const auto op = this->make_op(data, k, axis, "max", "value", element::i16),
                    NodeValidationFailure,
                    HasSubstr("Index element type attribute should be either \'i32\' or \'i64\'. Got:"));
}

REGISTER_TYPED_TEST_SUITE_P(topk_type_prop,
                            default_ctor,
                            default_ctor_no_arguments,
                            negative_axis_support,
                            default_index_element_type,
                            k_is_negative,
                            k_for_dynamic_dimension,
                            k_for_interval_dimension,
                            k_is_unknown_for_static_dimension,
                            k_is_unknown_for_dynamic_dimension,
                            k_is_unknown_for_interval_dimension,
                            k_is_unknown_for_interval_with_no_upper_bound_dimension,
                            data_and_k_shapes_are_dynamic,
                            preserve_partial_values_and_symbols_k_is_interval,
                            preserve_partial_values_and_symbols_k_is_interval_with_no_upper_bound,
                            negative_axis_dynamic_rank,
                            incorrect_index_element_type);

REGISTER_TYPED_TEST_SUITE_P(topk_type_prop_with_evaluate,
                            propagate_symbol_and_not_interval_value_max,
                            propagate_symbol_and_not_interval_value_min);

// TODO: merge the two instantiations into one when v11::TopK gets the evaluate() method
typedef Types<op::v1::TopK, op::v3::TopK, op::v11::TopK> TopKTypes;
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop, topk_type_prop, TopKTypes);

typedef Types<op::v1::TopK, op::v3::TopK> TopKTypesWithEvaluate;
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop, topk_type_prop_with_evaluate, TopKTypesWithEvaluate);

class TypePropTopKV1Test : public TypePropOpTest<op::v1::TopK> {};

TEST_F(TypePropTopKV1Test, k_is_u32) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{5, {-1, 2}});
    const auto k = Constant::create(element::u32, Shape{}, {1});

    OV_EXPECT_THROW(const auto op = this->make_op(data, k, 0, "max", "value"),
                    NodeValidationFailure,
                    HasSubstr("K input element type must be i8, i32 or i64 (got u32)"));
}

class TypePropTopKV3Test : public TypePropOpTest<op::v3::TopK> {};

TEST_F(TypePropTopKV3Test, k_is_u32) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{5, {-1, 2}});
    const auto k = Constant::create(element::u32, Shape{}, {1});

    const auto op = this->make_op(data, k, 0, "max", "value");

    EXPECT_THAT(op->outputs(),
                Each(Property("PartialShape", &Output<Node>::get_partial_shape, PartialShape({1, {-1, 2}}))));
}

TEST(type_prop, top_k_partial_value) {
    const auto data = std::make_shared<opset11::Parameter>(element::f32, PartialShape{{0, 16000}});
    const auto shape = std::make_shared<opset11::ShapeOf>(data);
    const auto concat =
        std::make_shared<Concat>(ov::OutputVector{shape, Constant::create(element::i64, {1}, {200})}, 0);
    const auto reduce_min = std::make_shared<opset11::ReduceMin>(concat, Constant::create(element::i64, {1}, {0}));
    const auto op = std::make_shared<op::v3::TopK>(data, reduce_min, 0, "max", "value");
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({{0, 200}}));
}

TEST(type_prop, topk_v11_stable_sort_by_none) {
    const auto data = std::make_shared<Parameter>(element::f32, Shape{2, 3, 4});
    const auto k = Constant::create(element::u32, Shape{}, {1});
    OV_EXPECT_THROW(const auto op = std::make_shared<ov::op::v11::TopK>(data,
                                                                        k,
                                                                        2,
                                                                        op::TopKMode::MIN,
                                                                        op::TopKSortType::NONE,
                                                                        element::i64,
                                                                        true),
                    NodeValidationFailure,
                    HasSubstr("Stable sort can only be used when TopK's sorting mode is set to 'VALUE' or 'INDEX'"));
}
