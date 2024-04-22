// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/select.hpp"

#include "common_test_utils/type_prop.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, select_deduce) {
    auto tv0_2_4_param_0 = make_shared<ov::op::v0::Parameter>(element::boolean, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_2 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    auto bc = make_shared<op::v1::Select>(tv0_2_4_param_0, tv0_2_4_param_1, tv0_2_4_param_2);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{2, 4}));
}

TEST(type_prop, select_default_constructor) {
    auto cond_param = make_shared<ov::op::v0::Parameter>(element::boolean, Shape{2, 4});
    auto then_param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    auto else_param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    auto op = make_shared<op::v1::Select>();

    EXPECT_EQ(op->get_auto_broadcast().m_type, op::AutoBroadcastType::NUMPY);

    op->set_argument(0, cond_param);
    op->set_argument(1, then_param);
    op->set_argument(2, else_param);

    op->set_auto_broadcast(op::AutoBroadcastType::NONE);
    EXPECT_EQ(op->get_auto_broadcast().m_type, op::AutoBroadcastType::NONE);

    op->validate_and_infer_types();

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_output_shape(0), (Shape{2, 4}));
}

TEST(type_prop, select_symbols_cond_numpy) {
    auto symboled_shape = PartialShape{{2, 8}, {3, 7}, {1, 10}, {1, 6}, {1, 10}};
    auto symbols = set_shape_symbols(symboled_shape);
    ov::TensorSymbol expected_symbols{symbols[0], symbols[1], symbols[2], nullptr, symbols[4]};

    auto cond_param = make_shared<ov::op::v0::Parameter>(element::boolean, symboled_shape);
    auto then_param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(5));
    auto else_param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape({{1, 5}, {1, 11}, 5, {1, 8}}));
    auto op = make_shared<op::v1::Select>(cond_param, then_param, else_param);

    const auto& out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(out_shape, PartialShape({{2, 8}, {3, 7}, -1, 5, -1}));
    EXPECT_EQ(get_shape_symbols(out_shape), expected_symbols);
}

TEST(type_prop, select_symbols_then_numpy) {
    auto symboled_shape = PartialShape::dynamic(5);
    auto symbols = set_shape_symbols(symboled_shape);
    ov::TensorSymbol expected_symbols{nullptr, nullptr, symbols[2], nullptr, symbols[4]};

    auto cond_param =
        make_shared<ov::op::v0::Parameter>(element::boolean, PartialShape{{2, 8}, {3, 7}, {1, 10}, {1, 6}, {1, 10}});
    auto then_param = make_shared<ov::op::v0::Parameter>(element::f32, symboled_shape);
    auto else_param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape({{1, 5}, {1, 11}, 5, {1, 8}}));
    auto op = make_shared<op::v1::Select>(cond_param, then_param, else_param);

    const auto& out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(out_shape, PartialShape({{2, 8}, {3, 7}, -1, 5, -1}));
    EXPECT_EQ(get_shape_symbols(out_shape), expected_symbols);
}

TEST(type_prop, select_symbols_else_numpy) {
    auto symboled_shape = PartialShape{{1, 5}, {1, 11}, 5, {1, 8}};
    auto symbols = set_shape_symbols(symboled_shape);

    ov::TensorSymbol expected_symbols{nullptr, nullptr, symbols[1], symbols[2], symbols[3]};

    auto cond_param =
        make_shared<ov::op::v0::Parameter>(element::boolean, PartialShape{{2, 8}, {3, 7}, {1, 10}, {1, 6}, {1, 10}});
    auto then_param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(5));
    auto else_param = make_shared<ov::op::v0::Parameter>(element::f32, symboled_shape);
    auto op = make_shared<op::v1::Select>(cond_param, then_param, else_param);

    const auto& out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(out_shape, PartialShape({{2, 8}, {3, 7}, -1, 5, -1}));
    EXPECT_EQ(get_shape_symbols(out_shape), expected_symbols);
}

TEST(type_prop, select_symbols_all_params_numpy) {
    auto symboled_shape_cond = PartialShape{-1, 2, 1, 3, 1, {2, 5}, {1, 8}, {5, -1}, {-1, 5}};
    auto symboled_shape_then = PartialShape{-1, 2, 4, 1, 1, {1, 5}, {2, 8}, {5, -1}, {-1, 5}};
    auto symboled_shape_else = PartialShape{-1, 2, 1, 3, 5, {1, 7}, {1, 8}, {5, -1}, {-1, 5}};

    auto cond_symbols = set_shape_symbols(symboled_shape_cond);
    auto then_symbols = set_shape_symbols(symboled_shape_then);
    auto else_symbols = set_shape_symbols(symboled_shape_else);

    ov::TensorSymbol expected_symbols{cond_symbols[0],
                                      else_symbols[1],
                                      then_symbols[2],
                                      else_symbols[3],
                                      else_symbols[4],
                                      cond_symbols[5],
                                      then_symbols[6],
                                      else_symbols[7],
                                      cond_symbols[8]};

    auto cond_param = make_shared<ov::op::v0::Parameter>(element::boolean, symboled_shape_cond);
    auto then_param = make_shared<ov::op::v0::Parameter>(element::f32, symboled_shape_then);
    auto else_param = make_shared<ov::op::v0::Parameter>(element::f32, symboled_shape_else);
    auto op = make_shared<op::v1::Select>(cond_param, then_param, else_param);

    const auto& out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(out_shape, (PartialShape{-1, 2, 4, 3, 5, {2, 5}, {2, 8}, {5, -1}, {-1, 5}}));
    EXPECT_EQ(get_shape_symbols(out_shape), expected_symbols);
}

TEST(type_prop, select_symbols_all_params_none) {
    auto symboled_shape_cond = PartialShape{-1, 2, 4, 3, 5, {2, 5}, {2, 8}, {5, -1}, {-1, 5}};
    auto symboled_shape_then = PartialShape{-1, 2, 4, 3, 5, {2, 5}, {2, 8}, {5, -1}, {-1, 5}};
    auto symboled_shape_else = PartialShape{-1, 2, 4, 3, 5, {2, 5}, {2, 8}, {5, -1}, {-1, 5}};

    auto cond_symbols = set_shape_symbols(symboled_shape_cond);
    auto then_symbols = set_shape_symbols(symboled_shape_then);
    auto else_symbols = set_shape_symbols(symboled_shape_else);

    ov::TensorSymbol expected_symbols{else_symbols};

    auto cond_param = make_shared<ov::op::v0::Parameter>(element::boolean, symboled_shape_cond);
    auto then_param = make_shared<ov::op::v0::Parameter>(element::f32, symboled_shape_then);
    auto else_param = make_shared<ov::op::v0::Parameter>(element::f32, symboled_shape_else);
    auto op = make_shared<op::v1::Select>(cond_param, then_param, else_param, op::AutoBroadcastType::NONE);

    const auto& out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(out_shape, (PartialShape{-1, 2, 4, 3, 5, {2, 5}, {2, 8}, {5, -1}, {-1, 5}}));
    EXPECT_THAT(get_shape_symbols(out_shape), expected_symbols);
}

TEST(type_prop, select_symbols_all_params_pdpd) {
    auto symboled_shape_cond = PartialShape{-1, 2, 1, 1, 1, {1, 5}, {1, 8}, {5, -1}, {-1, 5}};
    auto symboled_shape_then = PartialShape{-1, 2, 4, 3, 5, {2, 5}, {2, 8}, {5, -1}, {-1, 5}};
    auto symboled_shape_else = PartialShape{-1, 2, 1, 3, 1, {1, 5}, {1, 8}, {5, -1}, {-1, 5}};

    auto cond_symbols = set_shape_symbols(symboled_shape_cond);
    auto then_symbols = set_shape_symbols(symboled_shape_then);
    auto else_symbols = set_shape_symbols(symboled_shape_else);

    ov::TensorSymbol expected_symbols{cond_symbols[0],
                                      then_symbols[1],
                                      then_symbols[2],
                                      then_symbols[3],
                                      then_symbols[4],
                                      then_symbols[5],
                                      then_symbols[6],
                                      then_symbols[7],
                                      cond_symbols[8]};

    auto cond_param = make_shared<ov::op::v0::Parameter>(element::boolean, symboled_shape_cond);
    auto then_param = make_shared<ov::op::v0::Parameter>(element::f32, symboled_shape_then);
    auto else_param = make_shared<ov::op::v0::Parameter>(element::f32, symboled_shape_else);
    auto op = make_shared<op::v1::Select>(cond_param,
                                          then_param,
                                          else_param,
                                          op::AutoBroadcastSpec{op::AutoBroadcastType::PDPD});

    const auto& out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(out_shape, (PartialShape{-1, 2, 4, 3, 5, {2, 5}, {2, 8}, {5, -1}, {-1, 5}}));
    EXPECT_EQ(get_shape_symbols(out_shape), expected_symbols);
}

TEST(type_prop, select_dynamic) {
    auto param_0 =
        make_shared<ov::op::v0::Parameter>(element::boolean, PartialShape({{2, 8}, {3, 7}, {1, 10}, {1, 6}, {1, 10}}));
    auto param_1 = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(5));
    auto param_2 = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape({{1, 5}, {1, 11}, 5, {1, 8}}));
    auto bc = make_shared<op::v1::Select>(param_0, param_1, param_2);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_output_partial_shape(0), PartialShape({{2, 8}, {3, 7}, -1, 5, -1}));
}

TEST(type_prop, select_shape_mismatch_a) {
    auto tv0_2_4_param_0 = make_shared<ov::op::v0::Parameter>(element::boolean, Shape{3, 5});
    auto tv0_2_4_param_1 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_2 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    try {
        auto bc = make_shared<op::v1::Select>(tv0_2_4_param_0, tv0_2_4_param_1, tv0_2_4_param_2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument shapes are inconsistent"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_shape_mismatch_b) {
    auto tv0_2_4_param_0 = make_shared<ov::op::v0::Parameter>(element::boolean, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 5});
    auto tv0_2_4_param_2 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    try {
        auto bc = make_shared<op::v1::Select>(tv0_2_4_param_0, tv0_2_4_param_1, tv0_2_4_param_2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument shapes are inconsistent"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_shape_mismatch_c) {
    auto tv0_2_4_param_0 = make_shared<ov::op::v0::Parameter>(element::boolean, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_2 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 5});
    try {
        auto bc = make_shared<op::v1::Select>(tv0_2_4_param_0, tv0_2_4_param_1, tv0_2_4_param_2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument shapes are inconsistent"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_elem_mismatch_a) {
    auto tv0_2_4_param_0 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_2 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    try {
        auto bc = make_shared<op::v1::Select>(tv0_2_4_param_0, tv0_2_4_param_1, tv0_2_4_param_2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument 0 must have boolean element type"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_elem_mismatch_bc) {
    auto tv0_2_4_param_0 = make_shared<ov::op::v0::Parameter>(element::boolean, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_2 = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2, 4});
    try {
        auto bc = make_shared<op::v1::Select>(tv0_2_4_param_0, tv0_2_4_param_1, tv0_2_4_param_2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument 1 and 2 element types must match"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_partial_all_rank_dynamic) {
    auto param0 = make_shared<ov::op::v0::Parameter>(element::boolean, PartialShape::dynamic());
    auto param1 = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());

    auto sel = make_shared<op::v1::Select>(param0, param1, param2);

    ASSERT_EQ(sel->get_output_element_type(0), element::f32);
    ASSERT_TRUE(sel->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, select_partial_all_rank_dynamic_arg0_et_dynamic_arg1_arg2_et_mismatch) {
    auto param0 = make_shared<ov::op::v0::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param1 = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 = make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());

    try {
        auto sel = make_shared<op::v1::Select>(param0, param1, param2);
        FAIL() << "Did not detect mismatched element types for args 1 and 2 (element type-dynamic "
                  "arg0)";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument 1 and 2 element types must match"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_partial_all_rank_dynamic_arg0_arg1_et_dynamic) {
    auto param0 = make_shared<ov::op::v0::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param1 = make_shared<ov::op::v0::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param2 = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());

    auto sel = make_shared<op::v1::Select>(param0, param1, param2);

    ASSERT_EQ(sel->get_output_element_type(0), element::f32);
    ASSERT_TRUE(sel->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, select_partial_all_rank_dynamic_arg0_arg2_et_dynamic) {
    auto param0 = make_shared<ov::op::v0::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param1 = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 = make_shared<ov::op::v0::Parameter>(element::dynamic, PartialShape::dynamic());

    auto sel = make_shared<op::v1::Select>(param0, param1, param2);

    ASSERT_EQ(sel->get_output_element_type(0), element::f32);
    ASSERT_TRUE(sel->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, select_partial_all_rank_dynamic_arg0_arg1_arg2_et_dynamic) {
    auto param0 = make_shared<ov::op::v0::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param1 = make_shared<ov::op::v0::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param2 = make_shared<ov::op::v0::Parameter>(element::dynamic, PartialShape::dynamic());

    auto sel = make_shared<op::v1::Select>(param0, param1, param2);

    ASSERT_EQ(sel->get_output_element_type(0), element::dynamic);
    ASSERT_TRUE(sel->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, select_partial_all_rank_static_dynamic_ok) {
    auto param0 = make_shared<ov::op::v0::Parameter>(element::boolean,
                                                     PartialShape{2, Dimension::dynamic(), Dimension::dynamic()});
    auto param1 =
        make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 8, Dimension::dynamic()});
    auto param2 =
        make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3});

    auto sel = make_shared<op::v1::Select>(param0, param1, param2);

    ASSERT_EQ(sel->get_output_element_type(0), element::f32);
    ASSERT_TRUE(sel->get_output_partial_shape(0).is_static());
    ASSERT_EQ(sel->get_output_shape(0), (Shape{2, 8, 3}));
}

TEST(type_prop, select_partial_all_rank_static_intransitive_incompatibility) {
    auto param0 = make_shared<ov::op::v0::Parameter>(element::boolean,
                                                     PartialShape{2, Dimension::dynamic(), Dimension::dynamic()});
    auto param1 =
        make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 8, Dimension::dynamic()});
    auto param2 = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{3, Dimension::dynamic(), 3});

    try {
        auto sel = make_shared<op::v1::Select>(param0, param1, param2);
        FAIL() << "Did not detect intransitive partial-shape incompatibility";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument shapes are inconsistent"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

//------------------------------ v1::Select ---------------------------------//
//
//
struct SelectParams {
    std::vector<Shape> shapes;
    std::vector<element::Type> ets;
    op::AutoBroadcastSpec auto_broadcast;

    SelectParams(const std::vector<Shape>& shape,
                 const std::vector<element::Type>& et,
                 const op::AutoBroadcastSpec& auto_broadcast)
        : shapes(shape),
          ets(et),
          auto_broadcast(auto_broadcast) {}
};

struct DeduceV1SelectTest : ::testing::TestWithParam<SelectParams> {};

TEST_P(DeduceV1SelectTest, output_shape) {
    auto tp = GetParam();
    auto cond = make_shared<ov::op::v0::Parameter>(tp.ets[0], tp.shapes[0]);
    auto ptrue = make_shared<ov::op::v0::Parameter>(tp.ets[1], tp.shapes[1]);
    auto pfalse = make_shared<ov::op::v0::Parameter>(tp.ets[2], tp.shapes[2]);
    auto select = make_shared<op::v1::Select>(cond, ptrue, pfalse, tp.auto_broadcast);

    ASSERT_EQ(select->get_shape(), tp.shapes[3]);
    ASSERT_EQ(select->get_element_type(), tp.ets[3]);
}

INSTANTIATE_TEST_SUITE_P(
    type_prop,
    DeduceV1SelectTest,
    ::testing::Values(SelectParams({{2, 4}, {2, 4}, {2, 4}, {2, 4}},
                                   {element::boolean, element::f32, element::f32, element::f32},
                                   op::AutoBroadcastType::NONE),
                      SelectParams({{2, 4}, {2, 4}, {2, 4}, {2, 4}},
                                   {element::boolean, element::f32, element::f32, element::f32},
                                   op::AutoBroadcastType::NUMPY),
                      SelectParams({{}, {2, 4}, {2, 4}, {2, 4}},
                                   {element::boolean, element::f32, element::f32, element::f32},
                                   op::AutoBroadcastType::NUMPY),
                      SelectParams({{}, {4}, {2, 4}, {2, 4}},
                                   {element::boolean, element::f32, element::dynamic, element::f32},
                                   op::AutoBroadcastType::NUMPY),
                      SelectParams({{}, {2, 4}, {4}, {2, 4}},
                                   {element::boolean, element::f32, element::f32, element::f32},
                                   op::AutoBroadcastType::NUMPY),
                      SelectParams({{4}, {2, 4}, {4}, {2, 4}},
                                   {element::boolean, element::i8, element::dynamic, element::i8},
                                   op::AutoBroadcastType::NUMPY),
                      SelectParams({{4}, {4}, {2, 4}, {2, 4}},
                                   {element::dynamic, element::dynamic, element::i8, element::i8},
                                   op::AutoBroadcastType::NUMPY),
                      SelectParams({{2, 4}, {2, 4}, {2}, {2, 4}},
                                   {element::boolean, element::f32, element::dynamic, element::f32},
                                   {op::AutoBroadcastType::PDPD, 0}),
                      SelectParams({{4}, {2, 4}, {4}, {2, 4}},
                                   {element::boolean, element::f32, element::f32, element::f32},
                                   {op::AutoBroadcastType::PDPD, 1}),
                      SelectParams({{1}, {2, 4}, {4}, {2, 4}},
                                   {element::boolean, element::f32, element::dynamic, element::f32},
                                   {op::AutoBroadcastType::PDPD, 1}),
                      SelectParams({{4}, {4, 2, 3, 8}, {4, 2, 3, 1}, {4, 2, 3, 8}},
                                   {element::boolean, element::f32, element::f32, element::f32},
                                   {op::AutoBroadcastType::PDPD, 0})),

    PrintToDummyParamName());

TEST(type_prop, select_v1_partial_shape) {
    auto a = make_shared<ov::op::v0::Parameter>(element::boolean, PartialShape::dynamic());
    auto b = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    auto c = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});

    auto select = make_shared<op::v1::Select>(a, b, c, op::AutoBroadcastType::NONE);
    ASSERT_EQ(select->get_shape(), (Shape{2, 4}));
}

TEST(type_prop, select_v1_partial_shape_autob) {
    auto a = make_shared<ov::op::v0::Parameter>(element::boolean, PartialShape{Dimension::dynamic()});
    auto b = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic()});
    auto c = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{2, Dimension::dynamic()});

    auto select = make_shared<op::v1::Select>(a, b, c);
    ASSERT_TRUE(select->get_output_partial_shape(0).same_scheme(PartialShape{2, Dimension::dynamic()}));
}

TEST(type_prop, select_v1_wrong_et) {
    auto param0 = make_shared<ov::op::v0::Parameter>(element::i8, Shape{2, 4});
    auto param1 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    auto param2 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});

    try {
        auto sel = make_shared<op::v1::Select>(param0, param1, param2);
        FAIL() << "Did not detect wrong element type";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument 0 must have boolean element type"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_v1_et_mismatch) {
    auto param0 = make_shared<ov::op::v0::Parameter>(element::boolean, Shape{2, 4});
    auto param1 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    auto param2 = make_shared<ov::op::v0::Parameter>(element::i8, Shape{2, 4});

    try {
        auto sel = make_shared<op::v1::Select>(param0, param1, param2);
        FAIL() << "Did not detect element type mismatch";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument 1 and 2 element types must match."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_v1_shape_mismatch) {
    auto param0 = make_shared<ov::op::v0::Parameter>(element::boolean, Shape{2, 4});
    auto param1 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3});
    auto param2 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});

    try {
        auto sel = make_shared<op::v1::Select>(param0, param1, param2);
        FAIL() << "Did not detect shape mismatch";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument shapes are inconsistent."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_v1_partial_shape_mismatch) {
    auto param0 = make_shared<ov::op::v0::Parameter>(element::boolean, PartialShape{3, Dimension::dynamic()});
    auto param1 = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{2, Dimension::dynamic()});
    auto param2 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});

    try {
        auto sel = make_shared<op::v1::Select>(param0, param1, param2);
        FAIL() << "Did not detect shape mismatch";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument shapes are inconsistent."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
