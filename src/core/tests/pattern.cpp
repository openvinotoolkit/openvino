// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <list>
#include <memory>

#include "common_test_utils/matcher.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_tools.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/cosh.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/non_max_suppression.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/true.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace ov::pass;
using namespace std;

static std::shared_ptr<Node> construct_constant_node(int n) {
    return ov::op::v0::Constant::create(element::i32, Shape{}, {n});
}

static std::shared_ptr<pass::pattern::op::Label> construct_variance_graph() {
    // construct varaiance
    auto N = ov::op::v0::Constant::create(element::f32, Shape{3}, {2, 2, 2});
    auto input = std::make_shared<pass::pattern::op::Label>(element::f32, Shape{2, 3});
    auto input_sq = std::make_shared<op::v1::Multiply>(input, input);
    auto sum_input = std::make_shared<op::v1::ReduceSum>(input, ov::op::v0::Constant::create(element::i64, {1}, {0}));
    auto square_sumed_input = std::make_shared<op::v1::Multiply>(sum_input, sum_input);
    auto sum_squared_input =
        std::make_shared<op::v1::ReduceSum>(input_sq, ov::op::v0::Constant::create(element::i64, {1}, {0}));
    auto avg_input_sum_sq = std::make_shared<op::v1::Divide>(square_sumed_input, N);
    auto xmu = std::make_shared<op::v1::Subtract>(sum_squared_input, avg_input_sum_sq);
    auto variance = std::make_shared<op::v1::Divide>(xmu, N);
    auto variance_label = std::make_shared<pass::pattern::op::Label>(variance, nullptr, NodeVector{variance});

    return variance_label;
}

static std::shared_ptr<pattern::op::Label> construct_mean_graph() {
    // construct mean;
    auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{2, 3});
    auto N = ov::op::v0::Constant::create(element::f32, Shape{3}, {2, 2, 2});
    auto sum_input1 = std::make_shared<op::v1::ReduceSum>(input, ov::op::v0::Constant::create(element::i64, {1}, {0}));
    auto mean = std::make_shared<op::v1::Divide>(sum_input1, N);
    auto mean_label = std::make_shared<pattern::op::Label>(mean, nullptr, NodeVector{mean});
    return mean_label;
}

class TestGraphRewrite : public ov::pass::GraphRewrite {
public:
    void construct_multiply_by_one() {
        // pattern #1 : a * 1 = a
        auto iconst1 = construct_constant_node(1);
        auto pattern = std::make_shared<pattern::op::Label>(iconst1);

        auto callback = [pattern](pattern::Matcher& m) {
            OPENVINO_DEBUG("In a callback for construct_multiply_by_one against ", m.get_match_root()->get_name());
            OPENVINO_ASSERT(m.get_match_root()->input_values().size() == 2);

            auto pattern_map = m.get_pattern_map();

            size_t const_node_index = m.get_match_root()->input_value(0).get_node_shared_ptr() == pattern_map[pattern];
            auto const_node = ov::as_type_ptr<ov::op::v0::Constant>(
                m.get_match_root()->input_value(const_node_index).get_node_shared_ptr());
            auto second_node = m.get_match_root()->input_value(const_node_index).get_node_shared_ptr();
            OPENVINO_DEBUG("second_node = ",
                           second_node->get_name(),
                           " , pattern = ",
                           pattern_map[pattern]->get_name());

            if (pattern_map[pattern]->get_element_type() != const_node->get_element_type() ||
                pattern_map[pattern]->get_shape() != const_node->get_shape()) {
                OPENVINO_DEBUG("Operands' types and/or shape don't match");
                return false;
            }

            auto const_values = const_node->get_vector<int32_t>();
            bool all_ones = std::all_of(begin(const_values), end(const_values), [](int e) {
                return e == 1;
            });

            if (!all_ones) {
                OPENVINO_DEBUG("Constant vector's values aren't equal to 1");
                return false;
            }

            ov::replace_node(m.get_match_root(), pattern_map[pattern]);
            return true;
        };

        auto m = make_shared<TestMatcher>(make_shared<op::v1::Multiply>(pattern, iconst1));
        auto match_pass = std::make_shared<ov::pass::MatcherPass>(
            m->get_name(),
            m,
            [m, callback](const std::shared_ptr<Node>& node) -> bool {
                OPENVINO_DEBUG("Running matcher ", m->get_name(), " on ", node);
                if (std::dynamic_pointer_cast<ov::pass::pattern::Matcher>(m)->match(node->output(0))) {
                    OPENVINO_DEBUG("Matcher ", m->get_name(), " matched ", node);
                    bool status = callback(*m.get());
                    // explicitly clear Matcher state because it holds pointers to matched nodes
                    m->clear_state();
                    return status;
                }
                m->clear_state();
                return false;
            },
            ov::pass::PassProperty::REQUIRE_STATIC_SHAPE);
        this->add_matcher(match_pass);
    }

    void construct_add_zero() {
        // pattern #2 : a + 0 = a
        auto iconst0 = construct_constant_node(0);
        auto pattern = std::make_shared<pattern::op::Label>(iconst0);

        auto callback = [pattern](pattern::Matcher& m) {
            OPENVINO_DEBUG("In a callback for construct_add_zero against ", m.get_match_root()->get_name());
            OPENVINO_ASSERT(m.get_match_root()->input_values().size() == 2);

            auto pattern_map = m.get_pattern_map();

            size_t const_node_index = m.get_match_root()->input_value(0).get_node_shared_ptr() == pattern_map[pattern];
            auto const_node = ov::as_type_ptr<ov::op::v0::Constant>(
                m.get_match_root()->input_value(const_node_index).get_node_shared_ptr());
            auto second_node = m.get_match_root()->input_value(const_node_index).get_node_shared_ptr();
            OPENVINO_DEBUG("second_node = ",
                           second_node->get_name(),
                           " , pattern = ",
                           pattern_map[pattern]->get_name());

            if (pattern_map[pattern]->get_element_type() != const_node->get_element_type() ||
                pattern_map[pattern]->get_shape() != const_node->get_shape()) {
                OPENVINO_DEBUG("Operands' types and/or shape don't match");
                return false;
            }

            auto const_values = const_node->get_vector<int>();
            bool all_zeros = std::all_of(begin(const_values), end(const_values), [](int e) {
                return e == 0;
            });

            if (!all_zeros) {
                OPENVINO_DEBUG("Constant vector's values aren't equal to 0");
                return false;
            }

            ov::replace_node(m.get_match_root(), pattern_map[pattern]);
            return true;
        };

        auto add = make_shared<op::v1::Add>(pattern, iconst0);
        auto m = make_shared<TestMatcher>(add);
        auto match_pass = std::make_shared<ov::pass::MatcherPass>(
            m->get_name(),
            m,
            [m, callback](const std::shared_ptr<Node>& node) -> bool {
                OPENVINO_DEBUG("Running matcher ", m->get_name(), " on ", node);
                if (std::dynamic_pointer_cast<ov::pass::pattern::Matcher>(m)->match(node->output(0))) {
                    OPENVINO_DEBUG("Matcher ", m->get_name(), " matched ", node);
                    bool status = callback(*m.get());
                    // explicitly clear Matcher state because it holds pointers to matched nodes
                    m->clear_state();
                    return status;
                }
                m->clear_state();
                return false;
            },
            ov::pass::PassProperty::REQUIRE_STATIC_SHAPE);
        this->add_matcher(match_pass);
    }

    TestGraphRewrite() : GraphRewrite() {
        construct_multiply_by_one();
        construct_add_zero();
    }
};

static void run_passes(pass::Manager& pass_manager,
                       shared_ptr<Node> graph,
                       std::vector<shared_ptr<op::v0::Parameter>> parms) {
    auto func = make_shared<Model>(graph, ParameterVector{parms});
    pass_manager.run_passes(func);
}

TEST(pattern, graph_rewrite) {
    Shape shape{};
    pass::Manager pass_manager;
    pass_manager.register_pass<TestGraphRewrite>();

    {
        auto a = make_shared<op::v0::Parameter>(element::i32, shape);
        auto b = make_shared<op::v0::Parameter>(element::i32, shape);
        auto c = make_shared<op::v0::Parameter>(element::i32, shape);
        auto iconst0 = construct_constant_node(0);
        auto graph_a = make_shared<op::v1::Add>(a, iconst0);
        auto graph_b = make_shared<op::v1::Add>(b, iconst0);

        auto f = std::make_shared<Model>(ov::NodeVector{a, b, graph_a, c, graph_b}, ParameterVector{a, b, c});
        pass_manager.run_passes(f);

        ASSERT_TRUE(graph_a->get_output_target_inputs(0).empty());
        ASSERT_TRUE(graph_b->get_output_target_inputs(0).empty());

        auto expected = ov::NodeVector{a, b, a, c, b};
        ASSERT_TRUE(count_ops_of_type<op::v1::Add>(f) == 0);
    }

    {
        auto a = make_shared<op::v0::Parameter>(element::i32, shape);
        auto b = make_shared<op::v0::Parameter>(element::i32, shape);
        auto iconst0 = construct_constant_node(0);
        auto sum = make_shared<op::v1::Add>(a, iconst0);
        auto graph = make_shared<op::v1::Add>(b, sum);
        run_passes(pass_manager, graph, {a, b});
        ASSERT_EQ(graph->input_value(1).get_node_shared_ptr(), a);
        ASSERT_EQ(graph->input_value(1), a->output(0));           // graph's input points to a's output
        ASSERT_TRUE(sum->output(0).get_target_inputs().empty());  // graph's input is removed from sum's target inptus
        ASSERT_TRUE(a->get_output_target_inputs(0).count(graph->input(1)));  // a's output feeds into graph's input
    }

    {
        auto a = make_shared<op::v0::Parameter>(element::i32, shape);
        auto b = make_shared<op::v0::Parameter>(element::i32, shape);
        auto iconst1 = construct_constant_node(1);
        auto mul = make_shared<op::v1::Multiply>(a, iconst1);
        auto graph = make_shared<op::v1::Add>(b, mul);
        run_passes(pass_manager, graph, {a, b});
        ASSERT_EQ(graph->input_value(1).get_node_shared_ptr(), a);
        ASSERT_EQ(graph->input_value(1), a->output(0));           // graph's input points to a's output
        ASSERT_TRUE(mul->output(0).get_target_inputs().empty());  // graph's input is removed from sum's target inputs
        ASSERT_TRUE(a->get_output_target_inputs(0).count(graph->input(1)));  // a's output feeds into graph's input
    }

    {
        auto a = make_shared<op::v0::Parameter>(element::i32, shape);
        auto b = make_shared<op::v0::Parameter>(element::i32, shape);
        auto iconst1 = construct_constant_node(1);
        auto multiply = make_shared<op::v1::Multiply>(make_shared<op::v1::Multiply>(a, iconst1), iconst1);
        multiply = make_shared<op::v1::Multiply>(make_shared<op::v1::Multiply>(multiply, iconst1), iconst1);
        auto graph = make_shared<op::v1::Add>(multiply, b);
        run_passes(pass_manager, graph, {a, b});
        ASSERT_EQ(graph->input_value(0).get_node_shared_ptr(), a);
        ASSERT_EQ(graph->input_value(0), a->output(0));                      // graph's input points to a's output
        ASSERT_TRUE(a->get_output_target_inputs(0).count(graph->input(0)));  // a's output feeds into graph's input
    }

    {
        auto a = make_shared<op::v0::Parameter>(element::i32, shape);
        auto b = make_shared<op::v0::Parameter>(element::i32, shape);
        auto iconst0 = construct_constant_node(0);
        auto iconst1 = construct_constant_node(1);
        auto mul = make_shared<op::v1::Multiply>(make_shared<op::v1::Add>(a, iconst0), iconst1);
        auto graph = make_shared<op::v1::Add>(b, make_shared<op::v1::Add>(iconst0, mul));
        run_passes(pass_manager, graph, {a, b});
        ASSERT_EQ(graph->input_value(1).get_node_shared_ptr(), a);
        ASSERT_EQ(graph->input_value(1), a->output(0));                      // graph's input points to a's output
        ASSERT_TRUE(a->get_output_target_inputs(0).count(graph->input(1)));  // a's output feeds into graph's input
    }

    {
        auto a = make_shared<op::v0::Parameter>(element::i32, shape);
        auto b = make_shared<op::v0::Parameter>(element::i32, shape);
        auto iconst1 = construct_constant_node(1);
        auto mul = make_shared<op::v1::Multiply>(iconst1, make_shared<op::v1::Multiply>(iconst1, a));
        mul = make_shared<op::v1::Multiply>(iconst1, make_shared<op::v1::Multiply>(iconst1, mul));
        auto graph = make_shared<op::v1::Add>(b, mul);
        run_passes(pass_manager, graph, {a, b});
        ASSERT_EQ(graph->input_value(1).get_node_shared_ptr(), a);
        ASSERT_EQ(graph->input_value(1), a->output(0));                      // graph's input points to a's output
        ASSERT_TRUE(a->get_output_target_inputs(0).count(graph->input(1)));  // a's output feeds into graph's input
    }
}

TEST(pattern, matcher) {
    Shape shape{};
    auto a = make_shared<op::v0::Parameter>(element::i32, shape);
    TestMatcher n;
    ASSERT_TRUE(n.match(a, a));
    ASSERT_EQ(n.get_matched_nodes(), (NodeVector{a}));

    auto abs = make_shared<op::v0::Abs>(a);

    auto false_pred = [](std::shared_ptr<Node> /* no */) {
        return false;
    };

    auto pattern = std::make_shared<pattern::op::Label>(a);
    ASSERT_TRUE(n.match(pattern, a));
    ASSERT_EQ(n.get_pattern_map()[pattern], a);
    ASSERT_EQ(n.get_matched_nodes(), (NodeVector{a}));

    auto pattern_false = std::make_shared<pattern::op::Label>(a, false_pred);
    ASSERT_FALSE(n.match(pattern_false, a));
    ASSERT_EQ(n.get_matched_nodes(), (NodeVector{}));

    auto b = make_shared<op::v0::Parameter>(element::i32, shape);

    auto is_bea = [](std::shared_ptr<Node> node) -> bool {
        return op::util::is_binary_elementwise_arithmetic(node);
    };
    auto bea = std::make_shared<pattern::op::Any>(a, is_bea, NodeVector{a, b});
    auto add_ab = std::make_shared<op::v1::Add>(a, b);
    ASSERT_TRUE(n.match(bea, add_ab));
    ASSERT_EQ(n.get_matched_nodes(), (NodeVector{add_ab, a, b}));
    ASSERT_TRUE(n.match(bea, std::make_shared<op::v1::Add>(b, a)));

    auto bea_false = std::make_shared<pattern::op::Any>(a, false_pred, NodeVector{a, b});
    ASSERT_FALSE(n.match(bea_false, std::make_shared<op::v1::Add>(a, b)));

    auto add_abs_b = std::make_shared<op::v1::Add>(abs, b);
    auto bea_any_of = std::make_shared<pattern::op::AnyOf>(a, is_bea, NodeVector{abs});
    ASSERT_TRUE(n.match(bea_any_of, add_abs_b));

    auto add_b_abs = std::make_shared<op::v1::Add>(b, abs);
    ASSERT_TRUE(n.match(bea_any_of, add_b_abs));

    auto bea_any_of_label = std::make_shared<pattern::op::Label>(a, nullptr, NodeVector{bea_any_of});
    ASSERT_TRUE(n.match(bea_any_of_label, add_b_abs));
    ASSERT_EQ(n.get_pattern_map()[bea_any_of_label], add_b_abs);

    auto abs_label = std::make_shared<pattern::op::Label>(a, nullptr, NodeVector{abs});
    auto bea_label_any_of = std::make_shared<pattern::op::AnyOf>(a, is_bea, NodeVector{abs_label});
    ASSERT_TRUE(n.match(bea_label_any_of, add_b_abs));
    ASSERT_EQ(n.get_pattern_map()[abs_label], abs);

    auto bea_label = std::make_shared<pattern::op::Label>(a, nullptr, NodeVector{bea});
    auto ab = std::make_shared<op::v1::Add>(a, b);
    ASSERT_TRUE(n.match(bea_label, ab));
    ASSERT_EQ(n.get_pattern_map()[bea_label], ab);

    auto d = make_shared<op::v0::Parameter>(element::i32, shape);
    ASSERT_FALSE(n.match(d, b));

    ASSERT_FALSE(n.match(std::make_shared<op::v1::Add>(abs, b), std::make_shared<op::v1::Add>(b, b)));
    ASSERT_EQ(n.get_matched_nodes(), (NodeVector{}));

    auto iconst1_0 = construct_constant_node(1);
    auto iconst1_1 = construct_constant_node(1);
    ASSERT_TRUE(n.match(make_shared<op::v1::Multiply>(pattern, iconst1_0),
                        make_shared<op::v1::Multiply>(a, iconst1_1)));  // different iconst
    ASSERT_EQ(n.get_pattern_map()[pattern], a);
    auto fconst1_0 = ov::op::v0::Constant::create(element::f32, shape, {1});
    auto patternf = std::make_shared<pattern::op::Label>(fconst1_0);
    ASSERT_TRUE(n.match(make_shared<op::v1::Multiply>(patternf, fconst1_0),
                        make_shared<op::v1::Multiply>(a, iconst1_1)));  // different iconst

    // Subgraph labels
    auto add = std::make_shared<op::v1::Add>(a, b);
    auto label = std::make_shared<pattern::op::Label>(add, nullptr, NodeVector{add});
    ASSERT_TRUE(n.match(label, add));
    ASSERT_EQ(n.get_pattern_map()[label], add);
    ASSERT_EQ(n.get_matched_nodes(), (NodeVector{add, add, a, b}));

    ASSERT_FALSE(n.match(label, std::make_shared<op::v1::Subtract>(a, b)));

    ASSERT_TRUE(n.match(make_shared<op::v0::Abs>(label), make_shared<op::v0::Abs>(add)));
    ASSERT_EQ(n.get_pattern_map()[label], add);

    // Correct argument order
    ASSERT_FALSE(n.match(make_shared<op::v1::Subtract>(b, a), make_shared<op::v1::Subtract>(a, b)));
    auto aab = make_shared<op::v1::Multiply>(a, make_shared<op::v1::Subtract>(a, b));
    auto paab = make_shared<op::v1::Multiply>(pattern, make_shared<op::v1::Subtract>(pattern, b));
    ASSERT_TRUE(n.match(paab, aab));
    auto aba = make_shared<op::v1::Multiply>(a, make_shared<op::v1::Subtract>(b, a));
    ASSERT_FALSE(n.match(paab, aba));
    auto paba = make_shared<op::v1::Multiply>(pattern, make_shared<op::v1::Subtract>(b, pattern));
    ASSERT_FALSE(n.match(paba, aab));

    // Correlations
    auto label1 = std::make_shared<pattern::op::Label>(a);
    auto tmp = std::make_shared<op::v1::Add>(label1, b);
    auto label2 = std::make_shared<pattern::op::Label>(tmp, nullptr, NodeVector{tmp});
    auto sub_label1 = std::make_shared<op::v1::Subtract>(label1, label2);
    auto sub_add = std::make_shared<op::v1::Subtract>(a, add);
    ASSERT_TRUE(n.match(sub_label1, sub_add));
    ASSERT_EQ(n.get_pattern_map()[label1], a);
    ASSERT_EQ(n.get_pattern_map()[label2], add);
    ASSERT_EQ(n.get_matched_nodes(), (NodeVector{sub_add, a, add, add, a, b}));

    ASSERT_FALSE(n.match(sub_label1, std::make_shared<op::v1::Subtract>(add, a)));

    auto add_label1 = std::make_shared<op::v1::Add>(label1, label2);
    ASSERT_TRUE(n.match(add_label1, std::make_shared<op::v1::Add>(add, a)));
    ASSERT_EQ(n.get_pattern_map()[label1], a);
    ASSERT_EQ(n.get_pattern_map()[label2], add);

    // Or
    ASSERT_TRUE(n.match(std::make_shared<pattern::op::Or>(OutputVector{std::make_shared<op::v1::Add>(a, b),
                                                                       std::make_shared<op::v1::Subtract>(a, b)}),
                        std::make_shared<op::v1::Add>(a, b)));
    ASSERT_TRUE(n.match(std::make_shared<pattern::op::Or>(OutputVector{std::make_shared<op::v1::Add>(a, b),
                                                                       std::make_shared<op::v1::Subtract>(a, b)}),
                        std::make_shared<op::v1::Subtract>(a, b)));

    // strict mode
    {
        TestMatcher sm(Output<Node>{}, "TestMatcher", true);
        // exact shape and type
        auto scalar_param = make_shared<op::v0::Parameter>(element::i32, Shape{});
        auto label_dynamic_shape = make_shared<pattern::op::Label>(element::i32, PartialShape::dynamic());
        auto param = make_shared<op::v0::Parameter>(element::f32, Shape{});
        ASSERT_TRUE(sm.match(label_dynamic_shape, scalar_param));
        // wrong type
        auto scalar_param_wrong_type = make_shared<op::v0::Parameter>(element::f32, Shape{});
        ASSERT_FALSE(sm.match(label, scalar_param_wrong_type));
        // dynamic dimension
        auto label_dynamic_dimension =
            make_shared<pattern::op::Label>(element::i32, PartialShape{Dimension::dynamic()});
        auto vector_param = make_shared<op::v0::Parameter>(element::i32, Shape{10});
        ASSERT_TRUE(sm.match(label_dynamic_dimension, vector_param));
        // dynamic type
        auto label_dynamic_type = make_shared<pattern::op::Label>(element::dynamic, PartialShape{Dimension::dynamic()});
        ASSERT_TRUE(sm.match(label_dynamic_type, vector_param));
    }
}

// match optional nodes with single input
TEST(pattern, optional_match_node_with_single_input) {
    Shape shape{1, 2, 3};
    auto model_input_0 = make_shared<op::v0::Parameter>(element::f32, shape);
    auto model_exp = make_shared<op::v0::Exp>(model_input_0);

    auto model_const_input_0 =
        make_shared<op::v0::Constant>(element::f32, shape, std::vector<float>(ov::shape_size(shape), 0.1f));
    auto model_const_exp = make_shared<op::v0::Exp>(model_const_input_0);

    TestMatcher matcher;
    // is_on_const_path
    auto param_predicate = [](const Output<Node>& output) {
        return !ov::op::util::is_on_constant_path(output);
    };

    auto pattern_in_0 = ov::pass::pattern::any_input();
    auto pattern_exp = ov::pass::pattern::wrap_type<ov::op::v0::Exp>({pattern_in_0});

    // without optional op
    {
        ASSERT_TRUE(matcher.match(pattern_exp, model_exp));
        ASSERT_TRUE(matcher.match(pattern_exp, model_const_exp));
    }

    // with optional: 1 type
    {
        auto pattern = ov::pass::pattern::optional<op::v0::Abs>(pattern_exp);
        ASSERT_TRUE(matcher.match(pattern, model_exp));
        ASSERT_TRUE(matcher.match(pattern, model_const_exp));
        ASSERT_TRUE(matcher.match(pattern, make_shared<op::v0::Abs>(model_exp)));
        ASSERT_TRUE(matcher.match(pattern, make_shared<op::v0::Abs>(model_const_exp)));
        // negative scenario
        ASSERT_FALSE(matcher.match(pattern, make_shared<op::v0::Relu>(model_exp)));
        ASSERT_FALSE(matcher.match(pattern, make_shared<op::v0::Relu>(model_const_exp)));
    }

    // with optional: 1 type and predicate
    {
        auto pattern = ov::pass::pattern::optional<op::v0::Abs>(pattern_exp, param_predicate);
        ASSERT_TRUE(matcher.match(pattern, model_exp));
        ASSERT_TRUE(matcher.match(pattern, model_const_exp));
        ASSERT_TRUE(matcher.match(pattern, model_exp));
        ASSERT_TRUE(matcher.match(pattern, model_const_exp));
        ASSERT_TRUE(matcher.match(pattern, make_shared<op::v0::Abs>(model_exp)));
        ASSERT_FALSE(matcher.match(pattern, make_shared<op::v0::Abs>(model_const_exp)));
    }

    // with 2 types
    {
        auto pattern = ov::pass::pattern::optional<op::v0::Abs, op::v0::Relu>(pattern_exp);
        ASSERT_TRUE(matcher.match(pattern, model_exp));
        ASSERT_TRUE(matcher.match(pattern, model_const_exp));
        ASSERT_TRUE(matcher.match(pattern, make_shared<op::v0::Abs>(model_exp)));
        ASSERT_TRUE(matcher.match(pattern, make_shared<op::v0::Relu>(model_const_exp)));

        ASSERT_TRUE(matcher.match(pattern, make_shared<op::v0::Exp>(model_exp)));
        // negative
        ASSERT_FALSE(matcher.match(pattern, make_shared<op::v0::Cos>(model_exp)));
        ASSERT_FALSE(matcher.match(pattern, make_shared<op::v0::Cosh>(model_const_exp)));
    }

    // with 2 types and predicate
    {
        auto pattern = ov::pass::pattern::optional<op::v0::Abs, op::v0::Relu>(pattern_exp, param_predicate);

        ASSERT_TRUE(matcher.match(pattern, make_shared<op::v0::Abs>(model_exp)));
        // negative
        ASSERT_FALSE(matcher.match(pattern, make_shared<op::v0::Relu>(model_const_exp)));
        ASSERT_FALSE(matcher.match(pattern, make_shared<op::v0::Cos>(model_exp)));
        // true: match exp without optional + exp as an input
        ASSERT_TRUE(matcher.match(pattern, make_shared<op::v0::Exp>(model_const_exp)));
    }
}

// match optional nodes with multi input where order in not important
TEST(pattern, optional_match_cumulative_node_with_multi_input) {
    Shape shape{1, 2, 3};
    auto model_input_0 = make_shared<op::v0::Parameter>(element::f32, shape);
    auto model_relu = make_shared<op::v0::Relu>(model_input_0);
    auto model_input_1 =
        make_shared<op::v0::Constant>(element::f32, shape, std::vector<float>(ov::shape_size(shape), .1f));
    auto model_add = std::make_shared<op::v1::Add>(model_relu, model_input_1);
    auto model_equal = std::make_shared<op::v1::Equal>(model_relu, model_input_1);

    auto arithmetic_pattern = [](const ov::Output<ov::Node>& output) {
        return ov::op::util::is_binary_elementwise_arithmetic(output.get_node_shared_ptr());
    };
    auto relu_pattern = [](const ov::Output<ov::Node>& output) {
        return output.get_node_shared_ptr()->get_type_info().is_castable(ov::op::v0::Relu::get_type_info_static());
    };

    TestMatcher matcher;
    auto pattern_input_0 = ov::pass::pattern::any_input(relu_pattern);
    auto pattern_input_1 = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();

    // 1 type, correct in_order, without predicate
    {
        auto pattern = ov::pass::pattern::optional<op::v1::Add>({pattern_input_0, pattern_input_1});
        ASSERT_FALSE(matcher.match(pattern, model_input_0));
        ASSERT_TRUE(matcher.match(pattern, model_relu));
        ASSERT_FALSE(matcher.match(pattern, model_input_1));
        ASSERT_TRUE(matcher.match(pattern, model_add));
        ASSERT_FALSE(matcher.match(pattern, model_equal));
        auto pattern_negative = ov::pass::pattern::optional<op::v1::Equal>({pattern_input_0, pattern_input_1});
        ASSERT_FALSE(matcher.match(pattern_negative, model_add));
    }

    // 2 type, correct in_order, without predicate
    {
        auto pattern = ov::pass::pattern::optional<op::v1::Add, op::v1::Equal>({pattern_input_0, pattern_input_1});
        ASSERT_FALSE(matcher.match(pattern, model_input_0));
        ASSERT_TRUE(matcher.match(pattern, model_relu));
        ASSERT_FALSE(matcher.match(pattern, model_input_1));
        ASSERT_TRUE(matcher.match(pattern, model_add));
        ASSERT_TRUE(matcher.match(pattern, model_equal));
        // negative
        ASSERT_FALSE(matcher.match(pattern, std::make_shared<op::v1::Subtract>(model_relu, model_input_1)));
    }

    // 2 type, correct in_order, with predicate
    {
        auto pattern = ov::pass::pattern::optional<op::v1::Add, op::v1::Equal>({pattern_input_0, pattern_input_1},
                                                                               arithmetic_pattern);
        ASSERT_FALSE(matcher.match(pattern, model_input_0));
        ASSERT_TRUE(matcher.match(pattern, model_relu));
        ASSERT_FALSE(matcher.match(pattern, model_input_1));
        ASSERT_TRUE(matcher.match(pattern, model_add));
        ASSERT_FALSE(matcher.match(pattern, model_equal));
    }

    // 2 type, reverse in_order, without predicate
    {
        auto pattern = ov::pass::pattern::optional<op::v1::Add, op::v1::Equal>({pattern_input_1, pattern_input_0});
        ASSERT_FALSE(matcher.match(pattern, model_input_0));
        ASSERT_FALSE(matcher.match(pattern, model_relu));
        ASSERT_TRUE(matcher.match(pattern, model_input_1));
        ASSERT_TRUE(matcher.match(pattern, model_add));
        ASSERT_TRUE(matcher.match(pattern, model_equal));
    }

    // 2 type, reverse in_order, with predicate
    {
        auto pattern = ov::pass::pattern::optional<op::v1::Add, op::v1::Equal>({pattern_input_1, pattern_input_0},
                                                                               arithmetic_pattern);
        ASSERT_FALSE(matcher.match(pattern, model_input_0));
        ASSERT_FALSE(matcher.match(pattern, model_relu));
        ASSERT_TRUE(matcher.match(pattern, model_input_1));
        ASSERT_TRUE(matcher.match(pattern, model_add));
        ASSERT_FALSE(matcher.match(pattern, model_equal));
    }
}

// match optional nodes with multi input where order in important
TEST(pattern, optional_match_node_with_multi_input_order_is_cruical) {
    Shape shape{1, 2, 3};
    auto model_input_0 = make_shared<op::v0::Parameter>(element::f32, shape);
    auto model_relu = make_shared<op::v0::Relu>(model_input_0);
    auto model_input_1 =
        make_shared<op::v0::Constant>(element::f32, shape, std::vector<float>(ov::shape_size(shape), .1f));
    auto model_subtract = std::make_shared<op::v1::Subtract>(model_relu, model_input_1);
    auto model_greater = std::make_shared<op::v1::Greater>(model_relu, model_input_1);

    auto arithmetic_pattern = [](const ov::Output<ov::Node>& output) {
        return ov::op::util::is_binary_elementwise_arithmetic(output.get_node_shared_ptr());
    };
    auto relu_pattern = [](const ov::Output<ov::Node>& output) {
        return output.get_node_shared_ptr()->get_type_info().is_castable(ov::op::v0::Relu::get_type_info_static());
    };

    TestMatcher matcher;
    auto pattern_input_0 = ov::pass::pattern::any_input(relu_pattern);
    auto pattern_input_1 = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();

    // 1 type, correct in_order, without predicate
    {
        auto pattern = ov::pass::pattern::optional<op::v1::Subtract>({pattern_input_0, pattern_input_1});
        ASSERT_FALSE(matcher.match(pattern, model_input_0));
        ASSERT_TRUE(matcher.match(pattern, model_relu));
        ASSERT_FALSE(matcher.match(pattern, model_input_1));
        ASSERT_TRUE(matcher.match(pattern, model_subtract));
        ASSERT_FALSE(matcher.match(pattern, model_greater));
        auto pattern_negative = ov::pass::pattern::optional<op::v1::Greater>({pattern_input_0, pattern_input_1});
        ASSERT_FALSE(matcher.match(pattern_negative, model_subtract));
    }

    // 2 type, correct in_order, without predicate
    {
        auto pattern =
            ov::pass::pattern::optional<op::v1::Subtract, op::v1::Greater>({pattern_input_0, pattern_input_1});
        ASSERT_FALSE(matcher.match(pattern, model_input_0));
        ASSERT_TRUE(matcher.match(pattern, model_relu));
        ASSERT_FALSE(matcher.match(pattern, model_input_1));
        ASSERT_TRUE(matcher.match(pattern, model_greater));
        ASSERT_TRUE(matcher.match(pattern, model_subtract));
        // negative
        ASSERT_FALSE(matcher.match(pattern, std::make_shared<op::v1::Add>(model_relu, model_input_1)));
    }

    // 2 type, correct in_order, with predicate
    {
        auto pattern =
            ov::pass::pattern::optional<op::v1::Subtract, op::v1::Greater>({pattern_input_0, pattern_input_1},
                                                                           arithmetic_pattern);
        ASSERT_FALSE(matcher.match(pattern, model_input_0));
        ASSERT_TRUE(matcher.match(pattern, model_relu));
        ASSERT_FALSE(matcher.match(pattern, model_input_1));
        ASSERT_TRUE(matcher.match(pattern, model_subtract));
        ASSERT_FALSE(matcher.match(pattern, model_greater));
    }

    // 2 type, reverse in_order, without predicate
    {
        auto pattern =
            ov::pass::pattern::optional<op::v1::Subtract, op::v1::Greater>({pattern_input_1, pattern_input_0});
        ASSERT_FALSE(matcher.match(pattern, model_input_0));
        ASSERT_FALSE(matcher.match(pattern, model_relu));
        ASSERT_TRUE(matcher.match(pattern, model_input_1));
        ASSERT_FALSE(matcher.match(pattern, model_subtract));
        ASSERT_FALSE(matcher.match(pattern, model_greater));
    }

    // 2 type, reverse in_order, with predicate
    {
        auto pattern =
            ov::pass::pattern::optional<op::v1::Subtract, op::v1::Greater>({pattern_input_1, pattern_input_0},
                                                                           arithmetic_pattern);
        ASSERT_FALSE(matcher.match(pattern, model_input_0));
        ASSERT_FALSE(matcher.match(pattern, model_relu));
        ASSERT_TRUE(matcher.match(pattern, model_input_1));
        ASSERT_FALSE(matcher.match(pattern, model_subtract));
        ASSERT_FALSE(matcher.match(pattern, model_greater));
    }
}

// complex pattern matching with `optional` and `wrap_type`
TEST(pattern, optional_complex_pattern_matching) {
    auto model_param = make_shared<op::v0::Parameter>(element::f32, ov::Shape{2, 3, 4});
    auto model_constant = make_shared<op::v0::Constant>(element::i32, ov::Shape{3}, std::vector<int>{2, 0, 1});
    auto model_abs = make_shared<op::v0::Abs>(model_param);
    auto model_transpose_negative = std::make_shared<op::v1::Transpose>(model_abs, model_constant);
    auto model_negative = std::make_shared<op::v0::Relu>(model_transpose_negative);

    auto model_relu = make_shared<op::v0::Relu>(model_param);
    auto model_transpose_positive = std::make_shared<op::v1::Transpose>(model_relu, model_constant);
    auto model_positive = std::make_shared<op::v0::Relu>(model_transpose_positive);

    auto pattern_param = ov::pass::pattern::any_input();
    auto pattern_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto pattern_relu = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({pattern_param});
    auto pattern_transpose = ov::pass::pattern::optional<op::v1::Transpose>({pattern_relu, pattern_constant});
    auto pattern = ov::pass::pattern::wrap_type<op::v0::Relu>({pattern_transpose});

    TestMatcher matcher;
    ASSERT_FALSE(matcher.match(pattern, model_negative));
    ASSERT_TRUE(matcher.match(pattern, model_positive));
}

TEST(pattern, optional_full_match) {
    Shape shape{};
    auto model_input = std::make_shared<op::v0::Parameter>(element::i32, shape);
    auto model_relu = std::make_shared<op::v0::Relu>(model_input);
    auto model_relu1 = std::make_shared<op::v0::Relu>(model_relu->output(0));

    auto pattern_relu = ov::pass::pattern::optional<op::v0::Relu>();
    auto pattern_relu1 = std::make_shared<op::v0::Relu>(pattern_relu->output(0));

    TestMatcher tm;

    ASSERT_TRUE(tm.match(pattern_relu1, model_relu1));
}

TEST(pattern, optional_subgraph) {
    Shape shape{};
    auto model_input = std::make_shared<op::v0::Parameter>(element::i32, shape);
    auto model_relu = std::make_shared<op::v0::Relu>(model_input);
    auto model_abs = std::make_shared<op::v0::Abs>(model_relu->output(0));
    auto model_exp = std::make_shared<op::v0::Exp>(model_abs->output(0));

    auto pattern_input = ov::pass::pattern::optional<op::v0::Parameter>();
    auto pattern_relu = ov::pass::pattern::optional<op::v0::Relu>(pattern_input);
    auto pattern_abs = ov::pass::pattern::optional<op::v0::Abs>(pattern_relu);
    auto pattern_exp = ov::pass::pattern::optional<op::v0::Exp>(pattern_abs);

    TestMatcher tm;
    ASSERT_TRUE(tm.match(pattern_exp, model_exp));
    ASSERT_TRUE(tm.match(pattern_exp, model_abs));
    ASSERT_TRUE(tm.match(pattern_exp, model_relu));
    ASSERT_TRUE(tm.match(pattern_exp, model_input));
    auto constant = make_shared<op::v0::Constant>(element::i32, ov::Shape{3}, std::vector<int>{2, 0, 1});
    ASSERT_FALSE(tm.match(pattern_exp, constant));
}

// TODO: Issue: 139835
// The pattern matching does not support operations with optional inputs.
// For example, ov::op::v5::NonMaxSupression can be created without some optional input nodes (like
// `max_output_boxes_per_class`) (In case we would not specify input in constructor, the node input won't be created by
// default as a constant). Arguments matching will be failed due to different number of pattern and graph input args.
// Check `bool Matcher::match_arguments(Node* pattern_node, const std::shared_ptr<Node>& graph_node)`
TEST(pattern, DISABLED_optional_match_node_with_optional_input) {
    auto model_in_0 = std::make_shared<ov::op::v0::Parameter>(element::f32, ov::Shape{3, 100, 4});
    auto model_in_1 = std::make_shared<ov::op::v0::Parameter>(element::f32, ov::Shape{3, 5, 100});
    auto model_in_2 = std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape{}, std::vector<int>{10});
    auto model_nms_without_optional = std::make_shared<ov::op::v5::NonMaxSuppression>(model_in_0, model_in_1);
    auto model_nms_with_optional = std::make_shared<ov::op::v5::NonMaxSuppression>(model_in_0, model_in_1, model_in_2);

    TestMatcher matcher;
    auto positive_predicate = [](const Output<Node>& output) {
        return true;
    };
    auto negative_predicate = [](const Output<Node>& output) {
        return false;
    };

    auto pattern_in_0 = ov::pass::pattern::any_input();
    auto pattern_in_1 = ov::pass::pattern::any_input();

    // checking matching optional input
    {
        auto pattern_in_2 = ov::pass::pattern::optional<ov::op::v0::Constant>();
        ASSERT_TRUE(matcher.match(pattern_in_2, model_in_2));
    }

    // checking matching optional input: negative
    {
        auto pattern_in_2 = ov::pass::pattern::optional<ov::op::v0::Parameter>();
        ASSERT_FALSE(matcher.match(pattern_in_2, model_in_2));
    }

    // validate matching: pattern is without optional input
    {
        auto pattern_nms = ov::pass::pattern::wrap_type<ov::op::v5::NonMaxSuppression>({pattern_in_0, pattern_in_1});
        ASSERT_FALSE(matcher.match(pattern_nms, model_nms_with_optional));
        ASSERT_TRUE(matcher.match(pattern_nms, model_nms_without_optional));
    }

    // validate matching: pattern is with optional input
    {
        auto pattern_in_2 = ov::pass::pattern::optional<ov::op::v0::Constant>();
        auto pattern_nms =
            ov::pass::pattern::wrap_type<ov::op::v5::NonMaxSuppression>({pattern_in_0, pattern_in_1, pattern_in_2});
        ASSERT_TRUE(matcher.match(pattern_nms, model_nms_with_optional));
        ASSERT_TRUE(matcher.match(pattern_nms, model_nms_without_optional));
    }

    // validate matching: pattern is with optional input with positive predicate
    {
        auto pattern_in_2 = ov::pass::pattern::optional<ov::op::v0::Constant>(positive_predicate);
        auto pattern_nms =
            ov::pass::pattern::wrap_type<ov::op::v5::NonMaxSuppression>({pattern_in_0, pattern_in_1, pattern_in_2});
        ASSERT_TRUE(matcher.match(pattern_nms, model_nms_with_optional));
        ASSERT_TRUE(matcher.match(pattern_nms, model_nms_without_optional));
    }

    // validate matching: pattern is with optional input with negative predicate
    {
        auto pattern_in_2 = ov::pass::pattern::optional<ov::op::v0::Constant>(negative_predicate);
        auto pattern_nms =
            ov::pass::pattern::wrap_type<ov::op::v5::NonMaxSuppression>({pattern_in_0, pattern_in_1, pattern_in_2});
        ASSERT_FALSE(matcher.match(pattern_nms, model_nms_with_optional));
        ASSERT_TRUE(matcher.match(pattern_nms, model_nms_without_optional));
    }

    // validate matching: pattern is with optional input and optional relu as input 2
    {
        auto pattern_in_2 = ov::pass::pattern::optional<ov::op::v0::Constant>();
        auto pattern_relu = ov::pass::pattern::optional<ov::op::v0::Relu>(pattern_in_2);
        auto pattern_nms =
            ov::pass::pattern::wrap_type<ov::op::v5::NonMaxSuppression>({pattern_in_0, pattern_in_1, pattern_relu});
        ASSERT_TRUE(matcher.match(pattern_nms, model_nms_with_optional));
        ASSERT_TRUE(matcher.match(pattern_nms, model_nms_without_optional));
    }

    // validate matching: pattern is with optional input
    {
        auto pattern_in_2 = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
        auto pattern_relu_2 = ov::pass::pattern::optional<ov::op::v0::Relu>(pattern_in_2);
        auto pattern_nms =
            ov::pass::pattern::wrap_type<ov::op::v5::NonMaxSuppression>({pattern_in_0, pattern_in_1, pattern_relu_2});
        ASSERT_FALSE(matcher.match(pattern_nms, model_nms_with_optional));
        ASSERT_FALSE(matcher.match(pattern_nms, model_nms_without_optional));
    }
}

TEST(pattern, mean) {
    // construct mean
    TestMatcher n;

    auto input = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3});
    auto N = ov::op::v0::Constant::create(element::f32, Shape{3}, {2, 2, 2});
    auto sum_input1 = std::make_shared<op::v1::ReduceSum>(input, ov::op::v0::Constant::create(element::i64, {1}, {0}));
    auto mean = std::make_shared<op::v1::Divide>(sum_input1, N);

    auto mean_graph = construct_mean_graph();
    ASSERT_TRUE(n.match(mean_graph, mean));
    ASSERT_EQ(n.get_pattern_map()[mean_graph], mean);
}

TEST(pattern, variance) {
    // construct variance
    TestMatcher n;
    auto N = ov::op::v0::Constant::create(element::f32, Shape{3}, {2, 2, 2});
    auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{2, 3});
    auto input_sq = std::make_shared<op::v1::Multiply>(input, input);
    auto sum_input = std::make_shared<op::v1::ReduceSum>(input, ov::op::v0::Constant::create(element::i64, {1}, {0}));
    auto square_sumed_input = std::make_shared<op::v1::Multiply>(sum_input, sum_input);
    auto sum_squared_input =
        std::make_shared<op::v1::ReduceSum>(input_sq, ov::op::v0::Constant::create(element::i64, {1}, {0}));
    auto avg_input_sum_sq = std::make_shared<op::v1::Divide>(square_sumed_input, N);
    auto xmu = std::make_shared<op::v1::Subtract>(sum_squared_input, avg_input_sum_sq);
    auto variance = std::make_shared<op::v1::Divide>(xmu, N);

    auto var_graph = construct_variance_graph();
    ASSERT_TRUE(n.match(var_graph, variance));
    ASSERT_EQ(n.get_pattern_map()[var_graph], variance);
}

TEST(pattern, previous_matches) {
    Shape shape{};
    ov::pass::pattern::Matcher::PatternMap previous_matches;
    auto a = make_shared<op::v0::Parameter>(element::i32, shape);
    auto b = make_shared<op::v0::Parameter>(element::i32, shape);
    auto pattern = std::make_shared<pattern::op::Label>(b);
    auto abs = make_shared<op::v0::Abs>(a);
    auto add = make_shared<op::v1::Add>(abs, b);
    {
        pattern::Matcher n(make_shared<op::v1::Add>(pattern, b));
        ASSERT_TRUE(n.match(add, previous_matches));
        ASSERT_EQ(n.get_pattern_map()[pattern], abs);
    }

    {
        pattern::Matcher n(make_shared<op::v1::Add>(pattern, b));
        previous_matches.insert(std::make_pair(pattern, a));
        ASSERT_FALSE(n.match(add, previous_matches));
    }
}

TEST(pattern, test_sort) {
    Shape shape{};

    auto a = make_shared<op::v0::Parameter>(element::i32, shape);
    auto b = make_shared<op::v0::Parameter>(element::i32, shape);
    auto abs1 = make_shared<op::v0::Abs>(a);
    auto abs2 = make_shared<op::v0::Abs>(b);
    shared_ptr<Node> add = make_shared<op::v1::Add>(abs1, abs2);

    auto pa = make_shared<op::v0::Parameter>(element::i32, shape);
    auto pb = make_shared<op::v0::Parameter>(element::i32, shape);
    auto pabs1 = make_shared<op::v0::Abs>(pa);
    auto pabs1_label = std::make_shared<pattern::op::Label>(pabs1);
    auto pabs2 = make_shared<op::v0::Abs>(b);
    shared_ptr<Node> padd = make_shared<op::v1::Add>(pabs1_label, pabs2);

    {
        pattern::Matcher n1(padd);
        ASSERT_TRUE(n1.match(add));
        auto r1 = n1.get_pattern_map()[pabs1_label];
        ASSERT_TRUE(n1.match(add));
        ASSERT_EQ(r1, n1.get_pattern_map()[pabs1_label]);
    }
}

TEST(pattern, is_contained_match) {
    Shape shape{};
    auto a = make_shared<op::v0::Parameter>(element::i32, shape);
    auto absn = make_shared<op::v0::Abs>(a);
    TestMatcher n;

    auto label_a = std::make_shared<pattern::op::Label>(a);
    auto label_abs = make_shared<op::v0::Abs>(a);
    ASSERT_TRUE(n.match(label_abs, absn));
    auto result_absn = make_shared<ov::op::v0::Result>(absn);
    ASSERT_TRUE(n.is_contained_match());

    auto absn2 = make_shared<op::v0::Abs>(absn);
    auto result_absn2 = make_shared<ov::op::v0::Result>(absn2);
    auto label_abs2 = make_shared<op::v0::Abs>(label_abs);
    ASSERT_TRUE(n.match(label_abs2, absn2));
    ASSERT_FALSE(n.is_contained_match());
}

TEST(pattern, wrap_type_single_op) {
    auto a = make_shared<op::v0::Parameter>(element::f32, Shape{1, 3, 64, 64});
    auto b = make_shared<op::v0::Abs>(a);
    auto c = make_shared<ov::op::v0::Relu>(a);
    auto mul1 = make_shared<op::v1::Multiply>(a, ov::op::v0::Constant::create(element::f32, Shape{}, {1}));
    auto mul2 = make_shared<op::v1::Multiply>(ov::op::v0::Constant::create(element::f32, Shape{}, {1}), a);

    {
        auto m = pattern::wrap_type<op::v0::Abs>();
        auto matcher = std::make_shared<pattern::Matcher>(m, "AbsMatcher");
        ASSERT_TRUE(matcher->match(static_pointer_cast<Node>(b)));
        ASSERT_EQ(matcher->get_matched_nodes().size(), 1);
        ASSERT_EQ(matcher->get_matched_nodes()[0], b);
        ASSERT_EQ(matcher->get_pattern_map().count(m), 1);
        ASSERT_FALSE(matcher->match(static_pointer_cast<Node>(c)));
    }
    {
        auto m1 = pattern::wrap_type<op::v0::Parameter>();
        auto m2 = pattern::wrap_type<op::v0::Abs>({m1});
        auto matcher = std::make_shared<pattern::Matcher>(m2, "ParamAbsMatcher");
        ASSERT_TRUE(matcher->match(static_pointer_cast<Node>(b)));
        ASSERT_EQ(matcher->get_matched_nodes().size(), 2);
        ASSERT_EQ(matcher->get_pattern_map().count(m1), 1);
        ASSERT_EQ(matcher->get_pattern_map().count(m2), 1);
        ASSERT_FALSE(matcher->match(static_pointer_cast<Node>(c)));
    }
    {
        auto m1 =
            pattern::wrap_type<op::v1::Multiply>({pattern::any_input(), pattern::wrap_type<ov::op::v0::Constant>()});
        auto matcher = std::make_shared<pattern::Matcher>(m1, "MultiplyMatcher");
        ASSERT_TRUE(matcher->match(static_pointer_cast<Node>(mul1)));
        ASSERT_TRUE(matcher->match(static_pointer_cast<Node>(mul2)));
    }
    {
        auto m1 =
            pattern::wrap_type<op::v1::Multiply>({pattern::wrap_type<ov::op::v0::Constant>(), pattern::any_input()});
        auto matcher = std::make_shared<pattern::Matcher>(m1, "MultiplyMatcher");
        ASSERT_TRUE(matcher->match(static_pointer_cast<Node>(mul1)));
        ASSERT_TRUE(matcher->match(static_pointer_cast<Node>(mul2)));
    }
}

TEST(pattern, wrap_type_multi_op) {
    auto a = make_shared<op::v0::Parameter>(element::f32, Shape{1, 3, 64, 64});
    auto b = make_shared<op::v0::Abs>(a);
    auto c = make_shared<ov::op::v0::Relu>(a);
    auto mul = make_shared<op::v1::Multiply>(a, ov::op::v0::Constant::create(element::f32, Shape{}, {1}));
    auto add = make_shared<op::v1::Add>(ov::op::v0::Constant::create(element::f32, Shape{}, {1}), a);

    {
        auto m = pattern::wrap_type<op::v1::Multiply, op::v1::Add>();
        auto matcher = std::make_shared<pattern::Matcher>(m, "MulAddMatcher");
        ASSERT_TRUE(matcher->match(mul->output(0)));
        ASSERT_EQ(matcher->get_matched_nodes().size(), 1);
        ASSERT_EQ(matcher->get_matched_nodes()[0], mul);
        ASSERT_EQ(matcher->get_pattern_map().count(m), 1);

        ASSERT_TRUE(matcher->match(add->output(0)));
        ASSERT_EQ(matcher->get_matched_nodes().size(), 1);
        ASSERT_EQ(matcher->get_matched_nodes()[0], add);
        ASSERT_EQ(matcher->get_pattern_map().count(m), 1);

        ASSERT_FALSE(matcher->match(static_pointer_cast<Node>(a)));
        ASSERT_FALSE(matcher->match(static_pointer_cast<Node>(b)));
        ASSERT_FALSE(matcher->match(static_pointer_cast<Node>(c)));
    }
    {
        auto m = pattern::wrap_type<op::util::BinaryElementwiseArithmetic>();
        auto matcher = std::make_shared<pattern::Matcher>(m, "ElementwiseMatcher");
        ASSERT_TRUE(matcher->match(mul->output(0)));
        ASSERT_EQ(matcher->get_matched_nodes().size(), 1);
        ASSERT_EQ(matcher->get_matched_nodes()[0], mul);
        ASSERT_EQ(matcher->get_pattern_map().count(m), 1);

        ASSERT_TRUE(matcher->match(add->output(0)));
        ASSERT_EQ(matcher->get_matched_nodes().size(), 1);
        ASSERT_EQ(matcher->get_matched_nodes()[0], add);
        ASSERT_EQ(matcher->get_pattern_map().count(m), 1);

        ASSERT_FALSE(matcher->match(static_pointer_cast<Node>(a)));
        ASSERT_FALSE(matcher->match(static_pointer_cast<Node>(b)));
        ASSERT_FALSE(matcher->match(static_pointer_cast<Node>(c)));
    }
}

TEST(pattern, simple_model_and_pattern) {
    // Create a sample model
    PartialShape shape{2, 2};
    auto model_param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_param2 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_add = std::make_shared<ov::op::v1::Add>(model_param1->output(0), model_param2->output(0));
    auto model_param3 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_mul = std::make_shared<ov::op::v0::MatMul>(model_add->output(0), model_param3->output(0), false, false);
    auto model_abs = std::make_shared<ov::op::v0::Abs>(model_mul->output(0));
    auto model_relu = std::make_shared<ov::op::v0::Relu>(model_abs->output(0));
    auto model_result = std::make_shared<ov::op::v0::Result>(model_relu->output(0));

    // Create a sample model
    auto pattern_mul = std::make_shared<ov::op::v0::MatMul>(pattern::any_input(), pattern::any_input(), false, false);
    auto pattern_abs = std::make_shared<ov::op::v0::Abs>(pattern_mul->output(0));
    auto pattern_relu = std::make_shared<ov::op::v0::Relu>(pattern_abs->output(0));

    // Create a matcher and try to match the nodes
    TestMatcher tm;

    // Should perfectly match
    ASSERT_TRUE(tm.match(pattern_relu, model_relu));
}

TEST(pattern, simple_model_and_pattern_wrap_type) {
    // Create a sample model
    PartialShape shape{2, 2};
    auto model_param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_param2 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_add = std::make_shared<ov::op::v1::Add>(model_param1->output(0), model_param2->output(0));
    auto model_param3 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_mul = std::make_shared<ov::op::v0::MatMul>(model_add->output(0), model_param3->output(0), false, false);
    auto model_abs = std::make_shared<ov::op::v0::Abs>(model_mul->output(0));
    auto model_relu = std::make_shared<ov::op::v0::Relu>(model_abs->output(0));
    auto model_result = std::make_shared<ov::op::v0::Result>(model_relu->output(0));

    // Create a sample model
    auto pattern_mul = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({pattern::any_input(), pattern::any_input()});
    auto pattern_abs = ov::pass::pattern::wrap_type<ov::op::v0::Abs>({pattern_mul->output(0)});
    auto pattern_relu = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({pattern_abs->output(0)});

    // Create a matcher and try to match the nodes
    TestMatcher tm;

    // Should perfectly match
    ASSERT_TRUE(tm.match(pattern_relu, model_relu));
}

TEST(pattern, wrap_type_list) {
    // Create a sample model
    PartialShape shape{2, 2};
    auto model_param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_param2 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_add = std::make_shared<ov::op::v1::Add>(model_param1->output(0), model_param2->output(0));
    auto model_param3 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_mul = std::make_shared<ov::op::v0::MatMul>(model_add->output(0), model_param3->output(0), false, false);
    auto model_abs = std::make_shared<ov::op::v0::Abs>(model_mul->output(0));
    auto model_relu = std::make_shared<ov::op::v0::Relu>(model_abs->output(0));
    auto model_result = std::make_shared<ov::op::v0::Result>(model_relu->output(0));
    auto model_sig = std::make_shared<ov::op::v0::Sigmoid>(model_abs->output(0));
    auto model_result1 = std::make_shared<ov::op::v0::Result>(model_sig->output(0));

    // Create a sample model
    auto pattern_mul = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({pattern::any_input(), pattern::any_input()});
    auto pattern_abs = ov::pass::pattern::wrap_type<ov::op::v0::Abs>({pattern_mul->output(0)});
    auto pattern_relu = ov::pass::pattern::wrap_type<ov::op::v0::Relu, ov::op::v0::Sigmoid>({pattern_abs->output(0)});

    // Create a matcher and try to match the nodes
    TestMatcher tm;

    // The same pattern perfectly matches 2 different nodes
    ASSERT_TRUE(tm.match(pattern_relu, model_relu));
    ASSERT_TRUE(tm.match(pattern_relu, model_sig));
}

TEST(pattern, pattern_or) {
    // Create a sample model
    PartialShape shape{2, 2};
    auto model_param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_param2 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_add = std::make_shared<ov::op::v1::Add>(model_param1->output(0), model_param2->output(0));
    auto model_param3 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_mul = std::make_shared<ov::op::v0::MatMul>(model_add->output(0), model_param3->output(0), false, false);
    auto model_abs = std::make_shared<ov::op::v0::Abs>(model_mul->output(0));
    auto model_relu = std::make_shared<ov::op::v0::Relu>(model_abs->output(0));
    auto model_result = std::make_shared<ov::op::v0::Result>(model_relu->output(0));

    // Create a red branch
    auto red_pattern_add =
        ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({pattern::any_input(), pattern::any_input()});
    auto red_pattern_relu = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({red_pattern_add->output(0)});
    auto red_pattern_sigmoid = ov::pass::pattern::wrap_type<ov::op::v0::Sigmoid>({red_pattern_relu->output(0)});

    // Create a blue branch
    auto blue_pattern_mul =
        ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({pattern::any_input(), pattern::any_input()});
    auto blue_pattern_abs = ov::pass::pattern::wrap_type<ov::op::v0::Abs>({blue_pattern_mul->output(0)});
    auto blue_pattern_relu = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({blue_pattern_abs->output(0)});

    // Create Or node
    auto pattern_or = std::make_shared<ov::pass::pattern::op::Or>(
        OutputVector{red_pattern_sigmoid->output(0), blue_pattern_relu->output(0)});

    // Create a matcher and try to match the nodes
    TestMatcher tm;

    // The same pattern perfectly matches 2 different nodes
    ASSERT_TRUE(tm.match(pattern_or, model_relu));
}

TEST(pattern, pattern_optional_middle) {
    // Create a sample model
    PartialShape shape{2, 2};
    auto model_param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_param2 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_add = std::make_shared<ov::op::v1::Add>(model_param1->output(0), model_param2->output(0));
    auto model_param3 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_mul = std::make_shared<ov::op::v0::MatMul>(model_add->output(0), model_param3->output(0), false, false);
    auto model_abs = std::make_shared<ov::op::v0::Abs>(model_mul->output(0));
    auto model_relu = std::make_shared<ov::op::v0::Relu>(model_abs->output(0));
    auto model_result = std::make_shared<ov::op::v0::Result>(model_relu->output(0));

    // Create a sample pattern with an Optional node in the middle
    auto pattern_mul = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({pattern::any_input(), pattern::any_input()});
    auto pattern_abs = ov::pass::pattern::wrap_type<ov::op::v0::Abs>({pattern_mul->output(0)});
    auto pattern_sig_opt = ov::pass::pattern::optional<ov::op::v0::Sigmoid>({pattern_abs->output(0)});
    auto pattern_relu = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({pattern_sig_opt->output(0)});

    // Create a matcher and try to match the nodes
    TestMatcher tm;

    // Should perfectly match
    ASSERT_TRUE(tm.match(pattern_relu, model_relu));
}

TEST(pattern, pattern_optional_top) {
    // Create a sample model
    PartialShape shape{2, 2};
    auto model_param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_param2 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_add = std::make_shared<ov::op::v1::Add>(model_param1->output(0), model_param2->output(0));
    auto model_param3 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_mul = std::make_shared<ov::op::v0::MatMul>(model_add->output(0), model_param3->output(0), false, false);
    auto model_abs = std::make_shared<ov::op::v0::Abs>(model_mul->output(0));
    auto model_relu = std::make_shared<ov::op::v0::Relu>(model_abs->output(0));
    auto model_result = std::make_shared<ov::op::v0::Result>(model_relu->output(0));

    // Create a sample pattern an optional top node
    auto pattern_sig_opt = ov::pass::pattern::optional<ov::op::v0::Sigmoid>(pattern::any_input());
    auto pattern_mul = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({pattern_sig_opt, pattern::any_input()});
    auto pattern_abs = ov::pass::pattern::wrap_type<ov::op::v0::Abs>({pattern_mul->output(0)});
    auto pattern_relu = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({pattern_abs->output(0)});

    // Create a matcher and try to match the nodes
    TestMatcher tm;

    // Should perfectly match
    ASSERT_TRUE(tm.match(pattern_relu, model_relu));
}

TEST(pattern, pattern_optional_root) {
    // Create a sample model
    PartialShape shape{2, 2};
    auto model_param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_param2 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_add = std::make_shared<ov::op::v1::Add>(model_param1->output(0), model_param2->output(0));
    auto model_param3 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_mul = std::make_shared<ov::op::v0::MatMul>(model_add->output(0), model_param3->output(0), false, false);
    auto model_abs = std::make_shared<ov::op::v0::Abs>(model_mul->output(0));
    auto model_relu = std::make_shared<ov::op::v0::Relu>(model_abs->output(0));
    auto model_result = std::make_shared<ov::op::v0::Result>(model_relu->output(0));

    // Create a sample pattern an optional top node
    auto pattern_mul = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({pattern::any_input(), pattern::any_input()});
    auto pattern_abs = ov::pass::pattern::wrap_type<ov::op::v0::Abs>({pattern_mul->output(0)});
    auto pattern_relu = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({pattern_abs->output(0)});
    auto pattern_sig_opt = ov::pass::pattern::optional<ov::op::v0::Sigmoid>(pattern_relu);

    // Create a matcher and try to match the nodes
    TestMatcher tm;

    // Should perfectly match
    ASSERT_TRUE(tm.match(pattern_relu, model_relu));
}