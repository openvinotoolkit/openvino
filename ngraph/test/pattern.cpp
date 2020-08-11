//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <list>
#include <memory>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/branch.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/or.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/pattern/op/true.hpp"
#include "util/matcher.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

static std::shared_ptr<Node> construct_constant_node(int n)
{
    return op::Constant::create(element::i32, Shape{}, {n});
}

static std::shared_ptr<pattern::op::Label> construct_variance_graph()
{
    // construct varaiance
    auto N = op::Constant::create(element::f32, Shape{3}, {2, 2, 2});
    auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{2, 3});
    auto input_sq = std::make_shared<op::Multiply>(input, input);
    auto sum_input = std::make_shared<op::Sum>(input, AxisSet{0});
    auto square_sumed_input = std::make_shared<op::Multiply>(sum_input, sum_input);
    auto sum_squared_input = std::make_shared<op::Sum>(input_sq, AxisSet{0});
    auto avg_input_sum_sq = std::make_shared<op::Divide>(square_sumed_input, N);
    auto xmu = std::make_shared<op::Subtract>(sum_squared_input, avg_input_sum_sq);
    auto variance = std::make_shared<op::Divide>(xmu, N);
    auto variance_label =
        std::make_shared<pattern::op::Label>(variance, nullptr, NodeVector{variance});

    return variance_label;
}

static std::shared_ptr<pattern::op::Label> construct_mean_graph()
{
    // construct mean;
    auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{2, 3});
    auto N = op::Constant::create(element::f32, Shape{3}, {2, 2, 2});
    auto sum_input1 = std::make_shared<op::Sum>(input, AxisSet{0});
    auto mean = std::make_shared<op::Divide>(sum_input1, N);
    auto mean_label = std::make_shared<pattern::op::Label>(mean, nullptr, NodeVector{mean});
    return mean_label;
}

class TestGraphRewrite : public ngraph::pass::GraphRewrite
{
public:
    void construct_multiply_by_one()
    {
        // pattern #1 : a * 1 = a
        auto iconst1 = construct_constant_node(1);
        auto pattern = std::make_shared<pattern::op::Label>(iconst1);

        auto callback = [pattern](pattern::Matcher& m) {
            NGRAPH_DEBUG << "In a callback for construct_multiply_by_one against "
                         << m.get_match_root()->get_name();
            NGRAPH_CHECK(m.get_match_root()->input_values().size() == 2);

            auto pattern_map = m.get_pattern_map();

            size_t const_node_index =
                m.get_match_root()->input_value(0).get_node_shared_ptr() == pattern_map[pattern];
            auto const_node = as_type_ptr<op::Constant>(
                m.get_match_root()->input_value(const_node_index).get_node_shared_ptr());
            auto second_node =
                m.get_match_root()->input_value(const_node_index).get_node_shared_ptr();
            NGRAPH_DEBUG << "second_node = " << second_node->get_name()
                         << " , pattern = " << pattern_map[pattern]->get_name();

            if (pattern_map[pattern]->get_element_type() != const_node->get_element_type() ||
                pattern_map[pattern]->get_shape() != const_node->get_shape())
            {
                NGRAPH_DEBUG << "Operands' types and/or shape don't match";
                return false;
            }

            auto const_values = const_node->get_vector<int32_t>();
            bool all_ones =
                std::all_of(begin(const_values), end(const_values), [](int e) { return e == 1; });

            if (!all_ones)
            {
                NGRAPH_DEBUG << "Constant vector's values aren't equal to 1";
                return false;
            }

            ngraph::replace_node(m.get_match_root(), pattern_map[pattern]);
            return true;
        };

        auto m = make_shared<TestMatcher>(pattern * iconst1);
        NGRAPH_SUPPRESS_DEPRECATED_START
        this->add_matcher(m, callback);
        NGRAPH_SUPPRESS_DEPRECATED_END
    }

    void construct_add_zero()
    {
        // pattern #2 : a + 0 = a
        auto iconst0 = construct_constant_node(0);
        auto pattern = std::make_shared<pattern::op::Label>(iconst0);

        auto callback = [pattern](pattern::Matcher& m) {
            NGRAPH_DEBUG << "In a callback for construct_add_zero against "
                         << m.get_match_root()->get_name();
            NGRAPH_CHECK(m.get_match_root()->input_values().size() == 2);

            auto pattern_map = m.get_pattern_map();

            size_t const_node_index =
                m.get_match_root()->input_value(0).get_node_shared_ptr() == pattern_map[pattern];
            auto const_node = as_type_ptr<op::Constant>(
                m.get_match_root()->input_value(const_node_index).get_node_shared_ptr());
            auto second_node =
                m.get_match_root()->input_value(const_node_index).get_node_shared_ptr();
            NGRAPH_DEBUG << "second_node = " << second_node->get_name()
                         << " , pattern = " << pattern_map[pattern]->get_name();

            if (pattern_map[pattern]->get_element_type() != const_node->get_element_type() ||
                pattern_map[pattern]->get_shape() != const_node->get_shape())
            {
                NGRAPH_DEBUG << "Operands' types and/or shape don't match";
                return false;
            }

            auto const_values = const_node->get_vector<int>();
            bool all_zeros =
                std::all_of(begin(const_values), end(const_values), [](int e) { return e == 0; });

            if (!all_zeros)
            {
                NGRAPH_DEBUG << "Constant vector's values aren't equal to 0";
                return false;
            }

            ngraph::replace_node(m.get_match_root(), pattern_map[pattern]);
            return true;
        };

        auto add = pattern + iconst0;
        auto m = make_shared<TestMatcher>(add);
        NGRAPH_SUPPRESS_DEPRECATED_START
        this->add_matcher(m, callback);
        NGRAPH_SUPPRESS_DEPRECATED_END
    }

    TestGraphRewrite()
        : GraphRewrite()
    {
        construct_multiply_by_one();
        construct_add_zero();
    }
};

static void run_passes(pass::Manager& pass_manager,
                       shared_ptr<Node> graph,
                       std::vector<shared_ptr<op::Parameter>> parms)
{
    auto func = make_shared<Function>(graph, ParameterVector{parms});
    pass_manager.run_passes(func);
}

TEST(pattern, graph_rewrite)
{
    Shape shape{};
    pass::Manager pass_manager;
    pass_manager.register_pass<TestGraphRewrite>();

    {
        auto a = make_shared<op::Parameter>(element::i32, shape);
        auto b = make_shared<op::Parameter>(element::i32, shape);
        auto c = make_shared<op::Parameter>(element::i32, shape);
        auto iconst0 = construct_constant_node(0);
        auto graph_a = a + iconst0;
        auto graph_b = b + iconst0;

        auto f = std::make_shared<Function>(ngraph::NodeVector{a, b, graph_a, c, graph_b},
                                            ParameterVector{a, b, c});
        pass_manager.run_passes(f);

        ASSERT_TRUE(graph_a->get_output_target_inputs(0).empty());
        ASSERT_TRUE(graph_b->get_output_target_inputs(0).empty());

        auto expected = ngraph::NodeVector{a, b, a, c, b};
        ASSERT_TRUE(count_ops_of_type<op::Add>(f) == 0);
    }

    {
        auto a = make_shared<op::Parameter>(element::i32, shape);
        auto b = make_shared<op::Parameter>(element::i32, shape);
        auto iconst0 = construct_constant_node(0);
        auto sum = (a + iconst0);
        auto graph = b + sum;
        run_passes(pass_manager, graph, {a, b});
        ASSERT_EQ(graph->input_value(1).get_node_shared_ptr(), a);
        ASSERT_EQ(graph->input_value(1), a->output(0)); // graph's input points to a's output
        ASSERT_TRUE(sum->output(0)
                        .get_target_inputs()
                        .empty()); // graph's input is removed from sum's target inptus
        ASSERT_TRUE(a->get_output_target_inputs(0).count(
            graph->input(1))); // a's output feeds into graph's input
    }

    {
        auto a = make_shared<op::Parameter>(element::i32, shape);
        auto b = make_shared<op::Parameter>(element::i32, shape);
        auto iconst1 = construct_constant_node(1);
        auto mul = (a * iconst1);
        auto graph = b + mul;
        run_passes(pass_manager, graph, {a, b});
        ASSERT_EQ(graph->input_value(1).get_node_shared_ptr(), a);
        ASSERT_EQ(graph->input_value(1), a->output(0)); // graph's input points to a's output
        ASSERT_TRUE(mul->output(0)
                        .get_target_inputs()
                        .empty()); // graph's input is removed from sum's target inputs
        ASSERT_TRUE(a->get_output_target_inputs(0).count(
            graph->input(1))); // a's output feeds into graph's input
    }

    {
        auto a = make_shared<op::Parameter>(element::i32, shape);
        auto b = make_shared<op::Parameter>(element::i32, shape);
        auto iconst1 = construct_constant_node(1);
        auto graph = ((((a * iconst1) * iconst1) * iconst1) * iconst1) + b;
        run_passes(pass_manager, graph, {a, b});
        ASSERT_EQ(graph->input_value(0).get_node_shared_ptr(), a);
        ASSERT_EQ(graph->input_value(0), a->output(0)); // graph's input points to a's output
        ASSERT_TRUE(a->get_output_target_inputs(0).count(
            graph->input(0))); // a's output feeds into graph's input
    }

    {
        auto a = make_shared<op::Parameter>(element::i32, shape);
        auto b = make_shared<op::Parameter>(element::i32, shape);
        auto iconst0 = construct_constant_node(0);
        auto iconst1 = construct_constant_node(1);
        auto graph = b + (iconst0 + ((a + iconst0) * iconst1));
        run_passes(pass_manager, graph, {a, b});
        ASSERT_EQ(graph->input_value(1).get_node_shared_ptr(), a);
        ASSERT_EQ(graph->input_value(1), a->output(0)); // graph's input points to a's output
        ASSERT_TRUE(a->get_output_target_inputs(0).count(
            graph->input(1))); // a's output feeds into graph's input
    }

    {
        auto a = make_shared<op::Parameter>(element::i32, shape);
        auto b = make_shared<op::Parameter>(element::i32, shape);
        auto iconst1 = construct_constant_node(1);
        auto graph = b + (iconst1 * (iconst1 * (iconst1 * (iconst1 * a))));
        run_passes(pass_manager, graph, {a, b});
        ASSERT_EQ(graph->input_value(1).get_node_shared_ptr(), a);
        ASSERT_EQ(graph->input_value(1), a->output(0)); // graph's input points to a's output
        ASSERT_TRUE(a->get_output_target_inputs(0).count(
            graph->input(1))); // a's output feeds into graph's input
    }
}

TEST(pattern, matcher)
{
    Shape shape{};
    auto a = make_shared<op::Parameter>(element::i32, shape);
    TestMatcher n;
    ASSERT_TRUE(n.match(a, a));
    ASSERT_EQ(n.get_matched_nodes(), (NodeVector{a}));

    auto abs = make_shared<op::Abs>(a);
    auto any = std::make_shared<pattern::op::Skip>(a);
    ASSERT_TRUE(n.match(any, abs));
    ASSERT_EQ(n.get_matched_nodes(), (NodeVector{abs, a}));

    auto false_pred = [](std::shared_ptr<Node> /* no */) { return false; };
    auto any_false = std::make_shared<pattern::op::Skip>(a, false_pred);
    ASSERT_TRUE(n.match(any_false, a));
    ASSERT_EQ(n.get_matched_nodes(), (NodeVector{a, a}));

    auto pattern = std::make_shared<pattern::op::Label>(a);
    ASSERT_TRUE(n.match(pattern, a));
    ASSERT_EQ(n.get_pattern_map()[pattern], a);
    ASSERT_EQ(n.get_matched_nodes(), (NodeVector{a}));

    auto pattern_false = std::make_shared<pattern::op::Label>(a, false_pred);
    ASSERT_FALSE(n.match(pattern_false, a));
    ASSERT_EQ(n.get_matched_nodes(), (NodeVector{}));

    auto b = make_shared<op::Parameter>(element::i32, shape);

    auto is_bea = [](std::shared_ptr<Node> node) -> bool {
        return op::is_binary_elementwise_arithmetic(node);
    };
    auto bea = std::make_shared<pattern::op::Any>(a, is_bea, NodeVector{a, b});
    auto add_ab = a + b;
    ASSERT_TRUE(n.match(bea, add_ab));
    ASSERT_EQ(n.get_matched_nodes(), (NodeVector{add_ab, a, b}));
    ASSERT_TRUE(n.match(bea, b + a));

    auto bea_false = std::make_shared<pattern::op::Any>(a, false_pred, NodeVector{a, b});
    ASSERT_FALSE(n.match(bea_false, a + b));

    auto add_abs_b = abs + b;
    auto bea_any_of = std::make_shared<pattern::op::AnyOf>(a, is_bea, NodeVector{abs});
    ASSERT_TRUE(n.match(bea_any_of, add_abs_b));

    auto add_b_abs = b + abs;
    ASSERT_TRUE(n.match(bea_any_of, add_b_abs));

    auto bea_any_of_label =
        std::make_shared<pattern::op::Label>(a, nullptr, NodeVector{bea_any_of});
    ASSERT_TRUE(n.match(bea_any_of_label, add_b_abs));
    ASSERT_EQ(n.get_pattern_map()[bea_any_of_label], add_b_abs);

    auto abs_label = std::make_shared<pattern::op::Label>(a, nullptr, NodeVector{abs});
    auto bea_label_any_of = std::make_shared<pattern::op::AnyOf>(a, is_bea, NodeVector{abs_label});
    ASSERT_TRUE(n.match(bea_label_any_of, add_b_abs));
    ASSERT_EQ(n.get_pattern_map()[abs_label], abs);

    auto bea_label = std::make_shared<pattern::op::Label>(a, nullptr, NodeVector{bea});
    auto ab = a + b;
    ASSERT_TRUE(n.match(bea_label, ab));
    ASSERT_EQ(n.get_pattern_map()[bea_label], ab);

    auto d = make_shared<op::Parameter>(element::i32, shape);
    ASSERT_FALSE(n.match(d, b));

    ASSERT_FALSE(n.match(abs + b, b + b));
    ASSERT_EQ(n.get_matched_nodes(), (NodeVector{}));

    auto add_absb = abs + b;
    ASSERT_TRUE(n.match(any + b, add_absb));
    ASSERT_EQ(n.get_matched_nodes(), (NodeVector{add_absb, abs, a, b}));

    ASSERT_TRUE(n.match(pattern + b, add_absb));
    ASSERT_EQ(n.get_pattern_map()[pattern], abs);
    ASSERT_EQ(n.get_matched_nodes(), (NodeVector{add_absb, abs, b}));

    ASSERT_TRUE(n.match(b + pattern, add_absb));
    ASSERT_EQ(n.get_pattern_map()[pattern], abs);
    ASSERT_EQ(n.get_matched_nodes(), (NodeVector{add_absb, abs, b}));

    auto c = make_shared<op::Parameter>(element::i32, shape);
    auto mul_add_absb = c * (add_absb);
    ASSERT_TRUE(n.match(c * (b + pattern), mul_add_absb));
    ASSERT_EQ(n.get_pattern_map()[pattern], abs);
    ASSERT_EQ(n.get_matched_nodes(), (NodeVector{mul_add_absb, c, add_absb, abs, b}));

    ASSERT_TRUE(n.match(c * (any + b), mul_add_absb)); // nested any
    ASSERT_EQ(n.get_matched_nodes(), (NodeVector{mul_add_absb, c, add_absb, abs, a, b}));
    ASSERT_TRUE(n.match(c * (any + b), (b + abs) * c)); // permutations w/ any
    auto mul_c_add_ab = c * add_ab;
    ASSERT_TRUE(n.match(c * (any_false + b), c * (a + b)));  // nested any
    ASSERT_TRUE(n.match(c * (any_false + b), mul_c_add_ab)); // permutations w/ any_false
    ASSERT_EQ(n.get_matched_nodes(), (NodeVector{mul_c_add_ab, c, add_ab, a, a, b}));

    auto iconst1_0 = construct_constant_node(1);
    auto iconst1_1 = construct_constant_node(1);
    ASSERT_TRUE(n.match(pattern * iconst1_0, a * iconst1_1)); // different iconst
    ASSERT_EQ(n.get_pattern_map()[pattern], a);
    auto fconst1_0 = op::Constant::create(element::f32, shape, {1});
    auto patternf = std::make_shared<pattern::op::Label>(fconst1_0);
    ASSERT_TRUE(n.match(patternf * fconst1_0, a * iconst1_1)); // different iconst

    // Subgraph labels
    auto add = a + b;
    auto label = std::make_shared<pattern::op::Label>(add, nullptr, NodeVector{add});
    ASSERT_TRUE(n.match(label, add));
    ASSERT_EQ(n.get_pattern_map()[label], add);
    ASSERT_EQ(n.get_matched_nodes(), (NodeVector{add, add, a, b}));

    ASSERT_FALSE(n.match(label, a - b));

    ASSERT_TRUE(n.match(make_shared<op::Abs>(label), make_shared<op::Abs>(add)));
    ASSERT_EQ(n.get_pattern_map()[label], add);

    // Correct argument order
    ASSERT_FALSE(n.match(b - a, a - b));
    auto aab = a * (a - b);
    auto paab = pattern * (pattern - b);
    ASSERT_TRUE(n.match(paab, aab));
    auto aba = a * (b - a);
    ASSERT_FALSE(n.match(paab, aba));
    auto paba = pattern * (b - pattern);
    ASSERT_FALSE(n.match(paba, aab));

    // Correlations
    auto label1 = std::make_shared<pattern::op::Label>(a);
    auto tmp = label1 + b;
    auto label2 = std::make_shared<pattern::op::Label>(tmp, nullptr, NodeVector{tmp});
    auto sub_label1 = label1 - label2;
    auto sub_add = a - add;
    ASSERT_TRUE(n.match(sub_label1, sub_add));
    ASSERT_EQ(n.get_pattern_map()[label1], a);
    ASSERT_EQ(n.get_pattern_map()[label2], add);
    ASSERT_EQ(n.get_matched_nodes(), (NodeVector{sub_add, a, add, add, a, b}));

    ASSERT_FALSE(n.match(sub_label1, add - a));

    auto add_label1 = label1 + label2;
    ASSERT_TRUE(n.match(add_label1, add + a));
    ASSERT_EQ(n.get_pattern_map()[label1], a);
    ASSERT_EQ(n.get_pattern_map()[label2], add);

    // Or
    ASSERT_TRUE(n.match(std::make_shared<pattern::op::Or>(OutputVector{a + b, a - b}), a + b));
    ASSERT_TRUE(n.match(std::make_shared<pattern::op::Or>(OutputVector{a + b, a - b}), a - b));

    // Branch
    {
        auto branch = std::make_shared<pattern::op::Branch>();
        auto star = std::make_shared<pattern::op::Or>(
            OutputVector{branch, std::make_shared<pattern::op::True>()});
        auto pattern = star + star;
        branch->set_destination(pattern);
        ASSERT_TRUE(n.match(pattern, ((a + b) + (b + a) + a)));
        ASSERT_EQ(n.get_matched_nodes().size(), 4);
    }

    // strict mode
    {
        TestMatcher sm(Output<Node>{}, "TestMatcher", true);
        // exact shape and type
        auto scalar_param = make_shared<op::Parameter>(element::i32, Shape{});
        auto label_dynamic_shape =
            make_shared<pattern::op::Label>(element::i32, PartialShape::dynamic());
        auto param = make_shared<op::Parameter>(element::f32, Shape{});
        ASSERT_TRUE(sm.match(label_dynamic_shape, scalar_param));
        // wrong type
        auto scalar_param_wrong_type = make_shared<op::Parameter>(element::f32, Shape{});
        ASSERT_FALSE(sm.match(label, scalar_param_wrong_type));
        // dynamic dimension
        auto label_dynamic_dimension =
            make_shared<pattern::op::Label>(element::i32, PartialShape{Dimension::dynamic()});
        auto vector_param = make_shared<op::Parameter>(element::i32, Shape{10});
        ASSERT_TRUE(sm.match(label_dynamic_dimension, vector_param));
        // dynamic type
        auto label_dynamic_type =
            make_shared<pattern::op::Label>(element::dynamic, PartialShape{Dimension::dynamic()});
        ASSERT_TRUE(sm.match(label_dynamic_type, vector_param));
    }
}

TEST(pattern, mean)
{
    // construct mean
    TestMatcher n;

    auto input = std::make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto N = op::Constant::create(element::f32, Shape{3}, {2, 2, 2});
    auto sum_input1 = std::make_shared<op::Sum>(input, AxisSet{0});
    auto mean = std::make_shared<op::Divide>(sum_input1, N);

    auto mean_graph = construct_mean_graph();
    ASSERT_TRUE(n.match(mean_graph, mean));
    ASSERT_EQ(n.get_pattern_map()[mean_graph], mean);
}

TEST(pattern, variance)
{
    // construct variance
    TestMatcher n;
    auto N = op::Constant::create(element::f32, Shape{3}, {2, 2, 2});
    auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{2, 3});
    auto input_sq = std::make_shared<op::Multiply>(input, input);
    auto sum_input = std::make_shared<op::Sum>(input, AxisSet{0});
    auto square_sumed_input = std::make_shared<op::Multiply>(sum_input, sum_input);
    auto sum_squared_input = std::make_shared<op::Sum>(input_sq, AxisSet{0});
    auto avg_input_sum_sq = std::make_shared<op::Divide>(square_sumed_input, N);
    auto xmu = std::make_shared<op::Subtract>(sum_squared_input, avg_input_sum_sq);
    auto variance = std::make_shared<op::Divide>(xmu, N);

    auto var_graph = construct_variance_graph();
    ASSERT_TRUE(n.match(var_graph, variance));
    ASSERT_EQ(n.get_pattern_map()[var_graph], variance);
}

TEST(pattern, previous_matches)
{
    using ngraph::pattern::Matcher;
    Shape shape{};
    Matcher::PatternMap previous_matches;
    auto a = make_shared<op::Parameter>(element::i32, shape);
    auto b = make_shared<op::Parameter>(element::i32, shape);
    auto pattern = std::make_shared<pattern::op::Label>(b);
    auto abs = make_shared<op::Abs>(a);
    auto add = abs + b;
    {
        Matcher n(pattern + b);
        ASSERT_TRUE(n.match(add, previous_matches));
        ASSERT_EQ(n.get_pattern_map()[pattern], abs);
    }

    {
        Matcher n(pattern + b);
        previous_matches.insert(std::make_pair(pattern, a));
        ASSERT_FALSE(n.match(add, previous_matches));
    }
}

TEST(pattern, test_sort)
{
    using ngraph::pattern::Matcher;
    Shape shape{};

    auto a = make_shared<op::Parameter>(element::i32, shape);
    auto b = make_shared<op::Parameter>(element::i32, shape);
    auto abs1 = make_shared<op::Abs>(a);
    auto abs2 = make_shared<op::Abs>(b);
    auto add = abs1 + abs2;

    auto pa = make_shared<op::Parameter>(element::i32, shape);
    auto pb = make_shared<op::Parameter>(element::i32, shape);
    auto pabs1 = make_shared<op::Abs>(pa);
    auto pabs1_label = std::make_shared<pattern::op::Label>(pabs1);
    auto pabs2 = make_shared<op::Abs>(b);
    auto padd = pabs1_label + pabs2;

    {
        Matcher n1(padd);
        ASSERT_TRUE(n1.match(add));
        auto r1 = n1.get_pattern_map()[pabs1_label];
        ASSERT_TRUE(n1.match(add));
        ASSERT_EQ(r1, n1.get_pattern_map()[pabs1_label]);
    }
}

TEST(pattern, recurrent_pattern)
{
    using ngraph::pattern::RecurrentMatcher;
    Shape shape{};
    ngraph::pattern::Matcher::PatternMap previous_matches;
    auto a = make_shared<op::Parameter>(element::i32, shape);
    auto b = make_shared<op::Parameter>(element::i32, shape);
    auto rpattern = std::make_shared<pattern::op::Label>(b);
    auto iconst0 = construct_constant_node(0);
    auto abs = make_shared<op::Abs>(a);
    auto add1 = iconst0 + b;
    auto add2 = iconst0 + add1;
    auto add3 = iconst0 + add2;
    auto padd = iconst0 + rpattern;
    std::set<std::shared_ptr<pattern::op::Label>> empty_correlated_matches;
    RecurrentMatcher rm(padd, rpattern, empty_correlated_matches);
    ASSERT_TRUE(rm.match(add3));
    ASSERT_EQ(rm.get_number_of_bound_labels(), 3);
    auto recurrent_matches = rm.get_bound_nodes_for_pattern(rpattern);
    ASSERT_EQ(recurrent_matches.at(0), add2);
    ASSERT_EQ(recurrent_matches.at(1), add1);
    ASSERT_EQ(recurrent_matches.at(2), b);

    // Multiple labels in a reccuring pattern
    auto iconst1 = construct_constant_node(1);
    auto iconst_label = std::make_shared<pattern::op::Label>(iconst1, nullptr, NodeVector{iconst1});
    auto add2_2 = iconst1 + add1;
    auto add3_2 = iconst0 + add2_2;
    auto padd2 = iconst_label + rpattern;
    RecurrentMatcher rm2(padd2, rpattern, empty_correlated_matches);
    ASSERT_TRUE(rm2.match(add3_2));
    ASSERT_EQ(rm2.get_number_of_bound_labels(), 4);
    recurrent_matches = rm2.get_bound_nodes_for_pattern(rpattern);
    ASSERT_EQ(recurrent_matches.at(0), add2_2);
    ASSERT_EQ(recurrent_matches.at(1), add1);
    ASSERT_EQ(recurrent_matches.at(2), b);
    auto iconst_matches = rm2.get_bound_nodes_for_pattern(iconst_label);
    ASSERT_EQ(iconst_matches.at(0), iconst0);
    ASSERT_EQ(iconst_matches.at(1), iconst1);
    ASSERT_EQ(iconst_matches.at(2), iconst0);

    // Non-matching correlated labels
    std::set<std::shared_ptr<pattern::op::Label>> correlated_matches;
    correlated_matches.insert(iconst_label);
    RecurrentMatcher rm3(padd2, rpattern, correlated_matches);
    ASSERT_TRUE(rm3.match(add3_2));
    ASSERT_EQ(rm3.get_number_of_bound_labels(), 4);
    iconst_matches = rm3.get_bound_nodes_for_pattern(iconst_label);
    ASSERT_EQ(iconst_matches.size(), 1);
    ASSERT_EQ(iconst_matches.at(0), iconst0);

    // Matching correlated labels and
    // testing if RecurrentMatcher can be reused for different nodes
    ASSERT_TRUE(rm3.match(add3));
    ASSERT_EQ(rm3.get_number_of_bound_labels(), 4);
    recurrent_matches = rm3.get_bound_nodes_for_pattern(rpattern);
    ASSERT_EQ(recurrent_matches.at(0), add2);
    ASSERT_EQ(recurrent_matches.at(1), add1);
    ASSERT_EQ(recurrent_matches.at(2), b);
    iconst_matches = rm3.get_bound_nodes_for_pattern(iconst_label);
    ASSERT_EQ(iconst_matches.at(0), iconst0);
    ASSERT_EQ(iconst_matches.at(1), iconst0);
    ASSERT_EQ(iconst_matches.at(2), iconst0);
}

class TestRecurrentGraphRewrite : public ngraph::pass::RecurrentGraphRewrite
{
public:
    void construct_recurrent_add()
    {
        Shape shape{};
        auto iconst0 = construct_constant_node(0);
        auto iconst_label =
            std::make_shared<pattern::op::Label>(iconst0, nullptr, NodeVector{iconst0});
        auto rpattern = std::make_shared<pattern::op::Label>(element::i32, shape);
        auto padd = iconst_label + rpattern;

        auto callback = [iconst_label, rpattern](pattern::RecurrentMatcher& rm) {
            NGRAPH_DEBUG << "In a callback for construct_recurrent_add against "
                         << rm.get_match_root()->get_name();

            auto iconst_matches = rm.get_bound_nodes_for_pattern(iconst_label);

            auto is_iconst_zero = [](std::shared_ptr<Node> n) {
                bool result = ngraph::is_zero(n);
                NGRAPH_DEBUG << n->get_name() << " is " << (result ? " a zero " : " not a zero");
                return ngraph::is_zero(n);
            };

            bool are_all_iconst_zeros =
                std::all_of(iconst_matches.begin(), iconst_matches.end(), is_iconst_zero);

            if (!are_all_iconst_zeros)
            {
                return false;
            }

            auto number_of_adds = rm.get_number_of_recurrent_matches();
            // replace the topmost add with the seed (i.e. the first parameter to add)
            // matches are added in reverse order (i.e. the first match is the topmost node)
            auto arg = rm.get_bound_nodes_for_pattern(rpattern).at(number_of_adds - 1);
            NGRAPH_DEBUG << "Replacing " << rm.get_match_root()->get_name() << " with "
                         << arg->get_name();
            ngraph::replace_node(rm.get_match_root(), arg);
            return true;
        };

        std::set<std::shared_ptr<pattern::op::Label>> empty_correlated_matches;
        auto rm = make_shared<pattern::RecurrentMatcher>(padd, rpattern, empty_correlated_matches);
        NGRAPH_SUPPRESS_DEPRECATED_START
        this->add_matcher(rm, callback);
        NGRAPH_SUPPRESS_DEPRECATED_END
    }

    TestRecurrentGraphRewrite()
        : RecurrentGraphRewrite()
    {
        construct_recurrent_add();
    }
};

TEST(pattern, recurrent_graph_rewrite)
{
    Shape shape{};
    pass::Manager pass_manager;
    pass_manager.register_pass<TestRecurrentGraphRewrite>();

    {
        auto a = make_shared<op::Parameter>(element::i32, shape);
        auto iconst0 = construct_constant_node(0);
        auto add_a1 = a + iconst0;
        auto add_a2 = add_a1 + iconst0;
        auto add_a3 = add_a2 + iconst0;
        auto abs_add_a3 = std::make_shared<op::Abs>(add_a3);

        auto b = make_shared<op::Parameter>(element::i32, shape);
        auto add_b1 = b + iconst0;
        auto add_b2 = add_b1 + iconst0;
        auto abs_add_b2 = std::make_shared<op::Abs>(add_b2);

        auto graph = abs_add_a3 * abs_add_b2;

        auto f = std::make_shared<Function>(ngraph::NodeVector{graph}, ParameterVector{a, b});
        pass_manager.run_passes(f);

        auto left_abs = graph->input_value(0).get_node_shared_ptr();
        auto add_a = left_abs->input_value(0).get_node_shared_ptr();
        ASSERT_EQ(add_a, a);

        auto right_abs = graph->input_value(1).get_node_shared_ptr();
        auto add_b = right_abs->input_value(0).get_node_shared_ptr();
        ASSERT_EQ(add_b, b);
    }
}

TEST(pattern, label_on_skip)
{
    Shape shape{2, 2};
    auto a = make_shared<op::Parameter>(element::i32, shape);
    auto b = make_shared<op::Parameter>(element::i32, Shape{});
    auto iconst = ngraph::make_zero(element::i32, Shape{});
    auto label = std::make_shared<pattern::op::Label>(iconst);
    auto const_label =
        std::make_shared<pattern::op::Label>(iconst, ngraph::is_zero, NodeVector{iconst});

    auto bcst_pred = [](std::shared_ptr<Node> n) {
        return as_type_ptr<op::Broadcast>(n) != nullptr;
    };

    auto bcst = std::make_shared<pattern::op::Skip>(const_label, bcst_pred);
    auto bcst_label = std::make_shared<pattern::op::Label>(bcst, nullptr, NodeVector{bcst});
    auto matcher = std::make_shared<pattern::Matcher>(
        std::make_shared<op::Multiply>(label, bcst_label), "label_on_skip");

    auto const_broadcast = make_shared<op::Broadcast>(iconst, shape, AxisSet{0, 1});
    auto mul = a * const_broadcast;
    auto mul_scalar = b * iconst;
    ASSERT_TRUE(matcher->match(mul));
    ASSERT_EQ(matcher->get_pattern_map()[bcst_label], const_broadcast);
    ASSERT_EQ(matcher->get_pattern_map()[const_label], iconst);
    ASSERT_EQ(matcher->get_pattern_map()[label], a);
    ASSERT_TRUE(matcher->match(mul_scalar));
    ASSERT_EQ(matcher->get_pattern_map()[bcst_label], iconst);
    ASSERT_EQ(matcher->get_pattern_map()[const_label], iconst);
    ASSERT_EQ(matcher->get_pattern_map()[label], b);
}

TEST(pattern, is_contained_match)
{
    Shape shape{};
    auto a = make_shared<op::Parameter>(element::i32, shape);
    auto absn = make_shared<op::Abs>(a);
    TestMatcher n;

    auto label_a = std::make_shared<pattern::op::Label>(a);
    auto label_abs = make_shared<op::Abs>(a);
    ASSERT_TRUE(n.match(label_abs, absn));
    auto result_absn = make_shared<op::Result>(absn);
    ASSERT_TRUE(n.is_contained_match());

    auto absn2 = make_shared<op::Abs>(absn);
    auto result_absn2 = make_shared<op::Result>(absn2);
    auto label_abs2 = make_shared<op::Abs>(label_abs);
    ASSERT_TRUE(n.match(label_abs2, absn2));
    ASSERT_FALSE(n.is_contained_match());
}

TEST(pattern, wrap_type)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{1, 3, 64, 64});
    auto b = make_shared<op::Abs>(a);
    auto c = make_shared<op::Relu>(a);
    auto mul1 = make_shared<op::v1::Multiply>(a, op::Constant::create(element::f32, Shape{}, {1}));
    auto mul2 = make_shared<op::v1::Multiply>(op::Constant::create(element::f32, Shape{}, {1}), a);

    {
        auto m = pattern::wrap_type<op::Abs>();
        auto matcher = std::make_shared<pattern::Matcher>(m, "AbsMatcher");
        ASSERT_TRUE(matcher->match(static_pointer_cast<Node>(b)));
        ASSERT_EQ(matcher->get_matched_nodes().size(), 1);
        ASSERT_EQ(matcher->get_matched_nodes()[0], b);
        ASSERT_EQ(matcher->get_pattern_map().count(m), 1);
        ASSERT_FALSE(matcher->match(static_pointer_cast<Node>(c)));
    }
    {
        auto m1 = pattern::wrap_type<op::Parameter>();
        auto m2 = pattern::wrap_type<op::Abs>({m1});
        auto matcher = std::make_shared<pattern::Matcher>(m2, "ParamAbsMatcher");
        ASSERT_TRUE(matcher->match(static_pointer_cast<Node>(b)));
        ASSERT_EQ(matcher->get_matched_nodes().size(), 2);
        ASSERT_EQ(matcher->get_pattern_map().count(m1), 1);
        ASSERT_EQ(matcher->get_pattern_map().count(m2), 1);
        ASSERT_FALSE(matcher->match(static_pointer_cast<Node>(c)));
    }
    {
        auto m1 = pattern::wrap_type<op::v1::Multiply>(
            {pattern::any_input(), pattern::wrap_type<op::Constant>()});
        auto matcher = std::make_shared<pattern::Matcher>(m1, "MultiplyMatcher");
        ASSERT_TRUE(matcher->match(static_pointer_cast<Node>(mul1)));
        ASSERT_TRUE(matcher->match(static_pointer_cast<Node>(mul2)));
    }
    {
        auto m1 = pattern::wrap_type<op::v1::Multiply>(
            {pattern::wrap_type<op::Constant>(), pattern::any_input()});
        auto matcher = std::make_shared<pattern::Matcher>(m1, "MultiplyMatcher");
        ASSERT_TRUE(matcher->match(static_pointer_cast<Node>(mul1)));
        ASSERT_TRUE(matcher->match(static_pointer_cast<Node>(mul2)));
    }
}
