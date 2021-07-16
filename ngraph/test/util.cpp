// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/file_util.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/op_annotations.hpp"
#include "ngraph/opsets/opset6.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"

using namespace std;
using namespace ngraph;

TEST(util, split)
{
    {
        string s1 = "this,is,a,test";
        auto r1 = split(s1, ',');
        ASSERT_EQ(4, r1.size());
        EXPECT_STRCASEEQ("this", r1[0].c_str());
        EXPECT_STRCASEEQ("is", r1[1].c_str());
        EXPECT_STRCASEEQ("a", r1[2].c_str());
        EXPECT_STRCASEEQ("test", r1[3].c_str());
    }

    {
        string s1 = "this,is,a,test,";
        auto r1 = split(s1, ',');
        ASSERT_EQ(5, r1.size());
        EXPECT_STRCASEEQ("this", r1[0].c_str());
        EXPECT_STRCASEEQ("is", r1[1].c_str());
        EXPECT_STRCASEEQ("a", r1[2].c_str());
        EXPECT_STRCASEEQ("test", r1[3].c_str());
        EXPECT_STRCASEEQ("", r1[4].c_str());
    }

    {
        string s1 = ",this,is,a,test";
        auto r1 = split(s1, ',');
        ASSERT_EQ(5, r1.size());
        EXPECT_STRCASEEQ("", r1[0].c_str());
        EXPECT_STRCASEEQ("this", r1[1].c_str());
        EXPECT_STRCASEEQ("is", r1[2].c_str());
        EXPECT_STRCASEEQ("a", r1[3].c_str());
        EXPECT_STRCASEEQ("test", r1[4].c_str());
    }

    {
        string s1 = "this,,is,a,test";
        auto r1 = split(s1, ',');
        ASSERT_EQ(5, r1.size());
        EXPECT_STRCASEEQ("this", r1[0].c_str());
        EXPECT_STRCASEEQ("", r1[1].c_str());
        EXPECT_STRCASEEQ("is", r1[2].c_str());
        EXPECT_STRCASEEQ("a", r1[3].c_str());
        EXPECT_STRCASEEQ("test", r1[4].c_str());
    }

    {
        string s1 = "this";
        auto r1 = split(s1, ',');
        ASSERT_EQ(1, r1.size());
        EXPECT_STRCASEEQ("this", r1[0].c_str());
    }

    {
        string s1 = "";
        auto r1 = split(s1, ',');
        ASSERT_EQ(1, r1.size());
        EXPECT_STRCASEEQ("", r1[0].c_str());
    }
}

TEST(DISABLED_util, dump)
{
    string text = "this is a text string used to test the dump function.";

    dump(cout, text.data(), text.size());
}

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "windows.h"
#define usleep(a) Sleep(a / 1000)
#endif
TEST(util, stopwatch)
{
    stopwatch t1;

    t1.start();
    usleep(1000);
    t1.stop();

    t1.start();
    usleep(1000);
    t1.stop();

    t1.start();
    usleep(1000);
    t1.stop();

    EXPECT_EQ(3, t1.get_call_count());

    EXPECT_GT(t1.get_total_microseconds(), t1.get_microseconds());
}

TEST(util, trim)
{
    EXPECT_STREQ("test", trim("test").c_str());
    EXPECT_STREQ("test", trim(" test").c_str());
    EXPECT_STREQ("test", trim("test ").c_str());
    EXPECT_STREQ("test", trim(" test ").c_str());
    EXPECT_STREQ("test", trim("           test            ").c_str());
    EXPECT_STREQ("test", trim("\ttest").c_str());
    EXPECT_STREQ("test", trim("test\t").c_str());
    EXPECT_STREQ("test", trim("\ttest\t").c_str());
    EXPECT_STREQ("test", trim(" \t test \t ").c_str());
}

TEST(util, all_close)
{
    auto backend = runtime::Backend::create("INTERPRETER");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, Shape{2, 3});
    auto b = backend->create_tensor(element::f32, Shape{2, 3});

    copy_data(a, test::NDArray<float, 2>({{1, 2, 3}, {3, 4, 5}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{1, 2, 3}, {3, 4, 5}}).get_vector());

    EXPECT_TRUE(ngraph::test::all_close<float>(a, b));

    auto c = backend->create_tensor(element::f32, Shape{2, 3});
    copy_data(c, test::NDArray<float, 2>({{1.1f, 2, 3}, {3, 4, 5}}).get_vector());

    EXPECT_FALSE(ngraph::test::all_close<float>(c, a, 0, .05f));
    EXPECT_TRUE(ngraph::test::all_close<float>(c, a, 0, .11f));

    EXPECT_FALSE(ngraph::test::all_close<float>(c, a, .05f, 0));
    EXPECT_TRUE(ngraph::test::all_close<float>(c, a, .11f, 0));
}

class CloneTest : public ::testing::Test
{
public:
    // (A + B) * C
    Shape shape = Shape{2, 2};
    std::shared_ptr<op::Parameter> A = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<op::Parameter> B = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<op::Parameter> C = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<Node> AplusB = make_shared<op::v1::Add>(A, B);
    std::shared_ptr<Node> AplusBtimesC = make_shared<op::v1::Multiply>(AplusB, C);

    NodeMap node_map;
    std::vector<std::shared_ptr<ngraph::Node>> nodes;
    std::shared_ptr<Function> func =
        make_shared<Function>(AplusBtimesC, ParameterVector{A, B, C}, "f");

    void SetUp() override
    {
        nodes.push_back(AplusBtimesC);
        nodes.push_back(AplusB);
        nodes.push_back(A);
        nodes.push_back(B);
        nodes.push_back(C);
    }

    bool CompareNodeVector(const std::vector<std::shared_ptr<ngraph::Node>>& orig,
                           const std::vector<std::shared_ptr<ngraph::Node>>& clone,
                           const NodeMap& nm)
    {
        if (orig.size() != clone.size())
        {
            return false;
        }
        auto origit = orig.begin();
        auto cloneit = clone.begin();
        while (origit != orig.end() && cloneit != clone.end())
        {
            if (*cloneit != nm.at((*origit).get()))
            {
                return false;
            }
            ++origit;
            ++cloneit;
        }
        return true;
    }
};

TEST_F(CloneTest, clone_nodes_full)
{
    auto cloned_nodes = clone_nodes(nodes, node_map);
    ASSERT_TRUE(CompareNodeVector(nodes, cloned_nodes, node_map));

    ASSERT_NE(nullptr, as_type_ptr<op::Parameter>(node_map.at(A.get())));
    ASSERT_NE(nullptr, as_type_ptr<op::Parameter>(node_map.at(B.get())));
    ASSERT_NE(nullptr, as_type_ptr<op::Parameter>(node_map.at(C.get())));
    ASSERT_NE(nullptr, as_type_ptr<op::v1::Add>(node_map.at(AplusB.get())));
    ASSERT_NE(nullptr, as_type_ptr<op::v1::Multiply>(node_map.at(AplusBtimesC.get())));

    auto sorted_nodes = topological_sort(nodes);
    auto sorted_cloned_nodes = topological_sort(cloned_nodes);
    ASSERT_TRUE(CompareNodeVector(sorted_nodes, sorted_cloned_nodes, node_map));
}

TEST_F(CloneTest, clone_nodes_partial)
{
    // map A -> A' prior to clone
    auto Aprime = make_shared<op::Parameter>(element::f32, shape);
    node_map[A.get()] = Aprime;

    auto cloned_nodes = clone_nodes(nodes, node_map);
    ASSERT_TRUE(CompareNodeVector(nodes, cloned_nodes, node_map));

    // ensure A -> A' after clone
    ASSERT_EQ(Aprime, node_map.at(A.get()));
}

TEST_F(CloneTest, clone_function_full)
{
    auto cloned_func = clone_function(*func, node_map);
    ASSERT_TRUE(CompareNodeVector(func->get_ops(), cloned_func->get_ops(), node_map));
}

TEST(graph_util, clone_multiple_results)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto A_add_B = make_shared<op::v1::Add>(A, B);
    auto A_add_B_mul_C = make_shared<op::v1::Multiply>(A_add_B, C);

    auto f = make_shared<Function>(NodeVector{A_add_B, A_add_B_mul_C}, ParameterVector{A, B, C});

    auto copy = clone_function(*f);
}

TEST(graph_util, clone_rt_info)
{
    const std::string testAffinity = "CPU";
    std::shared_ptr<ngraph::Function> original_f;
    {
        ngraph::PartialShape shape({1, 84});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::opset6::Parameter>(type, shape);
        auto matMulWeights =
            ngraph::opset6::Constant::create(ngraph::element::Type_t::f32, {10, 84}, {1});
        auto shapeOf = std::make_shared<ngraph::opset6::ShapeOf>(matMulWeights);
        auto gConst1 = ngraph::opset6::Constant::create(ngraph::element::Type_t::i32, {1}, {1});
        auto gConst2 = ngraph::opset6::Constant::create(ngraph::element::Type_t::i64, {}, {0});
        auto gather = std::make_shared<ngraph::opset6::Gather>(shapeOf, gConst1, gConst2);
        auto concatConst = ngraph::opset6::Constant::create(ngraph::element::Type_t::i64, {1}, {1});
        auto concat =
            std::make_shared<ngraph::opset6::Concat>(ngraph::NodeVector{concatConst, gather}, 0);
        auto relu = std::make_shared<ngraph::opset6::Relu>(param);
        auto reshape = std::make_shared<ngraph::opset6::Reshape>(relu, concat, false);
        auto matMul = std::make_shared<ngraph::opset6::MatMul>(reshape, matMulWeights, false, true);
        auto matMulBias =
            ngraph::opset6::Constant::create(ngraph::element::Type_t::f32, {1, 10}, {1});
        auto addBias = std::make_shared<ngraph::opset6::Add>(matMul, matMulBias);
        auto result = std::make_shared<ngraph::opset6::Result>(addBias);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        original_f = std::make_shared<ngraph::Function>(results, params);
    }

    std::unordered_map<std::string, std::string> affinity;

    for (auto&& node : original_f->get_ordered_ops())
    {
        auto& nodeInfo = node->get_rt_info();

        nodeInfo["affinity"] = std::make_shared<ngraph::VariantWrapper<std::string>>(testAffinity);
        affinity[node->get_friendly_name()] = testAffinity;

        for (auto&& output : node->outputs())
        {
            auto& outputInfo = output.get_rt_info();
            outputInfo["affinity"] =
                std::make_shared<ngraph::VariantWrapper<std::string>>(testAffinity);
        }
    }

    auto clonedFunction = ngraph::clone_function(*original_f);

    for (auto&& node : clonedFunction->get_ordered_ops())
    {
        auto& nodeInfo = node->get_rt_info();
        auto itInfo = nodeInfo.find("affinity");
        ASSERT_TRUE(itInfo != nodeInfo.end());
        auto value =
            ngraph::as_type_ptr<ngraph::VariantWrapper<std::string>>(itInfo->second)->get();
        ASSERT_TRUE(affinity.find(node->get_friendly_name()) != affinity.end());
        ASSERT_TRUE(affinity[node->get_friendly_name()] == value);

        for (auto&& output : node->outputs())
        {
            auto& outputInfo = output.get_rt_info();
            ASSERT_TRUE(outputInfo.count("affinity"));
        }
    }
}

TEST(util, round_up)
{
    EXPECT_EQ(0, round_up(0, 4));
    EXPECT_EQ(4, round_up(1, 4));
    EXPECT_EQ(4, round_up(2, 4));
    EXPECT_EQ(4, round_up(3, 4));
    EXPECT_EQ(4, round_up(4, 4));
    EXPECT_EQ(8, round_up(5, 4));
}

TEST(util, parse_string)
{
    EXPECT_FLOAT_EQ(2, parse_string<float>("2"));
    EXPECT_FLOAT_EQ(2.125, parse_string<float>("2.125"));
    EXPECT_FLOAT_EQ(numeric_limits<float>::infinity(), parse_string<float>("INFINITY"));
    EXPECT_FLOAT_EQ(numeric_limits<float>::infinity(), parse_string<float>("infinity"));
    EXPECT_FLOAT_EQ(-numeric_limits<float>::infinity(), parse_string<float>("-INFINITY"));
    EXPECT_TRUE(isnan(parse_string<float>("NaN")));

    EXPECT_FLOAT_EQ(2, parse_string<double>("2"));
    EXPECT_FLOAT_EQ(2.125, parse_string<double>("2.125"));
    EXPECT_FLOAT_EQ(numeric_limits<double>::infinity(), parse_string<double>("INFINITY"));
    EXPECT_FLOAT_EQ(numeric_limits<double>::infinity(), parse_string<double>("infinity"));
    EXPECT_FLOAT_EQ(-numeric_limits<double>::infinity(), parse_string<double>("-INFINITY"));
    EXPECT_TRUE(std::isnan(parse_string<double>("NaN")));
}

TEST(graph_util, get_subgraph_outputs_trivial_tests)
{
    auto outputs = ngraph::get_subgraph_outputs(NodeVector{}, NodeVector{});
    ASSERT_EQ(outputs.size(), 0);

    Shape shape{};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto absn = make_shared<op::Abs>(A);
    auto neg_absn = make_shared<op::Negative>(absn);
    outputs = ngraph::get_subgraph_outputs(NodeVector{A}, NodeVector{});
    ASSERT_EQ(outputs, (NodeVector{A}));

    outputs = ngraph::get_subgraph_outputs(NodeVector{A}, NodeVector{A});
    ASSERT_EQ(outputs, (NodeVector{}));

    outputs = ngraph::get_subgraph_outputs(NodeVector{A, absn}, NodeVector{});
    ASSERT_EQ(outputs, (NodeVector{absn}));

    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto abs_b = make_shared<op::Abs>(B);
    auto neg_b = make_shared<op::Negative>(B);
    auto abs_b_neg = make_shared<op::Negative>(abs_b);
    outputs = ngraph::get_subgraph_outputs(NodeVector{B, abs_b}, NodeVector{});
    ASSERT_EQ(outputs, (NodeVector{B, abs_b}));

    outputs = ngraph::get_subgraph_outputs(NodeVector{B, abs_b}, NodeVector{B});
    ASSERT_EQ(outputs, (NodeVector{abs_b}));

    outputs = ngraph::get_subgraph_outputs(NodeVector{B, abs_b, abs_b_neg}, NodeVector{});
    ASSERT_EQ(outputs, (NodeVector{B}));

    auto add_b = make_shared<op::v1::Add>(neg_b, abs_b_neg);
    outputs =
        ngraph::get_subgraph_outputs(NodeVector{B, abs_b, neg_b, abs_b_neg, add_b}, NodeVector{});
    ASSERT_EQ(outputs, (NodeVector{}));

    // now add_b uses abs_b_neg
    outputs = ngraph::get_subgraph_outputs(NodeVector{B, abs_b, abs_b_neg}, NodeVector{});
    ASSERT_EQ(outputs, (NodeVector{B, abs_b_neg}));
}

TEST(graph_util, test_subgraph_topological_sort)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto add = make_shared<op::v1::Add>(A, B);
    auto mul = make_shared<op::v1::Multiply>(C, add);
    auto result = make_shared<op::Result>(mul);
    auto sorted = ngraph::subgraph_topological_sort(NodeVector{mul, add, A});
    std::vector<std::shared_ptr<Node>> expected{A, add, mul};
    ASSERT_EQ(expected, sorted);
}

TEST(graph_util, test_subgraph_topological_sort_control_dependencies)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto D = make_shared<op::Abs>(A);
    auto E = make_shared<op::Abs>(B);
    auto add = make_shared<op::v1::Add>(A, B);
    add->add_control_dependency(D);
    add->add_control_dependency(E);
    auto mul = make_shared<op::v1::Multiply>(C, add);
    auto result = make_shared<op::Result>(mul);
    auto sorted = ngraph::subgraph_topological_sort(NodeVector{mul, add, A, D});
    std::vector<std::shared_ptr<Node>> expected{A, D, add, mul};
    ASSERT_EQ(expected, sorted);
}

TEST(util, enum_mask_construction)
{
    enum class Type : uint32_t
    {
        a = 0x1,
        b = 1 << 1,
        c = 1 << 2,
        d = 1 << 3
    };
    {
        EnumMask<Type> m;
        EXPECT_EQ(0, m.value());
    }
    {
        EnumMask<Type> m(Type::c);
        EXPECT_EQ(static_cast<uint32_t>(Type::c), m.value());
    }
    {
        EnumMask<Type> a(Type::c);
        EnumMask<Type> b{a};
        EXPECT_EQ(a.value(), b.value());
    }
    {
        EnumMask<Type> a{Type::a, Type::c, Type::d};
        EXPECT_EQ((static_cast<uint32_t>(Type::a) | static_cast<uint32_t>(Type::c) |
                   static_cast<uint32_t>(Type::d)),
                  a.value());
    }
}

TEST(util, enum_mask_set_clear)
{
    enum class Type : uint32_t
    {
        a = 0x1,
        b = 1 << 1,
        c = 1 << 2,
        d = 1 << 3
    };
    EnumMask<Type> m;
    m.set(Type::b);
    EXPECT_EQ(static_cast<uint32_t>(Type::b), m.value());
    m.set(Type::c);
    EXPECT_EQ(static_cast<uint32_t>(Type::b) | static_cast<uint32_t>(Type::c), m.value());
    m.clear(Type::b);
    EXPECT_EQ(static_cast<uint32_t>(Type::c), m.value());
    m.clear_all();
    EXPECT_EQ(0, m.value());
    m.set(Type::d);
    m.set(Type::b);
    EXPECT_TRUE(m.is_set(Type::d));
    EXPECT_FALSE(m.is_set(Type::a));
    EXPECT_TRUE(m.is_set(Type::b));
    EXPECT_FALSE(m.is_set(Type::c));
    EXPECT_FALSE(m.is_set({Type::a, Type::b}));
    EXPECT_FALSE(m.is_set({Type::c, Type::d}));
    EXPECT_FALSE(m.is_set({Type::a, Type::c}));
    EXPECT_TRUE(m.is_set({Type::b, Type::d}));
    EXPECT_FALSE(m.is_clear(Type::d));
    EXPECT_TRUE(m.is_clear(Type::a));
    EXPECT_FALSE(m.is_clear(Type::b));
    EXPECT_TRUE(m.is_clear(Type::c));
    EXPECT_FALSE(m.is_clear({Type::c, Type::d}));
    EXPECT_FALSE(m.is_clear({Type::a, Type::b}));
    EXPECT_TRUE(m.is_clear({Type::a, Type::c}));
    EXPECT_FALSE(m.is_clear({Type::b, Type::d}));

    EXPECT_TRUE(m.is_any_set({Type::a, Type::b}));
    EXPECT_TRUE(m.is_any_set({Type::a, Type::d}));
    EXPECT_TRUE(m.is_any_set({Type::b, Type::c}));
    EXPECT_TRUE(m.is_any_set({Type::c, Type::d}));
    EXPECT_FALSE(m.is_any_set({Type::a, Type::c}));
    EXPECT_TRUE(m.is_any_clear({Type::c, Type::d}));
    EXPECT_TRUE(m.is_any_clear({Type::a, Type::b}));
    EXPECT_TRUE(m.is_any_clear({Type::a, Type::c}));
    EXPECT_TRUE(m.is_any_clear({Type::b, Type::c}));
    EXPECT_FALSE(m.is_any_clear({Type::b, Type::d}));

    m.set(Type::a);
    EXPECT_FALSE(m.is_clear(Type::a));
    EXPECT_FALSE(m.is_clear(Type::b));
    EXPECT_TRUE(m.is_clear(Type::c));
    EXPECT_FALSE(m.is_clear(Type::d));
}

TEST(util, enum_mask_operators)
{
    enum class Type : uint32_t
    {
        a = 0x1,
        b = 1 << 1,
        c = 1 << 2,
        d = 1 << 3
    };
    EnumMask<Type> m;
    m = Type::b;
    EXPECT_EQ(static_cast<uint32_t>(Type::b), m.value());
    EXPECT_TRUE(m[Type::b]);
    EXPECT_FALSE(m[Type::a]);
    EXPECT_FALSE(m[Type::c]);
    m |= Type::c;
    EXPECT_EQ(static_cast<uint32_t>(Type::b) | static_cast<uint32_t>(Type::c), m.value());
    m &= Type::d;
    EXPECT_EQ(0, m.value());

    m |= Type::a;
    m |= Type::c;
    EXPECT_TRUE(m.is_set(Type::a));
    EXPECT_FALSE(m.is_set(Type::b));
    EXPECT_TRUE(m.is_set(Type::c));
    EXPECT_FALSE(m.is_set(Type::d));
    EXPECT_TRUE(m.is_any_set(Type::a));
    EXPECT_FALSE(m.is_any_set(Type::b));
    EXPECT_TRUE(m.is_any_set(Type::c));
    EXPECT_FALSE(m.is_any_set(Type::d));
    EXPECT_TRUE(m.is_any_set({Type::a, Type::c}));
    EXPECT_FALSE(m.is_any_set({Type::b, Type::d}));

    EnumMask<Type> n;
    n = m | n;
    EXPECT_EQ(m, n);
    n = m & n;
    EXPECT_EQ(m, n);
    bool r = (n == m);
    EXPECT_TRUE(r);
    r = (n != m);
    EXPECT_FALSE(r);
    n.clear_all();
    n = {Type::a, Type::b};
    r = (n == m);
    EXPECT_FALSE(r);
    r = (n != m);
    EXPECT_TRUE(r);
    n = m & n;
    EXPECT_EQ(static_cast<uint32_t>(Type::a), n.value());
    n = m | Type::b;
    EXPECT_TRUE(n.is_set(Type::a));
    EXPECT_TRUE(n.is_set(Type::b));
    EXPECT_TRUE(n.is_set(Type::c));
    EXPECT_FALSE(n.is_set(Type::d));
    EXPECT_FALSE(n[Type::d]);
    EXPECT_TRUE(n[Type::b]);
}

TEST(graph, huge)
{
    std::vector<std::weak_ptr<Node>> weak_nodes;
    {
        auto param = make_shared<op::Parameter>(element::f32, Shape{3, 3});
        std::shared_ptr<Node> n = param;
        weak_nodes.push_back(n);
        for (size_t i = 0; i < 1000000; i++)
        {
            n = make_shared<op::Negative>(n);
            weak_nodes.push_back(n);
        }
        auto f = make_shared<Function>(NodeVector{n}, ParameterVector{param});
    }

    for (auto& weak_node : weak_nodes)
    {
        EXPECT_TRUE(weak_node.expired());
    }
}

TEST(util, apply_permutation)
{
    ASSERT_EQ(apply_permutation(Shape{0, 1, 2, 3}, AxisVector{2, 1, 0, 3}), (Shape{2, 1, 0, 3}));
}

TEST(util, apply_permutation_too_short_fails)
{
    ASSERT_THROW(apply_permutation(Shape{0, 1, 2, 3}, AxisVector{0, 1, 2}), CheckFailure);
}

TEST(util, apply_permutation_too_long_fails)
{
    ASSERT_THROW(apply_permutation(Shape{0, 1, 2, 3}, AxisVector{0, 1, 2, 3, 3}), CheckFailure);
}

TEST(util, apply_permutation_oob_axis_fails)
{
    ASSERT_THROW(apply_permutation(Shape{0, 1, 2, 3}, AxisVector{0, 1, 2, 4}), CheckFailure);
}

TEST(util, apply_permutation_repeated_axis_fails)
{
    ASSERT_THROW(apply_permutation(Shape{0, 1, 2, 3}, AxisVector{0, 1, 2, 2}), CheckFailure);
}

TEST(util, apply_permutation_pshape)
{
    ASSERT_TRUE(
        apply_permutation(PartialShape{0, Dimension::dynamic(), 2, 3}, AxisVector{2, 1, 0, 3})
            .same_scheme(PartialShape{2, Dimension::dynamic(), 0, 3}));
}

TEST(util, apply_permutation_pshape_rank_dynamic)
{
    ASSERT_TRUE(apply_permutation(PartialShape::dynamic(), AxisVector{2, 1, 0, 3})
                    .same_scheme(PartialShape::dynamic()));
}

TEST(util, apply_permutation_pshape_too_short_fails)
{
    ASSERT_THROW(
        apply_permutation(PartialShape{0, Dimension::dynamic(), 2, 3}, AxisVector{0, 1, 2}),
        CheckFailure);
}

TEST(util, apply_permutation_pshape_too_long_fails)
{
    ASSERT_THROW(
        apply_permutation(PartialShape{0, Dimension::dynamic(), 2, 3}, AxisVector{0, 1, 2, 3, 3}),
        CheckFailure);
}

TEST(util, apply_permutation_pshape_oob_axis_fails)
{
    ASSERT_THROW(
        apply_permutation(PartialShape{0, Dimension::dynamic(), 2, 3}, AxisVector{0, 1, 2, 4}),
        CheckFailure);
}

TEST(util, apply_permutation_pshape_repeated_axis_fails)
{
    ASSERT_THROW(
        apply_permutation(PartialShape{0, Dimension::dynamic(), 2, 3}, AxisVector{0, 1, 2, 2}),
        CheckFailure);
}

TEST(util, apply_permutation_pshape_rank_dynamic_inviable_permutation_fails)
{
    ASSERT_THROW(apply_permutation(PartialShape::dynamic(), AxisVector{0, 1, 2, 2}), CheckFailure);
}

TEST(util, clone_function_friendly_name)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Add>(A, B), ParameterVector{A, B});

    A->set_friendly_name("A");
    B->set_friendly_name("B");

    auto g = clone_function(*f);

    bool found_A = false;
    bool found_B = false;
    for (auto parameter : g->get_parameters())
    {
        found_A |= parameter->get_friendly_name() == "A";
        found_B |= parameter->get_friendly_name() == "B";
    }
    EXPECT_TRUE(found_A);
    EXPECT_TRUE(found_B);
}

TEST(util, clone_function_op_annotations)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Add>(make_shared<op::v1::Add>(A, B), C),
                                   ParameterVector{A, B, C});

    auto cacheable_op_annotation = std::make_shared<op::util::OpAnnotations>();
    cacheable_op_annotation->set_cacheable(true);
    A->set_op_annotations(cacheable_op_annotation);

    auto uncacheable_op_annotation = std::make_shared<op::util::OpAnnotations>();
    uncacheable_op_annotation->set_cacheable(false);
    B->set_op_annotations(uncacheable_op_annotation);

    auto g = clone_function(*f);

    bool found_A = false;
    bool found_B = false;
    for (auto parameter : g->get_parameters())
    {
        if (auto op_annotation = parameter->get_op_annotations())
        {
            if (op_annotation->is_cacheable())
            {
                found_A = true;
            }
            else
            {
                found_B = true;
            }
        }
    }
    EXPECT_TRUE(found_A);
    EXPECT_TRUE(found_B);
}

TEST(util, topological_sort_replace)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Add>(make_shared<op::v1::Add>(A, B), C),
                                   ParameterVector{A, B, C});
    bool custom_sorter_used = false;

    f->set_topological_sort(
        [&custom_sorter_used](const std::vector<std::shared_ptr<Node>>& root_nodes) {
            custom_sorter_used = true;
            return topological_sort(root_nodes);
        });

    // Need to now call topological sort but don't care about the results
    f->get_ordered_ops();

    EXPECT_TRUE(custom_sorter_used);
}

TEST(util, double_to_int_limits)
{
    auto round_func = [](double x) { return std::round(x); };

    double x = -std::numeric_limits<double>::infinity();

    EXPECT_TRUE(std::numeric_limits<int8_t>::min() == double_to_int<int8_t>(x, round_func));
    EXPECT_TRUE(std::numeric_limits<int16_t>::min() == double_to_int<int16_t>(x, round_func));
    EXPECT_TRUE(std::numeric_limits<int32_t>::min() == double_to_int<int32_t>(x, round_func));
    EXPECT_TRUE(std::numeric_limits<int64_t>::min() == double_to_int<int64_t>(x, round_func));

    EXPECT_TRUE(std::numeric_limits<uint8_t>::min() == double_to_int<uint8_t>(x, round_func));
    EXPECT_TRUE(std::numeric_limits<uint16_t>::min() == double_to_int<uint16_t>(x, round_func));
    EXPECT_TRUE(std::numeric_limits<uint32_t>::min() == double_to_int<uint32_t>(x, round_func));
    EXPECT_TRUE(std::numeric_limits<uint64_t>::min() == double_to_int<uint64_t>(x, round_func));

    x = std::numeric_limits<double>::infinity();

    EXPECT_TRUE(std::numeric_limits<int8_t>::max() == double_to_int<int8_t>(x, round_func));
    EXPECT_TRUE(std::numeric_limits<int16_t>::max() == double_to_int<int16_t>(x, round_func));
    EXPECT_TRUE(std::numeric_limits<int32_t>::max() == double_to_int<int32_t>(x, round_func));
    EXPECT_TRUE(std::numeric_limits<int64_t>::max() == double_to_int<int64_t>(x, round_func));

    EXPECT_TRUE(std::numeric_limits<uint8_t>::max() == double_to_int<uint8_t>(x, round_func));
    EXPECT_TRUE(std::numeric_limits<uint16_t>::max() == double_to_int<uint16_t>(x, round_func));
    EXPECT_TRUE(std::numeric_limits<uint32_t>::max() == double_to_int<uint32_t>(x, round_func));
    EXPECT_TRUE(std::numeric_limits<uint64_t>::max() == double_to_int<uint64_t>(x, round_func));
}

TEST(util, double_to_int_assert)
{
    auto round_func = [](double x) { return std::round(x); };
    ASSERT_THROW(double_to_int<float>(123.123, round_func), std::runtime_error);
    ASSERT_THROW(double_to_int<double>(123.123, round_func), std::runtime_error);
}

TEST(util, double_to_int)
{
    auto ceil_func = [](double x) { return std::ceil(x); };
    auto floor_func = [](double x) { return std::floor(x); };
    auto round_func = [](double x) { return std::round(x); };

    double x = -1.5;
    EXPECT_TRUE(double_to_int<int32_t>(x, ceil_func) == -1);
    EXPECT_TRUE(double_to_int<int32_t>(x, floor_func) == -2);
    EXPECT_TRUE(double_to_int<int32_t>(x, round_func) == -2);

    x = 1.5;
    EXPECT_TRUE(double_to_int<int32_t>(x, ceil_func) == 2);
    EXPECT_TRUE(double_to_int<int32_t>(x, floor_func) == 1);
    EXPECT_TRUE(double_to_int<int32_t>(x, round_func) == 2);
}

template <typename hosttensor_t, typename vector_t>
void host_tensor_2_vector_test(const vector<hosttensor_t>& input,
                               const vector<vector_t>& output,
                               const element::Type& hosttensor_elem_t)
{
    auto tensor = make_shared<HostTensor>(hosttensor_elem_t, Shape{2, 2});
    tensor->write(input.data(), input.size() * sizeof(hosttensor_t));
    auto result = host_tensor_2_vector<vector_t>(tensor);

    ASSERT_TRUE(test::all_close(result, output));
}

TEST(util_host_tensor_2_vector, tensor_nullptr)
{
    ASSERT_THROW(host_tensor_2_vector<int64_t>(nullptr), ngraph::CheckFailure);
}

TEST(util_host_tensor_2_vector, ht_boolean_2_vec_bool)
{
    vector<char> input{1, 0, 1, 0};
    vector<bool> output{true, false, true, false};
    host_tensor_2_vector_test<decltype(input)::value_type, decltype(output)::value_type>(
        input, output, element::boolean);
}

TEST(util_host_tensor_2_vector, ht_boolean_2_vec_int64)
{
    vector<char> input{1, 0, 1, 0};
    vector<int64_t> output{true, false, true, false};
    host_tensor_2_vector_test<decltype(input)::value_type, decltype(output)::value_type>(
        input, output, element::boolean);
}

TEST(util_host_tensor_2_vector, ht_i8_2_vec_int64)
{
    vector<int8_t> input{
        0, 1, std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()};
    vector<int64_t> output{
        0, 1, std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()};
    host_tensor_2_vector_test<decltype(input)::value_type, decltype(output)::value_type>(
        input, output, element::i8);
}

TEST(util_host_tensor_2_vector, ht_i16_2_vec_int64)
{
    vector<int16_t> input{
        0, 1, std::numeric_limits<int16_t>::min(), std::numeric_limits<int16_t>::max()};
    vector<int64_t> output{
        0, 1, std::numeric_limits<int16_t>::min(), std::numeric_limits<int16_t>::max()};
    host_tensor_2_vector_test<decltype(input)::value_type, decltype(output)::value_type>(
        input, output, element::i16);
}

TEST(util_host_tensor_2_vector, ht_i32_2_vec_int64)
{
    vector<int32_t> input{
        0, 1, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()};
    vector<int64_t> output{
        0, 1, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()};
    host_tensor_2_vector_test<decltype(input)::value_type, decltype(output)::value_type>(
        input, output, element::i32);
}

TEST(util_host_tensor_2_vector, ht_i64_2_vec_int64)
{
    vector<int64_t> input{
        0, 1, std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()};
    vector<int64_t> output{input};
    host_tensor_2_vector_test<decltype(input)::value_type, decltype(output)::value_type>(
        input, output, element::i64);
}

TEST(util_host_tensor_2_vector, ht_bf16_2_vec_double)
{
    vector<bfloat16> input{
        0, 1, std::numeric_limits<bfloat16>::min(), std::numeric_limits<bfloat16>::max()};
    vector<double> output{
        0, 1, std::numeric_limits<bfloat16>::min(), std::numeric_limits<bfloat16>::max()};
    host_tensor_2_vector_test<decltype(input)::value_type, decltype(output)::value_type>(
        input, output, element::bf16);
}

TEST(util_host_tensor_2_vector, ht_f16_2_vec_double)
{
    vector<float16> input{
        0, 1, std::numeric_limits<float16>::min(), std::numeric_limits<float16>::max()};
    vector<double> output{
        0, 1, std::numeric_limits<float16>::min(), std::numeric_limits<float16>::max()};
    host_tensor_2_vector_test<decltype(input)::value_type, decltype(output)::value_type>(
        input, output, element::f16);
}

TEST(util_host_tensor_2_vector, ht_f32_2_vec_double)
{
    vector<float> input{0, 1, std::numeric_limits<float>::min(), std::numeric_limits<float>::max()};
    vector<double> output{
        0, 1, std::numeric_limits<float>::min(), std::numeric_limits<float>::max()};
    host_tensor_2_vector_test<decltype(input)::value_type, decltype(output)::value_type>(
        input, output, element::f32);
}

TEST(util_host_tensor_2_vector, ht_f64_2_vec_double)
{
    vector<double> input{
        0, 1, std::numeric_limits<double>::min(), std::numeric_limits<double>::max()};
    vector<double> output{
        0, 1, std::numeric_limits<double>::min(), std::numeric_limits<double>::max()};
    host_tensor_2_vector_test<decltype(input)::value_type, decltype(output)::value_type>(
        input, output, element::f64);
}

TEST(util_host_tensor_2_vector, ht_u8_2_vec_uint64)
{
    vector<uint8_t> input{
        0, 1, std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max()};
    vector<uint64_t> output{
        0, 1, std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max()};
    host_tensor_2_vector_test<decltype(input)::value_type, decltype(output)::value_type>(
        input, output, element::u8);
}

TEST(util_host_tensor_2_vector, ht_u16_2_vec_uint64)
{
    vector<uint16_t> input{
        0, 1, std::numeric_limits<uint16_t>::min(), std::numeric_limits<uint16_t>::max()};
    vector<uint64_t> output{
        0, 1, std::numeric_limits<uint16_t>::min(), std::numeric_limits<uint16_t>::max()};
    host_tensor_2_vector_test<decltype(input)::value_type, decltype(output)::value_type>(
        input, output, element::u16);
}

TEST(util_host_tensor_2_vector, ht_u32_2_vec_uint64)
{
    vector<uint32_t> input{
        0, 1, std::numeric_limits<uint32_t>::min(), std::numeric_limits<uint32_t>::max()};
    vector<uint64_t> output{
        0, 1, std::numeric_limits<uint32_t>::min(), std::numeric_limits<uint32_t>::max()};
    host_tensor_2_vector_test<decltype(input)::value_type, decltype(output)::value_type>(
        input, output, element::u32);
}

TEST(util_host_tensor_2_vector, ht_u64_2_vec_uint64)
{
    vector<uint64_t> input{
        0, 1, std::numeric_limits<uint64_t>::min(), std::numeric_limits<uint64_t>::max()};
    vector<uint64_t> output{input};
    host_tensor_2_vector_test<decltype(input)::value_type, decltype(output)::value_type>(
        input, output, element::u64);
}
