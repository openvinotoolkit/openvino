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

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ngraph/builder/norm.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/fused_op_decomposition.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/opset1_upgrade.hpp"
#include "ngraph/provenance.hpp"
#include "opset0_downgrade.hpp"
#include "util/provenance_enabler.hpp"

using namespace std;
using namespace ngraph;
using ::testing::Return;

using ProvSet = std::unordered_set<std::string>;

TEST(provenance, provenance)
{
    test::ProvenanceEnabler provenance_enabler;

    //
    // Before:
    //
    //   A{tag_a}  B{tag_b}
    //         |   |
    //        C{tag_c}
    //
    // Replacement:
    //
    //       A{tag_a} B{tag_b}
    //              | |
    //         C := D{}
    //
    // After:
    //
    //   A{tag_a}  B{tag_b}
    //         |   |
    //        D{tag_c}
    //
    // Comment:
    //   * D is the replacement root, and its insertion kills C. We should not, however, consider
    //     A and B to be killed, because they are not post-dominated by D until after C is cut out
    //     of the graph.
    //
    {
        auto x = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});
        auto y = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});

        auto a = make_shared<op::Add>(x, y);
        a->add_provenance_tag("tag_a");
        auto b = make_shared<op::Multiply>(y, x);
        b->add_provenance_tag("tag_b");
        auto c = make_shared<op::Subtract>(a, b);
        c->add_provenance_tag("tag_c");

        auto f = make_shared<Function>(c, ParameterVector{x, y});

        auto new_c = make_shared<op::Subtract>(a, b);
        replace_node(c, new_c);

        EXPECT_EQ(new_c->get_provenance_tags(), ProvSet{"tag_c"});
    }

    //
    // Before:
    //
    //   A{tag_a}  B{tag_b}
    //         |   |
    //        C{tag_c}
    //
    // Replacement:
    //
    //
    //
    //     A{tag_a}  B{tag_b}
    //        |      |
    //   C -> D{tag_d}
    //
    // After:
    //
    //   A{tag_a}  B{tag_b}
    //         |   |
    //        D{tag_c,tag_d}
    //
    // Comment:
    //   * D is the replacement root, and its insertion kills C. We should not, however, consider
    //     A and B to be killed, because they are not post-dominated by D until after C is cut out
    //     of the graph.
    //
    {
        auto x = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});
        auto y = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});

        auto a = make_shared<op::Add>(x, y);
        a->add_provenance_tag("tag_a");
        auto b = make_shared<op::Multiply>(y, x);
        b->add_provenance_tag("tag_b");
        auto c = make_shared<op::Subtract>(a, b);
        c->add_provenance_tag("tag_c");

        auto f = make_shared<Function>(c, ParameterVector{x, y});

        auto d = make_shared<op::Subtract>(a, b);
        d->add_provenance_tag("tag_d");
        replace_node(c, d);

        EXPECT_EQ(d->get_provenance_tags(), (ProvSet{"tag_c", "tag_d"}));
    }

    //
    // Before:
    //
    //   A{tag_a}  B{tag_b}
    //         |   |
    //        C{tag_c}
    //
    // Replacement:
    //
    //   C -> D{tag_d}
    //
    // After:
    //
    //   D{tag_a,tag_b,tag_c,tag_d}
    //
    // Comment:
    //   * D is the replacement root, and its insertion kills A, B, and C.
    //
    {
        auto x = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});
        auto y = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});

        auto a = make_shared<op::Add>(x, y);
        a->add_provenance_tag("tag_a");
        auto b = make_shared<op::Multiply>(y, x);
        b->add_provenance_tag("tag_b");
        auto c = make_shared<op::Subtract>(a, b);
        c->add_provenance_tag("tag_c");

        auto f = make_shared<Function>(c, ParameterVector{x, y});

        auto d = make_zero(element::i32, Shape{2, 3, 4});
        d->add_provenance_tag("tag_d");
        replace_node(c, d);

        EXPECT_EQ(d->get_provenance_tags(), (ProvSet{"tag_a", "tag_b", "tag_c", "tag_d"}));
    }

    //
    // Before:
    //
    //   A{tag_a}  B{tag_b}
    //         |   |
    //        C{tag_c}
    //
    // Replacement:
    //
    //   C -> D{}
    //
    // After:
    //
    //   D{tag_a,tag_b,tag_c}
    //
    // Comment:
    //   * D is the replacement root, and its insertion kills A, B, and C.
    //
    {
        auto x = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});
        auto y = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});

        auto a = make_shared<op::Add>(x, y);
        a->add_provenance_tag("tag_a");
        auto b = make_shared<op::Multiply>(y, x);
        b->add_provenance_tag("tag_b");
        auto c = make_shared<op::Subtract>(a, b);
        c->add_provenance_tag("tag_c");

        auto f = make_shared<Function>(c, ParameterVector{x, y});

        auto d = make_zero(element::i32, Shape{2, 3, 4});
        replace_node(c, d);

        EXPECT_EQ(d->get_provenance_tags(), (ProvSet{"tag_a", "tag_b", "tag_c"}));
    }

    //
    // Before:
    //
    //   A{tag_a}  B{tag_b}
    //         |   |
    //        C{tag_c}
    //
    //
    // Replacement:
    //
    //   A{tag_a}  B{tag_b}
    //         |     |
    //        E{}    |
    //         |     |
    //    C -> D{tag_d}
    //
    //
    // After:
    //
    //   A{tag_a}          B{tag_b}
    //         |             |
    //      E{tag_c}         |
    //           |           |
    //          D{tag_c, tag_d}
    //
    // Comment:
    //   * D is the replacement root replacing C and creating a new argument node E
    //
    {
        auto x = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});
        auto y = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});

        auto a = make_shared<op::Add>(x, y);
        a->add_provenance_tag("tag_a");
        auto b = make_shared<op::Multiply>(y, x);
        b->add_provenance_tag("tag_b");
        auto c = make_shared<op::Subtract>(a, b);
        c->add_provenance_tag("tag_c");

        auto f = make_shared<Function>(c, ParameterVector{x, y});

        auto e = make_shared<op::Subtract>(a, x);
        auto d = make_shared<op::Subtract>(e, b);
        d->add_provenance_tag("tag_d");

        replace_node(c, d);

        EXPECT_EQ(d->get_provenance_tags(), (ProvSet{"tag_c", "tag_d"}));
        EXPECT_EQ(e->get_provenance_tags(), (ProvSet{"tag_c"}));
    }

    //
    // Before:
    //
    //   A{tag_a}  B{tag_b}
    //         |   |
    //        C{tag_c}
    //
    //
    // Replacement:
    //
    //   A{tag_a}  B{tag_b}
    //         |      |
    //       E{tag_e} |
    //           |    |
    //     C -> D{tag_d}
    //
    //
    // After:
    //
    //   A{tag_a}               B{tag_b}
    //       \                    /
    //   E{tag_c, tag_d, tag_e}  /
    //          \               /
    //           D{tag_c, tag_d}
    //
    // Comment:
    //   * D is the replacement root replacing C and creating a new argument node E
    //
    {
        auto x = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});
        auto y = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});

        auto a = make_shared<op::Add>(x, y);
        a->add_provenance_tag("tag_a");
        auto b = make_shared<op::Multiply>(y, x);
        b->add_provenance_tag("tag_b");
        auto c = make_shared<op::Subtract>(a, b);
        c->add_provenance_tag("tag_c");

        auto f = make_shared<Function>(c, ParameterVector{x, y});

        auto e = make_shared<op::Subtract>(a, x);
        e->add_provenance_tag("tag_e");
        auto d = make_shared<op::Subtract>(e, b);
        d->add_provenance_tag("tag_d");

        replace_node(c, d);

        EXPECT_EQ(d->get_provenance_tags(), (ProvSet{"tag_c", "tag_d"}));
        EXPECT_EQ(e->get_provenance_tags(), (ProvSet{"tag_c", "tag_e"}));
    }
}

TEST(provenance, add_group_above)
{
    auto p1 = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});
    p1->add_provenance_tag("P1");
    auto p2 = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});
    p2->add_provenance_tag("P2");
    auto a1 = p1 + p2;
    auto m1 = (a1 * a1)->add_provenance_group_members_above({p1, p2});
    m1->add_provenance_tag("m1");
    EXPECT_EQ(p1->get_provenance_tags(), (ProvSet{"P1"}));
    EXPECT_EQ(p2->get_provenance_tags(), (ProvSet{"P2"}));
    EXPECT_EQ(a1->get_provenance_tags(), (ProvSet{"m1"}));
    EXPECT_EQ(m1->get_provenance_tags(), (ProvSet{"m1"}));
}

TEST(provenance, add_tags_above)
{
    auto x = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});
    auto y = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});

    auto a = make_shared<op::Add>(x, y);
    auto b = make_shared<op::Multiply>(x, y);
    auto c = make_shared<op::Subtract>(a, b);
    auto d = make_shared<op::Abs>(c);

    // Add tags to Subtract and all nodes until Parameters (all above c, until params x, y)
    c->add_provenance_tags_above(OutputVector{x, y}, {"tag_above_c - until_params"});
    // Add tags to Abs and Subtract (above d, until c inputs)
    d->add_provenance_tags_above(c->input_values(), {"tag_above_d - until_c_inputs"});
    // Add tags to Abs and all nodes above
    d->add_provenance_tags_above(OutputVector{}, {"tag_all_above_d"});

    auto x_tags = x->get_provenance_tags();
    EXPECT_EQ(x_tags.size(), 1);
    EXPECT_TRUE(x_tags.find("tag_all_above_d") != x_tags.end());

    auto y_tags = y->get_provenance_tags();
    EXPECT_EQ(y_tags.size(), 1);
    EXPECT_TRUE(y_tags.find("tag_all_above_d") != y_tags.end());

    auto a_tags = a->get_provenance_tags();
    EXPECT_EQ(a_tags.size(), 2);
    EXPECT_TRUE(a_tags.find("tag_above_c - until_params") != a_tags.end());
    EXPECT_FALSE(a_tags.find("tag_above_d - until_c_inputs") != a_tags.end());
    EXPECT_TRUE(a_tags.find("tag_all_above_d") != a_tags.end());

    auto b_tags = b->get_provenance_tags();
    EXPECT_EQ(b_tags.size(), 2);
    EXPECT_TRUE(b_tags.find("tag_above_c - until_params") != b_tags.end());
    EXPECT_FALSE(b_tags.find("tag_above_d - until_c_inputs") != b_tags.end());
    EXPECT_TRUE(b_tags.find("tag_all_above_d") != b_tags.end());

    auto c_tags = c->get_provenance_tags();
    EXPECT_EQ(c_tags.size(), 3);
    EXPECT_TRUE(c_tags.find("tag_above_c - until_params") != c_tags.end());
    EXPECT_TRUE(c_tags.find("tag_above_d - until_c_inputs") != c_tags.end());
    EXPECT_TRUE(c_tags.find("tag_all_above_d") != c_tags.end());

    auto d_tags = d->get_provenance_tags();
    EXPECT_EQ(d_tags.size(), 2);
    EXPECT_FALSE(d_tags.find("tag_above_c - until_params") != d_tags.end());
    EXPECT_TRUE(d_tags.find("tag_above_d - until_c_inputs") != d_tags.end());
    EXPECT_TRUE(d_tags.find("tag_all_above_d") != d_tags.end());
}

TEST(provenance, builder)
{
    auto p1 = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});
    p1->add_provenance_tag("P1");
    auto norm = builder::opset1::lp_norm(p1, op::Constant::create(element::i64, {}, {0}), 1, 0);
    norm->add_provenance_tag("norm");
    for (auto node : topological_sort(NodeVector{norm}))
    {
        if (node == p1)
        {
            EXPECT_EQ(node->get_provenance_tags(), (ProvSet{"P1"}));
        }
        else
        {
            EXPECT_EQ(node->get_provenance_tags(), (ProvSet{"norm"}));
        }
    }
}

TEST(provenance, fused_copy_origin_tags)
{
    test::ProvenanceEnabler provenance_enabler;

    auto p1 = make_shared<op::Parameter>(element::f32, PartialShape{2, 3, 4});
    p1->add_provenance_tag("P1");
    auto g = make_shared<op::Gelu>(p1);
    g->add_provenance_tag("G");
    auto r = make_shared<op::Result>(g);
    auto f = make_shared<Function>(ResultVector{r}, ParameterVector{p1});

    pass::Manager manager;
    manager.register_pass<pass::FusedOpDecomposition>();
    manager.run_passes(f);

    traverse_nodes(f, [&](const std::shared_ptr<Node>& node) {
        auto tags = node->get_provenance_tags();
        if (node == p1)
        {
            EXPECT_EQ(tags.size(), 1);
            EXPECT_TRUE(tags.find("P1") != tags.end());
        }
        else if (node == r)
        {
        }
        else
        {
            EXPECT_TRUE(tags.find("G") != tags.end());
            EXPECT_TRUE(tags.find("<Decomposed from Gelu>") != tags.end());
        }
    });
}

TEST(provenance, fused_decomposition_tag)
{
    test::ProvenanceEnabler provenance_enabler;

    auto p1 = make_shared<op::Parameter>(element::f32, PartialShape{2, 3, 4});
    auto fused_op = make_shared<op::MVN>(p1);
    auto result = make_shared<op::Result>(fused_op);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{p1});

    pass::Manager manager;
    manager.register_pass<pass::FusedOpDecomposition>();
    manager.run_passes(f);

    const auto tag = "<Decomposed from MVN>";
    auto tag_check = [&tag](std::shared_ptr<ngraph::Node> node) {
        auto tags = node->get_provenance_tags();
        EXPECT_TRUE(tags.find(tag) != tags.end());
    };
    const auto decomposed_op = f->get_result()->get_input_node_shared_ptr(0);
    traverse_nodes(as_node_vector(decomposed_op->outputs()), tag_check, {p1});
}

TEST(provenance, topk_setk)
{
    auto p1 = make_shared<op::Parameter>(element::f32, PartialShape{20, 3, 4});
    p1->add_provenance_tag("P1");
    auto tk = make_shared<op::TopK>(p1, 0, element::i32, 10);
    tk->add_provenance_tag("TK");
    auto tkc0 = tk->input_value(1).get_node_shared_ptr();
    tkc0->add_provenance_tag("TKC0");
    for (auto node : topological_sort(NodeVector{tk}))
    {
        if (node == p1)
        {
            EXPECT_EQ(node->get_provenance_tags(), (ProvSet{"P1"}));
        }
        else if (node == tkc0)
        {
            EXPECT_EQ(node->get_provenance_tags(), (ProvSet{"TK", "TKC0"}));
        }
        else
        {
            EXPECT_EQ(node->get_provenance_tags(), (ProvSet{"TK"}));
        }
    }
    tk->set_k(5);
    auto tkc1 = tk->input_value(1).get_node_shared_ptr();
    tkc1->add_provenance_tag("TKC1");
    for (auto node : topological_sort(NodeVector{tk}))
    {
        if (node == p1)
        {
            EXPECT_EQ(node->get_provenance_tags(), (ProvSet{"P1"}));
        }
        else if (node == tkc1)
        {
            EXPECT_EQ(node->get_provenance_tags(), (ProvSet{"TK", "TKC0", "TKC1"}));
        }
        else
        {
            EXPECT_EQ(node->get_provenance_tags(), (ProvSet{"TK"}));
        }
    }
}

TEST(provenance, empty_group)
{
    auto p1 = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});
    p1->add_provenance_tag("P1");
    auto abs = make_shared<op::Abs>(p1);
    // Make sure group is empty
    abs->add_provenance_group_members_above({abs});
    abs->add_provenance_tag("abs");
    for (auto node : topological_sort(NodeVector{abs}))
    {
        if (node == p1)
        {
            EXPECT_EQ(node->get_provenance_tags(), (ProvSet{"P1"}));
        }
        else
        {
            EXPECT_EQ(node->get_provenance_tags(), (ProvSet{"abs"}));
        }
    }
}

TEST(provenance, scaled_quantize_concat_unsigned)
{
    ngraph::Shape shape_a{2, 2};
    auto A = make_shared<ngraph::op::Parameter>(ngraph::element::u8, shape_a);
    auto An = make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape{1});
    auto Ax = make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape{1});
    A->add_provenance_tag("in0");
    An->add_provenance_tag("in1");
    Ax->add_provenance_tag("in2");
    ngraph::Shape shape_r{2, 2};
    auto QConcat = ngraph::builder::QuantizedConcatBuilder(
        ngraph::NodeVector{A}, 0, ngraph::NodeVector{An}, ngraph::NodeVector{Ax});
    auto f = make_shared<ngraph::Function>(ngraph::NodeVector{QConcat},
                                           ngraph::ParameterVector{A, An, Ax});
    QConcat->add_provenance_tag("hello");
    auto check_if_result = [](shared_ptr<Node> n) {
        // Pointer will cast to nullptr if this node is not a Result
        auto ng_node = dynamic_pointer_cast<op::Result>(n);
        bool is_result = (ng_node != nullptr);
        return is_result;
    };

    for (auto n : f->get_ordered_ops())
    {
        if (!check_if_result(n))
        {
            ASSERT_EQ(n->get_provenance_tags().size(), 1);
        }
    }
}

TEST(provenance, opset1_upgrade_pass_topk)
{
    test::ProvenanceEnabler provenance_enabler;

    const size_t axis = 2;
    const size_t k = 10;
    const auto data = make_shared<op::Parameter>(element::i32, Shape{5, 10, 15});

    const auto topk_v0 = make_shared<op::v0::TopK>(data, axis, element::i32, k);
    const auto result = make_shared<op::Result>(topk_v0->output(0));
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{data});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    const auto pass_replacement_node = f->get_result()->get_input_node_shared_ptr(0);
    const auto topk_v1 = as_type_ptr<op::v1::TopK>(pass_replacement_node);

    const std::string tag = "<Opset1_Upgrade (v0 TopK)>";
    auto tag_check = [&tag](std::shared_ptr<ngraph::Node> node) {
        auto tags = node->get_provenance_tags();
        EXPECT_TRUE(tags.find(tag) != tags.end());
    };
    traverse_nodes({topk_v1}, tag_check, as_node_vector(topk_v0->input_values()));
}

TEST(provenance, opset0_downgrade_pass_topk)
{
    test::ProvenanceEnabler provenance_enabler;

    const auto data = make_shared<op::Parameter>(element::i32, Shape{5, 10, 15});
    const int32_t k = 10;
    const auto k_node = op::Constant::create(element::i64, Shape{}, {k});
    const size_t axis = 2;
    const auto mode = op::v1::TopK::Mode::MAX;
    const auto sort = op::v1::TopK::SortType::SORT_INDICES;
    const auto elem_type = element::i64;

    const auto topk_v1 = make_shared<op::v1::TopK>(data, k_node, axis, mode, sort, elem_type);
    const auto result = make_shared<op::Result>(topk_v1->output(0));
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{data});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    const auto pass_replacement_node = f->get_result()->get_input_node_shared_ptr(0);
    const auto topk_v0 = as_type_ptr<op::v0::TopK>(pass_replacement_node);

    const std::string tag = "<Opset0_Downgrade (v1 TopK)>";
    auto tag_check = [&tag](std::shared_ptr<ngraph::Node> node) {
        auto tags = node->get_provenance_tags();
        EXPECT_TRUE(tags.find(tag) != tags.end());
    };
    traverse_nodes({topk_v0}, tag_check, as_node_vector(topk_v1->input_values()));
}

TEST(provenance, opset1_upgrade_pass_graph)
{
    test::ProvenanceEnabler provenance_enabler;

    auto x = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});
    auto y = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});

    auto a = make_shared<op::v0::Add>(x, y);
    auto b = make_shared<op::v0::Subtract>(x, y);
    auto c = make_shared<op::v0::Abs>(b);
    auto d = make_shared<op::v0::Multiply>(a, b);

    auto f = make_shared<Function>(d, ParameterVector{x, y});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    for (auto node : f->get_ordered_ops())
    {
        auto tags = node->get_provenance_tags();
        if (as_type_ptr<op::v1::Add>(node))
        {
            EXPECT_EQ(tags.size(), 1);
            EXPECT_TRUE(tags.find("<Opset1_Upgrade (v0 Add)>") != tags.end());
        }
        else if (as_type_ptr<op::v1::Multiply>(node))
        {
            EXPECT_EQ(tags.size(), 1);
            EXPECT_TRUE(tags.find("<Opset1_Upgrade (v0 Multiply)>") != tags.end());
        }
        else if (as_type_ptr<op::v1::Subtract>(node))
        {
            EXPECT_EQ(tags.size(), 1);
            EXPECT_TRUE(tags.find("<Opset1_Upgrade (v0 Subtract)>") != tags.end());
        }
        else if (as_type_ptr<op::v0::Abs>(node))
        {
            EXPECT_TRUE(tags.empty());
        }
    }
}

TEST(provenance, opset0_downgrade_pass_graph)
{
    test::ProvenanceEnabler provenance_enabler;

    auto x = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});
    auto y = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});

    auto a = make_shared<op::v1::Add>(x, y);
    auto b = make_shared<op::v1::Subtract>(x, y);
    auto c = make_shared<op::v0::Abs>(b);
    auto d = make_shared<op::v1::Multiply>(a, b);

    auto f = make_shared<Function>(d, ParameterVector{x, y});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    for (auto node : f->get_ordered_ops())
    {
        auto tags = node->get_provenance_tags();
        if (as_type_ptr<op::v0::Add>(node))
        {
            EXPECT_EQ(tags.size(), 1);
            EXPECT_TRUE(tags.find("<Opset0_Downgrade (v1 Add)>") != tags.end());
        }
        else if (as_type_ptr<op::v0::Multiply>(node))
        {
            EXPECT_EQ(tags.size(), 1);
            EXPECT_TRUE(tags.find("<Opset0_Downgrade (v1 Multiply)>") != tags.end());
        }
        else if (as_type_ptr<op::v0::Subtract>(node))
        {
            EXPECT_EQ(tags.size(), 1);
            EXPECT_TRUE(tags.find("<Opset0_Downgrade (v1 Subtract)>") != tags.end());
        }
        else if (as_type_ptr<op::v0::Abs>(node))
        {
            EXPECT_TRUE(tags.empty());
        }
    }
}
