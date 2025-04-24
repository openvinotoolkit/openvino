// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <list>
#include <memory>

#include "common_test_utils/all_close.hpp"
#include "common_test_utils/ndarray.hpp"
#include "common_test_utils/test_tools.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/op.hpp"
#include "openvino/pass/pattern/matcher.hpp"

using namespace ov;
using namespace std;

class ControlDependencyOp : public ov::op::Op {
public:
    OPENVINO_OP("ControlDependencyOp");
    virtual std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override {
        auto clone = make_shared<ControlDependencyOp>(new_args, std::set<std::shared_ptr<Node>>{});
        return std::move(clone);
    }

    ControlDependencyOp(const OutputVector& args, const std::set<std::shared_ptr<Node>>& deps) : Op(args) {
        if (args.size() == 0 && deps.size() == 0) {
            OPENVINO_THROW("Expected some arguments or dependencies");
        }

        for (auto& node : deps) {
            add_control_dependency(node);
        }

        if (args.size() != 0) {
            set_output_type(0, args.at(0).get_element_type(), args.at(0).get_shape());
        } else {
            auto dn = *(deps.begin());
            set_output_type(0, dn->get_element_type(), dn->get_shape());
        }
    }
};

static ::testing::AssertionResult test_ordered_ops(const std::shared_ptr<ov::Model>& m,
                                                   const ov::NodeVector& required_ops) {
    std::unordered_set<ov::Node*> seen;
    for (auto& node_ptr : m->get_ordered_ops()) {
        ov::Node* node = node_ptr.get();
        if (seen.count(node) > 0) {
            return ::testing::AssertionFailure() << "Duplication in ordered ops";
        }
        size_t arg_count = node->get_input_size();
        for (size_t i = 0; i < arg_count; ++i) {
            ov::Node* dep = node->get_input_node_ptr(i);
            if (seen.count(dep) == 0) {
                return ::testing::AssertionFailure() << "Argument " << *dep << " does not occur before op" << *node;
            }
        }
        for (auto& dep_ptr : node->get_control_dependencies()) {
            if (seen.count(dep_ptr.get()) == 0) {
                return ::testing::AssertionFailure()
                       << "Control dependency " << *dep_ptr << " does not occur before op" << *node;
            }
        }
        seen.insert(node);
    }
    for (auto& node_ptr : required_ops) {
        if (seen.count(node_ptr.get()) == 0) {
            return ::testing::AssertionFailure() << "Required op " << *node_ptr << "does not occur in ordered ops";
        }
    }
    return ::testing::AssertionSuccess();
}

TEST(control_dependencies, cdep_ops) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto absn = make_shared<op::v0::Abs>(A);
    auto cdop = make_shared<ControlDependencyOp>(OutputVector{A}, std::set<std::shared_ptr<Node>>{absn});

    auto f = make_shared<Model>(cdop, ParameterVector{A, B});
    test_ordered_ops(f, NodeVector{absn});
}

TEST(control_dependencies, two_cdep_ops) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto absn = make_shared<op::v0::Abs>(A);
    auto C = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto absn_c = make_shared<op::v0::Abs>(C);
    auto cdop = make_shared<ControlDependencyOp>(OutputVector{A}, std::set<std::shared_ptr<Node>>{absn, absn_c});

    auto f = make_shared<Model>(cdop, ParameterVector{A, B, C});
    test_ordered_ops(f, NodeVector{absn, absn_c});
}

TEST(control_dependencies, two_cdep_ops_op_on_top) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto absn = make_shared<op::v0::Abs>(A);
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto absn_b = make_shared<ov::op::v0::Abs>(B);
    auto cdop = make_shared<ControlDependencyOp>(OutputVector{A}, std::set<std::shared_ptr<Node>>{absn, absn_b});
    auto absn_cdop = make_shared<ov::op::v0::Abs>(cdop);

    auto f = make_shared<Model>(absn_cdop, ParameterVector{A, B});
    test_ordered_ops(f, NodeVector{absn, absn_b});
}

TEST(control_dependencies, clone_function_cdop) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto absn = make_shared<ov::op::v0::Abs>(A);
    auto cdop = make_shared<ControlDependencyOp>(OutputVector{A}, std::set<std::shared_ptr<Node>>{absn});

    auto f = make_shared<Model>(cdop, ParameterVector{A});
    test_ordered_ops(f, NodeVector{absn});
    auto clone = f->clone();
    auto matcher = std::make_shared<pass::pattern::Matcher>(cdop);
    auto cdop_clone = clone->get_results().at(0)->input_value(0).get_node_shared_ptr();
    ASSERT_TRUE(matcher->match(cdop_clone));
    auto cloned_deps = cdop_clone->get_control_dependencies();
    ASSERT_EQ(cloned_deps.size(), 1);
    auto cloned_abs = *begin(cloned_deps);
    ASSERT_TRUE(is_type<ov::op::v0::Abs>(cloned_abs));
}

TEST(control_dependencies, clone_function_cdop_abs) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto absn = make_shared<ov::op::v0::Abs>(A);
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto absn_b = make_shared<ov::op::v0::Abs>(B);
    auto cdop = make_shared<ControlDependencyOp>(OutputVector{A}, std::set<std::shared_ptr<Node>>{absn, absn_b});
    auto absn_cdop = make_shared<ov::op::v0::Abs>(cdop);

    auto f = make_shared<Model>(absn_cdop, ParameterVector{A, B});
    auto clone = f->clone();
    auto matcher = std::make_shared<pass::pattern::Matcher>(cdop);
    auto cdop_clone =
        clone->get_results().at(0)->input_value(0).get_node_shared_ptr()->input_value(0).get_node_shared_ptr();
    ASSERT_TRUE(matcher->match(cdop_clone));
    auto cloned_deps = cdop_clone->get_control_dependencies();
    ASSERT_EQ(cloned_deps.size(), 2);
    for (auto ccdep : cloned_deps) {
        ASSERT_TRUE(is_type<ov::op::v0::Abs>(ccdep));
    }
}

static size_t count_control_dependencies(const shared_ptr<Node>& node, const shared_ptr<Node>& dependency) {
    auto& dependencies = node->get_control_dependencies();
    return count(dependencies.begin(), dependencies.end(), dependency);
}

TEST(control_dependencies, replace_node) {
    Shape shape{2, 2};
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, shape);
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, shape);
    auto MUL_AB = make_shared<op::v1::Multiply>(A, B);
    auto MUL_BA = make_shared<op::v1::Multiply>(B, A);
    auto ADD = make_shared<op::v1::Add>(A, B);
    auto SUM = make_shared<op::v1::Add>(MUL_AB, ADD);
    ADD->add_control_dependency(MUL_AB);
    ASSERT_TRUE(1 == count_control_dependencies(ADD, MUL_AB));
    ASSERT_TRUE(0 == count_control_dependencies(ADD, MUL_BA));
    ov::replace_node(MUL_AB, MUL_BA);
    ASSERT_TRUE(0 == count_control_dependencies(ADD, MUL_AB));
    ASSERT_TRUE(1 == count_control_dependencies(ADD, MUL_BA));
}
