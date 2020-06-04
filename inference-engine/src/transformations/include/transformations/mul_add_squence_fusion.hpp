// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include <transformations_visibility.hpp>

#include <ngraph/ngraph.hpp>

#include "ngraph/pattern/matcher.hpp"

#include <ngraph/opsets/opset1.hpp>

#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations/utils/annotations.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API MulAddFusion;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::MulAddFusion: public ngraph::pass::GraphRewrite {
public:
    MulAddFusion() : GraphRewrite() {
        mul_add_fusion<opset1::Multiply>();
        mul_add_fusion<opset1::Add>();
    }

private:
    template <class T>
    void mul_add_fusion();
};


template <class A, class B>
std::pair<std::shared_ptr<A>, std::shared_ptr<B>> parse_eltwise_inputs(std::shared_ptr<ngraph::Node> node) {
    auto eltwise = std::dynamic_pointer_cast<A>(node->input(0).get_source_output().get_node_shared_ptr());
    auto constant = std::dynamic_pointer_cast<B>(node->input(1).get_source_output().get_node_shared_ptr());

    if (!eltwise) {
        eltwise = std::dynamic_pointer_cast<A>(node->input(1).get_source_output().get_node_shared_ptr());
        constant = std::dynamic_pointer_cast<B>(node->input(0).get_source_output().get_node_shared_ptr());
    }

    if (!eltwise || !constant) {
        return {nullptr, nullptr};
    }

    return {eltwise, constant};
}

template <class T>
bool fusion(std::shared_ptr<T> m_eltwise) {
    using namespace ngraph;

    auto m_attrs = op::util::EltwiseAttrs::get_op_attrs(std::static_pointer_cast<op::Op>(m_eltwise));
    if (!m_attrs || !m_attrs->can_be_fused()) {
        return false;
    }

    std::shared_ptr<op::Op> eltwise, add, mul;
    std::shared_ptr<Node> constant, constant1, constant2;
    std::tie(add, constant1) = parse_eltwise_inputs<opset1::Add, Node>(m_eltwise);
    std::tie(mul, constant2) = parse_eltwise_inputs<opset1::Multiply, Node>(m_eltwise);

    if (add && add->output(0).get_target_inputs().size() != 1) {
        return false;
    }

    if (mul && mul->output(0).get_target_inputs().size() != 1) {
        return false;
    }

    if (add || mul) {
        std::tie(eltwise, constant) = (add ? std::make_tuple(add, constant1) : std::make_tuple(mul, constant2));
        auto res = parse_eltwise_inputs<Node, Node>(eltwise);

        auto attrs = op::util::EltwiseAttrs::get_op_attrs(eltwise);
        if (!attrs || !attrs->can_be_fused()) {
            return false;
        }

        // res.first should be data input and res.second should be constant
        if (attrs->get_const_input_id() == 0) {
            swap(res.first, res.second);
        }

        // Mul->Mul => Mul, Add->Add => Add
        if (std::dynamic_pointer_cast<T>(eltwise) && std::dynamic_pointer_cast<T>(m_eltwise)) {
            auto new_const = std::make_shared<T>(constant, res.second);
            auto new_eltwise = std::make_shared<T>(res.first, new_const);

            copy_runtime_info(m_eltwise, {new_const, new_eltwise});
            replace_node(m_eltwise, new_eltwise);
            new_eltwise->set_op_annotations(std::make_shared<op::util::EltwiseAttrs>(m_attrs));
            new_eltwise->set_friendly_name(m_eltwise->get_friendly_name());
            return true;
        }

        // Add->Mul => Mul->Add
        if (std::dynamic_pointer_cast<opset1::Add>(eltwise) && std::dynamic_pointer_cast<opset1::Multiply>(m_eltwise)) {
            auto new_mul = std::make_shared<opset1::Multiply>(res.first, constant);
            auto new_const = std::make_shared<opset1::Multiply>(constant, res.second);
            auto new_add = std::make_shared<opset1::Add> (new_mul, new_const);

            copy_runtime_info(m_eltwise, {new_mul, new_const, new_add});
            replace_node(m_eltwise, new_add);

            // We need to preserve op annotations and namings
            new_mul->set_op_annotations(std::make_shared<op::util::EltwiseAttrs>(attrs));
            new_add->set_op_annotations(std::make_shared<op::util::EltwiseAttrs>(m_attrs));
            new_add->set_friendly_name(m_eltwise->get_friendly_name());
            fusion(new_mul);
            return true;
        }
    }

    return false;
}

template<class T>
void ngraph::pass::MulAddFusion::mul_add_fusion() {
    auto input1 = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto input2 = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto eltwise = std::make_shared<T>(input1, input2);

    ngraph::graph_rewrite_callback callback = [&](ngraph::pattern::Matcher &m) {
        static_assert(std::is_same<T, opset1::Add>() || std::is_same<T, opset1::Multiply>(),
                      "Unsupported template parameter. Only Add or Multiply allowed!");

        if (auto m_eltwise = std::dynamic_pointer_cast<T>(m.get_match_root())) {
            return fusion(m_eltwise);
        }

        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(eltwise, "MulAddFusion");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
