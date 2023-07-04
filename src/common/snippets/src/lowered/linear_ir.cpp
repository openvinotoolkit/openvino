// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/linear_ir.hpp"

#include "snippets/itt.hpp"

#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/expression_factory.hpp"
#include "snippets/op/serialization_node.hpp"
#include "snippets/utils.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/type.hpp"

namespace ov {
namespace snippets {
namespace lowered {

LinearIR::LinearIR(const std::shared_ptr<ov::Model>& model, Config config)
        : m_io_expressions{}, m_config{std::move(config)}, m_loop_manager(std::make_shared<LoopManager>()) {
    constExprIt last_param = m_expressions.end();
    for (const auto& n : get_ordered_ops(model)) {
        constExprIt insertion_pos = m_expressions.end();
        const auto expr = create_expression(n, model);

        // Scalar should be on the Linear IR beginning after Parameters to have valid expression order after Loop passes.
        // After these passes we must call pass MoveScalarToConsumer() to have a correct accuracy.
        // For more details, please see the pass description
        if (const auto& scalar = as_type_ptr<op::Scalar>(n)) {
            insertion_pos = std::next(last_param);
        }

        register_expression(expr, true);
        const auto& it = m_expressions.insert(insertion_pos, expr);

        if (const auto io_expr = std::dynamic_pointer_cast<IOExpression>(expr)) {
            m_io_expressions.push_back(io_expr);
            if (ov::is_type<ov::op::v0::Parameter>(n))
                last_param = it;
        }
    }
}

ExpressionPtr LinearIR::create_expression(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model) {
    return ExpressionFactory::build(n, *this, model);
}

ExpressionPtr LinearIR::create_expression(const std::shared_ptr<Node>& n, const std::vector<PortConnectorPtr>& inputs) {
    return ExpressionFactory::build(n, inputs);
}

ov::NodeVector LinearIR::get_ordered_ops(const std::shared_ptr<ov::Model>& m) {
    if (!m->get_sinks().empty())
        OPENVINO_THROW("Linear IR is not supposed to work for model with sinks. Check your transformation pipeline.");

    // Note that an important difference between this impl and Model::get_ordered_ops is that Results and Parameters
    // are added in REVERSE order, so they will be visited in DIRECT order compared to get_parameters() and get_results()
    NodeVector nodes;
    const auto& results = m->get_results();
    std::copy(results.rbegin(), results.rend(), std::back_inserter(nodes));
    const auto& params = m->get_parameters();
    std::copy(params.rbegin(), params.rend(), std::back_inserter(nodes));

    return ov::topological_sort(nodes);
}

void LinearIR::serialize(const std::string& xml, const std::string& bin) {
    auto first_node = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    first_node->set_friendly_name("Start");
    first_node->get_rt_info()["execTimeMcs"] = 0;
    std::shared_ptr<Node> body_node = first_node;
    for (const auto& expr : m_expressions) {
        body_node = std::make_shared<op::SerializationNode>(body_node, expr);
    }
    auto last_node = std::make_shared<ov::op::v0::Result>(body_node);
    last_node->set_friendly_name("End");
    const auto tmp_model = std::make_shared<ov::Model>(ResultVector {last_node},
                                                       ParameterVector {first_node},
                                                       "Lowered_IR_Serialization");
    ov::pass::Serialize(xml, bin).run_on_model(tmp_model);
}

LinearIR::container LinearIR::deep_copy_range(LinearIR::container::const_iterator begin, LinearIR::container::const_iterator end) {
    auto deep_clone_ports = [](std::vector<PortDescriptorPtr>& ports) {
        for (auto& port : ports) { port = port->clone(); }
    };
    LinearIR::container result;
    NodeVector original_nodes;
    for (auto it = begin; it != end; it++)
        original_nodes.push_back((*it)->get_node());
    ngraph::NodeMap node_map;
    OPENVINO_SUPPRESS_DEPRECATED_START
    ngraph::clone_nodes(original_nodes,  node_map);
    OPENVINO_SUPPRESS_DEPRECATED_END
    for (auto it = begin; it != end; it++) {
        // copy by value, so result shared_pointer point to new objects
        Expression new_expr = **it;
        new_expr.m_source_node = node_map[(*it)->get_node().get()];
        deep_clone_ports(new_expr.m_input_port_descriptors);
        deep_clone_ports(new_expr.m_output_port_descriptors);
        result.emplace_back(std::make_shared<Expression>(new_expr));
    }
    return result;
}

void LinearIR::debug_print(bool tds_as_pointers) const {
    auto print_rinfo = [](const RegInfo& rinfo) {
        std::cerr << " : {";
        for (auto i : rinfo.first)
            std::cerr << i << " ";
        std::cerr << " => ";
        for (auto i : rinfo.second)
            std::cerr << i << " ";
        std::cerr << "}";
    };
    std::map<PortConnectorPtr, int> td2int;
    int td_counter = 0;
    int counter = 0;
    for (const auto& expr : m_expressions) {
        const auto& node = expr->get_node();
        std::cerr << counter++ << " : " <<
                  node->get_friendly_name() << " :  ";
        if (tds_as_pointers) {
            for (const auto& in : expr->m_input_port_connectors) {
                if (td2int.count(in) == 0)
                    OPENVINO_THROW("Undefined input descriptor for op");
                std::cerr << td2int.at(in) << ", ";
            }
            std::cerr << "\b\b => ";
            for (const auto& out : expr->m_output_port_connectors) {
                if (td2int.count(out) == 0)
                    td2int.insert({out, td_counter++});
                std::cerr << td2int.at(out) << ", ";
            }
        } else {
            for (const auto& port_desc : expr->m_input_port_descriptors)
                std::cerr << port_desc << ", ";
            std::cerr << "\b\b => ";
            for (const auto& port_desc : expr->m_output_port_descriptors)
                std::cerr << port_desc << ", ";
        }
        std::cerr << "\b\b";
        const auto& rinfo = expr->get_reg_info();
        if (!rinfo.first.empty() || !rinfo.second.empty())
            print_rinfo(expr->get_reg_info());
        std::cerr << "\n";
    }
}

void LinearIR::init_emitters(const std::shared_ptr<TargetMachine>& target) {
    for (auto& expr : m_expressions) {
        if (!expr->get_emitter())
            expr->init_emitter(target);
    }
}

const ExpressionPtr& LinearIR::get_expr_by_node(const std::shared_ptr<Node>& n) const {
    auto found = m_node2expression_map.find(n);
    OPENVINO_ASSERT(found != m_node2expression_map.end(), "The node " + n->get_friendly_name() + " hasn't been found in Linear IR");
    return found->second;
}

void LinearIR::replace_input(const std::set<ExpressionPort>& consumers, const PortConnectorPtr& to) {
    for (const auto& consumer_input : consumers) {
        replace_input(consumer_input, to);
    }
}

void LinearIR::replace_input(const ExpressionPort& expr_port, const PortConnectorPtr& to) {
    const auto port = expr_port.get_index();
    const auto& expr = expr_port.get_expr();

    OPENVINO_ASSERT(expr_port.get_type() == ExpressionPort::Type::Input, "Failed to replace: target input port must have Input type");
    OPENVINO_ASSERT(expr_port.get_index() < expr->get_input_count(), "Failed to replace: target input port must be less than input count!");

    const auto& from = expr->get_input_port_connector(port);
    if (from == to)
        return;

    if (!to->found_consumer(expr_port)) {
        to->add_consumer(expr_port);
    }
    from->remove_consumer(expr_port);
    expr->replace_input(port, to);
}

void LinearIR::register_expression(const ExpressionPtr& expr, bool io_allowed) {
    const auto& node = expr->get_node();
    if (!io_allowed && (is_type<ov::op::v0::Result>(node) || is_type<ov::op::v0::Parameter>(node)))
        OPENVINO_THROW("LinearIR::insert can't be used to add Parameters or Results to IR");
    {
        const auto& res = m_node2expression_map.insert({node, expr});
        if (!res.second)
            OPENVINO_THROW("Duplicate node is detected in linear IR: " + std::string(node->get_friendly_name()));
    }
}

void LinearIR::unregister_expression(const ExpressionPtr& expr) {
    for (size_t i = 0; i < expr->get_input_count(); ++i) {
        const auto& input = expr->get_input_port_connector(i);
        input->remove_consumer(expr->get_input_port(i));
    }

    m_node2expression_map.erase(expr->get_node());
}

LinearIR::exprIt LinearIR::insert(constExprIt pos, container::value_type&& value) {
    register_expression(value);
    return m_expressions.insert(pos, value);
}

LinearIR::exprIt LinearIR::insert(constExprIt pos, const container::value_type& value) {
    register_expression(value);
    return m_expressions.insert(pos, value);
}

LinearIR::exprIt LinearIR::insert(constExprIt pos, exprIt begin, exprIt end) {
    constExprIt cbegin = begin;
    constExprIt cend = end;
    return insert(pos, cbegin, cend);
}

LinearIR::exprIt LinearIR::insert(constExprIt pos, constExprIt begin, constExprIt end) {
    for (auto b = begin; b != end; b++)
        register_expression(*b);
    return m_expressions.insert(pos, begin, end);
}

LinearIR::exprIt LinearIR::insert(LinearIR::constExprIt pos, const NodeVector& nodes) {
    auto ret = m_expressions.end();
    for (const auto& n : nodes) {
        const auto& expr = create_expression(n);
        register_expression(expr);
        ret = m_expressions.insert(pos, expr);
    }
    // Need to return iterator to the first of the inserted values
    return std::prev(ret, static_cast<int64_t>(nodes.size()));
}

LinearIR::exprIt LinearIR::insert(LinearIR::constExprIt pos, const std::shared_ptr<Node>& n) {
    const auto& expr = create_expression(n);
    register_expression(expr);
    return m_expressions.insert(pos, expr);
}

LinearIR::exprIt LinearIR::erase(LinearIR::exprIt pos) {
    unregister_expression(*pos);
    return m_expressions.erase(pos);
}

LinearIR::exprIt LinearIR::erase(LinearIR::constExprIt pos) {
    unregister_expression(*pos);
    return m_expressions.erase(pos);
}

void LinearIR::move(LinearIR::constExprIt from, LinearIR::constExprIt to) {
    // Instead of `insert()` + `erase()`, we use `splice()` for the same list
    m_expressions.splice(to, m_expressions, from);
}

LinearIR::constExprIt LinearIR::find(const ExpressionPtr& target) const {
    return find(cbegin(), cend(), target);
}
template<>
LinearIR::constExprIt LinearIR::find_before(LinearIR::constExprIt it, const ExpressionPtr& target) const {
    return find(cbegin(), it, target);
}
template<>
LinearIR::constExprReverseIt LinearIR::find_before(LinearIR::constExprReverseIt it, const ExpressionPtr& target) const {
    return find(crbegin(), it, target);
}
template<>
LinearIR::constExprIt LinearIR::find_after(LinearIR::constExprIt it, const ExpressionPtr& target) const {
    return find(it, cend(), target);
}
template<>
LinearIR::constExprReverseIt LinearIR::find_after(LinearIR::constExprReverseIt it, const ExpressionPtr& target) const {
    return find(it, crend(), target);
}


}// namespace lowered
}// namespace snippets
}// namespace ov
