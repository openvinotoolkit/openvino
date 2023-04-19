// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/linear_ir.hpp"

#include <snippets/itt.hpp>

#include "snippets/lowered/loop_manager.hpp"
#include <snippets/op/serialization_node.hpp>
#include "snippets/tensor_descriptor.hpp"
#include "snippets/utils.hpp"

#include <openvino/core/graph_util.hpp>
#include <openvino/core/type.hpp>

namespace ngraph {
namespace snippets {
namespace lowered {

LinearIR::LinearIR(const std::shared_ptr<ov::Model>& model, Config config)
        : m_io_lowered_ops{}, m_config{std::move(config)}, m_loop_manager(std::make_shared<LoopManager>()) {
    constExprIt scalar_pos = m_lowered_ops.begin();
    ExpressionPtr last_param = nullptr;
    for (const auto& n : get_ordered_ops(model)) {
        constExprIt insertion_pos = m_lowered_ops.end();
        std::shared_ptr<Expression> expr;
        std::vector<TensorDescriptorPtr> input_tds;
        for (const auto& in : n->inputs()) {
            const auto& out = in.get_source_output();
            const auto& parent_out_tds = m_node2expression_map[out.get_node_shared_ptr()]->get_outputs();
            input_tds.push_back(parent_out_tds[out.get_index()]);
        }
        if (const auto& par = as_type_ptr<opset1::Parameter>(n)) {
            auto io_expr = std::make_shared<IOExpression>(par, model->get_parameter_index(par));
            m_io_lowered_ops.push_back(io_expr);
            expr = io_expr;
            last_param = expr;
        } else if (const auto& res = as_type_ptr<opset1::Result>(n)) {
            auto io_expr = std::make_shared<IOExpression>(res, model->get_result_index(res), input_tds);
            m_io_lowered_ops.push_back(io_expr);
            expr = io_expr;
        } else {
            if (const auto& scalar = as_type_ptr<op::Scalar>(n)) {
                // Scalar should be on the Linear IR beginning after Parameters to have valid expression order after Loop passes.
                // After these passes we must call pass MoveScalarToConsumer() to have a correct accuracy.
                // For more details, please see the pass description
                if (scalar_pos == m_lowered_ops.end()) {
                    OPENVINO_ASSERT(last_param, "Scalars must be executed after Parameters");
                    scalar_pos = std::find(m_lowered_ops.begin(), m_lowered_ops.end(), last_param);
                }
                insertion_pos = std::next(scalar_pos);
            }
            // Note that output tds must be empty since they are filled automatically from rt_info and/or tensor shapes
            expr = std::make_shared<Expression>(n, input_tds);
        }
        register_expression(expr);
        m_lowered_ops.insert(insertion_pos, expr);
    }
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
    auto first_node = std::make_shared<opset1::Parameter>(element::f32, Shape{});
    first_node->set_friendly_name("Start");
    first_node->get_rt_info()["execTimeMcs"] = 0;
    std::shared_ptr<Node> body_node = first_node;
    for (const auto& expr : m_lowered_ops) {
        body_node = std::make_shared<op::SerializationNode>(body_node, expr);
    }
    auto last_node = std::make_shared<opset1::Result>(body_node);
    last_node->set_friendly_name("End");
    const auto tmp_model = std::make_shared<ov::Model>(ResultVector {last_node},
                                                       ParameterVector {first_node},
                                                       "Lowered_IR_Serialization");
    ov::pass::Serialize(xml, bin).run_on_model(tmp_model);
}

LinearIR::container LinearIR::deep_copy_range(LinearIR::container::const_iterator begin, LinearIR::container::const_iterator end) {
    LinearIR::container result;
    NodeVector original_nodes;
    for (auto it = begin; it != end; it++)
        original_nodes.push_back((*it)->get_node());
    NodeMap node_map;
    ngraph::clone_nodes(original_nodes,  node_map);
    for (auto it = begin; it != end; it++) {
        // copy by value, so result shared_pointer point to new objects
        Expression new_expr = **it;
        new_expr.m_source_node = node_map[(*it)->get_node().get()];
        result.emplace_back(std::make_shared<Expression>(new_expr));
    }
    return result;
}

LinearIR LinearIR::deep_copy() const {
    LinearIR result;
    auto& result_ops = result.m_lowered_ops;
    for (const auto& expr : deep_copy_range(m_lowered_ops.begin(), m_lowered_ops.end()))
        result_ops.emplace_back(expr);
    result.m_config = m_config;
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
    std::map<TensorDescriptorPtr, int> td2int;
    int td_counter = 0;
    int counter = 0;
    for (const auto& expr : m_lowered_ops) {
        const auto& node = expr->get_node();
        std::cerr << counter++ << " : " <<
                  node->get_friendly_name() << " :  ";
        if (tds_as_pointers) {
            for (const auto& in : expr->get_inputs()) {
                if (td2int.count(in) == 0)
                    OPENVINO_THROW("Undefined input descriptor for op");
                std::cerr << td2int.at(in) << ", ";
            }
            std::cerr << "\b\b => ";
            for (const auto& out : expr->get_outputs()) {
                if (td2int.count(out) == 0)
                    td2int.insert({out, td_counter++});
                std::cerr << td2int.at(out) << ", ";
            }
        } else {
            for (const auto& in : expr->get_inputs())
                std::cerr << *in << ", ";
            std::cerr << "\b\b => ";
            for (const auto& out : expr->get_outputs())
                std::cerr << *out << ", ";
        }
        std::cerr << "\b\b";
        const auto& rinfo = expr->get_reg_info();
        if (!rinfo.first.empty() || !rinfo.second.empty())
            print_rinfo(expr->get_reg_info());
        std::cerr << "\n";
    }
}

void LinearIR::init_emitters(const std::shared_ptr<TargetMachine>& target) {
    for (auto& expr : m_lowered_ops) {
        if (!expr->get_emitter())
            expr->init_emitter(target);
    }
}

ExpressionPtr LinearIR::get_expr_by_node(const std::shared_ptr<Node>& n) const {
    auto found = m_node2expression_map.find(n);
    return found == m_node2expression_map.end() ? nullptr : found->second;
}

ExpressionPort LinearIR::get_expr_by_output(const TensorDescriptorPtr& td) const {
    auto found = m_output2expression_map.find(td);
    if (found == m_output2expression_map.end())
        OPENVINO_THROW("Failed to find expression by output tensor descriptor");
    return found->second;
}

const std::set<ExpressionPort>& LinearIR::get_exprs_by_input(const TensorDescriptorPtr& td) const {
    auto found = m_input2expression_map.find(td);
    if (found == m_input2expression_map.end())
        OPENVINO_THROW("Failed to find expression by input tensor descriptor");
    return found->second;
}

void LinearIR::replace_input(const ExpressionPtr& expr, size_t port, const TensorDescriptorPtr& to) {
    replace_input(expr->input_port(port), to);
}

void LinearIR::replace_input(const ExpressionPort& expr_port, const TensorDescriptorPtr& to) {
    const auto& expr = expr_port.expr;
    const auto port = expr_port.port;
    OPENVINO_ASSERT(expr_port.get_type() == ExpressionPort::Type::Input, "Failed to replace: target input port must have Input type");
    OPENVINO_ASSERT(port < expr->m_inputs.size(), "Failed to replace: target input port must be less than input count!");
    const auto from = expr->m_inputs[port];
    auto found = m_input2expression_map.find(from);
    if (found == m_input2expression_map.end() || found->second.count(expr_port) == 0)
        OPENVINO_THROW("Invalid expression of input was provided to replace_input");
    found->second.erase(expr_port);
    {
        const auto& res = m_input2expression_map.insert({to, std::set<ExpressionPort>{expr_port}});
        // If input is already in the map => add ExprPtr to the mapped set
        if (!res.second) {
            res.first->second.insert(expr_port);
        }
    }
    expr->replace_input(port, std::move(to));
}

void LinearIR::replace_output(const ExpressionPtr& expr, size_t port, const TensorDescriptorPtr& to) {
    replace_output(expr->output_port(port), to);
}

void LinearIR::replace_output(const ExpressionPort& expr_port, const TensorDescriptorPtr& to) {
    const auto& expr = expr_port.expr;
    const auto port = expr_port.port;
    OPENVINO_ASSERT(expr_port.get_type() == ExpressionPort::Type::Output, "Failed to replace: target output port must have Output type");
    OPENVINO_ASSERT(port < expr->m_outputs.size(), "Failed to replace: target output port must be less than output count!");
    const auto from = expr->m_outputs[port];
    auto found = m_output2expression_map.find(from);
    if (found == m_output2expression_map.end() || found->second != expr_port)
        OPENVINO_THROW("Invalid expression of output was provided to replace_output");
    m_output2expression_map.erase(found);
    m_output2expression_map[to] = expr_port;
    expr->replace_output(port, to);
}

void LinearIR::register_regular_expression(const ExpressionPtr& expr) {
    if (is_type<ov::op::v0::Result>(expr->get_node()) || is_type<ov::op::v0::Parameter>(expr->get_node()))
        OPENVINO_THROW("LinearIR::insert can't be used to add Parameters or Results to IR");
    register_expression(expr);
}

void LinearIR::register_expression(const ExpressionPtr& expr) {
    const auto& node = expr->get_node();
    {
        const auto& res = m_node2expression_map.insert({node, expr});
        if (!res.second)
            OPENVINO_THROW("Duplicate node is detected in linear IR: " + std::string(node->get_friendly_name()));
    }
    for (size_t i = 0; i < expr->m_outputs.size(); ++i) {
        const auto& out = expr->m_outputs[i];
        m_output2expression_map[out] = expr->output_port(i);
    }

    for (size_t i = 0; i < expr->m_inputs.size(); ++i) {
        const auto& in = expr->m_inputs[i];
        const auto expr_port = expr->input_port(i);
        const auto& res = m_input2expression_map.insert({in, std::set<ExpressionPort>{expr_port}});
        // If input is already in the map => add ExprPtr to the mapped set
        if (!res.second) {
            res.first->second.insert(expr_port);
        }
    }
}

void LinearIR::unregister_expression(const ExpressionPtr& expr) {
    for (const auto& out : expr->m_outputs)
        m_output2expression_map.erase(out);

    size_t in_port = 0;
    for (const auto& in : expr->m_inputs) {
        const auto& found = m_input2expression_map.find(in);
        if (found != m_input2expression_map.end()) {
            // Note: If the input is used by only by this expr => delete the whole entry
            //       Otherwise delete the expr from the users set
            auto& users = found->second;
            if (users.size() == 1)
                m_input2expression_map.erase(found);
            else
                users.erase(expr->input_port(in_port));
        }
        ++in_port;
    }

    m_node2expression_map.erase(expr->get_node());
}

LinearIR::exprIt LinearIR::insert(constExprIt pos, container::value_type&& value) {
    register_regular_expression(value);
    return m_lowered_ops.insert(pos, value);
}

LinearIR::exprIt LinearIR::insert(constExprIt pos, const container::value_type& value) {
    register_regular_expression(value);
    return m_lowered_ops.insert(pos, value);
}

LinearIR::exprIt LinearIR::insert(constExprIt pos, exprIt begin, exprIt end) {
    constExprIt cbegin = begin;
    constExprIt cend = end;
    return insert(pos, cbegin, cend);
}

LinearIR::exprIt LinearIR::insert(constExprIt pos, constExprIt begin, constExprIt end) {
    for (auto b = begin; b != end; b++)
        register_regular_expression(*b);
    return m_lowered_ops.insert(pos, begin, end);
}

LinearIR::exprIt LinearIR::insert(LinearIR::constExprIt pos, const NodeVector& nodes) {
    auto ret = m_lowered_ops.end();
    for (const auto& n : nodes) {
        std::vector<TensorDescriptorPtr> input_tds;
        for (const auto& in : n->inputs()) {
            const auto& out = in.get_source_output();
            const auto& parent_out_tds = m_node2expression_map[out.get_node_shared_ptr()]->get_outputs();
            input_tds.push_back(parent_out_tds[out.get_index()]);
        }
        // Note that output tds must be empty since they are filled automatically from rt_info and/or tensor shapes
        const auto& expr = std::make_shared<Expression>(n, input_tds);
        register_regular_expression(expr);
        ret = m_lowered_ops.insert(pos, expr);
    }
    // Need to return iterator to the first of the inserted values
    return std::prev(ret, static_cast<int64_t>(nodes.size()));
}

LinearIR::exprIt LinearIR::insert(LinearIR::constExprIt pos, const std::shared_ptr<Node>& n) {
    std::vector<TensorDescriptorPtr> input_tds;
    for (const auto& in : n->inputs()) {
        const auto& out = in.get_source_output();
        const auto& parent_out_tds = m_node2expression_map[out.get_node_shared_ptr()]->get_outputs();
        input_tds.push_back(parent_out_tds[out.get_index()]);
    }
    // Note that output tds must be empty since they are filled automatically from rt_info and/or tensor shapes
    const auto& expr = std::make_shared<Expression>(n, input_tds);
    register_regular_expression(expr);
    return m_lowered_ops.insert(pos, expr);
}

LinearIR::exprIt LinearIR::erase(LinearIR::exprIt pos) {
    unregister_expression(*pos);
    return m_lowered_ops.erase(pos);
}

LinearIR::exprIt LinearIR::erase(LinearIR::constExprIt pos) {
    unregister_expression(*pos);
    return m_lowered_ops.erase(pos);
}

void LinearIR::move(LinearIR::constExprIt from, LinearIR::constExprIt to) {
    // Instead of `insert()` + `erase()`, we use `splice()` for the same list
    m_lowered_ops.splice(to, m_lowered_ops, from);
}

}// namespace lowered
}// namespace snippets
}// namespace ngraph
