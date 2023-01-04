// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered_expr.hpp"
#include "snippets/pass/assign_registers.hpp"
#include "snippets/pass/vector_to_scalar.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/op/subgraph.hpp"
#include <snippets/itt.hpp>
#include <snippets/op/serialization_node.hpp>
#include "snippets/tensor_descriptor.hpp"

#include <openvino/core/graph_util.hpp>
#include <openvino/core/type.hpp>

namespace ngraph {
namespace snippets {

LoweredExpr::LoweredExpr(const std::shared_ptr<Node>& n) : m_source_node{n}, m_emitter{nullptr}, m_reg_info{{}, {}} {
    for (const auto& in : n->inputs())
        m_inputs.emplace_back(get_tensor_descriptor_ptr(in.get_source_output()));
    for (const auto& out : n->outputs())
        m_outputs.emplace_back(get_tensor_descriptor_ptr(out));
}

LoweredExpr::LoweredExpr(const std::shared_ptr<Node>& n, std::vector<TensorDescriptorPtr> inputs, std::vector<TensorDescriptorPtr> outputs)
    : m_source_node{n}, m_emitter{nullptr}, m_inputs(std::move(inputs)), m_outputs(std::move(outputs)),  m_reg_info{{}, {}} {
    if (m_outputs.empty())
        for (const auto& out : n->outputs())
            m_outputs.emplace_back(get_tensor_descriptor_ptr(out));
}

std::shared_ptr<Node> LoweredExpr::get_node() const {
    if (!m_source_node)
        throw ngraph_error("An attempt to get uninitialized node from lowered expression");
    return  m_source_node;
}

std::shared_ptr<Emitter> LoweredExpr::get_emitter() const {
    return  m_emitter;
}

void LoweredExpr::init_emitter(const std::shared_ptr<const TargetMachine>& target) {
    m_emitter = target->get(m_source_node->get_type_info())(m_source_node);
}

void LoweredExpr::replace_input(const TensorDescriptorPtr& from, TensorDescriptorPtr to) {
    const auto& found = std::find(m_inputs.begin(), m_inputs.end(), from);
    if (found == m_inputs.end())
        throw ngraph_error("Failed to replace: target input is not found");
    *found = std::move(to);
}

void LoweredExpr::replace_output(const TensorDescriptorPtr& from, TensorDescriptorPtr to) {
    const auto& found = std::find(m_outputs.begin(), m_outputs.end(), from);
    if (found == m_outputs.end())
        throw ngraph_error("Failed to replace: target output is not found");
    *found = std::move(to);
}

IOLoweredExpr::IOLoweredExpr(const std::shared_ptr<ov::opset1::Parameter>& par, int64_t index)
             : LoweredExpr(par), m_index(index), m_type{io_type::INPUT} {
}

IOLoweredExpr::IOLoweredExpr(const std::shared_ptr<ov::opset1::Result>& res, int64_t index, std::vector<TensorDescriptorPtr> inputs)
              : LoweredExpr(), m_index(index), m_type{io_type::OUTPUT} {
    m_source_node = res;
    if (inputs.size() != res->get_input_size())
        throw ngraph_error("Invalid number of inputs for IOLoweredExpr construction");
    m_inputs = std::move(inputs);
    m_outputs = {};
}

LoweredExprIR::LoweredExprIR(const std::shared_ptr<ov::Model>& model, LoweringConfig config)
    : m_config{std::move(config)}, m_io_lowered_ops{} {
    for (const auto& n : get_ordered_ops(model)) {
        std::shared_ptr<LoweredExpr> expr;
        std::vector<TensorDescriptorPtr> input_tds;
        for (const auto& in : n->inputs()) {
            const auto& out = in.get_source_output();
            const auto& parent_out_tds = m_node2expression_map[out.get_node_shared_ptr()]->get_outputs();
            input_tds.push_back(parent_out_tds[out.get_index()]);
        }
        if (const auto& par = as_type_ptr<opset1::Parameter>(n)) {
            auto io_expr = std::make_shared<IOLoweredExpr>(par, model->get_parameter_index(par));
            m_io_lowered_ops.push_back(io_expr);
            expr = io_expr;
        } else if (const auto& res = as_type_ptr<opset1::Result>(n)) {
            auto io_expr = std::make_shared<IOLoweredExpr>(res, model->get_result_index(res), input_tds);
            m_io_lowered_ops.push_back(io_expr);
            expr = io_expr;
        } else {
            // Note that output tds must be empty since they are filled automatically from rt_info and/or tensor shapes
            expr = std::make_shared<LoweredExpr>(n, input_tds, std::vector<TensorDescriptorPtr>{});
        }
        register_expression(expr);
        m_lowered_ops.emplace_back(expr);
    }
}

ov::NodeVector LoweredExprIR::get_ordered_ops(const std::shared_ptr<ov::Model>& m) {
    if (!m->get_sinks().empty())
        throw ngraph_error("Linear IR is not supposed to work for model with sinks. Check your transformation pipeline.");

    // Note that an important difference between this impl and Model::get_ordered_ops is that Results and Parameters
    // are added in REVERSE order, so they will be visited in DIRECT order compared to get_parameters() and get_results()
    NodeVector nodes;
    const auto& results = m->get_results();
    std::copy(results.rbegin(), results.rend(), std::back_inserter(nodes));
    const auto& params = m->get_parameters();
    std::copy(params.rbegin(), params.rend(), std::back_inserter(nodes));

    return ov::topological_sort(nodes);
}

void LoweredExprIR::serialize(const std::string& xml, const std::string& bin) {
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

LoweredExprIR::container LoweredExprIR::deep_copy_range(LoweredExprIR::container::const_iterator begin, LoweredExprIR::container::const_iterator end) {
    LoweredExprIR::container result;
    NodeVector original_nodes;
    for (auto it = begin; it != end; it++)
        original_nodes.push_back((*it)->get_node());
    NodeMap node_map;
    const NodeVector& new_nodes = ngraph::clone_nodes(original_nodes,  node_map);
    for (auto it = begin; it != end; it++) {
        // copy by value, so result shared_pointer point to new objects
        LoweredExpr new_expr = **it;
        new_expr.m_source_node = node_map[(*it)->get_node().get()];
        result.emplace_back(std::make_shared<LoweredExpr>(new_expr));
    }
    return result;
}

LoweredExprIR LoweredExprIR::deep_copy() const {
    LoweredExprIR result;
    auto& result_ops = result.m_lowered_ops;
    for (const auto& expr : deep_copy_range(m_lowered_ops.begin(), m_lowered_ops.end()))
        result_ops.emplace_back(expr);
    result.m_config = m_config;
    return result;
}

void LoweredExprIR::debug_print(bool tds_as_pointers) const {
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
                    throw ngraph_error("Undefined input descriptor for op");
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

void LoweredExprIR::init_emitters(const std::shared_ptr<TargetMachine>& target) {
    for (auto& expr : m_lowered_ops) {
        if (!expr->get_emitter())
            expr->init_emitter(target);
    }
}

LoweredExprPtr LoweredExprIR::get_expr_by_node(const std::shared_ptr<Node>& n) const {
    auto found = m_node2expression_map.find(n);
    return found == m_node2expression_map.end() ? nullptr : found->second;
}

LoweredExprPtr LoweredExprIR::get_expr_by_output(const TensorDescriptorPtr& td) const {
    auto found = m_output2expression_map.find(td);
    if (found == m_output2expression_map.end())
        throw ngraph_error("Failed to find expression by output tensor descriptor");
    return found->second;
}

const std::set<LoweredExprPtr>& LoweredExprIR::get_exprs_by_input(const TensorDescriptorPtr& td) const {
    auto found = m_input2expression_map.find(td);
    if (found == m_input2expression_map.end())
        throw ngraph_error("Failed to find expression by input tensor descriptor");
    return found->second;
}

void LoweredExprIR::replace_input(const LoweredExprPtr& expr, const TensorDescriptorPtr& from, TensorDescriptorPtr to) {
    auto found = m_input2expression_map.find(from);
    if (found == m_input2expression_map.end() || found->second.count(expr) == 0)
        throw ngraph_error("Invalid expression of input was provided to replace_input");
    found->second.erase(expr);
    {
        const auto& res = m_input2expression_map.insert({to, std::set<LoweredExprPtr> {expr}});
        // If input is already in the map => add ExprPtr to the mapped set
        if (!res.second) {
            res.first->second.insert(expr);
        }
    }
    expr->replace_input(from, std::move(to));
}

void LoweredExprIR::replace_output(const LoweredExprPtr& expr, const TensorDescriptorPtr& from, TensorDescriptorPtr to) {
    auto found = m_output2expression_map.find(from);
    if (found == m_output2expression_map.end() || found->second != expr)
        throw ngraph_error("Invalid expression of output was provided to replace_output");
    m_output2expression_map.erase(found);
    m_output2expression_map[to] = expr;
    expr->replace_output(from, to);
}

void LoweredExprIR::register_regular_expression(const LoweredExprPtr& expr) {
    if (is_type<ov::op::v0::Result>(expr->get_node()) || is_type<ov::op::v0::Parameter>(expr->get_node()))
        throw ngraph_error("LoweredExprIR::insert can't be used to add Parameters or Results to IR");
    register_expression(expr);
}

void LoweredExprIR::register_expression(const LoweredExprPtr& expr) {
    const auto& node = expr->get_node();
    {
        const auto& res = m_node2expression_map.insert({node, expr});
        if (!res.second)
            throw ngraph_error("Duplicate node is detected in linear IR: " + std::string(node->get_friendly_name()));
    }
    for (const auto& out : expr->m_outputs)
        m_output2expression_map[out] = expr;

    for (const auto& in : expr->m_inputs) {
        const auto& res = m_input2expression_map.insert({in, std::set<LoweredExprPtr>{expr}});
        // If input is already in the map => add ExprPtr to the mapped set
        if (!res.second) {
            res.first->second.insert(expr);
        }
    }
}

void LoweredExprIR::unregister_expression(const LoweredExprPtr& expr) {
    for (const auto& out : expr->m_outputs)
        m_output2expression_map.erase(out);

    for (const auto& in : expr->m_inputs) {
        const auto& found = m_input2expression_map.find(in);
        if (found != m_input2expression_map.end()) {
            // Note: If the input is used by only by this expr => delete the whole entry
            //       Otherwise delete the expr from the users set
            auto& users = found->second;
            if (users.size() == 1)
                m_input2expression_map.erase(found);
            else
                users.erase(expr);
        }
    }

    m_node2expression_map.erase(expr->get_node());
}

LoweredExprIR::exprIt LoweredExprIR::insert(constExprIt pos, container::value_type&& value) {
    register_regular_expression(value);
    return m_lowered_ops.insert(pos, value);
}

LoweredExprIR::exprIt LoweredExprIR::insert(constExprIt pos, const container::value_type& value) {
    register_regular_expression(value);
    return m_lowered_ops.insert(pos, value);
}

LoweredExprIR::exprIt LoweredExprIR::insert(constExprIt pos, exprIt begin, exprIt end) {
    constExprIt cbegin = begin;
    constExprIt cend = end;
    return insert(pos, cbegin, cend);
}

LoweredExprIR::exprIt LoweredExprIR::insert(constExprIt pos, constExprIt begin, constExprIt end) {
    for (auto b = begin; b != end; b++)
        register_regular_expression(*b);
    return m_lowered_ops.insert(pos, begin, end);
}

LoweredExprIR::exprIt LoweredExprIR::insert(LoweredExprIR::constExprIt pos, const NodeVector& nodes) {
    auto ret = m_lowered_ops.end();
    for (const auto& n : nodes) {
        std::vector<TensorDescriptorPtr> input_tds;
        for (const auto& in : n->inputs()) {
            const auto& out = in.get_source_output();
            const auto& parent_out_tds = m_node2expression_map[out.get_node_shared_ptr()]->get_outputs();
            input_tds.push_back(parent_out_tds[out.get_index()]);
        }
        // Note that output tds must be empty since they are filled automatically from rt_info and/or tensor shapes
        const auto& expr = std::make_shared<LoweredExpr>(n, input_tds, std::vector<TensorDescriptorPtr>{});
        register_regular_expression(expr);
        ret = m_lowered_ops.insert(pos, expr);
    }
    // Need to return iterator to the first of the inserted values
    return std::prev(ret, static_cast<int64_t>(nodes.size()));
}
// todo reuse for node vector to avoid code duplication
LoweredExprIR::exprIt LoweredExprIR::insert(LoweredExprIR::constExprIt pos, const std::shared_ptr<Node>& n) {
    std::vector<TensorDescriptorPtr> input_tds;
    for (const auto& in : n->inputs()) {
        const auto& out = in.get_source_output();
        const auto& parent_out_tds = m_node2expression_map[out.get_node_shared_ptr()]->get_outputs();
        input_tds.push_back(parent_out_tds[out.get_index()]);
    }
    // Note that output tds must be empty since they are filled automatically from rt_info and/or tensor shapes
    const auto& expr = std::make_shared<LoweredExpr>(n, input_tds, std::vector<TensorDescriptorPtr>{});
    register_regular_expression(expr);
    return m_lowered_ops.insert(pos, expr);
}

LoweredExprIR::exprIt LoweredExprIR::erase(LoweredExprIR::exprIt pos) {
    unregister_expression(*pos);
    return m_lowered_ops.erase(pos);
}

LoweredExprIR::exprIt LoweredExprIR::erase(LoweredExprIR::constExprIt pos) {
    unregister_expression(*pos);
    return m_lowered_ops.erase(pos);
}

}// namespace snippets
}// namespace ngraph
