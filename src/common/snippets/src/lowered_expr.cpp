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

size_t LoweredExpr::LOOP_NULL_ID = SIZE_MAX;

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

void LoweredExpr::replace_input(size_t port, TensorDescriptorPtr to) {
    OPENVINO_ASSERT(port < m_inputs.size(), "Failed to replace: target input port must be less than input count!");
    m_inputs[port] = std::move(to);
}

void LoweredExpr::replace_output(size_t port, TensorDescriptorPtr to) {
    OPENVINO_ASSERT(port < m_outputs.size(), "Failed to replace: target output port must be less than output count!");
    m_outputs[port] = std::move(to);
}

void LoweredExpr::set_loop_id(size_t id, size_t idx) {
    OPENVINO_ASSERT((std::find(m_loop_ids.begin(), m_loop_ids.end(), id) == m_loop_ids.end()),
                    "LoweredExpr cannot have several the same Loops");
    if (m_loop_ids.size() <= idx) {
        m_loop_ids.resize(idx + 1, LOOP_NULL_ID);
    }
    m_loop_ids[idx] = id;
}

void LoweredExpr::remove_loop_id(size_t id) {
    auto it = std::find(m_loop_ids.begin(), m_loop_ids.end(), id);
    OPENVINO_ASSERT(it == m_loop_ids.end(), "LoweredExpr doesn't have the Loop with ID " + std::to_string(id));
    *it = LoweredExpr::LOOP_NULL_ID;
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

LoweredExprPort::LoweredExprPort(const LoweredExprPtr& expr, size_t port, Type type) : expr(expr), port(port), type(type) {
    if (type == Type::Input) {
        OPENVINO_ASSERT(port < expr->get_inputs().size(), "The input port must be less than input count");
    } else if (type == Type::Output) {
        OPENVINO_ASSERT(port < expr->get_outputs().size(), "The output port must be less than output count");
    }
}

LoweredExprPort LoweredExprPort::make_input(const LoweredExprPtr& expr, size_t port) {
    return LoweredExprPort(expr, port, Type::Input);
}
LoweredExprPort LoweredExprPort::make_output(const LoweredExprPtr& expr, size_t port) {
    return LoweredExprPort(expr, port, Type::Output);
}

bool operator==(const LoweredExprPort& lhs, const LoweredExprPort& rhs) {
    if (&lhs == &rhs)
        return true;
    OPENVINO_ASSERT(lhs.type == rhs.type, "Incorrect comparison: Ports are from different types!");
    return lhs.expr == rhs.expr && lhs.port == rhs.port;
}

bool operator!=(const LoweredExprPort& lhs, const LoweredExprPort& rhs) {
    return !(lhs == rhs);
}

bool operator<(const LoweredExprPort& lhs, const LoweredExprPort& rhs) {
    OPENVINO_ASSERT(lhs.type == rhs.type, "Incorrect comparison: Ports are from different types!");
    return (lhs.expr < rhs.expr) || (lhs.expr == rhs.expr && lhs.port < rhs.port);
}

LoweredExprIR::LoweredExprIR(const std::shared_ptr<ov::Model>& model, LoweringConfig config)
    : m_io_lowered_ops{}, m_config{std::move(config)}, m_loop_manager(std::make_shared<LoweredLoopManager>()) {
    constExprIt scalar_pos = m_lowered_ops.begin();
    LoweredExprPtr last_param = nullptr;
    for (const auto& n : get_ordered_ops(model)) {
        constExprIt insertion_pos = m_lowered_ops.end();
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
            last_param = expr;
        } else if (const auto& res = as_type_ptr<opset1::Result>(n)) {
            auto io_expr = std::make_shared<IOLoweredExpr>(res, model->get_result_index(res), input_tds);
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
            expr = std::make_shared<LoweredExpr>(n, input_tds, std::vector<TensorDescriptorPtr>{});
        }
        register_expression(expr);
        m_lowered_ops.insert(insertion_pos, expr);
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
    ngraph::clone_nodes(original_nodes,  node_map);
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

LoweredExprPort LoweredExprIR::get_expr_by_output(const TensorDescriptorPtr& td) const {
    auto found = m_output2expression_map.find(td);
    if (found == m_output2expression_map.end())
        throw ngraph_error("Failed to find expression by output tensor descriptor");
    return found->second;
}

const std::set<LoweredExprPort>& LoweredExprIR::get_exprs_by_input(const TensorDescriptorPtr& td) const {
    auto found = m_input2expression_map.find(td);
    if (found == m_input2expression_map.end())
        throw ngraph_error("Failed to find expression by input tensor descriptor");
    return found->second;
}

void LoweredExprIR::replace_input(const LoweredExprPtr& expr, size_t port, const TensorDescriptorPtr& to) {
    replace_input(LoweredExprPort::make_input(expr, port), to);
}

void LoweredExprIR::replace_input(const LoweredExprPort& expr_port, const TensorDescriptorPtr& to) {
    const auto& expr = expr_port.expr;
    const auto port = expr_port.port;
    OPENVINO_ASSERT(expr_port.type == LoweredExprPort::Type::Input, "Failed to replace: target input port must have Input type");
    OPENVINO_ASSERT(port < expr->m_inputs.size(), "Failed to replace: target input port must be less than input count!");
    const auto from = expr->m_inputs[port];
    auto found = m_input2expression_map.find(from);
    if (found == m_input2expression_map.end() || found->second.count(expr_port) == 0)
        throw ngraph_error("Invalid expression of input was provided to replace_input");
    found->second.erase(expr_port);
    {
        const auto& res = m_input2expression_map.insert({to, std::set<LoweredExprPort>{expr_port}});
        // If input is already in the map => add ExprPtr to the mapped set
        if (!res.second) {
            res.first->second.insert(expr_port);
        }
    }
    expr->replace_input(port, std::move(to));
}

void LoweredExprIR::replace_output(const LoweredExprPtr& expr, size_t port, const TensorDescriptorPtr& to) {
    replace_output(LoweredExprPort::make_output(expr, port), to);
}

void LoweredExprIR::replace_output(const LoweredExprPort& expr_port, const TensorDescriptorPtr& to) {
    const auto& expr = expr_port.expr;
    const auto port = expr_port.port;
    OPENVINO_ASSERT(expr_port.type == LoweredExprPort::Type::Output, "Failed to replace: target output port must have Output type");
    OPENVINO_ASSERT(port < expr->m_outputs.size(), "Failed to replace: target output port must be less than output count!");
    const auto from = expr->m_outputs[port];
    auto found = m_output2expression_map.find(from);
    if (found == m_output2expression_map.end() || found->second != expr_port)
        throw ngraph_error("Invalid expression of output was provided to replace_output");
    m_output2expression_map.erase(found);
    m_output2expression_map[to] = expr_port;
    expr->replace_output(port, to);
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
    for (size_t i = 0; i < expr->m_outputs.size(); ++i) {
        const auto& out = expr->m_outputs[i];
        m_output2expression_map[out] = LoweredExprPort::make_output(expr, i);
    }

    for (size_t i = 0; i < expr->m_inputs.size(); ++i) {
        const auto& in = expr->m_inputs[i];
        const auto expr_port = LoweredExprPort::make_input(expr, i);
        const auto& res = m_input2expression_map.insert({in, std::set<LoweredExprPort>{expr_port}});
        // If input is already in the map => add ExprPtr to the mapped set
        if (!res.second) {
            res.first->second.insert(expr_port);
        }
    }
}

void LoweredExprIR::unregister_expression(const LoweredExprPtr& expr) {
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
                users.erase(LoweredExprPort::make_input(expr, in_port));
        }
        ++in_port;
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

void LoweredExprIR::move(LoweredExprIR::constExprIt from, LoweredExprIR::constExprIt to) {
    // Instead of `insert()` + `erase()`, we use `splice()` for the same list
    m_lowered_ops.splice(to, m_lowered_ops, from);
}

size_t LoweredExprIR::LoweredLoopManager::add_loop_info(const LoweredLoopInfoPtr& loop) {
    const auto index = next_id;
    m_map[index] = loop;
    next_id++;
    return index;
}

void LoweredExprIR::LoweredLoopManager::remove_loop_info(size_t index) {
    m_map.erase(index);
}

using LoweredLoopInfoPtr = LoweredExprIR::LoweredLoopManager::LoweredLoopInfoPtr;

const std::map<size_t, LoweredLoopInfoPtr>& LoweredExprIR::LoweredLoopManager::get_map() const {
    return m_map;
}

LoweredLoopInfoPtr LoweredExprIR::LoweredLoopManager::get_loop_info(size_t index) const {
    const auto it = m_map.find(index);
    OPENVINO_ASSERT(it != m_map.end(), "LoopInformation hasn't been found!");
    return it->second;
}

void LoweredExprIR::LoweredLoopManager::get_loop_bounds(const LoweredExprIR& linear_ir,
                                                        size_t loop_id,
                                                        LoweredExprIR::constExprIt& loop_begin_pos,
                                                        LoweredExprIR::constExprIt& loop_end_pos) const {
    const auto loop_info = get_loop_info(loop_id);
    get_loop_bounds(linear_ir, loop_info->entry_exprs, loop_info->exit_exprs, loop_begin_pos, loop_end_pos, loop_id);
}

void LoweredExprIR::LoweredLoopManager::get_loop_bounds(const LoweredExprIR& linear_ir,
                                                        const std::vector<LoweredExprPort>& entries,
                                                        const std::vector<LoweredExprPort>& exits,
                                                        LoweredExprIR::constExprIt& loop_begin_pos,
                                                        LoweredExprIR::constExprIt& loop_end_pos,
                                                        size_t loop_id) {
    OPENVINO_ASSERT(!entries.empty(), "Loop must have entry points");
    OPENVINO_ASSERT(!exits.empty(), "Loop must have entry points");
    loop_begin_pos = std::find(linear_ir.begin(), linear_ir.end(), entries.front().expr);
    OPENVINO_ASSERT(loop_begin_pos != linear_ir.end(), "Loop begin hasn't been found!");

    // Some operations in Loop can be before first entry points: Scalars, VectorBuffer.
    // We should iterate by them till the expr is in the corresponding Loop
    auto prev_loop_ids = (*std::prev(loop_begin_pos))->get_loop_ids();
    while (std::find(prev_loop_ids.begin(), prev_loop_ids.end(), loop_id) != prev_loop_ids.end()) {
        loop_begin_pos = std::prev(loop_begin_pos);
        prev_loop_ids = (*std::prev(loop_begin_pos))->get_loop_ids();
    }

    // At the moment all Loops must have exit points
    loop_end_pos = std::next(std::find(loop_begin_pos, linear_ir.end(), exits.back().expr));
    OPENVINO_ASSERT(loop_end_pos != linear_ir.end(), "Loop end hasn't been found!");
}

void LoweredExprIR::LoweredLoopManager::get_io_loop_ports(LoweredExprIR& linear_ir,
                                                          LoweredExprIR::constExprIt loop_begin_pos,
                                                          LoweredExprIR::constExprIt loop_end_pos,
                                                          std::vector<LoweredExprPort>& entries,
                                                          std::vector<LoweredExprPort>& exits) {
    entries.clear();
    exits.clear();
    for (auto expr_it = loop_begin_pos; expr_it != loop_end_pos; ++expr_it) {
        const auto& expr = *expr_it;
        const auto inputs = expr->get_inputs();
        const auto outputs = expr->get_outputs();

        for (size_t in_port = 0; in_port < inputs.size(); ++in_port) {
            const auto in_td = inputs[in_port];
            const auto parent_expr = linear_ir.get_expr_by_output(in_td).expr;
            if (!ov::is_type<opset1::Constant>(parent_expr->get_node()) &&
                std::find(loop_begin_pos, expr_it, parent_expr) == expr_it) {
                entries.push_back(LoweredExprPort::make_input(expr, in_port));
            }
        }

        for (size_t out_port = 0; out_port < outputs.size(); ++out_port) {
            const auto out_td = outputs[out_port];
            const auto consumer_exprs = linear_ir.get_exprs_by_input(out_td);
            for (const auto& conumer_expr : consumer_exprs) {
                if (std::find(expr_it, loop_end_pos, conumer_expr.expr) == loop_end_pos) {
                    exits.push_back(LoweredExprPort::make_output(expr, out_port));
                    break;
                }
            }
        }
    }
}

void LoweredExprIR::LoweredLoopManager::skipped_mark(LoweredExprIR::constExprIt loop_begin_pos,
                                                        LoweredExprIR::constExprIt loop_end_pos,
                                                        size_t loop_depth) {
    const auto loop_ids = std::vector<size_t>(loop_depth, LoweredExpr::LOOP_NULL_ID);
    for (auto& expr_it = loop_begin_pos; expr_it != loop_end_pos; ++expr_it) {
        const auto expr = *expr_it;
        expr->set_loop_ids(loop_ids);
    }
}

void LoweredExprIR::LoweredLoopManager::mark_loop(LoweredExprIR& linear_ir,
                                                  LoweredExprIR::constExprIt loop_begin_pos,
                                                  LoweredExprIR::constExprIt loop_end_pos,
                                                  size_t loop_depth, size_t vector_size) {
    std::vector<LoweredExprPort> loop_entry_points, loop_exit_points;
    LoweredLoopManager::get_io_loop_ports(linear_ir, loop_begin_pos, loop_end_pos, loop_entry_points, loop_exit_points);

    auto broadcast = [](std::vector<size_t>& lhs, const std::vector<size_t>& rhs) -> void {
        if (rhs == lhs)
            return;
        const auto lhs_size = lhs.size();
        const auto rhs_size = rhs.size();
        const auto size = std::max(lhs_size, rhs_size);
        std::vector<size_t> result(size, 1);
        lhs.resize(size, 1);
        for (size_t i = 0; i < size; ++i) {
            const auto lhs_value = i < lhs_size ? *(lhs.crbegin() + i) : 1;
            const auto rhs_value = i < rhs_size ? *(rhs.crbegin() + i) : 1;
            OPENVINO_ASSERT(lhs_value == rhs_value || lhs_value == 1 || rhs_value == 1, "Output shapes of Loop must be broadcastable!");
            *(lhs.rbegin() + i) = std::max(lhs_value, rhs_value);
        }
    };

    std::vector<size_t> loop_subtensor;
    std::vector<size_t> loop_layout;
    std::vector<size_t> loop_tensor(1, 1);  // Scalar
    for (const auto& exit_point : loop_exit_points) {
        const auto expr = exit_point.expr;
        const auto port = exit_point.port;
        const auto out_td = expr->get_outputs()[port];
        const auto out_tensor = out_td->get_tensor();
        const auto out_layout = out_td->get_layout();
        broadcast(loop_tensor, out_tensor);
        if (loop_layout.empty())
            loop_layout = out_layout;
        OPENVINO_ASSERT(loop_layout == out_layout, "Output layouts of Loop must be the same!");
    }

    for (const auto& entry_point : loop_entry_points) {
        const auto expr = entry_point.expr;
        const auto out_td = expr->get_outputs().front();
        const auto out_subtensor = out_td->get_subtensor();
        if (loop_subtensor.empty())
            loop_subtensor = out_subtensor;
        OPENVINO_ASSERT(loop_subtensor == out_subtensor, "Subtensors of Loop must be the same!");
    }

    for (size_t dim_idx = 0; dim_idx < loop_depth; ++dim_idx) {
        OPENVINO_ASSERT(dim_idx < loop_tensor.size(), "Incorrect indexes of Loop for markup");
        const auto dim = loop_layout.size() >= dim_idx ? *(loop_layout.rbegin() + dim_idx) : 0;
        const auto work_amount = loop_tensor.size() > dim ? loop_tensor[dim] : 0;
        const auto work_amount_increment = loop_subtensor.size() > dim_idx ? *(loop_subtensor.rbegin() + dim_idx) :
                                           dim_idx == 0 ? vector_size : 1;

        mark_loop(linear_ir, loop_begin_pos, loop_end_pos, loop_depth - dim_idx - 1, work_amount, work_amount_increment, loop_entry_points, loop_exit_points);
    }
}

void LoweredExprIR::LoweredLoopManager::mark_loop(LoweredExprIR& linear_ir,
                                                LoweredExprIR::constExprIt loop_begin_pos,
                                                LoweredExprIR::constExprIt loop_end_pos,
                                                size_t idx,
                                                size_t work_amount,
                                                size_t work_amount_increment,
                                                const std::vector<LoweredExprPort>& entries,
                                                const std::vector<LoweredExprPort>& exits) {
    const auto loop_info = std::make_shared<LoweredLoopManager::LoweredLoopInfo>(
            work_amount, work_amount_increment, entries, exits);
    const auto loop_id = this->add_loop_info(loop_info);
    exprs_marking(loop_begin_pos, loop_end_pos, loop_id, idx);
}

void LoweredExprIR::LoweredLoopManager::exprs_marking(LoweredExprIR::constExprIt loop_begin_pos,
                                                      LoweredExprIR::constExprIt loop_end_pos,
                                                      size_t loop_id, size_t idx) {
    for (auto expr_it = loop_begin_pos; expr_it != loop_end_pos; ++expr_it) {
        expr_it->get()->set_loop_id(loop_id, idx);
    }
}

}// namespace snippets
}// namespace ngraph
