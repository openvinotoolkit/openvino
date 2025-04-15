// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/linear_ir.hpp"

#include "snippets/itt.hpp"

#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/expression_factory.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/type.hpp"
#include "snippets/utils/utils.hpp"
#include "snippets/op/subgraph.hpp"

namespace ov {
namespace snippets {
namespace lowered {

LinearIR::LinearIR(Config config, const std::shared_ptr<IShapeInferSnippetsFactory>& factory)
    : m_expressions{},
      m_parameter_expressions{},
      m_result_expressions{},
      m_config(std::move(config)),
      m_loop_manager(std::make_shared<LoopManager>()),
      m_shape_infer_factory(factory),
      m_shape_infer(std::make_shared<LIRShapeInfer>(m_expressions, m_parameter_expressions, m_result_expressions)),
      m_expression_factory(std::make_shared<ExpressionFactory>(m_shape_infer_factory)) {}

LinearIR::LinearIR(const std::shared_ptr<ov::Model>& model,
                   const std::shared_ptr<IShapeInferSnippetsFactory>& factory,
                   Config config)
    : LinearIR(config, factory) {
    constExprIt last_param = m_expressions.end();
    for (const auto& n : get_ordered_ops(model)) {
        constExprIt insertion_pos = m_expressions.end();
        const auto expr = get_expr_factory()->build(n, get_expression_inputs_by_node(n));

        // Scalar should be on the Linear IR beginning after Parameters to have valid expression order after Loop passes.
        // After these passes we must call pass MoveScalarToConsumer() to have a correct accuracy.
        // For more details, please see the pass description
        if (const auto& scalar = as_type_ptr<op::Scalar>(n)) {
            insertion_pos = std::next(last_param);
        }

        register_expression(expr, true, 0);
        const auto& it = m_expressions.insert(insertion_pos, expr);
        if (ov::is_type<ov::op::v0::Parameter>(n))
            last_param = it;
    }
    for (const auto& param_expr : m_parameter_expressions)
        m_is_dynamic = m_is_dynamic || utils::is_dynamic_vdims(param_expr->get_output_port_descriptor(0)->get_shape());
    for (const auto& result_expr : m_result_expressions)
        m_is_dynamic = m_is_dynamic || utils::is_dynamic_vdims(result_expr->get_input_port_descriptor(0)->get_shape());

    enumerate_expressions();
}

const ExpressionFactoryPtr& LinearIR::get_expr_factory() const {
    OPENVINO_ASSERT(m_expression_factory, "ExpresstionFactory is missed!");
    return m_expression_factory;
}

std::vector<PortConnectorPtr> LinearIR::get_expression_inputs_by_node(const std::shared_ptr<Node>& n) const {
    OPENVINO_ASSERT(n != nullptr, "Failed expression inputs getting: node is null");
    std::vector<PortConnectorPtr> inputs(n->get_input_size(), nullptr);
    for (const auto& input : n->inputs()) {
        const auto input_source = input.get_source_output();
        const auto in_index = input.get_index();
        const auto& parent_expr = get_expr_by_node(input_source.get_node_shared_ptr());
        inputs[in_index] = parent_expr->get_output_port_connector(input_source.get_index());
    }
    return inputs;
}

namespace {
void update_consumers_and_regs(const ExpressionPtr& new_expr, const std::vector<std::set<ExpressionPort>>& consumers) {
    OPENVINO_ASSERT(consumers.empty() || consumers.size() == new_expr->get_output_count(),
                    "Failed to insert node: count of consumer sets must be sero or equal to output port count");
    for (size_t i = 0; i < consumers.size(); ++i) {
        const auto& port_consumers = consumers[i];
        replace_input_port_connectors(port_consumers, new_expr->get_output_port_connector(i));
        if (!port_consumers.empty())
            new_expr->get_output_port_descriptor(i)->set_reg(port_consumers.cbegin()->get_descriptor_ptr()->get_reg());
    }
    for (size_t i = 0; i < new_expr->get_input_count(); ++i) {
        new_expr->get_input_port_descriptor(i)->set_reg(new_expr->get_input_port_connector(i)->get_source().get_descriptor_ptr()->get_reg());
    }
}
} // namespace

ExpressionPtr LinearIR::create_expression(const std::shared_ptr<Node>& n, const std::vector<PortConnectorPtr>& new_inputs,
                                          const std::vector<size_t>& loop_ids, bool update_loop_ports,
                                          const std::vector<std::set<ExpressionPort>>& consumers) {
    const auto new_expr = get_expr_factory()->build(n, new_inputs);
    update_consumers_and_regs(new_expr, consumers);
    new_expr->set_loop_ids(loop_ids);

    if (update_loop_ports) {
        m_loop_manager->update_loop_ports(new_expr);
    }
    return new_expr;
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

bool LinearIR::is_dynamic() const {
    return m_is_dynamic;
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
        if (!expr->get_emitter()) {
            expr->m_emitter = target->get(expr->get_node()->get_type_info())(expr);
            OPENVINO_ASSERT(expr->m_emitter, "Emitter can't be created for the node ", expr->get_node());
        }
    }
}

const ExpressionPtr& LinearIR::get_expr_by_node(const std::shared_ptr<Node>& n) const {
    auto found = m_node2expression_map.find(n);
    OPENVINO_ASSERT(found != m_node2expression_map.end(), "The node " + n->get_friendly_name() + " hasn't been found in Linear IR");
    return found->second;
}

void LinearIR::register_expression(const ExpressionPtr& expr, bool io_allowed, double exec_num) {
    const auto& node = expr->get_node();
    OPENVINO_ASSERT(io_allowed || (!is_type<ov::op::v0::Result>(node) && !is_type<ov::op::v0::Parameter>(node)),
                    "LinearIR::insert can't be used to add Parameters or Results to IR");
    const auto& res = m_node2expression_map.insert({node, expr});
    OPENVINO_ASSERT(res.second, "Duplicate node is detected in linear IR: ", node);

    if (ov::is_type<ov::op::v0::Parameter>(node))
        m_parameter_expressions.push_back(expr);
    if (ov::is_type<ov::op::v0::Result>(node))
        m_result_expressions.push_back(expr);
    if (const auto buffer_expr = ov::as_type_ptr<BufferExpression>(expr))
        m_buffer_expressions.push_back(buffer_expr);
    expr->m_exec_num = exec_num;
}

void LinearIR::unregister_expression(const ExpressionPtr& expr) {
    for (size_t i = 0; i < expr->get_input_count(); ++i) {
        const auto& input = expr->get_input_port_connector(i);
        input->remove_consumer(expr->get_input_port(i));
    }

    const auto& node = expr->get_node();
    m_node2expression_map.erase(node);
    OPENVINO_ASSERT((!ov::is_type_any_of<ov::op::v0::Parameter, ov::op::v0::Result>(node)),
                    "unregister_expression mustn't be called for parameter or result expressions");
    if (const auto buffer_expr = ov::as_type_ptr<BufferExpression>(expr)) {
        const auto& it = std::find(m_buffer_expressions.cbegin(), m_buffer_expressions.cend(), buffer_expr);
        OPENVINO_ASSERT(it != m_buffer_expressions.cend(), "BufferExpression has not been found in the list of LinearIR Buffers!");
        m_buffer_expressions.erase(it);
    }
}

LinearIR::exprIt LinearIR::insert(constExprIt pos, container::value_type&& value) {
    register_expression(value, m_config.m_manual_build_support, get_inserted_expr_exec_num(pos));
    return m_expressions.insert(pos, value);
}

LinearIR::exprIt LinearIR::insert(constExprIt pos, const container::value_type& value) {
    register_expression(value, m_config.m_manual_build_support, get_inserted_expr_exec_num(pos));
    return m_expressions.insert(pos, value);
}

LinearIR::exprIt LinearIR::insert(constExprIt pos, exprIt begin, exprIt end) {
    constExprIt cbegin = begin;
    constExprIt cend = end;
    return insert(pos, cbegin, cend);
}

LinearIR::exprIt LinearIR::insert(constExprIt pos, constExprIt begin, constExprIt end) {
    // nothing to insert
    if (begin == end)
        return m_expressions.erase(pos, pos); // to convert constExprIt to exprIt
    const auto count = std::distance(begin, end);
    auto ret = m_expressions.end();
    for (auto b = begin; b != end; b++) {
        ret = insert(pos, *b);
    }
    // Need to return iterator to the first of the inserted values
    return std::prev(ret, static_cast<int64_t>(count) - 1);
}

LinearIR::exprIt LinearIR::insert(LinearIR::constExprIt pos, const NodeVector& nodes) {
    // nothing to insert
    if (nodes.empty())
        return m_expressions.erase(pos, pos); // to convert constExprIt to exprIt
    auto ret = m_expressions.end();
    for (const auto& n : nodes)
        ret = insert(pos, n);
    // Need to return iterator to the first of the inserted values
    return std::prev(ret, static_cast<int64_t>(nodes.size()) - 1);
}

LinearIR::exprIt LinearIR::insert(LinearIR::constExprIt pos, const std::shared_ptr<Node>& n) {
    const auto& expr = get_expr_factory()->build(n, get_expression_inputs_by_node(n));
    register_expression(expr, m_config.m_manual_build_support, get_inserted_expr_exec_num(pos));
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
    if (from == std::prev(to)) return;
    // Instead of `insert()` + `erase()`, we use `splice()` for the same list
    (*from)->m_exec_num = get_inserted_expr_exec_num(to);
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

IShapeInferSnippets::Result LinearIR::shape_infer(const std::vector<VectorDimsRef>& input_shapes) {
    OPENVINO_ASSERT(m_shape_infer, "Attempt to call shape_infer when the shapeInfer instance was not created");
    return m_shape_infer->infer(input_shapes);
}

VectorDims LinearIR::get_master_shape() const {
    VectorDims master_shape{};
    // Note: inputs and outputs must be broadcastable, so it's enough to broadcast-merge only outputs
    // Note: Snippets would benefit from a more generic master_shape calculation approach.
    //  It will be implemented in the scope of ROI propagation activity (ticket 120505)
    if (m_result_expressions.size() == 1) {
        const auto& out_expr = *m_result_expressions.begin();
        const auto& source = out_expr->get_input_port_connector(0)->get_source();
        if (!m_config.m_enable_domain_optimization && ov::is_type<snippets::op::Brgemm>(source.get_expr()->get_node())) {
            master_shape = utils::get_preordered_vdims(source);
        } else {
            const auto& shape_infer_seq = utils::get_first_parent_shape_infer_expr_seq(out_expr);
            const auto& expr = shape_infer_seq.empty() ? out_expr : shape_infer_seq.back();
            master_shape = utils::get_preordered_vdims(expr->get_input_port_connector(0)->get_source());
        }
    } else {
        for (const auto& oe : m_result_expressions) {
            const auto& port_desc = oe->get_input_port_descriptor(0);
            OPENVINO_ASSERT(ov::snippets::broadcast_merge_into(master_shape, port_desc->get_shape()),
                            "Failed to merge input shapes in infer_master_shape");
        }
    }
    return master_shape;
}

template<>
LinearIR::exprIt LinearIR::insert_node(const std::shared_ptr<ov::Node>& new_node, const std::vector<PortConnectorPtr>& new_inputs,
                                       const std::vector<size_t>& loop_ids, bool update_loop_ports, const constExprIt& place,
                                       const std::vector<std::set<ExpressionPort>>& consumers) {
    const auto new_expr = create_expression(new_node, new_inputs, loop_ids, update_loop_ports, consumers);
    const auto expr_it = insert(place, new_expr);
    get_loop_manager()->sort_loop_ports(loop_ids);
    return expr_it;
}
template<>
LinearIR::exprIt LinearIR::insert_node(const std::shared_ptr<ov::Node>& new_node, const std::vector<ExpressionPort>& args,
                                       const std::vector<size_t>& loop_ids, bool update_loop_ports, const constExprIt& place,
                                       const std::vector<std::set<ExpressionPort>>& consumers) {
    std::vector<PortConnectorPtr> new_inputs(args.size());
    for (size_t i = 0; i < args.size(); ++i) {
        OPENVINO_ASSERT(args[i].get_type() == ExpressionPort::Output, "Expected output expression ports!");
        new_inputs[i] = args[i].get_port_connector_ptr();
    }
    return insert_node(new_node, new_inputs, loop_ids, update_loop_ports, place, consumers);
}

LinearIR::exprIt LinearIR::insert_expr(const ExpressionPtr& new_expr, const std::vector<size_t>& loop_ids,
                                       bool update_loop_ports, const constExprIt& place, const std::vector<std::set<ExpressionPort>>& consumers) {
    update_consumers_and_regs(new_expr, consumers);
    new_expr->set_loop_ids(loop_ids);

    const auto expr_it = insert(place, new_expr);
    if (update_loop_ports)
        get_loop_manager()->update_loop_ports(new_expr);

    return expr_it;
}

LinearIR::exprIt LinearIR::replace_with_node(const std::vector<ExpressionPtr>& old_exprs, const std::shared_ptr<ov::Node>& new_node,
                                             const std::vector<size_t>& loop_ids, const constExprIt& place) {
    OPENVINO_ASSERT(!old_exprs.empty(), "Failed to replace node: there are no old expressions for replacing");
    OPENVINO_ASSERT(new_node->get_output_size() == old_exprs.back()->get_output_count(),
                   "Failed to replace node: node output port count is not equal to output count of last old expression");

    std::vector<PortConnectorPtr> new_inputs(new_node->get_input_size());
    for (size_t i = 0; i < new_node->get_input_size(); ++i) {
        const auto& source = new_node->get_input_source_output(i);
        new_inputs[i] = get_expr_by_node(source.get_node_shared_ptr())->get_output_port_connector(source.get_index());
    }

    const auto last_old_expr = old_exprs.back();
    for (size_t i = 0; i < new_node->get_output_size(); ++i)
        snippets::lowered::PortDescriptorUtils::set_port_descriptor_ptr(new_node->output(i), last_old_expr->get_output_port_descriptor(0)->clone());

    const auto new_expr = create_expression(new_node, new_inputs, loop_ids, false);
    return replace_with_expr(old_exprs, new_expr, place);
}

LinearIR::exprIt LinearIR::replace_with_node(const std::vector<ExpressionPtr>& old_exprs, const std::shared_ptr<ov::Node>& new_node) {
    OPENVINO_ASSERT(!old_exprs.empty(), "Failed to replace node: there are no old expressions for replacing");
    const auto loop_ids = old_exprs.front()->get_loop_ids();
    OPENVINO_ASSERT(std::all_of(old_exprs.cbegin(), old_exprs.cend(), [&loop_ids](const ExpressionPtr& expr) { return expr->get_loop_ids() == loop_ids; }),
                    "Failed to replace node: cannot replace node to nodes with inconsistent loop ids");
    const auto insertion_place = std::next(find(old_exprs.back()));
    return replace_with_node(old_exprs, new_node, loop_ids, insertion_place);
}

LinearIR::exprIt LinearIR::replace_with_expr(const std::vector<ExpressionPtr>& old_exprs, const ExpressionPtr& new_expr, const constExprIt& place) {
    OPENVINO_ASSERT(!old_exprs.empty(), "Failed to replace node: there are no old expressions for replacing");
    OPENVINO_ASSERT(new_expr->get_output_count() == old_exprs.back()->get_output_count(),
                   "Failed to replace expressions: new expr output port count is not equal to output count of last old expression");

    const auto& new_inputs = new_expr->get_input_port_connectors();

    auto is_old_expr = [&old_exprs](const ExpressionPtr& expr) {
        return std::find(old_exprs.cbegin(), old_exprs.cend(), expr) != old_exprs.cend();
    };
    auto is_input_source = [&new_inputs](const ExpressionPort& source) {
        return std::any_of(new_inputs.cbegin(), new_inputs.cend(), [&source](const PortConnectorPtr& input) { return input->get_source() == source; });
    };

    // Validate removable expressions - they must be a `sequence`
    const auto last_old_expr = old_exprs.back();
    std::vector<std::set<ExpressionPort>> consumers(last_old_expr->get_output_count());
    for (const auto& old_expr : old_exprs) {
        for (const auto& input : old_expr->get_input_port_connectors()) {
            const auto& source = input->get_source();
            OPENVINO_ASSERT(is_old_expr(source.get_expr()) || is_input_source(source),
                            "Failed to replace nodes: not all output ports of existing expressions will be connected!");
        }
        for (const auto& port_connector : old_expr->get_output_port_connectors()) {
            const auto old_expr_consumers = port_connector->get_consumers();
            if (old_expr != last_old_expr) {
                for (const auto& consumer : old_expr_consumers) {
                    OPENVINO_ASSERT(is_old_expr(consumer.get_expr()),
                                    "Failed to replace nodes: not all input ports of existing expressions will be connected!");
                }
            } else {
                consumers[port_connector->get_source().get_index()] = old_expr_consumers;
            }
        }
    }

    update_consumers_and_regs(new_expr, consumers);

    const auto new_expr_it = insert(place, new_expr);
    const auto& inserted_expr = *new_expr_it;
    const auto& loop_ids = inserted_expr->get_loop_ids();
    const auto input_ports = inserted_expr->get_input_ports();
    const auto output_ports = inserted_expr->get_output_ports();
    for (const auto& old_expr : old_exprs) {
        for (size_t i = 0; i < old_expr->get_input_count(); ++i) {
            for (const auto& loop_id : loop_ids) {
                const auto& loop_info = m_loop_manager->get_loop_info(loop_id);
                if (!loop_info->is_loop_port(old_expr->get_input_port(i))) {
                    continue;
                }
                std::vector<ExpressionPort> new_input_ports;
                const auto& old_expr_source = old_expr->get_input_port_connector(i)->get_source();
                for (size_t j = 0; j < inserted_expr->get_input_count(); ++j) {
                    if (inserted_expr->get_input_port_connector(j)->get_source() == old_expr_source) {
                        new_input_ports.push_back(inserted_expr->get_input_port(j));
                    }
                }
                if (!new_input_ports.empty()) {
                    m_loop_manager->replace_loop_ports(loop_ids, old_expr->get_input_port(i), new_input_ports);
                }
            }
        }
        for (size_t i = 0; i < old_expr->get_output_count(); ++i) {
            m_loop_manager->replace_loop_ports(loop_ids,
                                               old_expr->get_output_port(i),
                                               {new_expr_it->get()->get_output_port(i)});
        }
        erase(find(old_expr));
    }
    return new_expr_it;
}

LinearIR::exprIt LinearIR::replace_with_expr(const std::vector<ExpressionPtr>& old_exprs, const ExpressionPtr& new_expr) {
    OPENVINO_ASSERT(!old_exprs.empty(), "Failed to replace node: there are no old expressions for replacing");
    const auto loop_ids = new_expr->get_loop_ids();
    OPENVINO_ASSERT(std::all_of(old_exprs.cbegin(), old_exprs.cend(), [&loop_ids](const ExpressionPtr& expr) { return expr->get_loop_ids() == loop_ids; }),
                    "Failed to replace node: cannot replace node to nodes with inconsistent loop ids");
    const auto insertion_place = std::next(find(old_exprs.back()));
    return replace_with_expr(old_exprs, new_expr, insertion_place);
}

void LinearIR::enumerate_expressions() const {
    /* --------------- Scheme ----------------- */
    //      Before      Inside      After
    //  - |----------|-----0-----|----------| +
    //             Start        End
    /*------------------------------------------*/
    const double region = std::numeric_limits<double>::max() / 3; // 3 roughly equivalent working zones: before, inside, after
    const double step = region / static_cast<double>(m_expressions.size()); // distribute exprs evenly over the working zone
    double num = -1 * (region / 2); // main working zone is inside: [Start, End] = [-region/2, region/2]
    for (const auto& expr : m_expressions) {
        expr->m_exec_num = num;
        num += step;
    }
}

double LinearIR::get_inserted_expr_exec_num(constExprIt insertion_pos) const {
    if (empty()) return 0;
    const auto& right_pos = insertion_pos;
    const auto& left_pos = std::prev(insertion_pos);
    double left_order = 0, right_order = 0;
    if (right_pos == cbegin()) {      // On the list begin
        if (right_pos->get()->get_exec_num() == -1 * std::numeric_limits<double>::max())
            enumerate_expressions();
        return right_pos->get()->get_exec_num() - 1;
    } else if (right_pos == cend()) { // On the list end
        if (left_pos->get()->get_exec_num() == std::numeric_limits<double>::max())
            enumerate_expressions();
        return left_pos->get()->get_exec_num() + 1;
    } else {                          // In the list middle
        left_order = left_pos->get()->get_exec_num();
        right_order = right_pos->get()->get_exec_num();
        OPENVINO_ASSERT(right_order > left_order, "Incorrect expression enumeration!");

        // sync point to enumerate expressions
        // 10 * eps - is to avoid meaningless result after (right_order + left_order) / 2 below
        if (std::abs(1 - left_order/right_order) <= 10 * std::numeric_limits<double>::epsilon()) {
            enumerate_expressions();
            left_order = left_pos->get()->get_exec_num();
            right_order = right_pos->get()->get_exec_num();
        }
    }
    // to avoid possible oveflow in (right_order + left_order) / 2;
    return left_order + (right_order - left_order) / 2;
}

LinearIR::LIRShapeInfer::LIRShapeInfer(const container& body_exprs,
                                       const std::vector<ExpressionPtr>& param_exprs,
                                       const std::vector<ExpressionPtr>& result_exprs)
    : ShapeInferSnippetsNode(), m_exprs(body_exprs), m_input_exprs(param_exprs), m_output_exprs(result_exprs) {
    // Note that if all output shapes are static, as in the case when the first shape infer was performed on nGraph,
    // we can treat them as the last result
    std::vector<VectorDims> outputDims;
    outputDims.reserve(m_output_exprs.size());
    for (const auto& expr : m_output_exprs) {
        const auto &shape = expr->get_input_port_descriptor(0)->get_shape();
        if (utils::is_dynamic_vdims(shape)) {
            outputDims.clear();
            break;
        }
        outputDims.push_back(shape);
    }
    m_last_result = {outputDims, ShapeInferStatus::success};
}

IShapeInferSnippets::Result LinearIR::LIRShapeInfer::infer(const std::vector<VectorDimsRef>& input_shapes) {
    OPENVINO_ASSERT(m_input_exprs.size() == input_shapes.size(), "Got invalid number of input shapes in LIR ShapeInfer");
    size_t input_count = 0;
    for (const auto& expr : m_input_exprs)
        expr->get_output_port_descriptor(0)->set_shape(input_shapes[input_count++]);

    for (const auto& expr : m_exprs) {
        if (expr->needShapeInfer())
            expr->updateShapes();
    }

    std::vector<VectorDims> outputDims;
    outputDims.reserve(m_output_exprs.size());
    for (const auto& expr : m_output_exprs) {
        outputDims.push_back(expr->get_input_port_descriptor(0)->get_shape());
    }
    m_last_result = {outputDims, ShapeInferStatus::success};
    return m_last_result;
}

}// namespace lowered
}// namespace snippets
}// namespace ov
