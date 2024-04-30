// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/linear_ir.hpp"

#include "snippets/itt.hpp"

#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/expression_factory.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/type.hpp"
#include "snippets/utils.hpp"
#include "snippets/op/subgraph.hpp"

namespace ov {
namespace snippets {
namespace lowered {

LinearIR::LinearIR(const std::shared_ptr<ov::Model>& model, const std::shared_ptr<IShapeInferSnippetsFactory>& factory, Config config)
        : m_io_expressions{}, m_config{config}, m_loop_manager(std::make_shared<LoopManager>()), m_shape_infer_factory(factory) {
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
            switch (io_expr->get_type()) {
                case IOExpression::io_type::INPUT:
                    m_is_dynamic = m_is_dynamic || utils::is_dynamic_vdims(io_expr->get_output_port_descriptor(0)->get_shape());
                    break;
                case IOExpression::io_type::OUTPUT:
                    m_is_dynamic = m_is_dynamic || utils::is_dynamic_vdims(io_expr->get_input_port_descriptor(0)->get_shape());
                    break;
                default:
                    OPENVINO_THROW("Incorrect IO Expression type");
            }
        }
    }
    m_shape_infer = std::make_shared<LIRShapeInfer>(m_expressions, m_io_expressions);
}

std::shared_ptr<LinearIR> LinearIR::clone() const {
    auto cloned = std::make_shared<LinearIR>();
    cloned->m_config = m_config;

    ExpressionMap expression_map;
    cloned->m_expressions = deep_copy_range(m_expressions.cbegin(), m_expressions.cend(), expression_map);
    for (const auto& expr : cloned->m_expressions) {
        cloned->m_node2expression_map[expr->get_node()] = expr;
        if (const auto& io = std::dynamic_pointer_cast<IOExpression>(expr))
            cloned->m_io_expressions.push_back(io);
    }

    cloned->m_loop_manager = m_loop_manager->clone_with_new_expr(expression_map);
    // It's Ok to share shapeInfer factory ptr, since the factory doesn't depend on LIR in any way
    cloned->m_shape_infer_factory = m_shape_infer_factory;
    cloned->m_shape_infer = std::make_shared<LIRShapeInfer>(cloned->m_expressions, cloned->m_io_expressions);
    cloned->m_is_dynamic = m_is_dynamic;
    return cloned;
}

ExpressionPtr LinearIR::create_expression(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model) {
    return ExpressionFactory::build(n, *this, model);
}

ExpressionPtr LinearIR::create_expression(const std::shared_ptr<Node>& n, const std::vector<PortConnectorPtr>& inputs) const {
    return ExpressionFactory::build(n, inputs, *this);
}

ExpressionPtr LinearIR::create_expression(const std::shared_ptr<Node>& n, const std::vector<PortConnectorPtr>& new_inputs,
                                          const std::vector<size_t>& loop_ids, bool update_loop_ports,
                                          const std::vector<std::set<ExpressionPort>>& consumers) {
    const auto new_expr = create_expression(n, new_inputs);
    new_expr->set_loop_ids(loop_ids);

    OPENVINO_ASSERT(consumers.empty() || consumers.size() == new_expr->get_output_count(),
                    "Failed to insert node: count of consumer sets must be sero or equal to output port count");
    for (size_t i = 0; i < consumers.size(); ++i) {
        const auto& port_consumers = consumers[i];
        replace_input_port_connectors(port_consumers, new_expr->get_output_port_connector(i));
    }

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

namespace {
using NodeMap = std::unordered_map<ov::Node*, std::shared_ptr<ov::Node>>;

std::vector<std::shared_ptr<ov::Node>> clone_nodes(const std::vector<std::shared_ptr<ov::Node>>& nodes,
                                                   NodeMap& node_map) {
    // for each node in topological order
    auto sorted_nodes = topological_sort(nodes);
    for (const auto& node : sorted_nodes) {
        if (node_map.count(node.get()) == 0) {
            // get (already) cloned arguments and clone the node
            OutputVector cloned_args;
            for (auto input : node->inputs()) {
                ov::Output<Node> output = input.get_source_output();
                cloned_args.push_back(output.for_node(node_map.at(output.get_node())));
            }
            std::vector<std::shared_ptr<Node>> cloned_dependencies;
            for (auto& dependency : node->get_control_dependencies()) {
                std::shared_ptr<Node>& dependent = node_map.at(dependency.get());
                if (find(cloned_dependencies.begin(), cloned_dependencies.end(), dependent) ==
                    cloned_dependencies.end()) {
                    cloned_dependencies.push_back(dependent);
                }
            }
            auto cloned_node = node->copy_with_new_inputs(cloned_args, cloned_dependencies);
            // There is a friendly name for this node so copy it
            cloned_node->set_friendly_name(node->get_friendly_name());
            auto rt_info = node->get_rt_info();
            cloned_node->get_rt_info() = rt_info;

            for (auto output : node->outputs()) {
                const auto& output_rt_info = output.get_rt_info();
                auto new_output = output.for_node(cloned_node);
                new_output.get_rt_info() = output_rt_info;
            }

            for (auto input : node->inputs()) {
                const auto& output_rt_info = input.get_rt_info();
                auto new_input = cloned_node->input(input.get_index());
                new_input.get_rt_info() = output_rt_info;
            }

            node_map[node.get()] = cloned_node;
        }
    }

    // create and return vector of cloned nodes
    // order matches input vector (not necessarily topological)
    std::vector<std::shared_ptr<ov::Node>> cloned_nodes;
    for (const auto& node : nodes) {
        cloned_nodes.push_back(node_map.at(node.get()));
    }
    return cloned_nodes;
}
}  // namespace

LinearIR::container LinearIR::deep_copy_range(LinearIR::container::const_iterator begin,
                                              LinearIR::container::const_iterator end,
                                              ExpressionMap& expression_map) {
    OPENVINO_ASSERT(expression_map.empty(), "deep_copy_range expects empty expression_map as an input");
    LinearIR::container result;
    NodeVector original_nodes;
    for (auto it = begin; it != end; it++) {
        original_nodes.push_back((*it)->get_node());
    }

    // node_map and expr_map map original node pointer (expression) to a new pointer (expression)
    NodeMap node_map;
    clone_nodes(original_nodes, node_map);

    for (auto it = begin; it != end; it++) {
        const auto& expr = *it;
        const auto& new_expr = expr->clone_with_new_inputs(expression_map, node_map[expr->get_node().get()]);
        result.push_back(new_expr);
        expression_map[expr.get()] = new_expr;
    }

    // Checking that the cloning was successful: the cloned part of LinearIR is identical to the original one
    for (auto result_it = result.cbegin(), original_it = begin; result_it != result.cend(); ++result_it, ++original_it) {
        const auto& result_expr = *result_it;
        const auto& original_expr = *original_it;
        OPENVINO_ASSERT(result_expr->get_node()->get_type_info() == original_expr->get_node()->get_type_info() &&
                        result_expr->get_input_count() == original_expr->get_input_count() &&
                        result_expr->get_output_count() == original_expr->get_output_count(),
                        "Expressions after copying aren't matched!");
    }

    return result;
}

bool LinearIR::is_dynamic() const {
    return m_is_dynamic;
}

void LinearIR::debug_print(bool tds_as_pointers) const {
    auto print_rinfo = [](const RegInfo& rinfo) {
        std::cerr << " : {";
        for (auto i : rinfo.first)
            std::cerr << regTypeToStr(i.type) << "[" << i.idx << "] ";
        std::cerr << " => ";
        for (auto i : rinfo.second)
            std::cerr << regTypeToStr(i.type) << "[" << i.idx << "] ";
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

IShapeInferSnippets::Result LinearIR::shape_infer(const std::vector<VectorDimsRef>& input_shapes) {
    OPENVINO_ASSERT(m_shape_infer, "Attempt to call shape_infer when the shapeInfer instance was not created");
    return m_shape_infer->infer(input_shapes);
}

VectorDims LinearIR::get_master_shape() const {
    VectorDims master_shape{};
    // Note: inputs and outputs must be broadcastable, so it's enough to broadcast-merge only outputs
    std::vector<std::shared_ptr<IOExpression>> out_exprs;
    for (const auto& ioe : m_io_expressions) {
        if (ioe->get_type() == IOExpression::io_type::OUTPUT)
            out_exprs.push_back(ioe);
    }
    // Note: Snippets would benefit from a more generic master_shape calculation approach.
    //  It will be implemented in the scope of ROI propagation activity (ticket 120505)
    if (out_exprs.size() == 1) {
        const auto& source = out_exprs[0]->get_input_port_connector(0)->get_source();
        if (!m_config.m_enable_domain_optimization && ov::is_type<snippets::op::Brgemm>(source.get_expr()->get_node())) {
            master_shape = utils::get_preordered_vdims(source);
        } else {
            const auto& shape_infer_seq = utils::get_first_parent_shape_infer_expr_seq(out_exprs[0]);
            const auto& expr = shape_infer_seq.empty() ? out_exprs[0] : shape_infer_seq.back();
            master_shape = utils::get_preordered_vdims(expr->get_input_port_connector(0)->get_source());
        }
    } else {
        for (const auto& oe : out_exprs) {
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
    return insert(place, new_expr);
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

    OPENVINO_ASSERT(consumers.empty() || consumers.size() == new_expr->get_output_count(),
                    "Failed to insert node: count of consumer sets must be sero or equal to output port count");
    for (size_t i = 0; i < consumers.size(); ++i) {
        const auto& port_consumers = consumers[i];
        replace_input_port_connectors(port_consumers, new_expr->get_output_port_connector(i));
    }

    const auto new_expr_it = insert(place, new_expr);
    const auto& loop_ids = new_expr_it->get()->get_loop_ids();
    const auto input_ports = new_expr_it->get()->get_input_ports();
    const auto output_ports = new_expr_it->get()->get_output_ports();
    for (const auto& old_expr : old_exprs) {
        for (size_t i = 0; i < old_expr->get_input_count(); ++i)
            m_loop_manager->update_loops_port(loop_ids, old_expr->get_input_port(i), input_ports, true);
        for (size_t i = 0; i < old_expr->get_input_count(); ++i)
            m_loop_manager->update_loops_port(loop_ids, old_expr->get_output_port(i), output_ports, false);
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

LinearIR::LIRShapeInfer::LIRShapeInfer(container& body_exprs, io_container& io_exprs)
                                       : ShapeInferSnippetsNode(),
                                         m_exprs{std::make_shared<container>(body_exprs)} {
    // Note that here we rely on the assumption that io_expressions can't be changed after the LIR was created
    for (const auto& expr : io_exprs) {
        if (expr->get_type() == IOExpression::io_type::INPUT) {
            m_input_exprs.push_back(expr);
        } else if (expr->get_type() == IOExpression::io_type::OUTPUT) {
            m_output_exprs.emplace_back(expr);
        } else {
            OPENVINO_THROW("Invalid io expression type detected");
        }
    }
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
    for (size_t i = 0; i < m_input_exprs.size(); i++)
        m_input_exprs[i]->get_output_port_descriptor(0)->set_shape(input_shapes[i]);

    for (const auto& expr : *m_exprs) {
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
