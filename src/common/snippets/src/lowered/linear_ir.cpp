// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/linear_ir.hpp"

#include "snippets/itt.hpp"

#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/expression_factory.hpp"
#include "snippets/op/serialization_node.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/type.hpp"
#include "snippets/utils.hpp"

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
        }
    }
    m_shape_infer = std::make_shared<LIRShapeInfer>(m_expressions, m_io_expressions);
}

std::shared_ptr<LinearIR> LinearIR::clone() const {
    auto cloned = std::make_shared<LinearIR>();
    cloned->m_config = m_config;

    ExressionMap expression_map;
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
    return cloned;
}

ExpressionPtr LinearIR::create_expression(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model) {
    return ExpressionFactory::build(n, *this, model);
}

ExpressionPtr LinearIR::create_expression(const std::shared_ptr<Node>& n, const std::vector<PortConnectorPtr>& inputs) {
    return ExpressionFactory::build(n, inputs, *this);
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

void LinearIR::serialize(const std::string& xml, const std::string& bin) const {
    auto first_node = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    first_node->set_friendly_name("Start");
    first_node->get_rt_info()["execTimeMcs"] = 0;
    std::shared_ptr<Node> serialization_node = first_node;

    // This map allows to get LoopBegin serialization node by original LoopBegin node
    // It is used to draw an edge between LoopBegin and LoopEnd serialization nodes
    std::map<std::shared_ptr<snippets::op::LoopBegin>, std::shared_ptr<Node>> loops_map;
    for (const auto& expr : m_expressions) {
        const auto node = expr->get_node();
        if (auto loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(node)) {
            OPENVINO_ASSERT(loops_map.count(loop_end->get_loop_begin()),
                            "Serialization can't find LoopBegin that corresponds to LoopEnd with friendly name ",
                            loop_end->get_friendly_name());
            auto loop_begin_serialization_node = loops_map.at(loop_end->get_loop_begin());
            serialization_node = std::make_shared<op::SerializationNode>(ov::OutputVector{serialization_node, loop_begin_serialization_node}, expr);
        } else {
            serialization_node = std::make_shared<op::SerializationNode>(ov::OutputVector{serialization_node}, expr);
            if (auto loop_begin = ov::as_type_ptr<snippets::op::LoopBegin>(node)) {
                loops_map[loop_begin] = serialization_node;
            }
        }
    }
    auto last_node = std::make_shared<ov::op::v0::Result>(serialization_node);
    last_node->set_friendly_name("End");
    const auto tmp_model = std::make_shared<ov::Model>(ResultVector {last_node},
                                                       ParameterVector {first_node},
                                                       "Lowered_IR_Serialization");
    ov::pass::Serialize(xml, bin).run_on_model(tmp_model);
}

LinearIR::container LinearIR::deep_copy_range(LinearIR::container::const_iterator begin,
                                              LinearIR::container::const_iterator end,
                                              ExressionMap& expression_map) {
    OPENVINO_ASSERT(expression_map.empty(), "deep_copy_range expects empty expression_map as an input");
    LinearIR::container result;
    NodeVector original_nodes;
    for (auto it = begin; it != end; it++)
        original_nodes.push_back((*it)->get_node());

    // node_map and expr_map map original node pointer (expression) to a new pointer (expression)
    ngraph::NodeMap node_map;
    OPENVINO_SUPPRESS_DEPRECATED_START
    ngraph::clone_nodes(original_nodes,  node_map);
    OPENVINO_SUPPRESS_DEPRECATED_END

    for (auto it = begin; it != end; it++) {
        const auto& expr = *it;
        const auto& new_expr = expr->clone_with_new_inputs(expression_map, node_map[expr->get_node().get()]);
        result.push_back(new_expr);
        expression_map[expr.get()] = new_expr;
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
            expr->m_emitter = target->get(expr->get_node()->get_type_info())(expr);
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
    const auto& source = out_exprs[0]->get_input_port_connector(0)->get_source();
    if (!m_config.m_enable_domain_optimization && out_exprs.size() == 1 &&
        ov::is_type<snippets::op::Brgemm>(source.get_expr()->get_node())) {
        master_shape = utils::get_preordered_vdims(source);
    } else {
        for (const auto& oe : out_exprs) {
            const auto& port_desc = oe->get_input_port_descriptor(0);
            OPENVINO_ASSERT(ov::snippets::broadcast_merge_into(master_shape, port_desc->get_shape()),
                            "Failed to merge input shapes in infer_master_shape");
        }
    }
    return master_shape;
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
