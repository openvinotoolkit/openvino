// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>
#include "snippets/remarks.hpp"

#include "snippets/op/subgraph.hpp"
#include "snippets/op/convert_saturation.hpp"
#include "snippets/pass/insert_load_store.hpp"
#include "snippets/pass/insert_movebroadcast.hpp"
#include "snippets/pass/broadcast_to_movebroadcast.hpp"
#include "snippets/pass/load_movebroadcast_to_broadcastload.hpp"
#include "snippets/pass/assign_registers.hpp"
#include "snippets/pass/convert_constants.hpp"
#include "snippets/pass/convert_power_to_powerstatic.hpp"
#include "snippets/pass/vector_to_scalar.hpp"
#include "snippets/pass/insert_loops.hpp"
#include "snippets/pass/transpose_decomposition.hpp"
#include "snippets/pass/transform_convert.hpp"
#include "snippets/pass/align_element_type.hpp"
#include "snippets/pass/matmul_to_brgemm.hpp"
#include "snippets/pass/fuse_transpose_brgemm.hpp"
#include "snippets/pass/softmax_decomposition.hpp"
#include "snippets/pass/reset_buffer.hpp"
#include "snippets/pass/insert_buffer.hpp"
#include "snippets/pass/loop_fusion.hpp"
#include "snippets/utils.hpp"

#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/utils/utils.hpp"

#include <ngraph/pass/manager.hpp>
#include "ngraph/pass/constant_folding.hpp"
#include "ov_ops/type_relaxed.hpp"
#include <openvino/pass/serialize.hpp>

#include <algorithm>
#include <memory>
#include <array>

using namespace std;
using namespace ngraph;
using namespace ov::op::util;

void snippets::op::Subgraph::set_generator(std::shared_ptr<ngraph::snippets::Generator> generator) {
    m_generator = generator;
}

void snippets::op::Subgraph::set_virtual_port_count(const size_t count) {
    m_virtual_port_count = count;
}

void snippets::op::Subgraph::set_buffer_needed(const bool need) {
    m_buffer_needed = need;
}

void snippets::op::Subgraph::init_config() {
    const auto ops = body_ptr()->get_ops();
    for (const auto& op : ops) {
        config.m_is_quantized = config.m_is_quantized ||
            ov::is_type<ov::op::v0::FakeQuantize>(op);
        config.m_has_type_relaxed_ops = config.m_has_type_relaxed_ops ||
            std::dynamic_pointer_cast<ngraph::op::TypeRelaxedBase>(op);
        config.m_is_needed_to_align_precision = config.m_is_needed_to_align_precision ||
            is_quantized() ||
            has_type_relaxed_ops() ||
            snippets::pass::AlignElementType::opNeedsAlignElementType(op, execution_element_type);
        config.m_has_domain_sensitive_ops = config.m_has_domain_sensitive_ops ||
            ov::is_type<ov::op::v1::Transpose>(op) ||
            ov::is_type<ov::op::v1::Softmax>(op) ||
            ov::is_type<ov::op::v8::Softmax>(op) ||
            ov::is_type<ov::op::v0::MatMul>(op);
    }
    // Domain sensitive ops are decomposed with explicit Loops. So, we should explicitly insert Loops in Subgraph if it contains these ops
    config.m_explicit_loop_insertion = config.m_has_domain_sensitive_ops;
}

snippets::op::Subgraph::Subgraph(const OutputVector& args, std::shared_ptr<ov::Model> body)
    : SubGraphOp(args), m_generator(nullptr) {
    set_function(body);
    init_config();
    constructor_validate_and_infer_types();
    for (size_t i = 0; i < body->get_parameters().size(); ++i)
        m_input_descriptions[0].push_back(std::make_shared<InvariantInputDescription>(i, i));
    for (size_t i = 0; i < body->get_output_size(); ++i)
        m_output_descriptions[0].push_back(std::make_shared<BodyOutputDescription>(i, i));
    m_transformations_allowed = false;
}

snippets::op::Subgraph::Subgraph(const NodeVector& args, std::shared_ptr<ov::Model> body)
    : Subgraph(as_output_vector(args), std::move(body)) {}

std::shared_ptr<Node> snippets::op::Subgraph::clone_with_new_inputs(const OutputVector& inputs) const {
    INTERNAL_OP_SCOPE(Subgraph);
    return make_shared<Subgraph>(inputs, ov::clone_model(body()));
}

std::vector<PartialShape> snippets::op::Subgraph::reshape_body(const std::vector<PartialShape>& input_shapes) {
    auto& params = body_ptr()->get_parameters();
    OPENVINO_ASSERT(params.size() == input_shapes.size(), "Got invalid number of input shapes to reshape subgraph body");
    for (size_t i = 0; i < params.size(); ++i) {
        params[i]->set_partial_shape(input_shapes[i]);
    }
    body_ptr()->validate_nodes_and_infer_types();
    std::vector<PartialShape> output_shapes;
    for (const auto& res : body_ptr()->get_results()) {
        output_shapes.emplace_back(res->get_input_partial_shape(0));
    }
    return output_shapes;
}

std::vector<Shape> snippets::op::Subgraph::reshape_body(const std::vector<Shape>& input_shapes) {
    auto& params = body_ptr()->get_parameters();
    OPENVINO_ASSERT(params.size() == input_shapes.size(), "Got invalid number of input shapes to reshape subgraph body");
    for (size_t i = 0; i < params.size(); ++i) {
        params[i]->set_partial_shape(input_shapes[i]);
    }
    body_ptr()->validate_nodes_and_infer_types();
    std::vector<Shape> output_shapes;
    for (const auto& res : body_ptr()->get_results()) {
        auto pshape = res->get_input_partial_shape(0);
        OPENVINO_ASSERT(pshape.is_static(), "Subgraph inferred dynamic output shape during reshape with static inputs");
        output_shapes.emplace_back(res->get_input_partial_shape(0).get_shape());
    }
    return output_shapes;
}

void snippets::op::Subgraph::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(Subgraph);
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::validate_and_infer_types")
    ngraph::ParameterVector old_parameters;
    for (auto op : body_ptr()->get_parameters()) {
        old_parameters.push_back(op);
    }

    for (size_t i = 0; i < get_input_size(); ++i) {
        body_ptr()->replace_parameter(i, std::make_shared<opset1::Parameter>(get_input_element_type(i), get_input_partial_shape(i)));
    }

    body_ptr()->validate_nodes_and_infer_types();

    for (size_t i = 0; i < body_ptr()->get_parameters().size(); i++) {
        body_ptr()->get_parameters()[i]->set_friendly_name(old_parameters[i]->get_friendly_name());
    }

    set_output_size(body_ptr()->get_output_size());
    for (size_t i = 0; i < get_output_size(); ++i) {
        set_output_type(i, body_ptr()->get_output_element_type(i), body_ptr()->get_output_partial_shape(i));
    }
}

bool snippets::op::Subgraph::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("body", body_ptr());
    visitor.on_attribute("input_descriptions", m_input_descriptions[0]);
    visitor.on_attribute("output_descriptions", m_output_descriptions[0]);
    return true;
}

auto snippets::op::Subgraph::wrap_node_as_subgraph(const std::shared_ptr<ov::Node>& node) -> std::shared_ptr<op::Subgraph> {
    INTERNAL_OP_SCOPE(Subgraph);
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::wrap_node_as_subgraph")
    ngraph::ParameterVector body_parameters;
    ngraph::OutputVector body_inputs;

    ngraph::OutputVector subgraph_inputs;

    for (const auto& input : node->input_values()) {
        if (ov::is_type<ngraph::opset1::Constant>(input.get_node_shared_ptr()) &&
            (ngraph::shape_size(input.get_shape()) == 1 ||
             ov::is_type<ov::op::v0::FakeQuantize>(node) ||
             constant_input_should_be_inside_body(node))) {
            body_inputs.push_back(input);
        } else {
            auto parameter = std::make_shared<ngraph::opset1::Parameter>(input.get_element_type(), input.get_partial_shape());
            body_parameters.push_back(parameter);
            body_parameters.back()->set_friendly_name(input.get_node()->get_friendly_name());
            body_inputs.push_back(parameter->output(0));

            subgraph_inputs.push_back(input);
        }
    }

    auto body_node = node->clone_with_new_inputs(body_inputs);
    body_node->set_friendly_name(node->get_friendly_name());
    for (size_t i = 0; i < node->get_output_size(); i++) {
        fill_empty_output_names(body_node->output(i), node->output(i));
    }

    if (node->get_output_size() != body_node->get_output_size()) {
        throw ngraph::ngraph_error("original node outputs size and extracted subgraph node outputs size doesn't much");
    }

    ngraph::ResultVector body_results;
    for (auto output : node->outputs()) {
        body_results.push_back(std::make_shared<ngraph::opset1::Result>(body_node->output(output.get_index())));
    }

    auto body = create_body(node->get_friendly_name(), body_results, body_parameters);
    auto subgraph = build_subgraph(node, subgraph_inputs, body);

    bool need_buffer = false;
    size_t hidden_data_count = 0lu;
    if (auto fq_node = ov::as_type_ptr<ov::op::v0::FakeQuantize>(node)) {
        hidden_data_count += utils::get_non_scalar_constant_count_for_fq(fq_node);
    // Ops that requires Buffer
    } else if (ov::is_type<ov::op::v1::Softmax>(node) ||
               ov::is_type<ov::op::v8::Softmax>(node)) {
        need_buffer |= true;
    }
    subgraph->set_virtual_port_count(hidden_data_count);
    subgraph->set_buffer_needed(need_buffer);

    for (size_t i = 0; i < body->get_parameters().size(); i++) {
        body->get_parameters()[i]->set_friendly_name(body_parameters[i]->get_friendly_name());
    }

    if (subgraph->get_output_size() != body->get_results().size()) {
        throw ngraph::ngraph_error("newly create subgraph doesn't much number of original node results");
    }

    return subgraph;
}

void snippets::op::Subgraph::fill_empty_output_names(const Output<Node>& target_output_node, const Output<Node>& replacement_output_node) {
    NGRAPH_SUPPRESS_DEPRECATED_START
    auto& out_tensor = target_output_node.get_tensor();
    const std::string new_name = ngraph::op::util::get_ie_output_name(replacement_output_node);
    if (ov::descriptor::get_ov_tensor_legacy_name(out_tensor).empty()) {
        ov::descriptor::set_ov_tensor_legacy_name(out_tensor, new_name);
    }
    if (!replacement_output_node.get_names().empty()) {
        out_tensor.set_names(replacement_output_node.get_names());
    }
    NGRAPH_SUPPRESS_DEPRECATED_END
}

auto snippets::op::Subgraph::constant_input_should_be_inside_body(const std::shared_ptr<ov::Node>& node) -> bool {
    return ov::is_type<ov::op::v1::Transpose>(node) ||
           ov::is_type<ov::op::v1::Broadcast>(node) ||
           ov::is_type<ov::op::v3::Broadcast>(node) ||
           ov::is_type<ov::op::v1::Reshape>(node);
}

///
/// \brief  Canonization transforms original subgraph and to canonical form suitable for code generation. In particular,
///         it handles supported layout conversions, broadcasts inputs and outputs to a single rank and layout. Canonicalization
///         returns master-shape (max rank + max dimensions over all outputs) that can be used for scheduling.
///         Canonicalization currently supports only the following layout conversions:
///             * None: all inputs have the same layout
///             * Planar + blocked: some inputs have blocked, and some have planar layouts, e.g. <N, C, H, W, c> + <N, C, H, W>
///         Also there is precision aligning inside body of subgraph during canonicalization
ov::PartialShape snippets::op::Subgraph::canonicalize(const BlockedShapeVector& outputShapes,
                                                      const BlockedShapeVector& inputShapes) {
    INTERNAL_OP_SCOPE(Subgraph);
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::canonicalize")
    NODE_VALIDATION_CHECK(this, inputShapes.size() == body_ptr()->get_parameters().size(),
        "Number of parameters for snippet doesn't match passed to generate method: ", inputShapes.size(), " vs ", body_ptr()->get_parameters().size(), ".");

    NODE_VALIDATION_CHECK(this, outputShapes.size() == body_ptr()->get_results().size(),
        "number of results for snippet doesn't match passed to generate method: ", outputShapes.size(), " vs ", body_ptr()->get_results().size(), ".");

    auto getMaxRankBlockedShape = [](const BlockedShapeVector& blockedShapes) -> const BlockedShape& {
        return *std::max_element(blockedShapes.begin(), blockedShapes.end(),
                         [&](const BlockedShape& lhs, const BlockedShape& rhs) {
                            return std::get<0>(lhs).size() < std::get<0>(rhs).size();
                         });
    };
    PartialShape baseShape;
    AxisVector baseOrder;
    std::tie(baseShape, baseOrder, std::ignore) = getMaxRankBlockedShape(inputShapes);
    const auto baseRank = baseShape.size();
    const bool baseIsBlocked = baseOrder.size() != std::set<size_t>(baseOrder.begin(), baseOrder.end()).size();
    for (size_t i = 0; i < inputShapes.size(); i++) {
        const auto &blockedShape = inputShapes[i];
        PartialShape inShape;
        AxisVector inOrder;
        element::Type inType;
        std::tie(inShape, inOrder, inType) = blockedShape;
        const auto inRank = inShape.size();
        NODE_VALIDATION_CHECK(this, inRank <= baseRank, "Input rank can't be larger than output rank in snippets.");
        if (inRank < baseRank) {
            PartialShape newShape(ov::Shape(baseRank, 1));
            // todo: more complicated logics is needed if we want to merge smth else than blocked and planar
            if (baseIsBlocked) {
                const bool inIsNotBlocked = inOrder.size() == std::set<size_t>(inOrder.begin(), inOrder.end()).size();
                NODE_VALIDATION_CHECK(this, inIsNotBlocked, "Snippets don't support conversion between blocked layouts of different ranks");
                inShape.insert(inShape.end(), ov::Dimension(1));
            }
            NODE_VALIDATION_CHECK(this, PartialShape::broadcast_merge_into(newShape, inShape, ov::op::AutoBroadcastType::NUMPY),
                                  "Failed to broadcast_merge inputs in snippets canonicalization");
            inShape = std::move(newShape);
        } else {
            // todo: 4d blocked + 5d planar layouts are not supported: <N, C, H, W, c> + <N, C, D, H, W>
            NODE_VALIDATION_CHECK(this,
                                  equal(baseOrder.begin(), baseOrder.end(), inOrder.begin()),
                                  "Snippets canonicalization got input shapes of equal ranks but different layouts, which is not supported");
        }
        ov::PartialShape tmpPShape(baseShape);
        // todo: we need to generalize canonicalization for domain-sensitive ops. E.g. MatMul inputs can't be broadcasted one to another
        if (!config.m_has_domain_sensitive_ops)
            NODE_VALIDATION_CHECK(this,
                                  PartialShape::broadcast_merge_into(tmpPShape, inShape, ::ngraph::op::AutoBroadcastType::NUMPY),
                                  "Failed to create broadcastable shapes in snippets canonicalization");
        const auto paramShape = body_ptr()->get_parameters()[i]->get_partial_shape();
        const auto paramType =  body_ptr()->get_parameters()[i]->get_element_type();
        if (paramShape.size() != inShape.size() || !equal(paramShape.begin(), paramShape.end(), inShape.begin()))
                body_ptr()->replace_parameter(i, std::make_shared<opset1::Parameter>(paramType, inShape));
    }
    body_ptr()->validate_nodes_and_infer_types();
    auto skipStartEndOnes = [](const PartialShape& shape) {
        auto begin = shape.begin();
        auto end = shape.end();
        while (begin != end && *begin == 1)
            begin++;
        while (begin != end && *(end-1) == 1)
            end--;

        PartialShape trimmedShape(std::vector<ov::Dimension> (end - begin, 1));
        std::copy(begin, end, trimmedShape.begin());
        return trimmedShape;
    };

    // Check that output shapes are broadcastable => can be scheduled
    const auto& body_results = body_ptr()->get_results();
    PartialShape outPShape = body_results[0]->get_input_partial_shape(0);
    // todo: we need a slightly more general approach for backward ROI propagation
    const auto& result_parent = body_results[0]->get_input_node_shared_ptr(0);
    if (body_results.size() == 1 &&
        ov::is_type<opset1::Transpose>(result_parent) &&
        ov::is_type<opset1::MatMul>(result_parent->get_input_node_shared_ptr(0))) {
        outPShape = result_parent->get_input_partial_shape(0);
    } else {
        for (size_t i = 0; i < body_results.size(); i++) {
            auto shape_i = body_results[i]->get_input_partial_shape(0);
            auto outputShape_i = std::get<0>(outputShapes[i]);
            // Check that the produced output shape corresponds to the passed shape
            // Some produced shapes may have been changed to be broadcastable (e.g. blocked + planar outputs),
            // so we need to remove leading and trailing "1" before the comparison
            PartialShape pShape_i(skipStartEndOnes(shape_i));
            bool compatibleWithPassedShape = PartialShape::broadcast_merge_into(pShape_i,
                                                                                skipStartEndOnes(outputShape_i),
                                                                                ::ngraph::op::AutoBroadcastType::NUMPY);
            NODE_VALIDATION_CHECK(this, compatibleWithPassedShape,
                                  "Inferred and passed results shapes are incompatible for snippet ");
            // Check that output shapes are broadcastable to each other => can be scheduled
            bool compatibleWithOtherOutputs = PartialShape::broadcast_merge_into(outPShape, shape_i,
                                                                                 ::ngraph::op::AutoBroadcastType::NUMPY);
            NODE_VALIDATION_CHECK(this, compatibleWithOtherOutputs,
                                  "Snippets output shapes must be numpy broadcastable");
        }
    }

    // We should insert Converts after Parameters and Constant and before Results
    // to align precision inside Subgraph body that is supported by Plugin
    align_element_types(outputShapes, inputShapes);

    master_shape = outPShape;
    return master_shape;
}

void snippets::op::Subgraph::align_element_types(const BlockedShapeVector& outputShapes,
                                                 const BlockedShapeVector& inputShapes) {
    // We should insert Convert before Results to set original output element type if needed
    const auto& body_results = body_ptr()->get_results();
    for (size_t i = 0; i < outputShapes.size(); i++) {
        const auto needed_out_type = std::get<2>(outputShapes[i]);
        if (body_results[i]->get_input_element_type(0) != needed_out_type) {
            const auto convert = std::make_shared<ngraph::snippets::op::ConvertSaturation>(
                body_results[i]->get_input_node_shared_ptr(0), needed_out_type);
            body_results[i]->set_argument(0, convert);
        }
    }

    // We should change existing element type to original for Parameters if needed
    const auto& body_parameters = body_ptr()->get_parameters();
    for (size_t i = 0; i < inputShapes.size(); ++i) {
        const auto needed_in_type = std::get<2>(inputShapes[i]);
        if (body_parameters[i]->get_element_type() != needed_in_type) {
            body_parameters[i]->set_element_type(needed_in_type);
            config.m_is_needed_to_align_precision = true;
        }
    }

    // We should align element type inside body using the corresponding pass:
    //  - Insert Convert before operations that doesn't support original element type for execution
    //  - Insert reverse Convert before operations that support original element type
    //    but have inputs that doesn't support it (because before them will be inserted Convert with exec_type - first point)
    //  - Then we should use ConstantFolding pass to convert element type of Scalars before inference.
    //  - Eliminate redundant Converts which can be inserted in AlignElementType() pass
    ngraph::pass::Manager manager;
    if (config.m_is_needed_to_align_precision) {
        manager.register_pass<snippets::pass::AlignElementType>(execution_element_type);
        manager.register_pass<ngraph::pass::ConstantFolding>();
        // TODO [100041] : In some cases AlignElementType pass can insert extra Convert because
        //                 the pass doesn't know real precisions in real time.
        //                 We call EliminateConverts pass to remove them
        manager.register_pass<ngraph::pass::EliminateConvert>();
    }
    manager.run_passes(body_ptr());
}

void snippets::op::Subgraph::initialize_buffer_scratchpad_size() {
    auto is_transpose_loop = [](const ov::Output<ov::Node>& source_output) -> bool {
        const auto parent = source_output.get_node_shared_ptr();
        // Transpose op is decomposed into LoopBegin->LoadReshape->Store->LoopEnd subgraph. LoadReshape op can be only
        // in Transpose decomposition. So it's enough to verify that this Loop is Transpose pattern.
        // We cannot check for non-equality of input and output shape of Transpose Loop because Transpose may have the same
        // shapes on input and output.
        auto loop_end = ov::as_type_ptr<op::LoopEnd>(parent);
        if (!loop_end)
            return false;
        size_t idx = source_output.get_index();
        while (ov::is_type<op::LoopEnd>(loop_end->get_input_node_shared_ptr(idx))) {
            auto consumer = loop_end->input_value(idx);
            idx = consumer.get_index();
            loop_end = ov::as_type_ptr<op::LoopEnd>(consumer.get_node_shared_ptr());
        }

        const auto loop_begin = loop_end->get_loop_begin();
        // At the moment Transpose Loops cannot be fused with other Loops, so check for one input and one output is enough
        if (loop_begin->get_input_size() != 1 || loop_end->get_output_size() != 1 || loop_begin->get_output_target_inputs(0).size() != 1)
            return false;
        const auto consumer = loop_begin->get_output_target_inputs(0).begin()->get_node();
        return ov::is_type<op::LoadReshape>(consumer);
    };
    auto propagate_offset = [](const std::shared_ptr<ngraph::snippets::op::Buffer>& buffer, const size_t offset) {
        // If Buffer has offset We set this offset in the next Load and Store ops
        // to correctly read and write data because all buffers have the one register
        // Also if user sets offset to a Buffer It means that the Buffer has the corresponding Load and Store ops

        // Propagate to up: in Store. Buffer can have only one Store
        {
            auto parent = buffer->get_input_node_shared_ptr(0);
            auto idx = buffer->input(0).get_source_output().get_index();
            // There may be graph with several LoopBegin and LoopEnd between Store/Brgemm and Buffer,
            // so we should iterate through LoopBase
            while (ov::is_type<snippets::op::LoopBase>(parent)) {
                const auto source_output = parent->input_value(idx);
                parent = source_output.get_node_shared_ptr();
                idx = source_output.get_index();
            }
            if (auto store = ov::as_type_ptr<snippets::op::Store>(parent)) {
                store->set_offset(offset);
            } else if (const auto brgemm = ov::as_type_ptr<snippets::op::Brgemm>(parent)) {
                // Brgemm encapsulates work with loading and storing of data
                brgemm->set_offset_c(offset);
            } else {
                throw ngraph_error("Buffer::set_offset() was called when Buffer didn't have the corresponding Store op for offset propagation");
            }
        }

        // Propagate to down: in Load. Buffer can have several Load and Loops after himself. We should go through all target inputs
        {
            std::function<void(const Input<Node>&)> propagate_down;
            propagate_down = [&](const Input<Node>& target_input) {
                const auto child = target_input.get_node()->shared_from_this();
                // There may be graph with several LoopBegin and LoopEnd between Load/Brgemm and Buffer,
                // so we should iterate through LoopBase
                // Example: Softmax decomposition with ReduceMax
                if (ov::is_type<snippets::op::LoopBase>(child)) {
                    const auto index = target_input.get_index();
                    for (const auto loop_target_output : child->output(index).get_target_inputs()) {
                        propagate_down(loop_target_output);
                    }
                } else if (const auto load = ov::as_type_ptr<snippets::op::Load>(child)) {
                    load->set_offset(offset);
                } else if (const auto brgemm = ov::as_type_ptr<snippets::op::Brgemm>(child)) {
                    // Brgemm encapsulates work with loading and storing of data
                    if (target_input.get_index() == 0) {
                        brgemm->set_offset_a(offset);
                    } else if (target_input.get_index() == 1) {
                        brgemm->set_offset_b(offset);
                    }
                } else {
                    throw ngraph_error("Buffer::set_offset() was called when Buffer didn't have the corresponding Load op for offset propagation");
                }
            };

            for (const auto target_output : buffer->output(0).get_target_inputs()) {
                propagate_down(target_output);
            }
        }
    };
    m_buffer_scratchpad = 0;
    size_t offset = 0;
    const auto ops = body_ptr()->get_ordered_ops();
    for (const auto& op : ops) {
        if (const auto buffer = ov::as_type_ptr<ngraph::snippets::op::Buffer>(op)) {
            const auto buffer_size = buffer->get_byte_size();
            // We need to allocate memory for first buffer at least
            if (m_buffer_scratchpad == 0) {
                m_buffer_scratchpad += buffer_size;
                continue;
            }

            // Transpose and MatMul ops should have different memories on inputs and outputs to avoid data corruption,
            // so after them, we should allocate new memory. Other operations (Eltwises, Convert) can be executed inplace.
            const auto parent = buffer->get_input_node_shared_ptr(0);
            if (ov::is_type<op::Brgemm>(parent) || is_transpose_loop(parent)) {
                offset = m_buffer_scratchpad;
                propagate_offset(buffer, offset);
                m_buffer_scratchpad += buffer_size;
                continue;
            }

            // If Buffer op requires memory size more that has been already allocated,
            // we increase current memory size to the needed size
            // For example, it's possible when we have a sequence of Eltwise ops with broadcasting
            const auto current_allocated_memory_size = m_buffer_scratchpad - offset;
            if (buffer_size > current_allocated_memory_size) {
                m_buffer_scratchpad += (buffer_size - current_allocated_memory_size);
                // Note: we don't update offset because we just add memory to needed size
            }

            propagate_offset(buffer, offset);
        }
    }
}

void snippets::op::Subgraph::convert_to_snippet_dialect() {
    INTERNAL_OP_SCOPE(Subgraph);
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::convert_to_snippet_dialect")
    auto skip_matching_domain = [](const std::shared_ptr<const ov::Node>& n) -> bool {
        const auto& pshape = n->get_input_partial_shape(0);
        const auto& last_dim = pshape[pshape.size() - 1];
        return last_dim.is_dynamic() || last_dim.get_length() != 1;
    };

    // At the moment we support only full vector Load/Store and scalar Load/Store so that count is equal to lanes.
    // Then we are going to support variadic Load/Store with different element count
    const size_t count = m_generator->get_target_machine()->get_lanes();
    const auto & params = body_ptr()->get_parameters();

    bool inputs_has_dynamic_last_dims = std::any_of(params.begin(), params.end(),
                                                    [](const shared_ptr<ngraph::op::Parameter>& p){
                                                        return p->get_partial_shape().rbegin()->is_dynamic();
                                                    });
    ngraph::pass::Manager manager;
    if (config.m_has_domain_sensitive_ops) {
        manager.register_pass<snippets::pass::MatMulToBrgemm>();
        manager.register_pass<snippets::pass::FuseTransposeBrgemm>();
        manager.register_pass<snippets::pass::InsertBuffer>(tileRank);
        manager.register_pass<snippets::pass::SoftmaxDecomposition>(count, tileRank);
        manager.register_pass<snippets::pass::TransposeDecomposition>();
    }
    manager.register_pass<snippets::pass::BroadcastToMoveBroadcast>();
    manager.register_pass<snippets::pass::ConvertConstantsToScalars>();
    manager.register_pass<snippets::pass::ConvertPowerToPowerStatic>();
    manager.register_pass<snippets::pass::InsertLoad>(count);
    manager.register_pass<snippets::pass::InsertStore>(count);
    // todo: presently dynamic pipeline is activated even if the last two dimension are static
    //  In general, we can use static kernels in this case, but several parameters (src and dst memory pointers for example)
    //  should be passed as run-time args, so it's a mixed mode: kernel is shape-aware, but some additional runtime args are required
    // Presently Broadcasting is organized in the following way:
    // * ALL last dims are static => broadcasting is handled via MoveBroadcast and pointer arithmetics (even for dynamic upper dims)
    if (!inputs_has_dynamic_last_dims) {
        manager.register_pass<snippets::pass::InsertMoveBroadcast>();
        manager.register_pass<snippets::pass::LoadMoveBroadcastToBroadcastLoad>();
        // Note that, BrodacastMove is typically inserted right after the Load. Such cases are typical for
        // simple subgraphs where one of the ngraph::op's inputs is broadcasted to match the larger one. However, BroadcastMove
        // could also be inserted after the ngraph::op, if the op input don't need broadcasting, but the output does
        // (for example, to match the larger output of a child node). In such cases, Loads (and Stores) should be replaced
        // with ScalarLoads (ScalarStores) to avoid invalid read in vector Loop. Graph example:
        // Parameter_0    Parameter_1        Parameter_2
        // [1,2,5,16]      [1,2,5,1]          [1,2,5,1]
        //   Load        BroadcastLoad         Load*       Scalar
        //          Add                             Subtract
        //            \___________     ___________BroadcastMove
        //                        \   /
        //                       Multiply
        //                         Store
        //                        Result
        // Note: Load* should be replaced with ScalarLoad in this example to avoid invalid read in vector Loop.
        if (master_shape.size() != 0 && master_shape[master_shape.size() - 1] != 1) {
            manager.register_pass<snippets::pass::SetScalarCountForLoad>();
            manager.register_pass<snippets::pass::SetScalarCountForStore>();
            manager.get_pass_config()->
                    set_callback<ngraph::snippets::pass::SetScalarCountForLoad>(skip_matching_domain);
            manager.get_pass_config()->
                    set_callback<ngraph::snippets::pass::SetScalarCountForStore>(skip_matching_domain);
        }
        // Note that InsertLoops requires validate_and_infer_types afterwards, so add it manually if
        // automatic validation will be disabled in the pass manager
        manager.register_pass<snippets::pass::InsertLoops>(master_shape, tileRank,
            m_generator->get_target_machine()->get_lanes(), !config.m_explicit_loop_insertion);
        if (config.m_has_domain_sensitive_ops) {
            manager.register_pass<snippets::pass::LoopFusion>();
            manager.register_pass<snippets::pass::ResetBufferState>();
        }
    }
    manager.run_passes(body_ptr());
}

snippets::Schedule snippets::op::Subgraph::generate(const BlockedShapeVector& output_shapes,
                                                    const BlockedShapeVector& input_shapes,
                                                    const void* compile_params) {
    canonicalize(output_shapes, input_shapes);
    return generate(compile_params);
}

snippets::Schedule snippets::op::Subgraph::generate(const BlockedShapeVector& output_shapes,
                                                    const BlockedShapeVector& input_shapes,
                                                    ngraph::pass::Manager& opt,
                                                    const void* compile_params) {
    canonicalize(output_shapes, input_shapes);
    return generate(opt, compile_params);
}

snippets::Schedule snippets::op::Subgraph::generate(const void* compile_params) {
    auto mngr = ngraph::pass::Manager();
    return generate(mngr, compile_params);
}

snippets::Schedule snippets::op::Subgraph::generate(ngraph::pass::Manager& opt, const void* compile_params) {
    INTERNAL_OP_SCOPE(Subgraph);
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::generate")
    NGRAPH_CHECK(m_generator != nullptr, "generate is called while generator is not set");

    convert_to_snippet_dialect();
    opt.run_passes(body_ptr());

    // After all passes, when all optimizations are completed and all MemoryAccess ops are inserted,
    // we can calculate common buffer scratchpad size and propagate offset from Buffer to the corresponding MemoryAccess ops
    if (config.m_has_domain_sensitive_ops)
        initialize_buffer_scratchpad_size();

    snippets::pass::AssignRegisters().run_on_model(body_ptr());

    const auto ops = body_ptr()->get_ops();
    ngraph::snippets::Generator::GeneratorConfig generatorConfig;
    generatorConfig.m_save_lowered_code = config.m_has_domain_sensitive_ops;
    generatorConfig.m_need_fill_tail_register = config.m_has_domain_sensitive_ops;
    generatorConfig.m_optimize_single_evaluation = std::none_of(ops.begin(), ops.end(), [](const std::shared_ptr<ov::Node>& op) {
        return ov::is_type<ngraph::snippets::op::Buffer>(op);
    });

    // actual code emission
    ngraph::snippets::code ptr = m_generator->generate(body_ptr(), generatorConfig, compile_params);

    return {master_shape, false /*canBeLinearized*/, ptr};
}

void snippets::op::Subgraph::print() const {
    INTERNAL_OP_SCOPE(Subgraph);
    remark(13) << "subgraph " << this->get_friendly_name() << " "
        << this->get_type_name()
        << " which contains " << body_ptr()->get_ops().size() << " nodes" << std::endl;

    int qqq = 0;
    for (auto op : body_ptr()->get_ordered_ops()) {
        remark(13) << "op " << qqq++ << " " << op->get_friendly_name() << " (" << op->get_type_name() << ") " << op << std::endl;
    }

    for (auto& in : this->inputs()) {
        remark(13) << "  -> " << in.get_source_output().get_node_shared_ptr()->get_friendly_name() << " "
            << in.get_source_output().get_node_shared_ptr() << std::endl;
    }

    for (auto& out : this->outputs()) {
        for (auto& user : out.get_target_inputs()) {
            remark(13) << " <- " << user.get_node()->get_friendly_name() << " "  << user.get_node() << std::endl;
        }
        remark(13) << std::endl;
    }
}

void snippets::op::Subgraph::print_statistics(bool verbose) {
    INTERNAL_OP_SCOPE(Subgraph);
    auto getNodeInventory = [](std::shared_ptr<ov::Node> n) -> size_t {
        size_t total = 0;

        for (auto input : n->inputs()) {
            total += input.get_tensor().size();
        }

        for (auto output : n->outputs()) {
            total += output.get_tensor().size();
        }

        if (auto subgraph = ngraph::as_type_ptr<op::Subgraph>(n)) {
            for (auto op : subgraph->body_ptr()->get_ordered_ops()) {
                if (ngraph::as_type_ptr<ngraph::opset1::Constant>(op)) {
                    total += op->output(0).get_tensor().size();
                }
            }
        }

        return total;
    };

    auto getModelInventory = [getNodeInventory](const ov::Model & f) -> size_t {
        size_t total = 0;
        for (auto op : f.get_ordered_ops()) {
            // Results and parameters are artificially introduced,
            // while Constants are already considered if they are inputs of other operation
            // this should lead to 1:1 inventory for single node operations
            if (!ngraph::as_type_ptr<ngraph::opset1::Parameter>(op)
             && !ngraph::as_type_ptr<ngraph::opset1::Result>(op)
             && !ngraph::as_type_ptr<ngraph::opset1::Constant>(op)) {
                total += getNodeInventory(op);
            }
        }
        return total;
    };

    auto countConstants = [](const ov::Model & f) -> size_t {
        size_t count = 0;
        for (auto op : f.get_ordered_ops()) {
            count += !!ngraph::as_type_ptr<ngraph::opset1::Constant>(op) ? 1 : 0;
        }
        return count;
    };

    std::cout << get_friendly_name()
                << ";" << this
                << ";" << body_ptr()->get_ops().size()
                << ";" << body_ptr()->get_parameters().size()
                << ";" << body_ptr()->get_results().size()
                << ";" << countConstants(body())
                << ";" << getModelInventory(body())
                << ";" << getNodeInventory(shared_from_this()) << std::endl;

    if (verbose) {
        this->print();
    }
}

void snippets::op::Subgraph::serialize() const {
    std::stringstream xmlFile, binFile;
    ov::pass::Serialize serializer(xmlFile, xmlFile, ov::pass::Serialize::Version::IR_V10);
    serializer.run_on_model(body_ptr());
    auto m_constants = binFile.str();
    auto m_model = xmlFile.str();
    std::cout << m_model << std::endl;
}
