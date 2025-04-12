// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"
#include "snippets/remarks.hpp"

#include "snippets/op/subgraph.hpp"

#include "snippets/pass/broadcast_to_movebroadcast.hpp"
#include "snippets/pass/propagate_precision.hpp"
#include "snippets/pass/convert_constants.hpp"
#include "snippets/pass/convert_power_to_powerstatic.hpp"
#include "snippets/pass/transpose_decomposition.hpp"
#include "snippets/pass/softmax_decomposition.hpp"
#include "snippets/pass/matmul_to_brgemm.hpp"
#include "snippets/pass/fuse_transpose_brgemm.hpp"
#include "snippets/pass/canonicalization.hpp"
#include "snippets/pass/align_element_types.hpp"
#include "snippets/pass/reduce_to_snippets_reduce.hpp"
#include "snippets/pass/gn_decomposition.hpp"

#include "snippets/runtime_configurator.hpp"
#include "snippets/utils/utils.hpp"

#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/linear_ir_builder.hpp"
#include "snippets/lowered/pass/mark_loops.hpp"
#include "snippets/lowered/pass/split_loops.hpp"
#include "snippets/lowered/pass/fuse_loops.hpp"
#include "snippets/lowered/pass/init_loops.hpp"
#include "snippets/lowered/pass/insert_buffers.hpp"
#include "snippets/lowered/pass/insert_load_store.hpp"
#include "snippets/lowered/pass/insert_broadcastmove.hpp"
#include "snippets/lowered/pass/load_movebroadcast_to_broadcastload.hpp"
#include "snippets/lowered/pass/allocate_buffers.hpp"
#include "snippets/lowered/pass/move_scalar_to_consumer.hpp"
#include "snippets/lowered/pass/move_result_out_of_loop.hpp"
#include "snippets/lowered/pass/clean_repeated_ptr_shifts.hpp"
#include "snippets/lowered/pass/validate_unified_loops.hpp"
#include "snippets/lowered/pass/insert_loops.hpp"
#include "snippets/lowered/pass/optimize_domain.hpp"
#include "snippets/lowered/pass/insert_perf_count.hpp"
#include "snippets/lowered/pass/validate_shapes.hpp"
#include "snippets/lowered/pass/validate_buffers.hpp"
#include "snippets/lowered/pass/validate.hpp"
#include "snippets/lowered/pass/pass_config.hpp"
#include "snippets/lowered/pass/reduce_decomposition.hpp"
#include "snippets/lowered/pass/cleanup_loop_offsets.hpp"
#include "snippets/lowered/pass/insert_specific_iterations.hpp"
#include "snippets/lowered/pass/optimize_loop_single_evaluation.hpp"
#include "snippets/lowered/pass/normalize_loop_ids.hpp"
#include "snippets/lowered/pass/validate_expanded_loops.hpp"
#include "snippets/lowered/pass/set_load_store_scalar.hpp"
#include "snippets/lowered/pass/extract_loop_invariants.hpp"
#include "snippets/lowered/pass/set_dynamic_wa_to_outermost_loop.hpp"

#include "snippets/lowered/pass/init_registers.hpp"

#include "transformations/utils/utils.hpp"

#include "snippets/pass/manager.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "ov_ops/type_relaxed.hpp"

#include <algorithm>
#include <memory>
#include <array>

using namespace std;
using namespace ov::op::util;

namespace ov {
namespace snippets {
namespace op {

void Subgraph::set_generator(std::shared_ptr<ov::snippets::Generator> generator) {
    m_generator = std::move(generator);
}

void Subgraph::set_virtual_port_count(const size_t count) {
    m_virtual_port_count = count;
}

auto Subgraph::is_domain_sensitive_op(const std::shared_ptr<ov::Node>& op) -> bool {
    // Broadcast is domain sensetive op because the output shape depends on
    // the both input and broadcast shapes (the both - are inputs of op). Note: is used only in MHA pattern
    return ov::is_type_any_of<ov::op::v1::Transpose,
                              ov::op::v1::Softmax,
                              ov::op::v8::Softmax,
                              ov::op::v0::MatMul,
                              ov::op::v1::Broadcast,
                              ov::op::v3::Broadcast,
                              ov::op::v12::GroupNormalization,
                              op::Reshape>(op);
}

auto Subgraph::is_shape_infer_op(const std::shared_ptr<ov::Node>& op) -> bool {
    return ov::is_type<ov::snippets::op::ShapeInferOp>(op);
}

void Subgraph::init_config() {
    auto update = [](bool& flag, bool status) { flag = flag || status; };
    const auto ops = body_ptr()->get_ops();
    for (const auto& op : ops) {
        update(config.m_is_quantized, ov::is_type<ov::op::v0::FakeQuantize>(op));
        update(config.m_has_domain_sensitive_ops, is_domain_sensitive_op(op));
        update(config.m_has_broadcast_sensitive_ops, ov::is_type_any_of<ov::op::v12::GroupNormalization, op::Reshape>(op));
    }
}

auto Subgraph::get_estimated_buffer_count(const ov::NodeVector& ops) -> size_t {
    // The count of potential unique Buffers - it's hidden virtual ports as well
    // We should go through Subgraph and calculate potential non-inplace Buffers count.
    // These Buffers can be in 2 cases:
    // 1. Around Loops: we should check for element type size of nodes which use Buffer to get rating from above for unique Buffer count.
    // 2. Around MatMul: all buffers around Matmul must not be inplace because MatMul blocking implementation changes registers during computations.
    // The count is estimated because when we calculate this number, we have only original graph representation
    // and where will be Loops - we can just predict.
    // Note: The ops that create Buffers: MatMul, Transpose and Softmax (always FP32)
    std::vector<size_t> used_precision_size;

    auto push_prc_size = [&used_precision_size](size_t precision_size) {
        if (used_precision_size.empty() || used_precision_size.back() != precision_size) {
            used_precision_size.push_back(precision_size);
        }
    };

    for (const auto& op : ops) {
        if (const auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(op)) {
            // At the moment Transposes are supported only on Results and Parameters, but
            // then we should have the different Buffers for Transpose as well (Transpose isn't inplace)
            const auto consumers = transpose->get_output_target_inputs(0);
            // If after Transpose there is Result it means that there won't be Buffer after Transpose.
            // The same case is for Parameter before Transpose
            const auto are_prev_or_next_ops = std::none_of(consumers.begin(), consumers.end(),
                                                           [](const ov::Input<ov::Node>& in) {
                                                               return ov::is_type<ov::op::v0::Result>(in.get_node());
                                                           }) ||
                                              !ov::is_type<ov::op::v0::Parameter>(transpose->get_input_node_shared_ptr(0));
            if (are_prev_or_next_ops) {
                push_prc_size(transpose->get_element_type().size());
            }
        } else if (ov::is_type_any_of<ov::op::v1::Softmax, ov::op::v8::Softmax>(op)) {
            // Softmax always uses 2 FP32 Buffers after decomposition.
            // They are inplace and the same, so we can push precision size only once
            push_prc_size(ov::element::f32.size());
        } else if (const auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(op)) {
            // Since all buffers around Matmul must be unique, we explicitely add values to the vector without any checks
            if (!ov::is_type<ov::op::v0::Parameter>(matmul->get_input_node_shared_ptr(0)))
                used_precision_size.push_back(matmul->get_input_element_type(0).size());
            if (!ov::is_type<ov::op::v0::Parameter>(matmul->get_input_node_shared_ptr(1)))
                used_precision_size.push_back(matmul->get_input_element_type(1).size());

            const auto consumers = matmul->get_output_target_inputs(0);
            if (std::none_of(consumers.begin(), consumers.end(),
                             [](const ov::Input<ov::Node>& in) {
                                 return ov::is_type<ov::op::v0::Result>(in.get_node());
                             })) {
                used_precision_size.push_back(matmul->get_element_type().size());
            }
        }
    }

    return used_precision_size.size();
}

Subgraph::Subgraph(const OutputVector& args, const std::shared_ptr<ov::Model>& body)
        : SubGraphOp(args), m_generator(nullptr) {
    SubGraphOp::set_function(body);
    init_config();
    constructor_validate_and_infer_types();
    for (size_t i = 0; i < body->get_parameters().size(); ++i)
        m_input_descriptions[0].push_back(std::make_shared<InvariantInputDescription>(i, i));
    for (size_t i = 0; i < body->get_output_size(); ++i)
        m_output_descriptions[0].push_back(std::make_shared<BodyOutputDescription>(i, i));
    m_transformations_allowed = false;
    m_shape_infer = std::make_shared<OVShapeInfer>(body);
}

Subgraph::Subgraph(const NodeVector& args, const std::shared_ptr<ov::Model>& body)
        : Subgraph(as_output_vector(args), body) {}

std::shared_ptr<Node> Subgraph::clone_with_new_inputs(const OutputVector& inputs) const {
    INTERNAL_OP_SCOPE(Subgraph);
    return make_shared<Subgraph>(inputs, body().clone());
}

void Subgraph::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(Subgraph);
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::validate_and_infer_types")
    ov::ParameterVector old_parameters;
    for (const auto& op : body_ptr()->get_parameters()) {
        old_parameters.push_back(op);
    }

    for (size_t i = 0; i < get_input_size(); ++i) {
        body_ptr()->replace_parameter(i, std::make_shared<ov::op::v0::Parameter>(get_input_element_type(i), get_input_partial_shape(i)));
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

bool Subgraph::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("body", body_ptr());
    visitor.on_attribute("input_descriptions", m_input_descriptions[0]);
    visitor.on_attribute("output_descriptions", m_output_descriptions[0]);
    return true;
}

auto Subgraph::wrap_node_as_subgraph(const std::shared_ptr<ov::Node>& node) -> std::shared_ptr<op::Subgraph> {
    INTERNAL_OP_SCOPE(Subgraph);
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::wrap_node_as_subgraph")
    ov::ParameterVector body_parameters;
    ov::OutputVector body_inputs;

    ov::OutputVector subgraph_inputs;

    for (const auto& input : node->input_values()) {
        if (ov::is_type<ov::opset1::Constant>(input.get_node_shared_ptr()) &&
            (ov::shape_size(input.get_shape()) == 1 ||
             ov::is_type<ov::op::v0::FakeQuantize>(node) ||
             constant_input_should_be_inside_body(node))) {
            body_inputs.push_back(input);
        } else {
            auto parameter = std::make_shared<ov::opset1::Parameter>(input.get_element_type(), input.get_partial_shape());
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
        OPENVINO_THROW("original node outputs size and extracted subgraph node outputs size doesn't much");
    }

    ov::ResultVector body_results;
    for (const auto& output : node->outputs()) {
        body_results.push_back(std::make_shared<ov::opset1::Result>(body_node->output(output.get_index())));
    }

    auto body = create_body(node->get_friendly_name(), body_results, body_parameters);
    auto subgraph = build_subgraph(node, subgraph_inputs, body);

    size_t hidden_data_count = 0lu;
    if (auto fq_node = ov::as_type_ptr<ov::op::v0::FakeQuantize>(node)) {
        hidden_data_count += utils::get_non_scalar_constant_count_for_fq(fq_node);
    }
    subgraph->set_virtual_port_count(hidden_data_count);

    for (size_t i = 0; i < body->get_parameters().size(); i++) {
        body->get_parameters()[i]->set_friendly_name(body_parameters[i]->get_friendly_name());
    }

    if (subgraph->get_output_size() != body->get_results().size()) {
        OPENVINO_THROW("newly create subgraph doesn't much number of original node results");
    }

    return subgraph;
}

void Subgraph::fill_empty_output_names(const Output<Node>& target_output_node, const Output<Node>& replacement_output_node) {
    auto& out_tensor = target_output_node.get_tensor();
    if (!replacement_output_node.get_names().empty()) {
        out_tensor.set_names(replacement_output_node.get_names());
    }
}

auto Subgraph::constant_input_should_be_inside_body(const std::shared_ptr<ov::Node>& node) -> bool {
    return ov::is_type_any_of<ov::op::v1::Transpose, ov::op::v1::Broadcast, ov::op::v3::Broadcast, ov::op::v1::Reshape>(
        node);
}

bool Subgraph::check_broadcast(const std::shared_ptr<const ov::Node>& node) noexcept {
    const auto elementwise = ov::as_type_ptr<const ov::op::util::BinaryElementwiseArithmetic>(node);
    return
        (elementwise == nullptr) ||
        (elementwise->get_input_partial_shape(0).size() == elementwise->get_input_partial_shape(1).size()) ||
        (elementwise->get_autob().m_type != ov::op::AutoBroadcastType::PDPD);
}

IShapeInferSnippets::Result Subgraph::shape_infer(const std::vector<VectorDimsRef>& input_shapes) {
    OPENVINO_ASSERT(m_shape_infer, "Attempt to call shape_infer when it's not initialized");
    return m_shape_infer->infer(input_shapes);
}

Subgraph::OVShapeInfer::OVShapeInfer(const std::shared_ptr<ov::Model>& body) :
    m_ov_body(body) {
    OPENVINO_ASSERT(m_ov_body, "Can't initialize shape infer with empty body");
}

IShapeInferSnippets::Result Subgraph::OVShapeInfer::infer(const std::vector<VectorDimsRef>& input_shapes) {
    const ParameterVector& parameters = m_ov_body->get_parameters();
    const ResultVector& results = m_ov_body->get_results();
    OPENVINO_ASSERT(parameters.size() == input_shapes.size(), "Got invalid number of input shapes to reshape subgraph body");
    for (size_t i = 0; i < parameters.size(); ++i)
        parameters[i]->set_partial_shape(utils::vdims_to_pshape(input_shapes[i].get()));
    m_ov_body->validate_nodes_and_infer_types();
    std::vector<VectorDims> outputDims;
    for (const auto& res : results)
        outputDims.emplace_back(utils::pshape_to_vdims(res->get_input_partial_shape(0)));
    m_last_result = {outputDims, ShapeInferStatus::success};
    return m_last_result;
}

VectorDims Subgraph::infer_master_shape() {
    std::vector<VectorDims> output_dims;
    if (is_dynamic()) {
        // Note that in case of dynamic implementation shapeInfer() is called before PrepareParams,
        // so there must be last_result available
        // In principle, we can instantiate shape_infer here, but it's not an intended pipeline behavior.
        OPENVINO_ASSERT(m_shape_infer, "Can't calculate master_shape when shapeInfer is not initialized");
        output_dims = m_shape_infer->get_last_result().dims;
        OPENVINO_ASSERT(!output_dims.empty(), "Can't calculate master_shape before the first shape inference");
    } else {
        for (const auto& res : body_ptr()->get_results()) {
            const auto& shape_infer_leaf = utils::get_leaf_node_of_first_parent_shape_infer_seq(res);
            const auto& res_input = shape_infer_leaf ? shape_infer_leaf->input(0) : res->input(0);
            OPENVINO_ASSERT(res_input.get_partial_shape().is_static(), "Result have dynamic shape in static pipeline");
            // We need to account to the shape's layout stored in Output<Node> rt_info
            const auto& planar_shape = utils::get_preordered_pshape(res_input.get_source_output());
            output_dims.emplace_back(planar_shape.get_shape());
        }
    }

    if (output_dims.size() == 1)
        return output_dims.front();

    const auto& default_broadcasting = std::make_shared<NumpyBroadcastShapeInfer>();
    // Note: we have to convert vector<VectorDims> to vector<reference_wrapper<const VectorDims>>
    // because of shape inference interface
    std::vector<std::reference_wrapper<const VectorDims>> inputs;
    inputs.reserve(output_dims.size());
    for (const auto& d : output_dims)
        inputs.emplace_back(d);
    return default_broadcasting->infer(inputs).dims.front();
}

std::shared_ptr<lowered::LinearIR>
Subgraph::convert_body_to_linear_ir(size_t min_parallel_work_amount, size_t min_kernel_work_amount,
                                    const std::shared_ptr<IShapeInferSnippetsFactory>& shape_infer_factory) {
    lowered::Config lowering_config;
    lowering_config.m_need_fill_tail_register = config.m_has_domain_sensitive_ops;
    lowering_config.m_loop_depth = tile_rank;
    lowering_config.m_enable_domain_optimization = !config.m_has_domain_sensitive_ops;
    lowering_config.m_min_parallel_work_amount = min_parallel_work_amount;
    lowering_config.m_min_kernel_work_amount = min_kernel_work_amount;
#ifdef SNIPPETS_DEBUG_CAPS
    lowering_config.debug_config = config.m_debug_config;
    OPENVINO_ASSERT(lowering_config.debug_config, "Debug config is not initialized");
#endif  // SNIPPETS_DEBUG_CAPS

    m_linear_ir = std::make_shared<lowered::LinearIR>(body_ptr(), shape_infer_factory, lowering_config);
    m_shape_infer = m_linear_ir->get_shape_infer_instance();
    return m_linear_ir;
}

std::shared_ptr<Subgraph> Subgraph::clone() const {
    ov::OutputVector subgraph_node_inputs;
    for (const auto &input : input_values()) {
        auto new_input = std::make_shared<ov::opset1::Parameter>(input.get_element_type(), input.get_partial_shape());
        subgraph_node_inputs.push_back(new_input);
    }
    std::shared_ptr<ov::Model> new_body = body_ptr()->clone();
    auto result = std::make_shared<snippets::op::Subgraph>(subgraph_node_inputs, new_body);
    // Note: ov::copy_runtime_info accepts only shared_ptr<ov::Node> as "from" but never modifies it,
    // so we have to cast away constness to copy runtime info
    ov::copy_runtime_info(const_pointer_cast<Node>(shared_from_this()), result);
    result->set_friendly_name(get_friendly_name());
    if (m_linear_ir)
        result->m_linear_ir = lowered::LinearIRBuilder().clone(m_linear_ir);
    if (m_shape_infer_linear_ir)
        result->m_shape_infer_linear_ir = lowered::LinearIRBuilder().clone(m_shape_infer_linear_ir);
    // Note: we don't update shapeInfer here, since it's initialized in the constructor
    if (m_generator)
        result->m_generator = m_generator->clone();
    return result;
}

void Subgraph::data_flow_transformations(const BlockedShapeVector& blocked_input_shapes,
                                         const std::vector<ov::element::Type>& input_precisions,
                                         const std::vector<ov::element::Type>& output_precisions,
                                         const std::vector<snippets::pass::Manager::PositionedPassBase>& backend_passes) const {
    INTERNAL_OP_SCOPE(Subgraph);
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::data_flow_transformations")

    std::shared_ptr<ov::pass::PassConfig> pass_config = std::make_shared<ov::pass::PassConfig>();
    // If subgraph has its own specific canonicalization, which is different with common behavior, will skip the this common one.
    // for example in GN, scale and bias shape [c] are canonicalized to [1,c,1,1], not [1,1,1,c]. Common canonicalization is disabled in this case.
    if (blocked_input_shapes.empty() || config.m_has_broadcast_sensitive_ops)
        pass_config->disable<snippets::pass::Canonicalization>();
    if (input_precisions.empty() || output_precisions.empty())
        pass_config->disable<snippets::pass::AlignElementTypes>();

    ov::snippets::pass::Manager manager(pass_config, "SnippetsDataFlowManager");
    manager.register_pass<snippets::pass::Canonicalization>(blocked_input_shapes);
    manager.register_pass<snippets::pass::AlignElementTypes>(input_precisions, output_precisions);

    if (config.m_has_domain_sensitive_ops) {
        manager.register_pass<snippets::pass::MatMulToBrgemm>();
        manager.register_pass<snippets::pass::FuseTransposeBrgemm>();
        manager.register_pass<snippets::pass::TransposeDecomposition>();
        manager.register_pass<snippets::pass::SoftmaxDecomposition>();
        manager.register_pass<snippets::pass::GNDecomposition>();
    }
    manager.register_pass<snippets::pass::BroadcastToMoveBroadcast>();
    manager.register_pass<snippets::pass::ReduceToSnippetsReduce>();
    manager.register_pass<snippets::pass::ConvertConstantsToScalars>();
    manager.register_pass<snippets::pass::ConvertPowerToPowerStatic>();

    manager.register_pass<snippets::pass::PropagatePrecision>(m_generator->get_target_machine());
    manager.register_pass<ov::pass::ConstantFolding>();
    manager.register_pass<snippets::pass::ConvertConstantsToScalars>();

    manager.register_positioned_passes(backend_passes);
    manager.run_passes(body_ptr());
}

void Subgraph::control_flow_transformations(size_t min_parallel_work_amount, size_t min_kernel_work_amount,
                                            const std::shared_ptr<IShapeInferSnippetsFactory>& shape_infer_factory,
                                            const std::shared_ptr<lowered::pass::PassConfig>& lowered_pass_config,
                                            const std::vector<lowered::pass::PassPipeline::PositionedPassLowered>& lowered_backend_passes) {
    INTERNAL_OP_SCOPE(Subgraph);
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::control_flow_transformations")

    OV_ITT_TASK_CHAIN(CONTROL_FLOW, ov::pass::itt::domains::SnippetsTransform, "Snippets::op::control_flow_transformations", "::convert_body_to_linear_ir")

    convert_body_to_linear_ir(min_parallel_work_amount, min_kernel_work_amount, shape_infer_factory);
    OPENVINO_ASSERT(m_linear_ir, "LinearIR has not been inited for control flow transformations!");

    OV_ITT_TASK_NEXT(CONTROL_FLOW, "::control_flow_transformations")

    // Domain optimization must be the first pass, because all other transformations may depend on PortDescriptor shapes
    size_t loop_depth = m_linear_ir->get_config().m_loop_depth;
    if (!lowered_pass_config->is_disabled<lowered::pass::OptimizeDomain>()) {
        lowered::pass::OptimizeDomain(loop_depth).run(*m_linear_ir);
        m_linear_ir->set_loop_depth(loop_depth);
    }

    const size_t vector_size = get_generator()->get_target_machine()->get_lanes();

    lowered::pass::PassPipeline pipeline(lowered_pass_config);
    pipeline.register_pass<lowered::pass::MarkLoops>(vector_size);
    pipeline.register_pass<lowered::pass::ReduceDecomposition>(vector_size);
    pipeline.register_pass<lowered::pass::FuseLoops>();
    pipeline.register_pass<lowered::pass::SplitLoops>();
    pipeline.register_pass<lowered::pass::MoveResultOutOfLoop>();
    pipeline.register_pass<lowered::pass::InsertBuffers>();
    pipeline.register_pass<lowered::pass::InsertLoadStore>(vector_size);
    pipeline.register_pass<lowered::pass::MoveScalarToConsumer>();
    pipeline.register_pass<lowered::pass::InsertBroadcastMove>();
    pipeline.register_pass<lowered::pass::LoadMoveBroadcastToBroadcastLoad>();
    pipeline.register_pass<lowered::pass::ExtractLoopInvariants>();
    pipeline.register_pass<lowered::pass::ValidateShapes>();
    pipeline.register_pass<lowered::pass::ValidateUnifiedLoops>();
    pipeline.register_pass<lowered::pass::InitLoops>();
    pipeline.register_pass<lowered::pass::SetDynamicWAToOuterMostLoop>();
    pipeline.register_pass<lowered::pass::InsertLoops>();
    pipeline.register_pass<lowered::pass::AllocateBuffers>(m_linear_ir->get_config().m_are_buffers_optimized);
    pipeline.register_pass<lowered::pass::CleanRepeatedDataPointerShifts>();
    pipeline.register_positioned_passes(lowered_backend_passes);
    pipeline.run(*m_linear_ir);

    lowered::pass::PassPipeline validation_pipeline;
    validation_pipeline.register_pass<lowered::pass::ValidateBuffers>();
    validation_pipeline.register_pass<lowered::pass::Validate>();
    validation_pipeline.run(*m_linear_ir);

#ifdef SNIPPETS_DEBUG_CAPS
    if (m_linear_ir->get_config().debug_config->perf_count_mode != DebugCapsConfig::PerfCountMode::Disabled) {
        const std::map<std::string, std::string> bound_names = {};
        lowered::pass::InsertPerfCount perf_count_pass(bound_names);
        perf_count_pass.run(*m_linear_ir, m_linear_ir->cbegin(), m_linear_ir->cend());
    }
#endif

    OV_ITT_TASK_NEXT(CONTROL_FLOW, "::init_shape_infer_linear_ir")

    // After ControlFlow transformations we should to create LinearIR for ShapeInference - clone state of LinearIR before loop decomposition.
    const auto& cloning_config = lowered::LinearIRBuilder::Config(false);
    m_shape_infer_linear_ir = lowered::LinearIRBuilder(cloning_config).clone(m_linear_ir);
    OPENVINO_ASSERT(m_shape_infer_linear_ir, "LinearIR has not been successfully cloned!");
    m_shape_infer = m_shape_infer_linear_ir->get_shape_infer_instance();
    OPENVINO_ASSERT(m_shape_infer, "ShapeInference based on ShapeInferenceLinearIR has not been successfully created!");

    OV_ITT_TASK_NEXT(CONTROL_FLOW, "::pre_generation_pipeline")

    lowered::pass::PassPipeline gen_pipeline(lowered_pass_config);
    // Note: the order of all passes in this pipeline must not be changed since they have hard dependencies
    //    1. InsertSpecificIterations must be called after AssignRegisters since tail loop expressions must have the same
    //       assigned registers as the corresponding ops in the main body.
    //    2. CleanupLoopOffsets must be called after InsertSpecificIterations to avoid violating the proportionality of the pointer increments
    //       (this might happen if tail loop and main loop have different increments)
    //    3. OptimizeLoopSingleEvaluation must be called after CleanupLoopOffsets
    //       since CleanupLoopOffsets can't handle loops with evaluate_once = true

    gen_pipeline.register_pass<lowered::pass::InitRegisters>(get_generator(), lowered_pass_config);
    gen_pipeline.register_pass<lowered::pass::InsertSpecificIterations>();
    gen_pipeline.register_pass<lowered::pass::NormalizeLoopIDs>();
    gen_pipeline.register_pass<lowered::pass::ValidateExpandedLoops>();
    gen_pipeline.register_pass<lowered::pass::CleanupLoopOffsets>();
    gen_pipeline.register_pass<lowered::pass::OptimizeLoopSingleEvaluation>();
    gen_pipeline.run(*m_linear_ir);
}

snippets::Schedule Subgraph::generate(const BlockedShapeVector& blocked_input_shapes,
                                      const std::vector<ov::element::Type>& input_precisions,
                                      const std::vector<ov::element::Type>& output_precisions,
                                      const std::vector<snippets::pass::Manager::PositionedPassBase>& data_flow_backend_passes,
                                      const std::shared_ptr<lowered::pass::PassConfig>& lowered_pass_config,
                                      const std::vector<snippets::lowered::pass::PassPipeline::PositionedPassLowered>& lowered_backend_passes,
                                      size_t min_parallel_work_amount, size_t min_kernel_work_amount,
                                      const std::shared_ptr<IShapeInferSnippetsFactory>& factory,
                                      const void* compile_params) {
    data_flow_transformations(blocked_input_shapes, input_precisions, output_precisions, data_flow_backend_passes);
    control_flow_transformations(min_parallel_work_amount, min_kernel_work_amount, factory, lowered_pass_config, lowered_backend_passes);
    return generate(compile_params);
}

snippets::Schedule Subgraph::generate(const void* compile_params) const {
    INTERNAL_OP_SCOPE(Subgraph);
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::generate")
    OPENVINO_ASSERT(m_generator != nullptr, "generate is called while generator is not set");

    // actual code emission
    // Note: to not corrupt the lowered linear IR for the shape-dependent passes, we have to make a copy
    OPENVINO_ASSERT(m_linear_ir, "Attempt to call generate, when linear IR was not initialized");
    ov::snippets::lowered::ExpressionMap expression_map;
    const auto linear_ir = lowered::LinearIRBuilder().clone(m_linear_ir, expression_map);

    if (is_dynamic()) {
        ov::snippets::lowered::pass::PassPipeline shape_dependent_pipeline;
        shape_dependent_pipeline.register_pass<ov::snippets::lowered::pass::SetLoadStoreScalar>();
        shape_dependent_pipeline.register_pass<ov::snippets::lowered::pass::InsertBroadcastMove>();
        shape_dependent_pipeline.register_pass<ov::snippets::lowered::pass::LoadMoveBroadcastToBroadcastLoad>();
        shape_dependent_pipeline.run(*linear_ir);
    }

    auto lowering_result = m_generator->generate(linear_ir, compile_params);
    return {std::move(lowering_result)};
}

const std::shared_ptr<RuntimeConfigurator>& Subgraph::get_runtime_configurator() const {
    OPENVINO_ASSERT(m_generator, "Generator has not been inited!");
    return m_generator->get_target_machine()->get_runtime_configurator();
}

const std::shared_ptr<RuntimeConfig>& Subgraph::update_runtime_config() const {
    OPENVINO_ASSERT(m_linear_ir, "LoweredLinearIR has not been inited!");
    return get_runtime_configurator()->get_updated_config(m_linear_ir);
}

void Subgraph::print() const {
    INTERNAL_OP_SCOPE(Subgraph);
    remark(13) << "subgraph " << this->get_friendly_name() << " "
               << this->get_type_name()
               << " which contains " << body_ptr()->get_ops().size() << " nodes" << std::endl;

    int qqq = 0;
    for (const auto& op : body_ptr()->get_ordered_ops()) {
        remark(13) << "op " << qqq++ << " " << op->get_friendly_name() << " (" << op->get_type_name() << ") " << op
                   << std::endl;
    }

    for (auto& in : this->inputs()) {
        remark(13) << "  -> " << in.get_source_output().get_node_shared_ptr()->get_friendly_name() << " "
                   << in.get_source_output().get_node_shared_ptr() << std::endl;
    }

    for (auto& out : this->outputs()) {
        for (auto& user : out.get_target_inputs()) {
            remark(13) << " <- " << user.get_node()->get_friendly_name() << " " << user.get_node() << std::endl;
        }
        remark(13) << std::endl;
    }
}

} // namespace op
} // namespace snippets
} // namespace ov
