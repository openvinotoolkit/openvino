// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/plugin.hpp"

#include "openvino/op/tensor_iterator.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/util/sub_graph_base.hpp"
#include "transformations/utils/utils.hpp"

#include "intel_gpu/primitives/loop.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/graph/topology.hpp"

#include <vector>
#include <algorithm>

using Loop = ov::op::v5::Loop;
using TensorIterator = ov::op::v0::TensorIterator;

namespace ov {
namespace intel_gpu {

template<class DATA_TYPE>
static DATA_TYPE CreateScalarData(ProgramBuilder &p, const cldnn::primitive_id& id, ov::Shape& shape, cldnn::data_types dtype, int64_t num, int64_t rank) {
    auto mem = p.get_engine().allocate_memory({ shape, dtype, cldnn::format::get_default_format(rank) });
    cldnn::mem_lock<int64_t> ptr{mem, p.get_engine().get_service_stream()};
    *ptr.begin() = num;
    return {id, mem};
}

static cldnn::mutable_data CreateAdditionalOutputData(ProgramBuilder &p, const std::shared_ptr<ov::Node>& op,
                                                        const cldnn::primitive_id& id, const cldnn::primitive_id& input,
                                                        const int32_t output_idx) {
    const auto precision = cldnn::element_type_to_data_type(op->get_output_element_type(output_idx));
    const auto format = cldnn::format::get_default_format(op->get_output_shape(output_idx).size());
    const auto tensor = tensor_from_dims(op->get_output_shape(output_idx));
    cldnn::layout output_layout = cldnn::layout(precision, format, tensor);
    auto mem = p.get_engine().allocate_memory(output_layout);
    auto md = cldnn::mutable_data(id, {cldnn::input_info(input)}, std::move(mem)); // cldnn::data cannot set dependency
    return md;
}

static void SetLoopInputOutputMap(ProgramBuilder& p,
                                    const std::shared_ptr<ov::op::util::SubGraphOp>& op,
                                    cldnn::primitive::input_info_arr& inputs,
                                    std::vector<cldnn::loop::io_primitive_map>& input_primitive_maps,
                                    std::vector<cldnn::loop::io_primitive_map>& output_primitive_maps,
                                    std::vector<cldnn::loop::backedge_mapping>& back_edges_maps) {
    const std::string layerName = layer_type_name_ID(op);
    const auto& loop_input_descs = op->get_input_descriptions();
    const auto& loop_output_descs = op->get_output_descriptions();
    const auto& body_inputs = op->get_function()->get_parameters();
    const auto& body_outputs = op->get_function()->get_results();

    bool use_new_shape_infer = p.use_new_shape_infer();

    // set input mapping & back edges
    for (const auto& loop_input_desc : loop_input_descs) {
        auto external_id = inputs.at(loop_input_desc->m_input_index);
        auto& body_input = body_inputs.at(loop_input_desc->m_body_parameter_index);
        cldnn::primitive_id internal_id = layer_type_name_ID(body_input);

        // set input mapping
        if (const auto& sliceInfo =
            std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp::SliceInputDescription>(loop_input_desc)) {
            // sliced input
            input_primitive_maps.emplace_back(external_id, internal_id, sliceInfo->m_axis,
                sliceInfo->m_start, sliceInfo->m_end, sliceInfo->m_stride);
            GPU_DEBUG_LOG << "loop_input_descs[" << layerName << "][SliceInputDescription] = {m_input_index:"
                        << loop_input_desc->m_input_index << "(external_id: "
                        << external_id << "), m_body_parameter_index:" << loop_input_desc->m_body_parameter_index
                        << "(internal_id: " << internal_id << ")}" << std::endl;
        } else {
            // input without slicing
            input_primitive_maps.emplace_back(external_id, internal_id);
            GPU_DEBUG_LOG << "loop_input_descs[" << layerName << "][InputDescription] = {m_input_index:"
                        << loop_input_desc->m_input_index << "(external_id: "
                        << external_id << "), m_body_parameter_index:" << loop_input_desc->m_body_parameter_index
                        << "(internal_id: " << internal_id << ")}" << std::endl;
        }

        // set back edges
        if (const auto& mergedInput =
            std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp::MergedInputDescription>(loop_input_desc)) {
            // backedge
            const auto& to = body_inputs.at(mergedInput->m_body_parameter_index);
            const auto& from = body_outputs.at(mergedInput->m_body_value_index);

            cldnn::primitive_id to_id = layer_type_name_ID(to);
            cldnn::primitive_id from_id = layer_type_name_ID(from);

            back_edges_maps.emplace_back(from_id, to_id);
            GPU_DEBUG_LOG << "back_edge = {" << from_id << " => " << to_id << "}" << std::endl;
        }
    }

    // set output mapping
    if (use_new_shape_infer) {
        for (const auto& loop_output_desc : loop_output_descs) {
            cldnn::input_info external_input_info(layerName, loop_output_desc->m_output_index);
            p.primitive_ids[layerName] = layerName;

            const auto& body_output = body_outputs.at(loop_output_desc->m_body_value_index);
            cldnn::primitive_id internal_id = layer_type_name_ID(body_output);

            // update primitive_map
            if (const auto& concatOutput =
                std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp::ConcatOutputDescription>(loop_output_desc)) {
                // output which requires concatenation
                output_primitive_maps.emplace_back(external_input_info, internal_id, concatOutput->m_axis,
                    concatOutput->m_start, concatOutput->m_end, concatOutput->m_stride);
                GPU_DEBUG_LOG << "loop_output_descs[" << layerName << "][ConcatOutputDescription] external:"
                        << external_input_info << ", internal:"
                        << internal_id << "(axis, start, end, stride)={"
                        << concatOutput->m_axis << "," << concatOutput->m_start << ","
                        << concatOutput->m_end << "," << concatOutput->m_stride << "}" << std::endl;
            }
            if (std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp::BodyOutputDescription>(loop_output_desc)) {
                // output which requires no concatenation
                output_primitive_maps.emplace_back(external_input_info, internal_id);
                GPU_DEBUG_LOG << "loop_output_descs[" << layerName << "][BodyOutputDescription] external:"
                        << external_input_info << ", internal:" << internal_id << std::endl;
            }
        }
    } else {
        for (const auto& loop_output_desc : loop_output_descs) {
            const uint64_t output_idx = loop_output_desc->m_output_index;

            // Add additional mutable_data for multiple outputs
            // primitive ID should be <TI primitive ID>.<output_idx> if output_idx > 0
            // otherwise primitive ID should be equals to TI primitive ID
            const std::string layerNameWithIndex = layerName + ".out" + std::to_string(output_idx);
            std::string external_id;
            if (output_idx > 0) {
                cldnn::mutable_data output_data = CreateAdditionalOutputData(p, op, layerNameWithIndex, layerName, output_idx);
                p.add_primitive(*op, std::move(output_data));
                external_id = layerNameWithIndex;
            } else {
                p.primitive_ids[layerNameWithIndex] = layerName;
                p.primitive_ids[layerName] = layerName;
                external_id = layerName;
            }
            const auto& body_output = body_outputs.at(loop_output_desc->m_body_value_index);
            cldnn::primitive_id internal_id = layer_type_name_ID(body_output);

            // update primitive_map
            if (const auto& concatOutput =
                std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp::ConcatOutputDescription>(loop_output_desc)) {
                // output which requires concatenation
                output_primitive_maps.emplace_back(external_id, internal_id, concatOutput->m_axis,
                    concatOutput->m_start, concatOutput->m_end, concatOutput->m_stride);
                GPU_DEBUG_LOG << "loop_output_descs[" << layerName << "][ConcatOutputDescription] external:"
                        << external_id << ", internal:"
                        << internal_id << "(axis, start, end, stride)={"
                        << concatOutput->m_axis << "," << concatOutput->m_start << ","
                        << concatOutput->m_end << "," << concatOutput->m_stride << "}" << std::endl;
            }
            if (std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp::BodyOutputDescription>(loop_output_desc)) {
                // output which requires no concatenation
                output_primitive_maps.emplace_back(external_id, internal_id);
                GPU_DEBUG_LOG << "loop_output_descs[" << layerName << "][BodyOutputDescription] external:"
                        << external_id << ", internal:" << internal_id << std::endl;
            }
        }
    }
}

static std::vector<cldnn::primitive_id> GetOutputNames(const cldnn::primitive_id id,
                                                        const cldnn::primitive_id body_execution_condition_id,
                                                        const std::vector<cldnn::loop::io_primitive_map>& output_primitive_maps,
                                                        const std::vector<cldnn::loop::backedge_mapping>& back_edges) {
    std::vector<cldnn::primitive_id> output_names;
    OPENVINO_ASSERT(!output_primitive_maps.empty(), "[GPU] Output primitive map should have at least 1 mapping in primitive ", id);
    for (auto out_map : output_primitive_maps) {
        output_names.push_back(out_map.internal_id.pid);
    }

    // setup outputs for backedges
    for (auto& back_edge : back_edges) {
        auto iter = std::find(output_names.begin(), output_names.end(), back_edge.from);
        // Do not add duplicated output name
        if (iter == output_names.end()) {
            output_names.push_back(back_edge.from);
        }
    }

    // if execution_condition_id is specified, we need to add the id in build_option::outputs
    if (!body_execution_condition_id.empty()) {
        output_names.push_back(body_execution_condition_id);
    }

    return output_names;
}

static void CreateCommonLoopOp(ProgramBuilder& p, const std::shared_ptr<ov::op::util::SubGraphOp>& op, bool is_loop_op) {
    const std::string layerName = layer_type_name_ID(op);
    auto inputs = p.GetInputInfo(op);
    bool is_dynamic = p.use_new_shape_infer() || op->is_dynamic();

    int64_t num_iterations = op->get_num_iterations();

    auto num_outputs = is_dynamic? op->get_output_size() : 1;
    auto ov_model = op->get_function();

    // Set special body ports: current_iteration input , execution condition output
    cldnn::primitive_id body_current_iteration_id;
    cldnn::primitive_id body_execution_condition_id;
    cldnn::primitive_id trip_count_id;
    cldnn::primitive_id first_execution_condition_id;
    cldnn::primitive_id updated_current_iteration_id;

    std::shared_ptr<ov::op::v0::Parameter> current_iteration_input_op;
    if (is_loop_op) {
        auto loop_op = std::dynamic_pointer_cast<Loop>(op);
        auto special_body_ports = loop_op->get_special_body_ports();
        if (special_body_ports.current_iteration_input_idx >= 0) {
            const auto& body_inputs = loop_op->get_function()->get_parameters();
            current_iteration_input_op = body_inputs.at(special_body_ports.current_iteration_input_idx);
            body_current_iteration_id = layer_type_name_ID(current_iteration_input_op);
        }

        if (special_body_ports.body_condition_output_idx >= 0) {
            const auto& body_outputs = loop_op->get_function()->get_results();
            auto body_condition_output = body_outputs.at(special_body_ports.body_condition_output_idx)->get_input_node_shared_ptr(0);
            body_execution_condition_id = layer_type_name_ID(body_condition_output);
        }

        trip_count_id = layer_type_name_ID(loop_op->get_input_node_shared_ptr(0));
        // Update trip_count_id for cached constant primitive
        if (trip_count_id != p.primitive_ids[trip_count_id])
            trip_count_id = p.primitive_ids[trip_count_id];
        first_execution_condition_id = layer_type_name_ID(loop_op->get_input_node_shared_ptr(1));
    }

    // setup input_primitive_maps/ output_primitive_maps and back_edges
    std::vector<cldnn::loop::io_primitive_map> input_primitive_maps;
    std::vector<cldnn::loop::io_primitive_map> output_primitive_maps;
    std::vector<cldnn::loop::backedge_mapping> back_edges;

    SetLoopInputOutputMap(p, op, inputs, input_primitive_maps, output_primitive_maps, back_edges);

    auto shape = is_dynamic? ov::Shape{} : ov::Shape{1, 1, 1, 1};
    if (!is_dynamic) {
        for (size_t i = 4; i < op->get_output_shape(0).size(); ++i) {
            shape.push_back(1);
        }
    }
    auto prec = ov::element::i64;
    if (current_iteration_input_op) {
        OPENVINO_ASSERT(current_iteration_input_op->get_partial_shape().is_static(), "current_iteration should be static layout");
        shape = is_dynamic? current_iteration_input_op->get_partial_shape().to_shape() : shape;
        prec = current_iteration_input_op->get_element_type();

        auto increment_value_id = current_iteration_input_op->get_friendly_name() + "_inc";
        auto increment_value_op = std::make_shared<op::v0::Constant>(prec, shape, 1);
        increment_value_op->set_friendly_name(increment_value_id);

        auto update_current_iter_op_id = current_iteration_input_op->get_friendly_name() + "_update";
        auto update_current_iter_op = std::make_shared<op::v1::Add>(current_iteration_input_op, increment_value_op);
        update_current_iter_op->set_friendly_name(update_current_iter_op_id);
        updated_current_iteration_id = layer_type_name_ID(update_current_iter_op);

        auto result = std::make_shared<ov::op::v0::Result>(update_current_iter_op);
        ov_model->add_results({result});

        const cldnn::primitive_id first_iter_id = layerName + "_firstIter";
        cldnn::mutable_data first_iter_data = CreateScalarData<cldnn::mutable_data>(p, first_iter_id, shape, prec, 0,
                                static_cast<int64_t>(current_iteration_input_op->get_output_partial_shape(0).rank().get_length()));

        p.add_primitive(*op, std::move(first_iter_data));
        inputs.insert(inputs.begin(), cldnn::input_info(first_iter_id, 0));

        input_primitive_maps.emplace_back(cldnn::input_info(first_iter_id), cldnn::input_info(body_current_iteration_id));
        back_edges.emplace_back(updated_current_iteration_id, body_current_iteration_id);
    }

    // set trip count, num iteration primitives
    // they should be mutable_data to prevent from being optimized out
    const cldnn::primitive_id num_iteration_id = layerName + "_numIteration";
    cldnn::mutable_data num_iteration_data = CreateScalarData<cldnn::mutable_data>(p, num_iteration_id, shape, prec, 0,
                                                                                   static_cast<int64_t>(op->get_output_partial_shape(0).rank().get_length()));

    p.add_primitive(*op, std::move(num_iteration_data));
    inputs.insert(inputs.begin(), cldnn::input_info(num_iteration_id, 0));

    auto output_names_vec = GetOutputNames(layerName, body_execution_condition_id, output_primitive_maps, back_edges);

    auto config = p.get_config();
    config.set_property(ov::intel_gpu::custom_outputs(output_names_vec));
    config.set_property(ov::intel_gpu::max_dynamic_batch(1));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(is_dynamic));

    // get body program from ov::Model
    ProgramBuilder prog(ov_model, p.get_engine(), config, false, p.get_task_executor(), p.get_compilation_context(), true);
    auto body_program = prog.get_compiled_program();

    GPU_DEBUG_LOG << "* trip_count_id                 : " << trip_count_id << std::endl;
    GPU_DEBUG_LOG << "* num_iteration_id              : " << num_iteration_id << std::endl;
    GPU_DEBUG_LOG << "* body_current_iteration_id     : " << body_current_iteration_id << std::endl;
    GPU_DEBUG_LOG << "* first_execution_condition_id  : " << first_execution_condition_id << std::endl;
    GPU_DEBUG_LOG << "* body_execution_condition_id   : " << body_execution_condition_id << std::endl;

    const cldnn::loop loopPrimitive(
        layerName,                      /* layer name of this primitive (output id) */
        inputs,                         /* inputs of this layer */
        body_program,                   /* body network */
        trip_count_id,                  /* trip_count data in outer network, always same as num_iterations in TI */
        first_execution_condition_id,   /* initial_execution_condition data in outer network, always true in TI */
        num_iteration_id,               /* actual number of iteration data in body network */
        input_primitive_maps,           /* input mappings connecting outer network and inner network */
        output_primitive_maps,          /* output mappings connecting outer network and inner network */
        back_edges,                     /* back edge mapping */
        num_iterations,                 /* max iteration, i.e. length of iteration axis */
        body_current_iteration_id,
        body_execution_condition_id,
        num_outputs);

    p.add_primitive(*op, loopPrimitive);
}

static void CreateLoopOp(ProgramBuilder& p, const std::shared_ptr<Loop>& op) {
    CreateCommonLoopOp(p, op, true);
}

/* The above code is a comment in C++ programming language. It is not doing anything in terms of code
execution. It is simply providing information or documentation about the code. */
static void CreateTensorIteratorOp(ProgramBuilder& p, const std::shared_ptr<TensorIterator>& op) {
    CreateCommonLoopOp(p, op, false);
}

REGISTER_FACTORY_IMPL(v5, Loop);
REGISTER_FACTORY_IMPL(v0, TensorIterator);

}  // namespace intel_gpu
}  // namespace ov
