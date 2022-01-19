// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/plugin.hpp"

#include <cpp/ie_cnn_network.h>

#include "ngraph/op/loop.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/util/sub_graph_base.hpp"
#include "transformations/utils/utils.hpp"
#include "ie_ngraph_utils.hpp"

#include "intel_gpu/primitives/loop.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/graph/topology.hpp"

#include <vector>
#include <algorithm>

using Loop = ngraph::op::v5::Loop;

namespace ov {
namespace runtime {
namespace intel_gpu {

template<class DATA_TYPE>
static DATA_TYPE CreateScalarData(Program &p, const cldnn::primitive_id& id, int64_t num, const cldnn::primitive_id& ext_prim_id) {
    auto mem = p.GetEngine().allocate_memory({ cldnn::data_types::i64, cldnn::format::bfyx, { 1, 1, 1, 1 } });
    cldnn::mem_lock<int64_t> ptr{mem, p.GetEngine().get_program_stream()};
    *ptr.begin() = num;
    return {id, mem, ext_prim_id};
}

static cldnn::mutable_data CreateAdditionalOutputData(Program &p, const std::shared_ptr<ngraph::Node>& op,
                                                        const cldnn::primitive_id& id, const cldnn::primitive_id& input,
                                                        const int32_t output_idx) {
    const auto precision = DataTypeFromPrecision(op->get_output_element_type(output_idx));
    const auto format = DefaultFormatForDims(op->get_output_shape(output_idx).size());
    const auto tensor = tensor_from_dims(op->get_output_shape(output_idx));
    cldnn::layout output_layout = cldnn::layout(precision, format, tensor);
    auto mem = p.GetEngine().allocate_memory(output_layout);
    auto md = cldnn::mutable_data(id, {input}, mem, op->get_friendly_name()); // cldnn::data cannot set dependency
    return md;
}

static void UpdateBackedge(std::vector<cldnn::loop::backedge_mapping>& back_edges,
                            const cldnn::primitive_id& old_primitive_id, const cldnn::primitive_id& new_primitive_id) {
    for (auto& back_edge : back_edges) {
        if (back_edge.from == old_primitive_id) {
            back_edge.from = new_primitive_id;
        }
    }
}

static std::string GetExternalInputName(const int64_t body_parameter_index,
                                        const std::shared_ptr<Loop>& op) {
    const auto& loop_input_descs = op->get_input_descriptions();
    for (const auto& loop_input_desc : loop_input_descs) {
        if (loop_input_desc->m_body_parameter_index == body_parameter_index) {
            auto external_node = op->get_input_node_shared_ptr(loop_input_desc->m_input_index);
            return layer_type_name_ID(external_node);
        }
    }
    return {""};
}

static void CreateLoopOp(Program& p, const std::shared_ptr<Loop>& op) {
    const std::string layerName = layer_type_name_ID(op);
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    const auto& loop_input_descs = op->get_input_descriptions();
    const auto& loop_output_descs = op->get_output_descriptions();
    const auto& body_inputs = op->get_function()->get_parameters();
    const auto& body_outputs = op->get_function()->get_results();

    InferenceEngine::CNNNetwork body_network(op->get_function());
    auto networkInputs = body_network.getInputsInfo();
    auto networkOutputs = body_network.getOutputsInfo();

    // Set special body ports: current_iteration input , execution condition output
    auto special_body_ports = op->get_special_body_ports();

    std::string body_current_iteration_id;
    if (special_body_ports.current_iteration_input_idx >= 0) {
        auto current_iteration_input = body_inputs.at(special_body_ports.current_iteration_input_idx);
        body_current_iteration_id = layer_type_name_ID(current_iteration_input);
        std::string input_name = ngraph::op::util::create_ie_output_name(current_iteration_input);
        const auto networkInput = networkInputs.at(input_name);
        auto precision = InferenceEngine::details::convertPrecision(current_iteration_input->get_element_type());
        networkInput->setPrecision(precision);
    }

    cldnn::primitive_id body_execution_condition_id;
    if (special_body_ports.body_condition_output_idx >= 0) {
        auto body_condition_output = body_outputs.at(special_body_ports.body_condition_output_idx)->get_input_node_shared_ptr(0);
        body_execution_condition_id = layer_type_name_ID(body_condition_output);
        std::string output_name = ngraph::op::util::create_ie_output_name(body_condition_output);
        const auto networkOutput = networkOutputs.at(output_name);
        networkOutput->setPrecision(InferenceEngine::Precision::I64);
    }

    // get body topology from ngraph function
    Program body_program(body_network, p.GetEnginePtr(), p.GetConfig(), true);
    auto body_topology = *body_program.GetTopology();

    // setup input_primitive_maps/ output_primitive_maps and back_edges
    std::vector<cldnn::loop::io_primitive_map> input_primitive_maps;
    std::vector<cldnn::loop::io_primitive_map> output_primitive_maps;
    std::vector<cldnn::loop::backedge_mapping> back_edges;

    // set input mapping & back edges
    for (const auto& loop_input_desc : loop_input_descs) {
        const cldnn::primitive_id& external_id = inputPrimitives.at(loop_input_desc->m_input_index);
        auto& body_input = body_inputs.at(loop_input_desc->m_body_parameter_index);
        cldnn::primitive_id internal_id = layer_type_name_ID(body_input);

        // set input mapping
        if (const auto& sliceInfo =
            std::dynamic_pointer_cast<Loop::SliceInputDescription>(loop_input_desc)) {
            // sliced input
            input_primitive_maps.emplace_back(external_id, internal_id, sliceInfo->m_axis,
                sliceInfo->m_start, sliceInfo->m_end, sliceInfo->m_stride);
        } else {
            // input without slicing
            input_primitive_maps.emplace_back(external_id, internal_id);
        }

        // set back edges
        if (const auto& mergedInput =
            std::dynamic_pointer_cast<Loop::MergedInputDescription>(loop_input_desc)) {
            // backedge
            const auto& to = body_inputs.at(mergedInput->m_body_parameter_index);
            const auto& from = body_outputs.at(mergedInput->m_body_value_index);

            cldnn::primitive_id to_id = layer_type_name_ID(to);
            cldnn::primitive_id from_id = layer_type_name_ID(from);

            // reset output data type because the data types of the outputs of the
            // body topology are always FP32 regardless of ngraph data type
            {
                const auto from_prim = body_topology.at(from_id);
                const auto& to_ngraph_type = to->get_element_type();
                const auto to_cldnn_type = DataTypeFromPrecision(to_ngraph_type);
                from_prim->output_data_type = to_cldnn_type;
            }
            back_edges.emplace_back(from_id, to_id);
        }
    }

    // set trip count, initial execution condition, num iteration primitives
    // they should be mutable_data to prevent from being optimized out
    const cldnn::primitive_id trip_count_id = layer_type_name_ID(op->get_input_node_shared_ptr(0));
    const cldnn::primitive_id execution_condition_id = layer_type_name_ID(op->get_input_node_shared_ptr(1));
    const int64_t num_iterations = op->get_num_iterations();
    if (num_iterations < 0) {
        IE_THROW() << "loop's num_iteration cannot be negative";
    }
    const cldnn::primitive_id num_iteration_id = layerName + "_numIteration";
    {
        cldnn::mutable_data num_iteration = CreateScalarData<cldnn::mutable_data>(p, num_iteration_id, 0, op->get_friendly_name());
        p.primitiveIDs[num_iteration_id] = num_iteration_id;
        p.AddPrimitive(num_iteration);
        p.AddInnerPrimitiveToProfiler(num_iteration_id, layerName, op);
    }

    // set output mapping
    for (const auto& loop_output_desc : loop_output_descs) {
        const uint64_t output_idx = loop_output_desc->m_output_index;

        // Add additional mutable_data for multiple outputs
        // primitive ID should be <TI primitive ID>.<output_idx> if output_idx > 0
        // otherwise primitive ID should be equals to TI primitive ID
        const std::string layerNameWithIndex = layerName + "." + std::to_string(output_idx);
        std::string external_id;
        if (output_idx > 0) {
            cldnn::mutable_data output_data = CreateAdditionalOutputData(p, op, layerNameWithIndex, layerName, output_idx);
            p.AddPrimitive(output_data);
            p.AddInnerPrimitiveToProfiler(layerNameWithIndex, layerName, op);
            p.primitiveIDs[layerNameWithIndex] = layerNameWithIndex;
            external_id = layerNameWithIndex;
        } else {
            p.primitiveIDs[layerNameWithIndex] = layerName;
            p.primitiveIDs[layerName] = layerName;
            external_id = layerName;
        }
        const auto& body_output = body_outputs.at(loop_output_desc->m_body_value_index);
        cldnn::primitive_id internal_id = layer_type_name_ID(body_output);

        // update primitive_map
        if (const auto& concatOutput =
            std::dynamic_pointer_cast<Loop::ConcatOutputDescription>(loop_output_desc)) {
            // output which requires concatenation
            output_primitive_maps.emplace_back(external_id, internal_id, concatOutput->m_axis,
                concatOutput->m_start, concatOutput->m_end, concatOutput->m_stride);
        }
        if (std::dynamic_pointer_cast<Loop::BodyOutputDescription>(loop_output_desc)) {
            // output which requires no concatenation
            output_primitive_maps.emplace_back(external_id, internal_id);
        }
    }

    const cldnn::loop loopPrimitive(
        layerName,              /* layer name of this primitive (output id) */
        inputPrimitives,        /* inputs of this layer */
        body_topology,          /* body network */
        trip_count_id,          /* trip_count data in outer network, always same as num_iterations in TI */
        execution_condition_id, /* initial_execution_condition data in outer network, always true in TI */
        num_iteration_id,       /* actual number of iteration data in body network */
        input_primitive_maps,         /* input mappings connecting outer network and inner network */
        output_primitive_maps,        /* output mappings connecting outer network and inner network */
        back_edges,             /* back edge mapping */
        num_iterations,         /* max iteration, i.e. length of iteration axis */
        body_current_iteration_id,
        body_execution_condition_id,
        op->get_friendly_name());

    p.AddPrimitive(loopPrimitive);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v5, Loop);

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
