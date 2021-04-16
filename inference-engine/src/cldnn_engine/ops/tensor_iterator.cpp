// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"
#include "cldnn_engine.h"

#include <cpp/ie_cnn_network.h>

#include "ngraph/op/tensor_iterator.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/util/sub_graph_base.hpp"

#include "api/loop.hpp"
#include "api/mutable_data.hpp"

#include <vector>
#include <algorithm>

using TensorIterator = ngraph::op::v0::TensorIterator;

namespace CLDNNPlugin {

static cldnn::mutable_data CreateIntData(Program &p, const cldnn::primitive_id& id, int32_t num) {
    auto mem = cldnn::memory::allocate(p.GetEngine(),
        { cldnn::data_types::i32, cldnn::format::bfyx, { 1, 1, 1, 1 } });
    auto ptr = mem.pointer<int32_t>();
    *ptr.begin() = num;
    return {id, mem};
}

static cldnn::mutable_data CreateOutputData(Program &p, const std::shared_ptr<ngraph::Node>& op,
                                            const cldnn::primitive_id& id, const cldnn::primitive_id& input,
                                            const int32_t output_idx) {
    const auto precision = DataTypeFromPrecision(op->get_output_element_type(output_idx));
    const auto format = DefaultFormatForDims(op->get_output_shape(output_idx).size());
    const auto tensor = CldnnTensorFromIEDims(op->get_output_shape(output_idx));
    cldnn::layout output_layout = cldnn::layout(precision, format, tensor);
    auto mem = cldnn::memory::allocate(p.GetEngine(), output_layout);
    auto md = cldnn::mutable_data(id, {input}, mem);
    return md;
}

static cldnn::primitive_id GetOutputPrimitiveID(const Program& p, const std::shared_ptr<ngraph::Node>& op) {
    cldnn::primitive_id output_id = layer_type_lower(op) + ":" + op->get_friendly_name();
    auto found = std::find_if(p.primitiveIDs.begin(), p.primitiveIDs.end(),
        [&output_id, &p](const std::pair<cldnn::primitive_id, cldnn::primitive_id>& pm) {
            return pm.second == output_id;
        });
    assert(found != p.primitiveIDs.end());
    return found->first;
}

static cldnn::primitive_id GetPrimitiveID(const Program& p, const std::shared_ptr<ngraph::Node>& op) {
    auto found = std::find_if(p.primitivesToIRLayersMap.begin(), p.primitivesToIRLayersMap.end(),
        [&op, &p](const std::pair<cldnn::primitive_id, std::vector<std::string>>& pm) {
            assert(pm.second.size() == 1);
            return pm.second.front() == op->get_friendly_name();
        });
    assert(found != p.primitivesToIRLayersMap.end());
    return found->first;
}

static cldnn::primitive_id GetInputPrimitiveID(const std::shared_ptr<ngraph::Node>& op, const size_t index) {
    const std::shared_ptr<ngraph::op::Op> op2 = std::dynamic_pointer_cast<ngraph::op::Op>(op);
    return layer_type_name_ID(op2->inputs().at(index).get_source_output().get_node());
}

void CreateTensorIteratorOp(Program &p, const std::shared_ptr<TensorIterator> &op) {
    // loop can takes multiple inputs, no p.ValidateInputs()
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);

    // set body topology
    InferenceEngine::CNNNetwork body_network(op->get_body());
    Program body_program(body_network, p.GetEnginePtr(), p.GetConfig(), false); //TODO:
    auto body_topology = *body_program.GetTopology();

    // setup primitive_map and back_edges
    const auto& input_mappings = op->get_input_descriptions();
    const auto& output_mappings = op->get_output_descriptions();
    const auto& body_inputs = op->get_body()->get_parameters();
    const auto& body_outputs = op->get_body()->get_results();

    std::vector<cldnn::loop::primitive_mapping> primitive_map;
    std::vector<cldnn::loop::backedge_mapping> back_edges;

    for (const auto& input_mapping : input_mappings) {
        const cldnn::primitive_id& external_id = inputPrimitives.at(input_mapping->m_input_index);
        const auto& body_input = body_inputs.at(input_mapping->m_body_parameter_index);
        cldnn::primitive_id internal_id = GetPrimitiveID(body_program, body_input);
        if (const auto& sliceInfo =
            std::dynamic_pointer_cast<TensorIterator::SliceInputDescription>(input_mapping)) {
            // input with iteration axis
            primitive_map.emplace_back(cldnn::loop::INPUT, external_id, internal_id,
                sliceInfo->m_axis, sliceInfo->m_start, sliceInfo->m_end, sliceInfo->m_stride);
        } else {
            // InvariantInputDescription or InputDescription
            // input without iteration axis
            primitive_map.emplace_back(cldnn::loop::INPUT, external_id, internal_id);
        }
        if (const auto& mergedInput =
            std::dynamic_pointer_cast<TensorIterator::MergedInputDescription>(input_mapping)) {
            // backedge
            const auto& to = body_inputs.at(mergedInput->m_body_parameter_index);
            const auto& from = body_outputs.at(mergedInput->m_body_value_index);
            cldnn::primitive_id to_id = layer_type_name_ID(to);
            cldnn::primitive_id from_id = layer_type_name_ID(from);
            back_edges.emplace_back(from_id, to_id);
        }
    }

    std::string layerName = layer_type_name_ID(op);
    const cldnn::primitive_id trip_count_id = layerName + "_tripCount";
    const int32_t num_iterations = op->get_num_iterations();
    assert(num_iterations >= 0);
    {
        cldnn::mutable_data trip_count = CreateIntData(p, trip_count_id, num_iterations);
        p.primitivesToIRLayersMap[trip_count_id] = { op->get_friendly_name() };
        p.primitiveIDs[trip_count_id] = trip_count_id;
        p.AddPrimitive(trip_count);
        p.AddInnerPrimitiveToProfiler(trip_count_id, layerName, op);
    }
    const cldnn::primitive_id execution_condition_id = layerName + "_initialExecutionCondition";
    {
        cldnn::mutable_data execution_condition = CreateIntData(p, execution_condition_id, 1);
        p.primitivesToIRLayersMap[execution_condition_id] = { op->get_friendly_name() };
        p.primitiveIDs[execution_condition_id] = execution_condition_id;
        p.AddPrimitive(execution_condition);
        p.AddInnerPrimitiveToProfiler(execution_condition_id, layerName, op);
    }
    const cldnn::primitive_id num_iteration_id = layerName + "_numIteration";
    {
        cldnn::mutable_data num_iteration = CreateIntData(p, num_iteration_id, 0);
        p.primitivesToIRLayersMap[num_iteration_id] = { op->get_friendly_name() };
        p.primitiveIDs[num_iteration_id] = num_iteration_id;
        p.AddPrimitive(num_iteration);
        p.AddInnerPrimitiveToProfiler(num_iteration_id, layerName, op);
    }

    for (const auto& output_mapping : output_mappings) {
        const int output_idx = output_mapping->m_output_index;
        const std::string layerNameWithIndex = layerName + "." + std::to_string(output_idx);
        std::string external_id;
        if (output_idx > 0) {
            cldnn::mutable_data output_data = CreateOutputData(p, op, layerNameWithIndex, layerName, output_idx);
            p.primitiveIDs[layerNameWithIndex] = layerNameWithIndex;
            p.AddPrimitive(output_data);
            p.AddInnerPrimitiveToProfiler(layerNameWithIndex, layerName, op);
            external_id = layerNameWithIndex;
        } else {
            p.primitiveIDs[layerNameWithIndex] = layerName;
            p.primitiveIDs[layerName] = layerName;
            external_id = layerName;
        }
        const auto& body_output = body_outputs.at(output_mapping->m_body_value_index);
        const cldnn::primitive_id internal_id = layer_type_name_ID(body_output);
        if (const auto& concatOutput =
            std::dynamic_pointer_cast<TensorIterator::ConcatOutputDescription>(output_mapping)) {
            // output requires concatenation
            primitive_map.emplace_back(cldnn::loop::OUTPUT, external_id, internal_id,
                concatOutput->m_axis, concatOutput->m_start, concatOutput->m_end, concatOutput->m_stride);
        }
        if (const auto& body_desc =
            std::dynamic_pointer_cast<TensorIterator::BodyOutputDescription>(output_mapping)) {
            // output requires no concatenation
            primitive_map.emplace_back(cldnn::loop::OUTPUT, external_id, internal_id);
        }
    }

    const cldnn::loop loopPrimitive(
        layerName,              /* layer name of this primitive (output id) */
        inputPrimitives,        /* inputs of this layer */
        body_topology,          /* body network */
        trip_count_id,          /* trip_count data in outer network, always same as num_iterations in TI */
        execution_condition_id, /* initial_execution_condition data in outer network, always true in TI */
        num_iteration_id,       /* actual number of iteration data in body network */
        primitive_map,          /* primitive mapping connecting outer network and innter network */
        back_edges,             /* back edge mapping */
        num_iterations);        /* max iteration */

    p.AddPrimitive(loopPrimitive);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v0, TensorIterator);

} // namespace CLDNNPlugin
