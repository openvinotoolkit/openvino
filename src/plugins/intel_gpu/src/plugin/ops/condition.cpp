// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/op/if.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/condition.hpp"

#include "ie_ngraph_utils.hpp"

namespace ov {
namespace intel_gpu {

const size_t idx_true = 0;
const size_t idx_false = 1;

static cldnn::condition::branch gen_branch(ProgramBuilder& p, const std::shared_ptr<ov::op::v8::If>& op, size_t idx) {
    cldnn::condition::branch branch;
    const auto& internal_body = (idx == idx_true)? op->get_then_body() : op->get_else_body();
    InferenceEngine::CNNNetwork body_network(internal_body);
    {
        // CNNNetwork change the input/output data type to fp32 when input/output data type is fp16
        // To run internal body, rollback input/output data to original one.
        size_t tidx = 0;
        auto& model_inputs = internal_body->get_parameters();
        for (auto& in : body_network.getInputsInfo()) {
            auto input_data_type = InferenceEngine::details::convertPrecision(model_inputs[tidx++]->get_output_tensor(0).get_element_type());
            if (in.second->getPrecision() != input_data_type)
                in.second->setPrecision(input_data_type);
        }

        tidx = 0;
        for (auto& out : body_network.getOutputsInfo()) {
            const auto& model_output = internal_body->get_output_op(tidx++);
            auto output_data_type = InferenceEngine::details::convertPrecision(model_output->get_output_tensor(0).get_element_type());
            if (out.second->getPrecision() != output_data_type)
                out.second->setPrecision(output_data_type);
        }
    }

    auto config = p.get_config();
    config.set_property(ov::intel_gpu::max_dynamic_batch(1));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(op->is_dynamic()));

    ProgramBuilder prog(body_network, p.get_engine(), config, false, false, nullptr, nullptr, p.get_task_executor(), true);
    branch.inner_program = prog.GetCompiledProgram();

    auto& input_map = branch.input_map;
    auto external_inputs = p.GetInputInfo(op);
    auto internal_inputs = internal_body->get_parameters();
    auto input_desc_vec = op->get_input_descriptions(static_cast<int>(idx));
    for (auto& in_desc : input_desc_vec) {
        const auto& external_id = external_inputs.at(in_desc->m_input_index).pid;
        const auto& internal_id = layer_type_name_ID(internal_inputs.at(in_desc->m_body_parameter_index));
        input_map.insert({external_id, internal_id});
    }

    auto& output_map = branch.output_map;
    auto internal_outputs = internal_body->get_results();
    auto output_desc_vec = op->get_output_descriptions(static_cast<int>(idx));
    for (auto& out_desc : output_desc_vec) {
        const auto& internal_id = layer_type_name_ID(internal_outputs.at(out_desc->m_body_value_index));
        output_map.insert({out_desc->m_output_index, internal_id});
    }

    return branch;
}

static void CreateIfOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v8::If>& op) {
    auto inputs = p.GetInputInfo(op);
    OPENVINO_ASSERT(inputs.size() >= 1, "Invalid inputs count (Not allowed no input)");
    auto compare_node_pshape = op->get_input_partial_shape(0);
    auto p_input_name = inputs[0].pid;
    std::string type_name_str = op->get_input_node_ptr(0)->get_type_name();

    const std::string layerName = layer_type_name_ID(op);
    auto branch_true = gen_branch(p, op, idx_true);
    auto branch_false = gen_branch(p, op, idx_false);

    const cldnn::condition conditionPrimitive(layerName,
                                inputs,
                                branch_true,
                                branch_false);

    p.add_primitive(*op, conditionPrimitive);
}

REGISTER_FACTORY_IMPL(v8, If);

}  // namespace intel_gpu
}  // namespace ov
