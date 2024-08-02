// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/op/if.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/condition.hpp"

namespace ov {
namespace intel_gpu {

const size_t idx_true = 0;
const size_t idx_false = 1;

static cldnn::condition::branch gen_branch(ProgramBuilder& p, const std::shared_ptr<ov::op::v8::If>& op, size_t idx) {
    cldnn::condition::branch branch;
    const auto& internal_body = (idx == idx_true)? op->get_then_body() : op->get_else_body();
    GPU_DEBUG_LOG << "Generate inner program for " << "op::v"
                    << op->get_type_info().version_id << "::"
                    << op->get_type_name() << " operation "
                    << "(friendly_name=" << op->get_friendly_name() << ") : "
                    << internal_body->get_friendly_name()
                    << ", num inputs: " << op->get_input_size() << std::endl;

    auto config = p.get_config();
    {
        auto custom_outputs = config.get_property(ov::intel_gpu::custom_outputs);
        if (!custom_outputs.empty()) {
            config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>({})));
        }
    }
    config.set_property(ov::intel_gpu::max_dynamic_batch(1));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(op->is_dynamic() || p.use_new_shape_infer()));

    ProgramBuilder prog(internal_body, p.get_engine(), config, false, p.get_task_executor(), p.get_compilation_context(), true);
    branch.inner_program = prog.get_compiled_program();

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

    GPU_DEBUG_LOG << op->get_friendly_name() << " branch_info[" << internal_body->get_friendly_name() << "] : " << branch << std::endl;
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

    const size_t num_outputs = op->get_output_size();

    const cldnn::condition conditionPrimitive(layerName,
                                inputs,
                                branch_true,
                                branch_false,
                                num_outputs);

    p.add_primitive(*op, conditionPrimitive);
}

REGISTER_FACTORY_IMPL(v8, If);

}  // namespace intel_gpu
}  // namespace ov
