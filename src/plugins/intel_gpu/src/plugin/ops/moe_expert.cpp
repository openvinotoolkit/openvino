// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/op/if.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "ov_ops/moe_expert.hpp"
#include "intel_gpu/primitives/moe_expert.hpp"

namespace ov::intel_gpu {

const size_t idx_true = 0;
const size_t idx_false = 1;

static cldnn::moe_expert::branch gen_branch(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::MOEExpert>& op) {
    cldnn::moe_expert::branch branch;
    const auto& internal_body = op->get_function();
    GPU_DEBUG_LOG << "Generate inner program for " << "op::v"
                    << op->get_type_info().version_id << "::"
                    << op->get_type_name() << " operation "
                    << "(friendly_name=" << op->get_friendly_name() << ") : "
                    << internal_body->get_friendly_name()
                    << ", num inputs: " << op->get_input_size() << std::endl;

    auto config = p.get_config().clone();
    config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>({})));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(op->is_dynamic() || p.use_new_shape_infer()));
    config.finalize(p.get_engine());

    ProgramBuilder prog(internal_body, p.get_engine(), config, p.get_task_executor(), p.get_compilation_context(), true);
    branch.inner_program = prog.get_compiled_program();

    auto& input_map = branch.input_map;
    auto external_inputs = p.GetInputInfo(op);
    auto internal_inputs = internal_body->get_parameters();
    for (size_t i = 0; i < external_inputs.size(); i++) {
       const auto& external_id = external_inputs.at(i).pid;
       const auto& internal_id = layer_type_name_ID(internal_inputs.at(i));
       input_map.insert({external_id, internal_id});
    }

    GPU_DEBUG_LOG << op->get_friendly_name() << " branch_info[" << internal_body->get_friendly_name() << "] : " << branch << std::endl;
    return branch;
}

static void CreateMOEExpertOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::MOEExpert>& op) {
    auto inputs = p.GetInputInfo(op);
    OPENVINO_ASSERT(inputs.size() == 4, "Inputs count should be 4");

    const std::string layerName = layer_type_name_ID(op);
    auto branch = gen_branch(p, op);

    const cldnn::moe_expert moe(layerName,
                                inputs,
                                op->get_config(),
                                branch
                                );

    p.add_primitive(*op, moe);
}

REGISTER_FACTORY_IMPL(internal, MOEExpert);

}  // namespace ov::intel_gpu
