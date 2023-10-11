// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/generator.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/assign_registers.hpp"
#include "snippets/lowered/pass/insert_tail_loop.hpp"

#include "snippets/op/kernel.hpp"

#include "snippets/itt.hpp"

namespace ov {
namespace snippets {

void Generator::generate(lowered::LinearIR& linear_ir, LoweringResult& result, const void* compile_params) const {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::Generator::generate")
    OV_ITT_TASK_CHAIN(GENERATE, ov::pass::itt::domains::SnippetsTransform, "Snippets::Generator", "::Transformations")
    if (!target->is_supported())
        OPENVINO_THROW("unsupported architecture for code generation");

    std::function<opRegType(const std::shared_ptr<Node>& op)> reg_type_mapper = [&](const std::shared_ptr<Node>& op) -> opRegType {
        return get_op_reg_type(op);
    };
    lowered::pass::PassPipeline lowered_pipeline;
    lowered_pipeline.register_pass<lowered::pass::AssignRegisters>(reg_type_mapper);
    lowered_pipeline.register_pass<lowered::pass::InsertTailLoop>();
    lowered_pipeline.run(linear_ir);
    linear_ir.init_emitters(target);

    OV_ITT_TASK_NEXT(GENERATE, "::EmitCode")
    auto loops2DKernel = std::make_shared<op::Kernel>(linear_ir);
    loops2DKernel->compile_params = compile_params;
    auto loops2DKernelExpr = linear_ir.create_expression(loops2DKernel, std::vector<lowered::PortConnectorPtr>{});
    std::shared_ptr<Emitter> kernel = target->get(op::Kernel::get_type_info_static())(loops2DKernelExpr);

    kernel->emit_code({}, {});

    OV_ITT_TASK_NEXT(GENERATE, "::EmitData")
    for (auto& l : linear_ir.get_ops()) {
        l->get_emitter()->emit_data();
    }
    OV_ITT_TASK_NEXT(GENERATE, "::GetSnippet")

    // Note: some emitters use precompiled kernels. They need to be saved, so the kernels are accessible at runtime.
    if (linear_ir.get_config().m_enable_segfault_detector) {
        for (const auto& expr : linear_ir) {
            const auto& emitter = expr->get_emitter();
                result.m_saved_emitters.emplace_back(emitter);
        }
    } else if (linear_ir.get_config().m_save_expressions) {
        for (const auto& expr : linear_ir) {
            const auto& emitter = expr->get_emitter();
            if (uses_precompiled_kernel(emitter))
                result.m_saved_emitters.emplace_back(emitter);
        }
    }
    result.compiled_snippet = target->get_snippet();
}

std::shared_ptr<const TargetMachine> Generator::get_target_machine() const {
    return target;
}

Generator::opRegType Generator::get_op_reg_type(const std::shared_ptr<Node>& op) const {
    if (std::dynamic_pointer_cast<ov::op::v0::Parameter>(op) ||
        std::dynamic_pointer_cast<ov::op::v0::Result>(op) ||
        std::dynamic_pointer_cast<op::LoopBegin>(op) ||
        std::dynamic_pointer_cast<op::LoopEnd>(op) ||
        std::dynamic_pointer_cast<op::Brgemm>(op) ||
        std::dynamic_pointer_cast<op::Buffer>(op) ||
        std::dynamic_pointer_cast<op::RankNormalization>(op))
        return gpr2gpr;
    else if (std::dynamic_pointer_cast<snippets::op::Load>(op) ||
             std::dynamic_pointer_cast<snippets::op::BroadcastLoad>(op))
        return gpr2vec;
    else if (std::dynamic_pointer_cast<snippets::op::Store>(op))
        return vec2gpr;
    else if (ov::op::util::is_unary_elementwise_arithmetic(op) ||
             ov::op::util::is_binary_elementwise_arithmetic(op) ||
             ov::op::util::is_binary_elementwise_comparison(op) ||
             ov::op::util::is_binary_elementwise_logical(op) ||
             std::dynamic_pointer_cast<ov::op::v1::LogicalNot>(op) ||
             std::dynamic_pointer_cast<ov::op::v0::PRelu>(op) ||
             std::dynamic_pointer_cast<ov::op::v0::Convert>(op) ||
             std::dynamic_pointer_cast<ov::op::v1::Select>(op) ||
             std::dynamic_pointer_cast<op::VectorBuffer>(op) ||
             std::dynamic_pointer_cast<op::BroadcastMove>(op) ||
             std::dynamic_pointer_cast<op::Scalar>(op) ||
             std::dynamic_pointer_cast<op::HorizonMax>(op) ||
             std::dynamic_pointer_cast<op::HorizonSum>(op) ||
             std::dynamic_pointer_cast<op::Fill>(op))
        return vec2vec;
    else
        return get_specific_op_reg_type(op);
}

Generator::opRegType Generator::get_specific_op_reg_type(const std::shared_ptr<ov::Node>& op) const {
    OPENVINO_THROW("Register type of the operation " + std::string(op->get_type_name()) + " isn't determined!");
}

}// namespace snippets
}// namespace ov
