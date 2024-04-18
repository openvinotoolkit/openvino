// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/generator.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/op/kernel.hpp"
#include "snippets/op/memory_access.hpp"

namespace ov {
namespace snippets {

void Generator::generate(lowered::LinearIR& linear_ir, LoweringResult& result, const void* compile_params) const {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::Generator::generate")

    OV_ITT_TASK_CHAIN(GENERATE, ov::pass::itt::domains::SnippetsTransform, "Snippets::Generator", "::InitEmitters")

    OPENVINO_ASSERT(target->is_supported(), "unsupported architecture for code generation");
    linear_ir.init_emitters(target);

    OV_ITT_TASK_NEXT(GENERATE, "::EmitCode")

    const auto kernel_op = op::Kernel::make_kernel(linear_ir);
    kernel_op->compile_params = compile_params;
    const auto kernel_expr = linear_ir.create_expression(kernel_op, std::vector<lowered::PortConnectorPtr>{});
    const auto kernel = target->get(kernel_expr->get_node()->get_type_info())(kernel_expr);

    kernel->emit_code({}, {});

    OV_ITT_TASK_NEXT(GENERATE, "::EmitData")
    for (auto& l : linear_ir.get_ops()) {
        l->get_emitter()->emit_data();
    }
    OV_ITT_TASK_NEXT(GENERATE, "::GetSnippet")

    // 1. some emitters use precompiled kernels. They need to be saved, so the kernels are accessible at runtime.
    // 2. perf count node as field of emitter should be alive at runtime.
    // 3. Emitters with segfault detector debug capabilty also need to be accessible at runtime.
    for (const auto& expr : linear_ir) {
        const auto& emitter = expr->get_emitter();
        if (uses_precompiled_kernel(emitter))
            result.m_saved_emitters.emplace_back(emitter);
    }
    result.compiled_snippet = target->get_snippet();
}

std::shared_ptr<const TargetMachine> Generator::get_target_machine() const {
    return target;
}

RegType Generator::get_op_out_reg_type(const ov::Output<Node>& out) const {
    auto reg_type = get_specific_op_out_reg_type(out);
    if (reg_type != RegType::undefined)
        return reg_type;
    const auto op = out.get_node_shared_ptr();
    if (std::dynamic_pointer_cast<ov::op::v0::Parameter>(op) ||
        std::dynamic_pointer_cast<ov::op::v0::Result>(op) ||
        std::dynamic_pointer_cast<op::LoopBegin>(op) ||
        std::dynamic_pointer_cast<op::LoopEnd>(op) ||
        std::dynamic_pointer_cast<op::Brgemm>(op) ||
        std::dynamic_pointer_cast<op::IntermediateMemoryBuffer>(op) ||
        std::dynamic_pointer_cast<op::NewMemoryBuffer>(op) ||
        std::dynamic_pointer_cast<op::RankNormalization>(op) ||
        std::dynamic_pointer_cast<op::Reshape>(op) ||
        std::dynamic_pointer_cast<snippets::op::Store>(op)
#ifdef SNIPPETS_DEBUG_CAPS
        || std::dynamic_pointer_cast<op::PerfCountBeginBase>(op)
        || std::dynamic_pointer_cast<op::PerfCountEndBase>(op)
#endif
        )
        return RegType::gpr;
    else if (std::dynamic_pointer_cast<snippets::op::Load>(op) ||
             std::dynamic_pointer_cast<snippets::op::BroadcastLoad>(op) ||
             ov::op::util::is_unary_elementwise_arithmetic(op) ||
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
        return RegType::vec;
    else
        OPENVINO_THROW("Register type of the operation " + std::string(op->get_type_name()) + " isn't determined!");
    return reg_type;
}

RegType Generator::get_specific_op_out_reg_type(const ov::Output<Node>& out) const {
    OPENVINO_THROW("Register type of the operation " + std::string(out.get_node()->get_type_name()) + " isn't determined!");
}

const std::shared_ptr<RuntimeConfig>& Generator::update_runtime_config(const std::shared_ptr<lowered::LinearIR>& linear_ir) const {
    OPENVINO_ASSERT(target, "TargetMachine has not been inited!");
    return target->update_runtime_config(linear_ir);
}

}// namespace snippets
}// namespace ov
