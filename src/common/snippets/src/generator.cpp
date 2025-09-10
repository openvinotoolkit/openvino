// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/generator.hpp"

#include <memory>
#include <string>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/util/op_types.hpp"
#include "snippets/emitter.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/port_connector.hpp"
#include "snippets/lowered/reg_manager.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/op/broadcastload.hpp"
#include "snippets/op/broadcastmove.hpp"
#include "snippets/op/buffer.hpp"
#include "snippets/op/fill.hpp"
#include "snippets/op/horizon_max.hpp"
#include "snippets/op/horizon_sum.hpp"
#include "snippets/op/kernel.hpp"
#include "snippets/op/load.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/op/perf_count.hpp"
#include "snippets/op/rank_normalization.hpp"
#include "snippets/op/reorder.hpp"
#include "snippets/op/reshape.hpp"
#include "snippets/op/scalar.hpp"
#include "snippets/op/store.hpp"
#include "snippets/op/vector_buffer.hpp"
#include "snippets/runtime_configurator.hpp"
#include "snippets/target_machine.hpp"
#include "snippets/utils/reg_utils.hpp"

namespace ov::snippets {

LoweringResult Generator::generate(const lowered::LinearIRPtr& linear_ir, const void* compile_params) const {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::Generator::generate")

    // Before code gen we have to reset KernelExecutor Table - it should be empty
    const std::shared_ptr<ov::snippets::RuntimeConfigurator>& runtime_configurator = target->get_runtime_configurator();
    runtime_configurator->reset_kernel_executor_table();

    OV_ITT_TASK_CHAIN(GENERATE, ov::pass::itt::domains::SnippetsTransform, "Snippets::Generator", "::InitEmitters")

    OPENVINO_ASSERT(target->is_supported(), "unsupported architecture for code generation");
    linear_ir->init_emitters(target);

    OV_ITT_TASK_NEXT(GENERATE, "::EmitCode")
    const auto kernel_op = op::Kernel::make_kernel(linear_ir->is_dynamic(), *linear_ir);
    kernel_op->compile_params = compile_params;
    const lowered::RegManager reg_manager(shared_from_this());
    const auto kernel_expr = linear_ir->get_expr_factory()->build(kernel_op, std::vector<lowered::PortConnectorPtr>{});
    const auto kernel = target->get(kernel_expr->get_node()->get_type_info())(kernel_expr);

    kernel->emit_code(utils::transform_snippets_regs_to_idxs(reg_manager.get_kernel_call_regs(kernel_op)),
                      {},
                      utils::transform_snippets_regs_to_idxs(reg_manager.get_vec_reg_pool()),
                      utils::transform_snippets_regs_to_idxs(reg_manager.get_gp_regs_except_kernel_call(kernel_op)));

    OV_ITT_TASK_NEXT(GENERATE, "::EmitData")
    for (const auto& l : linear_ir->get_ops()) {
        l->get_emitter()->emit_data();
    }
    OV_ITT_TASK_NEXT(GENERATE, "::GetSnippet")

    LoweringResult result;
    // 1. some emitters use precompiled kernels. They need to be saved, so the kernels are accessible at runtime.
    // 2. perf count node as field of emitter should be alive at runtime.
    // 3. Emitters with segfault detector debug capabilty also need to be accessible at runtime.
    for (const auto& expr : *linear_ir) {
        const auto& emitter = expr->get_emitter();
        if (uses_precompiled_kernel(emitter)) {
            result.m_saved_emitters.emplace_back(emitter);
        }
    }
    result.compiled_snippet = target->get_snippet();
    result.kernel_executor_table = runtime_configurator->get_kernel_executor_table();
    // In static case some kernel executors might've been registered during code emission.
    // We need to update them, so appropriate kernels will be compiled.
    // In dynamic case it should be handled by RuntimeConfigurator
    if (!linear_ir->is_dynamic()) {
        result.kernel_executor_table->update_state(linear_ir);
    }

    return result;
}

std::shared_ptr<const TargetMachine> Generator::get_target_machine() const {
    return target;
}

RegType Generator::get_op_out_reg_type(const ov::Output<Node>& out) const {
    auto reg_type = get_specific_op_out_reg_type(out);
    if (reg_type != RegType::undefined) {
        return reg_type;
    }
    const auto op = out.get_node_shared_ptr();
    if (is_type_any_of<ov::op::v0::Parameter,
                       ov::op::v0::Result,
                       op::LoopBegin,
                       op::LoopEnd,
                       op::Brgemm,
                       op::Buffer,
                       op::RankNormalization,
                       op::Reshape,
                       op::Reorder,
                       snippets::op::Store>(op)
#ifdef SNIPPETS_DEBUG_CAPS
        || is_type_any_of<op::PerfCountBeginBase, op::PerfCountEndBase>(op)
#endif
    )
        return RegType::gpr;
    if (ov::op::util::is_unary_elementwise_arithmetic(op) || ov::op::util::is_binary_elementwise_arithmetic(op) ||
        ov::op::util::is_binary_elementwise_comparison(op) || ov::op::util::is_binary_elementwise_logical(op) ||
        is_type_any_of<snippets::op::Load,
                       snippets::op::BroadcastLoad,
                       ov::op::v1::LogicalNot,
                       ov::op::v0::PRelu,
                       ov::op::v0::Convert,
                       ov::op::v1::Select,
                       op::VectorBuffer,
                       op::BroadcastMove,
                       op::Scalar,
                       op::HorizonMax,
                       op::HorizonSum,
                       op::Fill>(op)) {
        return RegType::vec;
    }
    OPENVINO_THROW("Register type of the operation " + std::string(op->get_type_name()) + " isn't determined!");
}

RegType Generator::get_specific_op_out_reg_type(const ov::Output<Node>& out) const {
    OPENVINO_THROW("Register type of the operation " + std::string(out.get_node()->get_type_name()) +
                   " isn't determined!");
}

}  // namespace ov::snippets
