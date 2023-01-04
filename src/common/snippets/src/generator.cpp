// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/generator.hpp"
#include "snippets/lowered_expr.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/op/kernel.hpp"
#include <snippets/itt.hpp>
#include "snippets/pass/lowered/assign_registers.hpp"
#include "snippets/pass/lowered/insert_tail_loop.hpp"
#include "snippets/pass/lowered/insert_loops_layout.hpp"
#include "snippets/pass/lowered/move_scalar_to_consumer.hpp"
#include "snippets/pass/lowered/buffer_propagate_offset_and_reset.hpp"
#include "snippets/pass/lowered/propagate_layout.hpp"
#include "snippets/pass/lowered/cleanup_loop_offsets.hpp"
#include "snippets/pass/lowered/softmax_decomposition.hpp"
#include "snippets/lowered_expr.hpp"
#include "snippets/tensor_descriptor.hpp"

namespace ngraph {
namespace snippets {

Generator::LoweringResult Generator::generate(std::shared_ptr<ov::Model>& m, const LoweringConfig& config, const void* compile_params) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::Generator::generate")
    OV_ITT_TASK_CHAIN(GENERATE, ngraph::pass::itt::domains::SnippetsTransform, "Snippets::Generator", "::Transformations")
    if (!target->is_supported())
        OPENVINO_THROW("unsupported architecture for code generation");

    auto linear_ir = LoweredExprIR(m, config);
    const size_t vector_size = target->get_lanes();
    // todo: fix buffer allocation rank
    const int32_t buffer_allocation_rank = -1;
    auto propagate_buffer_offsets = std::make_shared<pass::lowered::PropagateOffsetAndResetBuffer>();
    std::vector<std::shared_ptr<pass::lowered::LinearIRTransformation>> transformation_pipeline {
            std::make_shared<pass::lowered::InsertLoopsLayout>(vector_size, buffer_allocation_rank),
            std::make_shared<pass::lowered::SoftmaxDecomposition>(vector_size, buffer_allocation_rank),
            std::make_shared<pass::lowered::MoveScalarToConsumer>(),
            std::make_shared<pass::lowered::PropagateLayout>(),
            propagate_buffer_offsets,
            std::make_shared<pass::lowered::CleanupLoopOffsets>(),
            std::make_shared<pass::lowered::AssignRegisters>(get_op_reg_type),
            std::make_shared<pass::lowered::InsertTailLoop>()
    };
    for (const auto& transform : transformation_pipeline) {
        transform->run(linear_ir);
    }
    const auto buffer_scratchpad_size = propagate_buffer_offsets->get_scratchpad_size();
    linear_ir.init_emitters(target);
    OV_ITT_TASK_NEXT(GENERATE, "::EmitCode")
    auto loops2DKernel = std::make_shared<op::Kernel>(linear_ir);
    loops2DKernel->compile_params = compile_params;
    std::shared_ptr<Emitter> kernel = target->get(op::Kernel::get_type_info_static())(loops2DKernel);

    kernel->emit_code({}, {});

    OV_ITT_TASK_NEXT(GENERATE, "::EmitData")
    for (auto& l : linear_ir.get_ops()) {
        l->get_emitter()->emit_data();
    }
    OV_ITT_TASK_NEXT(GENERATE, "::GetSnippet")

    // todo: we save lowered to access compiled brgemm kernels on execution time (normally lowered is destructed by then)
    //  remove this when kernel caching is implemented. Don't forget to make generate const method.
    if (config.m_save_lowered_code)
        lowered_saved = linear_ir;

    return {target->get_snippet(), buffer_scratchpad_size};
}

std::shared_ptr<const TargetMachine> Generator::get_target_machine() const {
    return target;
}

Generator::opRegType Generator::get_op_reg_type(const std::shared_ptr<Node>& op) const {
    if (std::dynamic_pointer_cast<opset1::Parameter>(op) ||
        std::dynamic_pointer_cast<opset1::Result>(op) ||
        std::dynamic_pointer_cast<op::LoopBegin>(op) ||
        std::dynamic_pointer_cast<op::LoopEnd>(op) ||
        std::dynamic_pointer_cast<op::Brgemm>(op) ||
        std::dynamic_pointer_cast<op::Buffer>(op))
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
             std::dynamic_pointer_cast<opset1::LogicalNot>(op) ||
             std::dynamic_pointer_cast<opset1::PRelu>(op) ||
             std::dynamic_pointer_cast<opset1::Convert>(op) ||
             std::dynamic_pointer_cast<opset1::Select>(op) ||
             std::dynamic_pointer_cast<op::VectorBuffer>(op) ||
             std::dynamic_pointer_cast<op::BroadcastMove>(op) ||
             std::dynamic_pointer_cast<op::Scalar>(op) ||
             std::dynamic_pointer_cast<op::HorizonMax>(op) ||
             std::dynamic_pointer_cast<op::HorizonSum>(op))
        return vec2vec;
    else
        return get_specific_op_reg_type(op);
}

Generator::opRegType Generator::get_specific_op_reg_type(const std::shared_ptr<ov::Node>& op) const {
    OPENVINO_THROW("Register type of the operation " + std::string(op->get_type_name()) + " isn't determined!");
}


}// namespace snippets
}// namespace ngraph
