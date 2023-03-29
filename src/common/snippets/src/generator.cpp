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
#include "snippets/pass/lowered/loop_markup.hpp"
#include "snippets/pass/lowered/loop_fusion.hpp"
#include "snippets/pass/lowered/loop_init.hpp"
#include "snippets/pass/lowered/buffer_insertion.hpp"
#include "snippets/pass/lowered/load_store_insertion.hpp"
#include "snippets/pass/lowered/vector_to_scalar.hpp"
#include "snippets/pass/lowered/load_movebroadcast_to_broadcastload.hpp"
#include "snippets/pass/lowered/buffer_allocation.hpp"
#include "snippets/pass/lowered/propagate_layout.hpp"
#include "snippets/pass/lowered/cleanup_loop_offsets.hpp"
#include "snippets/pass/lowered/softmax_decomposition.hpp"
#include "snippets/pass/lowered/move_scalar_to_consumer.hpp"
#include "snippets/pass/lowered/move_result_out_of_loop.hpp"
#include "snippets/tensor_descriptor.hpp"

namespace ngraph {
namespace snippets {

Generator::LoweringResult Generator::generate(std::shared_ptr<ov::Model>& m, const LoweringConfig& config, const void* compile_params) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::Generator::generate")
    OV_ITT_TASK_CHAIN(GENERATE, ngraph::pass::itt::domains::SnippetsTransform, "Snippets::Generator", "::Transformations")
    if (!target->is_supported())
        OPENVINO_THROW("unsupported architecture for code generation");

    auto linear_ir = LoweredExprIR(m, config);
    const size_t vector_size = get_target_machine()->get_lanes();
    const int32_t buffer_allocation_rank = static_cast<int32_t>(config.m_loop_depth);

    // Note: The pass LoopInit uses LoopInfo that contains entry and exit points of the corresponding Loop.
    //       To avoid the Loop information corruption, we should call the passes with Load/Store work
    //       (for example, LoadMoveBroadcastToBroadcastLoad()) after explicit Loop insertion (LoopInit())
    const auto buffer_allocation_pass = std::make_shared<pass::lowered::BufferAllocation>();
    pass::lowered::LinearIRTransformationPipeline common_pipeline;
    common_pipeline.register_transformation<pass::lowered::LoopMarkup>(vector_size);
    common_pipeline.register_transformation<pass::lowered::SoftmaxDecomposition>(vector_size);
    common_pipeline.register_transformation<pass::lowered::LoopFusion>();
    common_pipeline.register_transformation<pass::lowered::MoveResultOutOfLoop>();
    common_pipeline.register_transformation<pass::lowered::BufferInsertion>(buffer_allocation_rank);
    common_pipeline.register_transformation<pass::lowered::LoadStoreInsertion>(vector_size);
    common_pipeline.register_transformation<pass::lowered::SetScalarCountForLoadStore>();
    common_pipeline.register_transformation<pass::lowered::LoopInit>();
    common_pipeline.register_transformation<pass::lowered::MoveScalarToConsumer>();
    common_pipeline.register_transformation<pass::lowered::LoadMoveBroadcastToBroadcastLoad>();
    common_pipeline.register_transformation<pass::lowered::PropagateLayout>();
    common_pipeline.register_transformation(buffer_allocation_pass);
    common_pipeline.register_transformation<pass::lowered::CleanupLoopOffsets>();
    common_pipeline.run(linear_ir);

    pass::lowered::LinearIRTransformationPipeline target_pipeline = target_specific_transformations();
    target_pipeline.run(linear_ir);

    std::function<opRegType(const std::shared_ptr<Node>& op)> reg_type_mapper = [&](const std::shared_ptr<Node>& op) -> opRegType {
        return get_op_reg_type(op);
    };

    pass::lowered::LinearIRTransformationPipeline final_pipeline;
    final_pipeline.register_transformation<pass::lowered::AssignRegisters>(reg_type_mapper);
    final_pipeline.register_transformation<pass::lowered::InsertTailLoop>();
    final_pipeline.run(linear_ir);

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

    return {target->get_snippet(), buffer_allocation_pass->get_scratchpad_size()};
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

pass::lowered::LinearIRTransformationPipeline Generator::target_specific_transformations() const {
    return pass::lowered::LinearIRTransformationPipeline();
}

}// namespace snippets
}// namespace ngraph
