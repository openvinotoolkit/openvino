// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/generator.hpp"
#include "snippets/lowered_expr.hpp"
#include "snippets/pass/assign_registers.hpp"
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
#include "snippets/pass/lowered/buffer_propagate_offset_and_reset.hpp"
#include "snippets/pass/lowered/propagate_layout.hpp"
#include "snippets/pass/lowered/cleanup_loop_offsets.hpp"
#include "snippets/pass/lowered/softmax_decomposition.hpp"
#include "snippets/pass/lowered/move_scalar_to_consumer.hpp"
#include "snippets/tensor_descriptor.hpp"

namespace ngraph {
namespace snippets {

Generator::LoweringResult Generator::generate(std::shared_ptr<ov::Model>& m, const LoweringConfig& config, const void* compile_params) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::Generator::generate")
    OV_ITT_TASK_CHAIN(GENERATE, ngraph::pass::itt::domains::SnippetsTransform, "Snippets::Generator", "::Transformations")
    if (!target->is_supported())
        throw ngraph_error("unsupported architecture for code generation");

    auto linear_ir = LoweredExprIR(m, config);

    // At the moment we support only full vector Load/Store and scalar Load/Store so that count is equal to lanes.
    // Then we are going to support variadic Load/Store with different element count
    const size_t vector_size = get_target_machine()->get_lanes();
    const int32_t buffer_allocation_rank = static_cast<int32_t>(config.m_loop_depth);

    // Note: The pass LoopInit uses LoopInfo that contains entry and exit points of the corresponding Loop.
    //       To avoid the Loop information corruption, we should call the passes with Load/Store work
    //       (for example, LoadMoveBroadcastToBroadcastLoad()) after explicit Loop insertion (LoopInit())
    auto propagate_buffer_offsets = std::make_shared<pass::lowered::PropagateOffsetAndResetBuffer>();
    std::vector<std::shared_ptr<pass::lowered::LinearIRTransformation>> transformation_pipeline {
            std::make_shared<pass::lowered::LoopMarkup>(vector_size),
            std::make_shared<pass::lowered::SoftmaxDecomposition>(vector_size),
            std::make_shared<pass::lowered::LoopFusion>(),
            std::make_shared<pass::lowered::BufferInsertion>(buffer_allocation_rank),
            std::make_shared<pass::lowered::LoadStoreInsertion>(vector_size),
            std::make_shared<pass::lowered::SetScalarCountForLoadStore>(),
            std::make_shared<pass::lowered::LoopInit>(),
            std::make_shared<pass::lowered::MoveScalarToConsumer>(),
            std::make_shared<pass::lowered::LoadMoveBroadcastToBroadcastLoad>(),
            std::make_shared<pass::lowered::PropagateLayout>(),
            propagate_buffer_offsets,
            std::make_shared<pass::lowered::CleanupLoopOffsets>(),
            std::make_shared<pass::lowered::AssignRegisters>(),
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

}// namespace snippets
}// namespace ngraph
