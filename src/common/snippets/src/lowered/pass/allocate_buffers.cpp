// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "snippets/lowered/pass/allocate_buffers.hpp"

#include "snippets/lowered/pass/compute_buffer_allocation_size.hpp"
#include "snippets/lowered/pass/solve_buffer_memory.hpp"
#include "snippets/lowered/pass/init_buffers_default.hpp"
#include "snippets/lowered/pass/set_buffer_reg_group.hpp"
#include "snippets/lowered/pass/define_buffer_clusters.hpp"
#include "snippets/lowered/pass/normalize_buffer_reg_groups.hpp"
#include "snippets/lowered/pass/propagate_buffer_offset.hpp"
#include "snippets/pass/tokenization.hpp"
#include "snippets/itt.hpp"
#include "snippets/utils/utils.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

AllocateBuffers::AllocateBuffers(bool is_optimized) : m_is_optimized_mode(is_optimized) {}

bool AllocateBuffers::run(lowered::LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::AllocateBuffers");
    size_t buffer_scratchpad_size = 0;

    PassPipeline pipeline;
    pipeline.register_pass<ComputeBufferAllocationSize>();
    if (m_is_optimized_mode) {
        pipeline.register_pass<SetBufferRegGroup>();
        pipeline.register_pass<DefineBufferClusters>();
        pipeline.register_pass<SolveBufferMemory>(buffer_scratchpad_size);
        pipeline.register_pass<NormalizeBufferRegisterGroups>();
    } else {
        pipeline.register_pass<InitBuffersDefault>(buffer_scratchpad_size);
    }
    pipeline.register_pass<PropagateBufferOffset>();
    pipeline.run(linear_ir, linear_ir.cbegin(), linear_ir.cend());

    linear_ir.set_static_buffer_scratchpad_size(buffer_scratchpad_size);

    return buffer_scratchpad_size > 0;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
