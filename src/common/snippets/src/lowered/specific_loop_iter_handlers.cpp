// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/specific_loop_iter_handlers.hpp"

#include "snippets/lowered/pass/iter_handler.hpp"
#include "snippets/lowered/pass/propagate_subtensors.hpp"
#include "snippets/utils/utils.hpp"


namespace ov {
namespace snippets {
namespace lowered {

SpecificIterationHandlers::SpecificIterationHandlers(size_t loop_work_amount, size_t loop_increment, size_t processing_dim_idx) {
    // The following handlers are set only for Last Iter processing
    if (loop_increment > 1) {
        size_t last_iter_increment = utils::get_dynamic_value<size_t>();
        if (!utils::is_dynamic_value(loop_work_amount)) {
            last_iter_increment = loop_work_amount % loop_increment;
        } else if (utils::is_dynamic_value(loop_work_amount) && processing_dim_idx == 0) {
            // [149935] : Last Iterations of Loop processed last dimensions with Eltwise nodes inside should have increment = 1
            last_iter_increment = 1;
        }
        if (last_iter_increment != 0) {
            m_last_iter_handlers.register_pass<lowered::pass::UpdateMemoryAccessCounts>(last_iter_increment);
            m_last_iter_handlers.register_pass<lowered::pass::UpdateSubtensors>(last_iter_increment);
            // Last Iterations of Loop processed last dimensions with Eltwise nodes inside should have increment = 1
            if (last_iter_increment == 1)
                m_last_iter_handlers.register_pass<lowered::pass::SetLoopIncrementOne>();
        }
    }
}

SpecificIterationHandlers::SpecificIterationHandlers(lowered::pass::PassPipeline first_iter_handlers,
                                                     lowered::pass::PassPipeline main_body_handlers,
                                                     lowered::pass::PassPipeline last_iter_handlers)
    : m_first_iter_handlers(std::move(first_iter_handlers)),
      m_main_body_handlers(std::move(main_body_handlers)),
      m_last_iter_handlers(std::move(last_iter_handlers)) {}

SpecificIterationHandlers SpecificIterationHandlers::merge_handlers(
    const SpecificIterationHandlers& lhs,
    const SpecificIterationHandlers& rhs) {
    return SpecificIterationHandlers(
        pass::PassPipeline::merge_pipelines(lhs.m_first_iter_handlers, rhs.m_first_iter_handlers),
        pass::PassPipeline::merge_pipelines(lhs.m_main_body_handlers, rhs.m_main_body_handlers),
        pass::PassPipeline::merge_pipelines(lhs.m_last_iter_handlers, rhs.m_last_iter_handlers));
}

} // namespace lowered
} // namespace snippets
} // namespace ov
