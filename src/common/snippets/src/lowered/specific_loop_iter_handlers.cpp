// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/specific_loop_iter_handlers.hpp"

#include "snippets/lowered/pass/iter_handler.hpp"
#include "snippets/lowered/pass/propagate_subtensors.hpp"
#include "snippets/utils.hpp"


namespace ov {
namespace snippets {
namespace lowered {

SpecificIterationHandlers::SpecificIterationHandlers(size_t loop_work_amount, size_t loop_increment) {
    const auto tail_size = utils::is_dynamic_value(loop_work_amount) ? 1lu : loop_work_amount % loop_increment;
    if (tail_size != 0) {
        m_last_iter_handlers.register_pass<lowered::pass::UpdateMemoryAccessCounts>(tail_size);
        m_last_iter_handlers.register_pass<lowered::pass::UpdateSubtensors>(tail_size);
    }
}

SpecificIterationHandlers::SpecificIterationHandlers(lowered::pass::PassPipeline first_iter_handlers,
                                                     lowered::pass::PassPipeline main_body_handlers,
                                                     lowered::pass::PassPipeline last_iter_handlers)
    : m_first_iter_handlers(std::move(first_iter_handlers)),
      m_main_body_handlers(std::move(main_body_handlers)),
      m_last_iter_handlers(std::move(last_iter_handlers)) {}

const pass::PassPipeline& SpecificIterationHandlers::get_first_iter_handlers() const {
    return m_first_iter_handlers;
}

const pass::PassPipeline& SpecificIterationHandlers::get_main_iter_handlers() const {
    return m_main_body_handlers;
}

const pass::PassPipeline& SpecificIterationHandlers::get_last_iter_handlers() const {
    return m_last_iter_handlers;
}

const pass::PassPipeline& SpecificIterationHandlers::get_handlers_by_type(SpecificLoopIterType Type) const {
    switch (Type) {
        case (SpecificLoopIterType::FIRST_ITER):
            return get_first_iter_handlers();
        case (SpecificLoopIterType::MAIN_BODY):
            return get_main_iter_handlers();
        case (SpecificLoopIterType::LAST_ITER):
            return get_last_iter_handlers();
        default:
            OPENVINO_THROW("Unknown SpecificLoopIterType type!");
    }
}

SpecificIterationHandlers SpecificIterationHandlers::merge_handlers(
    const SpecificIterationHandlers& lhs,
    const SpecificIterationHandlers& rhs) {
    return SpecificIterationHandlers(
        pass::PassPipeline::merge_pipelines(lhs.get_first_iter_handlers(), rhs.get_first_iter_handlers()),
        pass::PassPipeline::merge_pipelines(lhs.get_main_iter_handlers(), rhs.get_main_iter_handlers()),
        pass::PassPipeline::merge_pipelines(lhs.get_last_iter_handlers(), rhs.get_last_iter_handlers()));
}

} // namespace lowered
} // namespace snippets
} // namespace ov
