// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/pass.hpp"
#include "snippets/lowered/specific_loop_iter_types.hpp"


namespace ov {
namespace snippets {
namespace lowered {

class SpecificIterationHandlers {
public:
    SpecificIterationHandlers() = default;
    SpecificIterationHandlers(size_t loop_work_amount, size_t loop_increment, size_t processing_dim_idx);
    SpecificIterationHandlers(pass::PassPipeline first_iter_handlers,
                              pass::PassPipeline main_body_handlers,
                              pass::PassPipeline last_iter_handlers);

    template <SpecificLoopIterType Type,
              typename std::enable_if<Type == SpecificLoopIterType::FIRST_ITER, bool>::type = true>
    const pass::PassPipeline& get_passes() const {
        return m_first_iter_handlers;
    }

    template <SpecificLoopIterType Type,
              typename std::enable_if<Type == SpecificLoopIterType::MAIN_BODY, bool>::type = true>
    const pass::PassPipeline& get_passes() const {
        return m_main_body_handlers;
    }

    template <SpecificLoopIterType Type,
              typename std::enable_if<Type == SpecificLoopIterType::LAST_ITER, bool>::type = true>
    const pass::PassPipeline& get_passes() const {
        return m_last_iter_handlers;
    }

    template <SpecificLoopIterType Type,
              typename T,
              class... Args,
              typename std::enable_if<Type == SpecificLoopIterType::FIRST_ITER, bool>::type = true>
    void register_pass(Args&&... args) {
        m_first_iter_handlers.register_pass<T>(args...);
    }

    template <SpecificLoopIterType Type,
              typename T,
              class... Args,
              typename std::enable_if<Type == SpecificLoopIterType::MAIN_BODY, bool>::type = true>
    void register_pass(Args&&... args) {
        m_main_body_handlers.register_pass<T>(args...);
    }

    template <SpecificLoopIterType Type,
              typename T,
              class... Args,
              typename std::enable_if<Type == SpecificLoopIterType::LAST_ITER, bool>::type = true>
    void register_pass(Args&&... args) {
        m_last_iter_handlers.register_pass<T>(args...);
    }

    static SpecificIterationHandlers merge_handlers(const SpecificIterationHandlers& lhs, const SpecificIterationHandlers& rhs);

private:
    pass::PassPipeline m_first_iter_handlers;
    pass::PassPipeline m_main_body_handlers;
    pass::PassPipeline m_last_iter_handlers;
};

} // namespace lowered
} // namespace snippets
} // namespace ov
