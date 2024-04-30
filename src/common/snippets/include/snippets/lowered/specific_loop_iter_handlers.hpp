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
    SpecificIterationHandlers(size_t loop_work_amount, size_t loop_increment);
    SpecificIterationHandlers(pass::PassPipeline first_iter_handlers,
                              pass::PassPipeline main_body_handlers,
                              pass::PassPipeline last_iter_handlers);

    template <SpecificLoopIterType Type>
    const pass::PassPipeline& get_passes() const {
        switch (Type) {
            case SpecificLoopIterType::FIRST_ITER:
                return m_first_iter_handlers;
            case SpecificLoopIterType::MAIN_BODY:
                return m_main_body_handlers;
            case SpecificLoopIterType::LAST_ITER:
                return m_last_iter_handlers;
            default:
                OPENVINO_THROW("Unknown SpecificLoopIterType");
        }
    }

    template <SpecificLoopIterType Type, typename Pass, class... Args>
    void register_pass(Args&&... args) {
        switch (Type) {
            case SpecificLoopIterType::FIRST_ITER:
                return m_first_iter_handlers.register_pass<Pass>(args...);
            case SpecificLoopIterType::MAIN_BODY:
                return m_main_body_handlers.register_pass<Pass>(args...);
            case SpecificLoopIterType::LAST_ITER:
                return m_last_iter_handlers.register_pass<Pass>(args...);
            default:
                OPENVINO_THROW("Unknown SpecificLoopIterType");
        }
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
