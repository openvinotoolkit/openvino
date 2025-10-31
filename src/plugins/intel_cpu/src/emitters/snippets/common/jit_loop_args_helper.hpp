// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::intel_cpu::snippets_common {

inline jit_snippets_call_args::loop_args_t compose_loop_args(
    const std::shared_ptr<ov::snippets::op::LoopEnd>& loop_end) {
    const auto& ptr_increments = loop_end->get_ptr_increments();
    const auto& fin_offsets = loop_end->get_finalization_offsets();
    const auto& is_incremented = loop_end->get_is_incremented();
    const auto wa_increment = loop_end->get_increment();

    const auto int_work_amount = ov::snippets::utils::is_dynamic_value(loop_end->get_work_amount())
                                     ? ov::snippets::utils::get_dynamic_value<int64_t>()
                                     : static_cast<int64_t>(loop_end->get_work_amount());
    auto loop_args = jit_snippets_call_args::loop_args_t(int_work_amount, ptr_increments, fin_offsets);

    const auto& data_sizes = loop_end->get_element_type_sizes();
    for (int64_t i = 0; i < loop_args.m_num_data_ptrs; ++i) {
        if (!is_incremented[i]) {
            loop_args.m_ptr_increments[i] = 0;
            loop_args.m_finalization_offsets[i] = 0;
            continue;
        }

        if (!ov::snippets::utils::is_dynamic_value(loop_args.m_ptr_increments[i])) {
            loop_args.m_ptr_increments[i] *= (wa_increment * data_sizes[i]);
        }
        if (!ov::snippets::utils::is_dynamic_value(loop_args.m_finalization_offsets[i])) {
            loop_args.m_finalization_offsets[i] *= data_sizes[i];
        }
    }

    return loop_args;
}

}  // namespace ov::intel_cpu::snippets_common
