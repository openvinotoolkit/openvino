// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <cstdint>

#include "onednn/iml_type_mapper.h"
#include "openvino/runtime/system_conf.hpp"

// What the dense QSAIndexer and the PagedQSAIndexer nodes share outside the cross-compiled kernels.
namespace ov::intel_cpu::node::qsa {

// The implementation type of a node that does not score on brgemm: the instruction set its cross-compiled kernel is
// dispatched to on this CPU.
inline impl_desc_type kernel_impl_type() {
    if (ov::with_cpu_x86_avx512f()) {
        return impl_desc_type::avx512;
    }
    return ov::with_cpu_x86_avx2() ? impl_desc_type::avx2 : impl_desc_type::ref;
}

// What one row of the batch contributes to a call of the dense QSAIndexer. Every quantity is derived from the row's
// own key_length and the window length L of the call alone -- no cross-call running counter is kept.
struct RowPlan {
    int64_t num_padding;  // left-padded positions of this call's window, skipped everywhere
    int64_t current;      // real tokens of this call
    int64_t tail;         // rows already sitting in the incomplete-block state
    int64_t past_blocks;  // complete blocks of the earlier calls
    int64_t new_blocks;   // blocks this call completes

    // Complete blocks visible to the token `v` of this call (`v` counts the real tokens, left padding excluded).
    [[nodiscard]] int64_t blocks_visible(int64_t v, int64_t ratio) const {
        return past_blocks + (tail + v + 1) / ratio;
    }
};

inline RowPlan make_row_plan(int64_t key_length, int64_t window, int64_t ratio) {
    const int64_t past = std::max<int64_t>(0, key_length - window);
    RowPlan plan{};
    plan.current = key_length - past;  // == min(window, key_length)
    plan.num_padding = window - plan.current;
    plan.tail = past % ratio;
    plan.past_blocks = past / ratio;
    plan.new_blocks = (plan.tail + plan.current) / ratio;
    return plan;
}

}  // namespace ov::intel_cpu::node::qsa
