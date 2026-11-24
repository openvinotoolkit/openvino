// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "exec_axis.h"

#include "intel_gpu/graph/program.hpp"
#include "program_node.h"
#include "runtime_concat_order.h"

namespace cldnn {

exec_axis exec_axis::build(const program& p) {
    exec_axis ax;
    if (p.get_config().get_queue_type() != QueueTypes::in_order) {
        ax.m_status = status::not_in_order;
        return ax;
    }
    const auto filtered = runtime_concat_order::filtered_processing_order(p.get_processing_order());
    if (!runtime_concat_order::is_normalized(filtered)) {
        ax.m_status = status::not_normalized;
        return ax;
    }
    ax.m_status = status::ok;
    ax.m_size = filtered.size();
    ax.m_pos.reserve(filtered.size());
    for (size_t i = 0; i < filtered.size(); ++i) {
        ax.m_pos.emplace(filtered[i], static_cast<uint32_t>(i));
    }
    return ax;
}

exec_axis exec_axis::for_test(status st, std::unordered_map<const program_node*, uint32_t> positions) {
    exec_axis ax;
    ax.m_status = st;
    ax.m_size = positions.size();
    ax.m_pos = std::move(positions);
    return ax;
}

std::optional<uint32_t> exec_axis::position(const program_node& n) const {
    if (m_status != status::ok) {
        return std::nullopt;
    }
    auto it = m_pos.find(&n);
    if (it == m_pos.end()) {
        return std::nullopt;
    }
    return it->second;
}

}  // namespace cldnn
