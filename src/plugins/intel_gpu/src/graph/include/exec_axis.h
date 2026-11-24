// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// Phase 0.5b: one-shot snapshot of the execution axis of a compiled graph. Positions are indices into the processing order filtered to
// executable nodes (runtime_concat_order::filtered_processing_order), which equals network::_exec_order (F9) for in-order queues whenever
// the order is normalized. The axis only provides positions; lifetime intervals (P2) are built on top of it and never invent another axis.
// Pure function of the graph, no influence on allocation.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>

namespace cldnn {

struct program;
struct program_node;

class exec_axis {
public:
    enum class status : uint8_t { ok, not_in_order, not_normalized };

    exec_axis() = default;

    static exec_axis build(const program& p);

    // Test-only constructor: inject arbitrary positions (e.g. shifted ones to prove that the V0 check can fail).
    static exec_axis for_test(status st, std::unordered_map<const program_node*, uint32_t> positions);

    status get_status() const {
        return m_status;
    }
    bool valid() const {
        return m_status == status::ok;
    }
    // Not executable nodes (data, mutable_data without dependencies) and nodes of other programs return nullopt; so does an invalid axis.
    std::optional<uint32_t> position(const program_node& n) const;
    size_t size() const {
        return m_size;
    }

private:
    status m_status = status::not_in_order;
    std::unordered_map<const program_node*, uint32_t> m_pos;
    size_t m_size = 0;
};

}  // namespace cldnn
