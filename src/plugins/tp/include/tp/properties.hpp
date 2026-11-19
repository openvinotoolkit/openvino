// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <string>

#include "openvino/core/except.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {
namespace tp {

/// How the model is partitioned across the ranks.
enum class TPStrategy {
    COLUMN_ROW = 0,  //!< Megatron-style column / row parallelism
};

inline std::ostream& operator<<(std::ostream& os, const TPStrategy& strategy) {
    switch (strategy) {
    case TPStrategy::COLUMN_ROW:
        return os << "COLUMN_ROW";
    default:
        OPENVINO_THROW("[TP] Unsupported partitioning strategy");
    }
}

inline std::istream& operator>>(std::istream& is, TPStrategy& strategy) {
    std::string str;
    is >> str;
    if (str == "COLUMN_ROW") {
        strategy = TPStrategy::COLUMN_ROW;
    } else {
        OPENVINO_THROW("[TP] Unsupported partitioning strategy \"", str, "\"; supported: COLUMN_ROW");
    }
    return is;
}

/// Partitioning strategy.
static constexpr Property<TPStrategy> tp_strategy{"TP_STRATEGY"};

/// Collective implementation of the ranks' device plugin, passed to it unchanged. The device plugin
/// defines the names, resolves "AUTO", and rejects names it does not know.
static constexpr Property<std::string> collective_backend{"TP_COLLECTIVE_BACKEND"};

/// Milliseconds a rank waits inside a collective before the whole group is aborted; 0 waits indefinitely.
static constexpr Property<int64_t> communication_timeout_ms{"TP_COMMUNICATION_TIMEOUT_MS"};

/// Bytes of collective staging per rank, allocated once at compile_model; larger messages are split
/// into chunks. 0 lets the staging grow with the largest message.
static constexpr Property<size_t> comm_buffer_size{"TP_COMM_BUFFER_SIZE"};

}  // namespace tp
}  // namespace ov
