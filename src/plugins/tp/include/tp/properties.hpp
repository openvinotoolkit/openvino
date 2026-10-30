// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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
        OPENVINO_THROW("TP: unsupported partitioning strategy");
    }
}

inline std::istream& operator>>(std::istream& is, TPStrategy& strategy) {
    std::string str;
    is >> str;
    if (str == "COLUMN_ROW") {
        strategy = TPStrategy::COLUMN_ROW;
    } else {
        OPENVINO_THROW("TP: unsupported partitioning strategy \"", str, "\"; supported: COLUMN_ROW");
    }
    return is;
}

/// Partitioning strategy.
static constexpr Property<TPStrategy> tp_strategy{"TP_STRATEGY"};

/// Collective implementation of the ranks' device plugin, passed to it unchanged. The device plugin
/// defines the names, resolves "AUTO", and rejects names it does not know.
static constexpr Property<std::string> collective_backend{"COLLECTIVE_BACKEND"};

}  // namespace tp
}  // namespace ov
