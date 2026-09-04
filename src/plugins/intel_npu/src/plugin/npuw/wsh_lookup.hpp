// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "openvino/core/model.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/weight_sharing_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/util/op_types.hpp"

namespace ov {
namespace npuw {
namespace wsh {

struct Origin {
    std::size_t offset;
    std::size_t size;
    ov::element::Type dtype;
};

// Resolve a Constant's origin metadata (offset in the source bin, byte size,
// original dtype). Delegates to ov::weight_sharing, which resolves in this
// order:
//   1) The weight-sharing identity carried by the Constant's buffer descriptor
//      (source id + offset). Self-describing, needs no external Context.
//   2) The deprecated ov::WeightlessCacheAttribute rt_info as a transitional
//      fallback.
// Returns std::nullopt when neither source has an entry for this Constant.
inline std::optional<Origin> resolve_origin(const ov::op::v0::Constant& c) {
    if (auto origin = ov::weight_sharing::Extension::get_constant_origin(c)) {
        return Origin{origin->m_offset, origin->m_size, origin->m_type};
    }
    return std::nullopt;
}

// True iff at least one Constant in the model has a resolvable origin.
inline bool any_origin_available(const ov::Model& model) {
    for (const auto& n : model.get_ordered_ops()) {
        if (!ov::op::util::is_constant(n)) {
            continue;
        }
        const auto& c = *std::static_pointer_cast<ov::op::v0::Constant>(n);
        if (resolve_origin(c).has_value()) {
            return true;
        }
    }
    return false;
}

}  // namespace wsh
}  // namespace npuw
}  // namespace ov
