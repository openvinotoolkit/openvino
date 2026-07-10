// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "openvino/core/any.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/weight_sharing_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/runtime/internal_properties.hpp"

namespace ov {
namespace npuw {
namespace wsh {

// Extract the weight sharing Context from a properties AnyMap. Returns nullptr
// if the property is absent. Accepts either the typed WeightSharingCtxPtr or a
// bare shared_ptr<const Context> stashed under the property's string key.
inline ov::internal::WeightSharingCtxPtr context_from(const ov::AnyMap& props) {
    auto it = props.find(ov::internal::model_sharing_context.name());
    if (it == props.end()) {
        return nullptr;
    }
    return it->second.as<ov::internal::WeightSharingCtxPtr>();
}

struct Origin {
    std::size_t offset;
    std::size_t size;
    ov::element::Type dtype;
};

// Resolve a Constant's origin metadata (offset in the source bin, byte size,
// original dtype). Order of resolution:
//   1) ov::WeightlessCacheAttribute on rt_info  -> legacy path, unchanged.
//   2) ov::weight_sharing::Context lookup via the Constant's buffer descriptor
//      (source_id from get_constant_source_id, constant_id from
//      get_constant_id) into ctx->m_weight_registry.
// Returns std::nullopt when neither source has an entry for this Constant.
inline std::optional<Origin> resolve_origin(const ov::op::v0::Constant& c,
                                            const ov::weight_sharing::Context* ctx) {
    const auto& rt_info = c.get_rt_info();
    if (auto it = rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static()); it != rt_info.end()) {
        const auto& wl = it->second.as<ov::WeightlessCacheAttribute>();
        return Origin{wl.bin_offset, wl.original_size, wl.original_dtype};
    }
    if (ctx) {
        const auto src_id = ov::weight_sharing::Extension::get_constant_source_id(c);
        const auto cst_id = ov::weight_sharing::Extension::get_constant_id(c);
        if (auto reg_it = ctx->m_weight_registry.find(src_id); reg_it != ctx->m_weight_registry.end()) {
            if (auto meta_it = reg_it->second.find(cst_id); meta_it != reg_it->second.end()) {
                const auto& m = meta_it->second;
                return Origin{m.m_offset, m.m_size, m.m_type};
            }
        }
    }
    return std::nullopt;
}

// True iff at least one Constant in the model has a resolvable origin
// (either WLCA rt_info or a matching Context registry entry).
inline bool any_origin_available(const ov::Model& model, const ov::weight_sharing::Context* ctx) {
    for (const auto& n : model.get_ordered_ops()) {
        if (!ov::op::util::is_constant(n)) {
            continue;
        }
        const auto& c = *std::static_pointer_cast<ov::op::v0::Constant>(n);
        if (resolve_origin(c, ctx).has_value()) {
            return true;
        }
    }
    return false;
}

}  // namespace wsh
}  // namespace npuw
}  // namespace ov
