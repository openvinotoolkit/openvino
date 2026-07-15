// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "intel_gpu/runtime/layout.hpp"
#include "openvino/core/shape_util.hpp"

namespace cldnn {

// Computes a lower-rank representation of a fused eltwise peer when collapsing its leading spatial
// axes is an order-preserving reshape and the result broadcasts directly to the host layout.
inline std::optional<ov::PartialShape> fold_higher_rank_fused_peer(const layout& peer_layout, const layout& host_layout) {
    const auto& peer_shape = peer_layout.get_partial_shape();
    const auto& host_shape = host_layout.get_partial_shape();

    const size_t peer_rank = peer_shape.size();
    const size_t host_rank = host_shape.size();
    if (peer_rank <= host_rank || host_rank < 3)
        return std::nullopt;
    if (peer_shape.is_dynamic() || host_shape.is_dynamic())
        return std::nullopt;
    if (peer_layout.data_padding || host_layout.data_padding)
        return std::nullopt;

    const auto& peer_format = peer_layout.format;
    const auto& host_format = host_layout.format;
    if (!format::is_simple_data_format(peer_format) || !format::is_simple_data_format(host_format))
        return std::nullopt;
    if (!format::is_default_format(peer_format) || !format::is_default_format(host_format))
        return std::nullopt;
    if (format::adjust_to_rank(peer_format, host_rank) != host_format)
        return std::nullopt;

    const auto peer_dims = peer_shape.to_shape();
    const auto host_dims = host_shape.to_shape();
    const size_t fold_count = peer_rank - host_rank + 1;
    ov::Shape folded_dims;
    folded_dims.reserve(host_rank);
    folded_dims.push_back(peer_dims[0]);
    folded_dims.push_back(peer_dims[1]);

    size_t grouped = 1;
    for (size_t i = 2; i < 2 + fold_count; ++i) {
        const auto grouped_size = ov::util::shape_size_safe({grouped, peer_dims[i]});
        if (!grouped_size.has_value())
            return std::nullopt;
        grouped = grouped_size.value();
    }
    folded_dims.push_back(grouped);
    folded_dims.insert(folded_dims.end(), peer_dims.begin() + 2 + fold_count, peer_dims.end());

    const auto peer_total = ov::util::shape_size_safe(peer_dims);
    const auto folded_total = ov::util::shape_size_safe(folded_dims);
    if (!peer_total.has_value() || !folded_total.has_value() || peer_total.value() != folded_total.value())
        return std::nullopt;
    if (folded_dims.size() != host_dims.size())
        return std::nullopt;

    for (size_t i = 0; i < folded_dims.size(); ++i) {
        if (folded_dims[i] != 1 && folded_dims[i] != host_dims[i])
            return std::nullopt;
    }

    return ov::PartialShape(folded_dims);
}

}  // namespace cldnn