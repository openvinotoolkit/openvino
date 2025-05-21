// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jitter.hpp"

#include <cstddef>

#include "openvino/core/dimension.hpp"
#include "openvino/core/except.hpp"

namespace ov::intel_gpu {
namespace {

std::vector<ChannelName> get_data_channels_order(size_t rank) {
    using channel = ChannelName;
    switch (rank) {
    case 1:
        return {channel::BATCH};
    case 2:
        return {channel::BATCH, channel::FEATURE};
    case 3:
        return {channel::BATCH, channel::FEATURE, channel::Y};
    case 4:
        return {channel::BATCH, channel::FEATURE, channel::Y, channel::X};
    case 5:
        return {channel::BATCH, channel::FEATURE, channel::Z, channel::Y, channel::X};
    case 6:
        return {channel::BATCH, channel::FEATURE, channel::W, channel::Z, channel::Y, channel::X};
    case 7:
        return {channel::BATCH, channel::FEATURE, channel::U, channel::W, channel::Z, channel::Y, channel::X};
    case 8:
        return {channel::BATCH, channel::FEATURE, channel::V, channel::U, channel::W, channel::Z, channel::Y, channel::X};
    default:
        OPENVINO_ASSERT("[GPU] Unexpected rank ", rank, " in get_data_channels_order() func");
    }

    return {};
}

std::vector<ChannelName> get_weights_channels_order(size_t rank, bool is_grouped) {
    using channel = ChannelName;
    if (is_grouped) {
        switch (rank) {
        case 4:
            return {channel::G, channel::OFM, channel::IFM, channel::Y, channel::X};
        case 5:
            return {channel::G, channel::OFM, channel::IFM, channel::Z, channel::Y, channel::X};
        case 6:
            return {channel::G, channel::OFM, channel::IFM, channel::Z, channel::Y, channel::X};
        default:
            break;
        }
    } else {
        switch (rank) {
        case 3:
            return {channel::OFM, channel::IFM, channel::X};
        case 4:
            return {channel::OFM, channel::IFM, channel::Y, channel::X};
        case 5:
            return {channel::OFM, channel::IFM, channel::Z, channel::Y, channel::X};
        default:
            break;
        }
    }

    return {};
}

}  // namespace

std::vector<ChannelName> get_default_channels_order(size_t rank, bool is_weights_fmt, bool is_grouped) {
    if (is_weights_fmt) {
        return get_weights_channels_order(rank, is_grouped);
    }
    return get_data_channels_order(rank);
}

int get_channel_index(ChannelName channel_name, size_t rank, bool is_weights_fmt, bool is_grouped) {
    auto order = get_default_channels_order(rank, is_weights_fmt, is_grouped);
    auto it = std::find(order.begin(), order.end(), channel_name);
    if (it == order.end()) {
        return -1;
    }

    return static_cast<int>(std::distance(order.begin(), it));
}

size_t extract_channel(ChannelName channel, const cldnn::layout& l) {
    using cldnn::format;
    const auto& pshape = l.get_partial_shape();
    auto idx = get_channel_index(channel, pshape.size(), format::is_weights_format(l.format), format::is_grouped(l.format));
    return (idx < 0 || idx >= static_cast<int>(pshape.size())) ? 1 : static_cast<size_t>(pshape[idx].get_length());
}

}  // namespace ov::intel_gpu
