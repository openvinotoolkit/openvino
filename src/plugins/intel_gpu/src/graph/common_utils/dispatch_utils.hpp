// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <vector>

#include "intel_gpu/runtime/device_info.hpp"
#include "intel_gpu/runtime/format.hpp"

namespace ov::intel_gpu {

enum class ChannelName : uint8_t { X = 0, Y = 1, Z = 2, W = 3, U = 4, V = 5, FEATURE = 6, BATCH = 7, IFM = 8, OFM = 9, G = 10, UNKNOWN = 11 };

std::vector<size_t> get_optimal_lws(const std::vector<size_t>& gws,
                                    const cldnn::device_info& info,
                                    cldnn::format::type input_fmt = cldnn::format::bfyx,
                                    cldnn::format::type output_fmt = cldnn::format::bfyx,
                                    const std::vector<std::vector<ChannelName>>& dims_by_gws = {{ChannelName::X, ChannelName::Y},
                                                                                                {ChannelName::FEATURE},
                                                                                                {ChannelName::BATCH}});

}  // namespace ov::intel_gpu
