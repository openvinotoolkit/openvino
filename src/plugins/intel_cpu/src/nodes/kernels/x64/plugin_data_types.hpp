// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common/c_types_map.hpp>

// After reverting the oneDNN fork commits that added u2 support, the
// dnnl::impl::data_type::u2 constant (value 18) will no longer exist in
// oneDNN headers. This header provides the constant for plugin-internal
// weight decompression kernels. It is never passed to oneDNN APIs.

namespace ov::intel_cpu::plugin_data_type {
constexpr dnnl::impl::data_type_t u2 = static_cast<dnnl::impl::data_type_t>(18);
}  // namespace ov::intel_cpu::plugin_data_type
