// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <unordered_map>
#include <dnnl.hpp>

namespace ov {
namespace intel_cpu {

using dnnl_primitive_args = std::unordered_map<int, dnnl::memory>;

}  // namespace intel_cpu
}  // namespace ov
