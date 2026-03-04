// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// MOECompressed has moved to common transformations.
// This header provides a backward-compatible alias for GPU code.
#include "ov_ops/moe_compressed.hpp"

namespace ov::intel_gpu::op {
using MOECompressed = ov::op::internal::MOECompressed;
}  // namespace ov::intel_gpu::op
