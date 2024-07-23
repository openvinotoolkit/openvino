// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ov {
namespace intel_gpu {
namespace op {
enum class TP_MODE {
    ALL_GATHERH = 0,
    ALL_GATHERV,
    ALL_REDUCE
};
}   // namespace op
}   // namespace intel_gpu
}   // namespace ov