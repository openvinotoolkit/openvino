// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

namespace ov {
namespace npuw {

struct KVAxesPosition {
    uint32_t batch;
    uint32_t seq_len;
};

}  // namespace npuw
}  // namespace ov
