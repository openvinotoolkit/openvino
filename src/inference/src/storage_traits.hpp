// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

namespace ov::storage {

using tag_type = uint32_t;
using length_type = uint64_t;

static constexpr tag_type shared_context_tag = 0x0100;
static constexpr tag_type content_index_tag = 0x0200;
static constexpr tag_type blob_tag = 0x0300;
}  // namespace ov::storage
