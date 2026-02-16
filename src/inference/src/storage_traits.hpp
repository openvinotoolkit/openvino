// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

namespace ov {

struct TLVStorage {
    using tag_type = uint32_t;
    using length_type = uint64_t;
    using blob_id_type = uint64_t;

    enum class Tag : tag_type {
        SharedContext = 0x01,
        String = 0x02,
        Blob = 0x03,
        BlobMap = 0x04,
    };
};
}  // namespace ov
