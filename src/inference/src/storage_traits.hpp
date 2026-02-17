// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

namespace ov {
// todo Move it to src/inference/src/single_file_storage.hpp?
struct TLVStorage {
    using tag_type = uint32_t;
    using length_type = uint64_t;
    using blob_id_type = uint64_t;
    using padding_size_type = uint64_t;

    struct blob_info {
        blob_id_type id;
        length_type size;
        length_type offset;
        std::string model_name;
    };
    using blob_map_type = std::unordered_map<TLVStorage::blob_id_type, blob_info>;

    enum class Tag : tag_type {
        SharedContext = 0x01,
        String = 0x02,
        Blob = 0x03,
        BlobMap = 0x04,
    };
};
}  // namespace ov
