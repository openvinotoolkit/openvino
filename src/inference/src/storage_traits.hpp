// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_map>

// todo Rename file to tlv_helpers and move common codecs into here

namespace ov {

// todo Separate CacheManager and TLV format traits
struct TLVStorage {
    using tag_type = uint32_t;
    using length_type = uint64_t;
    using blob_id_type = uint64_t;
    using pad_size_type = uint64_t;

    struct blob_info {
        blob_id_type id;  // todo It's likely redundant information - remove this field if used only as key in blob_map
        std::streampos offset;
        std::streampos size;
        std::string model_name;
    };
    using blob_map_type = std::unordered_map<TLVStorage::blob_id_type, blob_info>;

    enum class Tag : tag_type {
        SharedContext = 0x01,
        String = 0x02,
        Blob = 0x03,
        BlobMap = 0x04,
    };

    struct Version {
        uint16_t major{};
        uint16_t minor{};
        uint16_t patch{};

        bool operator==(const Version& other) const {
            return major == other.major && minor == other.minor && patch == other.patch;
        }
    };
};
}  // namespace ov
