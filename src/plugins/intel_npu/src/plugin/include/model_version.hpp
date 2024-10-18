// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npu.hpp"

namespace intel_npu {

constexpr std::string_view DELIMITER = "OVNPU";

constexpr int CURRENT_METAVERSION_MAJOR = 1;
constexpr int CURRENT_METAVERSION_MINOR = 0;

struct MetadataVersion {
    uint32_t major;
    uint32_t minor;
};

struct OpenvinoVersion {
    uint32_t size;
    std::string version;

    OpenvinoVersion(const std::string& version);

    void read(std::istream& stream);
};

template<int Major, int Minor>
struct Metadata { };

template<>
struct Metadata<1, 0> {
    MetadataVersion version;
    OpenvinoVersion ovVersion;

    Metadata();

    std::stringstream data() const;

    void write(std::ostream& stream);

    void read(std::istream& stream);
};

void check_blob_version(std::vector<uint8_t>& blob);

} // namespace intel_npu
