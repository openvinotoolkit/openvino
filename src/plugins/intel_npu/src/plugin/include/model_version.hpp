// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <optional>

#include "intel_npu/al/icompiled_model.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "npu.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace intel_npu {

constexpr int CURRENT_METAVERSION_MAJOR = 1;
constexpr int CURRENT_METAVERSION_MINOR = 0;

constexpr std::string_view VERSION_HEADER = "OVNPU";

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

    std::stringstream data();

    void write(std::ostream& stream);

    void read(std::istream& stream);
};

void check_blob_version(std::vector<uint8_t>& blob, std::istream& stream);

} // namespace intel_npu