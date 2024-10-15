// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <optional>

#include "intel_npu/al/icompiled_model.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "npu.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace intel_npu {

struct MetadataVersion {
    uint32_t major;
    uint32_t minor;
} typedef MetadataVersion;

struct OpenvinoVersion {
    const std::string version;
} typedef OpenvinoVersion;

struct ModelLayout {
    int something;
    double somethingElse;
} typedef ModelLayout;

void check_blob_version(std::vector<uint8_t>& blob, std::istream& stream);

struct Metadata_v1 {
    MetadataVersion version;
    OpenvinoVersion ovVersion;

    std::vector<uint8_t> data();

    void read_metadata(std::vector<uint8_t>::iterator& metadataIterator);

    void write_metadata(std::ostream& stream);
} typedef Metadata_v1;

struct Metadata_v2 : Metadata_v1 {
    ModelLayout layout;

    std::vector<uint8_t> data();

    void read_metadata(std::vector<uint8_t>::iterator& metadataIterator);

    void write_metadata(std::ostream& stream);
} typedef Metadata_v2;

struct Metadata_v3 {
    MetadataVersion version;
    ModelLayout layout;
    OpenvinoVersion ovVersion;
    double extra;

    std::vector<uint8_t> data();

    void read_metadata(std::vector<uint8_t>::iterator metadataIterator);

    void write_metadata(std::ostream& stream);
} typedef Metadata_v3;

} // namespace intel_npu