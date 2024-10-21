// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdint.h>

#include <memory>
#include <string>
#include <vector>

namespace intel_npu {

constexpr std::string_view DELIMITER = "OVNPU";

constexpr int CURRENT_METAVERSION_MAJOR = 1;
constexpr int CURRENT_METAVERSION_MINOR = 0;

struct MetadataVersion {
    uint32_t major;
    uint32_t minor;
};

struct OpenvinoVersion {
    std::string version;
    uint32_t size;

    OpenvinoVersion(const std::string& version);

    void read(std::istream& stream);
};

struct MetadataBase {
    virtual void read(std::istream& stream) = 0;
    virtual void write(std::ostream& stream) = 0;
    virtual bool isCompatible(const MetadataBase& other) const = 0;
    virtual ~MetadataBase() = default;
};

template <int Major, int Minor>
struct Metadata : public MetadataBase {};

template <>
struct Metadata<1, 0> : public MetadataBase {
    MetadataVersion version;
    OpenvinoVersion ovVersion;

    Metadata();

    std::stringstream data() const;

    void write(std::ostream& stream) override;

    void read(std::istream& stream) override;

    bool isCompatible(const MetadataBase& other) const override;
};

std::unique_ptr<MetadataBase> createMetadata(int major, int minor);

std::unique_ptr<MetadataBase> read_metadata_from(std::vector<uint8_t>& blob);

}  // namespace intel_npu
