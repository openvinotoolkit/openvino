// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "icache_manager.hpp"
#include "openvino/core/weight_sharing_util.hpp"
#include "storage_traits.hpp"

// todo Remove this inclusion and use weight_sharing::Context instead of SharedContext in ISharedContextStore and
// SingleFileStorage.
#include "openvino/runtime/internal_properties.hpp"

namespace ov {
class SingleFileStorage final : public ICacheManager, public ISharedContextStore {
public:
    // todo Whose version should it be and how to handle incompatibilities?
    static constexpr TLVStorage::Version m_version = {0, 1, 0};

    explicit SingleFileStorage(const std::filesystem::path& path);

    void write_cache_entry(const std::string& blob_id, StreamWriter writer) override;
    void read_cache_entry(const std::string& blob_id, bool mmap_enabled, StreamReader reader) override;
    void remove_cache_entry(const std::string& blob_id) override;

    void write_context_entry(const weight_sharing::Context& context) override;
    weight_sharing::Context get_context() const override;

private:
    // todo Make it configurable and/or detect actual file system page size
    static constexpr uint64_t m_alignment = 4096;

    std::filesystem::path m_file_path;
    TLVStorage::blob_map_type m_blob_map;
    void scan_blob_map(std::ifstream& stream);
    void scan_context(std::ifstream& stream);

    static TLVStorage::blob_id_type convert_blob_id(const std::string& blob_id);
    void write_blob_entry(TLVStorage::blob_id_type blob_id, StreamWriter& writer, std::ofstream& stream);
    bool has_blob_id(TLVStorage::blob_id_type blob_id) const;

    weight_sharing::Context m_shared_context;
};
}  // namespace ov
