// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "icache_manager.hpp"
#include "storage_traits.hpp"

namespace ov {
class SingleFileStorage final : public ICacheManager, public ISharedContextStore {
public:
    static constexpr uint64_t major_version = 0;
    static constexpr uint64_t minor_version = 1;

    explicit SingleFileStorage(const std::filesystem::path& path);

    void write_cache_entry(const std::string& blob_id, StreamWriter writer) override;
    void read_cache_entry(const std::string& blob_id, bool mmap_enabled, StreamReader reader) override;
    void remove_cache_entry(const std::string& blob_id) override;

    void write_context_entry(const SharedContext& context) override;
    SharedContext get_shared_context() const override;

private:
    TLVStorage::blob_map_type m_blob_map;
    static uint64_t convert_blob_id(const std::string& blob_id);
    void write_blob_entry(uint64_t blob_id, StreamWriter& writer, std::ofstream& stream);
    bool has_blob_id(uint64_t blob_id) const;

    // todo Below parts are for refactor and might be removed. Don't leave it - reuse or remove.
private:
    static constexpr size_t blob_id_size = 24;
    void update_shared_ctx(const SharedContext& new_ctx);
    void update_shared_ctx_from_file();
    // rename to append or sth??
    void write_ctx_diff(std::ostream& stream);

    std::filesystem::path m_cache_file_path;  // rename to m_storage_file_path or sth
    SharedContext m_shared_context;
    SharedContext m_context_diff;
    std::streampos m_context_end;
};
}  // namespace ov
