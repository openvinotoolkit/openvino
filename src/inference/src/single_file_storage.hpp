// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "icache_manager.hpp"

namespace ov {
class SingleFileStorage final : public ICacheManager, public ISharedContextStore {
public:
    explicit SingleFileStorage(const std::filesystem::path& path);

    void write_cache_entry(const std::string& id, StreamWriter writer) override;
    void read_cache_entry(const std::string& id, bool mmap_enabled, StreamReader reader) override;
    void remove_cache_entry(const std::string& id) override;

    void write_context_entry(const SharedContext& context) override;
    SharedContext get_shared_context() const override;

private:
    bool has_blob_id(const std::string& blob_id) const;
    void populate_cache_index();
    void update_shared_ctx(const SharedContext& new_ctx);
    void update_shared_ctx_from_file();
    void write_ctx_diff(std::ostream& stream);
    void write_blob_entry(const std::string& id, StreamWriter& writer, std::ofstream& stream);

    std::filesystem::path m_cache_file_path;
    std::unordered_map<std::string, std::tuple<size_t, size_t>> m_cache_index;  // blob_id -> (offset, size)
    SharedContext m_shared_context;
    SharedContext m_context_diff;
    std::streampos m_context_end;
};
}  // namespace ov
