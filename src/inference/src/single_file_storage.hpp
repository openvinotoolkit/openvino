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
    std::filesystem::path m_cache_file_path;
    std::unordered_map<std::string, std::tuple<size_t, size_t>> m_cache_index;  // blob_id -> (offset, size)
    SharedContext m_shared_context;
    SharedContext m_context_diff;
    std::streampos m_context_end;
};
}  // namespace ov
