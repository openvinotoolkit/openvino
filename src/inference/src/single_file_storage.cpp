// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_file_storage.hpp"

#include "openvino/util/file_util.hpp"

namespace ov {
SingleFileStorage::SingleFileStorage(const std::filesystem::path& path) : m_cache_file_path{path}, m_context_end{0} {
    util::create_directory_recursive(m_cache_file_path.parent_path());
    if (!util::file_exists(m_cache_file_path)) {
        std::ofstream stream(m_cache_file_path, std::ios_base::binary);
    } else {
        // Populate cache index or shared context if needed
    }
}

void SingleFileStorage::write_cache_entry(const std::string& id, StreamWriter writer) {}
void SingleFileStorage::read_cache_entry(const std::string& id, bool mmap_enabled, StreamReader reader) {}
void SingleFileStorage::remove_cache_entry(const std::string& id) {}

void SingleFileStorage::write_context_entry(const SharedContext& context) {}
SharedContext SingleFileStorage::get_shared_context() const {
    return {};
}

};  // namespace ov
