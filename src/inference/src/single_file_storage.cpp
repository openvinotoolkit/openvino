// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_file_storage.hpp"

#include "openvino/util/file_util.hpp"
#include "storage_codecs.hpp"
#include "storage_traits.hpp"

namespace ov {

SingleFileStorage::SingleFileStorage(const std::filesystem::path& path) : m_cache_file_path{path}, m_context_end{0} {
    util::create_directory_recursive(m_cache_file_path.parent_path());
    if (!util::file_exists(m_cache_file_path)) {
        std::ofstream stream(m_cache_file_path, std::ios_base::binary);
    } else {
        populate_cache_index();
        update_shared_ctx_from_file();
    }
}

bool SingleFileStorage::has_blob_id(const std::string& blob_id) const {
    return m_cache_index.find(blob_id) != m_cache_index.end();
}

void SingleFileStorage::populate_cache_index() {
    const auto f_size = ov::util::file_size(m_cache_file_path);
    if (std::ifstream blob_file(m_cache_file_path, std::ios_base::binary); blob_file.is_open()) {
        // Read shared context from the cache file
        TLVStorage::Tag tag{};
        TLVStorage::length_type size{};
        while (blob_file.good() && blob_file.tellg() < f_size) {
            blob_file.read(reinterpret_cast<char*>(&tag), sizeof(tag));
            blob_file.read(reinterpret_cast<char*>(&size), sizeof(size));
            if (blob_file.eof()) {
                break;
            }
            if (tag == TLVStorage::Tag::Blob) {
                std::string blob_id(blob_id_size, '\0');
                blob_file.read(blob_id.data(), blob_id.size());
                blob_id.erase(std::find(blob_id.begin(), blob_id.end(), '\0'), blob_id.end());
                size -= blob_id_size;
                m_cache_index.emplace_hint(m_cache_index.end(),
                                           blob_id,
                                           std::make_tuple(static_cast<size_t>(blob_file.tellg()), size));
            }
            blob_file.seekg(size, std::ios::cur);
        }
    }
    // std::cout << "Populating cache index end" << std::endl;
    // for (const auto& [id, info] : m_cache_index) {
    //     std::cout << "Cache index - id: " << id << " offset: " << std::get<0>(info) << " size: " << std::get<1>(info)
    //               << "\n";
    // }
}

void SingleFileStorage::update_shared_ctx(const SharedContext& new_ctx) {
    for (const auto& [src_id, consts] : new_ctx) {
        for (const auto& [const_id, props] : consts) {
            if (auto id_it = m_shared_context.find(src_id); id_it != m_shared_context.end()) {
                id_it->second[const_id] = props;
            } else {
                m_shared_context[src_id] = {{const_id, props}};
            }
        }
    }
}

void SingleFileStorage::update_shared_ctx_from_file() {
    if (std::ifstream blob_file(m_cache_file_path, std::ios_base::binary | std::ios_base::ate);
        blob_file.is_open() && (m_context_end < blob_file.tellg())) {
        blob_file.seekg(m_context_end);
        // Read shared context from the cache file
        SharedContext shared_ctx;
        SharedContextStreamCodec ctx_cache{&shared_ctx};
        blob_file >> ctx_cache;
        update_shared_ctx(shared_ctx);
        m_context_end = blob_file.tellg();
    }
}

void SingleFileStorage::write_blob_entry(const std::string& id, StreamWriter& writer, std::ofstream& stream) {
    OPENVINO_ASSERT(!has_blob_id(id), "Blob with id ", id, " already exists in cache.");
    TLVStorage::length_type blob_size = 0;
    TLVStorage::Tag blob_tag = TLVStorage::Tag::Blob;
    stream.write(reinterpret_cast<const char*>(&blob_tag), sizeof(blob_tag));
    const auto blob_size_pos = stream.tellp();
    stream.write(reinterpret_cast<const char*>(&blob_size), sizeof(blob_size));

    const auto blob_id_pos = stream.tellp();
    auto ids = id;
    ids.resize(blob_id_size, '\0');
    stream.write(ids.data(), blob_id_size);

    const auto blob_pos = stream.tellp();
    writer(stream);
    const auto end = stream.tellp();
    blob_size = end - blob_id_pos;
    stream.seekp(blob_size_pos);
    stream.write(reinterpret_cast<const char*>(&blob_size), sizeof(blob_size));
    stream.seekp(end);
    m_cache_index[id] = std::make_tuple(blob_pos, blob_size - ids.size());  // Update with actual offset and size
}

void SingleFileStorage::write_cache_entry(const std::string& id, StreamWriter writer) {
    ScopedLocale plocal_C(LC_ALL, "C");
    std::ofstream stream(m_cache_file_path, std::ios_base::binary | std::ios_base::in | std::ios_base::ate);
    write_blob_entry(id, writer, stream);
}

void SingleFileStorage::read_cache_entry(const std::string& id, bool enable_mmap, StreamReader reader) {
    ScopedLocale plocal_C(LC_ALL, "C");

    if (std::filesystem::exists(m_cache_file_path) && has_blob_id(id)) {
        const auto& [offset, size] = m_cache_index[id];
        if (enable_mmap) {
            CompiledBlobVariant compiled_blob{
                std::in_place_index<0>,
                ov::read_tensor_data(m_cache_file_path, element::u8, PartialShape{static_cast<int>(size)}, offset)};
            reader(compiled_blob);
        } else {
            std::ifstream stream(m_cache_file_path, std::ios_base::binary);
            stream.seekg(offset);
            CompiledBlobVariant compiled_blob{std::in_place_index<1>, std::ref(stream)};
            reader(compiled_blob);
        }
    }
}

void SingleFileStorage::remove_cache_entry(const std::string& id) {}

void SingleFileStorage::write_ctx_diff(std::ostream& stream) {
    if (!m_context_diff.empty()) {
        SharedContextStreamCodec ctx_cache{&m_context_diff};
        stream << ctx_cache;
        m_context_end = stream.tellp();
        update_shared_ctx(m_context_diff);
        m_context_diff.clear();
    }
}

SharedContext SingleFileStorage::get_shared_context() const {
    return m_shared_context;
}

void SingleFileStorage::write_context_entry(const SharedContext& ctx) {
    for (auto&& [src_id, consts] : ctx) {
        for (auto&& [const_id, props] : consts) {
            if (auto id_it = m_context_diff.find(src_id); id_it != m_context_diff.end()) {
                id_it->second[const_id] = props;
            } else {
                m_context_diff[src_id] = {{const_id, props}};
            }
        }
    }
    update_shared_ctx(ctx);
    std::ofstream stream(m_cache_file_path, std::ios_base::binary | std::ios_base::in | std::ios_base::ate);
    write_ctx_diff(stream);
}
};  // namespace ov
