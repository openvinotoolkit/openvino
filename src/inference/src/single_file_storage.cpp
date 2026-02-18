// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_file_storage.hpp"

#include "openvino/util/file_util.hpp"
#include "storage_codecs.hpp"
#include "storage_traits.hpp"

namespace ov {
namespace {
void validate_version(const TLVStorage::Version& version) {
    // todo Implement version compatibility check
}
}  // namespace

SingleFileStorage::SingleFileStorage(const std::filesystem::path& path) : m_file_path{path}, m_context_end{0} {
    util::create_directory_recursive(m_file_path.parent_path());
    if (!util::file_exists(m_file_path)) {
        std::ofstream stream(m_file_path, std::ios_base::binary);
        SingleFileStorageHeaderCodec header{m_version};
        stream << header;
    } else {
        std::ifstream stream(m_file_path, std::ios_base::binary);
        SingleFileStorageHeaderCodec header{};
        stream >> header;
        validate_version(header.version);

        fill_blob_map(stream);
        update_shared_ctx_from_file();
    }
}

void SingleFileStorage::fill_blob_map(std::ifstream& stream) {
    m_blob_map.clear();

    const auto beginning_pos = stream.tellg();
    stream.seekg(0, std::ios::end);
    const auto stream_end = stream.tellg();
    stream.seekg(beginning_pos);

    while (stream.good() && stream.tellg() < stream_end) {
        TLVStorage::Tag tag{};
        TLVStorage::length_type size{};
        stream.read(reinterpret_cast<char*>(&tag), sizeof(tag));
        if (!stream.good()) {
            break;
        }
        stream.read(reinterpret_cast<char*>(&size), sizeof(size));
        if (!stream.good() || size == 0) {
            break;
        }
        if (tag == TLVStorage::Tag::Blob) {
            TLVStorage::blob_id_type id;
            stream.read(reinterpret_cast<char*>(&id), sizeof(id));
            if (!stream.good()) {
                break;
            }
            TLVStorage::pad_size_type padding_size{};
            stream.read(reinterpret_cast<char*>(&padding_size), sizeof(padding_size));
            if (!stream.good()) {
                break;
            }
            stream.seekg(padding_size, std::ios::cur);

            const auto blob_data_pos = stream.tellg();
            const auto blob_data_size = size - sizeof(id) - sizeof(padding_size) - padding_size;
            m_blob_map[id].offset = blob_data_pos;
            m_blob_map[id].size = blob_data_size;
            stream.seekg(blob_data_size, std::ios::cur);
        } else if (tag == TLVStorage::Tag::BlobMap) {
            TLVStorage::blob_id_type id;
            stream.read(reinterpret_cast<char*>(&id), sizeof(id));
            if (!stream.good()) {
                break;
            }

            if (std::string model_name; read_tlv_string(stream, model_name)) {
                m_blob_map[id].model_name = model_name;
                // std::cout << "Read blob map entry: id=" << id << ", model_name=" << model_name
                //           << ", offset=" << m_blob_map[id].offset << ", size=" << m_blob_map[id].size << std::endl;
            } else {
                break;
            }
        } else {
            stream.seekg(size, std::ios::cur);
        }
    };

    stream.seekg(beginning_pos);
}

uint64_t SingleFileStorage::convert_blob_id(const std::string& blob_id) {
    return static_cast<uint64_t>(std::stoull(blob_id.c_str()));
}

bool SingleFileStorage::has_blob_id(uint64_t blob_id) const {
    return m_blob_map.find(blob_id) != m_blob_map.end();
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
    if (std::ifstream blob_file(m_file_path, std::ios_base::binary | std::ios_base::ate);
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

void SingleFileStorage::write_blob_entry(uint64_t blob_id, StreamWriter& writer, std::ofstream& stream) {
    OPENVINO_ASSERT(!has_blob_id(blob_id), "Blob with id ", blob_id, " already exists in cache.");
    TLVStorage::length_type entry_size = 0;
    TLVStorage::Tag tag = TLVStorage::Tag::Blob;
    stream.write(reinterpret_cast<const char*>(&tag), sizeof(tag));
    const auto blob_size_pos = stream.tellp();
    stream.write(reinterpret_cast<const char*>(&entry_size), sizeof(entry_size));

    auto blob_id_pos = stream.tellp();
    stream.write(reinterpret_cast<const char*>(&blob_id), sizeof(blob_id));

    // todo Apply actual padding
    const auto pad_size_pos =
        static_cast<uint64_t>(stream.tellp()) + static_cast<uint64_t>(sizeof(TLVStorage::pad_size_type));
    auto aligned_pos = pad_size_pos + m_alignment - 1;
    aligned_pos -= aligned_pos % m_alignment;

    TLVStorage::pad_size_type pad_size = aligned_pos - pad_size_pos;
    stream.write(reinterpret_cast<const char*>(&pad_size), sizeof(pad_size));
    if (pad_size > 0) {
        std::vector<char> padding(pad_size, 0);
        stream.write(padding.data(), padding.size());
    }

    const auto blob_pos = stream.tellp();
    std::cout << "Writing blob entry: id=" << blob_id << ", offset=" << blob_pos << ", pad_size=" << pad_size
              << std::endl;
    writer(stream);
    auto entry_end = stream.tellp();
    entry_size = entry_end - blob_id_pos;
    const auto blob_size = entry_end - blob_pos;
    stream.seekp(blob_size_pos);
    stream.write(reinterpret_cast<const char*>(&entry_size), sizeof(entry_size));
    stream.seekp(entry_end);

    tag = TLVStorage::Tag::BlobMap;
    entry_size = 0;
    stream.write(reinterpret_cast<const char*>(&tag), sizeof(tag));
    const auto blob_map_size_pos = stream.tellp();
    stream.write(reinterpret_cast<const char*>(&entry_size), sizeof(entry_size));

    blob_id_pos = stream.tellp();
    stream.write(reinterpret_cast<const char*>(&blob_id), sizeof(blob_id));

    std::string model_name{"dev/invalid name"};  // todo Where to get it from?
    write_tlv_string(stream, model_name);
    entry_end = stream.tellp();
    entry_size = entry_end - blob_id_pos;
    stream.seekp(blob_map_size_pos);
    stream.write(reinterpret_cast<const char*>(&entry_size), sizeof(entry_size));
    stream.seekp(entry_end);

    m_blob_map[blob_id] = {blob_id, blob_pos, blob_size, model_name};
}

void SingleFileStorage::write_cache_entry(const std::string& blob_id, StreamWriter writer) {
    ScopedLocale plocal_C(LC_ALL, "C");
    std::ofstream stream(m_file_path, std::ios_base::binary | std::ios_base::in | std::ios_base::ate);
    write_blob_entry(convert_blob_id(blob_id), writer, stream);
}

void SingleFileStorage::read_cache_entry(const std::string& blob_id, bool enable_mmap, StreamReader reader) {
    ScopedLocale plocal_C(LC_ALL, "C");

    const auto cid = convert_blob_id(blob_id);

    if (std::filesystem::exists(m_file_path) && has_blob_id(cid)) {
        const auto& [id, blob_pos, blob_size, model_name] = m_blob_map[cid];
        std::cout << "Reading blob entry: id=" << id << ", model_name=" << model_name << ", offset=" << blob_pos
                  << ", size=" << blob_size << std::endl;
        if (enable_mmap) {
            CompiledBlobVariant compiled_blob{
                std::in_place_index<0>,
                ov::read_tensor_data(m_file_path, element::u8, PartialShape{static_cast<int>(blob_size)}, blob_pos)};
            reader(compiled_blob);
        } else {
            std::ifstream stream(m_file_path, std::ios_base::binary);
            stream.seekg(blob_pos);
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
    std::ofstream stream(m_file_path, std::ios_base::binary | std::ios_base::in | std::ios_base::ate);
    write_ctx_diff(stream);
}
};  // namespace ov
