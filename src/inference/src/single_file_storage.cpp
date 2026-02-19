// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_file_storage.hpp"

#include "openvino/util/file_util.hpp"
#include "storage_codecs.hpp"
#include "storage_traits.hpp"

namespace ov {
namespace {

void write_version(std::ostream& stream, const TLVStorage::Version& version) {
    stream.write(reinterpret_cast<const char*>(&version.major), sizeof(version.major));
    stream.write(reinterpret_cast<const char*>(&version.minor), sizeof(version.minor));
    stream.write(reinterpret_cast<const char*>(&version.patch), sizeof(version.patch));
}

void read_version(std::istream& stream, TLVStorage::Version& version) {
    stream.read(reinterpret_cast<char*>(&version.major), sizeof(version.major));
    stream.read(reinterpret_cast<char*>(&version.minor), sizeof(version.minor));
    stream.read(reinterpret_cast<char*>(&version.patch), sizeof(version.patch));
}

void validate_version(const TLVStorage::Version& version) {
    // todo Implement version compatibility check
}

void write_tlv_string(std::ostream& stream, const std::string& str) {
    TLVFormat::write_entry(stream,
                           static_cast<TLVFormat::tag_type>(TLVStorage::Tag::String),
                           static_cast<TLVFormat::length_type>(str.size()),
                           reinterpret_cast<const uint8_t*>(str.data()));
}

bool read_tlv_string(std::istream& stream, std::string& str) {
    TLVFormat::tag_type tag{};
    TLVFormat::length_type size{};
    const auto read = TLVFormat::read_entry(stream, tag, size, str);
    OPENVINO_ASSERT(TLVStorage::Tag{tag} == TLVStorage::Tag::String);
    return read;
}
}  // namespace

SingleFileStorage::SingleFileStorage(const std::filesystem::path& path) : m_file_path{path}, m_context_end{0} {
    util::create_directory_recursive(m_file_path.parent_path());
    if (!util::file_exists(m_file_path)) {
        std::ofstream stream(m_file_path, std::ios_base::binary);
        write_version(stream, m_version);
    } else {
        std::ifstream stream(m_file_path, std::ios_base::binary);
        TLVStorage::Version file_version{};
        read_version(stream, file_version);
        validate_version(file_version);

        scan_blob_map(stream);
        update_shared_ctx_from_file();
    }
}

void SingleFileStorage::scan_blob_map(std::ifstream& stream) {
    const TLVFormat::value_reader_callable blob_scanner = [this](std::istream& stream, TLVFormat::length_type size) {
        if (size == 0) {
            return;
        }
        TLVStorage::blob_id_type id;
        TLVStorage::pad_size_type padding_size{};
        stream.read(reinterpret_cast<char*>(&id), sizeof(id));
        stream.read(reinterpret_cast<char*>(&padding_size), sizeof(padding_size));
        stream.seekg(padding_size, std::ios::cur);
        if (!stream.good()) {
            return;
        }
        const auto blob_data_pos = stream.tellg();
        const auto blob_data_size = size - sizeof(id) - sizeof(padding_size) - padding_size;
        m_blob_map[id].offset = blob_data_pos;
        m_blob_map[id].size = blob_data_size;
        stream.seekg(blob_data_size, std::ios::cur);
    };
    const TLVFormat::value_reader_callable blob_map_scanner = [this](std::istream& stream,
                                                                     TLVFormat::length_type size) {
        if (size == 0) {
            return;
        }
        TLVStorage::blob_id_type id;
        stream.read(reinterpret_cast<char*>(&id), sizeof(id));
        if (!stream.good()) {
            return;
        }
        if (std::string model_name; read_tlv_string(stream, model_name)) {
            m_blob_map[id].model_name = model_name;
        }
    };

    const TLVFormat::value_scanners scanners = {
        {static_cast<TLVFormat::tag_type>(TLVStorage::Tag::Blob), blob_scanner},
        {static_cast<TLVFormat::tag_type>(TLVStorage::Tag::BlobMap), blob_map_scanner},
    };
    TLVFormat::scan_entries(stream, scanners, true);
}

TLVStorage::blob_id_type SingleFileStorage::convert_blob_id(const std::string& blob_id) {
    // todo stoull used unconditionally - what if blob_id_type isn't uint64_t?
    return static_cast<TLVStorage::blob_id_type>(std::stoull(blob_id.c_str()));
}

bool SingleFileStorage::has_blob_id(TLVStorage::blob_id_type blob_id) const {
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

void SingleFileStorage::write_blob_entry(TLVStorage::blob_id_type blob_id,
                                         StreamWriter& writer,
                                         std::ofstream& stream) {
    OPENVINO_ASSERT(!has_blob_id(blob_id), "Blob with id ", blob_id, " already exists in cache.");

    std::streampos blob_pos;
    std::streamoff blob_size;
    const TLVFormat::value_writer_callable blob_writer = [&](std::ostream& s) {
        s.write(reinterpret_cast<const char*>(&blob_id), sizeof(blob_id));

        const auto padding_pos =
            static_cast<uint64_t>(s.tellp()) + static_cast<uint64_t>(sizeof(TLVStorage::pad_size_type));
        auto aligned_pos = padding_pos + m_alignment - 1;
        aligned_pos -= aligned_pos % m_alignment;

        TLVStorage::pad_size_type pad_size = aligned_pos - padding_pos;
        s.write(reinterpret_cast<const char*>(&pad_size), sizeof(pad_size));
        if (pad_size > 0) {
            std::vector<char> padding(pad_size, 0);
            s.write(padding.data(), padding.size());
        }

        blob_pos = s.tellp();
        writer(s);
        blob_size = s.tellp() - blob_pos;
    };
    TLVFormat::write_entry(stream, static_cast<TLVFormat::tag_type>(TLVStorage::Tag::Blob), blob_writer);

    std::string model_name{"dev/invalid name"};  // todo Where to get it from?

    const TLVFormat::value_writer_callable blob_map_writer = [&](std::ostream& s) {
        s.write(reinterpret_cast<const char*>(&blob_id), sizeof(blob_id));
        write_tlv_string(s, model_name);
    };
    TLVFormat::write_entry(stream, static_cast<TLVFormat::tag_type>(TLVStorage::Tag::BlobMap), blob_map_writer);

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
        if (enable_mmap) {
            // todo Extend memory mapping helpers to suport partial file mapping
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
