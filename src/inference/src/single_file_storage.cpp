// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/single_file_storage.hpp"

#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/mmap_object.hpp"
#include "openvino/util/variant_visitor.hpp"

namespace ov::runtime {
namespace {
void write_version(std::ostream& stream, const SingleFileStorage::FormatVersion& version) {
    stream.write(reinterpret_cast<const char*>(&version.major), sizeof(version.major));
    stream.write(reinterpret_cast<const char*>(&version.minor), sizeof(version.minor));
    stream.write(reinterpret_cast<const char*>(&version.patch), sizeof(version.patch));
}

void read_version(std::istream& stream, SingleFileStorage::FormatVersion& version) {
    stream.read(reinterpret_cast<char*>(&version.major), sizeof(version.major));
    stream.read(reinterpret_cast<char*>(&version.minor), sizeof(version.minor));
    stream.read(reinterpret_cast<char*>(&version.patch), sizeof(version.patch));
}

void validate_version(const SingleFileStorage::FormatVersion& version) {
    // todo Implement version compatibility check
}

void write_tlv_string(std::ostream& stream, const std::string& str) {
    TLVFormat::write_entry(stream,
                           static_cast<TLVFormat::TagType>(SingleFileStorage::Tag::String),
                           static_cast<TLVFormat::LeghtType>(str.size()),
                           reinterpret_cast<const uint8_t*>(str.data()));
}

bool read_tlv_string(std::istream& stream, std::string& str) {
    TLVFormat::TagType tag;
    TLVFormat::LeghtType size;
    const auto read = TLVFormat::read_entry(stream, tag, size, str);
    OPENVINO_ASSERT(SingleFileStorage::Tag{tag} == SingleFileStorage::Tag::String);
    return read;
}

void write_padding(std::ostream& stream, uint64_t alignment) {
    const uint64_t padding_pos = static_cast<uint64_t>(stream.tellp()) + sizeof(SingleFileStorage::PadSizeType);
    auto aligned_pos = padding_pos + alignment - 1;
    aligned_pos -= aligned_pos % alignment;

    SingleFileStorage::PadSizeType pad_size = aligned_pos - padding_pos;
    stream.write(reinterpret_cast<const char*>(&pad_size), sizeof(pad_size));
    if (pad_size > 0) {
        std::vector<char> padding(pad_size, 0);
        stream.write(padding.data(), padding.size());
    }
}
}  // namespace

bool SingleFileStorage::FormatVersion::operator==(const FormatVersion& other) const {
    return major == other.major && minor == other.minor && patch == other.patch;
}

SingleFileStorage::SingleFileStorage(const std::filesystem::path& path) : m_file_path{path} {
    util::create_directory_recursive(m_file_path.parent_path());
    if (!util::file_exists(m_file_path)) {
        std::ofstream stream(m_file_path, std::ios_base::binary);
        write_version(stream, m_version);
    } else {
        std::ifstream stream(m_file_path, std::ios_base::binary);
        FormatVersion file_version;
        read_version(stream, file_version);
        validate_version(file_version);

        scan_blob_map(stream);
        scan_context(stream);

        // todo Add error handling and validation (e.g. check that blob entries are present for all blob ids in blob
        //      map). If file is corrupted, it should be handled gracefully without throwing exceptions or crashing
        //      (e.g. by ignoring invalid entries and logging warnings).
    }
}

void SingleFileStorage::scan_blob_map(std::ifstream& stream) {
    const auto blob_reader = [this](std::istream& stream, TLVFormat::LeghtType size) {
        if (size == 0) {
            return;
        }
        BlobIdType id;
        PadSizeType padding_size;
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

    const auto blob_map_reader = [this](std::istream& stream, TLVFormat::LeghtType size) {
        if (size == 0) {
            return;
        }
        BlobIdType id;
        stream.read(reinterpret_cast<char*>(&id), sizeof(id));
        if (!stream.good()) {
            return;
        }
        if (std::string model_name; read_tlv_string(stream, model_name)) {
            m_blob_map[id].model_name = model_name;
        }
    };

    const TLVFormat::ValueScanner scanners = {
        {static_cast<TLVFormat::TagType>(Tag::Blob), blob_reader},
        {static_cast<TLVFormat::TagType>(Tag::BlobMap), blob_map_reader},
    };
    TLVFormat::scan_entries(stream, scanners, true);
}

SingleFileStorage::BlobIdType SingleFileStorage::convert_blob_id(const std::string& blob_id) {
    // todo stoull used unconditionally - what if BlobIdType isn't uint64_t?
    return static_cast<BlobIdType>(std::stoull(blob_id.c_str()));
}

bool SingleFileStorage::has_blob_id(BlobIdType blob_id) const {
    return m_blob_map.find(blob_id) != m_blob_map.end();
}

void SingleFileStorage::write_blob_entry(BlobIdType blob_id, StreamWriter& writer, std::ofstream& stream) {
    OPENVINO_ASSERT(!has_blob_id(blob_id), "Blob with id ", blob_id, " already exists in cache.");

    std::streampos blob_pos;
    std::streamoff blob_size;

    const auto blob_writer = [&](std::ostream& s) {
        s.write(reinterpret_cast<const char*>(&blob_id), sizeof(blob_id));
        write_padding(s, m_alignment);
        blob_pos = s.tellp();
        writer(s);
        blob_size = s.tellp() - blob_pos;
    };
    TLVFormat::write_entry(stream, static_cast<TLVFormat::TagType>(Tag::Blob), blob_writer);

    std::string model_name{"dev/invalid name"};  // todo Where to get it from?

    const auto blob_map_writer = [&](std::ostream& s) {
        s.write(reinterpret_cast<const char*>(&blob_id), sizeof(blob_id));
        write_tlv_string(s, model_name);
    };
    TLVFormat::write_entry(stream, static_cast<TLVFormat::TagType>(Tag::BlobMap), blob_map_writer);

    m_blob_map[blob_id] = {blob_pos, blob_size, model_name};
}

void SingleFileStorage::write_cache_entry(const std::string& blob_id, StreamWriter writer) {
    ScopedLocale plocal_C(LC_ALL, "C");
    // todo Check whether `std::ios_base::in` is needed
    std::ofstream stream(m_file_path, std::ios_base::binary | std::ios_base::in | std::ios_base::ate);
    write_blob_entry(convert_blob_id(blob_id), writer, stream);
}

void SingleFileStorage::read_cache_entry(const std::string& blob_id, bool enable_mmap, StreamReader reader) {
    ScopedLocale plocal_C(LC_ALL, "C");

    const auto cid = convert_blob_id(blob_id);

    if (std::filesystem::exists(m_file_path) && has_blob_id(cid)) {
        const auto& [blob_pos, blob_size, model_name] = m_blob_map[cid];
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

weight_sharing::Context SingleFileStorage::get_context() const {
    return m_shared_context;
}

void SingleFileStorage::scan_context(std::ifstream& stream) {
    const auto constant_meta_reader = [this](std::istream& stream, TLVFormat::LeghtType size) {
        if (size == 0) {
            return;
        }
        uint64_t source_id;
        stream.read(reinterpret_cast<char*>(&source_id), sizeof(source_id));
        auto left_size = size - sizeof(source_id);

        while (stream.good() && left_size > 0) {
            uint64_t const_id, const_offset, const_size;
            uint8_t const_type;
            constexpr auto const_meta_size =
                sizeof(const_id) + sizeof(const_offset) + sizeof(const_size) + sizeof(const_type);
            stream.read(reinterpret_cast<char*>(&const_id), sizeof(const_id));
            stream.read(reinterpret_cast<char*>(&const_offset), sizeof(const_offset));
            stream.read(reinterpret_cast<char*>(&const_size), sizeof(const_size));
            stream.read(reinterpret_cast<char*>(&const_type), sizeof(const_type));
            if (!stream.good()) {
                break;
            }
            left_size -= const_meta_size;

            m_shared_context.m_constants_meta_data[source_id][const_id] = {const_offset,
                                                                           const_size,
                                                                           element::Type_t{const_type}};
        }
    };

    const auto constant_source_reader = [this](std::istream& stream, TLVFormat::LeghtType size) {
        if (size == 0) {
            return;
        }
        DataIdType device_id, source_id;
        PadSizeType padding_size;
        stream.read(reinterpret_cast<char*>(&device_id), sizeof(device_id));
        stream.read(reinterpret_cast<char*>(&source_id), sizeof(source_id));
        stream.read(reinterpret_cast<char*>(&padding_size), sizeof(padding_size));
        stream.seekg(padding_size, std::ios::cur);
        if (!stream.good()) {
            return;
        }
        // const auto weight_pos = stream.tellg();
        const auto weight_size = size - sizeof(device_id) - sizeof(source_id) - sizeof(padding_size) - padding_size;
        m_shared_context.m_weight_sources[source_id] = {};
        stream.seekg(weight_size, std::ios::cur);
    };

    const TLVFormat::ValueScanner scanners = {
        {static_cast<TLVFormat::TagType>(Tag::ConstantMeta), constant_meta_reader},
        {static_cast<TLVFormat::TagType>(Tag::WeightSource), constant_source_reader},
    };
    TLVFormat::scan_entries(stream, scanners, true);
}

void SingleFileStorage::write_context(const weight_sharing::Context& context) {
    ScopedLocale plocal_C(LC_ALL, "C");
    std::ofstream stream(m_file_path, std::ios_base::binary | std::ios_base::in | std::ios_base::ate);

    // todo Add delta writing - not the whole.

    for (const auto& [key, const_meta] : context.m_constants_meta_data) {
        const auto const_meta_writer = [&](std::ostream& s) {
            const auto source_id = static_cast<DataIdType>(key);
            s.write(reinterpret_cast<const char*>(&source_id), sizeof(source_id));
            for (const auto& [id, props] : const_meta) {
                const auto const_id = static_cast<DataIdType>(id);
                const auto const_offset = static_cast<uint64_t>(props.m_offset);
                const auto const_size = static_cast<uint64_t>(props.m_size);
                const auto const_type = static_cast<uint8_t>(element::Type_t{props.m_type});
                s.write(reinterpret_cast<const char*>(&const_id), sizeof(const_id));
                s.write(reinterpret_cast<const char*>(&const_offset), sizeof(const_offset));
                s.write(reinterpret_cast<const char*>(&const_size), sizeof(const_size));
                s.write(reinterpret_cast<const char*>(&const_type), sizeof(const_type));

                m_shared_context.m_constants_meta_data[key][id] = props;
            }
        };
        TLVFormat::write_entry(stream, static_cast<TLVFormat::TagType>(Tag::ConstantMeta), const_meta_writer);
    }

    for (const auto& [key, weight_buffer] : context.m_weight_sources) {
        const auto weight_source_writer = [&](std::ostream& s) {
            const auto device_id = static_cast<DataIdType>(0);  // todo Where to get it from?
            const auto source_id = static_cast<DataIdType>(key);
            s.write(reinterpret_cast<const char*>(&device_id), sizeof(device_id));
            s.write(reinterpret_cast<const char*>(&source_id), sizeof(source_id));
            write_padding(s, m_alignment);
            // todo Write actual weight data if needed

            const auto buffer_writer =
                ov::util::VariantVisitor{[&](const std::weak_ptr<ov::AlignedBuffer>& weak_buf) {
                                             if (auto buf = weak_buf.lock()) {
                                                 s.write(reinterpret_cast<const char*>(buf->get_ptr()), buf->size());
                                             }
                                         },
                                         [&](const std::weak_ptr<ov::MappedMemory>& weak_mem) {
                                             if (auto mem = weak_mem.lock()) {
                                                 s.write(reinterpret_cast<const char*>(mem->data()), mem->size());
                                             }
                                         }};
            std::visit(buffer_writer, weight_buffer);

            m_shared_context.m_weight_sources[key] = weight_buffer;
        };
        TLVFormat::write_entry(stream, static_cast<TLVFormat::TagType>(Tag::WeightSource), weight_source_writer);
    }
}
};  // namespace ov::runtime
