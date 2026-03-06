// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/single_file_storage.hpp"

#ifdef _WIN32
#    include <windows.h>
#else
#    include <unistd.h>
#endif

#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/mmap_object.hpp"
#include "openvino/util/variant_visitor.hpp"

namespace ov::runtime {

namespace {
void write_version(std::ostream& stream, const util::Version& version) {
    const uint16_t major = static_cast<uint16_t>(version.major);
    const uint16_t minor = static_cast<uint16_t>(version.minor);
    const uint16_t patch = static_cast<uint16_t>(version.patch);
    stream.write(reinterpret_cast<const char*>(&major), sizeof(major));
    stream.write(reinterpret_cast<const char*>(&minor), sizeof(minor));
    stream.write(reinterpret_cast<const char*>(&patch), sizeof(patch));
}

void read_version(std::istream& stream, util::Version& version) {
    uint16_t major = 0, minor = 0, patch = 0;
    stream.read(reinterpret_cast<char*>(&major), sizeof(major));
    stream.read(reinterpret_cast<char*>(&minor), sizeof(minor));
    stream.read(reinterpret_cast<char*>(&patch), sizeof(patch));
    if (stream.good()) {
        version.major = major;
        version.minor = minor;
        version.patch = patch;
    }
}

void validate_version(const util::Version& version) {
    // todo Implement version compatibility check
}

void write_tlv_string(std::ostream& stream, const std::string& str) {
    write_tlv_record(stream, static_cast<TLVTraits::TagType>(SingleFileStorage::Tag::String), str.size(), str.data());
}

bool read_tlv_string(std::istream& stream, std::string& str) {
    TLVTraits::TagType tag;
    TLVTraits::LengthType size;
    std::vector<char> buffer;
    const auto read = read_tlv_record(stream, tag, size, buffer);
    OPENVINO_ASSERT(SingleFileStorage::Tag{tag} == SingleFileStorage::Tag::String);
    if (read) {
        str = std::string{buffer.begin(), buffer.end()};
    }
    return read;
}

void write_padding(std::ostream& stream, size_t alignment) {
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

SingleFileStorage::SingleFileStorage(const std::filesystem::path& path) : m_file_path{path} {
    util::create_directory_recursive(m_file_path.parent_path());
    if (!util::file_exists(m_file_path)) {
        std::ofstream stream(m_file_path, std::ios::binary);
        write_version(stream, m_version);
    } else {
        std::ifstream stream(m_file_path, std::ios::binary);
        util::Version file_version{"0.0.0-0-"};
        read_version(stream, file_version);
        validate_version(file_version);

        if (!build_content_index(stream)) {
            m_blob_index.clear();
            m_shared_context.m_cache_sources.clear();
            m_shared_context.m_weight_registry.clear();
        }
    }
}

bool SingleFileStorage::build_content_index(std::ifstream& stream) {
    const auto blob_reader = [this](std::istream& s, TLVTraits::LengthType size) {
        if (size == 0) {
            return true;
        }
        BlobIdType id;
        PadSizeType padding_size;
        s.read(reinterpret_cast<char*>(&id), sizeof(id));
        s.read(reinterpret_cast<char*>(&padding_size), sizeof(padding_size));
        s.seekg(padding_size, std::ios::cur);
        if (!s.good()) {
            return false;
        }
        const auto blob_data_pos = s.tellg();
        const auto blob_data_size = size - sizeof(id) - sizeof(padding_size) - padding_size;
        m_blob_index[id].offset = blob_data_pos;
        m_blob_index[id].size = blob_data_size;
        s.seekg(blob_data_size, std::ios::cur);
        return s.good();
    };
    const auto blob_map_reader = [this](std::istream& s, TLVTraits::LengthType size) {
        if (size == 0) {
            return true;
        }
        BlobIdType id;
        s.read(reinterpret_cast<char*>(&id), sizeof(id));
        if (!s.good()) {
            return false;
        }
        if (std::string model_name; read_tlv_string(s, model_name)) {
            m_blob_index[id].model_name = model_name;
        }
        return s.good();
    };
    const auto constant_meta_reader = [this](std::istream& s, TLVTraits::LengthType size) {
        if (size == 0) {
            return true;
        }
        uint64_t source_id;
        s.read(reinterpret_cast<char*>(&source_id), sizeof(source_id));
        auto left_size = size - sizeof(source_id);

        while (s.good() && left_size > 0) {
            uint64_t const_id, const_offset, const_size;
            uint8_t const_type;
            constexpr auto const_meta_size =
                sizeof(const_id) + sizeof(const_offset) + sizeof(const_size) + sizeof(const_type);
            s.read(reinterpret_cast<char*>(&const_id), sizeof(const_id));
            s.read(reinterpret_cast<char*>(&const_offset), sizeof(const_offset));
            s.read(reinterpret_cast<char*>(&const_size), sizeof(const_size));
            s.read(reinterpret_cast<char*>(&const_type), sizeof(const_type));
            if (!s.good()) {
                return false;
            }
            left_size -= const_meta_size;

            m_shared_context.m_weight_registry[source_id][const_id] = {static_cast<size_t>(const_offset),
                                                                       static_cast<size_t>(const_size),
                                                                       element::Type_t{const_type}};
        }
        return s.good();
    };
    const auto constant_source_reader = [this](std::istream& s, TLVTraits::LengthType size) {
        if (size == 0) {
            return true;
        }
        DataIdType device_id, source_id;
        PadSizeType padding_size;
        s.read(reinterpret_cast<char*>(&device_id), sizeof(device_id));
        s.read(reinterpret_cast<char*>(&source_id), sizeof(source_id));
        s.read(reinterpret_cast<char*>(&padding_size), sizeof(padding_size));
        s.seekg(padding_size, std::ios::cur);
        if (!s.good()) {
            return false;
        }
        const auto weight_size = size - sizeof(device_id) - sizeof(source_id) - sizeof(padding_size) - padding_size;
        m_shared_context.m_cache_sources[source_id] = {};
        s.seekg(weight_size, std::ios::cur);
        return s.good();
    };
    const TLVValueScanner scanners = {
        {static_cast<TLVTraits::TagType>(Tag::Blob), blob_reader},
        {static_cast<TLVTraits::TagType>(Tag::BlobMap), blob_map_reader},
        {static_cast<TLVTraits::TagType>(Tag::ConstantMeta), constant_meta_reader},
        {static_cast<TLVTraits::TagType>(Tag::WeightSource), constant_source_reader},
    };
    return scan_tlv_records(stream, scanners);
}

SingleFileStorage::BlobIdType SingleFileStorage::convert_blob_id(const std::string& blob_id) {
    return static_cast<BlobIdType>(std::stoull(blob_id.c_str()));
}

bool SingleFileStorage::has_blob_id(BlobIdType blob_id) const {
    return m_blob_index.find(blob_id) != m_blob_index.end();
}

void SingleFileStorage::write_blob_entry(std::ofstream& stream, BlobIdType blob_id, StreamWriter& writer) {
    OPENVINO_ASSERT(!has_blob_id(blob_id), "Blob with id ", blob_id, " already exists in cache.");

    std::streampos blob_pos;
    std::streamoff blob_size;

    const auto blob_writer = [&](std::ostream& s) {
        s.write(reinterpret_cast<const char*>(&blob_id), sizeof(blob_id));
        write_padding(s, util::get_system_page_size());
        blob_pos = s.tellp();
        writer(s);
        blob_size = s.tellp() - blob_pos;
    };
    write_tlv_record(stream, static_cast<TLVTraits::TagType>(Tag::Blob), blob_writer);

    std::string model_name;  // Intentionally empty

    const auto blob_map_writer = [&](std::ostream& s) {
        s.write(reinterpret_cast<const char*>(&blob_id), sizeof(blob_id));
        write_tlv_string(s, model_name);
    };
    write_tlv_record(stream, static_cast<TLVTraits::TagType>(Tag::BlobMap), blob_map_writer);

    m_blob_index[blob_id] = {blob_pos, blob_size, model_name};
}

void SingleFileStorage::write_cache_entry(const std::string& blob_id, StreamWriter writer) {
    ScopedLocale plocal_C(LC_ALL, "C");
    std::ofstream stream(m_file_path, std::ios::binary | std::ios::in | std::ios::ate);
    write_blob_entry(stream, convert_blob_id(blob_id), writer);
}

void SingleFileStorage::read_cache_entry(const std::string& blob_id, bool enable_mmap, StreamReader reader) {
    ScopedLocale plocal_C(LC_ALL, "C");

    const auto cid = convert_blob_id(blob_id);

    if (std::filesystem::exists(m_file_path) && has_blob_id(cid)) {
        const auto& [blob_pos, blob_size, model_name] = m_blob_index[cid];
        if (enable_mmap) {
            // CVS-181859 Extend memory mapping helpers to suport partial file mapping
            CompiledBlobVariant compiled_blob{
                std::in_place_index<0>,
                make_tensor_from(m_file_path, blob_pos, {static_cast<Shape::value_type>(blob_size)})};
            reader(compiled_blob);
        } else {
            std::ifstream stream(m_file_path, std::ios::binary);
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

void SingleFileStorage::write_context(const weight_sharing::Context& context) {
    ScopedLocale plocal_C(LC_ALL, "C");
    std::ofstream stream(m_file_path, std::ios::binary | std::ios::in | std::ios::ate);

    weight_sharing::WeightRegistry delta_weight_registry;
    for (const auto& [source_id, const_meta_map] : context.m_weight_registry) {
        for (const auto& [const_id, const_meta] : const_meta_map) {
            if (m_shared_context.m_weight_registry.count(source_id) == 0 ||
                m_shared_context.m_weight_registry[source_id].count(const_id) == 0) {
                delta_weight_registry[source_id][const_id] = const_meta;
            }
        }
    }
    for (const auto& meta_map : delta_weight_registry) {
        const auto const_meta_writer = [&](std::ostream& s) {
            const auto& [source_id, const_meta] = meta_map;
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

                m_shared_context.m_weight_registry[source_id][const_id] = props;
            }
        };
        write_tlv_record(stream, static_cast<TLVTraits::TagType>(Tag::ConstantMeta), const_meta_writer);
    }

    weight_sharing::WeightSourceRegistry delta_cache_sources;
    for (const auto& [source_id, weight_buffer] : context.m_cache_sources) {
        if (m_shared_context.m_cache_sources.count(source_id) == 0) {
            delta_cache_sources[source_id] = weight_buffer;
        }
    }
    for (const auto& cache_registry : delta_cache_sources) {
        const auto weight_source_writer = [&](std::ostream& s) {
            const auto& [source_id, weight_buffer] = cache_registry;
            if (auto buf = weight_buffer.m_weights.lock()) {
                const auto device_id = weight_buffer.m_device;
                s.write(reinterpret_cast<const char*>(&device_id), sizeof(device_id));
                s.write(reinterpret_cast<const char*>(&source_id), sizeof(source_id));
                write_padding(s, util::get_system_page_size());
                s.write(reinterpret_cast<const char*>(buf->get_ptr()), buf->size());
            }

            m_shared_context.m_cache_sources[source_id] = weight_buffer;
        };
        write_tlv_record(stream, static_cast<TLVTraits::TagType>(Tag::WeightSource), weight_source_writer);
    }
}
};  // namespace ov::runtime
