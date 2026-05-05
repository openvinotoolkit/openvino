// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "orc.hpp"

#include <algorithm>
#include <cstring>
#include <istream>
#include <ostream>
#include <sstream>

namespace {

constexpr std::array<std::uint8_t, 8> ORC_FILE_MAGIC = {'N', 'P', 'U', 'W', 'O', 'R', 'C', '\0'};
constexpr std::uint16_t ORC_FILE_VERSION = 1u;

std::streamsize checked_stream_size(const std::size_t size) {
    if (size > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        OPENVINO_THROW("ORC size exceeds std::streamsize range");
    }
    return static_cast<std::streamsize>(size);
}

std::size_t checked_size(const std::uint64_t size) {
    if (size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        OPENVINO_THROW("ORC section size exceeds std::size_t range");
    }
    return static_cast<std::size_t>(size);
}

std::uint64_t checked_add(const std::uint64_t lhs, const std::uint64_t rhs) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        OPENVINO_THROW("ORC section size overflow");
    }
    return lhs + rhs;
}

constexpr std::uint64_t header_size() {
    // Section header layout: type_id (u16) + version (u16) + flags (u32) + payload_size (u64)
    return sizeof(ov::npuw::orc::TypeId) + sizeof(ov::npuw::orc::Version) + sizeof(ov::npuw::orc::SectionFlags) +
           sizeof(std::uint64_t);
}

void validate_section_shape(const ov::npuw::orc::Section& section) {
    if (section.is_container()) {
        if (!section.payload.empty()) {
            OPENVINO_THROW("ORC container sections cannot store a raw payload");
        }
        return;
    }

    if (!section.children.empty()) {
        OPENVINO_THROW("ORC raw sections cannot store nested child sections");
    }
}

std::uint64_t encoded_size(const ov::npuw::orc::Section& section);

std::uint64_t body_size(const ov::npuw::orc::Section& section) {
    validate_section_shape(section);
    if (!section.is_container()) {
        return section.payload.size();
    }

    std::uint64_t size = 0u;
    for (const auto& child : section.children) {
        size = checked_add(size, encoded_size(child));
    }
    return size;
}

std::uint64_t encoded_size(const ov::npuw::orc::Section& section) {
    return checked_add(header_size(), body_size(section));
}

void write_section(ov::npuw::orc::Stream& stream, const ov::npuw::orc::Section& section) {
    validate_section_shape(section);

    ov::npuw::orc::SectionHeader header;
    header.type = section.type;
    header.version = section.version;
    header.flags = section.flags;
    header.size = body_size(section);
    stream & header.type & header.version & header.flags & header.size;

    if (!section.is_container()) {
        if (!section.payload.empty()) {
            stream.bytes(const_cast<std::byte*>(section.payload.data()), section.payload.size());
        }
        return;
    }

    for (const auto& child : section.children) {
        write_section(stream, child);
    }
}

ov::npuw::orc::Section read_section(ov::npuw::orc::Stream& stream) {
    ov::npuw::orc::SectionHeader header;
    stream & header.type & header.version & header.flags & header.size;

    ov::npuw::orc::Section section;
    section.type = header.type;
    section.version = header.version;
    section.flags = header.flags;

    const auto size = checked_size(header.size);
    if (!section.is_container()) {
        section.payload.resize(size);
        if (size != 0u) {
            stream.bytes(section.payload.data(), size);
        }
        return section;
    }

    std::vector<std::byte> bytes(size);
    if (size != 0u) {
        stream.bytes(bytes.data(), size);
    }
    auto nested = ov::npuw::orc::Stream::memory_reader(bytes.data(), bytes.size());
    while (nested.remaining() != 0u) {
        section.children.push_back(read_section(nested));
    }
    return section;
}

}  // namespace

ov::npuw::orc::Stream ov::npuw::orc::Stream::reader(std::istream& stream) {
    Stream s;
    s.m_input = &stream;
    return s;
}

ov::npuw::orc::Stream ov::npuw::orc::Stream::memory_reader(const void* data, const std::size_t size) {
    Stream s;
    s.m_memory = reinterpret_cast<const std::byte*>(data);
    s.m_memory_size = size;
    return s;
}

ov::npuw::orc::Stream ov::npuw::orc::Stream::writer(std::ostream& stream) {
    Stream s;
    s.m_output = &stream;
    return s;
}

bool ov::npuw::orc::Stream::input() const {
    return m_input != nullptr || m_memory != nullptr;
}

bool ov::npuw::orc::Stream::output() const {
    return m_output != nullptr;
}

bool ov::npuw::orc::Stream::memory() const {
    return m_memory != nullptr;
}

std::size_t ov::npuw::orc::Stream::remaining() const {
    if (!memory()) {
        OPENVINO_THROW("ORC remaining() is only available for memory-backed input streams");
    }
    return m_memory_size - m_memory_offset;
}

void ov::npuw::orc::Stream::bytes(void* data, const std::size_t size) {
    if (output()) {
        m_output->write(reinterpret_cast<const char*>(data), checked_stream_size(size));
        if (!m_output->good()) {
            OPENVINO_THROW("Failed to write ORC stream bytes");
        }
        return;
    }

    if (memory()) {
        if (size > remaining()) {
            OPENVINO_THROW("Unexpected end of ORC memory stream");
        }
        if (size != 0u) {
            std::memcpy(data, m_memory + m_memory_offset, size);
            m_memory_offset += size;
        }
        return;
    }

    m_input->read(reinterpret_cast<char*>(data), checked_stream_size(size));
    if (!m_input->good()) {
        OPENVINO_THROW("Unexpected end of ORC input stream");
    }
}

void ov::npuw::orc::serialize(Stream& stream, std::string& value) {
    if (stream.output()) {
        auto size = value.size();
        stream & size;
        if (!value.empty()) {
            stream.bytes(value.data(), value.size());
        }
        return;
    }

    std::size_t size = 0u;
    stream & size;
    value.resize(size);
    if (size != 0u) {
        stream.bytes(value.data(), value.size());
    }
}

ov::npuw::orc::Section ov::npuw::orc::Section::raw(const TypeId type,
                                                   const Version version,
                                                   std::vector<std::byte> payload,
                                                   const SectionFlags flags) {
    Section section;
    section.type = type;
    section.version = version;
    section.flags = flags | SectionFlag::LEAF;
    section.payload = std::move(payload);
    return section;
}

ov::npuw::orc::Section ov::npuw::orc::Section::container(const TypeId type,
                                                         const Version version,
                                                         std::vector<Section> children,
                                                         const SectionFlags flags) {
    Section section;
    section.type = type;
    section.version = version;
    section.flags = flags;  // LEAF not set — this is a structural container section
    section.children = std::move(children);
    return section;
}

void ov::npuw::orc::write_file(std::ostream& stream, const Section& root, const SchemaUUID& uuid) {
    auto writer = Stream::writer(stream);

    auto magic = ORC_FILE_MAGIC;
    writer & magic;
    auto version = ORC_FILE_VERSION;
    writer & version;
    auto uuid_copy = uuid;
    writer & uuid_copy;
    write_section(writer, root);
}

ov::npuw::orc::Section ov::npuw::orc::read_file(std::istream& stream) {
    const auto header = is_orc(stream);
    if (!header) {
        OPENVINO_THROW("Not an ORC file (invalid magic)");
    }
    if (header->version != ORC_FILE_VERSION) {
        OPENVINO_THROW("Unsupported ORC file version ", header->version);
    }

    // is_orc() restores the stream position; skip past the file header before reading sections.
    constexpr auto skip =
        static_cast<std::streamoff>(ORC_FILE_MAGIC.size() + sizeof(ORC_FILE_VERSION) + sizeof(SchemaUUID));
    stream.seekg(skip, std::ios::cur);

    auto reader = Stream::reader(stream);
    return read_section(reader);
}

bool ov::npuw::orc::Schema::knows(const TypeId type) const {
    return m_entries.count(type) != 0u;
}

std::any ov::npuw::orc::Schema::load_any(const Section& section) const {
    const auto it = m_entries.find(section.type);
    if (it == m_entries.end()) {
        OPENVINO_THROW("No ORC loader registered for section type ID ", section.type);
    }
    return it->second.loader(section, *this);
}

std::vector<ov::npuw::orc::Schema::LoadedChild> ov::npuw::orc::Schema::load_children(const Section& container) const {
    if (!container.is_container()) {
        OPENVINO_THROW("ORC child loading requires a container section");
    }

    std::vector<LoadedChild> out;
    out.reserve(container.children.size());
    for (const auto& child : container.children) {
        if (!knows(child.type)) {
            if (child.is_optional()) {
                continue;
            }
            OPENVINO_THROW("Required ORC child section type ID ", child.type, " is not registered in the schema");
        }
        out.push_back({child.type, load_any(child)});
    }
    return out;
}

const ov::npuw::orc::Section* ov::npuw::orc::Schema::find_child(const Section& container, const TypeId type) const {
    if (!container.is_container()) {
        OPENVINO_THROW("ORC child lookup requires a container section");
    }

    const auto it = std::find_if(container.children.begin(), container.children.end(), [type](const Section& child) {
        return child.type == type;
    });
    return it == container.children.end() ? nullptr : &(*it);
}

const ov::npuw::orc::Section& ov::npuw::orc::Schema::require_child(const Section& container, const TypeId type) const {
    const auto* child = find_child(container, type);
    if (child == nullptr) {
        OPENVINO_THROW("Required ORC child section type ID ", type, " is missing");
    }
    return *child;
}

std::optional<ov::npuw::orc::OrcHeader> ov::npuw::orc::is_orc(std::istream& stream) {
    const auto saved = stream.tellg();
    const auto restore = [&] {
        stream.clear();
        stream.seekg(saved);
    };

    std::array<std::uint8_t, 8> magic{};
    stream.read(reinterpret_cast<char*>(magic.data()), magic.size());
    if (!stream.good() || magic != ORC_FILE_MAGIC) {
        restore();
        return std::nullopt;
    }

    std::uint16_t version = 0u;
    stream.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!stream.good()) {
        restore();
        return std::nullopt;
    }

    SchemaUUID uuid{};
    stream.read(reinterpret_cast<char*>(uuid.data()), uuid.size());
    if (!stream.good()) {
        restore();
        return std::nullopt;
    }

    restore();
    return OrcHeader{version, uuid};
}
