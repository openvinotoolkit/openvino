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
constexpr std::uint16_t ORC_FILE_VERSION = 0u;

std::streampos checked_tellp(std::ostream& stream, const char* context) {
    const auto pos = stream.tellp();
    if (pos == std::streampos(-1)) {
        OPENVINO_THROW("ORC export requires a seekable output stream for ", context);
    }
    return pos;
}

std::streampos checked_tellg(std::istream& stream, const char* context) {
    const auto pos = stream.tellg();
    if (pos == std::streampos(-1)) {
        OPENVINO_THROW("ORC import requires a seekable input stream for ", context);
    }
    return pos;
}

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

void ov::npuw::orc::serialize(Stream& stream, Section& section) {
    if (stream.output()) {
        write_section(stream, section);
    } else {
        section = read_section(stream);
    }
}

void ov::npuw::orc::serialize(Stream& stream, SectionHeader& header) {
    stream & header.type & header.version & header.flags & header.size;
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

bool ov::npuw::orc::try_read_bytes(std::istream& stream, void* data, const std::size_t size) {
    const auto saved = stream.tellg();
    stream.read(reinterpret_cast<char*>(data), checked_stream_size(size));
    if (stream.good()) {
        return true;
    }

    stream.clear();
    stream.seekg(saved);
    return false;
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
    section.flags = flags;  // LEAF not set; this is a structural container section
    section.children = std::move(children);
    return section;
}

void ov::npuw::orc::write_file_header(std::ostream& stream, const SchemaUUID& uuid) {
    auto writer = Stream::writer(stream);

    auto magic = ORC_FILE_MAGIC;
    writer & magic;
    auto version = ORC_FILE_VERSION;
    writer & version;
    auto uuid_copy = uuid;
    writer & uuid_copy;
}

ov::npuw::orc::OrcHeader ov::npuw::orc::read_file_header(std::istream& stream) {
    const auto header = is_orc(stream);
    if (!header) {
        OPENVINO_THROW("Not an ORC file (invalid magic)");
    }
    if (header->version != ORC_FILE_VERSION) {
        OPENVINO_THROW("Unsupported ORC file version ", header->version);
    }

    constexpr auto skip =
        static_cast<std::streamoff>(ORC_FILE_MAGIC.size() + sizeof(ORC_FILE_VERSION) + sizeof(SchemaUUID));
    stream.seekg(skip, std::ios::cur);
    if (!stream.good()) {
        OPENVINO_THROW("Failed to seek past ORC file header");
    }
    return *header;
}

void ov::npuw::orc::write_file(std::ostream& stream, const Section& root, const SchemaUUID& uuid) {
    write_file_header(stream, uuid);
    auto writer = Stream::writer(stream);
    write_section(writer, root);
}

ov::npuw::orc::Section ov::npuw::orc::read_file(std::istream& stream) {
    read_file_header(stream);
    auto reader = Stream::reader(stream);
    return read_section(reader);
}

std::streamsize ov::npuw::orc::checked_stream_size(const std::size_t size) {
    return ::checked_stream_size(size);
}

std::size_t ov::npuw::orc::checked_size(const std::uint64_t size) {
    return ::checked_size(size);
}

std::streamoff ov::npuw::orc::checked_streamoff(const std::uint64_t size) {
    if (size > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        OPENVINO_THROW("ORC section size exceeds std::streamoff range");
    }
    return static_cast<std::streamoff>(size);
}

ov::npuw::orc::ScopedWriteSection::ScopedWriteSection(std::ostream& stream,
                                                      const TypeId type,
                                                      const Version version,
                                                      const SectionFlags flags)
    : m_stream(stream) {
    SectionHeader header;
    header.type = type;
    header.version = version;
    header.flags = flags;
    header.size = 0u;

    auto writer = Stream::writer(stream);
    writer & header.type & header.version & header.flags;
    m_size_pos = checked_tellp(stream, "section size patching");
    writer & header.size;
    m_body_pos = checked_tellp(stream, "section body tracking");
}

void ov::npuw::orc::ScopedWriteSection::close() {
    if (m_closed) {
        return;
    }

    const auto end_pos = checked_tellp(m_stream, "section size patching");
    const auto body_size = end_pos - m_body_pos;
    if (body_size < 0) {
        OPENVINO_THROW("ORC section body size underflow");
    }

    const auto restore_pos = end_pos;
    m_stream.seekp(m_size_pos);
    if (!m_stream.good()) {
        OPENVINO_THROW("Failed to seek to ORC section size field");
    }

    auto writer = Stream::writer(m_stream);
    auto size = static_cast<std::uint64_t>(body_size);
    writer & size;
    m_stream.seekp(restore_pos);
    if (!m_stream.good()) {
        OPENVINO_THROW("Failed to restore ORC output position after patching section size");
    }
    m_closed = true;
}

ov::npuw::orc::ScopedReadSection::ScopedReadSection(std::istream& stream) : m_stream(stream) {
    auto reader = Stream::reader(stream);
    reader & m_header;
    m_end = checked_tellg(stream, "section bounds") + checked_streamoff(m_header.size);
}

const ov::npuw::orc::SectionHeader& ov::npuw::orc::ScopedReadSection::header() const {
    return m_header;
}

bool ov::npuw::orc::ScopedReadSection::done() const {
    return checked_tellg(m_stream, "section iteration") >= m_end;
}

std::size_t ov::npuw::orc::ScopedReadSection::remaining() const {
    const auto pos = checked_tellg(m_stream, "section remaining");
    if (pos > m_end) {
        OPENVINO_THROW("ORC stream advanced past the end of the current section");
    }
    return checked_size(static_cast<std::uint64_t>(m_end - pos));
}

void ov::npuw::orc::ScopedReadSection::expect_end() const {
    if (!done() || checked_tellg(m_stream, "section completion") != m_end) {
        OPENVINO_THROW("Malformed ORC section contents");
    }
}

std::string ov::npuw::orc::ScopedReadSection::read_remaining_blob() {
    std::string blob(remaining(), '\0');
    if (!blob.empty()) {
        m_stream.read(blob.data(), checked_stream_size(blob.size()));
        if (!m_stream.good()) {
            OPENVINO_THROW("Unexpected end of ORC input stream");
        }
    }
    return blob;
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
