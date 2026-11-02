// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/hsm_reader_handler.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <optional>
#include <utility>

#include "common_utils/parallel_mem_streambuf.hpp"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/plugin/compiled_model.hpp"
#include "openvino/runtime/hsm_reader.hpp"

namespace ov::intel_gpu::hsm {
namespace {

class PrefixInputStreamBuf final : public std::streambuf {
public:
    PrefixInputStreamBuf(std::streambuf* source, char* prefix, size_t size) : m_source(source) {
        setg(prefix, prefix, prefix + size);
    }

protected:
    std::streamsize xsgetn(char* destination, std::streamsize count) override {
        const auto prefix = std::min(count, static_cast<std::streamsize>(egptr() - gptr()));
        std::memcpy(destination, gptr(), static_cast<size_t>(prefix));
        gbump(static_cast<int>(prefix));
        return prefix + m_source->sgetn(destination + prefix, count - prefix);
    }
    int_type underflow() override {
        return m_source->sgetc();
    }
    int_type uflow() override {
        return m_source->sbumpc();
    }

private:
    std::streambuf* m_source;
};

// The delegated mmap file reader can see the rest of the file: bound reads and prefetch alike.
class BoundedInputStreamBuf final : public std::streambuf {
public:
    BoundedInputStreamBuf(std::istream& source, pos_type start, size_t size) : m_source(source.rdbuf()), m_start(start) {
        OPENVINO_ASSERT(m_source && start >= pos_type(0) && size <= static_cast<size_t>(std::numeric_limits<off_type>::max()) &&
                            static_cast<off_type>(start) <= std::numeric_limits<off_type>::max() - static_cast<off_type>(size),
                        "[GPU] HSM section input range overflow");
        m_size = static_cast<off_type>(size);
        OPENVINO_ASSERT(m_source->pubseekpos(start, std::ios::in) == start, "[GPU] HSM section input requires seeking");
    }

    std::streambuf* source() const {
        return m_source;
    }
    std::streamsize remaining() const {
        return m_size - m_position;
    }

protected:
    std::streamsize xsgetn(char* destination, std::streamsize count) override {
        if (count <= 0)
            return 0;
        const auto read = m_source->sgetn(destination, std::min(count, remaining()));
        m_position += read;
        return read;
    }
    int_type underflow() override {
        return remaining() ? m_source->sgetc() : traits_type::eof();
    }
    int_type uflow() override {
        const auto ch = remaining() ? m_source->sbumpc() : traits_type::eof();
        if (!traits_type::eq_int_type(ch, traits_type::eof()))
            ++m_position;
        return ch;
    }
    std::streamsize showmanyc() override {
        return remaining();
    }
    pos_type seekoff(off_type offset, std::ios_base::seekdir direction, std::ios_base::openmode mode) override {
        const auto base = direction == std::ios_base::beg   ? off_type(0)
                          : direction == std::ios_base::cur ? m_position
                          : direction == std::ios_base::end ? m_size
                                                            : off_type(-1);
        if (!(mode & std::ios::in) || base < 0 || offset < -base || offset > m_size - base)
            return pos_type(off_type(-1));
        if (offset == 0 && direction == std::ios_base::cur)
            return pos_type(m_position);
        const auto next = base + offset;
        if (m_source->pubseekpos(m_start + next, std::ios::in) != m_start + next)
            return pos_type(off_type(-1));
        m_position = next;
        return pos_type(next);
    }
    pos_type seekpos(pos_type position, std::ios_base::openmode mode) override {
        return seekoff(static_cast<off_type>(position), std::ios_base::beg, mode);
    }

private:
    std::streambuf* m_source;
    pos_type m_start;
    off_type m_size = 0;
    off_type m_position = 0;
};

class Sections final : public wire::ISectionReaderHandler {
public:
    explicit Sections(const wire::Reader& reader) {
        reader.read_sections({this});
        OPENVINO_ASSERT(m_info && m_requirements && m_model, "[GPU] Missing mandatory HSM section");
        OPENVINO_ASSERT(m_info->size() == sizeof(PayloadInfo) && m_info->read(&info), "[GPU] Invalid HSM payload metadata");
        OPENVINO_ASSERT(info.version == 1 && info.cache_mode <= 1 && !(info.flags & ~encrypted) && info.requirements_version == 2 && info.reserved == 0,
                        "[GPU] Unsupported HSM GPU payload schema");
        OPENVINO_ASSERT(m_requirements->size() > 0 && m_requirements->size() <= max_requirements_size, "[GPU] Invalid HSM runtime requirements size");
        OPENVINO_ASSERT(m_model->size() > 0, "[GPU] Empty HSM model payload");
    }

    bool handle_section(const wire::ManifestEntry& entry, const wire::Section& section) override {
        if (entry.device != wire::gpu_device_id)
            return false;
        if (entry.tag.id() == payload_info_tag.id()) {
            accept(m_info, payload_info_tag, entry, section);
        } else if (entry.tag.id() == wire::runtime_requirements_tag.id()) {
            accept(m_requirements, wire::runtime_requirements_tag, entry, section);
        } else if (entry.tag.id() == wire::model_tag.id()) {
            accept(m_model, wire::model_tag, entry, section);
            model_entry = entry;
        } else {
            return false;
        }
        return true;
    }

    const wire::Section& model() const {
        return *m_model;
    }
    std::string requirements() const {
        std::string value(m_requirements->size(), '\0');
        OPENVINO_ASSERT(m_requirements->read(value.data()), "[GPU] Cannot read HSM runtime requirements");
        return value;
    }

    PayloadInfo info;
    wire::ManifestEntry model_entry{};

private:
    static void accept(std::optional<wire::Section>& target, wire::SectionTag expected, const wire::ManifestEntry& entry, const wire::Section& section) {
        OPENVINO_ASSERT(entry.tag.value == expected.value, "[GPU] Invalid HSM section storage mode");
        OPENVINO_ASSERT(!target, "[GPU] Duplicate HSM singleton section");
        target = section;
    }

    std::optional<wire::Section> m_info;
    std::optional<wire::Section> m_requirements;
    std::optional<wire::Section> m_model;
};

std::string decrypt_payload(const wire::Section& section, const Crypt& decrypt) {
    OPENVINO_ASSERT(decrypt, "[GPU] Encrypted HSM payload requires a decryption callback");
    size_t size = 0;
    OPENVINO_ASSERT(section.size() >= sizeof(size) && section.read(0, sizeof(size), &size) && size > 0 && size == section.size() - sizeof(size),
                    "[GPU] Invalid encrypted HSM payload length");
    std::string ciphertext(size, '\0');
    OPENVINO_ASSERT(section.read(sizeof(size), size, ciphertext.data()), "[GPU] Cannot read encrypted HSM payload");
    auto plaintext = decrypt(ciphertext);
    OPENVINO_ASSERT(!plaintext.empty(), "[GPU] Decryption produced an empty HSM payload");
    return plaintext;
}

}  // namespace

struct ReaderHandler::Impl {
    Impl(std::istream& model, cldnn::engine& engine, std::shared_ptr<ov::Tensor> backing, Crypt decrypt) : source(model.rdbuf()), backing(std::move(backing)) {
        OPENVINO_ASSERT(source, "[GPU] Invalid compiled-model input stream");
        const auto start = source->pubseekoff(0, std::ios::cur, std::ios::in);
        const auto probe_size = source->sgetn(magic.data(), magic.size());
        is_hsm = probe_size == static_cast<std::streamsize>(magic.size()) && std::memcmp(magic.data(), "OVBLS", magic.size()) == 0;
        const bool seekable = start != std::streampos(-1);
        if (seekable)
            OPENVINO_ASSERT(source->pubseekpos(start, std::ios::in) == start, "[GPU] Cannot restore blob stream position");

        if (!is_hsm) {
            std::istream* legacy_stream = &model;
            if (!seekable) {
                prefix_buffer = std::make_unique<PrefixInputStreamBuf>(source, magic.data(), static_cast<size_t>(probe_size));
                payload_stream = std::make_unique<std::istream>(prefix_buffer.get());
                legacy_stream = payload_stream.get();
            }
            input = decrypt ? std::make_unique<cldnn::EncryptedBinaryInputBuffer>(*legacy_stream, engine, decrypt)
                            : std::make_unique<cldnn::BinaryInputBuffer>(
                                  *legacy_stream,
                                  engine,
                                  this->backing ? static_cast<const size_t*>(std::as_const(*this->backing).data()) : nullptr);
            *input >> cldnn::make_data(&cache_mode, sizeof(cache_mode));
            return;
        }

        OPENVINO_ASSERT(seekable, "[GPU] HSM import requires a seekable input stream");
        wire::Header header{};
        OPENVINO_ASSERT(source->sgetn(reinterpret_cast<char*>(&header), sizeof(header)) == sizeof(header) && source->pubseekpos(start, std::ios::in) == start,
                        "[GPU] Truncated HSM header");
        // Master Reader lacks the CPU PR's overflow guard. Validate locally before calling it.
        OPENVINO_ASSERT(
            wire::is_valid_header_fields(header) && start >= std::streampos(0) && header.container_size <= std::numeric_limits<size_t>::max() &&
                header.container_size <= static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()) &&
                static_cast<std::streamoff>(start) <= std::numeric_limits<std::streamoff>::max() - static_cast<std::streamoff>(header.container_size),
            "[GPU] Invalid or unsupported HSM container header");
        container_end = start + static_cast<std::streamoff>(header.container_size);
        const auto source_end = source->pubseekoff(0, std::ios::end, std::ios::in);
        OPENVINO_ASSERT(source_end >= container_end && source->pubseekpos(start, std::ios::in) == start,
                        "[GPU] Truncated HSM container or unsupported input seeking");
        auto reader = this->backing ? wire::Reader::open(static_cast<const std::byte*>(std::as_const(*this->backing).data()), this->backing->get_byte_size())
                                    : wire::Reader::open(model);
        OPENVINO_ASSERT(reader, "[GPU] Invalid or unsupported HSM container");
        Sections sections(*reader);
        requirements = sections.requirements();
        const auto& device_info = engine.get_device_info();
        OPENVINO_ASSERT(CompiledModel::is_runtime_requirements_compatible(requirements, device_info),
                        "[GPU] Cannot import HSM GPU payload: incompatible runtime configuration.\n  blob: ",
                        requirements,
                        "\n  current: ",
                        CompiledModel::build_runtime_requirements(device_info));
        cache_mode = sections.info.cache_mode == 0 ? ov::CacheMode::OPTIMIZE_SPEED : ov::CacheMode::OPTIMIZE_SIZE;
        const bool is_encrypted = sections.info.flags & encrypted;
        size_t payload_size = sections.model().size();
        const size_t* tensor_base = nullptr;
        if (is_encrypted) {
            // Preserve original file-I/O delegation rather than faulting retained ciphertext mmap pages.
            const wire::Section ciphertext(sections.model_entry, model, start + static_cast<std::streamoff>(sections.model_entry.offset));
            plaintext = decrypt_payload(ciphertext, decrypt);
            payload_size = plaintext.size();
            memory_buffer = std::make_unique<ParallelMemStreamBuf>(plaintext.data(), payload_size);
        } else if (const auto view = sections.model().view(); !view.empty()) {
            tensor_base = reinterpret_cast<const size_t*>(view.data());
            memory_buffer = std::make_unique<ParallelMemStreamBuf>(view.data(), view.size());
        } else {
            section_buffer = std::make_unique<BoundedInputStreamBuf>(model, start + static_cast<std::streamoff>(sections.model_entry.offset), payload_size);
        }
        if (memory_buffer) {
            memory_stream = std::make_unique<std::istream>(memory_buffer.get());
            section_buffer = std::make_unique<BoundedInputStreamBuf>(*memory_stream, 0, payload_size);
        }
        payload_stream = std::make_unique<std::istream>(section_buffer.get());
        input = std::make_unique<cldnn::BinaryInputBuffer>(*payload_stream, engine, tensor_base, is_encrypted);
        input->set_read_limit(payload_size);
    }

    std::streambuf* source;
    std::shared_ptr<ov::Tensor> backing;
    std::array<char, 5> magic{};
    bool is_hsm = false;
    ov::CacheMode cache_mode = ov::CacheMode::OPTIMIZE_SPEED;
    std::streampos container_end{};
    std::string requirements;
    std::string plaintext;
    std::unique_ptr<PrefixInputStreamBuf> prefix_buffer;
    std::unique_ptr<ParallelMemStreamBuf> memory_buffer;
    std::unique_ptr<std::istream> memory_stream;
    std::unique_ptr<BoundedInputStreamBuf> section_buffer;
    std::unique_ptr<std::istream> payload_stream;
    std::unique_ptr<cldnn::BinaryInputBuffer> input;
};

ReaderHandler::ReaderHandler(std::istream& source, cldnn::engine& engine, std::shared_ptr<ov::Tensor> backing, Crypt decrypt)
    : m_impl(std::make_unique<Impl>(source, engine, std::move(backing), std::move(decrypt))) {}

ReaderHandler::~ReaderHandler() = default;

cldnn::BinaryInputBuffer& ReaderHandler::buffer() {
    return *m_impl->input;
}

ov::CacheMode ReaderHandler::cache_mode() const {
    return m_impl->cache_mode;
}

const std::string& ReaderHandler::requirements() const {
    return m_impl->requirements;
}

void ReaderHandler::finish() {
    if (m_impl->is_hsm)
        OPENVINO_ASSERT(m_impl->source->pubseekpos(m_impl->container_end, std::ios::in) == m_impl->container_end, "[GPU] Cannot seek past HSM container");
}

bool prefetch_cache_data(std::streambuf* buffer, std::streamsize size) {
    while (auto* bounded = dynamic_cast<BoundedInputStreamBuf*>(buffer)) {
        size = std::min(size, bounded->remaining());
        buffer = bounded->source();
    }
    if (auto* file = dynamic_cast<ov::util::ParallelReadStreamBuf*>(buffer))
        return file->prefetch(size);
    if (auto* memory = dynamic_cast<ParallelMemStreamBuf*>(buffer))
        return memory->prefetch(size);
    return false;
}

}  // namespace ov::intel_gpu::hsm
