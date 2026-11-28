// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/hsm_writer_handler.hpp"

#include <array>
#include <cstring>
#include <limits>
#include <utility>

#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/plugin/compiled_model.hpp"
#include "openvino/runtime/hsm_compile_options_codec.hpp"
#include "openvino/runtime/hsm_deferred_writer.hpp"
#include "openvino/util/memory.hpp"

namespace ov::intel_gpu::hsm {
namespace {

// The graph is written as ~10^5 small records; one sink call per record measurably slowed down weightless export.
class SectionOutputStreamBuf final : public std::streambuf {
public:
    explicit SectionOutputStreamBuf(const runtime::hsm::SectionSink& sink) : m_sink(sink) {}

protected:
    std::streamsize xsputn(const char* data, std::streamsize size) override {
        OPENVINO_ASSERT(size >= 0, "[GPU] Invalid HSM section output size");
        const auto count = static_cast<size_t>(size);
        if (count == 0) {
            return 0;
        }
        if (count <= m_buffer.size() - m_used) {
            std::memcpy(m_buffer.data() + m_used, data, count);
            m_used += count;
        } else {
            sync();
            if (count >= m_buffer.size()) {
                emit(data, count);
            } else {
                std::memcpy(m_buffer.data(), data, count);
                m_used = count;
            }
        }
        return size;
    }

    int_type overflow(int_type ch) override {
        if (!traits_type::eq_int_type(ch, traits_type::eof())) {
            const auto value = traits_type::to_char_type(ch);
            xsputn(&value, 1);
        }
        return traits_type::not_eof(ch);
    }

    int sync() override {
        if (m_used) {
            emit(m_buffer.data(), m_used);
            m_used = 0;
        }
        return 0;
    }

    pos_type seekoff(off_type offset, std::ios_base::seekdir direction, std::ios_base::openmode /*mode*/) override {
        if (offset != 0 || direction != std::ios_base::cur || m_used > static_cast<size_t>(std::numeric_limits<off_type>::max()) - m_emitted) {
            return pos_type(off_type(-1));
        }
        return pos_type(static_cast<off_type>(m_emitted + m_used));
    }

private:
    void emit(const char* data, size_t size) {
        OPENVINO_ASSERT(size <= static_cast<size_t>(std::numeric_limits<off_type>::max()) - m_emitted, "[GPU] HSM section output offset overflow");
        m_sink({reinterpret_cast<const std::byte*>(data), size});
        m_emitted += size;
    }

    const runtime::hsm::SectionSink& m_sink;
    std::array<char, 64 * 1024> m_buffer;
    size_t m_used = 0;
    size_t m_emitted = 0;
};

class PlaintextBuffer final : public cldnn::BinaryOutputBuffer {
public:
    PlaintextBuffer(std::ostream& stream, size_t capacity) : BinaryOutputBuffer(stream, true) {
        if (capacity <= m_plaintext.max_size()) {
            m_plaintext.reserve(capacity);
        }
    }

    void write(const void* data, std::streamsize size) override {
        OPENVINO_ASSERT(size >= 0, "[GPU] Invalid HSM plaintext size");
        m_plaintext.append(static_cast<const char*>(data), static_cast<size_t>(size));
    }

    const std::string& plaintext() const {
        return m_plaintext;
    }

private:
    std::string m_plaintext;
};

class ModelEncoder final : public runtime::hsm::ISectionEncoder {
public:
        ModelEncoder(std::function<void(cldnn::BinaryOutputBuffer&)> serialize,
                                 decltype(ov::EncryptionCallbacks::encrypt) encrypt,
                                 size_t plaintext_capacity)
        : m_serialize(std::move(serialize)),
                    m_encrypt(std::move(encrypt)),
                    m_plaintext_capacity(plaintext_capacity) {}

    void encode(const runtime::hsm::SectionSink& sink) const override {
        SectionOutputStreamBuf buffer(sink);
        std::ostream stream(&buffer);
        stream.exceptions(std::ios::badbit | std::ios::failbit);
        if (m_encrypt) {
            PlaintextBuffer output(stream, m_plaintext_capacity);
            m_serialize(output);
            const auto ciphertext = m_encrypt(output.plaintext());
            OPENVINO_ASSERT(!ciphertext.empty(), "[GPU] HSM encryption produced an empty model");
            sink(ov::util::as_memory_view(ciphertext));
            return;
        }
        cldnn::BinaryOutputBuffer output(stream);
        m_serialize(output);
        stream.flush();
        OPENVINO_ASSERT(stream.tellp() > std::streampos(0), "[GPU] HSM serialization produced an empty model");
    }

private:
    std::function<void(cldnn::BinaryOutputBuffer&)> m_serialize;
    decltype(ov::EncryptionCallbacks::encrypt) m_encrypt;
    size_t m_plaintext_capacity;
};

}  // namespace

WriterHandler::WriterHandler(ov::CacheMode cache_mode,
                             std::string requirements,
                             std::function<void(cldnn::BinaryOutputBuffer&)> serialize,
                             const decltype(ov::EncryptionCallbacks::encrypt)& encrypt,
                             size_t plaintext_capacity)
    : m_requirements(std::move(requirements)),
      m_compile_options{ov::cache_mode(cache_mode)},
    m_encoder(std::make_shared<ModelEncoder>(std::move(serialize), encrypt, plaintext_capacity)) {
    OPENVINO_ASSERT(cache_mode == ov::CacheMode::OPTIMIZE_SPEED || cache_mode == ov::CacheMode::OPTIMIZE_SIZE, "[GPU] Invalid stored HSM cache mode");
    OPENVINO_ASSERT(!m_requirements.empty() && m_requirements.size() <= CompiledModel::runtime_requirements_max_size,
                    "[GPU] Invalid HSM runtime requirements size");
    m_info.flags = encrypt ? encrypted_flag : 0;
    m_info.requirements_version = CompiledModel::runtime_requirements_version;
}

void WriterHandler::handle_section(runtime::hsm::IWriter& writer) const {
    const auto add = [&writer](const SectionId& id, auto&&... args) {
        OPENVINO_ASSERT(writer.add_section(id.device, id.tag, std::forward<decltype(args)>(args)...), "[GPU] Cannot add HSM section ", id.tag.id());
    };
    add(payload_info, ov::util::as_memory_view(m_info));
    add(runtime_requirements, ov::util::as_memory_view(m_requirements));
    add(compile_options, std::make_shared<runtime::hsm::CompileOptionsEncoder>(m_compile_options));
    add(model, m_encoder, runtime::hsm::SectionAlignment{ov::util::min_page_alignment, 0});
}

void write_model(std::ostream& destination,
                 ov::CacheMode cache_mode,
                 const std::string& requirements,
                 std::function<void(cldnn::BinaryOutputBuffer&)> serialize,
                 const decltype(ov::EncryptionCallbacks::encrypt)& encrypt,
                 size_t plaintext_capacity) {
    WriterHandler handler(cache_mode, requirements, std::move(serialize), encrypt, plaintext_capacity);
    auto writer = runtime::hsm::DeferredWriter::open(destination);
    OPENVINO_ASSERT(writer, "[GPU] HSM export requires a seekable output stream");
    writer->add_sections({&handler});
    const auto error = writer->finalize();
    OPENVINO_ASSERT(!error, "[GPU] HSM export failed: ", error.message());
}

}  // namespace ov::intel_gpu::hsm
