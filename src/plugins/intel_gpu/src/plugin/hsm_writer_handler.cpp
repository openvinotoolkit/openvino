// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/hsm_writer_handler.hpp"

#include <array>
#include <cstring>
#include <limits>
#include <utility>

#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "openvino/runtime/hsm_deferred_writer.hpp"
#include "openvino/util/file_util.hpp"

namespace ov::intel_gpu::hsm {
namespace {

// Amortize small graph records without bypassing the destination ostream in SectionSink.
class SectionOutputStreamBuf final : public std::streambuf {
public:
    explicit SectionOutputStreamBuf(const wire::SectionSink& sink) : m_sink(sink) {}

protected:
    std::streamsize xsputn(const char* data, std::streamsize size) override {
        OPENVINO_ASSERT(size >= 0, "[GPU] Invalid HSM section output size");
        const auto count = static_cast<size_t>(size);
        if (count == 0)
            return 0;
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

    pos_type seekoff(off_type offset, std::ios_base::seekdir direction, std::ios_base::openmode) override {
        if (offset != 0 || direction != std::ios_base::cur || m_used > static_cast<size_t>(std::numeric_limits<off_type>::max()) - m_emitted)
            return pos_type(off_type(-1));
        return pos_type(static_cast<off_type>(m_emitted + m_used));
    }

private:
    void emit(const char* data, size_t size) {
        OPENVINO_ASSERT(size <= static_cast<size_t>(std::numeric_limits<off_type>::max()) - m_emitted, "[GPU] HSM section output offset overflow");
        m_sink({reinterpret_cast<const std::byte*>(data), size});
        m_emitted += size;
    }

    const wire::SectionSink& m_sink;
    std::array<char, 64 * 1024> m_buffer;
    size_t m_used = 0;
    size_t m_emitted = 0;
};

class ModelEncoder final : public wire::ISectionEncoder {
public:
    ModelEncoder(std::function<void(cldnn::BinaryOutputBuffer&)> serialize, Crypt encrypt) : m_serialize(std::move(serialize)), m_encrypt(std::move(encrypt)) {}

    void encode(const wire::SectionSink& sink) const override {
        SectionOutputStreamBuf buffer(sink);
        std::ostream stream(&buffer);
        stream.exceptions(std::ios::badbit | std::ios::failbit);
        if (m_encrypt) {
            cldnn::EncryptedBinaryOutputBuffer output(stream, m_encrypt);
            m_serialize(output);
            output.flush();
        } else {
            cldnn::BinaryOutputBuffer output(stream);
            m_serialize(output);
        }
        stream.flush();
        OPENVINO_ASSERT(stream.tellp() > std::streampos(0), "[GPU] HSM serialization produced an empty model");
    }

private:
    std::function<void(cldnn::BinaryOutputBuffer&)> m_serialize;
    Crypt m_encrypt;
};

class WriterHandler final : public wire::ISectionWriterHandler {
public:
    WriterHandler(ov::CacheMode cache_mode, const std::string& requirements, std::function<void(cldnn::BinaryOutputBuffer&)> serialize, Crypt encrypt)
        : m_requirements(requirements),
          m_encoder(std::make_shared<ModelEncoder>(std::move(serialize), encrypt)) {
        OPENVINO_ASSERT(cache_mode == ov::CacheMode::OPTIMIZE_SPEED || cache_mode == ov::CacheMode::OPTIMIZE_SIZE, "[GPU] Invalid stored HSM cache mode");
        OPENVINO_ASSERT(!requirements.empty() && requirements.size() <= max_requirements_size, "[GPU] Invalid HSM runtime requirements size");
        m_info.cache_mode = cache_mode == ov::CacheMode::OPTIMIZE_SPEED ? 0 : 1;
        m_info.flags = encrypt ? encrypted : 0;
    }

    void handle_section(wire::IWriter& writer) const override {
        const auto view = [](const void* data, size_t size) {
            return ov::util::MemoryView{static_cast<const std::byte*>(data), size};
        };
        OPENVINO_ASSERT(writer.add_section(wire::gpu_device_id, payload_info_tag, view(&m_info, sizeof(m_info))), "[GPU] Cannot register HSM payload metadata");
        OPENVINO_ASSERT(writer.add_section(wire::gpu_device_id, wire::runtime_requirements_tag, view(m_requirements.data(), m_requirements.size())),
                        "[GPU] Cannot register HSM runtime requirements");
        OPENVINO_ASSERT(writer.add_section(wire::gpu_device_id, wire::model_tag, m_encoder, wire::SectionAlignment{ov::util::min_page_alignment, 0}),
                        "[GPU] Cannot register HSM model payload");
    }

private:
    PayloadInfo m_info;
    const std::string& m_requirements;
    wire::SectionEncoderPtr m_encoder;
};

}  // namespace

void write_model(std::ostream& destination,
                 ov::CacheMode cache_mode,
                 const std::string& requirements,
                 std::function<void(cldnn::BinaryOutputBuffer&)> serialize,
                 Crypt encrypt) {
    WriterHandler handler(cache_mode, requirements, std::move(serialize), std::move(encrypt));
    auto writer = wire::DeferredWriter::open(destination);
    OPENVINO_ASSERT(writer, "[GPU] HSM export requires a seekable output stream");
    writer->add_sections({&handler});
    const auto error = writer->finalize();
    OPENVINO_ASSERT(!error, "[GPU] HSM export failed: ", error.message());
}

}  // namespace ov::intel_gpu::hsm
