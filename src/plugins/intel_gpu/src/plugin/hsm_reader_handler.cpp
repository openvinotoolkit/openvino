// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/hsm_reader_handler.hpp"

#include <sstream>
#include <utility>

#include "common_utils/prefetch_streambuf.hpp"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"
#include "intel_gpu/plugin/compiled_model.hpp"
#include "openvino/runtime/shared_buffer.hpp"

namespace ov::intel_gpu::hsm {
namespace {

// CompiledModel reads the requirements descriptor ahead of the payload; serve the one already checked.
class ModelInputBuffer final : public cldnn::BinaryInputBuffer {
public:
    ModelInputBuffer(std::istream& stream,
                     cldnn::engine& engine,
                     const size_t* tensor_base,
                     bool encrypted,
                     const std::string& requirements)
        : BinaryInputBuffer(stream, engine, tensor_base, encrypted) {
        cldnn::BinaryOutputBuffer header(m_header);
        header << CompiledModel::runtime_requirements_magic << CompiledModel::runtime_requirements_version << requirements;
    }

    void read(void* data, std::streamsize size) override {
        // Bypasses BinaryInputBuffer::read so payload offsets stay relative to the model section.
        const auto read = m_header.rdbuf()->sgetn(static_cast<char*>(data), size);
        if (read < size)
            BinaryInputBuffer::read(static_cast<char*>(data) + read, size - read);
    }

private:
    std::stringstream m_header;
};

bool is_section(const runtime::hsm::ManifestEntry& entry, const SectionId& id) {
    return entry.device == id.device && entry.tag.id() == id.tag.id();
}

void accept(std::optional<runtime::hsm::Section>& target,
            const SectionId& id,
            const runtime::hsm::ManifestEntry& entry,
            const runtime::hsm::Section& section) {
    OPENVINO_ASSERT(entry.tag.value == id.tag.value, "[GPU] Invalid HSM section storage mode");
    OPENVINO_ASSERT(!target, "[GPU] Duplicate HSM section");
    target = section;
}

}  // namespace

ReaderHandler::ReaderHandler(const runtime::hsm::Reader& reader, cldnn::engine& engine) : m_engine(engine) {
    reader.read_sections({this});
    OPENVINO_ASSERT(m_info_section && m_requirements_section && m_model, "[GPU] Missing mandatory HSM section");
    OPENVINO_ASSERT(m_info_section->size() == sizeof(PayloadInfo) && m_info_section->read(&m_info),
                    "[GPU] Invalid HSM payload info");
    OPENVINO_ASSERT(m_info.version == 1 && m_info.cache_mode <= 1 && (m_info.flags & ~encrypted_flag) == 0 &&
                        m_info.requirements_version == CompiledModel::runtime_requirements_version && m_info.reserved == 0,
                    "[GPU] Unsupported HSM payload info");
    OPENVINO_ASSERT(m_requirements_section->size() <= CompiledModel::runtime_requirements_max_size,
                    "[GPU] HSM runtime requirements are too large");
    m_requirements.resize(m_requirements_section->size());
    OPENVINO_ASSERT(m_requirements_section->read(m_requirements.data()), "[GPU] Cannot read HSM runtime requirements");
    const auto& device_info = engine.get_device_info();
    OPENVINO_ASSERT(CompiledModel::is_runtime_requirements_compatible(m_requirements, device_info),
                    "[GPU] Cannot import HSM blob: incompatible runtime configuration.\n  blob: ",
                    m_requirements,
                    "\n  current: ",
                    CompiledModel::build_runtime_requirements(device_info));
}

ReaderHandler::~ReaderHandler() = default;

bool ReaderHandler::handle_section(const runtime::hsm::ManifestEntry& entry, const runtime::hsm::Section& section) {
    if (is_section(entry, payload_info)) {
        accept(m_info_section, payload_info, entry, section);
    } else if (is_section(entry, runtime_requirements)) {
        accept(m_requirements_section, runtime_requirements, entry, section);
    } else if (is_section(entry, model)) {
        accept(m_model, model, entry, section);
        m_model_entry = entry;
    } else {
        return false;
    }
    return true;
}

ov::CacheMode ReaderHandler::cache_mode() const {
    return m_info.cache_mode == 0 ? ov::CacheMode::OPTIMIZE_SPEED : ov::CacheMode::OPTIMIZE_SIZE;
}

bool ReaderHandler::open_encrypted_model(const ov::EncryptionCallbacks& callbacks) {
    if ((m_info.flags & encrypted_flag) == 0)
        return false;
    OPENVINO_ASSERT(callbacks.decrypt, "[GPU] Encrypted HSM payload requires a decryption callback");
    size_t size = 0;
    OPENVINO_ASSERT(m_model->read(0, sizeof(size), &size) && size == m_model->size() - sizeof(size),
                    "[GPU] Invalid encrypted HSM payload size");
    {
        // The callback works on whole strings, so ciphertext and plaintext are both held while it runs.
        std::string ciphertext(size, '\0');
        OPENVINO_ASSERT(m_model->read(sizeof(size), size, ciphertext.data()), "[GPU] Cannot read encrypted HSM payload");
        m_plaintext = callbacks.decrypt(ciphertext);
    }
    m_buffer = std::make_unique<ov::SharedStreamBuffer>(m_plaintext.data(), m_plaintext.size());
    m_stream = std::make_unique<std::istream>(m_buffer.get());
    m_input = std::make_unique<ModelInputBuffer>(*m_stream, m_engine, nullptr, true, m_requirements);
    return true;
}

cldnn::BinaryInputBuffer& ReaderHandler::open_model(std::istream& source,
                                                    std::streampos container_start,
                                                    const ov::EncryptionCallbacks& callbacks) {
    if (!open_encrypted_model(callbacks)) {
        // Read the source directly so that the prefetch in program::load() still reaches its streambuf.
        source.clear();
        source.seekg(container_start + static_cast<std::streamoff>(m_model_entry.offset));
        OPENVINO_ASSERT(source.good(), "[GPU] Cannot seek to the HSM model section");
        m_input = std::make_unique<ModelInputBuffer>(source, m_engine, nullptr, false, m_requirements);
    }
    return *m_input;
}

cldnn::BinaryInputBuffer& ReaderHandler::open_model(const ov::Tensor& blob, const ov::EncryptionCallbacks& callbacks) {
    if (open_encrypted_model(callbacks))
        return *m_input;
    const auto view = m_model->view();
    const auto offset = static_cast<size_t>(view.data() - static_cast<const std::byte*>(blob.data()));
    // A byte ROI keeps the memory hints of a mapped blob, which drive the prefetch.
    m_model_tensor = blob.get_element_type() == ov::element::u8 && blob.get_shape().size() == 1
                         ? ov::Tensor(blob, ov::Coordinate{offset}, ov::Coordinate{offset + view.size()})
                         : ov::Tensor(ov::element::u8, ov::Shape{view.size()}, const_cast<std::byte*>(view.data()));
#ifdef _WIN32
    // TODO: CVS-198510 enable on Windows once prefetch shows a compile-time gain there.
    constexpr bool prefetch = false;
#else
    constexpr bool prefetch = true;
#endif
    m_buffer = std::make_unique<PrefetchStreamBuf>(m_model_tensor, prefetch);
    m_stream = std::make_unique<std::istream>(m_buffer.get());
    const auto* tensor_base = static_cast<const size_t*>(std::as_const(m_model_tensor).data());
    m_input = std::make_unique<ModelInputBuffer>(*m_stream, m_engine, tensor_base, false, m_requirements);
    return *m_input;
}

}  // namespace ov::intel_gpu::hsm
