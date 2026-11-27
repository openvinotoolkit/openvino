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
#include "openvino/runtime/hsm_compile_options_codec.hpp"
#include "openvino/runtime/shared_buffer.hpp"

namespace ov::intel_gpu::hsm {
namespace {

// CompiledModel reads the requirements descriptor ahead of the payload; serve the one already checked.
class ModelInputBuffer final : public cldnn::BinaryInputBuffer {
public:
    ModelInputBuffer(std::istream& stream, cldnn::engine& engine, const size_t* tensor_base, bool encrypted, size_t model_size, const std::string& requirements)
        : BinaryInputBuffer(stream, engine, tensor_base, encrypted),
          m_model_size(model_size) {
        cldnn::BinaryOutputBuffer header(m_header);
        header << CompiledModel::runtime_requirements_magic << CompiledModel::runtime_requirements_version << requirements;
        m_header_remaining = header.get_offset();
    }

    void read(void* data, std::streamsize size) override {
        OPENVINO_ASSERT(size >= 0, "[GPU] Invalid HSM model read size");
        // Bypasses BinaryInputBuffer::read so payload offsets stay relative to the model section.
        auto* destination = static_cast<char*>(data);
        if (m_header_remaining != 0) {
            const auto read = m_header.rdbuf()->sgetn(destination, size);
            m_header_remaining -= static_cast<size_t>(read);
            if (m_header_remaining == 0) {
                set_read_limit(m_model_size);
            }
            destination += read;
            size -= read;
        }
        if (size != 0) {
            check_size(static_cast<size_t>(size));
            BinaryInputBuffer::read(destination, size);
        }
    }

    void seek_current_ptr(std::streamsize size) override {
        OPENVINO_ASSERT(size >= 0, "[GPU] Invalid HSM model seek size");
        check_size(static_cast<size_t>(size));
        BinaryInputBuffer::seek_current_ptr(size);
    }

private:
    std::stringstream m_header;
    size_t m_header_remaining = 0;
    const size_t m_model_size;
};

bool is_section(const runtime::hsm::ManifestEntry& entry, const SectionId& id) {
    return entry.device == id.device && entry.tag.id() == id.tag.id();
}

void check_section(bool seen, const SectionId& id, const runtime::hsm::ManifestEntry& entry) {
    OPENVINO_ASSERT(entry.tag.value == id.tag.value, "[GPU] Invalid HSM section storage mode");
    OPENVINO_ASSERT(!seen, "[GPU] Duplicate HSM section");
}

}  // namespace

ReaderHandler::ReaderHandler(const runtime::hsm::Reader& reader, cldnn::engine& engine) : m_engine(engine) {
    reader.read_sections({this});
    OPENVINO_ASSERT(m_info && m_requirements && m_cache_mode && m_model, "[GPU] Missing mandatory HSM section");
    const auto& device_info = engine.get_device_info();
    OPENVINO_ASSERT(CompiledModel::is_runtime_requirements_compatible(*m_requirements, device_info),
                    "[GPU] Cannot import HSM blob: incompatible runtime configuration.\n  blob: ",
                    *m_requirements,
                    "\n  current: ",
                    CompiledModel::build_runtime_requirements(device_info));
}

ReaderHandler::~ReaderHandler() = default;

bool ReaderHandler::handle_section(const runtime::hsm::ManifestEntry& entry, const runtime::hsm::Section& section) {
    if (is_section(entry, payload_info)) {
        check_section(m_info.has_value(), payload_info, entry);
        auto& info = m_info.emplace();
        OPENVINO_ASSERT(section.size() == sizeof(info) && section.read(&info), "[GPU] Invalid HSM payload info");
        OPENVINO_ASSERT(info.version == PayloadInfo::current_version && (info.flags & ~encrypted_flag) == 0 &&
                            info.requirements_version == CompiledModel::runtime_requirements_version && info.reserved32 == 0 && info.reserved == 0,
                        "[GPU] Unsupported HSM payload info");
    } else if (is_section(entry, runtime_requirements)) {
        check_section(m_requirements.has_value(), runtime_requirements, entry);
        OPENVINO_ASSERT(section.size() > 0 && section.size() <= CompiledModel::runtime_requirements_max_size, "[GPU] Invalid HSM runtime requirements size");
        auto& requirements = m_requirements.emplace(section.size(), '\0');
        OPENVINO_ASSERT(section.read(requirements.data()), "[GPU] Cannot read HSM runtime requirements");
    } else if (is_section(entry, compile_options)) {
        check_section(m_cache_mode.has_value(), compile_options, entry);
        const auto options = runtime::hsm::decode_compile_options(section);
        OPENVINO_ASSERT(options, "[GPU] Invalid HSM compile options");
        const auto mode = options->find(ov::cache_mode.name());
        OPENVINO_ASSERT(mode != options->end() && mode->second.is<std::string>(), "[GPU] HSM compile options must include a cache mode");
        const auto& value = mode->second.as<std::string>();
        OPENVINO_ASSERT(value == "optimize_speed" || value == "OPTIMIZE_SPEED" || value == "optimize_size" || value == "OPTIMIZE_SIZE",
                        "[GPU] Invalid HSM cache mode");
        m_cache_mode = mode->second.as<ov::CacheMode>();
    } else if (is_section(entry, model)) {
        check_section(m_model.has_value(), model, entry);
        OPENVINO_ASSERT(section.size() > 0, "[GPU] Empty HSM model section");
        m_model = section;
        m_model_offset = entry.offset;
    } else {
        return false;
    }
    return true;
}

ov::CacheMode ReaderHandler::cache_mode() const {
    return *m_cache_mode;
}

bool ReaderHandler::open_encrypted_model(const ov::EncryptionCallbacks& callbacks) {
    if ((m_info->flags & encrypted_flag) == 0) {
        OPENVINO_ASSERT(!callbacks.encrypt, "[GPU] HSM payload is not encrypted");
        return false;
    }
    OPENVINO_ASSERT(callbacks.decrypt, "[GPU] Encrypted HSM payload requires a decryption callback");
    {
        // The callback works on whole strings, so ciphertext and plaintext are both held while it runs.
        std::string ciphertext(m_model->size(), '\0');
        OPENVINO_ASSERT(m_model->read(ciphertext.data()), "[GPU] Cannot read encrypted HSM payload");
        m_plaintext = callbacks.decrypt(ciphertext);
    }
    m_buffer = std::make_unique<ov::SharedStreamBuffer>(m_plaintext.data(), m_plaintext.size());
    m_stream = std::make_unique<std::istream>(m_buffer.get());
    m_input = std::make_unique<ModelInputBuffer>(*m_stream, m_engine, nullptr, true, m_plaintext.size(), *m_requirements);
    return true;
}

cldnn::BinaryInputBuffer& ReaderHandler::open_model(std::istream& source,
                                                    std::streampos container_start,
                                                    const ov::EncryptionCallbacks& callbacks) {
    if (!open_encrypted_model(callbacks)) {
        // Read the source directly so that the prefetch in program::load() still reaches its streambuf.
        source.clear();
        source.seekg(container_start + static_cast<std::streamoff>(m_model_offset));
        OPENVINO_ASSERT(source.good(), "[GPU] Cannot seek to the HSM model section");
        m_input = std::make_unique<ModelInputBuffer>(source, m_engine, nullptr, false, m_model->size(), *m_requirements);
    }
    return *m_input;
}

cldnn::BinaryInputBuffer& ReaderHandler::open_model(const ov::Tensor& blob, const ov::EncryptionCallbacks& callbacks) {
    if (open_encrypted_model(callbacks)) {
        return *m_input;
    }
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
    const auto address = reinterpret_cast<std::uintptr_t>(view.data());
    const auto fits_host_import = [&](size_t alignment) {
        return alignment != 0 && address % alignment == 0 && view.size() % alignment == 0;
    };
    const auto cacheline = m_engine.get_device_info().cacheline_size.value_or(0);
    // The writer's slot padding is outside the model view. Do not register it as model memory.
    const bool can_alias = fits_host_import(cacheline) &&
                           (m_engine.runtime_type() != cldnn::runtime_types::ze ||
                            fits_host_import(static_cast<size_t>(ov::util::get_system_page_size())));
    const auto* tensor_base = can_alias ? static_cast<const size_t*>(std::as_const(m_model_tensor).data()) : nullptr;
    m_input = std::make_unique<ModelInputBuffer>(*m_stream, m_engine, tensor_base, false, view.size(), *m_requirements);
    return *m_input;
}

}  // namespace ov::intel_gpu::hsm
