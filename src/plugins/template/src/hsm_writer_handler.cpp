// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "hsm_writer_handler.hpp"

#include <functional>
#include <optional>
#include <ostream>
#include <streambuf>
#include <string>
#include <string_view>
#include <type_traits>

#include "openvino/core/except.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/compilation_context.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/hash_util.hpp"

namespace ov::template_plugin {

namespace {
// Page-aligned start keeps the large sections mmap/DMA friendly; no slot padding, nothing follows that needs it.
constexpr runtime::hsm::SectionAlignment page_alignment{util::min_page_alignment, 0};

// Forwards written chunks to a sink (may be empty) and tracks the position for tellp(); never stores data.
class ForwardingStreamBuf final : public std::streambuf {
public:
    using Callback = runtime::hsm::SectionSink;

    explicit ForwardingStreamBuf(Callback callback = {}) : m_callback(std::move(callback)), m_pos(0) {}

    size_t size() const noexcept {
        return static_cast<size_t>(m_pos);
    }

protected:
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        put(s, static_cast<size_t>(n));
        return n;
    }

    int_type overflow(int_type ch) override {
        if (!traits_type::eq_int_type(ch, traits_type::eof())) {
            const auto c = traits_type::to_char_type(ch);
            put(&c, 1);
        }
        return traits_type::not_eof(ch);
    }

    pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode) override {
        return off == 0 && dir == std::ios_base::cur ? pos_type(m_pos) : pos_type(off_type(-1));
    }

private:
    void put(const char* data, size_t size) {
        if (m_callback) {
            m_callback({reinterpret_cast<const std::byte*>(data), size});
        }
        m_pos += static_cast<off_type>(size);
    }

    Callback m_callback;
    off_type m_pos;
};

struct ModelStats {
    size_t weights_size = 0;
    uint64_t hash = 0;  // of the serialized structure and the weights size, not the weight values
};

// Serializes into counting/hashing buffers only, so nothing of the model is kept in memory.
ModelStats measure_model(const std::shared_ptr<ov::Model>& model) {
    uint64_t hash = 0;
    ForwardingStreamBuf xml_buf{[&hash](ov::util::MemoryView chunk) {
        const std::string_view text{reinterpret_cast<const char*>(chunk.data()), chunk.size()};
        hash = util::u64_hash_combine(hash, std::hash<std::string_view>{}(text));
    }};
    ForwardingStreamBuf bin_buf;
    std::ostream xml(&xml_buf), bin(&bin_buf);
    ov::pass::Serialize(xml, bin).run_on_model(model);
    return {bin_buf.size(), util::u64_hash_combine(hash, bin_buf.size())};
}

// Serializes the model in one pass: write_weights() streams the weights as produced and keeps the small XML for
// write_structure(), so calling them in that order avoids a second pass.
class ModelSerializer {
public:
    explicit ModelSerializer(std::shared_ptr<ov::Model> model) : m_model(std::move(model)) {}

    void write_weights(const runtime::hsm::SectionSink& sink) {
        auto& xml = m_xml.emplace();
        ForwardingStreamBuf xml_buf{[&xml](ov::util::MemoryView chunk) {
            xml.append(reinterpret_cast<const char*>(chunk.data()), chunk.size());
        }};
        ForwardingStreamBuf bin_buf{sink};
        std::ostream xml_out(&xml_buf), bin_out(&bin_buf);
        ov::pass::Serialize(xml_out, bin_out).run_on_model(m_model);
    }

    // Runs its own pass when write_weights() was not called before.
    void write_structure(const runtime::hsm::SectionSink& sink) {
        if (m_xml) {
            sink({reinterpret_cast<const std::byte*>(m_xml->data()), m_xml->size()});
            m_xml.reset();
        } else {
            ForwardingStreamBuf xml_buf{sink}, discard_buf;
            std::ostream xml_out(&xml_buf), bin_out(&discard_buf);
            ov::pass::Serialize(xml_out, bin_out).run_on_model(m_model);
        }
    }

private:
    std::shared_ptr<ov::Model> m_model;
    std::optional<std::string> m_xml;
};

class SerializeEncoder final : public runtime::hsm::ISectionEncoder {
public:
    using Write = void (ModelSerializer::*)(const runtime::hsm::SectionSink&);

    SerializeEncoder(std::shared_ptr<ModelSerializer> serializer, Write write)
        : m_serializer(std::move(serializer)),
          m_write(write) {}

    void encode(const runtime::hsm::SectionSink& sink) const override {
        ((*m_serializer).*m_write)(sink);
    }

private:
    std::shared_ptr<ModelSerializer> m_serializer;
    Write m_write;
};

// Cache-relevant options plus the weights path, which a weightless blob needs on import; restored through
// Configuration.
ov::AnyMap get_compiled_options(const Configuration& cfg, ov::AnyMap cache_options) {
    if (!cfg.weights_path.empty()) {
        cache_options.emplace(ov::weights_path.name(), cfg.weights_path.string());
    }
    return cache_options;
}

// One `key=value` per line; inverse of decode_compiled_options() in hsm_reader_handler.cpp.
std::string encode_compiled_options(const ov::AnyMap& options) {
    std::string encoded;
    for (const auto& [key, value] : options) {
        encoded.append(key).append(1, '=').append(value.as<std::string>()).append(1, '\n');
    }
    return encoded;
}
}  // namespace

/// Owns a small section's bytes (metadata only); large payloads must be streamed to avoid a second in-memory copy.
class HsmWriterHandler::BytesEncoder final : public runtime::hsm::ISectionEncoder {
    static constexpr auto max_size = util::min_page_alignment;  // one page
public:
    explicit BytesEncoder(std::string bytes) : m_bytes(std::move(bytes)) {
        OPENVINO_ASSERT(m_bytes.size() <= max_size, "TEMPLATE: HSM section too large to buffer: ", m_bytes.size());
    }

    template <class T>
    static std::shared_ptr<BytesEncoder> from_value(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        return std::make_shared<BytesEncoder>(std::string(reinterpret_cast<const char*>(&value), sizeof(T)));
    }

    size_t size() const noexcept {
        return m_bytes.size();
    }

    ov::util::MemoryView view() const noexcept {
        return {reinterpret_cast<const std::byte*>(m_bytes.data()), m_bytes.size()};
    }

    void encode(const runtime::hsm::SectionSink& sink) const override {
        sink(view());
    }

private:
    std::string m_bytes;
};

HsmWriterHandler::HsmWriterHandler(const std::shared_ptr<ov::Model>& model,
                                   const Configuration& cfg,
                                   const ov::AnyMap& cache_options,
                                   std::string_view runtime_requirements) {
    const auto stats = measure_model(model);
    const auto is_weights_embedded = stats.weights_size > 0 && cfg.cache_mode == ov::CacheMode::OPTIMIZE_SPEED;
    hsm::BlobInfo info;
    info.flags = (is_weights_embedded ? hsm::BlobInfo::weights_embedded : 0u) |
                 (stats.weights_size == 0 ? hsm::BlobInfo::weightless : 0u);
    const auto model_hash = std::stoull(ov::ModelCache::compute_hash(std::to_string(stats.hash), {}, cache_options));

    m_encoders.emplace(hsm::section_key(hsm::model_id),
                       Slot{hsm::model_id, BytesEncoder::from_value(static_cast<uint64_t>(model_hash))});
    m_encoders.emplace(
        hsm::section_key(hsm::runtime_requirements),
        Slot{hsm::runtime_requirements, std::make_shared<BytesEncoder>(std::string(runtime_requirements))});
    m_encoders.emplace(
        hsm::section_key(hsm::compiled_options),
        Slot{hsm::compiled_options,
             std::make_shared<BytesEncoder>(encode_compiled_options(get_compiled_options(cfg, cache_options)))});

    const auto serializer = std::make_shared<ModelSerializer>(model);
    m_encoders.emplace(hsm::section_key(hsm::model_struct),
                       Slot{hsm::model_struct,
                            std::make_shared<SerializeEncoder>(serializer, &ModelSerializer::write_structure),
                            std::nullopt,
                            page_alignment});

    if (is_weights_embedded) {
        m_encoders.emplace(hsm::section_key(hsm::weights),
                           Slot{hsm::weights,
                                std::make_shared<SerializeEncoder>(serializer, &ModelSerializer::write_weights),
                                stats.weights_size,
                                page_alignment});
    }
    m_encoders.emplace(hsm::section_key(hsm::blob_info), Slot{hsm::blob_info, BytesEncoder::from_value(info)});
}

HsmWriterHandler::Slot::Slot(const hsm::SectionId& id,
                             const std::shared_ptr<BytesEncoder>& bytes,
                             runtime::hsm::SectionAlignment align)
    : m_id(id),
      m_align(align),
      m_encoder(bytes),
      m_size(bytes->size()),
      m_view(bytes->view()) {}

HsmWriterHandler::Slot::Slot(const hsm::SectionId& id,
                             runtime::hsm::SectionEncoderPtr encoder,
                             std::optional<size_t> size,
                             runtime::hsm::SectionAlignment align)
    : m_id(id),
      m_align(align),
      m_encoder(std::move(encoder)),
      m_size(size),
      m_view{} {}

bool HsmWriterHandler::Slot::add_to(runtime::hsm::IWriter& writer) const {
    if (m_id.tag.is_inline()) {
        return writer.add_section(m_id.device, m_id.tag, m_view);
    } else if (m_size) {
        return writer.add_section(m_id.device, m_id.tag, *m_size, m_encoder, m_align);
    } else {
        return writer.add_section(m_id.device, m_id.tag, m_encoder, m_align);
    }
}

// ! [hsm_writer_handler:handle_section]
void HsmWriterHandler::handle_section(runtime::hsm::IWriter& writer) const {
    const auto add_slot = [&writer](uint32_t key, const Slot& slot) {
        OPENVINO_ASSERT(slot.add_to(writer), "TEMPLATE: cannot add HSM section ", key);
    };

    // Weights first: the serialization pass streams them before the structure section is written.
    const auto weights_key = hsm::section_key(hsm::weights);
    if (const auto weights = m_encoders.find(weights_key); weights != m_encoders.end()) {
        add_slot(weights->first, weights->second);
    }
    for (const auto& [key, slot] : m_encoders) {
        if (key != weights_key) {
            add_slot(key, slot);
        }
    }
}
// ! [hsm_writer_handler:handle_section]

}  // namespace ov::template_plugin
