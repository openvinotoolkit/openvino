// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/graph_serializer/hsm_writer_handler.hpp"

#include <cstddef>
#include <functional>
#include <ios>
#include <memory>
#include <optional>
#include <ostream>
#include <streambuf>
#include <string>
#include <utility>

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/hsm_writer.hpp"
#include "openvino/util/memory.hpp"
#include "utils/graph_serializer/hsm_common.hpp"
#include "utils/graph_serializer/serializer.hpp"

namespace ov::intel_cpu {

namespace {
// Page-aligned start and slot size: each section covers whole pages
constexpr runtime::hsm::SectionAlignment section_alignment{ov::util::min_page_alignment, 0};

// Forwards every written chunk to the sink without storing it; tellp() is the only position query it supports.
class SinkStreamBuf final : public std::streambuf {
public:
    explicit SinkStreamBuf(runtime::hsm::SectionSink sink) : m_sink(std::move(sink)) {}

protected:
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        m_sink({reinterpret_cast<const std::byte*>(s), static_cast<size_t>(n)});
        m_pos += n;
        return n;
    }

    int_type overflow(int_type ch) override {
        if (!traits_type::eq_int_type(ch, traits_type::eof())) {
            const auto c = traits_type::to_char_type(ch);
            xsputn(&c, 1);
        }
        return traits_type::not_eof(ch);
    }

    pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode /*mode*/) override {
        return off == 0 && dir == std::ios_base::cur ? pos_type(m_pos) : pos_type(static_cast<off_type>(-1));
    }

private:
    runtime::hsm::SectionSink m_sink;
    off_type m_pos = 0;
};

// One serialization pass feeds both sections: write_weights() streams the constants out as they are produced and
// keeps the (small) XML for write_model(), so the encoders must run in that order.
class HsmModelSerialization {
public:
    HsmModelSerialization(std::shared_ptr<ov::Model> model,
                          ModelSerializer::CacheEncrypt encrypt_fn,
                          bool weightless_mode)
        : m_model(std::move(model)),
          m_encrypt_fn(std::move(encrypt_fn)),
          m_weightless_mode(weightless_mode) {}

    void write_weights(const runtime::hsm::SectionSink& sink) {
        SinkStreamBuf weights_buf{sink};
        std::ostream weights(&weights_buf);
        std::string xml_text;
        SinkStreamBuf xml_buf{[&xml_text](ov::util::MemoryView chunk) {
            xml_text.append(reinterpret_cast<const char*>(chunk.data()), chunk.size());
        }};
        std::ostream xml(&xml_buf);
        serialize_model(m_model, weights, xml, m_encrypt_fn, m_weightless_mode);
        OPENVINO_ASSERT(weights.good() && xml.good(), "[CPU] Cannot serialize the model into the HSM blob.");
        m_xml = std::move(xml_text);
    }

    void write_model(const runtime::hsm::SectionSink& sink) {
        OPENVINO_ASSERT(m_xml, "[CPU] The weights section must be encoded before the model section.");
        sink({reinterpret_cast<const std::byte*>(m_xml->data()), m_xml->size()});
        m_xml.reset();
    }

private:
    std::shared_ptr<ov::Model> m_model;
    ModelSerializer::CacheEncrypt m_encrypt_fn;
    bool m_weightless_mode;
    std::optional<std::string> m_xml;
};

class FunctionEncoder final : public runtime::hsm::ISectionEncoder {
public:
    explicit FunctionEncoder(std::function<void(const runtime::hsm::SectionSink&)> encode)
        : m_encode(std::move(encode)) {}

    void encode(const runtime::hsm::SectionSink& sink) const override {
        m_encode(sink);
    }

private:
    std::function<void(const runtime::hsm::SectionSink&)> m_encode;
};
}  // namespace

HsmWriterHandler::HsmWriterHandler(std::shared_ptr<ov::Model> model,
                                   std::string runtime_requirements,
                                   ModelSerializer::CacheEncrypt encrypt_fn,
                                   bool weightless_mode)
    : m_runtime_requirements(std::move(runtime_requirements)) {
    const auto serialization =
        std::make_shared<HsmModelSerialization>(std::move(model), std::move(encrypt_fn), weightless_mode);
    m_weights_encoder = std::make_shared<FunctionEncoder>([serialization](const runtime::hsm::SectionSink& sink) {
        serialization->write_weights(sink);
    });
    m_model_encoder = std::make_shared<FunctionEncoder>([serialization](const runtime::hsm::SectionSink& sink) {
        serialization->write_model(sink);
    });
}

void HsmWriterHandler::handle_section(runtime::hsm::IWriter& writer) const {
    const auto add = [&writer](const hsm::SectionId& id, auto&&... args) {
        OPENVINO_ASSERT(writer.add_section(id.device, id.tag, std::forward<decltype(args)>(args)...),
                        "[CPU] Cannot add HSM section ",
                        id.tag.id());
    };

    // Requirements come first so that an import can reject an incompatible blob before reading any payload.
    // CVS-196454: encode with the common Runtime Requirements codec.
    add(hsm::runtime_requirements,
        ov::util::MemoryView{reinterpret_cast<const std::byte*>(m_runtime_requirements.data()),
                             m_runtime_requirements.size()});
    if (m_model_id) {
        add(hsm::model_id, ov::util::as_memory_view(*m_model_id));
    }
    // The weights are encoded before the model, see HsmModelSerialization.
    add(hsm::weights, m_weights_encoder, section_alignment);
    add(hsm::model_struct, m_model_encoder, section_alignment);
}

}  // namespace ov::intel_cpu
