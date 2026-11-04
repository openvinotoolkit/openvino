// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "hsm_writer_handler.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <streambuf>
#include <utility>

#include "hsm_common.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/util/memory.hpp"

namespace ov::hetero {

namespace {
// Page-aligned start and slot size: a section covers whole pages, so a mapped blob can hand it out as is.
constexpr runtime::hsm::SectionAlignment hsm_page_alignment{ov::util::min_page_alignment};

// Seekable view of the section being written: appends go through the writer's sink, so it keeps its own size
// accounting, while a write behind the end (a producer patching a header) overwrites the bytes already in the
// destination. Seeks are limited to [0, bytes appended].
class HsmSectionWindowBuf final : public std::streambuf {
public:
    HsmSectionWindowBuf(std::ostream& destination, const runtime::hsm::SectionSink& sink)
        : m_destination(destination),
          m_sink(sink),
          m_start(destination.tellp()) {}

    size_t written() const {
        return m_high;
    }

protected:
    std::streamsize xsputn(const char* data, std::streamsize count) override {
        if (count <= 0) {
            return 0;
        }
        auto left = static_cast<size_t>(count);
        if (m_pos < m_high) {
            const auto patch = std::min(left, m_high - m_pos);
            m_destination.seekp(m_start + static_cast<std::streamoff>(m_pos));
            m_destination.write(data, static_cast<std::streamsize>(patch));
            m_destination.seekp(m_start + static_cast<std::streamoff>(m_high));
            if (!m_destination) {
                return 0;
            }
            data += patch;
            left -= patch;
            m_pos += patch;
        }
        if (left > 0) {
            m_sink({reinterpret_cast<const std::byte*>(data), left});
            m_pos += left;
            m_high += left;
            if (!m_destination) {
                return 0;
            }
        }
        return count;
    }

    int_type overflow(int_type ch) override {
        if (traits_type::eq_int_type(ch, traits_type::eof())) {
            return traits_type::not_eof(ch);
        }
        const char c = traits_type::to_char_type(ch);
        return xsputn(&c, 1) == 1 ? ch : traits_type::eof();
    }

    pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which) override {
        if ((which & std::ios_base::out) == 0) {
            return pos_type(off_type(-1));
        }
        off_type base = 0;
        if (dir == std::ios_base::cur) {
            base = static_cast<off_type>(m_pos);
        } else if (dir == std::ios_base::end) {
            base = static_cast<off_type>(m_high);
        } else if (dir != std::ios_base::beg) {
            return pos_type(off_type(-1));
        }
        const auto target = base + off;
        if (target < 0 || static_cast<size_t>(target) > m_high) {
            return pos_type(off_type(-1));
        }
        m_pos = static_cast<size_t>(target);
        return pos_type(target);
    }

    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override {
        return seekoff(static_cast<off_type>(pos), std::ios_base::beg, which);
    }

private:
    std::ostream& m_destination;
    const runtime::hsm::SectionSink& m_sink;
    std::streampos m_start;
    size_t m_pos = 0;
    size_t m_high = 0;
};

// One serialization pass feeds both IR sections: the weights go to the sink as they are produced, only the small
// XML is kept for the section that follows.
struct HsmIrSerialization {
    std::shared_ptr<ov::Model> model;
    std::string xml;
};

class HsmIrWeightsEncoder final : public runtime::hsm::ISectionEncoder {
public:
    HsmIrWeightsEncoder(std::ostream& destination, std::shared_ptr<HsmIrSerialization> state)
        : m_destination(destination),
          m_state(std::move(state)) {}

    void encode(const runtime::hsm::SectionSink& sink) const override {
        OPENVINO_ASSERT(m_state->model, "OpenVINO Model is empty");
        HsmSectionWindowBuf weights_buf{m_destination, sink};
        std::ostream weights_stream{&weights_buf};
        std::ostringstream xml_stream;
        ov::pass::Serialize{xml_stream, weights_stream}.run_on_model(m_state->model);
        OPENVINO_ASSERT(weights_stream, "[HETERO] Failed to serialize the IR weights into the HSM blob");
        m_state->xml = xml_stream.str();
    }

private:
    std::ostream& m_destination;
    std::shared_ptr<HsmIrSerialization> m_state;
};

class HsmIrXmlEncoder final : public runtime::hsm::ISectionEncoder {
public:
    explicit HsmIrXmlEncoder(std::shared_ptr<HsmIrSerialization> state) : m_state(std::move(state)) {}

    void encode(const runtime::hsm::SectionSink& sink) const override {
        sink(ov::util::as_memory_view(m_state->xml));
        std::string{}.swap(m_state->xml);
    }

private:
    std::shared_ptr<HsmIrSerialization> m_state;
};

class HsmBlobEncoder final : public runtime::hsm::ISectionEncoder {
public:
    HsmBlobEncoder(std::ostream& destination, std::function<void(std::ostream&)> export_blob)
        : m_destination(destination),
          m_export_blob(std::move(export_blob)) {}

    void encode(const runtime::hsm::SectionSink& sink) const override {
        HsmSectionWindowBuf window{m_destination, sink};
        std::ostream stream{&window};
        try {
            m_export_blob(stream);
        } catch (const ov::NotImplemented& ex) {
            // Only a producer that has not written anything can still be replaced by the IR sections.
            OPENVINO_ASSERT(window.written() == 0, "[HETERO] Submodel export failed after partial output: ", ex.what());
            throw;
        }
        OPENVINO_ASSERT(stream && m_destination, "[HETERO] Failed to export a compiled submodel into the HSM blob");
    }

private:
    std::ostream& m_destination;
    std::function<void(std::ostream&)> m_export_blob;
};
}  // namespace

std::string build_hsm_runtime_requirements() {
    return "hetero=" + std::to_string(hsm::requirements_format_version);
}

HsmWriterHandler::HsmWriterHandler(std::ostream& destination,
                                   std::string metadata,
                                   std::vector<HsmSubmodelPayload> submodels)
    : m_destination(destination),
      m_runtime_requirements(build_hsm_runtime_requirements()),
      m_metadata(std::move(metadata)),
      m_submodels(std::move(submodels)) {
    OPENVINO_ASSERT(m_submodels.size() <= std::numeric_limits<uint32_t>::max(),
                    "[HETERO] Too many submodels for the HSM blob");
}

void HsmWriterHandler::handle_section(runtime::hsm::IWriter& writer) const {
    const auto add = [&writer](const hsm::SectionId& id,
                               uint32_t index,
                               ov::util::MemoryView payload,
                               runtime::hsm::SectionAlignment alignment = {}) {
        OPENVINO_ASSERT(writer.add_section(id.device,
                                           runtime::hsm::SectionTagReserved{id.tag, hsm::encode_section_index(index)},
                                           payload,
                                           alignment),
                        "[HETERO] Cannot add HSM section ",
                        id.tag.id());
    };

    // Requirements come first so that an import can reject an incompatible blob before reading any payload.
    add(hsm::runtime_requirements, 0, ov::util::as_memory_view(m_runtime_requirements));
    add(hsm::metadata, 0, ov::util::as_memory_view(m_metadata));

    for (size_t i = 0; i < m_submodels.size(); ++i) {
        const auto index = static_cast<uint32_t>(i);
        const auto& submodel = m_submodels[i];
        if (submodel.export_blob) {
            // Unsized: the size is whatever the sub-device writes.
            OPENVINO_ASSERT(writer.add_section(hsm::submodel_blob.device,
                                               runtime::hsm::SectionTagReserved{hsm::submodel_blob.tag,
                                                                                hsm::encode_section_index(index)},
                                               std::make_shared<HsmBlobEncoder>(m_destination, submodel.export_blob),
                                               hsm_page_alignment),
                            "[HETERO] Cannot add HSM section ",
                            hsm::submodel_blob.tag.id());
        } else {
            // The weights are encoded first: the same pass produces the XML kept for the next section.
            const auto state = std::make_shared<HsmIrSerialization>(HsmIrSerialization{submodel.ir_model, {}});
            const auto add_encoded = [&](const hsm::SectionId& id,
                                         std::shared_ptr<runtime::hsm::ISectionEncoder> encoder,
                                         runtime::hsm::SectionAlignment alignment) {
                OPENVINO_ASSERT(
                    writer.add_section(id.device,
                                       runtime::hsm::SectionTagReserved{id.tag, hsm::encode_section_index(index)},
                                       std::move(encoder),
                                       alignment),
                    "[HETERO] Cannot add HSM section ",
                    id.tag.id());
            };
            add_encoded(hsm::submodel_ir_weights,
                        std::make_shared<HsmIrWeightsEncoder>(m_destination, state),
                        hsm_page_alignment);
            add_encoded(hsm::submodel_ir_xml, std::make_shared<HsmIrXmlEncoder>(state), {});
        }
    }
}

}  // namespace ov::hetero
