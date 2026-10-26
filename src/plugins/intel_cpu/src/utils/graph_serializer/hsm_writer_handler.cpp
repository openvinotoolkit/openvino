// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/graph_serializer/hsm_writer_handler.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <ios>
#include <memory>
#include <ostream>
#include <streambuf>
#include <string>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/hsm_writer.hpp"
#include "openvino/util/memory.hpp"
#include "utils/graph_serializer/hsm_common.hpp"
#include "utils/graph_serializer/serializer.hpp"

namespace ov::intel_cpu {

namespace {

// Seekable, because ModelSerializer patches its header at the end; chunked to avoid reallocating a large buffer.
// Not needed once the model and weights are separate sections and can be streamed straight to the writer.
class ChunkedStreamBuf final : public std::streambuf {
public:
    static constexpr size_t chunk_size = 4 * ov::util::one_mib;

    void emit(const runtime::hsm::SectionSink& sink) const {
        for (size_t offset = 0; offset < m_size; offset += chunk_size) {
            sink({reinterpret_cast<const std::byte*>(m_chunks[offset / chunk_size].get()),
                  std::min(chunk_size, m_size - offset)});
        }
    }

protected:
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        auto remaining = static_cast<size_t>(n);
        while (remaining > 0) {
            const auto index = m_pos / chunk_size;
            if (index == m_chunks.size()) {
                m_chunks.push_back(std::make_unique<char[]>(chunk_size));
            }
            const auto offset = m_pos % chunk_size;
            const auto count = std::min(remaining, chunk_size - offset);
            std::memcpy(m_chunks[index].get() + offset, s, count);
            s += count;
            m_pos += count;
            remaining -= count;
        }
        m_size = std::max(m_size, m_pos);
        return n;
    }

    int_type overflow(int_type ch) override {
        if (!traits_type::eq_int_type(ch, traits_type::eof())) {
            const auto c = traits_type::to_char_type(ch);
            xsputn(&c, 1);
        }
        return traits_type::not_eof(ch);
    }

    pos_type seekoff(off_type off,
                     std::ios_base::seekdir dir,
                     [[maybe_unused]] std::ios_base::openmode /** open mode*/) override {
        off_type base = 0;
        switch (dir) {
        case std::ios_base::cur:
            base = static_cast<off_type>(m_pos);
            break;
        case std::ios_base::end:
            base = static_cast<off_type>(m_size);
            break;
        default:
            break;
        }
        const auto target = base + off;
        if (target < 0 || static_cast<size_t>(target) > m_size) {
            return {off_type{-1}};
        }
        m_pos = static_cast<size_t>(target);
        return {target};
    }

    pos_type seekpos(pos_type pos, std::ios_base::openmode mode) override {
        return seekoff(static_cast<off_type>(pos), std::ios_base::beg, mode);
    }

private:
    std::vector<std::unique_ptr<char[]>> m_chunks;
    size_t m_size = 0;
    size_t m_pos = 0;
};

// Runs the unchanged legacy serialization and hands the resulting stream to the writer.
class LegacyStreamEncoder final : public runtime::hsm::ISectionEncoder {
public:
    LegacyStreamEncoder(std::shared_ptr<ov::Model> model,
                        ModelSerializer::CacheEncrypt encrypt_fn,
                        bool weightless_mode)
        : m_model(std::move(model)),
          m_encrypt_fn(std::move(encrypt_fn)),
          m_weightless_mode(weightless_mode) {}

    void encode(const runtime::hsm::SectionSink& sink) const override {
        ChunkedStreamBuf buf;
        std::ostream stream(&buf);
        ModelSerializer serializer(stream, m_encrypt_fn, m_weightless_mode);
        serializer << m_model;
        OPENVINO_ASSERT(stream.good(), "[CPU] Cannot serialize the model into the HSM blob.");
        buf.emit(sink);
    }

private:
    std::shared_ptr<ov::Model> m_model;
    ModelSerializer::CacheEncrypt m_encrypt_fn;
    bool m_weightless_mode;
};

// Last section: its page-aligned end puts the manifest on a page boundary.
constexpr runtime::hsm::SectionAlignment model_alignment{ov::util::min_page_alignment, ov::util::min_page_alignment};

}  // namespace

HsmWriterHandler::HsmWriterHandler(std::shared_ptr<ov::Model> model,
                                   std::string runtime_requirements,
                                   ModelSerializer::CacheEncrypt encrypt_fn,
                                   bool weightless_mode)
    : m_runtime_requirements(std::move(runtime_requirements)),
      m_model_encoder(std::make_shared<LegacyStreamEncoder>(std::move(model), std::move(encrypt_fn), weightless_mode)) {
}

void HsmWriterHandler::handle_section(runtime::hsm::IWriter& writer) const {
    const auto add = [&writer](const hsm::SectionId& id, auto&&... args) {
        OPENVINO_ASSERT(writer.add_section(id.device, id.tag, std::forward<decltype(args)>(args)...),
                        "[CPU] Cannot add HSM section ",
                        id.tag.id());
    };

    // TODO CVS-196454: encode with the common Runtime Requirements codec.
    add(hsm::runtime_requirements,
        ov::util::MemoryView{reinterpret_cast<const std::byte*>(m_runtime_requirements.data()),
                             m_runtime_requirements.size()});
    add(hsm::serialized_model, m_model_encoder, model_alignment);
}

}  // namespace ov::intel_cpu
