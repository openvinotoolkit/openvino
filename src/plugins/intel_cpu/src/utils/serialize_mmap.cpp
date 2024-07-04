// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "serialize_mmap.hpp"

#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/core/parallel.hpp"

namespace ov {
namespace intel_cpu {

void codec_xor(char* dst_str, const char* src_str, size_t len) {
    static const char codec_key[] = {0x30, 0x60, 0x70, 0x02, 0x04, 0x08, 0x3F, 0x6F, 0x72, 0x74, 0x78, 0x7F};
    auto key_size = sizeof(codec_key);

    parallel_for(len, [&](size_t key_idx) {
        dst_str[key_idx] = src_str[key_idx] ^ codec_key[key_idx % key_size];
    });
}

ModelMmapDeserializer::ModelMmapDeserializer(const std::shared_ptr<ov::MappedMemory>& buffer, model_builder fn)
    : m_model_buffer(buffer), m_model_builder(fn) {}

void ModelMmapDeserializer::parse(std::shared_ptr<ov::Model>& model) {
    using namespace ov::pass;

    ov::Tensor data_blob;
    auto buffer_base = m_model_buffer->data();
    const auto hdr_pos = m_model_buffer->get_offset();
    const auto file_size = m_model_buffer->size();

    StreamSerialize::DataHeader hdr = {};
    std::memcpy(reinterpret_cast<char*>(&hdr), buffer_base + hdr_pos, sizeof hdr);

    // Check if model header contains valid data.
    bool is_valid_model = (hdr.custom_data_offset == sizeof(hdr) + hdr_pos) &&
                          (hdr.custom_data_size == hdr.consts_offset - hdr.custom_data_offset) &&
                          (hdr.consts_size == hdr.model_offset - hdr.consts_offset) &&
                          (hdr.model_size = file_size - hdr.model_offset);
    if (!is_valid_model) {
        OPENVINO_THROW("[CPU] Could not deserialize xml header.");
    }

    // Read model input/output precisions.
    pugi::xml_document xml_in_out_doc;
    if (hdr.custom_data_size > 0lu) {
        auto res = xml_in_out_doc.load_buffer(buffer_base + hdr.custom_data_offset, hdr.custom_data_size, pugi::parse_default, pugi::encoding_utf8);
        if (res.status != pugi::status_ok) {
            OPENVINO_THROW("[CPU] Could to deserialize custom data.");
        }
    }

    // Map blob content
    std::shared_ptr<ov::AlignedBuffer> weights_buf;
    if (hdr.consts_size) {
        weights_buf = std::make_shared<ov::SharedBuffer<std::shared_ptr<MappedMemory>>>(buffer_base + hdr.consts_offset,
                                                                                        hdr.consts_size,
                                                                                        m_model_buffer);
    }

    // XML content
    std::string xml_buff;
    xml_buff.reserve(hdr.model_size + 1);
    codec_xor(&(xml_buff.front()), buffer_base + hdr.model_offset, hdr.model_size);
    std::shared_ptr<ov::AlignedBuffer> model_buf =
            std::make_shared<ov::SharedBuffer<std::shared_ptr<MappedMemory>>>(&(xml_buff.front()),
                                                                              hdr.model_size,
                                                                              m_model_buffer);

    model = m_model_builder(model_buf, weights_buf);

    // Set Info
    pugi::xml_node root = xml_in_out_doc.child("cnndata");
    set_info(root, model);
}

}   // namespace intel_cpu
}   // namespace ov
