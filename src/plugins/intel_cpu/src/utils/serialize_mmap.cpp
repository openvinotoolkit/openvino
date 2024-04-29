// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "serialize_mmap.hpp"

#include "nodes/common/cpu_memcpy.h"
#include "openvino/pass/serialize.hpp"
#include "openvino/util/codec_xor.hpp"

// #include "../../../../inference/src/compilation_context.hpp"

namespace ov {
namespace intel_cpu {

ModelMmapDeserializer::ModelMmapDeserializer(const std::shared_ptr<ov::MappedMemory>& buffer, model_builder fn)
    : ModelDeserializerBase(fn), m_model_buffer(buffer) {}

void ModelMmapDeserializer::parse(std::shared_ptr<ov::Model>& model) {
printf("[CPU][ModelMmapDeserializer][parse]\n");
    using namespace ov::pass;

    ov::Tensor data_blob;
    auto buffer_base = m_model_buffer->data();
    const auto hdr_pos = m_model_buffer->get_offset();
    const auto file_size = m_model_buffer->size();
printf("    data: %s\n", buffer_base);
// ov::CompiledBlobHeader header;
// m_model_buffer->data() >> header;
// printf("[CPU][CoreImpl][load_model_from_cache]\n    IeVersion: %s\n    FileInfo: %s\n    "
//         "RuntimeInfo: %s\n",
//         header.getIeVersion().c_str(),
//         header.getFileInfo().c_str(),
//         header.getRuntimeInfo().c_str());

    StreamSerialize::DataHeader hdr = {};
    std::memcpy(reinterpret_cast<char*>(&hdr), buffer_base + hdr_pos, sizeof hdr);
printf("    hdr size: %zu\n", sizeof hdr);
// printf("    hdr: %s\n", reinterpret_cast<char*>(&hdr));
printf("    hdr_pos: %zu; file_size: %zu; consts_offset: %zu; model_offset: %zu\n", hdr_pos, file_size, hdr.consts_offset, hdr.model_offset);
printf("    custom_data_offset(%zu) == %zu\n    custom_data_size(%zu) == %zu\n    consts_size(%zu) == %zu\n    model_size(%zu) == %zu\n",
    hdr.custom_data_offset, sizeof(hdr) + hdr_pos,
    hdr.custom_data_size, hdr.consts_offset - hdr.custom_data_offset,
    hdr.consts_size, hdr.model_offset - hdr.consts_offset,
    hdr.model_size, file_size - hdr.model_offset);

    // Check if model header contains valid data.
    bool is_valid_model = (hdr.custom_data_offset == sizeof(hdr) + hdr_pos) &&
                          (hdr.custom_data_size == hdr.consts_offset - hdr.custom_data_offset) &&
                          (hdr.consts_size == hdr.model_offset - hdr.consts_offset) &&
                          (hdr.model_size = file_size - hdr.model_offset);
    if (!is_valid_model) {
        OPENVINO_THROW("[CPU] Could not deserialize xml header.");
    }
printf("[CPU][ModelMmapDeserializer][parse][2]\n");

    // Read model input/output precisions.
    pugi::xml_document xml_in_out_doc;
    if (hdr.custom_data_size > 0) {
        auto res = xml_in_out_doc.load_string(buffer_base + hdr.custom_data_offset);
        if (res.status != pugi::status_ok) {
            OPENVINO_THROW("[CPU] Could to deserialize custom data.");
        }
    }
printf("[CPU][ModelMmapDeserializer][parse][3]\n");

    // Map blob content
    if (hdr.consts_size) {
        data_blob = ov::Tensor(element::u8, ov::Shape({hdr.consts_size}));
        cpu_parallel_memcpy(data_blob.data(element::u8), reinterpret_cast<std::uint8_t*>(buffer_base) + hdr.consts_offset, hdr.consts_size);
    }
printf("[CPU][ModelMmapDeserializer][parse][4]\n");

    // XML content
    std::string xml_string(buffer_base + hdr.model_offset, hdr.model_size);
    // xml_string = ov::util::codec_xor(xml_string);

    model = m_model_builder(xml_string, std::move(data_blob));
printf("[CPU][ModelMmapDeserializer][parse][5]\n");

    // Set Info
    pugi::xml_node root = xml_in_out_doc.child("cnndata");
printf("[CPU][ModelMmapDeserializer][parse][6]\n");
    set_info(root, model);
printf("[CPU][ModelMmapDeserializer][parse][7]\n");
}

}   // namespace intel_cpu
}   // namespace ov
