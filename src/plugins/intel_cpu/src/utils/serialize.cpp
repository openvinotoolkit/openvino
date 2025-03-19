// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "serialize.hpp"

#include <utility>

#include "openvino/core/descriptor_tensor.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/runtime/shared_buffer.hpp"

namespace ov::intel_cpu {

////////// ModelSerializer //////////

ModelSerializer::ModelSerializer(std::ostream& ostream, CacheEncrypt encrypt_fn)
    : m_ostream(ostream),
      m_cache_encrypt(std::move(encrypt_fn)) {}

void ModelSerializer::operator<<(const std::shared_ptr<ov::Model>& model) {
    auto serialize_info = [&](std::ostream& stream) {
        pugi::xml_document xml_doc;
        pugi::xml_node root = xml_doc.append_child("cnndata");
        root.append_child("outputs");
        xml_doc.save(stream);
    };

    ov::pass::StreamSerialize serializer(m_ostream, serialize_info, m_cache_encrypt);
    serializer.run_on_model(std::const_pointer_cast<ov::Model>(model->clone()));
}

////////// ModelDeserializer //////////

ModelDeserializer::ModelDeserializer(std::istream& model_stream,
                                     std::shared_ptr<ov::AlignedBuffer> model_buffer,
                                     ModelBuilder fn,
                                     const CacheDecrypt& decrypt_fn,
                                     bool decript_from_string)
    : m_istream(model_stream),
      m_model_builder(std::move(fn)),
      m_decript_from_string(decript_from_string),
      m_model_buffer(std::move(model_buffer)) {
    if (m_decript_from_string) {
        m_cache_decrypt.m_decrypt_str = decrypt_fn.m_decrypt_str;
    } else {
        m_cache_decrypt.m_decrypt_char = decrypt_fn.m_decrypt_char;
    }
}

void ModelDeserializer::set_info(pugi::xml_node& root, std::shared_ptr<ov::Model>& model) {}

void ModelDeserializer::operator>>(std::shared_ptr<ov::Model>& model) {
    if (m_model_buffer) {
        process_mmap(model, m_model_buffer);
    } else {
        process_stream(model);
    }
}

void ModelDeserializer::process_mmap(std::shared_ptr<ov::Model>& model,
                                     const std::shared_ptr<ov::AlignedBuffer>& mmemory) {
    // Note: Don't use seekg with mmaped stream. This may affect the performance of some models.
    // Get file size before seek content.
    // Blob from cache may have other header, so need to skip this.
    auto buffer_base = reinterpret_cast<char*>(mmemory->get_ptr());
    const auto file_size = mmemory->size();
    const size_t hdr_pos = m_istream.tellg();

    pass::StreamSerialize::DataHeader hdr = {};
    std::memcpy(reinterpret_cast<char*>(&hdr), buffer_base + hdr_pos, sizeof hdr);

    // Check if model header contains valid data.
    bool is_valid_model = (hdr.custom_data_offset == sizeof(hdr) + hdr_pos) &&
                          (hdr.custom_data_size == hdr.consts_offset - hdr.custom_data_offset) &&
                          (hdr.consts_size == hdr.model_offset - hdr.consts_offset) &&
                          (hdr.model_size = file_size - hdr.model_offset);
    if (!is_valid_model) {
        OPENVINO_THROW("[CPU] Could not deserialize by device xml header.");
    }

    // Read model input/output precisions.
    pugi::xml_document xml_in_out_doc;
    if (hdr.custom_data_size > 0lu) {
        auto res = xml_in_out_doc.load_buffer(buffer_base + hdr.custom_data_offset,
                                              hdr.custom_data_size,
                                              pugi::parse_default,
                                              pugi::encoding_utf8);
        if (res.status != pugi::status_ok) {
            OPENVINO_THROW("[CPU] Could to deserialize custom data.");
        }
    }

    // Map blob content
    std::shared_ptr<ov::AlignedBuffer> weights_buf;
    if (hdr.consts_size) {
        weights_buf =
            std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(buffer_base + hdr.consts_offset,
                                                                                   hdr.consts_size,
                                                                                   mmemory);
    }

    // XML content
    auto xml_buff = std::make_shared<std::string>();
    if (m_cache_decrypt) {
        if (m_decript_from_string) {
            xml_buff->assign(buffer_base + hdr.model_offset, hdr.model_size);
            *xml_buff = m_cache_decrypt.m_decrypt_str(*xml_buff);
        } else {
            xml_buff->reserve(hdr.model_size + 1);
            m_cache_decrypt.m_decrypt_char(&((*xml_buff)[0]), buffer_base + hdr.model_offset, hdr.model_size);
        }
    } else {
        xml_buff->assign(buffer_base + hdr.model_offset, hdr.model_size);
    }
    std::shared_ptr<ov::AlignedBuffer> model_buf =
        std::make_shared<ov::SharedBuffer<std::shared_ptr<std::string>>>(&((*xml_buff)[0]), hdr.model_size, xml_buff);

    model = m_model_builder(model_buf, weights_buf);

    // Set Info
    pugi::xml_node root = xml_in_out_doc.child("cnndata");
    set_info(root, model);
}

void ModelDeserializer::process_stream(std::shared_ptr<ov::Model>& model) {
    const size_t hdr_pos = m_istream.tellg();
    m_istream.seekg(0, m_istream.end);
    const size_t file_size = m_istream.tellg();
    m_istream.seekg(hdr_pos, m_istream.beg);

    pass::StreamSerialize::DataHeader hdr = {};
    m_istream.read(reinterpret_cast<char*>(&hdr), sizeof hdr);

    // Check if model header contains valid data.
    bool is_valid_model = (hdr.custom_data_offset == sizeof(hdr) + hdr_pos) &&
                          (hdr.custom_data_size == hdr.consts_offset - hdr.custom_data_offset) &&
                          (hdr.consts_size == hdr.model_offset - hdr.consts_offset) &&
                          (hdr.model_size = file_size - hdr.model_offset);
    if (!is_valid_model) {
        OPENVINO_THROW("[CPU] Could not deserialize by device xml header.");
    }

    // read model input/output precisions
    m_istream.seekg(hdr.custom_data_offset);

    pugi::xml_document xmlInOutDoc;
    if (hdr.custom_data_size > 0) {
        std::string xmlInOutString;
        xmlInOutString.resize(hdr.custom_data_size);
        m_istream.read(const_cast<char*>(xmlInOutString.c_str()), hdr.custom_data_size);
        auto res = xmlInOutDoc.load_string(xmlInOutString.c_str());
        if (res.status != pugi::status_ok) {
            OPENVINO_THROW("NetworkNotRead: The inputs and outputs information is invalid.");
        }
    }

    // read blob content
    auto data_blob = std::make_shared<ov::Tensor>(ov::element::u8, ov::Shape({hdr.consts_size}));
    m_istream.seekg(hdr.consts_offset);
    if (hdr.consts_size) {
        m_istream.read(static_cast<char*>(data_blob->data(ov::element::u8)), hdr.consts_size);
    }

    // read XML content
    auto xml_string = std::make_shared<std::string>();
    m_istream.seekg(hdr.model_offset);
    xml_string->resize(hdr.model_size);
    m_istream.read(const_cast<char*>(xml_string->data()), hdr.model_size);
    if (m_cache_decrypt) {
        if (m_decript_from_string) {
            *xml_string = m_cache_decrypt.m_decrypt_str(*xml_string);
        } else {
            m_cache_decrypt.m_decrypt_char(const_cast<char*>(xml_string->data()),
                                           xml_string->data(),
                                           xml_string->size());
        }
    }

    auto model_buf =
        std::make_shared<ov::SharedBuffer<std::shared_ptr<std::string>>>(const_cast<char*>(xml_string->data()),
                                                                         xml_string->size(),
                                                                         xml_string);
    auto weights_buf = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::Tensor>>>(
        reinterpret_cast<char*>(data_blob->data(ov::element::u8)),
        hdr.consts_size,
        data_blob);

    model = m_model_builder(model_buf, weights_buf);

    // Set Info
    pugi::xml_node root = xmlInOutDoc.child("cnndata");
    set_info(root, model);
}

}  // namespace ov::intel_cpu
