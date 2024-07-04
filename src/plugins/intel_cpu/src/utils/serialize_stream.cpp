// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "serialize_stream.hpp"

#include "openvino/pass/serialize.hpp"
#include "openvino/util/codec_xor.hpp"

namespace ov {
namespace intel_cpu {

ModelStreamDeserializer::ModelStreamDeserializer(std::istream& istream, model_builder fn)
    : m_istream(&istream), m_model_builder(fn) {}

void ModelStreamDeserializer::parse(std::shared_ptr<ov::Model>& model) {
    using namespace ov::pass;

    std::string xml_string;
    ov::Tensor data_blob;

    // get file size before seek content
    // blob from cache may have other header, skip it
    const size_t _pos = m_istream->tellg();
    m_istream->seekg(0, m_istream->end);
    const size_t file_size = m_istream->tellg();
    m_istream->seekg(_pos, m_istream->beg);

    StreamSerialize::DataHeader hdr = {};
    m_istream->read(reinterpret_cast<char*>(&hdr), sizeof hdr);

    // check if model header contains valid data
    bool isValidModel = (hdr.custom_data_offset == sizeof(hdr) + _pos) &&
                        (hdr.custom_data_size == hdr.consts_offset - hdr.custom_data_offset) &&
                        (hdr.consts_size == hdr.model_offset - hdr.consts_offset) &&
                        (hdr.model_size = file_size - hdr.model_offset);
    if (!isValidModel) {
        OPENVINO_THROW("Failed to read CPU device xml header");
    }
    // read model input/output precisions
    m_istream->seekg(hdr.custom_data_offset);

    pugi::xml_document xmlInOutDoc;
    if (hdr.custom_data_size > 0) {
        std::string xmlInOutString;
        xmlInOutString.resize(hdr.custom_data_size);
        m_istream->read(const_cast<char*>(xmlInOutString.c_str()), hdr.custom_data_size);
        auto res = xmlInOutDoc.load_string(xmlInOutString.c_str());
        if (res.status != pugi::status_ok) {
            OPENVINO_THROW("NetworkNotRead: The inputs and outputs information is invalid.");
        }
    }

    // read blob content
    m_istream->seekg(hdr.consts_offset);
    if (hdr.consts_size) {
        data_blob = ov::Tensor(ov::element::u8, ov::Shape({hdr.consts_size}));
        m_istream->read(static_cast<char *>(data_blob.data(ov::element::u8)), hdr.consts_size);
    }

    // read XML content
    m_istream->seekg(hdr.model_offset);
    xml_string.resize(hdr.model_size);
    m_istream->read(const_cast<char*>(xml_string.c_str()), hdr.model_size);
    xml_string = ov::util::codec_xor(xml_string);

    model = m_model_builder(xml_string, std::move(data_blob));

    // Set Info
    pugi::xml_node root = xmlInOutDoc.child("cnndata");
    set_info(root, model);
}

}   // namespace intel_cpu
}   // namespace ov
