// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "serialize.h"

#include <pugixml.hpp>

#include "openvino/pass/serialize.hpp"
#include "openvino/util/codec_xor.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_cpu {

static void setInfo(pugi::xml_node& root, std::shared_ptr<ov::Model>& model) {
    pugi::xml_node outputs = root.child("outputs");
    auto nodes_it = outputs.children("out").begin();
    size_t size = model->outputs().size();
    for (size_t i = 0; i < size; ++nodes_it, i++) {
        std::string name = nodes_it->attribute("name").value();
        if (name.empty())
            continue;
        auto result = model->output(i).get_node_shared_ptr();
        ov::descriptor::set_ov_tensor_legacy_name(result->input_value(0).get_tensor(), name);
    }
}

ModelSerializer::ModelSerializer(std::ostream& ostream)
    : _ostream(ostream) {}

void ModelSerializer::operator<<(const std::shared_ptr<ov::Model>& model) {
    auto serializeInfo = [&](std::ostream& stream) {
        const std::string name = "cnndata";
        pugi::xml_document xml_doc;
        pugi::xml_node root = xml_doc.append_child(name.c_str());
        pugi::xml_node outputs = root.append_child("outputs");
        for (const auto& out : model->get_results()) {
            auto out_node = outputs.append_child("out");
            const std::string name = ov::descriptor::get_ov_tensor_legacy_name(out->input_value(0).get_tensor());
            out_node.append_attribute("name").set_value(name.c_str());
        }
        xml_doc.save(stream);
    };

    ov::pass::StreamSerialize serializer(_ostream, serializeInfo, ov::util::codec_xor);
    serializer.run_on_model(std::const_pointer_cast<ov::Model>(model->clone()));
}

ModelDeserializer::ModelDeserializer(std::istream& istream, model_builder fn)
    : m_istream(&istream),
      m_model_buffer(nullptr),
      m_model_builder(fn) {}

ModelDeserializer::ModelDeserializer(const std::shared_ptr<ov::MappedMemory>& buffer, model_builder fn)
    : m_istream(nullptr),
      m_model_buffer(buffer),
      m_model_builder(fn) {}

void ModelDeserializer::operator>>(std::shared_ptr<ov::Model>& model) {
    if (m_model_buffer) {
        parse_buffer(model);
    } else {
        parse_stream(model);
    }
}

void ModelDeserializer::parse_stream(std::shared_ptr<ov::Model>& model) {
    using namespace ov::pass;

    std::string xmlString;
    ov::Tensor dataBlob;

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
        dataBlob = ov::Tensor(ov::element::u8, ov::Shape({hdr.consts_size}));
        m_istream->read(static_cast<char *>(dataBlob.data(ov::element::u8)), hdr.consts_size);
    }

    // read XML content
    m_istream->seekg(hdr.model_offset);
    xmlString.resize(hdr.model_size);
    m_istream->read(const_cast<char*>(xmlString.c_str()), hdr.model_size);
    xmlString = ov::util::codec_xor(xmlString);

    model = m_model_builder(xmlString, std::move(dataBlob));

    // Set Info
    pugi::xml_node root = xmlInOutDoc.child("cnndata");
    setInfo(root, model);
}

void ModelDeserializer::parse_buffer(std::shared_ptr<ov::Model>& model) {
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
    if (hdr.custom_data_size > 0) {
        auto res = xml_in_out_doc.load_string(buffer_base + hdr.custom_data_offset);
        if (res.status != pugi::status_ok) {
            OPENVINO_THROW("[CPU] Could to deserialize custom data.");
        }
    }

    // Map blob content
    if (hdr.consts_size) {
        data_blob = ov::Tensor(element::u8, ov::Shape({hdr.consts_size}), reinterpret_cast<std::uint8_t*>(buffer_base) + hdr.consts_offset);
    }

    // XML content
    const std::string xml_string(buffer_base + hdr.model_offset, hdr.model_size);

    model = m_model_builder(xml_string, std::move(data_blob));

    // Set Info
    pugi::xml_node root = xml_in_out_doc.child("cnndata");
    setInfo(root, model);
}

}   // namespace intel_cpu
}   // namespace ov
