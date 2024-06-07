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

ModelDeserializer::ModelDeserializer(std::istream & istream, model_builder fn)
    : _istream(istream)
    , _model_builder(fn) {
}

void ModelDeserializer::operator>>(std::shared_ptr<ov::Model>& model) {
    using namespace ov::pass;

    std::string xmlString;
    ov::Tensor dataBlob;

    // get file size before seek content
    // blob from cache may have other header, skip it
    const size_t _pos = _istream.tellg();
    _istream.seekg(0, _istream.end);
    const size_t file_size = _istream.tellg();
    _istream.seekg(_pos, _istream.beg);

    StreamSerialize::DataHeader hdr = {};
    _istream.read(reinterpret_cast<char*>(&hdr), sizeof hdr);

    // check if model header contains valid data
    bool isValidModel = (hdr.custom_data_offset == sizeof(hdr) + _pos) &&
                        (hdr.custom_data_size == hdr.consts_offset - hdr.custom_data_offset) &&
                        (hdr.consts_size == hdr.model_offset - hdr.consts_offset) &&
                        (hdr.model_size = file_size - hdr.model_offset);
    if (!isValidModel) {
        OPENVINO_THROW("Failed to read CPU device xml header");
    }
    // read model input/output precisions
    _istream.seekg(hdr.custom_data_offset);

    pugi::xml_document xmlInOutDoc;
    if (hdr.custom_data_size > 0) {
        std::string xmlInOutString;
        xmlInOutString.resize(hdr.custom_data_size);
        _istream.read(const_cast<char*>(xmlInOutString.c_str()), hdr.custom_data_size);
        auto res = xmlInOutDoc.load_string(xmlInOutString.c_str());
        if (res.status != pugi::status_ok) {
            OPENVINO_THROW("NetworkNotRead: The inputs and outputs information is invalid.");
        }
    }

    // read blob content
    _istream.seekg(hdr.consts_offset);
    if (hdr.consts_size) {
        dataBlob = ov::Tensor(ov::element::u8, ov::Shape({hdr.consts_size}));
        _istream.read(static_cast<char *>(dataBlob.data(ov::element::u8)), hdr.consts_size);
    }

    // read XML content
    _istream.seekg(hdr.model_offset);
    xmlString.resize(hdr.model_size);
    _istream.read(const_cast<char*>(xmlString.c_str()), hdr.model_size);
    xmlString = ov::util::codec_xor(xmlString);

    model = _model_builder(xmlString, std::move(dataBlob));

    // Set Info
    pugi::xml_node root = xmlInOutDoc.child("cnndata");
    setInfo(root, model);
}

}   // namespace intel_cpu
}   // namespace ov
