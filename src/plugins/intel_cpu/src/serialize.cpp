// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "serialize.h"

#include <pugixml.hpp>

#include "openvino/pass/serialize.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_cpu {

ModelSerializer::ModelSerializer(std::ostream& ostream) : _ostream(ostream) {}

void ModelSerializer::operator<<(const std::shared_ptr<ov::Model>& model) {
    ov::pass::StreamSerialize serializer(_ostream);
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

    StreamSerialize::DataHeader hdr = {};
    _istream.read(reinterpret_cast<char*>(&hdr), sizeof hdr);

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

    model = _model_builder(xmlString, std::move(dataBlob));
}

}   // namespace intel_cpu
}   // namespace ov
