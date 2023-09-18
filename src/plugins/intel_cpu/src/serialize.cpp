// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "serialize.h"

#include <pugixml.hpp>

#include "openvino/pass/serialize.hpp"
#include "transformations/utils/utils.hpp"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {

ModelSerializer::ModelSerializer(std::ostream & ostream, ExtensionManager::Ptr extensionManager)
    : _ostream(ostream)
    , _extensionManager(extensionManager) {
}

void ModelSerializer::operator<<(const std::shared_ptr<ov::Model>& model) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    auto getCustomOpSets = [this]() {
        std::map<std::string, ngraph::OpSet> custom_opsets;

        if (_extensionManager) {
            auto extensions = _extensionManager->Extensions();
            for (const auto& extension : extensions) {
                auto opset = extension->getOpSets();
                custom_opsets.insert(std::begin(opset), std::end(opset));
            }
        }

        return custom_opsets;
    };

    // Serialize to old representation in case of old API
    ov::pass::StreamSerialize serializer(_ostream, getCustomOpSets(), {});
    OPENVINO_SUPPRESS_DEPRECATED_END
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

    OPENVINO_SUPPRESS_DEPRECATED_START
    if (hdr.custom_data_size > 0) {
        std::string xmlInOutString;
        xmlInOutString.resize(hdr.custom_data_size);
        _istream.read(const_cast<char*>(xmlInOutString.c_str()), hdr.custom_data_size);
        pugi::xml_document xmlInOutDoc;
        auto res = xmlInOutDoc.load_string(xmlInOutString.c_str());
        if (res.status != pugi::status_ok) {
            OPENVINO_THROW("NetworkNotRead: The inputs and outputs information is invalid.");
        }
    }
    OPENVINO_SUPPRESS_DEPRECATED_END

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
