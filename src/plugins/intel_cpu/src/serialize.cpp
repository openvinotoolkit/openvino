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

    // Serialize to old representation in case of old API
    ov::pass::StreamSerialize serializer(_ostream, getCustomOpSets(), serializeInfo);
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

    // Set Info
    pugi::xml_node root = xmlInOutDoc.child("cnndata");
    setInfo(root, model);
}

}   // namespace intel_cpu
}   // namespace ov
