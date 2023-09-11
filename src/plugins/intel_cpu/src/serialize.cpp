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
namespace {

template <typename T>
void setInfo(pugi::xml_object_range<pugi::xml_named_node_iterator>&& nodes, T&& info) {
    auto nodes_it = nodes.begin();
    auto info_iter = info.begin();
    for (; nodes_it != nodes.end(); ++nodes_it, ++info_iter) {
        auto name_attr = nodes_it->attribute("name");
        auto precision_attr = nodes_it->attribute("precision");
        auto shape_attr = nodes_it->attribute("shape");

        if (!name_attr || !precision_attr || !shape_attr || info_iter == info.end()) {
            OPENVINO_THROW("NetworkNotRead: the inputs/outputs information is invalid.");
        }
        info_iter->get_tensor_ptr()->set_element_type(ov::element::Type(precision_attr.value()));
        info_iter->get_tensor_ptr()->set_tensor_type(ov::element::Type(precision_attr.value()),
                                                     ov::PartialShape(shape_attr.value()));
    }
}
};  // namespace

ModelSerializer::ModelSerializer(std::ostream & ostream, ExtensionManager::Ptr extensionManager)
    : _ostream(ostream)
    , _extensionManager(extensionManager) {
}

void ModelSerializer::operator<<(const std::shared_ptr<ov::Model>& model) {
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

    auto serializeInputsAndOutputs = [&](std::ostream& stream) {
        const std::string name = "cnndata";
        pugi::xml_document xml_doc;
        pugi::xml_node root = xml_doc.append_child(name.c_str());
        pugi::xml_node inputs = root.append_child("inputs");
        pugi::xml_node outputs = root.append_child("outputs");

        // Need it?
        for (const auto& in : model->inputs()) {
            auto in_node = inputs.append_child("in");
            in_node.append_attribute("name").set_value(ov::op::util::get_ie_output_name(in).c_str());
            in_node.append_attribute("precision").set_value(in.get_element_type().get_type_name().c_str());
            in_node.append_attribute("shape").set_value(in.get_partial_shape().to_string().c_str());
        }

        for (const auto& out : model->outputs()) {
            auto out_node = outputs.append_child("out");
            const auto node = out.get_node_shared_ptr();
            out_node.append_attribute("name").set_value(ov::op::util::get_ie_output_name(node->input_value(0)).c_str());
            out_node.append_attribute("precision").set_value(out.get_element_type().get_type_name().c_str());
            out_node.append_attribute("shape").set_value(out.get_partial_shape().to_string().c_str());
        }
        xml_doc.save(stream);
    };

    // Serialize to old representation in case of old API
    OPENVINO_SUPPRESS_DEPRECATED_START
    ov::pass::StreamSerialize serializer(_ostream, getCustomOpSets(), serializeInputsAndOutputs);
    OPENVINO_SUPPRESS_DEPRECATED_END
    serializer.run_on_model(std::const_pointer_cast<ov::Model>(model->clone()));
}

ModelDeserializer::ModelDeserializer(std::istream & istream, model_builder fn)
    : _istream(istream)
    , _model_builder(fn) {
}

void ModelDeserializer::operator>>(std::shared_ptr<ov::Model>& model) {
    using namespace ov::pass;

    std::string xmlString, xmlInOutString;
    ov::Tensor dataBlob;

    StreamSerialize::DataHeader hdr = {};
    _istream.read(reinterpret_cast<char*>(&hdr), sizeof hdr);

    // read model input/output precisions
    _istream.seekg(hdr.custom_data_offset);
    xmlInOutString.resize(hdr.custom_data_size);
    _istream.read(const_cast<char*>(xmlInOutString.c_str()), hdr.custom_data_size);
    pugi::xml_document xmlInOutDoc;
    auto res = xmlInOutDoc.load_string(xmlInOutString.c_str());
    if (res.status != pugi::status_ok) {
        IE_THROW(NetworkNotRead) << "The inputs and outputs information is invalid.";
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
