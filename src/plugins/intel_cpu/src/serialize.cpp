// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "serialize.h"

#include <openvino/pass/serialize.hpp>

#include <pugixml.hpp>

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace {
    std::string to_string(InferenceEngine::Layout layout) {
        std::stringstream ss;
        ss << layout;
        return ss.str();
    }

    InferenceEngine::Layout layout_from_string(const std::string & name) {
        static const std::unordered_map<std::string, InferenceEngine::Layout> layouts = {
            { "ANY", InferenceEngine::Layout::ANY },
            { "NCHW", InferenceEngine::Layout::NCHW },
            { "NHWC", InferenceEngine::Layout::NHWC },
            { "NCDHW", InferenceEngine::Layout::NCDHW },
            { "NDHWC", InferenceEngine::Layout::NDHWC },
            { "OIHW", InferenceEngine::Layout::OIHW },
            { "C", InferenceEngine::Layout::C },
            { "CHW", InferenceEngine::Layout::CHW },
            { "HWC", InferenceEngine::Layout::HWC },
            { "HW", InferenceEngine::Layout::HW },
            { "NC", InferenceEngine::Layout::NC },
            { "CN", InferenceEngine::Layout::CN },
            { "BLOCKED", InferenceEngine::Layout::BLOCKED }
        };
        auto it = layouts.find(name);
        if (it != layouts.end()) {
            return it->second;
        }
        IE_THROW(NetworkNotRead) << "Unknown layout with name '" << name << "'";
    }

    template<typename T>
    void setPrecisionsAndLayouts(
        pugi::xml_object_range<pugi::xml_named_node_iterator> && nodes,
        T && info) {
        for (auto n : nodes) {
            auto name_attr = n.attribute("name");
            auto precision_attr = n.attribute("precision");
            auto layout_attr = n.attribute("layout");

            if (!name_attr
                || !precision_attr
                || !layout_attr) {
                IE_THROW(NetworkNotRead) << "The inputs/outputs information is invalid.";
            }

            auto it = info.find(name_attr.value());
            if (it == info.end()) {
                IE_THROW(NetworkNotRead) << "The input/output with name '" << name_attr.value() << "' not found";
            }

            it->second->setPrecision(Precision::FromStr(precision_attr.value()));
            it->second->setLayout(layout_from_string(layout_attr.value()));
        }
    }
};  // namespace

CNNNetworkSerializer::CNNNetworkSerializer(std::ostream & ostream, ExtensionManager::Ptr extensionManager)
    : _ostream(ostream)
    , _extensionManager(extensionManager) {
}

void CNNNetworkSerializer::operator << (const CNNNetwork & network) {
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

    auto serializeInputsAndOutputs = [&](std::ostream & stream) {
        const std::string name = "cnndata";
        pugi::xml_document xml_doc;
        pugi::xml_node root = xml_doc.append_child(name.c_str());
        pugi::xml_node inputs = root.append_child("inputs");
        pugi::xml_node outputs = root.append_child("outputs");

        for (const auto & in : network.getInputsInfo()) {
            auto in_node = inputs.append_child("in");

            in_node.append_attribute("name")
                    .set_value(in.first.c_str());
            in_node.append_attribute("precision")
                    .set_value(in.second->getPrecision().name());
            in_node.append_attribute("layout")
                    .set_value(to_string(in.second->getLayout()).c_str());
        }

        for (const auto & out : network.getOutputsInfo()) {
            auto out_node = outputs.append_child("out");
            out_node.append_attribute("name")
                    .set_value(out.first.c_str());
            out_node.append_attribute("precision")
                    .set_value(out.second->getPrecision().name());
            out_node.append_attribute("layout")
                    .set_value(to_string(out.second->getLayout()).c_str());
        }

        xml_doc.save(stream);
    };

    // Serialize to old representation in case of old API
    OPENVINO_SUPPRESS_DEPRECATED_START
    ov::pass::StreamSerialize serializer(_ostream, getCustomOpSets(), serializeInputsAndOutputs);
    OPENVINO_SUPPRESS_DEPRECATED_END
    serializer.run_on_model(std::const_pointer_cast<ngraph::Function>(network.getFunction()));
}

CNNNetworkDeserializer::CNNNetworkDeserializer(std::istream & istream, cnn_network_builder fn)
    : _istream(istream)
    , _cnn_network_builder(fn) {
}

void CNNNetworkDeserializer::operator >> (InferenceEngine::CNNNetwork & network) {
    using namespace ov::pass;

    std::string xmlString, xmlInOutString;
    InferenceEngine::Blob::Ptr dataBlob;

    StreamSerialize::DataHeader hdr = {};
    _istream.read(reinterpret_cast<char*>(&hdr), sizeof hdr);

    // read CNNNetwork input/output precisions
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
        dataBlob = InferenceEngine::make_shared_blob<std::uint8_t>(
            InferenceEngine::TensorDesc(InferenceEngine::Precision::U8, {hdr.consts_size}, InferenceEngine::Layout::C));
        dataBlob->allocate();
        _istream.read(dataBlob->buffer(), hdr.consts_size);
    }

    // read XML content
    _istream.seekg(hdr.model_offset);
    xmlString.resize(hdr.model_size);
    _istream.read(const_cast<char*>(xmlString.c_str()), hdr.model_size);

    network = _cnn_network_builder(xmlString, std::move(dataBlob));

    // Set input and output precisions
    pugi::xml_node root = xmlInOutDoc.child("cnndata");
    pugi::xml_node inputs = root.child("inputs");
    pugi::xml_node outputs = root.child("outputs");

    setPrecisionsAndLayouts(inputs.children("in"), network.getInputsInfo());
    setPrecisionsAndLayouts(outputs.children("out"), network.getOutputsInfo());
}

}   // namespace intel_cpu
}   // namespace ov
