// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "network_serializer.hpp"

#include <map>
#include <deque>
#include <string>
#include <vector>

#include "exec_graph_info.hpp"
#include "xml_parse_utils.h"
#include <legacy/ie_ngraph_utils.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/function.hpp>

namespace InferenceEngine {
namespace Serialization {

namespace {

void FillXmlDocWithExecutionNGraph(const InferenceEngine::ICNNNetwork& network,
                                   pugi::xml_document& doc) {
    std::shared_ptr<const ngraph::Function> function = network.getFunction();
    if (function == nullptr) {
        THROW_IE_EXCEPTION << network.getName() << " does not represent ngraph::Function";
    }

    std::vector<std::shared_ptr<ngraph::Node>> ordered = function->get_ordered_ops();
    pugi::xml_node netXml = doc.append_child("net");
    netXml.append_attribute("name").set_value(network.getName().c_str());

    pugi::xml_node layers = netXml.append_child("layers");
    std::unordered_map<std::shared_ptr<ngraph::Node>, size_t> matching;

    for (size_t i = 0; i < ordered.size(); ++i) {
        matching[ordered[i]] = i;
        const std::shared_ptr<ngraph::Node> node = ordered[i];
        auto params = node->get_rt_info();

        auto layerTypeVariant = params.find(ExecGraphInfoSerialization::LAYER_TYPE);
        if (layerTypeVariant == params.end()) {
            THROW_IE_EXCEPTION << node->get_friendly_name() << " does not define "
                               << ExecGraphInfoSerialization::LAYER_TYPE << " attribute.";
        }
        using VariantString = ngraph::VariantImpl<std::string>;
        auto layerTypeValueStr = std::dynamic_pointer_cast<VariantString>(layerTypeVariant->second);
        IE_ASSERT(layerTypeValueStr != nullptr);
        params.erase(layerTypeVariant);

        pugi::xml_node layer = layers.append_child("layer");
        layer.append_attribute("name").set_value(node->get_friendly_name().c_str());
        layer.append_attribute("type").set_value(layerTypeValueStr->get().c_str());
        layer.append_attribute("id").set_value(i);

        if (!params.empty()) {
            pugi::xml_node data = layer.append_child("data");

            for (const auto& it : params) {
                if (auto strValue = std::dynamic_pointer_cast<VariantString>(it.second))
                    data.append_attribute(it.first.c_str()).set_value(strValue->get().c_str());
            }
        }

        if (node->get_input_size() > 0) {
            pugi::xml_node input = layer.append_child("input");

            for (size_t iport = 0; iport < node->get_input_size(); iport++) {
                const ngraph::Shape & dims = node->get_input_shape(iport);
                pugi::xml_node port = input.append_child("port");

                port.append_attribute("id").set_value(iport);
                for (auto dim : dims) {
                    port.append_child("dim").text().set(dim);
                }
            }
        }
        if (node->get_output_size() > 0 &&
            // ngraph::op::Result still have single output while we should not print it
            !std::dynamic_pointer_cast<ngraph::op::Result>(node)) {
            pugi::xml_node output = layer.append_child("output");

            for (size_t oport = 0; oport < node->get_output_size(); oport++) {
                pugi::xml_node port = output.append_child("port");
                Precision outputPrecision = details::convertPrecision(node->get_output_element_type(oport));

                port.append_attribute("id").set_value(node->get_input_size() + oport);
                port.append_attribute("precision").set_value(outputPrecision.name());

                for (const auto dim : node->get_output_shape(oport)) {
                    port.append_child("dim").text().set(dim);
                }
            }
        }
    }

    pugi::xml_node edges = netXml.append_child("edges");

    for (const auto& ord : ordered) {
        const std::shared_ptr<ngraph::Node> parentNode = ord;

        if (parentNode->get_output_size() > 0) {
            auto itFrom = matching.find(parentNode);
            if (itFrom == matching.end()) {
                THROW_IE_EXCEPTION << "Internal error, cannot find " << parentNode->get_friendly_name()
                                   << " in matching container during serialization of IR";
            }
            for (size_t oport = 0; oport < parentNode->get_output_size(); oport++) {
                ngraph::Output<ngraph::Node> parentPort = parentNode->output(oport);
                for (const auto& childPort : parentPort.get_target_inputs()) {
                    ngraph::Node * childNode = childPort.get_node();
                    for (int iport = 0; iport < childNode->get_input_size(); iport++) {
                        if (childNode->input_value(iport).get_node() == parentPort.get_node()) {
                            auto itTo = matching.find(childNode->shared_from_this());
                            if (itTo == matching.end()) {
                                THROW_IE_EXCEPTION << "Broken edge form layer "
                                                   << parentNode->get_friendly_name() << " to layer "
                                                   << childNode->get_friendly_name()
                                                   << "during serialization of IR";
                            }
                            pugi::xml_node edge = edges.append_child("edge");
                            edge.append_attribute("from-layer").set_value(itFrom->second);
                            edge.append_attribute("from-port").set_value(oport + parentNode->get_input_size());

                            edge.append_attribute("to-layer").set_value(itTo->second);
                            edge.append_attribute("to-port").set_value(iport);
                        }
                    }
                }
            }
        }
    }
}

}  // namespace

void SerializeV10(const std::string& xmlPath, const std::string& binPath,
                  const InferenceEngine::ICNNNetwork& network) {
    if (auto function = network.getFunction()) {
        // A flag for serializing executable graph information (not complete IR)
        bool execGraphInfoSerialization = true;

        // go over all operations and check whether performance stat is set
        for (const auto & op : function->get_ops()) {
            auto & rtInfo = op->get_rt_info();
            if (rtInfo.find(ExecGraphInfoSerialization::PERF_COUNTER) == rtInfo.end()) {
                execGraphInfoSerialization = false;
                break;
            }
        }

        if (execGraphInfoSerialization) {
            pugi::xml_document doc;
            FillXmlDocWithExecutionNGraph(network, doc);

            if (!doc.save_file(xmlPath.c_str())) {
                THROW_IE_EXCEPTION << "File '" << xmlPath << "' was not serialized";
            }
        } else {
            THROW_IE_EXCEPTION << "Serialization to IR v10 is not implemented in Inference Engine";
        }
    } else {
        THROW_IE_EXCEPTION << "Serialization to IR v7 is removed from Inference Engine";
    }
}
}  //  namespace Serialization
}  //  namespace InferenceEngine
