// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_graph_dumper.h"
#include "cnn_network_impl.hpp"
#include "ie_util_internal.hpp"
#include "ie_ngraph_utils.hpp"
#include "exec_graph_info.hpp"
#include "mkldnn_debug.h"
#include "generic_ie.hpp"
#include <ngraph/variant.hpp>

#include <vector>
#include <string>
#include <memory>
#include <map>

using namespace InferenceEngine;

namespace MKLDNNPlugin {

namespace {

std::map<std::string, std::string> extract_node_metadata(const MKLDNNNodePtr &);
void drawer_callback(const InferenceEngine::CNNLayerPtr, ordered_properties &, ordered_properties &);

}  // namespace

CNNLayer::Ptr create_cnnlayer(const MKLDNNNodePtr &node) {
    CNNLayer::Ptr layer(new CNNLayer({node->getName(), "type", Precision::FP32}));

    layer->params = extract_node_metadata(node);
    layer->type = layer->params[ExecGraphInfoSerialization::LAYER_TYPE];
    layer->params.erase(ExecGraphInfoSerialization::LAYER_TYPE);

    auto &cfg = node->getSelectedPrimitiveDescriptor()->getConfig();
    layer->insData.resize(cfg.inConfs.size());
    layer->outData.resize(cfg.outConfs.size());

    return layer;
}

std::shared_ptr<ICNNNetwork> dump_graph_as_ie_ngraph_net(const MKLDNNGraph &graph) {
    std::map<MKLDNNNodePtr, std::shared_ptr<ngraph::Node> > node2layer;

    ngraph::ResultVector results;
    ngraph::ParameterVector params;

    auto get_inputs = [&] (const MKLDNNNodePtr & node) {
        auto pr_edges = node->getParentEdges();
        ngraph::OutputVector inputs(pr_edges.size());

        for (int i = 0; i < pr_edges.size(); i++) {
            auto edge = node->getParentEdgeAt(i);
            int pr_port = edge->getInputNum();
            int ch_port = edge->getOutputNum();
            auto pr_node = edge->getParent();

            IE_ASSERT(node2layer.count(pr_node) == 1);
            auto pr = node2layer[pr_node];

            inputs[ch_port] = pr->output(pr_port);
        }

        return inputs;
    };

    auto create_ngraph_node = [&](const MKLDNNNodePtr &node) {
        bool is_input = false, is_output = false;
        for (auto && kvp : graph.inputNodes) {
            if (kvp.second == node) {
                is_input = true;
                break;
            }
        }

        for (auto && onode : graph.outputNodes) {
            if (onode == node) {
                is_output = true;
                break;
            }
        }

        auto meta_data = extract_node_metadata(node);
        std::shared_ptr<ngraph::Node> return_node;
        if (is_input) {
            auto desc = node->getChildEdgeAt(0)->getDesc();
            auto param = std::make_shared<ngraph::op::Parameter>(
                details::convertPrecision(desc.getPrecision()),
                ngraph::PartialShape(desc.getDims()));
            return_node = param;
            params.push_back(param);
        } else if (is_output) {
            results.emplace_back(std::make_shared<ngraph::op::Result>(get_inputs(node).back()));
            return_node = results.back();
        } else {
            return_node = std::make_shared<ExecGraphInfoSerialization::ExecutionNode>(
                get_inputs(node), node->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size());

            for (size_t port = 0; port < return_node->get_output_size(); ++port) {
                auto desc = node->getChildEdgeAt(port)->getDesc();
                return_node->set_output_type(port,
                    details::convertPrecision(desc.getPrecision()),
                    ngraph::PartialShape(desc.getDims()));
            }
        }

        for (auto && kvp : meta_data)
            return_node->get_rt_info()[kvp.first] = std::make_shared<::ngraph::VariantWrapper<std::string>>(kvp.second);
        return_node->set_friendly_name(node->getName());

        return return_node;
    };

    ngraph::NodeVector nodes;
    nodes.reserve(graph.graphNodes.size());
    for (auto &node : graph.graphNodes) {  // important: graph.graphNodes are in topological order
        nodes.emplace_back(create_ngraph_node(node));
        node2layer[node] = nodes.back();
    }

    ngraph::op::GenericIE::DisableReshape reshape(nodes);
    auto function = std::make_shared<ngraph::Function>(results, params, graph._name);
    InferenceEngine::CNNNetwork net(function);
    return net;
}

std::shared_ptr<ICNNNetwork> dump_graph_as_ie_net(const MKLDNNGraph &graph) {
    auto net = std::make_shared<details::CNNNetworkImpl>();

    net->setName(graph._name);
    std::map<MKLDNNNodePtr, CNNLayerPtr> node2layer;

    // Copy all nodes to network
    for (auto &node : graph.graphNodes) {
        auto layer = create_cnnlayer(node);
        node2layer[node] = layer;
        net->addLayer(layer);
    }

    // Copy all edges to network
    for (auto &node : graph.graphNodes) {
        auto pr = node2layer[node];
        auto ch_edges = node->getChildEdges();

        for (int i = 0; i < ch_edges.size(); i++) {
            auto edge = node->getChildEdgeAt(i);
            int in_port = edge->getOutputNum();
            auto ch_node = edge->getChild();
            auto ch  = node2layer[ch_node];

            DataPtr data;
            if (i < pr->outData.size()) {
                std::string data_name = node->getName() + "_out" + std::to_string(i);
                pr->outData[i] = std::make_shared<Data>(data_name, edge->getDesc());
                data = pr->outData[i];
                data->getCreatorLayer() = pr;
            } else {
                data = pr->outData[0];
            }

            data->getInputTo()[ch->name] = ch;
            ch->insData[in_port] = data;
        }
    }

    // Specify inputs data
    for (auto kvp : graph.inputNodes) {
        auto in_node = kvp.second;
        auto in_layer = node2layer[in_node];

        auto in_info = std::make_shared<InputInfo>();
        in_info->setInputData(in_layer->outData[0]);
        net->setInputInfo(in_info);
    }

    return net;
}

void dump_graph_as_dot(const MKLDNNGraph &graph, std::ostream &out) {
    auto dump_net = dump_graph_as_ie_net(graph);
    if (dump_net == nullptr)
        THROW_IE_EXCEPTION << "Nullable net dump";
    InferenceEngine::saveGraphToDot(*dump_net, out, drawer_callback);
}

//**********************************
// Special converters of meta data
//**********************************

namespace {

std::map<std::string, std::string> extract_node_metadata(const MKLDNNNodePtr &node) {
    std::map<std::string, std::string> serialization_info;

    if (node->getType() == Input && node->isConstant()) {
        // We need to separate Input and Const layers
        serialization_info[ExecGraphInfoSerialization::LAYER_TYPE] = "Const";
    } else if (node->getType() == Generic) {
        // Path to print actual name for extension layers
        serialization_info[ExecGraphInfoSerialization::LAYER_TYPE] = node->getTypeStr();
    } else {
        serialization_info[ExecGraphInfoSerialization::LAYER_TYPE] = NameFromType(node->getType());
    }

    // Original layers
    serialization_info[ExecGraphInfoSerialization::ORIGINAL_NAMES] = node->getOriginalLayers();

    // Implementation type name
    serialization_info[ExecGraphInfoSerialization::IMPL_TYPE] = node->getPrimitiveDescriptorType();

    std::string outputPrecisionsStr;
    if (!node->getChildEdges().empty()) {
        outputPrecisionsStr = node->getChildEdgeAt(0)->getDesc().getPrecision().name();

        bool isAllEqual = true;
        for (size_t i = 1; i < node->getChildEdges().size(); i++) {
            if (node->getChildEdgeAt(i-1)->getDesc().getPrecision() != node->getChildEdgeAt(i)->getDesc().getPrecision()) {
                isAllEqual = false;
                break;
            }
        }

        // If all output precisions are the same, we store the name only once
        if (!isAllEqual) {
            for (size_t i = 1; i < node->getChildEdges().size(); i++)
                outputPrecisionsStr += "," + std::string(node->getChildEdgeAt(i)->getDesc().getPrecision().name());
        }
    } else {
        // Branch to correctly handle output nodes
        if (!node->getParentEdges().empty()) {
            outputPrecisionsStr = node->getParentEdgeAt(0)->getDesc().getPrecision().name();
        }
    }
    serialization_info[ExecGraphInfoSerialization::OUTPUT_PRECISIONS] = outputPrecisionsStr;

    std::string outputLayoutsStr;
    auto outDataConfigs = node->getSelectedPrimitiveDescriptor()->getConfig().outConfs;
    std::vector<std::string> outLayouts;
    outLayouts.reserve(outDataConfigs.size());
    for (auto &cfg : outDataConfigs) {
        outLayouts.push_back(format_tag_to_string(get_format_tag(cfg.desc)));
    }
    if (!outLayouts.empty()) {
        outputLayoutsStr = outLayouts[0];

        bool isAllEqual = std::equal(outLayouts.begin() + 1, outLayouts.end(), outLayouts.begin());

        // If all output layouts are the same, we store the name only once
        if (!isAllEqual) {
            for (size_t i = 1; i < outLayouts.size(); i++)
                outputLayoutsStr += "," + outLayouts[i];
        }
    } else {
        outputLayoutsStr = format_tag_to_string(mkldnn::memory::format_undef);
    }
    serialization_info[ExecGraphInfoSerialization::OUTPUT_LAYOUTS] = outputLayoutsStr;

    // Performance
    if (node->PerfCounter().avg() != 0) {
        serialization_info[ExecGraphInfoSerialization::PERF_COUNTER] = std::to_string(node->PerfCounter().avg());
    } else {
        serialization_info[ExecGraphInfoSerialization::PERF_COUNTER] = "not_executed";  // it means it was not calculated yet
    }

    serialization_info[ExecGraphInfoSerialization::EXECUTION_ORDER] = std::to_string(node->getExecIndex());

    return serialization_info;
}

const char BLUE[]  = "#D8D9F1";
const char GREEN[] = "#D9EAD3";

void drawer_callback(const InferenceEngine::CNNLayerPtr layer,
                     ordered_properties &printed_properties,
                     ordered_properties &node_properties) {
    const auto &params = layer->params;

    // Implementation
    auto impl = params.find(ExecGraphInfoSerialization::IMPL_TYPE);
    if (impl != params.end()) {
        printed_properties.push_back({"impl", impl->second});
    }

    // Original names
    auto orig = params.find(ExecGraphInfoSerialization::ORIGINAL_NAMES);
    if (orig != params.end()) {
        printed_properties.push_back({"originals", orig->second});
    }

    // Precision
    auto prec = params.find(ExecGraphInfoSerialization::OUTPUT_PRECISIONS);
    if (prec != params.end()) {
        printed_properties.push_back({"precision", prec->second});
        // Set color
        node_properties.push_back({"fillcolor", prec->second == "FP32" ? GREEN : BLUE});
    }

    // Set xlabel containing PM data if calculated
    auto perf = layer->params.find(ExecGraphInfoSerialization::PERF_COUNTER);
    node_properties.push_back({"xlabel", (perf != layer->params.end()) ? perf->second : ""});
}

}  // namespace

}  // namespace MKLDNNPlugin
