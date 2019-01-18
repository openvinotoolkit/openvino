// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_graph_dumper.h"
#include "cnn_network_impl.hpp"
#include "ie_util_internal.hpp"

#include <vector>
#include <string>
#include <memory>
#include <map>

using namespace InferenceEngine;

namespace MKLDNNPlugin {

static void copy_node_metadata(const MKLDNNNodePtr &, CNNLayer::Ptr &);
static void drawer_callback(const InferenceEngine::CNNLayerPtr, ordered_properties &, ordered_properties &);

CNNLayer::Ptr convert_node(const MKLDNNNodePtr &node) {
    CNNLayer::Ptr layer(new CNNLayer({"name", "type", Precision::FP32}));
    copy_node_metadata(node, layer);

    auto &cfg = node->getSelectedPrimitiveDescriptor()->getConfig();
    layer->insData.resize(cfg.inConfs.size());
    layer->outData.resize(cfg.outConfs.size());

    return layer;
}

std::shared_ptr<ICNNNetwork> dump_graph_as_ie_net(const MKLDNNGraph &graph) {
    auto net = std::make_shared<details::CNNNetworkImpl>();

    net->setPrecision(Precision::FP32);
    net->setName("internal_cpu_graph");
    std::map<MKLDNNNodePtr, CNNLayerPtr> node2layer;

    // Copy all nodes to network
    for (auto &node : graph.graphNodes) {
        auto layer = convert_node(node);
        node2layer[node] = layer;
        net->addLayer(layer);
    }

    // Copy all edges to network
    for (auto &node : graph.graphNodes) {
        auto pr = node2layer[node];
        auto ch_edges = node->getChildEdges();

        for (int i = 0; i < ch_edges.size(); i++) {
            auto edge = node->getChildEdgeAt(i);
            int out_port = edge->getInputNum();
            int in_port = edge->getOutputNum();
            auto ch_node = edge->getChild();
            auto ch  = node2layer[ch_node];

            DataPtr data;
            if (i < pr->outData.size()) {
                std::string data_name = node->getName() + "_out" + std::to_string(i);
                pr->outData[i] = std::make_shared<Data>(data_name, edge->getDesc());
                data = pr->outData[i];
                data->creatorLayer = pr;
            } else {
                data = pr->outData[0];
            }

            data->inputTo[ch->name] = ch;
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
    InferenceEngine::saveGraphToDot(*dump_net, out, drawer_callback);
}

//**********************************
// Special converters of meta data
//**********************************

static std::map<Type, std::string> type_n2l {
    {Unknown, "Unknown"},
    {Generic, "Unknown"},
    {Reorder, "Reorder"},
    {Copy, "Reorder"},
    {Input, "Input"},
    {Output, "Output"},
    {Convolution, "Conv"},
    {Deconvolution, "Deconv"},
    {Convolution_Sum, "Conv_Eltw"},
    {Convolution_Activation, "Conv_Activ"},
    {Convolution_Sum_Activation, "Conv_Eltw_Activ"},
    {Activation, "Activation"},
    {Depthwise, "Depthwise"},
    {Lrn, "Lrn"},
    {Pooling, "Pool"},
    {FullyConnected, "FC"},
    {SoftMax, "SoftMax"},
    {Split, "Split"},
    {Concatenation, "Concat"},
    {Power, "Power"},
    {Eltwise, "Eltwise"},
    {Crop, "Crop"},
    {Reshape, "Reshape"},
    {Tile, "Tile"},
    {SimplerNMS, "Proposal"},
    {ROIPooling, "ROIPooling"},
    {BatchNormalization, "BatchNorm"},
    {Flatten, "Flatten"},
    {Permute, "Permute"},
    {MemoryOutput, "MemoryIn"},
    {MemoryInput, "MemoryOut"}
};

static const std::string ORIGIN_NAMES = "origin";
static const std::string IMPL_TYPE    = "impl";
static const std::string PRECISION    = "prec";
static const std::string PERF_COUNTER = "perf";

static const std::string BLUE  = "#D8D9F1";
static const std::string GREEN = "#D9EAD3";

void copy_node_metadata(const MKLDNNNodePtr &node, CNNLayer::Ptr &layer) {
    layer->type = type_n2l[node->getType()];
    layer->name = node->getName();  // Is ID

    if (node->getCnnLayer()) {
        // Original layer names
        std::vector<MKLDNNNodePtr> internal = node->getFusedWith();
        auto &merged = node->getMergeWith();
        internal.insert(internal.end(), merged.begin(), merged.end());

        std::string orig_names = node->getCnnLayer()->name;
        for (auto &sub_node : internal)
            orig_names += " " + sub_node->getCnnLayer()->name;

        layer->params[ORIGIN_NAMES] = orig_names;
    }

    // Implementation type name
    layer->params[IMPL_TYPE] = node->getPrimitiveDescriptorType();

    // Precision
    // TODO: That is not fully correct mapping type to precision.
    std::string precision = "FP32";
    auto desc = node->getSelectedPrimitiveDescriptor();
    if (desc == nullptr) {
        THROW_IE_EXCEPTION << "Internal error - descriptor is empty";
    }
    impl_desc_type impl_type = desc->getImplementationType();

    if (impl_type == gemm_blas &&
        node->getParentEdgeAt(0)->getDesc().getPrecision() == Precision::U8)  precision = "INT8";

    if (impl_type & jit && impl_type & avx512 &&
        node->getParentEdgeAt(0)->getDesc().getPrecision() == Precision::U8)  precision = "INT8";

    layer->params[PRECISION] = precision;

    // Performance
    if (node->PerfCounter().avg() != 0) {
        layer->params[PERF_COUNTER] = std::to_string(node->PerfCounter().avg())+ " mcs";
    }
}

void drawer_callback(const InferenceEngine::CNNLayerPtr layer,
        ordered_properties &printed_properties,
        ordered_properties &node_properties) {
    const auto &params = layer->params;

    // Implementation
    auto impl = params.find(IMPL_TYPE);
    if (impl != params.end()) {
        printed_properties.push_back({"impl", impl->second});
    }

    // Original names
    auto orig = params.find(ORIGIN_NAMES);
    if (orig != params.end()) {
        printed_properties.push_back({"originals", orig->second});
    }

    // Precision
    auto prec = params.find(PRECISION);
    if (prec != params.end()) {
        printed_properties.push_back({"precision", prec->second});
    }

    // Set color
    node_properties.push_back({"fillcolor", prec->second == "FP32" ? GREEN : BLUE});
}

}  // namespace MKLDNNPlugin
