// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "mkldnn_graph.h"
#include "mkldnn_graph_dumper.h"
#include "ie_blob.h"
#include "ie_util_internal.hpp"
#include "details/ie_cnn_network_tools.h"
#include "xml_net_builder.hpp"
#include "graph_tools.hpp"

#include <string>
#include <map>

using namespace InferenceEngine;
using namespace MKLDNNPlugin;
using std::string;
using std::map;

class NetGen : testing::V2NetBuilder {
    string model;
    TBlob<uint8_t>::Ptr weights;

public:
    NetGen(): testing::V2NetBuilder(buildNetworkWithOneInput(
            "SomeNet", {2,3,16,16}, "FP32")) {
        using prm_t = map<string, string>;

        testing::InOutShapes inout = {{{2,3,16,16}},{{2,16,16,16}}};

        prm_t conv_prm = {
                {"stride-x", std::to_string(1)},
                {"stride-y", std::to_string(1)},
                {"pad-x",    std::to_string(1)},
                {"pad-y",    std::to_string(1)},
                {"kernel-x", std::to_string(3)},
                {"kernel-y", std::to_string(3)},
                {"output",   std::to_string(16)},
                {"group",    std::to_string(1)}
        };
        size_t wght = 3*16*3*3*sizeof(float);
        size_t bias = 16*sizeof(float);

        prm_t relu_prm = {{"negative_slope", std::to_string(0)}};

        addLayer("Convolution", "FP32", &conv_prm, {{{2,3,16,16}},{{2,16,16,16}}}, wght, bias);
        addLayer("Relu", "FP32", &relu_prm, {{{2,16,16,16}},{{2,16,16,16}}});

        model = finish();

        weights.reset(new TBlob<uint8_t>({Precision::U8, {wght+bias}, C}));
        weights->allocate();
    }

    CNNNetwork net() {
        CNNNetReader net_reader;
        net_reader.ReadNetwork(model.data(), model.length());
        net_reader.SetWeights(weights);

        return net_reader.getNetwork();
    }
};

TEST(MKLDNNLayersTests, DumpSimpleGraph) {
    auto net = NetGen().net();
    MKLDNNGraph graph;
    MKLDNNExtensionManager::Ptr extMgr;
    graph.CreateGraph(net, extMgr);

    auto dump_net = dump_graph_as_ie_net(graph);
    auto layers = details::CNNNetSortTopologically(*dump_net);

    ASSERT_EQ(layers.size(), 4);
    ASSERT_EQ(layers[0]->type, "Input");
    ASSERT_EQ(layers[1]->type, "Convolution");
    ASSERT_EQ(layers[2]->type, "Reorder");
    ASSERT_EQ(layers[3]->type, "Output");
}

TEST(MKLDNNLayersTests, DumpSimpleGraphToDot) {
    auto net = NetGen().net();
    MKLDNNGraph graph;
    MKLDNNExtensionManager::Ptr extMgr;
    graph.CreateGraph(net, extMgr);

    std::stringstream buff;
    dump_graph_as_dot(graph, buff);

    std::string dot = buff.str();
    std::cout << dot;
    ASSERT_EQ(std::count(dot.begin(), dot.end(), '{'), 1); // 1-graph
    ASSERT_EQ(std::count(dot.begin(), dot.end(), '}'), 1);
    ASSERT_EQ(std::count(dot.begin(), dot.end(), '['), 10); // 4-node 3-data 3-shape
    ASSERT_EQ(std::count(dot.begin(), dot.end(), ']'), 10);
    ASSERT_EQ(std::count(dot.begin(), dot.end(), '>'), 6); // connection
}
