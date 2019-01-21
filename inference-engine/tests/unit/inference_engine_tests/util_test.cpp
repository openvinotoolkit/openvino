// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>

#include <initializer_list>
#include <string>
#include <utility>
#include <unordered_set>
#include <unordered_map>

#include <ie_util_internal.hpp>
#include <tests_common.hpp>
#include <graph_transformer.h>
#include "ie_utils.hpp"

namespace IE = InferenceEngine;

namespace {
class NetBuilder {
    using LayersMap = std::unordered_map<std::string, IE::CNNLayerPtr>;
    using DataMap = std::unordered_map<std::string, IE::DataPtr>;
    using InputsSet = std::unordered_set<IE::InputInfo::Ptr>;
    LayersMap _layers;
    DataMap _data;
    InputsSet _inputs;
public:
    NetBuilder() = default;
    NetBuilder(const NetBuilder&) = delete;

    template<typename... Args>
    NetBuilder& data(Args&&... args) {
        auto newData = std::make_shared<IE::Data>(std::forward<Args>(args)...);
        assert(!IE::contains(_data, newData->getName()));
        _data[newData->getName()] = newData;
        return *this;
    }

    template<typename T,typename... Args>
    NetBuilder& layer(Args&&... args) {
        auto newLayer = std::make_shared<T>(std::forward<Args>(args)...);
        assert(!IE::contains(_layers, newLayer->name));
        _layers[newLayer->name] = std::static_pointer_cast<IE::CNNLayer>(newLayer);
        return *this;
    }

    const LayersMap& getLayersMap() const {
        return _layers;
    }

    const DataMap& getDataMap() const {
        return _data;
    }

    NetBuilder& linkDataTo(const std::string& dataName,
                           const std::string& nextlayerName) {
        assert(IE::contains(_layers, nextlayerName));
        assert(IE::contains(_data,   dataName));

        auto nextlayer = _layers[nextlayerName];
        auto data = _data[dataName];

        nextlayer->insData.push_back(data);
        data->getInputTo().insert({nextlayerName, nextlayer});
        return *this;
    }

    NetBuilder& linkToData(const std::string& prevlayerName,
                           const std::string& dataName) {
        assert(IE::contains(_layers, prevlayerName));
        assert(IE::contains(_data,   dataName));

        auto prevlayer = _layers[prevlayerName];
        auto data = _data[dataName];
        assert(nullptr == data->getCreatorLayer().lock());

        prevlayer->outData.push_back(data);
        data->getCreatorLayer() = prevlayer;
        return *this;
    }

    NetBuilder& linkLayers(const std::string& prevlayerName,
                           const std::string& nextlayerName,
                           const std::string& dataName) {
        linkToData(prevlayerName, dataName);
        linkDataTo(dataName, nextlayerName);
        return *this;
    }

    NetBuilder& linkData(const std::string& prevDataName,
                         const std::string& nextDataName,
                         const std::string& layerName) {
        linkDataTo(prevDataName, layerName);
        linkToData(layerName, nextDataName);
        return *this;
    }

    template<typename... Args>
    NetBuilder& addInput(const std::string& dataName, Args&&... args) {
        assert(!dataName.empty());
        assert(IE::contains(_data, dataName));
        auto input = std::make_shared<IE::InputInfo>(
                         std::forward<Args>(args)...);
        input->setInputData(_data[dataName]);
        _inputs.insert(std::move(input));
        return *this;
    }

    IE::details::CNNNetworkImplPtr finalize() {
        auto net = std::make_shared<IE::details::CNNNetworkImpl>();

        for (auto&& it: _data) {
            auto& data = it.second;
            net->getData(it.first) = data;
            if (nullptr == data->getCreatorLayer().lock()) {
                auto input  = std::make_shared<IE::InputInfo>();
                input->setInputData(data);
                net->setInputInfo(input);
            }
        }
        for (auto&& it: _layers) {
            net->addLayer(it.second);
        }
        for (auto& i : _inputs) {
            net->setInputInfo(std::move(i));
        }

        net->resolveOutput();

        return net;
    }
};

bool checkLayers(const std::vector<IE::CNNLayerPtr>& layers, std::initializer_list<const char*> layersToCheck) {
    if (layers.size() != layersToCheck.size()) {
        return false;
    }
    for (auto&& layerToCheck: layersToCheck) {
        bool found = false;
        for (auto&& layer: layers) {
            if (layerToCheck == layer->name) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    return true;
}
}

TEST(UtilTests, contains) {
    std::unordered_set<int> temp_set;
    temp_set.insert(42);
    EXPECT_TRUE(IE::contains(temp_set, 42));
    EXPECT_FALSE(IE::contains(temp_set, 5));
}

TEST(UtilTests, groupSubraphsEmpty) {
    auto net = NetBuilder().finalize();
    auto subgraphs = IE::groupSubgraphs(*net,
                                        [&](const IE::CNNLayerPtr&,
                                            const IE::CNNLayerPtr&)
    {
        return false;
    });
    ASSERT_EQ(0, subgraphs.size());
}

TEST(UtilTests, groupSubraphsNoSplit) {
    auto net = NetBuilder()
               .data("data1",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data2",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data3",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data4",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data5",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .layer<IE::CNNLayer>(IE::LayerParams{"layer1","dummy",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer2","dummy",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer3","dummy",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer4","dummy",IE::Precision::UNSPECIFIED})
               .linkData("data1","data2","layer1")
               .linkData("data2","data3","layer2")
               .linkData("data3","data4","layer3")
               .linkData("data4","data5","layer4")
               .finalize();

    auto subgraphs = IE::groupSubgraphs(*net,
                                        [&](const IE::CNNLayerPtr&,
                                            const IE::CNNLayerPtr&)
    {
        return false;
    });
    ASSERT_EQ(1, subgraphs.size());
    ASSERT_TRUE(checkLayers(subgraphs.front(), {"layer1","layer2","layer3","layer4"}));
}

TEST(UtilTests, groupSubraphsAlwaysSplit) {
    auto net = NetBuilder()
               .data("data1",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data2",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data3",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data4",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data5",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .layer<IE::CNNLayer>(IE::LayerParams{"layer1","dummy",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer2","dummy",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer3","dummy",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer4","dummy",IE::Precision::UNSPECIFIED})
               .linkData("data1","data2","layer1")
               .linkData("data2","data3","layer2")
               .linkData("data3","data4","layer3")
               .linkData("data4","data5","layer4")
               .finalize();

    auto subgraphs = IE::groupSubgraphs(*net,
                                        [&](const IE::CNNLayerPtr&,
                                            const IE::CNNLayerPtr&)
    {
        return true;
    });
    ASSERT_EQ(4, subgraphs.size());

    for (auto&& it: net->allLayers()) {
        auto& layer = it.second;
        bool found = false;
        for (auto&& subgraphLayer: subgraphs) {
            ASSERT_EQ(1, subgraphLayer.size());
            if (layer == subgraphLayer.front()) {
                found = true;
                break;
            }
        }
        ASSERT_TRUE(found);
    }
}

TEST(UtilTests, groupSubraphsUnconnectedSubgraphs) {
    auto net = NetBuilder()
               .data("data1",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data2",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data3",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data4",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data5",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .layer<IE::CNNLayer>(IE::LayerParams{"layer1","dummy",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer2","dummy",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer3","dummy",IE::Precision::UNSPECIFIED})
               .linkData("data1","data2","layer1")
               .linkData("data3","data4","layer2")
               .linkData("data4","data5","layer3")
               .finalize();

    auto subgraphs = IE::groupSubgraphs(*net,
                                        [&](const IE::CNNLayerPtr&,
                                            const IE::CNNLayerPtr&)
    {
        return false;
    });
    ASSERT_EQ(2, subgraphs.size());

    for (auto&& subgraph: subgraphs) {
        if (1 == subgraph.size()) {
            ASSERT_TRUE(checkLayers(subgraph, {"layer1"}));
        }
        else if (2 == subgraph.size()) {
            ASSERT_TRUE(checkLayers(subgraph, {"layer2","layer3"}));
        }
        else {
            FAIL();
        }
    }
}

TEST(UtilTests, groupSubraphsLinear) {
    auto net = NetBuilder()
               .data("data1",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data2",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data3",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data4",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .layer<IE::CNNLayer>(IE::LayerParams{"layer1","dummy",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer2","dummy",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer3","dummy",IE::Precision::UNSPECIFIED})
               .linkData("data1","data2","layer1")
               .linkData("data2","data3","layer2")
               .linkData("data3","data4","layer3")
               .finalize();

    auto subgraphs = IE::groupSubgraphs(*net,
                                        [&](const IE::CNNLayerPtr& layer1,
                                            const IE::CNNLayerPtr& layer2)
    {
        if ("layer1" == layer1->name) {
            EXPECT_EQ("layer2", layer2->name);
            return true;
        }
        if ("layer2" == layer1->name) {
            EXPECT_EQ("layer3", layer2->name);
            return false;
        }
        else {
            ADD_FAILURE();
            return false;
        }
    });
    ASSERT_EQ(2, subgraphs.size());

    for (auto&& subgraph: subgraphs) {
        if (1 == subgraph.size()) {
            ASSERT_TRUE(checkLayers(subgraph, {"layer1"}));
        }
        else if (2 == subgraph.size()) {
            ASSERT_TRUE(checkLayers(subgraph, {"layer2","layer3"}));
        }
        else {
            FAIL();
        }
    }
}

TEST(UtilTests, groupSubraphsDeadBranch) {
    //
    // L1->L2->L3->L4->L5
    //      \      /
    //      L6 -> L7
    //
    // Split between L2 - L6 and L7 - L4
    //
    // Subgraphs:
    // L1, L2, L3, L4, L5
    // L6, L7
    auto net = NetBuilder()
               .data("data1",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data2",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data3",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data4",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data5",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data6",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data7",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data8",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data9",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .layer<IE::CNNLayer>(IE::LayerParams{"layer1","dummy",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer2","dummy",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer3","dummy",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer4","dummy",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer5","dummy",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer6","dummy",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer7","dummy",IE::Precision::UNSPECIFIED})
               .linkData("data1","data2","layer1")
               .linkData("data2","data3","layer2")
               .linkData("data3","data4","layer3")
               .linkData("data4","data5","layer4")
               .linkData("data5","data6","layer5")

               .linkLayers("layer2", "layer6", "data7")
               .linkLayers("layer6", "layer7", "data8")
               .linkLayers("layer7", "layer4", "data9")

               .finalize();

    auto subgraphs = IE::groupSubgraphs(*net,
                                        [&](const IE::CNNLayerPtr& layer1,
                                            const IE::CNNLayerPtr& layer2)
    {
        if ("layer2" == layer1->name) {
            if ("layer3" == layer2->name) {
                return false;
            }
            else if ("layer6" == layer2->name) {
                return true;
            }
            else {
                ADD_FAILURE();
            }
        }

        if ("layer7" == layer1->name) {
            EXPECT_EQ("layer4", layer2->name);
            return true;
        }
        return false;
    });
    ASSERT_EQ(2, subgraphs.size());

    for (auto&& subgraph: subgraphs) {
        if (5 == subgraph.size()) {
            ASSERT_TRUE(checkLayers(subgraph, {"layer1","layer2","layer3","layer4","layer5"}));
        }
        else if (2 == subgraph.size()) {
            ASSERT_TRUE(checkLayers(subgraph, {"layer6","layer7"}));
        }
        else {
            FAIL();
        }
    }
}

TEST(UtilTests, cloneData) {
    IE::Data srcData("test",IE::SizeVector{1,2,3},IE::Precision::FP32,IE::CHW);
    IE::CNNLayerPtr layer1 = std::make_shared<IE::CNNLayer>(IE::LayerParams{});
    IE::CNNLayerPtr layer2 = std::make_shared<IE::CNNLayer>(IE::LayerParams{});
    srcData.getCreatorLayer() = layer1;
    srcData.getInputTo().insert({"foo",layer2});
    auto cloned = IE::cloneData(srcData);
    ASSERT_NE(nullptr, cloned);
    EXPECT_EQ(srcData.getName(), cloned->getName());
    EXPECT_EQ(srcData.getPrecision(), cloned->getPrecision());
    EXPECT_EQ(srcData.getLayout(), cloned->getLayout());
    EXPECT_EQ(srcData.getDims(), cloned->getDims());
    EXPECT_EQ(nullptr, cloned->getCreatorLayer().lock());
    EXPECT_TRUE(cloned->getInputTo().empty());
}

namespace {
template<typename T>
bool checkLayerCloning() {
    T srcLayer(IE::LayerParams{"layer","dummy",IE::Precision::FP32});
    auto cloned = IE::clonelayer(srcLayer);
    return nullptr != std::dynamic_pointer_cast<T>(cloned);
}
}

TEST(UtilTests, cloneLayers) {
    {
        IE::CNNLayer srclayer(IE::LayerParams{"layer","dummy",IE::Precision::FP32});
        auto data1 = std::make_shared<IE::Data>("data1",IE::Precision::FP32);
        auto data2 = std::make_shared<IE::Data>("data1",IE::Precision::FP32);
        srclayer.insData.push_back(data1);
        srclayer.outData.push_back(data2);
        int dummy = 123;
        srclayer.userValue.v_ptr = &dummy;
        srclayer.params["foo"] = "1";
        srclayer.params["bar"] = "2";
        auto blob = std::make_shared<IE::TBlob<float>>(IE::Precision::FP32, IE::NCHW);
        srclayer.blobs["baz"] = blob;
        auto cloned = IE::clonelayer(srclayer);
        ASSERT_NE(nullptr, cloned);
        EXPECT_EQ(srclayer.name, cloned->name);
        EXPECT_EQ(srclayer.type, cloned->type);
        EXPECT_EQ(srclayer.precision, cloned->precision);
        EXPECT_EQ(srclayer.userValue.v_ptr, cloned->userValue.v_ptr);
        EXPECT_EQ(srclayer.params, cloned->params);
        EXPECT_EQ(srclayer.blobs, cloned->blobs);
        EXPECT_EQ(0, cloned->insData.size());
        EXPECT_EQ(0, cloned->outData.size());
    }
    EXPECT_TRUE(checkLayerCloning<IE::BatchNormalizationLayer>());
    EXPECT_TRUE(checkLayerCloning<IE::PowerLayer             >());
    EXPECT_TRUE(checkLayerCloning<IE::ScaleShiftLayer        >());
    EXPECT_TRUE(checkLayerCloning<IE::TileLayer              >());
    EXPECT_TRUE(checkLayerCloning<IE::ReshapeLayer           >());
    EXPECT_TRUE(checkLayerCloning<IE::CropLayer              >());
    EXPECT_TRUE(checkLayerCloning<IE::EltwiseLayer           >());
    EXPECT_TRUE(checkLayerCloning<IE::ClampLayer             >());
    EXPECT_TRUE(checkLayerCloning<IE::ReLULayer              >());
    EXPECT_TRUE(checkLayerCloning<IE::SoftMaxLayer           >());
    EXPECT_TRUE(checkLayerCloning<IE::GRNLayer               >());
    EXPECT_TRUE(checkLayerCloning<IE::NormLayer              >());
    EXPECT_TRUE(checkLayerCloning<IE::SplitLayer             >());
    EXPECT_TRUE(checkLayerCloning<IE::ConcatLayer            >());
    EXPECT_TRUE(checkLayerCloning<IE::FullyConnectedLayer    >());
    EXPECT_TRUE(checkLayerCloning<IE::PoolingLayer           >());
    EXPECT_TRUE(checkLayerCloning<IE::DeconvolutionLayer     >());
    EXPECT_TRUE(checkLayerCloning<IE::ConvolutionLayer       >());
    EXPECT_TRUE(checkLayerCloning<IE::WeightableLayer        >());
    EXPECT_TRUE(checkLayerCloning<IE::CNNLayer               >());
}

namespace {
IE::CNNLayerPtr getLayer(const IE::details::CNNNetworkImplPtr n,
                         const char* name) {
    if (IE::contains(n->allLayers(), name)) {
        return n->allLayers().find(name)->second;
    }
    return nullptr;
};
} // namespace

TEST(UtilTests, cloneNet) {
    //
    //      I      O
    //       \    /
    // I-L1->L2->L3->L4->L5->O
    //        \      /
    //        L6 -> L7
    //
    auto net = NetBuilder()
                // input is 4d to allow setting of batch, otherwise, CHW notation doesnt have batches
               .data("data1",IE::SizeVector{1,1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::NCHW)
               .data("data2",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data3",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data4",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data5",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data6",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data7",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data8",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data9",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data10",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data11",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .layer<IE::CNNLayer>(IE::LayerParams{"layer1","dummy",IE::Precision::Q78})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer2","dummy",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer3","dummy",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer4","dummy",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer5","dummy",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer6","dummy",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer7","dummy",IE::Precision::UNSPECIFIED})
               .linkData("data1","data2","layer1")
               .linkData("data2","data3","layer2")
               .linkData("data3","data4","layer3")
               .linkData("data4","data5","layer4")
               .linkData("data5","data6","layer5")

               .linkLayers("layer2", "layer6", "data7")
               .linkLayers("layer6", "layer7", "data8")
               .linkLayers("layer7", "layer4", "data9")

               .linkDataTo("data10","layer2")
               .linkToData("layer3","data11")

               .finalize();

    net->setPrecision(IE::Precision::Q78);
    net->setBatchSize(42);
    net->setName("net");
    net->setTargetDevice(IE::TargetDevice::eHETERO);

    {
        IE::InputsDataMap inputs;
        net->getInputsInfo(inputs);
        for (auto &&it : inputs) {
            it.second->getPreProcess().init(1);
            it.second->getPreProcess().setMeanImage(IE::make_shared_blob<float>(IE::Precision::FP32, IE::Layout::CHW, {1,1,1}));
            it.second->getPreProcess().setResizeAlgorithm(IE::ResizeAlgorithm::RESIZE_BILINEAR);
        }
    }

    {
        auto layer = getLayer(net, "layer1");
        auto cloned = IE::cloneNet({layer});
        EXPECT_EQ(2, cloned->layerCount());
        auto clonedLayer = getLayer(cloned, "layer1");
        ASSERT_NE(nullptr, clonedLayer);
        EXPECT_EQ(layer->type, clonedLayer->type);

        IE::InputsDataMap inputs;
        IE::OutputsDataMap outputs;
        cloned->getInputsInfo(inputs);
        cloned->getOutputsInfo(outputs);
        ASSERT_EQ(1, inputs.size());
        ASSERT_EQ(1, outputs.size());
        EXPECT_TRUE(IE::contains(inputs,"data1"));
        EXPECT_TRUE(IE::contains(outputs,"data2"));
    }
    {
        auto layer1 = getLayer(net, "layer1");
        auto layer2 = getLayer(net, "layer2");
        auto cloned = IE::cloneNet({layer1,layer2});
        EXPECT_EQ(4, cloned->layerCount());
        auto clonedLayer1 = getLayer(cloned, "layer1");
        auto clonedLayer2 = getLayer(cloned, "layer2");
        ASSERT_NE(nullptr, clonedLayer1);
        ASSERT_NE(nullptr, clonedLayer2);

        IE::InputsDataMap inputs;
        IE::OutputsDataMap outputs;
        cloned->getInputsInfo(inputs);
        cloned->getOutputsInfo(outputs);
        ASSERT_EQ(2, inputs.size());
        ASSERT_EQ(2, outputs.size());
        EXPECT_TRUE(IE::contains(inputs,"data1"));
        EXPECT_TRUE(IE::contains(inputs,"data10"));
        EXPECT_TRUE(IE::contains(outputs,"data3"));
        EXPECT_TRUE(IE::contains(outputs,"data7"));
    }
    {
        auto layer4 = getLayer(net, "layer4");
        auto layer5 = getLayer(net, "layer5");
        auto cloned = IE::cloneNet({layer4,layer5});
        EXPECT_EQ(4, cloned->layerCount());
        auto clonedLayer4 = getLayer(cloned, "layer4");
        auto clonedLayer5 = getLayer(cloned, "layer5");
        ASSERT_NE(nullptr, clonedLayer4);
        ASSERT_NE(nullptr, clonedLayer5);

        ASSERT_EQ(2, clonedLayer4->insData.size());
        ASSERT_EQ(1, clonedLayer4->outData.size());

        EXPECT_EQ("data4", clonedLayer4->insData[0].lock()->getName());
        EXPECT_EQ("data9", clonedLayer4->insData[1].lock()->getName());
        EXPECT_EQ("data5", clonedLayer4->outData[0]->getName());

        ASSERT_EQ(1, clonedLayer5->insData.size());
        ASSERT_EQ(1, clonedLayer5->outData.size());

        EXPECT_EQ("data5", clonedLayer5->insData[0].lock()->getName());
        EXPECT_EQ("data6", clonedLayer5->outData[0]->getName());

        IE::InputsDataMap inputs;
        IE::OutputsDataMap outputs;
        cloned->getInputsInfo(inputs);
        cloned->getOutputsInfo(outputs);
        ASSERT_EQ(2, inputs.size());
        ASSERT_EQ(1, outputs.size());
        EXPECT_TRUE(IE::contains(inputs,"data4"));
        EXPECT_TRUE(IE::contains(inputs,"data9"));
        EXPECT_TRUE(IE::contains(outputs,"data6"));
    }
    {
        auto layer3 = getLayer(net, "layer3");
        auto cloned = IE::cloneNet({layer3});
        EXPECT_EQ(2, cloned->layerCount());
        auto clonedLayer3 = getLayer(cloned, "layer3");
        ASSERT_NE(nullptr, clonedLayer3);

        ASSERT_EQ(1, clonedLayer3->insData.size());
        ASSERT_EQ(2, clonedLayer3->outData.size());

        EXPECT_EQ("data3", clonedLayer3->insData[0].lock()->getName());
        EXPECT_EQ("data4", clonedLayer3->outData[0]->getName());
        EXPECT_EQ("data11", clonedLayer3->outData[1]->getName());

        IE::InputsDataMap inputs;
        IE::OutputsDataMap outputs;
        cloned->getInputsInfo(inputs);
        cloned->getOutputsInfo(outputs);
        ASSERT_EQ(1, inputs.size());
        ASSERT_EQ(2, outputs.size());
        EXPECT_TRUE(IE::contains(inputs,"data3"));
        EXPECT_TRUE(IE::contains(outputs,"data4"));
        EXPECT_TRUE(IE::contains(outputs,"data11"));
    }
    {
        auto layer1 = getLayer(net, "layer1");
        auto layer2 = getLayer(net, "layer2");
        auto layer3 = getLayer(net, "layer3");
        auto layer4 = getLayer(net, "layer4");
        auto layer5 = getLayer(net, "layer5");
        auto layer6 = getLayer(net, "layer6");
        auto layer7 = getLayer(net, "layer7");
        auto cloned = IE::cloneNet({layer1,layer2,layer3,layer4,layer5,layer6,layer7});
        EXPECT_EQ(9, cloned->layerCount());
        auto clonedLayer1 = getLayer(cloned, "layer1");
        auto clonedLayer2 = getLayer(cloned, "layer2");
        auto clonedLayer3 = getLayer(cloned, "layer3");
        auto clonedLayer4 = getLayer(cloned, "layer4");
        auto clonedLayer5 = getLayer(cloned, "layer5");
        auto clonedLayer6 = getLayer(cloned, "layer6");
        auto clonedLayer7 = getLayer(cloned, "layer7");
        ASSERT_NE(nullptr, clonedLayer1);
        ASSERT_NE(nullptr, clonedLayer2);
        ASSERT_NE(nullptr, clonedLayer3);
        ASSERT_NE(nullptr, clonedLayer4);
        ASSERT_NE(nullptr, clonedLayer5);
        ASSERT_NE(nullptr, clonedLayer6);
        ASSERT_NE(nullptr, clonedLayer7);

        ASSERT_EQ(1, clonedLayer1->insData.size());
        ASSERT_EQ(1, clonedLayer1->outData.size());
        EXPECT_EQ("data1", clonedLayer1->insData[0].lock()->getName());
        EXPECT_EQ("data2", clonedLayer1->outData[0]->getName());

        ASSERT_EQ(2, clonedLayer2->insData.size());
        ASSERT_EQ(2, clonedLayer2->outData.size());
        EXPECT_EQ("data2", clonedLayer2->insData[0].lock()->getName());
        EXPECT_EQ("data10", clonedLayer2->insData[1].lock()->getName());
        EXPECT_EQ("data3", clonedLayer2->outData[0]->getName());
        EXPECT_EQ("data7", clonedLayer2->outData[1]->getName());

        ASSERT_EQ(1, clonedLayer3->insData.size());
        ASSERT_EQ(2, clonedLayer3->outData.size());
        EXPECT_EQ("data3", clonedLayer3->insData[0].lock()->getName());
        EXPECT_EQ("data4", clonedLayer3->outData[0]->getName());
        EXPECT_EQ("data11", clonedLayer3->outData[1]->getName());

        ASSERT_EQ(2, clonedLayer4->insData.size());
        ASSERT_EQ(1, clonedLayer4->outData.size());
        EXPECT_EQ("data4", clonedLayer4->insData[0].lock()->getName());
        EXPECT_EQ("data9", clonedLayer4->insData[1].lock()->getName());
        EXPECT_EQ("data5", clonedLayer4->outData[0]->getName());

        ASSERT_EQ(1, clonedLayer5->insData.size());
        ASSERT_EQ(1, clonedLayer5->outData.size());
        EXPECT_EQ("data5", clonedLayer5->insData[0].lock()->getName());
        EXPECT_EQ("data6", clonedLayer5->outData[0]->getName());

        ASSERT_EQ(1, clonedLayer6->insData.size());
        ASSERT_EQ(1, clonedLayer6->outData.size());
        EXPECT_EQ("data7", clonedLayer6->insData[0].lock()->getName());
        EXPECT_EQ("data8", clonedLayer6->outData[0]->getName());

        ASSERT_EQ(1, clonedLayer7->insData.size());
        ASSERT_EQ(1, clonedLayer7->outData.size());
        EXPECT_EQ("data8", clonedLayer7->insData[0].lock()->getName());
        EXPECT_EQ("data9", clonedLayer7->outData[0]->getName());

        IE::InputsDataMap inputs;
        IE::OutputsDataMap outputs;
        cloned->getInputsInfo(inputs);
        cloned->getOutputsInfo(outputs);
        ASSERT_EQ(2, inputs.size());
        ASSERT_EQ(2, outputs.size());
        EXPECT_TRUE(IE::contains(inputs,"data1"));
        EXPECT_TRUE(IE::contains(inputs,"data10"));
        EXPECT_TRUE(IE::contains(outputs,"data11"));
        EXPECT_TRUE(IE::contains(outputs,"data6"));
    }
    {
        auto cloned = IE::cloneNet(*net);
        auto layer1 = getLayer(cloned, "layer1");
        auto layer2 = getLayer(cloned, "layer2");
        EXPECT_TRUE(IE::Precision::Q78          == layer1->precision);
        EXPECT_TRUE(IE::Precision::UNSPECIFIED  == layer2->precision);
    }
    {
        auto cloned = IE::cloneNet(*net);
        EXPECT_TRUE(IE::Precision::Q78        == cloned->getPrecision());
        EXPECT_EQ(42,                            cloned->getBatchSize());
        EXPECT_EQ("net",                         cloned->getName());
        EXPECT_TRUE(IE::TargetDevice::eHETERO == cloned->getTargetDevice());
    }
    {
        auto cloned = IE::cloneNet(*net);
        IE::InputsDataMap clonedInputs;
        cloned->getInputsInfo(clonedInputs);
        for (auto &&clonedInput : clonedInputs) {
           EXPECT_EQ(1,                                        clonedInput.second->getPreProcess().getNumberOfChannels());
           EXPECT_TRUE(IE::MeanVariant::MEAN_IMAGE          == clonedInput.second->getPreProcess().getMeanVariant());
           EXPECT_TRUE(IE::ResizeAlgorithm::RESIZE_BILINEAR == clonedInput.second->getPreProcess().getResizeAlgorithm());
        }
    }
}

TEST(UtilTests, cloneNet_input) {
    //
    // I1-d1-L1
    //
    // I2-d2
    //      \
    //       L2
    //      /
    // I3-d3
    //      \
    //       L3
    //
    auto net = NetBuilder()
               .data("data1",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data2",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data3",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .layer<IE::CNNLayer>(IE::LayerParams{"input1","input",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"input2","Input",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"input3","input",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer1","dummy",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer2","dummy",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer3","dummy",IE::Precision::UNSPECIFIED})

               .linkToData("input1", "data1")
               .linkToData("input2", "data2")
               .linkToData("input3", "data3")

               .linkDataTo("data1", "layer1")
               .linkDataTo("data2", "layer2")
               .linkDataTo("data3", "layer2")
               .linkDataTo("data3", "layer3")

               .finalize();

    getLayer(net, "input1")->params["custom_param1"] = "custom_val1";
    getLayer(net, "input2")->params["custom_param2"] = "custom_val2";
    getLayer(net, "input3")->params["custom_param3"] = "custom_val3";

    auto cloned = IE::cloneNet({getLayer(net, "layer1"),
                                getLayer(net, "layer2"),
                                getLayer(net, "layer3")});

    ASSERT_EQ(6, cloned->layerCount());
    ASSERT_NE(nullptr, getLayer(cloned, "input1"));
    ASSERT_NE(nullptr, getLayer(cloned, "input2"));
    ASSERT_NE(nullptr, getLayer(cloned, "input3"));
    ASSERT_EQ("input", getLayer(cloned, "input1")->type);
    ASSERT_EQ("Input", getLayer(cloned, "input2")->type);
    ASSERT_EQ("input", getLayer(cloned, "input3")->type);
    ASSERT_EQ("custom_val1", getLayer(cloned, "input1")->params["custom_param1"]);
    ASSERT_EQ("custom_val2", getLayer(cloned, "input2")->params["custom_param2"]);
    ASSERT_EQ("custom_val3", getLayer(cloned, "input3")->params["custom_param3"]);
}

TEST(UtilTests, cloneNet_const) {
    //
    // C1-d1-L1
    //
    // C2-d2
    //      \
    //       L2
    //      /
    // C3-d3
    //      \
    //       L3
    //
    auto net = NetBuilder()
               .data("data1",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data2",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data3",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .layer<IE::CNNLayer>(IE::LayerParams{"input1","const",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"input2","Const",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"input3","const",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer1","dummy",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer2","dummy",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer3","dummy",IE::Precision::UNSPECIFIED})

               .linkToData("input1", "data1")
               .linkToData("input2", "data2")
               .linkToData("input3", "data3")

               .linkDataTo("data1", "layer1")
               .linkDataTo("data2", "layer2")
               .linkDataTo("data3", "layer2")
               .linkDataTo("data3", "layer3")

               .finalize();

    getLayer(net, "input1")->params["custom_param1"] = "custom_val1";
    getLayer(net, "input2")->params["custom_param2"] = "custom_val2";
    getLayer(net, "input3")->params["custom_param3"] = "custom_val3";

    auto cloned = IE::cloneNet({getLayer(net, "layer1"),
                                getLayer(net, "layer2"),
                                getLayer(net, "layer3")});

    ASSERT_EQ(6, cloned->layerCount());
    ASSERT_NE(nullptr, getLayer(cloned, "input1"));
    ASSERT_NE(nullptr, getLayer(cloned, "input2"));
    ASSERT_NE(nullptr, getLayer(cloned, "input3"));
    ASSERT_EQ("const", getLayer(cloned, "input1")->type);
    ASSERT_EQ("Const", getLayer(cloned, "input2")->type);
    ASSERT_EQ("const", getLayer(cloned, "input3")->type);
    ASSERT_EQ("custom_val1", getLayer(cloned, "input1")->params["custom_param1"]);
    ASSERT_EQ("custom_val2", getLayer(cloned, "input2")->params["custom_param2"]);
    ASSERT_EQ("custom_val3", getLayer(cloned, "input3")->params["custom_param3"]);
}

TEST(UtilTests, getRootDataObjects) {
    //
    // I1-d1-L1-d7
    //            \
    // I2-d2       L6
    //      \     /
    //       L2-d8
    //      /
    // I3-d3
    //      \
    //       L3
    //      /
    // C1-d4
    //      \
    //       L4-d9
    //      /     \
    // C2-d5       L7
    //            /
    // C3-d6-L5-d10
    auto net = NetBuilder()
               .data("data1",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data2",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data3",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data4",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data5",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data6",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data7",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data8",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data9",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data10",IE::SizeVector{1,1,1},IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .layer<IE::CNNLayer>(IE::LayerParams{"input1","input",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"input2","Input",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"input3","input",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"const1","const",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"const2","Const",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"const3","const",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer1","dummy",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer2","dummy",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer3","dummy",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer4","dummy",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer5","dummy",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer6","dummy",IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer7","dummy",IE::Precision::UNSPECIFIED})

               .linkToData("input1", "data1")
               .linkToData("input2", "data2")
               .linkToData("input3", "data3")

               .linkToData("const1", "data4")
               .linkToData("const2", "data5")
               .linkToData("const3", "data6")

               .linkDataTo("data1", "layer1")
               .linkDataTo("data2", "layer2")
               .linkDataTo("data3", "layer2")
               .linkDataTo("data3", "layer3")

               .linkDataTo("data4", "layer3")
               .linkDataTo("data4", "layer4")
               .linkDataTo("data5", "layer4")
               .linkDataTo("data6", "layer5")

               .linkToData("layer1", "data7")
               .linkToData("layer2", "data8")
               .linkToData("layer4", "data9")
               .linkToData("layer5", "data10")

               .linkDataTo("data7", "layer6")
               .linkDataTo("data8", "layer6")

               .linkDataTo("data9", "layer7")
               .linkDataTo("data10", "layer7")

               .addInput("data1")
               .addInput("data2")
               .addInput("data3")

               .finalize();

    auto cloned = IE::cloneNet(*net);

    ASSERT_EQ(13, cloned->layerCount());
    auto root_data = IE::getRootDataObjects(*cloned);
    ASSERT_EQ(6, root_data.size());
    std::unordered_set<std::string> data_names;
    for (auto& data : root_data) {
        data_names.insert(data->name);
    }
    ASSERT_TRUE(IE::contains(data_names, "data1"));
    ASSERT_TRUE(IE::contains(data_names, "data2"));
    ASSERT_TRUE(IE::contains(data_names, "data3"));
    ASSERT_TRUE(IE::contains(data_names, "data4"));
    ASSERT_TRUE(IE::contains(data_names, "data5"));
    ASSERT_TRUE(IE::contains(data_names, "data6"));
}

TEST(UtilTests, networkComplexity) {
    std::string model =
            "<net name=\"Top\" version=\"2\" batch=\"1\">"
            "    <layers>"
            "        <layer name=\"data\" type=\"Input\" precision=\"FP32\" id=\"0\">"
            "            <output>"
            "                <port id=\"0\">"
            "                    <dim>1</dim>"
            "                    <dim>3</dim>"
            "                    <dim>227</dim>"
            "                    <dim>227</dim>"
            "                </port>"
            "            </output>"
            "        </layer>"
            "        <layer name=\"conv1\" type=\"Convolution\" precision=\"FP32\" id=\"1\">"
            "            <convolution_data stride-x=\"4\" stride-y=\"4\" pad-x=\"0\" pad-y=\"0\" kernel-x=\"11\" kernel-y=\"11\" output=\"96\" group=\"1\"/>"
            "            <input>"
            "                <port id=\"1\">"
            "                    <dim>1</dim>"
            "                    <dim>3</dim>"
            "                    <dim>227</dim>"
            "                    <dim>227</dim>"
            "                </port>"
            "            </input>"
            "            <output>"
            "                <port id=\"2\">"
            "                    <dim>1</dim>"
            "                    <dim>96</dim>"
            "                    <dim>55</dim>"
            "                    <dim>55</dim>"
            "                </port>"
            "            </output>"
            "            <weights offset=\"0\" size=\"139392\"/>"
            "            <biases offset=\"139392\" size=\"384\"/>"
            "        </layer>"
            "        <layer name=\"relu1\" type=\"ReLU\" precision=\"FP32\" id=\"2\">"
            "            <input>"
            "                <port id=\"3\">"
            "                    <dim>1</dim>"
            "                    <dim>96</dim>"
            "                    <dim>55</dim>"
            "                    <dim>55</dim>"
            "                </port>"
            "            </input>"
            "            <output>"
            "                <port id=\"4\">"
            "                    <dim>1</dim>"
            "                    <dim>96</dim>"
            "                    <dim>55</dim>"
            "                    <dim>55</dim>"
            "                </port>"
            "            </output>"
            "        </layer>"
            "        <layer name=\"norm1\" type=\"Norm\" precision=\"FP32\" id=\"3\">"
            "            <norm_data alpha=\"9.9999997e-05\" beta=\"0.75\" local-size=\"5\" region=\"across\"/>"
            "            <input>"
            "                <port id=\"5\">"
            "                    <dim>1</dim>"
            "                    <dim>96</dim>"
            "                    <dim>55</dim>"
            "                    <dim>55</dim>"
            "                </port>"
            "            </input>"
            "            <output>"
            "                <port id=\"6\">"
            "                    <dim>1</dim>"
            "                    <dim>96</dim>"
            "                    <dim>55</dim>"
            "                    <dim>55</dim>"
            "                </port>"
            "            </output>"
            "        </layer>"
            "        <layer name=\"pool1\" type=\"Pooling\" precision=\"FP32\" id=\"4\">"
            "            <pooling_data kernel-x=\"3\" kernel-y=\"3\" pad-x=\"0\" pad-y=\"0\" stride-x=\"2\" stride-y=\"2\" rounding-type=\"ceil\" pool-method=\"max\"/>"
            "            <input>"
            "                <port id=\"7\">"
            "                    <dim>1</dim>"
            "                    <dim>96</dim>"
            "                    <dim>55</dim>"
            "                    <dim>55</dim>"
            "                </port>"
            "            </input>"
            "            <output>"
            "                <port id=\"8\">"
            "                    <dim>1</dim>"
            "                    <dim>96</dim>"
            "                    <dim>27</dim>"
            "                    <dim>27</dim>"
            "                </port>"
            "            </output>"
            "        </layer>"
            "        <layer name=\"conv2_split\" type=\"Split\" precision=\"FP32\" id=\"24\">"
            "            <input>"
            "                <port id=\"47\">"
            "                    <dim>1</dim>"
            "                    <dim>96</dim>"
            "                    <dim>27</dim>"
            "                    <dim>27</dim>"
            "                </port>"
            "            </input>"
            "            <output>"
            "                <port id=\"49\">"
            "                    <dim>1</dim>"
            "                    <dim>48</dim>"
            "                    <dim>27</dim>"
            "                    <dim>27</dim>"
            "                </port>"
            "                <port id=\"53\">"
            "                    <dim>1</dim>"
            "                    <dim>48</dim>"
            "                    <dim>27</dim>"
            "                    <dim>27</dim>"
            "                </port>"
            "            </output>"
            "        </layer>"
            "        <layer name=\"conv2_1\" type=\"Convolution\" precision=\"FP32\" id=\"27\">"
            "            <convolution_data stride-x=\"1\" stride-y=\"1\" pad-x=\"2\" pad-y=\"2\" kernel-x=\"5\" kernel-y=\"5\" output=\"128\" group=\"1\"/>"
            "            <input>"
            "                <port id=\"54\">"
            "                    <dim>1</dim>"
            "                    <dim>48</dim>"
            "                    <dim>27</dim>"
            "                    <dim>27</dim>"
            "                </port>"
            "            </input>"
            "            <output>"
            "                <port id=\"55\">"
            "                    <dim>1</dim>"
            "                    <dim>128</dim>"
            "                    <dim>27</dim>"
            "                    <dim>27</dim>"
            "                </port>"
            "            </output>"
            "            <weights offset=\"139776\" size=\"614400\"/>"
            "            <biases offset=\"754176\" size=\"512\"/>"
            "        </layer>"
            "        <layer name=\"conv2_0\" type=\"Convolution\" precision=\"FP32\" id=\"26\">"
            "            <convolution_data stride-x=\"1\" stride-y=\"1\" pad-x=\"2\" pad-y=\"2\" kernel-x=\"5\" kernel-y=\"5\" output=\"128\" group=\"1\"/>"
            "            <input>"
            "                <port id=\"50\">"
            "                    <dim>1</dim>"
            "                    <dim>48</dim>"
            "                    <dim>27</dim>"
            "                    <dim>27</dim>"
            "                </port>"
            "            </input>"
            "            <output>"
            "                <port id=\"51\">"
            "                    <dim>1</dim>"
            "                    <dim>128</dim>"
            "                    <dim>27</dim>"
            "                    <dim>27</dim>"
            "                </port>"
            "            </output>"
            "            <weights offset=\"754688\" size=\"614400\"/>"
            "            <biases offset=\"1369088\" size=\"512\"/>"
            "        </layer>"
            "        <layer name=\"conv2_merge\" type=\"Concat\" precision=\"FP32\" id=\"25\">"
            "            <concat_data axis=\"1\"/>"
            "            <input>"
            "                <port id=\"52\">"
            "                    <dim>1</dim>"
            "                    <dim>128</dim>"
            "                    <dim>27</dim>"
            "                    <dim>27</dim>"
            "                </port>"
            "                <port id=\"56\">"
            "                    <dim>1</dim>"
            "                    <dim>128</dim>"
            "                    <dim>27</dim>"
            "                    <dim>27</dim>"
            "                </port>"
            "            </input>"
            "            <output>"
            "                <port id=\"48\">"
            "                    <dim>1</dim>"
            "                    <dim>256</dim>"
            "                    <dim>27</dim>"
            "                    <dim>27</dim>"
            "                </port>"
            "            </output>"
            "        </layer>"
            "        <layer name=\"relu2\" type=\"ReLU\" precision=\"FP32\" id=\"6\">"
            "            <input>"
            "                <port id=\"11\">"
            "                    <dim>1</dim>"
            "                    <dim>256</dim>"
            "                    <dim>27</dim>"
            "                    <dim>27</dim>"
            "                </port>"
            "            </input>"
            "            <output>"
            "                <port id=\"12\">"
            "                    <dim>1</dim>"
            "                    <dim>256</dim>"
            "                    <dim>27</dim>"
            "                    <dim>27</dim>"
            "                </port>"
            "            </output>"
            "        </layer>"
            "        <layer name=\"norm2\" type=\"Norm\" precision=\"FP32\" id=\"7\">"
            "            <norm_data alpha=\"9.9999997e-05\" beta=\"0.75\" local-size=\"5\" region=\"across\"/>"
            "            <input>"
            "                <port id=\"13\">"
            "                    <dim>1</dim>"
            "                    <dim>256</dim>"
            "                    <dim>27</dim>"
            "                    <dim>27</dim>"
            "                </port>"
            "            </input>"
            "            <output>"
            "                <port id=\"14\">"
            "                    <dim>1</dim>"
            "                    <dim>256</dim>"
            "                    <dim>27</dim>"
            "                    <dim>27</dim>"
            "                </port>"
            "            </output>"
            "        </layer>"
            "        <layer name=\"pool2\" type=\"Pooling\" precision=\"FP32\" id=\"8\">"
            "            <pooling_data kernel-x=\"3\" kernel-y=\"3\" pad-x=\"0\" pad-y=\"0\" stride-x=\"2\" stride-y=\"2\" rounding-type=\"ceil\" pool-method=\"max\"/>"
            "            <input>"
            "                <port id=\"15\">"
            "                    <dim>1</dim>"
            "                    <dim>256</dim>"
            "                    <dim>27</dim>"
            "                    <dim>27</dim>"
            "                </port>"
            "            </input>"
            "            <output>"
            "                <port id=\"16\">"
            "                    <dim>1</dim>"
            "                    <dim>256</dim>"
            "                    <dim>13</dim>"
            "                    <dim>13</dim>"
            "                </port>"
            "            </output>"
            "        </layer>"
            "        <layer name=\"conv3\" type=\"Convolution\" precision=\"FP32\" id=\"9\">"
            "            <convolution_data stride-x=\"1\" stride-y=\"1\" pad-x=\"1\" pad-y=\"1\" kernel-x=\"3\" kernel-y=\"3\" output=\"384\" group=\"1\"/>"
            "            <input>"
            "                <port id=\"17\">"
            "                    <dim>1</dim>"
            "                    <dim>256</dim>"
            "                    <dim>13</dim>"
            "                    <dim>13</dim>"
            "                </port>"
            "            </input>"
            "            <output>"
            "                <port id=\"18\">"
            "                    <dim>1</dim>"
            "                    <dim>384</dim>"
            "                    <dim>13</dim>"
            "                    <dim>13</dim>"
            "                </port>"
            "            </output>"
            "            <weights offset=\"1369600\" size=\"3538944\"/>"
            "            <biases offset=\"4908544\" size=\"1536\"/>"
            "        </layer>"
            "        <layer name=\"relu3\" type=\"ReLU\" precision=\"FP32\" id=\"10\">"
            "            <input>"
            "                <port id=\"19\">"
            "                    <dim>1</dim>"
            "                    <dim>384</dim>"
            "                    <dim>13</dim>"
            "                    <dim>13</dim>"
            "                </port>"
            "            </input>"
            "            <output>"
            "                <port id=\"20\">"
            "                    <dim>1</dim>"
            "                    <dim>384</dim>"
            "                    <dim>13</dim>"
            "                    <dim>13</dim>"
            "                </port>"
            "            </output>"
            "        </layer>"
            "        <layer name=\"conv4_split\" type=\"Split\" precision=\"FP32\" id=\"28\">"
            "            <input>"
            "                <port id=\"57\">"
            "                    <dim>1</dim>"
            "                    <dim>384</dim>"
            "                    <dim>13</dim>"
            "                    <dim>13</dim>"
            "                </port>"
            "            </input>"
            "            <output>"
            "                <port id=\"59\">"
            "                    <dim>1</dim>"
            "                    <dim>192</dim>"
            "                    <dim>13</dim>"
            "                    <dim>13</dim>"
            "                </port>"
            "                <port id=\"63\">"
            "                    <dim>1</dim>"
            "                    <dim>192</dim>"
            "                    <dim>13</dim>"
            "                    <dim>13</dim>"
            "                </port>"
            "            </output>"
            "        </layer>"
            "        <layer name=\"conv4_1\" type=\"Convolution\" precision=\"FP32\" id=\"31\">"
            "            <convolution_data stride-x=\"1\" stride-y=\"1\" pad-x=\"1\" pad-y=\"1\" kernel-x=\"3\" kernel-y=\"3\" output=\"192\" group=\"1\"/>"
            "            <input>"
            "                <port id=\"64\">"
            "                    <dim>1</dim>"
            "                    <dim>192</dim>"
            "                    <dim>13</dim>"
            "                    <dim>13</dim>"
            "                </port>"
            "            </input>"
            "            <output>"
            "                <port id=\"65\">"
            "                    <dim>1</dim>"
            "                    <dim>192</dim>"
            "                    <dim>13</dim>"
            "                    <dim>13</dim>"
            "                </port>"
            "            </output>"
            "            <weights offset=\"4910080\" size=\"1327104\"/>"
            "            <biases offset=\"6237184\" size=\"768\"/>"
            "        </layer>"
            "        <layer name=\"conv4_0\" type=\"Convolution\" precision=\"FP32\" id=\"30\">"
            "            <convolution_data stride-x=\"1\" stride-y=\"1\" pad-x=\"1\" pad-y=\"1\" kernel-x=\"3\" kernel-y=\"3\" output=\"192\" group=\"1\"/>"
            "            <input>"
            "                <port id=\"60\">"
            "                    <dim>1</dim>"
            "                    <dim>192</dim>"
            "                    <dim>13</dim>"
            "                    <dim>13</dim>"
            "                </port>"
            "            </input>"
            "            <output>"
            "                <port id=\"61\">"
            "                    <dim>1</dim>"
            "                    <dim>192</dim>"
            "                    <dim>13</dim>"
            "                    <dim>13</dim>"
            "                </port>"
            "            </output>"
            "            <weights offset=\"6237952\" size=\"1327104\"/>"
            "            <biases offset=\"7565056\" size=\"768\"/>"
            "        </layer>"
            "        <layer name=\"conv4_merge\" type=\"Concat\" precision=\"FP32\" id=\"29\">"
            "            <concat_data axis=\"1\"/>"
            "            <input>"
            "                <port id=\"62\">"
            "                    <dim>1</dim>"
            "                    <dim>192</dim>"
            "                    <dim>13</dim>"
            "                    <dim>13</dim>"
            "                </port>"
            "                <port id=\"66\">"
            "                    <dim>1</dim>"
            "                    <dim>192</dim>"
            "                    <dim>13</dim>"
            "                    <dim>13</dim>"
            "                </port>"
            "            </input>"
            "            <output>"
            "                <port id=\"58\">"
            "                    <dim>1</dim>"
            "                    <dim>384</dim>"
            "                    <dim>13</dim>"
            "                    <dim>13</dim>"
            "                </port>"
            "            </output>"
            "        </layer>"
            "        <layer name=\"relu4\" type=\"ReLU\" precision=\"FP32\" id=\"12\">"
            "            <input>"
            "                <port id=\"23\">"
            "                    <dim>1</dim>"
            "                    <dim>384</dim>"
            "                    <dim>13</dim>"
            "                    <dim>13</dim>"
            "                </port>"
            "            </input>"
            "            <output>"
            "                <port id=\"24\">"
            "                    <dim>1</dim>"
            "                    <dim>384</dim>"
            "                    <dim>13</dim>"
            "                    <dim>13</dim>"
            "                </port>"
            "            </output>"
            "        </layer>"
            "        <layer name=\"conv5_split\" type=\"Split\" precision=\"FP32\" id=\"32\">"
            "            <input>"
            "                <port id=\"67\">"
            "                    <dim>1</dim>"
            "                    <dim>384</dim>"
            "                    <dim>13</dim>"
            "                    <dim>13</dim>"
            "                </port>"
            "            </input>"
            "            <output>"
            "                <port id=\"69\">"
            "                    <dim>1</dim>"
            "                    <dim>192</dim>"
            "                    <dim>13</dim>"
            "                    <dim>13</dim>"
            "                </port>"
            "                <port id=\"73\">"
            "                    <dim>1</dim>"
            "                    <dim>192</dim>"
            "                    <dim>13</dim>"
            "                    <dim>13</dim>"
            "                </port>"
            "            </output>"
            "        </layer>"
            "        <layer name=\"conv5_1\" type=\"Convolution\" precision=\"FP32\" id=\"35\">"
            "            <convolution_data stride-x=\"1\" stride-y=\"1\" pad-x=\"1\" pad-y=\"1\" kernel-x=\"3\" kernel-y=\"3\" output=\"128\" group=\"1\"/>"
            "            <input>"
            "                <port id=\"74\">"
            "                    <dim>1</dim>"
            "                    <dim>192</dim>"
            "                    <dim>13</dim>"
            "                    <dim>13</dim>"
            "                </port>"
            "            </input>"
            "            <output>"
            "                <port id=\"75\">"
            "                    <dim>1</dim>"
            "                    <dim>128</dim>"
            "                    <dim>13</dim>"
            "                    <dim>13</dim>"
            "                </port>"
            "            </output>"
            "            <weights offset=\"7565824\" size=\"884736\"/>"
            "            <biases offset=\"8450560\" size=\"512\"/>"
            "        </layer>"
            "        <layer name=\"conv5_0\" type=\"Convolution\" precision=\"FP32\" id=\"34\">"
            "            <convolution_data stride-x=\"1\" stride-y=\"1\" pad-x=\"1\" pad-y=\"1\" kernel-x=\"3\" kernel-y=\"3\" output=\"128\" group=\"1\"/>"
            "            <input>"
            "                <port id=\"70\">"
            "                    <dim>1</dim>"
            "                    <dim>192</dim>"
            "                    <dim>13</dim>"
            "                    <dim>13</dim>"
            "                </port>"
            "            </input>"
            "            <output>"
            "                <port id=\"71\">"
            "                    <dim>1</dim>"
            "                    <dim>128</dim>"
            "                    <dim>13</dim>"
            "                    <dim>13</dim>"
            "                </port>"
            "            </output>"
            "            <weights offset=\"8451072\" size=\"884736\"/>"
            "            <biases offset=\"9335808\" size=\"512\"/>"
            "        </layer>"
            "        <layer name=\"conv5_merge\" type=\"Concat\" precision=\"FP32\" id=\"33\">"
            "            <concat_data axis=\"1\"/>"
            "            <input>"
            "                <port id=\"72\">"
            "                    <dim>1</dim>"
            "                    <dim>128</dim>"
            "                    <dim>13</dim>"
            "                    <dim>13</dim>"
            "                </port>"
            "                <port id=\"76\">"
            "                    <dim>1</dim>"
            "                    <dim>128</dim>"
            "                    <dim>13</dim>"
            "                    <dim>13</dim>"
            "                </port>"
            "            </input>"
            "            <output>"
            "                <port id=\"68\">"
            "                    <dim>1</dim>"
            "                    <dim>256</dim>"
            "                    <dim>13</dim>"
            "                    <dim>13</dim>"
            "                </port>"
            "            </output>"
            "        </layer>"
            "        <layer name=\"relu5\" type=\"ReLU\" precision=\"FP32\" id=\"14\">"
            "            <input>"
            "                <port id=\"27\">"
            "                    <dim>1</dim>"
            "                    <dim>256</dim>"
            "                    <dim>13</dim>"
            "                    <dim>13</dim>"
            "                </port>"
            "            </input>"
            "            <output>"
            "                <port id=\"28\">"
            "                    <dim>1</dim>"
            "                    <dim>256</dim>"
            "                    <dim>13</dim>"
            "                    <dim>13</dim>"
            "                </port>"
            "            </output>"
            "        </layer>"
            "        <layer name=\"pool5\" type=\"Pooling\" precision=\"FP32\" id=\"15\">"
            "            <pooling_data kernel-x=\"3\" kernel-y=\"3\" pad-x=\"0\" pad-y=\"0\" stride-x=\"2\" stride-y=\"2\" rounding-type=\"ceil\" pool-method=\"max\"/>"
            "            <input>"
            "                <port id=\"29\">"
            "                    <dim>1</dim>"
            "                    <dim>256</dim>"
            "                    <dim>13</dim>"
            "                    <dim>13</dim>"
            "                </port>"
            "            </input>"
            "            <output>"
            "                <port id=\"30\">"
            "                    <dim>1</dim>"
            "                    <dim>256</dim>"
            "                    <dim>6</dim>"
            "                    <dim>6</dim>"
            "                </port>"
            "            </output>"
            "        </layer>"
            "        <layer name=\"fc6\" type=\"FullyConnected\" precision=\"FP32\" id=\"16\">"
            "            <fc_data out-size=\"4096\"/>"
            "            <input>"
            "                <port id=\"31\">"
            "                    <dim>1</dim>"
            "                    <dim>256</dim>"
            "                    <dim>6</dim>"
            "                    <dim>6</dim>"
            "                </port>"
            "            </input>"
            "            <output>"
            "                <port id=\"32\">"
            "                    <dim>1</dim>"
            "                    <dim>4096</dim>"
            "                </port>"
            "            </output>"
            "            <weights offset=\"9336320\" size=\"150994944\"/>"
            "            <biases offset=\"160331264\" size=\"16384\"/>"
            "        </layer>"
            "        <layer name=\"relu6\" type=\"ReLU\" precision=\"FP32\" id=\"17\">"
            "            <input>"
            "                <port id=\"33\">"
            "                    <dim>1</dim>"
            "                    <dim>4096</dim>"
            "                </port>"
            "            </input>"
            "            <output>"
            "                <port id=\"34\">"
            "                    <dim>1</dim>"
            "                    <dim>4096</dim>"
            "                </port>"
            "            </output>"
            "        </layer>"
            "        <layer name=\"fc7\" type=\"FullyConnected\" precision=\"FP32\" id=\"19\">"
            "            <fc_data out-size=\"4096\"/>"
            "            <input>"
            "                <port id=\"37\">"
            "                    <dim>1</dim>"
            "                    <dim>4096</dim>"
            "                </port>"
            "            </input>"
            "            <output>"
            "                <port id=\"38\">"
            "                    <dim>1</dim>"
            "                    <dim>4096</dim>"
            "                </port>"
            "            </output>"
            "            <weights offset=\"160347648\" size=\"67108864\"/>"
            "            <biases offset=\"227456512\" size=\"16384\"/>"
            "        </layer>"
            "        <layer name=\"relu7\" type=\"ReLU\" precision=\"FP32\" id=\"20\">"
            "            <input>"
            "                <port id=\"39\">"
            "                    <dim>1</dim>"
            "                    <dim>4096</dim>"
            "                </port>"
            "            </input>"
            "            <output>"
            "                <port id=\"40\">"
            "                    <dim>1</dim>"
            "                    <dim>4096</dim>"
            "                </port>"
            "            </output>"
            "        </layer>"
            "        <layer name=\"fc8\" type=\"FullyConnected\" precision=\"FP32\" id=\"22\">"
            "            <fc_data out-size=\"1000\"/>"
            "            <input>"
            "                <port id=\"43\">"
            "                    <dim>1</dim>"
            "                    <dim>4096</dim>"
            "                </port>"
            "            </input>"
            "            <output>"
            "                <port id=\"44\">"
            "                    <dim>1</dim>"
            "                    <dim>1000</dim>"
            "                </port>"
            "            </output>"
            "            <weights offset=\"227472896\" size=\"16384000\"/>"
            "            <biases offset=\"243856896\" size=\"4000\"/>"
            "        </layer>"
            "        <layer name=\"prob\" type=\"SoftMax\" precision=\"FP32\" id=\"23\">"
            "            <input>"
            "                <port id=\"45\">"
            "                    <dim>1</dim>"
            "                    <dim>1000</dim>"
            "                </port>"
            "            </input>"
            "            <output>"
            "                <port id=\"46\">"
            "                    <dim>1</dim>"
            "                    <dim>1000</dim>"
            "                </port>"
            "            </output>"
            "        </layer>"
            "    </layers>"
            "    <edges>"
            "        <edge from-layer=\"0\" from-port=\"0\" to-layer=\"1\" to-port=\"1\"/>"
            "        <edge from-layer=\"1\" from-port=\"2\" to-layer=\"2\" to-port=\"3\"/>"
            "        <edge from-layer=\"2\" from-port=\"4\" to-layer=\"3\" to-port=\"5\"/>"
            "        <edge from-layer=\"3\" from-port=\"6\" to-layer=\"4\" to-port=\"7\"/>"
            "        <edge from-layer=\"4\" from-port=\"8\" to-layer=\"24\" to-port=\"47\"/>"
            "        <edge from-layer=\"25\" from-port=\"48\" to-layer=\"6\" to-port=\"11\"/>"
            "        <edge from-layer=\"6\" from-port=\"12\" to-layer=\"7\" to-port=\"13\"/>"
            "        <edge from-layer=\"7\" from-port=\"14\" to-layer=\"8\" to-port=\"15\"/>"
            "        <edge from-layer=\"8\" from-port=\"16\" to-layer=\"9\" to-port=\"17\"/>"
            "        <edge from-layer=\"9\" from-port=\"18\" to-layer=\"10\" to-port=\"19\"/>"
            "        <edge from-layer=\"10\" from-port=\"20\" to-layer=\"28\" to-port=\"57\"/>"
            "        <edge from-layer=\"29\" from-port=\"58\" to-layer=\"12\" to-port=\"23\"/>"
            "        <edge from-layer=\"12\" from-port=\"24\" to-layer=\"32\" to-port=\"67\"/>"
            "        <edge from-layer=\"33\" from-port=\"68\" to-layer=\"14\" to-port=\"27\"/>"
            "        <edge from-layer=\"14\" from-port=\"28\" to-layer=\"15\" to-port=\"29\"/>"
            "        <edge from-layer=\"15\" from-port=\"30\" to-layer=\"16\" to-port=\"31\"/>"
            "        <edge from-layer=\"16\" from-port=\"32\" to-layer=\"17\" to-port=\"33\"/>"
            "        <edge from-layer=\"19\" from-port=\"38\" to-layer=\"20\" to-port=\"39\"/>"
            "        <edge from-layer=\"22\" from-port=\"44\" to-layer=\"23\" to-port=\"45\"/>"
            "        <edge from-layer=\"24\" from-port=\"49\" to-layer=\"26\" to-port=\"50\"/>"
            "        <edge from-layer=\"26\" from-port=\"51\" to-layer=\"25\" to-port=\"52\"/>"
            "        <edge from-layer=\"24\" from-port=\"53\" to-layer=\"27\" to-port=\"54\"/>"
            "        <edge from-layer=\"27\" from-port=\"55\" to-layer=\"25\" to-port=\"56\"/>"
            "        <edge from-layer=\"28\" from-port=\"59\" to-layer=\"30\" to-port=\"60\"/>"
            "        <edge from-layer=\"30\" from-port=\"61\" to-layer=\"29\" to-port=\"62\"/>"
            "        <edge from-layer=\"28\" from-port=\"63\" to-layer=\"31\" to-port=\"64\"/>"
            "        <edge from-layer=\"31\" from-port=\"65\" to-layer=\"29\" to-port=\"66\"/>"
            "        <edge from-layer=\"32\" from-port=\"69\" to-layer=\"34\" to-port=\"70\"/>"
            "        <edge from-layer=\"34\" from-port=\"71\" to-layer=\"33\" to-port=\"72\"/>"
            "        <edge from-layer=\"32\" from-port=\"73\" to-layer=\"35\" to-port=\"74\"/>"
            "        <edge from-layer=\"35\" from-port=\"75\" to-layer=\"33\" to-port=\"76\"/>"
            "        <edge from-layer=\"17\" from-port=\"34\" to-layer=\"19\" to-port=\"37\"/>"
            "        <edge from-layer=\"20\" from-port=\"40\" to-layer=\"22\" to-port=\"43\"/>"
            "    </edges>"
            "    <pre-process reference-layer-name=\"data\">"
            "        <channel id=\"0\">"
            "            <mean value=\"104.00698793\"/>"
            "        </channel>"
            "        <channel id=\"1\">"
            "            <mean value=\"116.66876762\"/>"
            "        </channel>"
            "        <channel id=\"2\">"
            "            <mean value=\"122.67891434\"/>"
            "        </channel>"
            "    </pre-process>"
            "</net>";
    InferenceEngine::CNNNetReader reader;
    ASSERT_NO_THROW(reader.ReadNetwork(model.data(), model.length()));

    auto topology = reader.getNetwork();
    auto topology_complexity = getNetworkComplexity(topology);

    std::map<std::string, InferenceEngine::LayerComplexity>
        reference
        {
            {"conv1", {210830400, 34944}},
            {"relu1", {290400, 0}},
            {"norm1", {14520000, 0}},
            {"pool1", {629856, 0}},

            {"conv2_0", {223948800, 153728}},
            {"conv2_1", {223948800, 153728}},
            {"relu2", {186624, 0}},
            {"norm2", {9331200, 0}},
            {"pool2", {389376, 0}},

            {"conv3", {299040768, 885120}},
            {"relu3", {64896, 0}},

            {"conv4_0", {112140288, 331968}},
            {"conv4_1", {112140288, 331968}},
            {"relu4", {64896, 0}},
            {"conv5_0", {74760192, 221312}},
            {"conv5_1", {74760192, 221312}},

            {"relu5", {43264, 0}},
            {"pool5", {82944, 0}},

            {"fc6", {75497472, 37752832}},
            {"relu6", {4096, 0}},

            {"fc7", {33554432, 16781312}},
            {"relu7", {4096, 0}},

            {"fc8", {8192000, 4097000}},
            {"prob", {4000, 0}}
        };

    for (auto &item: reference) {
        ASSERT_TRUE(topology_complexity.count(item.first) > 0);
        auto flops = topology_complexity[item.first].flops;
        auto params = topology_complexity[item.first].params;

        ASSERT_EQ(flops, item.second.flops);
        ASSERT_EQ(params, item.second.params);
    }
}

TEST(UtilTests, replaceLayerWithNewLayer) {
    //
    // I->L1->L3->O
    //     \  /
    //      L2
    //

    NetBuilder netBuilder;
    auto net = netBuilder
               .data("data1", IE::SizeVector{1, 1, 1}, IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data2", IE::SizeVector{1, 1, 1}, IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data3", IE::SizeVector{1, 1, 1}, IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .data("data4", IE::SizeVector{1, 1, 1}, IE::Precision::UNSPECIFIED, IE::Layout::CHW)
               .layer<IE::CNNLayer>(IE::LayerParams{"layer1", "dummy", IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer2", "dummy", IE::Precision::UNSPECIFIED})
               .layer<IE::CNNLayer>(IE::LayerParams{"layer3", "dummy", IE::Precision::UNSPECIFIED})
               .linkData("data1", "data2", "layer1")
               .linkData("data2", "data3", "layer2")
               .linkData("data2", "data4", "layer3")
               .linkDataTo("data3", "layer3")
               .finalize();

    const auto& layers = netBuilder.getLayersMap();
    const auto& data   = netBuilder.getDataMap();

    {   // Replace L1
        auto newLayer1 = std::make_shared<IE::CNNLayer>(IE::LayerParams{"layer1", "dummy", IE::Precision::UNSPECIFIED});
        auto layer1    = layers.find("layer1");
        EXPECT_TRUE(layer1 != layers.end());
        IE::replaceLayerWithNewLayer(*net, layer1->second, newLayer1);
        IE::CNNLayerPtr layer1Check = nullptr;
        net->getLayerByName("layer1", layer1Check, nullptr);
        ASSERT_EQ(layer1Check, newLayer1);
        ASSERT_EQ(layer1Check->outData.size(), 1);
        ASSERT_EQ(layer1Check->outData[0], data.find("data2")->second);
        ASSERT_EQ(layer1Check->outData[0]->creatorLayer.lock(), newLayer1);
    }
    {   // Replace L2
        auto newLayer2 = std::make_shared<IE::CNNLayer>(IE::LayerParams{"layer2", "dummy", IE::Precision::UNSPECIFIED});
        auto layer2    = layers.find("layer2");
        EXPECT_TRUE(layer2 != layers.end());
        IE::replaceLayerWithNewLayer(*net, layer2->second, newLayer2);
        IE::CNNLayerPtr layer2Check = nullptr;
        net->getLayerByName("layer2", layer2Check, nullptr);
        ASSERT_EQ(layer2Check, newLayer2);
        ASSERT_EQ(layer2Check->outData.size(), 1);
        ASSERT_EQ(layer2Check->outData[0], data.find("data3")->second);
        ASSERT_EQ(layer2Check->outData[0]->creatorLayer.lock(), newLayer2);
    }
    {   // Replace L3
        auto newLayer3 = std::make_shared<IE::CNNLayer>(IE::LayerParams{"layer3", "dummy", IE::Precision::UNSPECIFIED});
        auto layer3    = layers.find("layer3");
        EXPECT_TRUE(layer3 != layers.end());
        IE::replaceLayerWithNewLayer(*net, layer3->second, newLayer3);
        IE::CNNLayerPtr layer3Check = nullptr;
        net->getLayerByName("layer3", layer3Check, nullptr);
        ASSERT_EQ(layer3Check, newLayer3);
        ASSERT_EQ(layer3Check->outData.size(), 1);
        ASSERT_EQ(layer3Check->outData[0], data.find("data4")->second);
        ASSERT_EQ(layer3Check->outData[0]->creatorLayer.lock(), newLayer3);
        ASSERT_TRUE(layer3Check->insData[0].lock() == data.find("data2")->second ||
                    layer3Check->insData[0].lock() == data.find("data3")->second);
        ASSERT_TRUE(layer3Check->insData[1].lock() == data.find("data2")->second ||
                    layer3Check->insData[1].lock() == data.find("data3")->second);
    }
}
