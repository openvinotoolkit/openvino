// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <initializer_list>
#include <string>
#include <unordered_set>

#include <ie_algorithm.hpp>
#include <legacy/ie_util_internal.hpp>
#include <legacy/graph_transformer.h>
#include "util_test.hpp"
#include <legacy/graph_tools.hpp>

namespace {
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
    EXPECT_TRUE(contains(temp_set, 42));
    EXPECT_FALSE(contains(temp_set, 5));
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
        auto blob = IE::make_shared_blob<float>(IE::TensorDesc(IE::Precision::FP32, IE::NCHW));
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
    if (InferenceEngine::details::contains(n->allLayers(), name)) {
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
               .data("data1", IE::TensorDesc(IE::Precision::UNSPECIFIED, IE::SizeVector{ 1,1,1,1 }, IE::Layout::NCHW))
               .data("data2", IE::TensorDesc(IE::Precision::UNSPECIFIED, IE::SizeVector{ 1,1,1 }, IE::Layout::CHW))
               .data("data3", IE::TensorDesc(IE::Precision::UNSPECIFIED, IE::SizeVector{ 1,1,1 }, IE::Layout::CHW))
               .data("data4", IE::TensorDesc(IE::Precision::UNSPECIFIED, IE::SizeVector{ 1,1,1 }, IE::Layout::CHW))
               .data("data5", IE::TensorDesc(IE::Precision::UNSPECIFIED, IE::SizeVector{ 1,1,1 }, IE::Layout::CHW))
               .data("data6", IE::TensorDesc(IE::Precision::UNSPECIFIED, IE::SizeVector{ 1,1,1 }, IE::Layout::CHW))
               .data("data7", IE::TensorDesc(IE::Precision::UNSPECIFIED, IE::SizeVector{ 1,1,1 }, IE::Layout::CHW))
               .data("data8", IE::TensorDesc(IE::Precision::UNSPECIFIED, IE::SizeVector{ 1,1,1 }, IE::Layout::CHW))
               .data("data9", IE::TensorDesc(IE::Precision::UNSPECIFIED, IE::SizeVector{ 1,1,1 }, IE::Layout::CHW))
               .data("data10", IE::TensorDesc(IE::Precision::UNSPECIFIED, IE::SizeVector{ 1,1,1 }, IE::Layout::CHW))
               .data("data11", IE::TensorDesc(IE::Precision::UNSPECIFIED, IE::SizeVector{ 1,1,1 }, IE::Layout::CHW))
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

    InferenceEngine::ResponseDesc resp;
    net->setName("net");

    {
        IE::InputsDataMap inputs;
        net->getInputsInfo(inputs);
        for (auto &&it : inputs) {
            it.second->getPreProcess().init(1);
            it.second->getPreProcess().setMeanImage(IE::make_shared_blob<float>({ IE::Precision::FP32, {1,1,1}, IE::Layout::CHW }));
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
        EXPECT_TRUE(contains(inputs,"data1"));
        EXPECT_TRUE(contains(outputs,"data2"));
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
        EXPECT_TRUE(contains(inputs,"data1"));
        EXPECT_TRUE(contains(inputs,"data10"));
        EXPECT_TRUE(contains(outputs,"data3"));
        EXPECT_TRUE(contains(outputs,"data7"));
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
        EXPECT_TRUE(contains(inputs,"data4"));
        EXPECT_TRUE(contains(inputs,"data9"));
        EXPECT_TRUE(contains(outputs,"data6"));
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
        EXPECT_TRUE(contains(inputs,"data3"));
        EXPECT_TRUE(contains(outputs,"data4"));
        EXPECT_TRUE(contains(outputs,"data11"));
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
        EXPECT_TRUE(contains(inputs,"data1"));
        EXPECT_TRUE(contains(inputs,"data10"));
        EXPECT_TRUE(contains(outputs,"data11"));
        EXPECT_TRUE(contains(outputs,"data6"));
    }
    {
        auto cloned = IE::cloneNet(IE::CNNNetwork(net));
        auto layer1 = getLayer(cloned, "layer1");
        auto layer2 = getLayer(cloned, "layer2");
        EXPECT_TRUE(IE::Precision::Q78          == layer1->precision);
        EXPECT_TRUE(IE::Precision::UNSPECIFIED  == layer2->precision);
    }
    {
        auto cloned = IE::cloneNet(IE::CNNNetwork(net));
        EXPECT_EQ("net",                         cloned->getName());
    }
    {
        auto cloned = IE::cloneNet(IE::CNNNetwork(net));
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
               .data("data1", IE::TensorDesc(IE::Precision::UNSPECIFIED, IE::SizeVector{ 1,1,1 }, IE::Layout::CHW))
               .data("data2", IE::TensorDesc(IE::Precision::UNSPECIFIED, IE::SizeVector{ 1,1,1 }, IE::Layout::CHW))
               .data("data3", IE::TensorDesc(IE::Precision::UNSPECIFIED, IE::SizeVector{ 1,1,1 }, IE::Layout::CHW))
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
               .data("data1", IE::TensorDesc(IE::Precision::UNSPECIFIED, IE::SizeVector{ 1,1,1 }, IE::Layout::CHW))
               .data("data2", IE::TensorDesc(IE::Precision::UNSPECIFIED, IE::SizeVector{ 1,1,1 }, IE::Layout::CHW))
               .data("data3", IE::TensorDesc(IE::Precision::UNSPECIFIED, IE::SizeVector{ 1,1,1 }, IE::Layout::CHW))
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

TEST(UtilTests, DISABLED_getRootDataObjects) {
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
               .data("data1",IE::TensorDesc(IE::Precision::UNSPECIFIED, IE::SizeVector{ 1,1,1 }, IE::Layout::CHW))
               .data("data2",IE::TensorDesc(IE::Precision::UNSPECIFIED, IE::SizeVector{ 1,1,1 }, IE::Layout::CHW))
               .data("data3",IE::TensorDesc(IE::Precision::UNSPECIFIED, IE::SizeVector{ 1,1,1 }, IE::Layout::CHW))
               .data("data4",IE::TensorDesc(IE::Precision::UNSPECIFIED, IE::SizeVector{ 1,1,1 }, IE::Layout::CHW))
               .data("data5",IE::TensorDesc(IE::Precision::UNSPECIFIED, IE::SizeVector{ 1,1,1 }, IE::Layout::CHW))
               .data("data6",IE::TensorDesc(IE::Precision::UNSPECIFIED, IE::SizeVector{ 1,1,1 }, IE::Layout::CHW))
               .data("data7",IE::TensorDesc(IE::Precision::UNSPECIFIED, IE::SizeVector{ 1,1,1 }, IE::Layout::CHW))
               .data("data8",IE::TensorDesc(IE::Precision::UNSPECIFIED, IE::SizeVector{ 1,1,1 }, IE::Layout::CHW))
               .data("data9",IE::TensorDesc(IE::Precision::UNSPECIFIED, IE::SizeVector{ 1,1,1 }, IE::Layout::CHW))
               .data("data10",IE::TensorDesc(IE::Precision::UNSPECIFIED, IE::SizeVector{ 1,1,1 }, IE::Layout::CHW))
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

    auto cloned = IE::cloneNet(IE::CNNNetwork(net));

    ASSERT_EQ(13, cloned->layerCount());
    auto root_data = IE::getRootDataObjects(IE::CNNNetwork(cloned));
    ASSERT_EQ(6, root_data.size());
    std::unordered_set<std::string> data_names;
    for (auto& data : root_data) {
        data_names.insert(data->getName());
    }
    ASSERT_TRUE(contains(data_names, "data1"));
    ASSERT_TRUE(contains(data_names, "data2"));
    ASSERT_TRUE(contains(data_names, "data3"));
    ASSERT_TRUE(contains(data_names, "data4"));
    ASSERT_TRUE(contains(data_names, "data5"));
    ASSERT_TRUE(contains(data_names, "data6"));
}
