// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <inference_engine/graph_tools.hpp>
#include "test_assertions.hpp"
#include <unordered_set>
#include <gmock/gmock-generated-function-mockers.h>
#include <gmock/gmock-generated-matchers.h>
#include <gmock/gmock-more-actions.h>
#include "xml_father.hpp"
#include "ie_common.h"
#include "graph_test_base.hpp"
#include <memory>

#ifdef ENABLE_GNA
# include <gna_plugin/quantization/model_quantizer.hpp>
#endif

using namespace testing;
using namespace InferenceEngine;
using namespace std;
using namespace GraphTest;


class GraphCopyTests : public GraphTestsBase {

protected:
    MockCopier mc;

    void SetUp() override {
        GraphTestsBase::_batchSize = 12;
        GraphTestsBase::SetUp();
        CONNECT(1, 2);
        CONNECT(3, 4);
        CONNECT(4, 2);
        CONNECT(3, 5);
        CONNECT(5, 2);

        EXPECT_CALL(mockNet, getInputsInfo(_)).WillRepeatedly(WithArg<0>(Invoke([&](InputsDataMap &maps) {
            prepareInputs(maps, 12);
        })));

        EXPECT_CALL(mockNet, getOutputsInfo(_)).WillRepeatedly(WithArg<0>(Invoke([&](OutputsDataMap &maps) {
            prepareOutputs(maps);
        })));

        EXPECT_CALL(mockNet, getTargetDevice()).WillRepeatedly(Return(TargetDevice::eCPU));
        EXPECT_CALL(mockNet, getPrecision()).WillRepeatedly(Return(Precision::FP16));
        EXPECT_CALL(mockNet, getBatchSize()).WillRepeatedly(Return(12));
        EXPECT_CALL(mockNet, getName(_, _)).WillRepeatedly(Invoke([](char *pName, size_t len) {
            memcpy(pName, "nm", 3);
        }));

        EXPECT_CALL(mc, copyLayer(_)).WillRepeatedly(Invoke([](CNNLayerPtr ptr) {
            return ptr;
        }));
    }
};

TEST_F(GraphCopyTests, copyNetworkPreserveBasicParams) {

    auto clone = CNNNetCopy<MockCopier>(mockNet, mc);

    //network was copied not just assigned
    ASSERT_NE(clone.get(), &mockNet);
    ASSERT_EQ(clone->getTargetDevice(), TargetDevice::eCPU);
    ASSERT_EQ(clone->getPrecision(), Precision::FP16);

    char name[20];
    clone->getName(name, sizeof(name));
    ASSERT_STREQ(name, "nm");
}

TEST_F(GraphCopyTests, canPreserveBatchWhenCopyNetwork) {
    auto clone = CNNNetCopy<MockCopier>(mockNet, mc);
    ASSERT_EQ(clone->getBatchSize(), 12);
}


TEST_F(GraphCopyTests, canPreserveInputs) {
    auto clone = CNNNetCopy<MockCopier>(mockNet, mc);

    InputsDataMap inputs, inputsTarget;
    InputsDataMap heads, headsTarget;

    clone->getInputsInfo(inputs);
    mockNet.getInputsInfo(inputsTarget);
    ASSERT_INPUTS_INFO_EQ(inputs, inputsTarget);
}

TEST_F(GraphCopyTests, canPreserveOutputs) {

    auto clone = CNNNetCopy<MockCopier>(mockNet, mc);

    OutputsDataMap outTarget, outSource;
    clone->getOutputsInfo(outTarget);
    mockNet.getOutputsInfo(outSource);

    ASSERT_OUTPUTS_INFO_EQ(outSource, outTarget);
}

TEST_F(GraphCopyTests, canPreserveAttributes) {
    auto clone = CNNNetCopy<MockCopier>(mockNet, mc);
    ADD_ATTR(1, "id", "r-1-2-3");
    ADD_ATTR(2, "id", "r-1-2-3");

    IE_SUPPRESS_DEPRECATED_START
    CNNNetwork cloned (clone.get());
    IE_SUPPRESS_DEPRECATED_END
    auto idMemOutput = cloned.getLayerByName("1")->GetParamAsString("id");
    auto idMemInput  = cloned.getLayerByName("2")->GetParamAsString("id");

    ASSERT_STREQ(idMemInput.c_str(), idMemOutput.c_str());
    ASSERT_STREQ(idMemInput.c_str(), "r-1-2-3");
}

TEST_F(GraphCopyTests, canPreserveGetData) {
    auto clone = CNNNetCopy<MockCopier>(mockNet, mc);

    ASSERT_NE(clone->getData("1"), nullptr);
    ASSERT_NE(clone->getData("2"), nullptr);
    ASSERT_NE(clone->getData("3"), nullptr);
    ASSERT_NE(clone->getData("4"), nullptr);
    ASSERT_NE(clone->getData("5"), nullptr);
}

TEST_F(GraphCopyTests, canPreserveTopology) {
    auto iclone = CNNNetCopy<MockCopier>(mockNet, mc);
    auto clone = CNNNetwork(iclone);

    ASSERT_EQ(clone.layerCount(), 5);

    EXPECT_CALL(*this, visited(1, 0)).Times(1);
    EXPECT_CALL(*this, visited(2, 1)).Times(1);

    EXPECT_CALL(*this, visited2(3, 0)).Times(1);
    EXPECT_CALL(*this, visited2(4, AnyOf(1, 2))).Times(1);
    EXPECT_CALL(*this, visited2(5, AnyOf(1, 2))).Times(1);
    EXPECT_CALL(*this, visited2(2, 3)).Times(1);

    int idx = 0;
    CNNNetBFS(clone.getLayerByName("1"), [&](CNNLayerPtr layer) {
        visited(ID(layer), idx++);
    });

    idx = 0;
    CNNNetBFS(clone.getLayerByName("3"), [&](CNNLayerPtr layer) {
        visited2(ID(layer), idx++);
    });
}

#ifdef ENABLE_GNA
using namespace GNAPluginNS;
struct _FP32_2_FP32  : public GNAPluginNS::details::QuantDescTmpl<float, float, float, float, float> {
};
using FP32_2_FP32 = GNAPluginNS::details::QuantPair<_FP32_2_FP32 , _FP32_2_FP32 >;

TEST_F(GraphCopyTests, canQuantizeTopology) {

    auto iclone = ModelQuantizer<FP32_2_FP32>().quantize(mockNet, std::vector<float >({1.0f, 1.0f}));
    auto clone = CNNNetwork(iclone);

    CNNNetBFS(clone.getLayerByName("1"), [&](CNNLayerPtr layer) {
        auto params = getInjectedData<QuantizedLayerParams>(layer);
        ASSERT_NE(params, nullptr);
    });

    CNNNetBFS(clone.getLayerByName("3"), [&](CNNLayerPtr layer) {
        auto params = getInjectedData<QuantizedLayerParams>(layer);
        ASSERT_NE(params, nullptr);
    });
}

#endif

TEST(CNNSpecificGraphCopyTests, copyNetworkWithClampLayer) {
    CNNNetReader netReader;
    //define minimal network with Clamp layer
    const std::string SINGLE_LAYER_MODEL = R"V0G0N(
    <net name="SingleLayer" version="2" batch="1">
        <layers>
                <layer id="0" name="InputLayer" precision="FP16" type="Input">
                        <output>
                                <port id="0">
                                        <dim>1</dim>
                                        <dim>3</dim>
                                        <dim>224</dim>
                                        <dim>224</dim>
                                </port>
                        </output>
                </layer>
                <layer id="1" name="ClampLayer" precision="FP16" type="Clamp">
                    <data max="6" min="0"/>
                    <input>
                            <port id="0">
                                    <dim>1</dim>
                                    <dim>3</dim>
                                    <dim>224</dim>
                                    <dim>224</dim>
                            </port>
                    </input>
                    <output>
                            <port id="1">
                                    <dim>1</dim>
                                    <dim>3</dim>
                                    <dim>224</dim>
                                    <dim>224</dim>
                            </port>
                    </output>
                </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        </edges>
    </net>
    )V0G0N";
    ASSERT_NO_THROW(netReader.ReadNetwork(SINGLE_LAYER_MODEL.data(), SINGLE_LAYER_MODEL.length()));
    ASSERT_TRUE(netReader.isParseSuccess());
    auto network = netReader.getNetwork();

    //copy the network
    struct EmptyStruct {};
    auto visitor = [&](CNNLayerPtr lp) { return injectData<EmptyStruct>(lp); };
    auto copied_net_ptr = CNNNetCopy(network, visitor);
    IE_SUPPRESS_DEPRECATED_START
    auto copied_net = CNNNetwork(copied_net_ptr.get());
    IE_SUPPRESS_DEPRECATED_END

    //check that Clamp layer was properly copied
    auto layer = std::dynamic_pointer_cast<ClampLayer>(copied_net.getLayerByName("ClampLayer"));
    ASSERT_NE(layer, nullptr) << "Could not perform dynamic cast from base pointer to Clamp layer pointer. "
            "Net copy could be incorrect.";
}

TEST(CNNSpecificGraphCopyTests, copyPreprocess) {
    CNNNetReader netReader;
    //define minimal network with Clamp layer
    const std::string SINGLE_LAYER_MODEL = R"V0G0N(
    <net name="SingleLayer" version="2" batch="1">
        <layers>
                <layer id="0" name="InputLayer" precision="FP16" type="Input">
                        <output>
                                <port id="0">
                                        <dim>1</dim>
                                        <dim>3</dim>
                                        <dim>224</dim>
                                        <dim>224</dim>
                                </port>
                        </output>
                </layer>
                <layer id="1" name="ClampLayer" precision="FP16" type="Clamp">
                    <data max="6" min="0"/>
                    <input>
                            <port id="0">
                                    <dim>1</dim>
                                    <dim>3</dim>
                                    <dim>224</dim>
                                    <dim>224</dim>
                            </port>
                    </input>
                    <output>
                            <port id="1">
                                    <dim>1</dim>
                                    <dim>3</dim>
                                    <dim>224</dim>
                                    <dim>224</dim>
                            </port>
                    </output>
                </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        </edges>
        <pre-process reference-layer-name="InputLayer">
            <channel id="0">
                <mean value="104"/>
            </channel>
            <channel id="1">
                <mean value="116"/>
            </channel>
            <channel id="2">
                <mean value="122"/>
            </channel>
        </pre-process>
    </net>
    )V0G0N";
    ASSERT_NO_THROW(netReader.ReadNetwork(SINGLE_LAYER_MODEL.data(), SINGLE_LAYER_MODEL.length()));
    ASSERT_TRUE(netReader.isParseSuccess());
    auto network = netReader.getNetwork();

    //copy the network
    struct EmptyStruct {};
    auto visitor = [&](CNNLayerPtr lp) { return injectData<EmptyStruct>(lp); };
    auto copied_net_ptr = CNNNetCopy(network, visitor);
    IE_SUPPRESS_DEPRECATED_START
    auto copied_net = CNNNetwork(copied_net_ptr.get());
    IE_SUPPRESS_DEPRECATED_END

    //check that pre process Info existed in copied network
    auto &pp = copied_net.getInputsInfo().begin()->second->getPreProcess();
    ASSERT_EQ(MEAN_VALUE, pp.getMeanVariant());
    ASSERT_EQ(3, pp.getNumberOfChannels());


    ASSERT_FLOAT_EQ(pp[0]->meanValue, 104);
    ASSERT_FLOAT_EQ(pp[1]->meanValue, 116);
    ASSERT_FLOAT_EQ(pp[2]->meanValue, 122);
}

TEST(CNNSpecificGraphCopyTests, copyNetworkWithDeconvolution) {
    CNNNetReader netReader;
    //define minimal network with deconvolution layer
    const std::string SINGLE_LAYER_MODEL = R"V0G0N(
    <net name="SingleLayer" version="2" batch="1">
        <layers>
                <layer id="0" name="InputLayer" precision="FP16" type="Input">
                        <output>
                                <port id="0">
                                        <dim>1</dim>
                                        <dim>384</dim>
                                        <dim>4</dim>
                                        <dim>2</dim>
                                </port>
                        </output>
                </layer>
            <layer name="upsample_merged" type="Deconvolution" precision="FP16" id="1">
            <deconvolution_data stride-x="2" stride-y="2" pad-x="1" pad-y="1" kernel-x="4" kernel-y="4" output="384" group="384"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>384</dim>
                    <dim>4</dim>
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>384</dim>
                    <dim>8</dim>
                    <dim>4</dim>
                </port>
            </output>
            <weights offset="5517824" size="12288"/>
        </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        </edges>
    </net>
    )V0G0N";
    ASSERT_NO_THROW(netReader.ReadNetwork(SINGLE_LAYER_MODEL.data(), SINGLE_LAYER_MODEL.length()));
    ASSERT_TRUE(netReader.isParseSuccess());
    auto network = netReader.getNetwork();

    // copy the network
    struct EmptyStruct {};
    auto visitor = [&](CNNLayerPtr lp) { return injectData<EmptyStruct>(lp); };
    auto copied_net_ptr = CNNNetCopy(network, visitor);
    IE_SUPPRESS_DEPRECATED_START
    auto copied_net = CNNNetwork(copied_net_ptr.get());
    IE_SUPPRESS_DEPRECATED_END

    // check that Clamp layer was properly copied
    auto layer = std::dynamic_pointer_cast<DeconvolutionLayer>(copied_net.getLayerByName("upsample_merged"));
    ASSERT_NE(layer, nullptr) << "Could not perform dynamic cast from base pointer to Deconvolution layer pointer. "
                                 "Net copy could be incorrect.";
}
