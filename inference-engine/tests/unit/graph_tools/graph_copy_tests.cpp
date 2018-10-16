// Copyright (C) 2018 Intel Corporation
//
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

    CNNNetwork cloned (clone.get());
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

    auto iclone = ModelQuantizer<FP32_2_FP32>().quantize(mockNet, 1.0f);
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
    auto copied_net = CNNNetwork(copied_net_ptr.get());

    //check that Clamp layer was properly copied
    auto layer = std::dynamic_pointer_cast<ClampLayer>(copied_net.getLayerByName("ClampLayer"));
    ASSERT_NE(layer, nullptr) << "Could not perform dynamic cast from base pointer to Clamp layer pointer. "
            "Net copy could be incorrect.";
}
