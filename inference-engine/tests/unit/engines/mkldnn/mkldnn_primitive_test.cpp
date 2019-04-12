// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "mkldnn_plugin/mkldnn_memory.h"
#include "mkldnn_plugin/mkldnn_graph.h"

using namespace std;
using namespace MKLDNNPlugin;
using namespace mkldnn;
using namespace ::testing;

class MKLDNNPrimitiveTest : public ::testing::Test {
protected:
    virtual void TearDown() override{
    }

    virtual void SetUp() override{
    }
};

//class ChildConv : public MKLDNNConvolution {
// public:
//    explicit ChildConv(const engine& eng) : MKLDNNConvolution(eng) {}
//    // Add the following two lines to the mock class.
//    MOCK_METHOD0(die, void());
//    ~ChildConv () { die(); }
//};


TEST_F(MKLDNNPrimitiveTest, DISABLED_canDeleteWeightInweitableLayer) {
    //simulate how convlayer gets created
    engine e(engine::cpu, 0);
    //auto node = MKLDNNPlugin::MKLDNNNodePtr(MKLDNNPlugin::MKLDNNNode::CreateNode(MKLDNNPlugin::Generic, InferenceEngine::Precision::FP32, ""));
//    ChildConv *conv = new ChildConv(e);
//    EXPECT_CALL(*conv, die()).Times(1);

    std::vector<float> weights = {1,2,3,4};
    std::vector<void *> weightsData = {(void*)&*weights.begin()};
    std::vector <size_t> weightsSize = {weights.size() * sizeof(float)};

    memory::dims dims(4);
    dims[0] = weights.size();

//    conv->CreateWeightsMemory(dims, memory::f32, memory::nchw);
//    conv->SetWeights(weightsData, weightsSize);
    FAIL() << "Should change the test";
//    node->SetPrimitive(conv);
//    node.reset();

//    Mock::VerifyAndClear(conv);
}