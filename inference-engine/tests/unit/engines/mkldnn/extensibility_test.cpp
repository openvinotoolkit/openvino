// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include <mkldnn_plugin/mkldnn_graph.h>
#include <mkldnn_plugin/nodes/mkldnn_generic_node.h>
#include "mkldnn_plugin/mkldnn_extension_mngr.h"
#include "mock_mkldnn_primitive.hpp"

using namespace ::testing;
using namespace std;
using namespace mkldnn;

class MKLDNNExtensionTests: public ::testing::Test {
 protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
    }
    MockMKLDNNGenericPrimitive * mock_mkldnn_primitive()  {
        return new MockMKLDNNGenericPrimitive();
    }
};

class MKLDNNFakeGenericNode: public MKLDNNPlugin::MKLDNNGenericNode {
public:
    explicit MKLDNNFakeGenericNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng): MKLDNNPlugin::MKLDNNGenericNode(layer, eng) {}

    void SetPrimitive(shared_ptr<InferenceEngine::MKLDNNPlugin::IMKLDNNGenericPrimitive>& prim) {
        genericPrimitive = prim;
    }
};

TEST_F(MKLDNNExtensionTests, canDeleteGenericPrimitiveFromNode ) {
    mkldnn::engine eng(mkldnn::engine(mkldnn::engine::kind::cpu, 0));

    InferenceEngine::CNNLayerPtr layer(new InferenceEngine::CNNLayer({"", "Generic", InferenceEngine::Precision::FP32}));
    InferenceEngine::DataPtr data(new InferenceEngine::Data("Test", {1, 1, 2, 3}, InferenceEngine::Precision::FP32));
    layer->outData.push_back(data);
    auto node = MKLDNNPlugin::MKLDNNNodePtr(new MKLDNNFakeGenericNode(layer, eng));

    engine e(engine::cpu, 0);
    auto mock_ptr = mock_mkldnn_primitive();
    EXPECT_CALL(*mock_ptr, die()).Times(1);

    auto generic = shared_ptr<InferenceEngine::MKLDNNPlugin::IMKLDNNGenericPrimitive>(mock_ptr);
    MKLDNNFakeGenericNode *nodePtr = dynamic_cast<MKLDNNFakeGenericNode *>(node.get());
    nodePtr->SetPrimitive(generic);

    generic.reset();
    node.reset();

    Mock::VerifyAndClear(mock_ptr);
}
