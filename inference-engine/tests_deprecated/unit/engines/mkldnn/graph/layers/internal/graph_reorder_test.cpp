// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_graph.hpp"

#include "tests_common.hpp"
#include <ie_core.hpp>

using namespace ::testing;
using namespace std;
using namespace mkldnn;

class MKLDNNGraphReorderTests: public TestsCommon {
protected:
    virtual void SetUp() {
        TestsCommon::SetUp();
    }
};

TEST_F(MKLDNNGraphReorderTests, cannotCreatePrimitiveDescriprorsWithoutOtherLayers) {
    std::shared_ptr<MKLDNNPlugin::MKLDNNNode> node;
    mkldnn::engine eng(mkldnn::engine(mkldnn::engine::kind::cpu, 0));

    InferenceEngine::CNNLayerPtr layer(new InferenceEngine::CNNLayer({"TestReorder", "Reorder", InferenceEngine::Precision::FP32}));
    MKLDNNPlugin::MKLDNNWeightsSharing::Ptr cache;
    node.reset(MKLDNNPlugin::MKLDNNNode::CreateNode(layer, eng, {}, cache));
    ASSERT_EQ(MKLDNNPlugin::Type::Reorder, node->getType());

    ASSERT_THROW(node->getSupportedDescriptors(), InferenceEngine::details::InferenceEngineException);
}

TEST_F(MKLDNNGraphReorderTests, CreateReorder) {
    std::string model = R"V0G0N(
<Net Name="Convolution_Only" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>9</dim>
                    <dim>16</dim>
                    <dim>32</dim>
                </port>
            </output>
        </layer>
        <layer name="conv1" id="1" type="Convolution" precision="FP32">
            <convolution stride-x="1" stride-y="1"
                         pad-x="0"    pad-y="0"
                         kernel-x="1" kernel-y="1"
                         output="17"   group="1"/>

            <weights offset="0" size="612" />
            <biases offset="612" size="68" />

            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>9</dim>
                    <dim>16</dim>
                    <dim>32</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>17</dim>
                    <dim>16</dim>
                    <dim>32</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</Net>
)V0G0N";

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8,
                                                                                   {(1 * 1 * 17 * 9 / 1 + 17)
                                                      * sizeof(float)}, InferenceEngine::C });
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork network;
    ASSERT_NO_THROW(network = core.ReadNetwork(model, weights_ptr));

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(network);

    auto& nodes = graph.getNodes();
    for (int i = 0; i < nodes.size(); i++) {
        if (nodes[i]->getType() == MKLDNNPlugin::Reorder) {
            ASSERT_EQ(1, nodes[i]->getSupportedPrimitiveDescriptors().size());
            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::ref_any,
                      nodes[i]->getSupportedPrimitiveDescriptors()[0].getImplementationType());
            ASSERT_EQ(1, nodes[i]->getSupportedPrimitiveDescriptors()[0].getConfig().inConfs.size());
            if (i == 1) {
                ASSERT_EQ(InferenceEngine::Layout::NCHW, nodes[i]->getSupportedPrimitiveDescriptors()[0].getConfig().inConfs[0].desc.getLayout());
                ASSERT_NE(InferenceEngine::Layout::NCHW, nodes[i]->getSupportedPrimitiveDescriptors()[0].getConfig().outConfs[0].desc.getLayout());
            } else {
                ASSERT_NE(InferenceEngine::Layout::NCHW, nodes[i]->getSupportedPrimitiveDescriptors()[0].getConfig().inConfs[0].desc.getLayout());
                ASSERT_EQ(InferenceEngine::Layout::NCHW, nodes[i]->getSupportedPrimitiveDescriptors()[0].getConfig().outConfs[0].desc.getLayout());
            }
            ASSERT_EQ(1, nodes[i]->getSupportedPrimitiveDescriptors()[0].getConfig().outConfs.size());
        }
    }
}

TEST_F(MKLDNNGraphReorderTests, CreateInPlaceReorder) {
    std::string model = R"V0G0N(
<Net Name="InPlaceReorder_Only" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>9</dim>
                    <dim>16</dim>
                    <dim>32</dim>
                </port>
            </output>
        </layer>
        <layer name="reshape1" id="1" type="Reshape" precision="FP32">
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>9</dim>
                    <dim>16</dim>
                    <dim>32</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>32</dim>
                    <dim>144</dim>
                </port>
            </output>
        </layer>
        <layer name="reshape2" id="2" type="Reshape" precision="FP32">
            <data axis="0" num_axes="-1" dim="1, 4608"/>
            <input>
                <port id="1">
                    <dim>32</dim>
                    <dim>144</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>48</dim>
                    <dim>32</dim>
                </port>
            </output>
        </layer>
        <layer name="scaleshift" id="3" type="ScaleShift" precision="FP32">
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>48</dim>
                    <dim>32</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>48</dim>
                    <dim>32</dim>
                </port>
            </output>
            <blobs>
                <weights offset="0" size="12"/>
                <biases offset="12" size="12"/>
            </blobs>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="1"/>
    </edges>
</Net>
)V0G0N";

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, {24}, InferenceEngine::C });
    weights->allocate();
    float *data = weights->buffer().as<float *>();
    size_t dataSize = weights->byteSize() / sizeof(float);
    for (size_t i = 0; i < dataSize; i++) {
        data[i] = 2;
    }
    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);
    
    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork network;
    ASSERT_NO_THROW(network = core.ReadNetwork(model, weights_ptr));

    network.addOutput("reshape1");

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(network);

    InferenceEngine::SizeVector dims_src = {1, 9, 16, 32};

    InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src, InferenceEngine::NCHW});
    src->allocate();
    data = src->buffer().as<float *>();
    dataSize = src->size();
    for (size_t i = 0; i < dataSize; i++) {
        data[i] = 1;
    }

    auto * srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());

    if (srcPtr == nullptr)
        FAIL() << "Cannot cast blob to TBlob<float>.";

    InferenceEngine::BlobMap srcs;
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src));

    InferenceEngine::OutputsDataMap out;
    out = network.getOutputsInfo();
    InferenceEngine::BlobMap outputBlobs;

    auto it = out.begin();
    std::pair<std::string, InferenceEngine::DataPtr> item = *it;

    InferenceEngine::TBlob<float>::Ptr output1;
    output1 = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    output1->allocate();
    outputBlobs[item.first] = output1;

    item = *(++it);

    InferenceEngine::TBlob<float>::Ptr output2;
    output2 = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    output2->allocate();
    outputBlobs[item.first] = output2;

    graph.Infer(srcs, outputBlobs);

    data = output1->data();
    for (size_t i = 0; i < output1->size(); i++) {
        ASSERT_EQ(data[i], 1);
    }
    data = output2->data();
    for (size_t i = 0; i < output2->size(); i++) {
        ASSERT_EQ(data[i], 4);
    }
}
