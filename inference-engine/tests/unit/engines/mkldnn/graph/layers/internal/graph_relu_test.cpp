// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "mkldnn_plugin/mkldnn_graph.h"
#include "mock_mkldnn_primitive.hpp"

#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include <mkldnn_plugin/mkldnn_extension_utils.h>
#include "tests_common.hpp"


using namespace ::testing;
using namespace std;
using namespace mkldnn;


struct relu_test_params {
    struct {
        size_t n;
        size_t c;
        size_t h;
        size_t w;
    } in;

    float n_clope;

    size_t num_prim_desc;

    MKLDNNPlugin::impl_desc_type selectedType;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

template <typename data_t>
void ref_relu(const InferenceEngine::TBlob<data_t> &src, InferenceEngine::TBlob<data_t> &dst, relu_test_params prm)
{
    size_t IW = prm.in.w;
    size_t IH = prm.in.h;
    size_t IC = prm.in.c;

    const data_t *src_data = src.readOnly();
    data_t *dst_data = dst.data();

    for (uint32_t c = 0; c < IC; c++) {
        for (uint32_t h = 0; h < IH; h++) {
            for (uint32_t w = 0; w < IW; w++) {
                uint32_t oidx = c * IH * IW
                                + h * IW + w;

                dst_data[oidx] = src_data[oidx] >= 0.0 ?
                                 src_data[oidx] :
                                 src_data[oidx] * prm.n_clope;
            }
        }
    }
}

class MKLDNNGraphReluTests: public TestsCommon,
                                     public WithParamInterface<relu_test_params> {
    std::string model_t = R"V0G0N(
<Net Name="Relu_Only" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
        <layer name="norm" id="1" type="ReLU" precision="FP32">
            <input>
                <port id="1">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</Net>
)V0G0N";

    std::string getModel(relu_test_params p) {
        std::string model = model_t;

        REPLACE_WITH_NUM(model, "_IW_", p.in.w);
        REPLACE_WITH_NUM(model, "_IH_", p.in.h);
        REPLACE_WITH_NUM(model, "_IC_", p.in.c);
        REPLACE_WITH_NUM(model, "_IN_", p.in.n);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            relu_test_params p = ::testing::WithParamInterface<relu_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());
            auto& nodes = graph.getNodes();
            for (int i = 0; i < nodes.size(); i++) {
                if (nodes[i]->getType() == MKLDNNPlugin::Activation) {
                    ASSERT_LE(p.num_prim_desc, nodes[i]->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(nodes[i]->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, nodes[i]->getSelectedPrimitiveDescriptor());
                    ASSERT_TRUE(nodes[i]->getSelectedPrimitiveDescriptor()->getImplementationType() | p.selectedType);
                }
            }

            InferenceEngine::SizeVector dims_src = {p.in.n, p.in.c, p.in.h, p.in.w};

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, InferenceEngine::NCHW, dims_src);
            src->allocate();
            fill_data(src->buffer(), src->size());

            InferenceEngine::TBlob<float>* srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());

            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src));

            InferenceEngine::OutputsDataMap out;
            out = net_reader.getNetwork().getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;

            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;

            graph.Infer(srcs, outputBlobs);

            InferenceEngine::TBlob<float> dst_ref(item.second->getTensorDesc());
            dst_ref.allocate();

            ref_relu(*srcPtr, dst_ref, p);

            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphReluTests, TestsRelu) {}


INSTANTIATE_TEST_CASE_P(
        TestsRelu, MKLDNNGraphReluTests,
        ::testing::Values(
                relu_test_params{
                        {1, 3, 228, 228}, 0.0f, 9, MKLDNNPlugin::impl_desc_type::jit, {
                                [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                                    ASSERT_TRUE(impl.getImplementationType() | MKLDNNPlugin::impl_desc_type::jit);
                                    ASSERT_EQ(1, impl.getConfig().inConfs.size());
                                    ASSERT_EQ(1, impl.getConfig().outConfs.size());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                                },
                                [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                                    ASSERT_TRUE(impl.getImplementationType() | MKLDNNPlugin::impl_desc_type::jit);
                                    ASSERT_EQ(1, impl.getConfig().inConfs.size());
                                    ASSERT_EQ(1, impl.getConfig().outConfs.size());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                                }
                        }}));
