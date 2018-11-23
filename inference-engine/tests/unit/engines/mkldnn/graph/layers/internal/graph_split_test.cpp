// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "mkldnn_plugin/mkldnn_graph.h"

#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include <mkldnn_plugin/mkldnn_extension_utils.h>
#include <inference_engine/cnn_network_impl.hpp>
#include "tests_common.hpp"

using namespace ::testing;
using namespace std;
using namespace mkldnn;

struct split_test_params {
    struct {
        size_t n;
        size_t c;
        size_t h;
        size_t w;
    } in;

    struct {
        size_t n;
        size_t c;
        size_t h;
        size_t w;
    } out1;

    struct {
        size_t n;
        size_t c;
        size_t h;
        size_t w;
    } out2;

    size_t num_prim_desc;

    MKLDNNPlugin::impl_desc_type selectedType;
    std::vector<MKLDNNPlugin::impl_desc_type> preferTypes;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

template <typename data_t>
void ref_split(InferenceEngine::TBlob<data_t> &src, InferenceEngine::TBlob<data_t> &dst1, InferenceEngine::TBlob<data_t> &dst2) {
    const float * srcData = src.readOnly();

    int MB = dst1.dims()[dst1.dims().size() - 1];

    float * dstData1 = dst1.data();
    int dstSize1 = dst1.size() / MB;

    float *dstData2 = dst2.data();
    int dstSize2 = dst2.size() / MB;

    for (int b = 0; b < MB; b++) {
        for (size_t j = 0; j < dstSize1; j++, srcData++) {
            dstData1[b*dstSize1 + j] = *srcData;
        }

        for (size_t j = 0; j < dstSize2; j++, srcData++) {
            dstData2[b*dstSize1 + j] = *srcData;
        }
    }
}

class MKLDNNGraphSplitTests: public TestsCommon,
                              public WithParamInterface<split_test_params> {
    // TODO: remove power layers from the test
    std::string model_t = R"V0G0N(
<net name="ConcatOnly" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
        <layer name="split" id="2" type="Split" precision="FP32">
            <split_data axis="1" PrimitivesPriority="_IMPLS_"/>
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
                    <dim>_ON1_</dim>
                    <dim>_OC1_</dim>
                    <dim>_OH1_</dim>
                    <dim>_OW1_</dim>
                </port>
                <port id="3">
                    <dim>_ON2_</dim>
                    <dim>_OC2_</dim>
                    <dim>_OH2_</dim>
                    <dim>_OW2_</dim>
                </port>
            </output>
        </layer>
        <layer name="power1" id="3" type="Power" precision="FP32">
            <power_data power="1" scale="1" shift="0"/>
            <input>
                <port id="1">
                    <dim>_ON1_</dim>
                    <dim>_OC1_</dim>
                    <dim>_OH1_</dim>
                    <dim>_OW1_</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>_ON1_</dim>
                    <dim>_OC1_</dim>
                    <dim>_OH1_</dim>
                    <dim>_OW1_</dim>
                </port>
            </output>
        </layer>
        <layer name="power2" id="4" type="Power" precision="FP32">
            <power_data power="1" scale="1" shift="0"/>
            <input>
                <port id="1">
                    <dim>_ON2_</dim>
                    <dim>_OC2_</dim>
                    <dim>_OH2_</dim>
                    <dim>_OW2_</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>_ON2_</dim>
                    <dim>_OC2_</dim>
                    <dim>_OH2_</dim>
                    <dim>_OW2_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="3" to-layer="4" to-port="1"/>
    </edges>
</net>
)V0G0N";

protected:
    std::string getModel(split_test_params p) {
        std::string model = model_t;
        REPLACE_WITH_NUM(model, "_IN_", p.in.n);
        REPLACE_WITH_NUM(model, "_IC_", p.in.c);
        REPLACE_WITH_NUM(model, "_IW_", p.in.w);
        REPLACE_WITH_NUM(model, "_IH_", p.in.h);

        REPLACE_WITH_NUM(model, "_ON1_", p.out1.n);
        REPLACE_WITH_NUM(model, "_OC1_", p.out1.c);
        REPLACE_WITH_NUM(model, "_OH1_", p.out1.h);
        REPLACE_WITH_NUM(model, "_OW1_", p.out1.w);

        REPLACE_WITH_NUM(model, "_ON2_", p.out2.n);
        REPLACE_WITH_NUM(model, "_OC2_", p.out2.c);
        REPLACE_WITH_NUM(model, "_OH2_", p.out2.h);
        REPLACE_WITH_NUM(model, "_OW2_", p.out2.w);
        std::string impls;
        for (const auto& preferType : p.preferTypes) {
            if (!impls.empty())
                impls += ",";
            impls += "cpu:" + MKLDNNGraphTestClass::getStrPrimitiveDescriptorType(preferType);
        }
        REPLACE_WITH_STR(model, "_IMPLS_", impls);
        return model;
    }

    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            split_test_params p = ::testing::WithParamInterface<split_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());
            auto& nodes = graph.getNodes();
            for (int i = 0; i < nodes.size(); i++) {
                if (nodes[i]->getType() == MKLDNNPlugin::Split) {
                    ASSERT_EQ(p.num_prim_desc, nodes[i]->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(nodes[i]->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, nodes[i]->getSelectedPrimitiveDescriptor());
                    ASSERT_EQ(p.selectedType, nodes[i]->getSelectedPrimitiveDescriptor()->getImplementationType());
                }
            }
            ASSERT_LE(3, nodes.size());

            InferenceEngine::SizeVector dims_src = {p.in.n, p.in.c, p.in.h, p.in.w};

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, InferenceEngine::NCHW, dims_src);
            src->allocate();
            fill_data(src->buffer(), src->size());

            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src));

            InferenceEngine::TBlob<float>* srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());

            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            InferenceEngine::OutputsDataMap out;
            out = net_reader.getNetwork().getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;
            auto it = out.begin();

            std::pair<std::string, InferenceEngine::DataPtr> item = *it;

            InferenceEngine::TBlob<float>::Ptr output1;
            output1 = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output1->allocate();
            outputBlobs[item.first] = output1;

            InferenceEngine::TBlob<float> dst_ref1(item.second->getTensorDesc());
            dst_ref1.allocate();

            item = *(++it);
            InferenceEngine::TBlob<float>::Ptr output2;
            output2 = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output2->allocate();
            outputBlobs[item.first] = output2;

            InferenceEngine::TBlob<float> dst_ref2(item.second->getTensorDesc());
            dst_ref2.allocate();

            graph.Infer(srcs, outputBlobs);

            ref_split(*srcPtr, dst_ref1, dst_ref2);

            compare(*output1, dst_ref1);
            compare(*output2, dst_ref2);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphSplitTests, TestsSplit) {}

INSTANTIATE_TEST_CASE_P(
        TestsSplit, MKLDNNGraphSplitTests,
        ::testing::Values(
                split_test_params {
                        {1, 24, 2, 5},
                        {1, 16, 2, 5},
                        {1, 8, 2, 5},
                        3, MKLDNNPlugin::impl_desc_type::unknown, {}, {
                                [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                                    ASSERT_EQ(MKLDNNPlugin::impl_desc_type::ref, impl.getImplementationType());
                                    ASSERT_EQ(1, impl.getConfig().inConfs.size());
                                    ASSERT_EQ(2, impl.getConfig().outConfs.size());
                                    ASSERT_EQ(InferenceEngine::Layout::ANY, impl.getConfig().inConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::ANY, impl.getConfig().outConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::ANY, impl.getConfig().outConfs.at(1).desc.getLayout());
                                },
                                [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                                    ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
                                    ASSERT_EQ(1, impl.getConfig().inConfs.size());
                                    ASSERT_EQ(2, impl.getConfig().outConfs.size());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(1).desc.getLayout());
                                },
                                [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                                    ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
                                    ASSERT_EQ(1, impl.getConfig().inConfs.size());
                                    ASSERT_EQ(2, impl.getConfig().outConfs.size());
                                    ASSERT_EQ(InferenceEngine::Layout::BLOCKED, impl.getConfig().inConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::BLOCKED, impl.getConfig().outConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::BLOCKED, impl.getConfig().outConfs.at(1).desc.getLayout());
                                }
                        }
                },
                split_test_params {
                        {1, 20, 2, 5},
                        {1, 13, 2, 5},
                        {1, 7, 2, 5},
                        2, MKLDNNPlugin::impl_desc_type::unknown, {}, {
                                [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                                    ASSERT_EQ(MKLDNNPlugin::impl_desc_type::ref, impl.getImplementationType());
                                    ASSERT_EQ(1, impl.getConfig().inConfs.size());
                                    ASSERT_EQ(2, impl.getConfig().outConfs.size());
                                    ASSERT_EQ(InferenceEngine::Layout::ANY, impl.getConfig().inConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::ANY, impl.getConfig().outConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::ANY, impl.getConfig().outConfs.at(1).desc.getLayout());
                                },
                                [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                                    ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
                                    ASSERT_EQ(1, impl.getConfig().inConfs.size());
                                    ASSERT_EQ(2, impl.getConfig().outConfs.size());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(1).desc.getLayout());
                                }
                        }
                },
                split_test_params {
                        {1, 20, 2, 5},
                        {1, 10, 2, 5},
                        {1, 10, 2, 5},
                        2, MKLDNNPlugin::impl_desc_type::unknown, {}, {
                                [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                                    ASSERT_EQ(MKLDNNPlugin::impl_desc_type::ref, impl.getImplementationType());
                                    ASSERT_EQ(1, impl.getConfig().inConfs.size());
                                    ASSERT_EQ(2, impl.getConfig().outConfs.size());
                                    ASSERT_EQ(InferenceEngine::Layout::ANY, impl.getConfig().inConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::ANY, impl.getConfig().outConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::ANY, impl.getConfig().outConfs.at(1).desc.getLayout());
                                },
                                [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                                    ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
                                    ASSERT_EQ(1, impl.getConfig().inConfs.size());
                                    ASSERT_EQ(2, impl.getConfig().outConfs.size());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(1).desc.getLayout());
                                }
                        }
                },
                split_test_params {
                        {2, 20, 2, 5},
                        {2, 10, 2, 5},
                        {2, 10, 2, 5},
                        2, MKLDNNPlugin::impl_desc_type::unknown, {}, {
                                [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                                    ASSERT_EQ(MKLDNNPlugin::impl_desc_type::ref, impl.getImplementationType());
                                    ASSERT_EQ(1, impl.getConfig().inConfs.size());
                                    ASSERT_EQ(2, impl.getConfig().outConfs.size());
                                    ASSERT_EQ(InferenceEngine::Layout::ANY, impl.getConfig().inConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::ANY, impl.getConfig().outConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::ANY, impl.getConfig().outConfs.at(1).desc.getLayout());
                                },
                                [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                                    ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
                                    ASSERT_EQ(1, impl.getConfig().inConfs.size());
                                    ASSERT_EQ(2, impl.getConfig().outConfs.size());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(1).desc.getLayout());
                                }
                        }
                },
                split_test_params {
                        {1, 24, 2, 5},
                        {1, 16, 2, 5},
                        {1, 8, 2, 5},
                        3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}
                },
                split_test_params {
                        {1, 20, 2, 5},
                        {1, 13, 2, 5},
                        {1, 7, 2, 5},
                        2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}
                },
                split_test_params {
                        {1, 20, 2, 5},
                        {1, 10, 2, 5},
                        {1, 10, 2, 5},
                        2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}
                },
                split_test_params {
                        {2, 20, 2, 5},
                        {2, 10, 2, 5},
                        {2, 10, 2, 5},
                        2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}}));

class MKLDNNGraphDynBatchSplitTests: public MKLDNNGraphSplitTests {
protected:
    virtual void SetUp() {
        try {
            split_test_params p = ::testing::WithParamInterface<split_test_params>::GetParam();
            std::string model = getModel(p);
            size_t MB = p.in.n;
            if (MB < 2)
                MB = 2;

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));
            InferenceEngine::CNNNetwork network = net_reader.getNetwork();
            auto implNet = dynamic_cast<InferenceEngine::details::CNNNetworkImpl *>(&((InferenceEngine::ICNNNetwork&)network));
            ASSERT_NE(nullptr, implNet) << "Failed to cast ICNNNetwork to CNNNetworkImpl";
            InferenceEngine::ResponseDesc resp;
            InferenceEngine::StatusCode sts  = implNet->setBatchSizeReshape(MB, &resp);
            ASSERT_EQ((int)InferenceEngine::StatusCode::OK, sts) << resp.msg;

            MKLDNNGraphTestClass graph;
            graph.setProperty({{InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED, InferenceEngine::PluginConfigParams::YES}});
            graph.CreateGraph(net_reader.getNetwork());

            InferenceEngine::SizeVector dims_src = {MB, p.in.c, p.in.h, p.in.w};

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, InferenceEngine::NCHW, dims_src);
            src->allocate();
            fill_data(src->buffer(), src->size());

            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src));

            InferenceEngine::TBlob<float>* srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());

            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            InferenceEngine::OutputsDataMap out;
            out = net_reader.getNetwork().getOutputsInfo();
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

            auto checkSplit = [](const MKLDNNPlugin::MKLDNNNodePtr& node) {
                return node->getType() == MKLDNNPlugin::Split;
            };

            graph.checkDynBatch(srcs, outputBlobs, MB, MB, checkSplit);
            graph.checkDynBatch(srcs, outputBlobs, 1, MB, checkSplit);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphDynBatchSplitTests, TestsDynBatchSplit) {}

INSTANTIATE_TEST_CASE_P(
        TestsDynBatchSplit, MKLDNNGraphDynBatchSplitTests,
        ::testing::Values(
                split_test_params {
                        {1, 24, 2, 5},
                        {1, 16, 2, 5},
                        {1, 8, 2, 5},
                        3, MKLDNNPlugin::impl_desc_type::unknown, {}, {
                                [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                                    ASSERT_EQ(MKLDNNPlugin::impl_desc_type::ref, impl.getImplementationType());
                                    ASSERT_EQ(1, impl.getConfig().inConfs.size());
                                    ASSERT_EQ(2, impl.getConfig().outConfs.size());
                                    ASSERT_EQ(InferenceEngine::Layout::ANY, impl.getConfig().inConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::ANY, impl.getConfig().outConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::ANY, impl.getConfig().outConfs.at(1).desc.getLayout());
                                },
                                [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                                    ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
                                    ASSERT_EQ(1, impl.getConfig().inConfs.size());
                                    ASSERT_EQ(2, impl.getConfig().outConfs.size());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(1).desc.getLayout());
                                },
                                [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                                    ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
                                    ASSERT_EQ(1, impl.getConfig().inConfs.size());
                                    ASSERT_EQ(2, impl.getConfig().outConfs.size());
                                    ASSERT_EQ(InferenceEngine::Layout::BLOCKED, impl.getConfig().inConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::BLOCKED, impl.getConfig().outConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::BLOCKED, impl.getConfig().outConfs.at(1).desc.getLayout());
                                }
                        }
                },
                split_test_params {
                        {1, 20, 2, 5},
                        {1, 13, 2, 5},
                        {1, 7, 2, 5},
                        2, MKLDNNPlugin::impl_desc_type::unknown, {}, {
                                [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                                    ASSERT_EQ(MKLDNNPlugin::impl_desc_type::ref, impl.getImplementationType());
                                    ASSERT_EQ(1, impl.getConfig().inConfs.size());
                                    ASSERT_EQ(2, impl.getConfig().outConfs.size());
                                    ASSERT_EQ(InferenceEngine::Layout::ANY, impl.getConfig().inConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::ANY, impl.getConfig().outConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::ANY, impl.getConfig().outConfs.at(1).desc.getLayout());
                                },
                                [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                                    ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
                                    ASSERT_EQ(1, impl.getConfig().inConfs.size());
                                    ASSERT_EQ(2, impl.getConfig().outConfs.size());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(1).desc.getLayout());
                                }
                        }
                },
                split_test_params {
                        {1, 20, 2, 5},
                        {1, 10, 2, 5},
                        {1, 10, 2, 5},
                        2, MKLDNNPlugin::impl_desc_type::unknown, {}, {
                                [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                                    ASSERT_EQ(MKLDNNPlugin::impl_desc_type::ref, impl.getImplementationType());
                                    ASSERT_EQ(1, impl.getConfig().inConfs.size());
                                    ASSERT_EQ(2, impl.getConfig().outConfs.size());
                                    ASSERT_EQ(InferenceEngine::Layout::ANY, impl.getConfig().inConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::ANY, impl.getConfig().outConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::ANY, impl.getConfig().outConfs.at(1).desc.getLayout());
                                },
                                [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                                    ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
                                    ASSERT_EQ(1, impl.getConfig().inConfs.size());
                                    ASSERT_EQ(2, impl.getConfig().outConfs.size());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(1).desc.getLayout());
                                }
                        }
                },
                split_test_params {
                        {2, 20, 2, 5},
                        {2, 10, 2, 5},
                        {2, 10, 2, 5},
                        2, MKLDNNPlugin::impl_desc_type::unknown, {}, {
                                [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                                    ASSERT_EQ(MKLDNNPlugin::impl_desc_type::ref, impl.getImplementationType());
                                    ASSERT_EQ(1, impl.getConfig().inConfs.size());
                                    ASSERT_EQ(2, impl.getConfig().outConfs.size());
                                    ASSERT_EQ(InferenceEngine::Layout::ANY, impl.getConfig().inConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::ANY, impl.getConfig().outConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::ANY, impl.getConfig().outConfs.at(1).desc.getLayout());
                                },
                                [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                                    ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
                                    ASSERT_EQ(1, impl.getConfig().inConfs.size());
                                    ASSERT_EQ(2, impl.getConfig().outConfs.size());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(1).desc.getLayout());
                                }
                        }
                },
                split_test_params {
                        {1, 24, 2, 5},
                        {1, 16, 2, 5},
                        {1, 8, 2, 5},
                        3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}
                },
                split_test_params {
                        {1, 20, 2, 5},
                        {1, 13, 2, 5},
                        {1, 7, 2, 5},
                        2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}
                },
                split_test_params {
                        {1, 20, 2, 5},
                        {1, 10, 2, 5},
                        {1, 10, 2, 5},
                        2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}
                },
                split_test_params {
                        {2, 20, 2, 5},
                        {2, 10, 2, 5},
                        {2, 10, 2, 5},
                        2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}}));
