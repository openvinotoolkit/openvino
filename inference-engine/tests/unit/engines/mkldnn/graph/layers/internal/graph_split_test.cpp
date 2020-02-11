// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "mkldnn_graph.h"

#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include <mkldnn_extension_utils.h>
#include <cnn_network_impl.hpp>
#include "tests_common.hpp"

using namespace ::testing;
using namespace std;
using namespace mkldnn;

struct split_test_params {
    // Formats: NCHW, NCDHW
    vector<size_t> dims;
    std::vector<vector<size_t>> outs;

    int axis;

    size_t num_prim_desc;

    MKLDNNPlugin::impl_desc_type selectedType;
    std::vector<MKLDNNPlugin::impl_desc_type> preferTypes;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

template <typename data_t>
void ref_split(InferenceEngine::TBlob<data_t> &src, std::vector<InferenceEngine::TBlob<data_t>>& dsts, split_test_params& prm) {
    const float * srcData = src.readOnly();

    int outerSize = 1;
    for (int i = 0; i < prm.axis; i++)
        outerSize *= src.getTensorDesc().getDims()[i];

    for (size_t osIdx = 0; osIdx < outerSize; osIdx++) {
        for (size_t dstIdx = 0; dstIdx < dsts.size(); dstIdx++) {
            float* dstData = dsts[dstIdx].data();
            int innerSize = dsts[dstIdx].size() / outerSize;

            for (size_t j = 0; j < innerSize; j++, srcData++) {
                dstData[osIdx*innerSize + j] = *srcData;
            }
        }
    }
}

class MKLDNNGraphSplitTests: public TestsCommon,
                              public WithParamInterface<split_test_params> {
    std::string model_t = R"V0G0N(
<net name="ConcatOnly" version="3" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_ID_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
        <layer name="split" id="2" type="Split" precision="FP32">
            <split_data axis="_AXIS_" PrimitivesPriority="_IMPLS_"/>
            <input>
                <port id="1">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_ID_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </input>
            <output>
                _OP_
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
    </edges>
</net>
)V0G0N";

    std::string port_t = R"V0G0N(
<port id="_ID_">
    <dim>_N_</dim>
    <dim>_C_</dim>
    <dim>_D_</dim>
    <dim>_H_</dim>
    <dim>_W_</dim>
</port>
)V0G0N";

protected:
    std::string getModel(split_test_params p) {
        std::string model = model_t;
        auto dims_size = p.dims.size();

        switch (dims_size) {
            case 3:
                REMOVE_LINE(model, "<dim>_IH_</dim>");
            case 4:
                REMOVE_LINE(model, "<dim>_ID_</dim>");
        }
        REPLACE_WITH_NUM(model, "_IN_", p.dims[0]);
        REPLACE_WITH_NUM(model, "_IC_", p.dims[1]);
        REPLACE_WITH_NUM(model, "_IW_", p.dims[dims_size - 1]);
        switch (dims_size) {
            case 5:
                REPLACE_WITH_NUM(model, "_ID_", p.dims[dims_size - 3]);
            case 4:
                REPLACE_WITH_NUM(model, "_IH_", p.dims[dims_size - 2]);
        }

        std::string outPorts;
        for (int idx = 0; idx < p.outs.size(); idx++) {
            std::string outPort = port_t;
            switch (dims_size) {
                case 3:
                    REMOVE_LINE(outPort, "<dim>_H_</dim>");
                case 4:
                    REMOVE_LINE(outPort, "<dim>_D_</dim>");
            }
            REPLACE_WITH_NUM(outPort, "_ID_", idx);
            REPLACE_WITH_NUM(outPort, "_N_", p.outs[idx][0]);
            REPLACE_WITH_NUM(outPort, "_C_", p.outs[idx][1]);
            REPLACE_WITH_NUM(outPort, "_W_", p.outs[idx][dims_size - 1]);
            switch (dims_size) {
                case 5:
                    REPLACE_WITH_NUM(outPort, "_D_", p.outs[idx][dims_size - 3]);
                case 4:
                    REPLACE_WITH_NUM(outPort, "_H_", p.outs[idx][dims_size - 2]);
            }

            outPorts += outPort;
        }
        REPLACE_WITH_STR(model, "_OP_", outPorts);

        REPLACE_WITH_NUM(model, "_AXIS_", p.axis);

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
            net_reader.ReadNetwork(model.data(), model.length());

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

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>(net_reader.getNetwork().getInputsInfo().begin()->second->getTensorDesc());
            src->allocate();
            fill_data(src->buffer(), src->size());

            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src));

            auto srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());

            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            InferenceEngine::OutputsDataMap out;
            out = net_reader.getNetwork().getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;
            std::vector<InferenceEngine::TBlob<float>> dst_refs;
            for (auto& item : out) {
                InferenceEngine::TBlob<float>::Ptr output;
                output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
                output->allocate();
                outputBlobs[item.first] = output;

                InferenceEngine::TBlob<float> dst_ref(item.second->getTensorDesc());
                dst_ref.allocate();
                dst_refs.push_back(dst_ref);
            }

            graph.Infer(srcs, outputBlobs);

            ref_split(*srcPtr, dst_refs, p);

            int ref_idx = 0;
            for (auto& output : outputBlobs) {
                compare(*output.second, dst_refs[ref_idx++], 0.0005f);
            }
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
                        {{1, 16, 2, 5}, {1, 8, 2, 5}},
                        1, 3, MKLDNNPlugin::impl_desc_type::unknown, {}, {
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
                        {{1, 13, 2, 5}, {1, 7, 2, 5}},
                        1, 2, MKLDNNPlugin::impl_desc_type::unknown, {}, {
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
                        {{1, 10, 2, 5}, {1, 10, 2, 5}},
                        1, 2, MKLDNNPlugin::impl_desc_type::unknown, {}, {
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
                        {{2, 10, 2, 5}, {2, 10, 2, 5}},
                        1, 2, MKLDNNPlugin::impl_desc_type::unknown, {}, {
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
                        {{1, 16, 2, 5}, {1, 8, 2, 5}},
                        1, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}
                },
                split_test_params {
                        {1, 20, 2, 5},
                        {{1, 13, 2, 5}, {1, 7, 2, 5}},
                        1, 2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}
                },
                split_test_params {
                        {1, 20, 2, 5},
                        {{1, 10, 2, 5}, {1, 10, 2, 5}},
                        1, 2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}
                },
                split_test_params {
                        {2, 20, 2, 5},
                        {{2, 10, 2, 5}, {2, 10, 2, 5}},
                        1, 2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}
                },
                split_test_params {
                        {2, 20, 2, 5},
                        {{2, 15, 2, 5}, {2,  5, 2, 5}},
                        1, 2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}
                },
                split_test_params {
                        {9, 11, 7, 5},
                        {{3, 11, 7, 5}, {6, 11, 7, 5}},
                        0, 2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}
                },
                split_test_params {
                        {3, 11, 7, 5},
                        {{3, 11, 4, 5}, {3, 11, 3, 5}},
                        2, 2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}
                },
                split_test_params {
                        {3, 11, 7, 5},
                        {{3, 11, 7, 1}, {3, 11, 7, 4}},
                        3, 2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}
                },
                split_test_params {
                        {5, 6, 7, 15},
                        {{1, 6, 7, 15}, {2, 6, 7, 15}, {1, 6, 7, 15}, {1, 6, 7, 15}},
                        0, 2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}
                },
                split_test_params {
                        {5, 6, 7, 15},
                        {{5, 1, 7, 15}, {5, 2, 7, 15}, {5, 1, 7, 15}, {5, 2, 7, 15}},
                        1, 2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}
                },
                split_test_params {
                        {5, 6, 7, 15},
                        {{5, 6, 3, 15}, {5, 6, 1, 15}, {5, 6, 2, 15}, {5, 6, 1, 15}},
                        2, 2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}
                },
                split_test_params {
                        {5, 6, 7, 15},
                        {{5, 6, 7, 5}, {5, 6, 7, 3}, {5, 6, 7, 4}, {5, 6, 7, 3}},
                        3, 2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}
                },
                split_test_params {
                        {5, 6, 7, 15},
                        {{5, 6, 7, 15}},
                        1, 2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}},
                split_test_params {
                        {1, 32, 16, 16, 16},
                        {{1, 8, 16, 16, 16}, {1, 8, 16, 16, 16}, {1, 8, 16, 16, 16}, {1, 8, 16, 16, 16}},
                        1, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}},
                split_test_params {
                        {1, 32, 16, 16, 16},
                        {{1, 8, 16, 16, 16}, {1, 8, 16, 16, 16}, {1, 8, 16, 16, 16}, {1, 8, 16, 16, 16}},
                        1, 3, MKLDNNPlugin::impl_desc_type::unknown, {}}));

class MKLDNNGraphDynBatchSplitTests: public MKLDNNGraphSplitTests {
protected:
    virtual void SetUp() {
        try {
            split_test_params p = ::testing::WithParamInterface<split_test_params>::GetParam();
            std::string model = getModel(p);
            size_t MB = p.dims[0];
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

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>(net_reader.getNetwork().getInputsInfo().begin()->second->getTensorDesc());
            src->allocate();
            fill_data(src->buffer(), src->size());

            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src));

            auto* srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());

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
                        {{1, 16, 2, 5}, {1, 8, 2, 5}},
                        1, 3, MKLDNNPlugin::impl_desc_type::unknown, {}, {
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
                        {{1, 13, 2, 5}, {1, 7, 2, 5}},
                        1, 2, MKLDNNPlugin::impl_desc_type::unknown, {}, {
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
                        {{1, 10, 2, 5}, {1, 10, 2, 5}},
                        1, 2, MKLDNNPlugin::impl_desc_type::unknown, {}, {
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
                        {{2, 10, 2, 5}, {2, 10, 2, 5}},
                        1, 2, MKLDNNPlugin::impl_desc_type::unknown, {}, {
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
                        {2, 24, 2, 5},
                        {{2, 16, 2, 5}, {2, 8, 2, 5}},
                        1, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}
                },
                split_test_params {
                        {1, 20, 2, 5},
                        {{1, 13, 2, 5}, {1, 7, 2, 5}},
                        1, 2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}
                },
                split_test_params {
                        {1, 20, 2, 5},
                        {{1, 10, 2, 5}, {1, 10, 2, 5}},
                        1, 2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}
                },
                split_test_params {
                        {2, 20, 2, 5},
                        {{2, 10, 2, 5}, {2, 10, 2, 5}},
                        1, 2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}
                },
                split_test_params {
                        {2, 20, 2, 5},
                        {{2, 15, 2, 5}, {2,  5, 2, 5}},
                        1, 2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}
                },
                split_test_params {
                        {3, 11, 7, 5},
                        {{3, 11, 4, 5}, {3, 11, 3, 5}},
                        2, 2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}
                },
                split_test_params {
                        {3, 11, 7, 5},
                        {{3, 11, 7, 1}, {3, 11, 7, 4}},
                        3, 2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}
                },
                split_test_params {
                        {5, 6, 7, 15},
                        {{5, 1, 7, 15}, {5, 2, 7, 15}, {5, 1, 7, 15}, {5, 2, 7, 15}},
                        1, 2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}
                },
                split_test_params {
                        {5, 6, 7, 15},
                        {{5, 6, 3, 15}, {5, 6, 1, 15}, {5, 6, 2, 15}, {5, 6, 1, 15}},
                        2, 2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}
                },
                split_test_params {
                        {5, 6, 7, 15},
                        {{5, 6, 7, 5}, {5, 6, 7, 3}, {5, 6, 7, 4}, {5, 6, 7, 3}},
                        3, 2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref}}));
