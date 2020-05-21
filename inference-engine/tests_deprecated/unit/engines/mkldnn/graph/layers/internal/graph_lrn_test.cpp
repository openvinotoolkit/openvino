// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include "tests_common.hpp"
#include <ie_core.hpp>
#include <ie_plugin_config.hpp>

using namespace ::testing;
using namespace std;
using namespace mkldnn;


struct lrn_test_params {
    struct {
        size_t n;
        size_t c;
        size_t h;
        size_t w;
    } in;

    size_t local_size;
    float alpha;
    float beta;
    size_t k;

    size_t num_prim_desc;

    int selectedType;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

template <typename data_t>
void ref_lrn(const InferenceEngine::TBlob<data_t> &src, InferenceEngine::TBlob<data_t> &dst, lrn_test_params prm)
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

                uint32_t sz = prm.local_size;
                int32_t c_start = c - sz / 2;
                int32_t c_end = c_start + sz;
                if (c_start < 0) c_start = 0;
                if (c_end > (int32_t)IC) c_end = IC;
                data_t sum = 0.0;
                for (int32_t c1 = c_start; c1 < c_end; c1++) {
                    uint32_t idx = c1 * IH * IW + h * IW + w;
                    data_t s = src_data[idx];

                    sum += s * s;
                }

                data_t norm_coef = powf(1. + prm.alpha * sum / sz, -prm.beta);
                dst_data[oidx] = norm_coef * src_data[oidx];
            }
        }
    }
}

class MKLDNNGraphLrnTests: public TestsCommon,
                                     public WithParamInterface<lrn_test_params> {
    std::string model_t = R"V0G0N(
<Net Name="Lrn_Only" version="2" precision="FP32" batch="1">
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
        <layer name="norm" id="1" type="LRN" precision="FP32">
            <lrn local_size="_LS_" alpha="_A_" beta="_B_" k="_K_" region="ACROSS" />

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

protected:
    std::string getModel(lrn_test_params p) {
        std::string model = model_t;

        REPLACE_WITH_NUM(model, "_IW_", p.in.w);
        REPLACE_WITH_NUM(model, "_IH_", p.in.h);
        REPLACE_WITH_NUM(model, "_IC_", p.in.c);
        REPLACE_WITH_NUM(model, "_IN_", p.in.n);

        REPLACE_WITH_NUM(model, "_LS_", p.local_size);
        REPLACE_WITH_NUM(model, "_A_", p.alpha);
        REPLACE_WITH_NUM(model, "_B_", p.beta);
        REPLACE_WITH_NUM(model, "_K_", p.k);

        return model;
    }

    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            lrn_test_params p = ::testing::WithParamInterface<lrn_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;
            ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(network);
            auto& nodes = graph.getNodes();
            for (int i = 0; i < nodes.size(); i++) {
                if (nodes[i]->getType() == MKLDNNPlugin::Lrn) {
                    ASSERT_LE(p.num_prim_desc, nodes[i]->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(nodes[i]->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, nodes[i]->getSelectedPrimitiveDescriptor());
                    ASSERT_EQ(p.selectedType,
                              nodes[i]->getSelectedPrimitiveDescriptor()->getImplementationType() & p.selectedType);
                }
            }
            if (nodes.size() != 3 && nodes.size() != 5)
                FAIL() << "Nodes amount should be 3 or 5 (in reorder case)";

            InferenceEngine::SizeVector dims_src = {p.in.n, p.in.c, p.in.h, p.in.w};

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src, InferenceEngine::NCHW});
            src->allocate();
            fill_data(src->buffer(), src->size());

            InferenceEngine::TBlob<float>* srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());

            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src));

            InferenceEngine::OutputsDataMap out;
            out = network.getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;

            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;

            graph.Infer(srcs, outputBlobs);

            InferenceEngine::TBlob<float> dst_ref(item.second->getTensorDesc());
            dst_ref.allocate();

            ref_lrn(*srcPtr, dst_ref, p);

            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphLrnTests, TestsLrn) {}

INSTANTIATE_TEST_CASE_P(
        TestsLrn, MKLDNNGraphLrnTests,
        ::testing::Values(
                lrn_test_params{
                        {1, 3, 228, 228},
                        5, 0.0001f, 0.75f, 1, 3, MKLDNNPlugin::impl_desc_type::ref_any, {
                                [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                                    ASSERT_EQ(MKLDNNPlugin::impl_desc_type::ref_any, impl.getImplementationType());
                                    ASSERT_EQ(1, impl.getConfig().inConfs.size());
                                    ASSERT_EQ(1, impl.getConfig().outConfs.size());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                                },
                                [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                                    ASSERT_EQ(MKLDNNPlugin::impl_desc_type::ref_any, impl.getImplementationType());
                                    ASSERT_EQ(1, impl.getConfig().inConfs.size());
                                    ASSERT_EQ(1, impl.getConfig().outConfs.size());
                                    ASSERT_EQ(InferenceEngine::Layout::BLOCKED, impl.getConfig().inConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::BLOCKED, impl.getConfig().outConfs.at(0).desc.getLayout());
                                },
                                [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                                    ASSERT_EQ(MKLDNNPlugin::impl_desc_type::ref_any, impl.getImplementationType());
                                    ASSERT_EQ(1, impl.getConfig().inConfs.size());
                                    ASSERT_EQ(1, impl.getConfig().outConfs.size());
                                    ASSERT_EQ(InferenceEngine::Layout::BLOCKED, impl.getConfig().inConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::BLOCKED, impl.getConfig().outConfs.at(0).desc.getLayout());
                                }
                        }},
                lrn_test_params{{1, 16, 228, 228}, 5, 0.0001f, 0.75f, 1, 3, MKLDNNPlugin::impl_desc_type::jit}));

class MKLDNNGraphDynBatchLrnTests: public MKLDNNGraphLrnTests {
protected:
    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            lrn_test_params p = ::testing::WithParamInterface<lrn_test_params>::GetParam();
            std::string model = getModel(p);
            size_t MB = p.in.n;
            if (MB < 2)
                MB = 2;

            InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;
            ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

            auto implNet = dynamic_cast<InferenceEngine::details::CNNNetworkImpl *>(&((InferenceEngine::ICNNNetwork&)network));
            ASSERT_NE(nullptr, implNet) << "Failed to cast ICNNNetwork to CNNNetworkImpl";
            InferenceEngine::ResponseDesc resp;
            InferenceEngine::StatusCode sts  = implNet->setBatchSizeReshape(MB, &resp);
            ASSERT_EQ((int)InferenceEngine::StatusCode::OK, sts) << resp.msg;

            MKLDNNGraphTestClass graph;
            graph.setProperty({{InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED, InferenceEngine::PluginConfigParams::YES}});
            graph.CreateGraph(network);

            InferenceEngine::SizeVector dims_src = {MB, p.in.c, p.in.h, p.in.w};

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src, InferenceEngine::NCHW});
            src->allocate();
            fill_data(src->buffer(), src->size());

            InferenceEngine::TBlob<float>* srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());

            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src));

            InferenceEngine::OutputsDataMap out;
            out = network.getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;

            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;

            auto checkLRN = [](const MKLDNNPlugin::MKLDNNNodePtr& node) {
                return node->getType() == MKLDNNPlugin::Lrn;
            };
            graph.checkDynBatch(srcs, outputBlobs, MB, MB, checkLRN);
            graph.checkDynBatch(srcs, outputBlobs, 1, MB, checkLRN);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphDynBatchLrnTests, TestsDynBatchLrn) {}

INSTANTIATE_TEST_CASE_P(
        TestsDynBatchLrn, MKLDNNGraphDynBatchLrnTests,
        ::testing::Values(
                lrn_test_params{{1, 3, 228, 228}, 5, 0.0001f, 0.75f, 1, 3, MKLDNNPlugin::impl_desc_type::ref_any},
                lrn_test_params{{1, 16, 228, 228}, 5, 0.0001f, 0.75f, 1, 3, MKLDNNPlugin::impl_desc_type::jit}));
