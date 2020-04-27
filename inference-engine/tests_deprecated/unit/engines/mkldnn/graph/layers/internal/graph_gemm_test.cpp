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
#include <ie_core.hpp>
#include <ie_plugin_config.hpp>

using namespace ::testing;
using namespace std;
using namespace mkldnn;

struct gemm_test_params {
    struct {
        size_t MB1_A;
        size_t MB2_A;
        size_t MB1_B;
        size_t MB2_B;
        size_t MB1_C;
        size_t MB2_C;
        size_t MB1_D;
        size_t MB2_D;
    } batches;

    size_t M;
    size_t N;
    size_t K;

    float alpha;
    float beta;

    bool transposeA;
    bool transposeB;

    size_t num_prim_desc;

    MKLDNNPlugin::impl_desc_type selectedType;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

template<typename data_t>
void ref_gemm(const std::vector<InferenceEngine::TBlob<data_t>> &src, InferenceEngine::TBlob<data_t> &dst,
              gemm_test_params prm) {
    const data_t *src0_data = src[0].readOnly();
    const data_t *src1_data = src[1].readOnly();
    const data_t *src2_data = src.size() == 3 ? src[2].readOnly() : dst.readOnly();
    data_t *dst_data = dst.data();

    size_t MB1 = prm.batches.MB1_D;
    size_t MB2 = prm.batches.MB2_D;
    size_t M  = prm.M;
    size_t N  = prm.N;
    size_t K  = prm.K;

    for (int mb1 = 0; mb1 < MB1; mb1++) {
        const data_t *a_data = src0_data;
        const data_t *b_data = src1_data;
        const data_t *c_data = src2_data;
        data_t *d_data = dst_data;

        for (int mb2 = 0; mb2 < MB2; mb2++) {
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    d_data[i * N + j] = src.size() == 3 ? prm.beta * c_data[i * N + j] : 0;

                    for (int k = 0; k < K; k++) {
                        size_t src0_off = prm.transposeA ? k * M + i : i * K + k;
                        size_t src1_off = prm.transposeB ? j * K + k : k * N + j;
                        d_data[i * N + j] += prm.alpha * a_data[src0_off] * b_data[src1_off];
                    }
                }
            }
            a_data += prm.batches.MB2_A == MB2 ? M*K : 0;
            b_data += prm.batches.MB2_B == MB2 ? K*N : 0;
            c_data += prm.batches.MB2_C == MB2 ? M*N : 0;
            d_data += M*N;
        }

        src0_data += prm.batches.MB1_A == MB1 ? prm.batches.MB2_A*M*K : 0;
        src1_data += prm.batches.MB1_B == MB1 ? prm.batches.MB2_B*K*N : 0;
        src2_data += prm.batches.MB1_C == MB1 ? prm.batches.MB2_C*M*N : 0;
        dst_data += prm.batches.MB2_D*M*N;
    }
}

class MKLDNNGraphGemmTests: public TestsCommon,
                                     public WithParamInterface<gemm_test_params> {
    std::string model_t = R"V0G0N(
<net name="gemmOnly" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    <dim>_MB1_A_</dim>
                    <dim>_MB2_A_</dim>
                    <dim>_M_A_</dim>
                    <dim>_N_A_</dim>
                </port>
            </output>
        </layer>
        <layer name="in2" type="Input" precision="FP32" id="2">
            <output>
                <port id="1">
                    <dim>_MB1_B_</dim>
                    <dim>_MB2_B_</dim>
                    <dim>_M_B_</dim>
                    <dim>_N_B_</dim>
                </port>
            </output>
        </layer>
        <layer name="in3" type="Input" precision="FP32" id="3">
            <output>
                <port id="1">
                    <dim>_MB1_C_</dim>
                    <dim>_MB2_C_</dim>
                    <dim>_M_</dim>
                    <dim>_N_</dim>
                </port>
            </output>
        </layer>
        <layer name="gemm" id="4" type="GEMM" precision="FP32">
            <data alpha="_A_" beta="_B_" transpose_a="_TA_" transpose_b="_TB_"/>
            <input>
                <port id="1">
                    <dim>_MB1_A_</dim>
                    <dim>_MB2_A_</dim>
                    <dim>_M_A_</dim>
                    <dim>_N_A_</dim>
                </port>
                <port id="2">
                    <dim>_MB1_B_</dim>
                    <dim>_MB2_B_</dim>
                    <dim>_M_B_</dim>
                    <dim>_N_B_</dim>
                </port>
                <port id="3">
                    <dim>_MB1_C_</dim>
                    <dim>_MB2_C_</dim>
                    <dim>_M_</dim>
                    <dim>_N_</dim>
                </port>
            </input>
            <output>
                <port id="4">
                    <dim>_MB1_D_</dim>
                    <dim>_MB2_D_</dim>
                    <dim>_M_</dim>
                    <dim>_N_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="4" to-port="1"/>
        <edge from-layer="2" from-port="1" to-layer="4" to-port="2"/>
        <edge from-layer="3" from-port="1" to-layer="4" to-port="3"/>
    </edges>
</net>
)V0G0N";

protected:
    std::string getModel(gemm_test_params p) {
        std::string model = model_t;
        std::string op;

        REPLACE_WITH_NUM(model, "_MB1_A_", p.batches.MB1_A);
        REPLACE_WITH_NUM(model, "_MB2_A_", p.batches.MB2_A);
        REPLACE_WITH_NUM(model, "_MB1_B_", p.batches.MB1_B);
        REPLACE_WITH_NUM(model, "_MB2_B_", p.batches.MB2_B);
        REPLACE_WITH_NUM(model, "_MB1_C_", p.batches.MB1_C);
        REPLACE_WITH_NUM(model, "_MB2_C_", p.batches.MB2_C);
        REPLACE_WITH_NUM(model, "_MB1_D_", p.batches.MB1_D);
        REPLACE_WITH_NUM(model, "_MB2_D_", p.batches.MB2_D);

        auto m_A = p.transposeA ? p.K : p.M;
        auto n_A = p.transposeA ? p.M : p.K;
        auto m_B = p.transposeB ? p.N : p.K;
        auto n_B = p.transposeB ? p.K : p.N;

        REPLACE_WITH_NUM(model, "_M_A_", m_A);
        REPLACE_WITH_NUM(model, "_N_A_", n_A);
        REPLACE_WITH_NUM(model, "_M_B_", m_B);
        REPLACE_WITH_NUM(model, "_N_B_", n_B);

        REPLACE_WITH_NUM(model, "_M_", p.M);
        REPLACE_WITH_NUM(model, "_N_", p.N);
        REPLACE_WITH_NUM(model, "_K_", p.K);

        REPLACE_WITH_NUM(model, "_A_", p.alpha);
        REPLACE_WITH_NUM(model, "_B_", p.beta);
        REPLACE_WITH_NUM(model, "_TA_", p.transposeA);
        REPLACE_WITH_NUM(model, "_TB_", p.transposeB);

        return model;
    }

    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            gemm_test_params p = ::testing::WithParamInterface<gemm_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;
            ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(network);

            auto& nodes = graph.getNodes();
            for (int i = 0; i < nodes.size(); i++) {
                if (nodes[i]->getType() == MKLDNNPlugin::Gemm) {
                    ASSERT_EQ(p.num_prim_desc, nodes[i]->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(nodes[i]->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, nodes[i]->getSelectedPrimitiveDescriptor());
                    ASSERT_EQ(p.selectedType, nodes[i]->getSelectedPrimitiveDescriptor()->getImplementationType());
                }
            }

            auto m_A = p.transposeA ? p.K : p.M;
            auto n_A = p.transposeA ? p.M : p.K;
            auto m_B = p.transposeB ? p.N : p.K;
            auto n_B = p.transposeB ? p.K : p.N;

            InferenceEngine::SizeVector dims_src1 = {p.batches.MB1_A, p.batches.MB2_A, m_A, n_A};
            InferenceEngine::SizeVector dims_src2 = {p.batches.MB1_B, p.batches.MB2_B, m_B, n_B};
            InferenceEngine::SizeVector dims_src3 = {p.batches.MB1_C, p.batches.MB2_C, p.M, p.N};
            InferenceEngine::SizeVector dims_dst  = {p.batches.MB1_D, p.batches.MB2_D, p.M, p.N};

            InferenceEngine::Blob::Ptr src1 = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src1, InferenceEngine::NCHW});
            src1->allocate();
            InferenceEngine::TBlob<float>* srcPtr1 = dynamic_cast<InferenceEngine::TBlob<float>*>(src1.get());
            if (srcPtr1 == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";
            fill_data(src1->buffer(), src1->size());

            InferenceEngine::Blob::Ptr src2 = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src2, InferenceEngine::NCHW});
            src2->allocate();
            InferenceEngine::TBlob<float>* srcPtr2 = dynamic_cast<InferenceEngine::TBlob<float>*>(src2.get());
            if (srcPtr2 == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";
            fill_data(src2->buffer(), src2->size());

            InferenceEngine::Blob::Ptr src3 = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src3, InferenceEngine::NCHW});
            src3->allocate();
            InferenceEngine::TBlob<float>* srcPtr3 = dynamic_cast<InferenceEngine::TBlob<float>*>(src3.get());
            if (srcPtr3 == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";
            fill_data(src3->buffer(), src3->size());

            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src1));
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in2", src2));
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in3", src3));

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

            std::vector<InferenceEngine::TBlob<float>> src_vec = {*srcPtr1, *srcPtr2, *srcPtr3};

            ref_gemm(src_vec, dst_ref, p);

            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphGemmTests, TestsGemm) {}

INSTANTIATE_TEST_CASE_P(
        TestsGemm, MKLDNNGraphGemmTests,
        ::testing::Values(
                gemm_test_params{{2, 1, 2, 1, 2, 1, 2, 1}, 3, 3, 2, 1, 1, false, false, 1, MKLDNNPlugin::impl_desc_type::gemm_any, {
                        [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::gemm_any, impl.getImplementationType());
                            ASSERT_EQ(3, impl.getConfig().inConfs.size());
                            ASSERT_EQ(1, impl.getConfig().outConfs.size());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(1).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(2).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                        }
                } },
                gemm_test_params{{3, 2, 3, 2, 3, 2, 3, 2}, 8, 5, 4, 1, 1, false, false, 1, MKLDNNPlugin::impl_desc_type::gemm_any, {
                        [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::gemm_any, impl.getImplementationType());
                            ASSERT_EQ(3, impl.getConfig().inConfs.size());
                            ASSERT_EQ(1, impl.getConfig().outConfs.size());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(1).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(2).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                        }
                } },
                gemm_test_params{{3, 2, 3, 2, 3, 2, 3, 2}, 16, 10, 12, 1, 1, false, false, 1, MKLDNNPlugin::impl_desc_type::gemm_any, {
                        [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::gemm_any, impl.getImplementationType());
                            ASSERT_EQ(3, impl.getConfig().inConfs.size());
                            ASSERT_EQ(1, impl.getConfig().outConfs.size());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(1).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(2).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                        }
                } },
                gemm_test_params{{3, 2, 3, 2, 3, 2, 3, 2}, 11, 10, 20, 1, 1, false, false, 1, MKLDNNPlugin::impl_desc_type::gemm_any, {
                        [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::gemm_any, impl.getImplementationType());
                            ASSERT_EQ(3, impl.getConfig().inConfs.size());
                            ASSERT_EQ(1, impl.getConfig().outConfs.size());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(1).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(2).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                        }
                } },
                gemm_test_params{{3, 2, 3, 2, 3, 2, 3, 2}, 5, 13, 2, 1, 1, false, false, 1, MKLDNNPlugin::impl_desc_type::gemm_any, {
                        [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::gemm_any, impl.getImplementationType());
                            ASSERT_EQ(3, impl.getConfig().inConfs.size());
                            ASSERT_EQ(1, impl.getConfig().outConfs.size());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(1).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(2).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                        }
                } },
                gemm_test_params{{3, 2, 3, 2, 3, 2, 3, 2}, 5, 15, 10, 1, 1, false, false, 1, MKLDNNPlugin::impl_desc_type::gemm_any, {
                        [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::gemm_any, impl.getImplementationType());
                            ASSERT_EQ(3, impl.getConfig().inConfs.size());
                            ASSERT_EQ(1, impl.getConfig().outConfs.size());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(1).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(2).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                        }
                } },
                gemm_test_params{{3, 2, 3, 2, 3, 2, 3, 2}, 5, 6, 7, 2, 0, false, false, 1, MKLDNNPlugin::impl_desc_type::gemm_any},
                gemm_test_params{{3, 2, 3, 2, 3, 2, 3, 2}, 5, 6, 7, 0, 2, false, false, 1, MKLDNNPlugin::impl_desc_type::gemm_any},
                gemm_test_params{{3, 2, 3, 2, 3, 2, 3, 2}, 3, 7, 4, 2, 3, true, false, 1, MKLDNNPlugin::impl_desc_type::gemm_any},
                gemm_test_params{{3, 2, 3, 2, 3, 2, 3, 2}, 7, 3, 4, 2, 3, true, false, 1, MKLDNNPlugin::impl_desc_type::gemm_any},
                gemm_test_params{{3, 2, 3, 2, 3, 2, 3, 2}, 7, 4, 3, 2, 3, true, false, 1, MKLDNNPlugin::impl_desc_type::gemm_any},
                gemm_test_params{{3, 2, 3, 2, 3, 2, 3, 2}, 3, 7, 4, 2, 3, false, true, 1, MKLDNNPlugin::impl_desc_type::gemm_any},
                gemm_test_params{{3, 2, 3, 2, 3, 2, 3, 2}, 7, 3, 4, 2, 3, false, true, 1, MKLDNNPlugin::impl_desc_type::gemm_any},
                gemm_test_params{{3, 2, 3, 2, 3, 2, 3, 2}, 7, 4, 3, 2, 3, false, true, 1, MKLDNNPlugin::impl_desc_type::gemm_any},
                gemm_test_params{{3, 2, 3, 2, 3, 2, 3, 2}, 3, 7, 4, 2, 3, true, true, 1, MKLDNNPlugin::impl_desc_type::gemm_any},
                gemm_test_params{{3, 2, 3, 2, 3, 2, 3, 2}, 7, 3, 4, 2, 3, true, true, 1, MKLDNNPlugin::impl_desc_type::gemm_any},
                gemm_test_params{{3, 2, 3, 2, 3, 2, 3, 2}, 7, 4, 3, 2, 3, true, true, 1, MKLDNNPlugin::impl_desc_type::gemm_any},
                gemm_test_params{{1, 3, 2, 3, 2, 3, 2, 3}, 7, 4, 3, 2, 3, true, false, 1, MKLDNNPlugin::impl_desc_type::gemm_any},
                gemm_test_params{{1, 3, 2, 3, 1, 3, 2, 3}, 7, 4, 3, 2, 3, false, false, 1, MKLDNNPlugin::impl_desc_type::gemm_any},
                gemm_test_params{{2, 3, 1, 3, 1, 3, 2, 3}, 7, 4, 3, 2, 3, false, true, 1, MKLDNNPlugin::impl_desc_type::gemm_any},
                gemm_test_params{{5, 3, 5, 1, 5, 3, 5, 3}, 7, 4, 3, 2, 3, true, false, 1, MKLDNNPlugin::impl_desc_type::gemm_any},
                gemm_test_params{{5, 3, 5, 1, 5, 1, 5, 3}, 7, 4, 3, 2, 3, false, true, 1, MKLDNNPlugin::impl_desc_type::gemm_any},
                gemm_test_params{{5, 1, 5, 1, 5, 3, 5, 3}, 7, 4, 3, 2, 3, true, false, 1, MKLDNNPlugin::impl_desc_type::gemm_any},
                gemm_test_params{{1, 1, 5, 3, 5, 3, 5, 3}, 7, 4, 3, 2, 3, false, false, 1, MKLDNNPlugin::impl_desc_type::gemm_any},
                gemm_test_params{{1, 1, 1, 1, 5, 3, 5, 3}, 7, 4, 3, 2, 3, true, true, 1, MKLDNNPlugin::impl_desc_type::gemm_any},
                gemm_test_params{{5, 4, 1, 1, 1, 1, 5, 4}, 7, 4, 3, 2, 3, false, false, 1, MKLDNNPlugin::impl_desc_type::gemm_any}
        ));

class MKLDNNGraphDynBatchGemmTests: public MKLDNNGraphGemmTests {
protected:
    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            gemm_test_params p = ::testing::WithParamInterface<gemm_test_params>::GetParam();
            std::string model = getModel(p);
            size_t MB = p.batches.MB1_D;
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

            auto m_A = p.transposeA ? p.K : p.M;
            auto n_A = p.transposeA ? p.M : p.K;
            auto m_B = p.transposeB ? p.N : p.K;
            auto n_B = p.transposeB ? p.K : p.N;

            InferenceEngine::SizeVector dims_src1 = {MB, p.batches.MB2_A, m_A, n_A};
            InferenceEngine::SizeVector dims_src2 = {MB, p.batches.MB2_B, m_B, n_B};
            InferenceEngine::SizeVector dims_src3 = {MB, p.batches.MB2_C, p.M, p.N};
            InferenceEngine::SizeVector dims_dst  = {MB, p.batches.MB2_D, p.M, p.N};

            InferenceEngine::Blob::Ptr src1 = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src1, InferenceEngine::NCHW});
            src1->allocate();
            InferenceEngine::TBlob<float>* srcPtr1 = dynamic_cast<InferenceEngine::TBlob<float>*>(src1.get());
            if (srcPtr1 == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";
            fill_data(src1->buffer(), src1->size());

            InferenceEngine::Blob::Ptr src2 = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src2, InferenceEngine::NCHW});
            src2->allocate();
            InferenceEngine::TBlob<float>* srcPtr2 = dynamic_cast<InferenceEngine::TBlob<float>*>(src2.get());
            if (srcPtr2 == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";
            fill_data(src2->buffer(), src2->size());

            InferenceEngine::Blob::Ptr src3 = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src3, InferenceEngine::NCHW});
            src3->allocate();
            InferenceEngine::TBlob<float>* srcPtr3 = dynamic_cast<InferenceEngine::TBlob<float>*>(src3.get());
            if (srcPtr3 == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";
            fill_data(src3->buffer(), src3->size());

            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src1));
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in2", src2));
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in3", src3));

            InferenceEngine::OutputsDataMap out;
            out = network.getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;

            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;

            auto check = [](const MKLDNNPlugin::MKLDNNNodePtr& node) {
                return node->getType() == MKLDNNPlugin::Gemm;
            };

            graph.checkDynBatch(srcs, outputBlobs, MB, MB, check);
            graph.checkDynBatch(srcs, outputBlobs, 1, MB, check);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphDynBatchGemmTests, TestsDynBatchGemm) {}

INSTANTIATE_TEST_CASE_P(
        TestsDynBatchGemm, MKLDNNGraphDynBatchGemmTests,
        ::testing::Values(
                gemm_test_params{{1, 3, 1, 3, 1, 3, 1, 3}, 3, 3, 3, 1, 1, false, false, 1, MKLDNNPlugin::impl_desc_type::gemm_any},
                gemm_test_params{{1, 3, 1, 1, 1, 3, 1, 3}, 16, 15, 12, 1, 1, false, false, 1, MKLDNNPlugin::impl_desc_type::gemm_any}
));

class MKLDNNGraphSingleBatchDimGemmTests: public TestsCommon,
                                     public WithParamInterface<gemm_test_params> {
    std::string model_t = R"V0G0N(
<net name="gemmOnly" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    <dim>_MB_A_</dim>
                    <dim>_M_A_</dim>
                    <dim>_N_A_</dim>
                </port>
            </output>
        </layer>
        <layer name="in2" type="Input" precision="FP32" id="2">
            <output>
                <port id="1">
                    <dim>_MB_B_</dim>
                    <dim>_M_B_</dim>
                    <dim>_N_B_</dim>
                </port>
            </output>
        </layer>
        <layer name="gemm" id="3" type="GEMM" precision="FP32">
            <data alpha="_A_" beta="_B_" transpose_a="_TA_" transpose_b="_TB_"/>
            <input>
                <port id="1">
                    <dim>_MB_A_</dim>
                    <dim>_M_A_</dim>
                    <dim>_N_A_</dim>
                </port>
                <port id="2">
                    <dim>_MB_B_</dim>
                    <dim>_M_B_</dim>
                    <dim>_N_B_</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>_MB_D_</dim>
                    <dim>_M_</dim>
                    <dim>_N_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="1" to-layer="3" to-port="2"/>
    </edges>
</net>
)V0G0N";

protected:
    std::string getModel(gemm_test_params p) {
        std::string model = model_t;
        std::string op;

        REPLACE_WITH_NUM(model, "_MB_A_", p.batches.MB2_A);
        REPLACE_WITH_NUM(model, "_MB_B_", p.batches.MB2_B);
        REPLACE_WITH_NUM(model, "_MB_D_", p.batches.MB2_D);

        auto m_A = p.transposeA ? p.K : p.M;
        auto n_A = p.transposeA ? p.M : p.K;
        auto m_B = p.transposeB ? p.N : p.K;
        auto n_B = p.transposeB ? p.K : p.N;

        REPLACE_WITH_NUM(model, "_M_A_", m_A);
        REPLACE_WITH_NUM(model, "_N_A_", n_A);
        REPLACE_WITH_NUM(model, "_M_B_", m_B);
        REPLACE_WITH_NUM(model, "_N_B_", n_B);

        REPLACE_WITH_NUM(model, "_M_", p.M);
        REPLACE_WITH_NUM(model, "_N_", p.N);
        REPLACE_WITH_NUM(model, "_K_", p.K);

        REPLACE_WITH_NUM(model, "_A_", p.alpha);
        REPLACE_WITH_NUM(model, "_B_", p.beta);
        REPLACE_WITH_NUM(model, "_TA_", p.transposeA);
        REPLACE_WITH_NUM(model, "_TB_", p.transposeB);

        return model;
    }

    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            gemm_test_params p = ::testing::WithParamInterface<gemm_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;
            ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(network);

            auto& nodes = graph.getNodes();
            for (int i = 0; i < nodes.size(); i++) {
                if (nodes[i]->getType() == MKLDNNPlugin::Gemm) {
                    ASSERT_EQ(p.num_prim_desc, nodes[i]->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(nodes[i]->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, nodes[i]->getSelectedPrimitiveDescriptor());
                    ASSERT_EQ(p.selectedType, nodes[i]->getSelectedPrimitiveDescriptor()->getImplementationType());
                }
            }

            auto m_A = p.transposeA ? p.K : p.M;
            auto n_A = p.transposeA ? p.M : p.K;
            auto m_B = p.transposeB ? p.N : p.K;
            auto n_B = p.transposeB ? p.K : p.N;

            InferenceEngine::SizeVector dims_src1 = {p.batches.MB2_A, m_A, n_A};
            InferenceEngine::SizeVector dims_src2 = {p.batches.MB2_B, m_B, n_B};
            InferenceEngine::SizeVector dims_dst  = {p.batches.MB2_D, p.M, p.N};

            InferenceEngine::Blob::Ptr src1 = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src1, InferenceEngine::CHW});
            src1->allocate();
            InferenceEngine::TBlob<float>* srcPtr1 = dynamic_cast<InferenceEngine::TBlob<float>*>(src1.get());
            if (srcPtr1 == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";
            fill_data(src1->buffer(), src1->size());

            InferenceEngine::Blob::Ptr src2 = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src2, InferenceEngine::CHW});
            src2->allocate();
            InferenceEngine::TBlob<float>* srcPtr2 = dynamic_cast<InferenceEngine::TBlob<float>*>(src2.get());
            if (srcPtr2 == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";
            fill_data(src2->buffer(), src2->size());

            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src1));
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in2", src2));

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

            std::vector<InferenceEngine::TBlob<float>> src_vec = {*srcPtr1, *srcPtr2};

            ref_gemm(src_vec, dst_ref, p);

            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphSingleBatchDimGemmTests, TestsGemm) {}

INSTANTIATE_TEST_CASE_P(
        TestsGemm, MKLDNNGraphSingleBatchDimGemmTests,
        ::testing::Values(
                gemm_test_params{{1, 1, 1, 1, 1, 1, 1, 1}, 7, 4, 3, 2, 3, false, false, 1, MKLDNNPlugin::impl_desc_type::gemm_any},
                gemm_test_params{{1, 3, 1, 3, 1, 1, 1, 3}, 7, 4, 3, 2, 3, false, false, 1, MKLDNNPlugin::impl_desc_type::gemm_any},
                gemm_test_params{{1, 3, 1, 1, 1, 1, 1, 3}, 7, 4, 3, 2, 3, false, false, 1, MKLDNNPlugin::impl_desc_type::gemm_any},
                gemm_test_params{{1, 1, 1, 1, 1, 1, 1, 1}, 7, 4, 3, 2, 3, true, false, 1, MKLDNNPlugin::impl_desc_type::gemm_any},
                gemm_test_params{{1, 3, 1, 3, 1, 1, 1, 3}, 7, 4, 3, 2, 3, true, false, 1, MKLDNNPlugin::impl_desc_type::gemm_any},
                gemm_test_params{{1, 3, 1, 1, 1, 1, 1, 3}, 7, 4, 3, 2, 3, true, false, 1, MKLDNNPlugin::impl_desc_type::gemm_any},
                gemm_test_params{{1, 1, 1, 1, 1, 1, 1, 1}, 7, 4, 3, 2, 3, false, true, 1, MKLDNNPlugin::impl_desc_type::gemm_any},
                gemm_test_params{{1, 3, 1, 3, 1, 1, 1, 3}, 7, 4, 3, 2, 3, false, true, 1, MKLDNNPlugin::impl_desc_type::gemm_any},
                gemm_test_params{{1, 3, 1, 1, 1, 1, 1, 3}, 7, 4, 3, 2, 3, false, true, 1, MKLDNNPlugin::impl_desc_type::gemm_any},
                gemm_test_params{{1, 1, 1, 1, 1, 1, 1, 1}, 7, 4, 3, 2, 3, true, true, 1, MKLDNNPlugin::impl_desc_type::gemm_any},
                gemm_test_params{{1, 3, 1, 3, 1, 1, 1, 3}, 7, 4, 3, 2, 3, true, true, 1, MKLDNNPlugin::impl_desc_type::gemm_any},
                gemm_test_params{{1, 3, 1, 1, 1, 1, 1, 3}, 7, 4, 3, 2, 3, true, true, 1, MKLDNNPlugin::impl_desc_type::gemm_any}
        ));
