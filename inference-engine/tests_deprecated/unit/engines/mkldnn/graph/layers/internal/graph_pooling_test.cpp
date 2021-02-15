// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "test_graph.hpp"

#include <ie_plugin_config.hpp>
#include "single_layer_common.hpp"
#include <legacy/ie_layers.h>
#include "tests_common.hpp"
#include "ir_gen_helper.hpp"
#include <math.h>

#include <ie_core.hpp>

using namespace InferenceEngine;
using namespace ::testing;
using namespace std;
using namespace mkldnn;
using namespace single_layer_tests;

struct pooling_test_params {
    // Formats: NCHW, NCDHW
    vector<size_t> dims;
    // Formats: WH, WHD
    vector<size_t> kernel;
    vector<size_t> strides;
    vector<size_t> pads_begin;
    vector<size_t> pads_end;

    PoolingLayer::PoolType _type;
    bool _exclude_pad;

    size_t num_prim_desc;

    MKLDNNPlugin::impl_desc_type selectedType;
    vector<MKLDNNPlugin::impl_desc_type> preferTypes;

    vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

template <typename data_t>
void ref_pool(const InferenceEngine::TBlob<data_t> &src, InferenceEngine::TBlob<data_t> &dst, pooling_test_params prm)
{
    int dims_size = prm.dims.size();

    int KW = prm.kernel[X_AXIS];
    int KH = prm.kernel[Y_AXIS];
    int KD = dims_size == 5 ? prm.kernel[Z_AXIS] : 1;

    int SW = prm.strides[X_AXIS];
    int SH = prm.strides[Y_AXIS];
    int SD = prm.strides.size() > Z_AXIS ? prm.strides[Z_AXIS] : 1;

    int IW = prm.dims[dims_size - 1];
    int IH = prm.dims[dims_size - 2];
    int ID = dims_size == 5 ? prm.dims[dims_size - 3] : 1;
    
    int PWB = prm.pads_begin[X_AXIS];
    int PHB = prm.pads_begin[Y_AXIS];
    int PDB = prm.pads_begin.size() > Z_AXIS ? prm.pads_begin[Z_AXIS] : 0;
    int PWE = prm.pads_end[X_AXIS];
    int PHE = prm.pads_end[Y_AXIS];
    int PDE = prm.pads_end.size() > Z_AXIS ? prm.pads_end[Z_AXIS] : 0;

    int OW = (IW + PWB + PWE - KW) / SW + 1;
    int OH = (IH + PHB + PHE - KH) / SH + 1;
    int OD = dims_size == 5 ? (ID + PDB + PDE - KD) / SD + 1 : 1;
    int OC = prm.dims[1];

    const data_t *src_data = src.readOnly();
    data_t *dst_data = dst.data();

    InferenceEngine::SizeVector dims = dst.getTensorDesc().getDims(); 
    IE_ASSERT(OC == dims[1]);

    int k1 = OH * OW,
           k2 = k1 * OD,
           k3 = IH * IW,
           k4 = k3 * ID;

    if (prm._type == PoolingLayer::MAX) {
        for (int c = 0; c < OC; c++) {
            int cc = c * k2;
            for (int od = 0; od < OD; od++) {
                int cd = cc + od * k1;
                for (int oh = 0; oh < OH; oh++) {
                    int ch = cd + oh * OW;
                    for (int ow = 0; ow < OW; ow++) {

                        int oidx = ch + ow;
                        data_t out_ref = data_t(0);
                        bool is_initialized = false;

                        for (int kd = 0; kd < KD; kd++) {
                            int id = dims_size == 5 ? od * SD - PDB + kd : 0lu;
                            if (id < 0 || id >= ID) continue;
                            for (int kh = 0; kh < KH; kh++) {
                                int ih = oh * SH - PHB + kh;
                                if (ih < 0 || ih >= IH) continue;
                                for (int kw = 0; kw < KW; kw++) {
                                    int iw = ow * SW - PWB + kw;
                                    if (iw < 0 || iw >= IW) continue;
                                    int iidx = c * k4
                                                + id * k3
                                                + ih * IW
                                                + iw;

                                    data_t d = src_data[iidx];
                                    if (!is_initialized) {
                                        out_ref = d;
                                        is_initialized = true;
                                    } else {
                                        if (out_ref < d)
                                            out_ref = d;
                                    }
                                }
                            }
                        }
                        dst_data[oidx] = out_ref;
                    }
                }
            }
        }
    } else if (prm._type == PoolingLayer::AVG) {

        bool include_padding = false;
        bool not_zero_l = false;
        for (auto lr : prm.pads_begin) {
            if (lr) {
                not_zero_l = true;
                break;
            }
        }
        if (!prm._exclude_pad && not_zero_l)
            include_padding = true;

        int PDBKD = KD - PDB,
            PHBKH = KH - PHB,
            PWBKW = KW - PWB,
            IDPDE = ID + PDE,
            IHPHE = IH + PHE,
            IWPWE = IW + PWE;

        for (int c = 0; c < OC; c++) {
            int cc = c * k2;
            for (int od = 0; od < OD; od++) {
                int cd = cc + od * k1;
                int id_start = od * SD - PDB;
                int id_end = std::min(od * SD + PDBKD, IDPDE);
                for (int oh = 0; oh < OH; oh++) {
                    int ch = cd + oh * OW;
                    int ih_start = oh * SH - PHB;
                    int ih_end = std::min(oh * SH + PHBKH, IHPHE);
                    for (int ow = 0; ow < OW; ow++) {
                        size_t oidx = ch + ow;
                        dst_data[oidx] = (data_t)0;
                        int iw_start = ow * SW - PWB;
                        int iw_end = std::min(ow * SW + PWBKW, IWPWE);

                        // include_padding
                        double num_summands = (ih_end - ih_start) * (iw_end - iw_start) * (id_end - id_start);

                        id_start = std::max(id_start, 0);
                        ih_start = std::max(ih_start, 0);
                        iw_start = std::max(iw_start, 0);
                        id_end = std::min(id_end, ID);
                        ih_end = std::min(ih_end, IH);
                        iw_end = std::min(iw_end, IW);

                        if (!include_padding)
                            num_summands = (id_end - id_start) * (ih_end - ih_start) * (iw_end - iw_start);
                        if (num_summands == 0.0) continue;

                        double dst = 0.0;
                        for (int id = id_start; id < id_end; ++id) {
                            for (int ih = ih_start; ih < ih_end; ++ih) {
                                for (int iw = iw_start; iw < iw_end; ++iw) {
                                    size_t iidx = c * k4
                                                + id * k3
                                                + ih * IW
                                                + iw;

                                    dst += (double)src_data[iidx];
                        }   }   }

                        dst_data[oidx] = (data_t)(dst / num_summands);
    }   }   }   }   }
}

class MKLDNNGraphPoolingTests: public TestsCommon,
                                     public WithParamInterface<pooling_test_params> {
    std::string layers_t = R"V0G0N(
        <layer name="pool" id="1" type="Pooling" precision="FP32">

            <pooling kernel="_K_"
                     strides="_KS_"
                     pads_begin="_PB_" pads_end="_PE_"
                     pool-method="_PM_" exclude-pad="_EP_" rounding_type="floor"
                     PrimitivesPriority="_IMPLS_"/>

            <input>
                <port id="1">
                    __SRC_DIMS__
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    __DST_DIMS__
                </port>
            </output>
        </layer>
)V0G0N";

    std::string edges_t = R"V0G0N(
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
)V0G0N";

protected:
    std::string getModel(pooling_test_params p) {
        std::string model = layers_t;

        std::string s_dims;
        for (auto& dim : p.dims) {
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(dim) + "</dim>";
        }
	REPLACE_WITH_STR(model, "__SRC_DIMS__", s_dims);

        s_dims = "";
        int k_len = p.kernel.size();
        for (size_t i = 2lu; i < p.dims.size(); i++) {
            size_t inx = k_len - i + 1lu;
            size_t dim = (p.dims[i] + p.pads_begin[inx] + p.pads_end[inx] - p.kernel[inx]) / p.strides[inx] + 1lu;
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(dim) + "</dim>";
        }
	REPLACE_WITH_STR(model, "__DST_DIMS__", s_dims);
        
        std::string pool_method;
        switch (p._type) {
            case PoolingLayer::AVG: pool_method = "avg";
                break;
            case PoolingLayer::ROI: pool_method = "roi";
                break;
            default: pool_method = "max";
        }
        REPLACE_WITH_STR(model, "_PM_", pool_method);
        
        std::string exclude_pad = "false";
        if (p._exclude_pad) exclude_pad = "true";
        REPLACE_WITH_STR(model, "_EP_", exclude_pad);

        REPLACE_WITH_NUM(model, "_IN_", p.dims[0]);
        REPLACE_WITH_NUM(model, "_IC_", p.dims[1]);

        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_K_", p.kernel);
        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_KS_", p.strides);
        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_PB_", p.pads_begin);
        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_PE_", p.pads_end);

        std::string impls;
        for (const auto& preferType : p.preferTypes) {
            if (!impls.empty())
                impls += ",";
            impls += "cpu:" + MKLDNNGraphTestClass::getStrPrimitiveDescriptorType(preferType);
        }
        REPLACE_WITH_STR(model, "_IMPLS_", impls);

        model = IRTemplateGenerator::getIRTemplate("Pooling_Only", p.dims, "FP32", model, edges_t);

        return model;
    }

    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            pooling_test_params p = ::testing::WithParamInterface<pooling_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;
            ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(network);
            auto& nodes = graph.getNodes();
            for (int i = 0; i < nodes.size(); i++) {
                if (nodes[i]->getType() == MKLDNNPlugin::Pooling) {
                    ASSERT_LE(p.num_prim_desc, nodes[i]->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(nodes[i]->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, nodes[i]->getSelectedPrimitiveDescriptor());
                    ASSERT_TRUE(nodes[i]->getSelectedPrimitiveDescriptor()->getImplementationType() | p.selectedType);
                }
            }

            InferenceEngine::Layout layout = ANY;
            switch (p.dims.size()) {
                case 4:
                    layout = InferenceEngine::NCHW;
                    break;
                case 5:
                    layout = InferenceEngine::NCDHW;
                    break;
            }

            InferenceEngine::Blob::Ptr src =
                InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, p.dims, layout});
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

            ref_pool(*srcPtr, dst_ref, p);

            compare(*output, dst_ref, 0.0001f);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphPoolingTests, TestsPooling) {}

INSTANTIATE_TEST_CASE_P(
        TestsPooling, MKLDNNGraphPoolingTests,
        ::testing::Values(
        /*0*/   pooling_test_params{{1, 3, 228, 228}, {2, 2}, {2, 2}, {0, 0}, {0, 0}, PoolingLayer::MAX, false, 3, MKLDNNPlugin::impl_desc_type::jit},
                pooling_test_params{{1, 3, 228, 228}, {4, 2}, {2, 2}, {0, 0}, {0, 0}, PoolingLayer::MAX, false, 3, MKLDNNPlugin::impl_desc_type::jit},
                pooling_test_params{{1, 3, 228, 228}, {4, 2}, {2, 1}, {0, 0}, {0, 0}, PoolingLayer::MAX, false, 3, MKLDNNPlugin::impl_desc_type::jit},
                pooling_test_params{{1, 3, 228, 228}, {2, 2}, {2, 2}, {0, 0}, {0, 0}, PoolingLayer::MAX, false, 3, MKLDNNPlugin::impl_desc_type::ref,
                            {MKLDNNPlugin::impl_desc_type::ref_any}},
                pooling_test_params{{1, 3, 228, 228}, {4, 2}, {2, 2}, {0, 0}, {0, 0}, PoolingLayer::MAX, false, 3, MKLDNNPlugin::impl_desc_type::ref,
                            {MKLDNNPlugin::impl_desc_type::ref_any}},
                pooling_test_params{{1, 3, 228, 228}, {4, 2}, {2, 1}, {0, 0}, {0, 0}, PoolingLayer::MAX, false, 3, MKLDNNPlugin::impl_desc_type::ref,
                            {MKLDNNPlugin::impl_desc_type::ref_any}},
                pooling_test_params{{1u, 4u, 128u, 128u}, {2u, 2u}, {2u, 2u}, {1u, 0u}, {0u, 0u}, PoolingLayer::AVG, false, 3u,
                            MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                pooling_test_params{{1u, 4u, 128u, 128u}, {2u, 2u}, {2u, 2u}, {1u, 0u}, {0u, 0u}, PoolingLayer::AVG, false, 3u,
                            MKLDNNPlugin::impl_desc_type::jit },
                pooling_test_params{{1u, 4u, 128u, 128u}, {2u, 2u}, {2u, 2u}, {0u, 0u}, {0u, 0u}, PoolingLayer::AVG, true, 3u,
                            MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
        /*9*/   pooling_test_params{{1u, 4u, 128u, 128u}, {2u, 2u}, {2u, 2u}, {0u, 0u}, {0u, 0u}, PoolingLayer::AVG, true, 3u,
                            MKLDNNPlugin::impl_desc_type::jit },
                pooling_test_params{{1u, 4u, 128u, 128u}, {2u, 2u}, {2u, 2u}, {2u, 2u}, {2u, 2u}, PoolingLayer::AVG, true, 3u,
                            MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                pooling_test_params{{1u, 4u, 128u, 128u}, {2u, 2u}, {2u, 2u}, {2u, 2u}, {2u, 2u}, PoolingLayer::AVG, false, 3u,
                            MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
//                pooling_test_params{{1u, 4u, 128u, 128u}, {2u, 2u}, {2u, 2u}, {2u, 2u}, {2u, 2u}, PoolingLayer::MAX, false, 3u,
//                            MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
//                pooling_test_params{{1u, 1u, 128u, 128u}, {2u, 2u}, {2u, 2u}, {2u, 2u}, {2u, 2u}, PoolingLayer::MAX, false, 1,
//                                    MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                // TODO Fix jit implementation. End paddings
//                pooling_test_params{{1u, 4u, 128u, 128u}, {2u, 2u}, {2u, 2u}, {2u, 2u}, {2u, 0u}, PoolingLayer::AVG, true, 3u,
//                            MKLDNNPlugin::impl_desc_type::jit },
//                pooling_test_params{{1u, 4u, 128u, 128u}, {2u, 2u}, {2u, 2u}, {2u, 2u}, {2u, 0u}, PoolingLayer::AVG, false, 3u,
//                            MKLDNNPlugin::impl_desc_type::jit },
//                pooling_test_params{{1u, 4u, 128u, 128u}, {2u, 2u}, {2u, 2u}, {2u, 2u}, {2u, 0u}, PoolingLayer::MAX, false, 3u,
//                            MKLDNNPlugin::impl_desc_type::jit },

                // 5D tensor
                pooling_test_params{{1u, 3u, 16u, 32u, 32u}, {2u, 2u, 2u}, {1u, 1u, 1u}, {0u, 0u, 0u}, {0u, 0u, 0u}, PoolingLayer::MAX, false, 3u,
                            MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                pooling_test_params{{1u, 3u, 16u, 32u, 32u}, {2u, 2u, 2u}, {1u, 1u, 1u}, {0u, 0u, 0u}, {0u, 0u, 0u}, PoolingLayer::MAX, false, 3u,
                            MKLDNNPlugin::impl_desc_type::jit },
                pooling_test_params{{1u, 3u, 16u, 32u, 32u}, {2u, 2u, 2u}, {1u, 1u, 1u}, {1u, 1u, 1u}, {1u, 1u, 1u}, PoolingLayer::MAX, false, 3u,
                            MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                pooling_test_params{{1u, 32u, 60u, 60u, 60u}, {2u, 3u, 4u}, {2u, 2u, 2u}, {1u, 1u, 1u}, {1u, 2u, 3u}, PoolingLayer::MAX, false, 3u,
                            MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
//                pooling_test_params{{1u, 3u, 16u, 32u, 32u}, {2u, 2u, 2u}, {1u, 1u, 1u}, {1u, 2u, 3u}, {1u, 2u, 3u}, PoolingLayer::MAX, false, 1u,
//                            MKLDNNPlugin::impl_desc_type::jit },
                pooling_test_params{{1u, 4u, 128u, 128u, 128u}, {2u, 2u, 2u}, {2u, 2u, 2u}, {1u, 0u, 0u}, {0u, 0u, 0u}, PoolingLayer::AVG, false, 3u,
                            MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                pooling_test_params{{1u, 4u, 128u, 128u, 128u}, {2u, 2u, 2u}, {2u, 2u, 2u}, {1u, 0u, 0u}, {0u, 0u, 0u}, PoolingLayer::AVG, false, 3u,
                            MKLDNNPlugin::impl_desc_type::jit },
                pooling_test_params{{1u, 4u, 128u, 128u, 128u}, {2u, 2u, 2u}, {2u, 2u, 2u}, {0u, 0u, 0u}, {0u, 0u, 0u}, PoolingLayer::AVG, true, 3u,
                            MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                pooling_test_params{{1u, 4u, 128u, 128u, 128u}, {2u, 2u, 2u}, {2u, 2u, 2u}, {0u, 0u, 0u}, {0u, 0u, 0u}, PoolingLayer::AVG, true, 3u,
                            MKLDNNPlugin::impl_desc_type::jit },
                pooling_test_params{{1u, 4u, 128u, 128u, 128u}, {2u, 2u, 2u}, {2u, 2u, 2u}, {2u, 2u, 2u}, {0u, 0u, 0u}, PoolingLayer::AVG, true, 3u,
                            MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                pooling_test_params{{1u, 4u, 128u, 128u, 128u}, {2u, 2u, 2u}, {2u, 2u, 2u}, {2u, 2u, 2u}, {2u, 2u, 2u}, PoolingLayer::AVG, true, 3u,
                            MKLDNNPlugin::impl_desc_type::jit },
                pooling_test_params{{1u, 4u, 128u, 128u, 128u}, {2u, 2u, 2u}, {2u, 2u, 2u}, {2u, 2u, 2u}, {2u, 2u, 2u}, PoolingLayer::AVG, false, 3u,
                            MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                pooling_test_params{{1u, 4u, 128u, 128u, 128u}, {2u, 2u, 2u}, {2u, 2u, 2u}, {2u, 2u, 2u}, {2u, 2u, 2u}, PoolingLayer::AVG, false, 3u,
                            MKLDNNPlugin::impl_desc_type::jit },
                pooling_test_params{{1u, 1u, 128u, 128u, 128u}, {2u, 2u, 2u}, {2u, 2u, 2u}, {2u, 2u, 2u}, {2u, 2u, 2u}, PoolingLayer::AVG, false, 1u,
                                    MKLDNNPlugin::impl_desc_type::ref }));


class MKLDNNGraphDynBatchPoolingTests: public MKLDNNGraphPoolingTests {
protected:
    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            pooling_test_params p = ::testing::WithParamInterface<pooling_test_params>::GetParam();
            std::string model = getModel(p);
            size_t MB = p.dims[0];
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


            InferenceEngine::Layout layout = ANY;
            switch (p.dims.size()) {
                case 4:
                    layout = InferenceEngine::NCHW;
                    break;
                case 5:
                    layout = InferenceEngine::NCDHW;
                    break;
            }
            InferenceEngine::Blob::Ptr src =
                InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, p.dims, layout});
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

            auto checkPooling = [](const MKLDNNPlugin::MKLDNNNodePtr& node) {
                return node->getType() == MKLDNNPlugin::Pooling;
            };
            graph.checkDynBatch(srcs, outputBlobs, MB, MB, checkPooling);
            graph.checkDynBatch(srcs, outputBlobs, 1, MB, checkPooling);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphDynBatchPoolingTests, TestsDynBatchPooling) {}

// TODO: rewrite to ngraph to have reshape functionality
INSTANTIATE_TEST_CASE_P(
        DISABLED_TestsDynBatchPooling, MKLDNNGraphDynBatchPoolingTests,
        ::testing::Values(
                pooling_test_params{{1, 3, 228, 228}, {4, 2}, {2, 1}, {0, 0}, {0, 0}, PoolingLayer::MAX, false, 4, MKLDNNPlugin::impl_desc_type::jit},
                pooling_test_params{{1, 3, 228, 228}, {2, 2}, {2, 2}, {0, 0}, {0, 0}, PoolingLayer::MAX, false, 6, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                pooling_test_params{{1, 3, 228, 228}, {4, 2}, {2, 2}, {0, 0}, {0, 0}, PoolingLayer::MAX, false, 4, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                pooling_test_params{{1, 3, 228, 228}, {4, 2}, {2, 1}, {0, 0}, {0, 0}, PoolingLayer::MAX, false, 4, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}}));
