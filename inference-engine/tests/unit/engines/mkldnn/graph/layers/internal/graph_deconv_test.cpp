// Copyright (C) 2018 Intel Corporation
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


using namespace InferenceEngine;
using namespace ::testing;
using namespace std;
using namespace mkldnn;


struct deconv_test_params {
    // Formats: NCHW, NCDHW
    vector<size_t> dims;
    // Formats: WH, WHD
    vector<size_t> kernel;
    vector<size_t> strides;
    vector<size_t> pads_begin;
    vector<size_t> pads_end;

    size_t out_c;
    size_t grp_c;

    bool with_bias;
    string auto_pad;

    size_t num_prim_desc;

    std::vector<int> selectedTypes;
    std::vector<MKLDNNPlugin::impl_desc_type> preferTypes;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

template <typename data_t>
void ref_deconv(const InferenceEngine::TBlob<data_t> &src, const InferenceEngine::Blob::Ptr &weights, const InferenceEngine::Blob::Ptr &bias,
                InferenceEngine::TBlob<data_t> &dst, deconv_test_params prm) {
    auto dims_size = src.dims().size();

    size_t G  = prm.grp_c;
    size_t KW = prm.kernel[X_AXIS];
    size_t KH = prm.kernel[Y_AXIS];
    size_t KD = prm.kernel.size() > Z_AXIS ? prm.kernel[Z_AXIS] : 1u;

    size_t PW = prm.pads_begin[X_AXIS];
    size_t PH = prm.pads_begin[Y_AXIS];
    size_t PD = prm.pads_begin.size() > Z_AXIS ? prm.pads_begin[Z_AXIS] : 0u;

    size_t SW = prm.strides[X_AXIS];
    size_t SH = prm.strides[Y_AXIS];
    size_t SD = prm.strides.size() > Z_AXIS ? prm.strides[Z_AXIS] : 1u;

    size_t IW = src.dims()[dims_size - 1];
    size_t IH = src.dims()[dims_size - 2];
    size_t ID = dims_size == 5 ? src.dims()[dims_size - 3] : 1u;
    size_t IC = src.dims()[1];
    size_t MB = src.dims()[0];

    size_t OC = prm.out_c;

    size_t OW = SW * (IW - 1) + KW - 2 * PW;
    size_t OH = SH * (IH - 1) + KH - 2 * PH;
    size_t OD = dims_size == 5 ? (SD * (ID - 1) + KD - 2 * PD) : 1u;

    const data_t *src_data = src.readOnly();
    const data_t *weights_data = weights->buffer().as<data_t*>();
    const data_t *bias_data = bias->buffer().as<data_t*>();

    data_t *dst_data = dst.data();

    size_t CS1 = OH * OW;
    size_t CS2 = CS1 * OD;
    size_t CS3 = CS2 * OC;

    size_t CI1 = IH * IW;
    size_t CI2 = CI1 * ID;
    size_t CI3 = CI2 * IC;

    size_t CK1 = KH * KW;
    size_t CK2 = CK1 * KD;
    size_t CK3 = CK2 * (OC / G);
    size_t CK4 = CK3 * (IC / G);

    for (int g = 0; g < G; ++g) {
        for (int mb = 0; mb < MB; ++mb) {
            for (int oc = 0; oc < OC / G; ++oc) {
                for (int od = 0; od < OD; ++od) {
                    for (int oh = 0; oh < OH; ++oh) {
                        for (int ow = 0; ow < OW; ++ow) {
                            size_t didx = mb * CS3
                                          + (g * OC / G + oc) * CS2
                                          + od * CS1
                                          + oh * OW
                                          + ow;

                            dst_data[didx] = data_t(0);
                            if (prm.with_bias) dst_data[didx] += bias_data[g * OC / G + oc];

                            for (int ic = 0; ic < IC / G; ic++) {
                                for (int kd = 0; kd < KD; kd++) {
                                    for (int kh = 0; kh < KH; kh++) {
                                        for (int kw = 0; kw < KW; kw++) {
                                            if (ow + PW < kw || oh + PH < kh || od + PD < kd)
                                                continue;

                                            size_t iw = ow - kw + PW;
                                            size_t ih = oh - kh + PH;
                                            size_t id = od - kd + PD;

                                            if (iw % SW != 0 || ih % SH != 0 || id % SD != 0)
                                                continue;

                                            iw /= SW;
                                            ih /= SH;
                                            id /= SD;

                                            if (ih < IH && iw < IW && id < ID) {
                                                size_t sidx = mb * CI3
                                                              + (g * IC / G + ic) * CI2
                                                              + id * CI1
                                                              + ih * IW
                                                              + iw;

                                                size_t widx = g * CK4
                                                              + ic * CK3
                                                              + oc * CK2
                                                              + kd * CK1
                                                              + kh * KW
                                                              + kw;

                                                dst_data[didx] += src_data[sidx] * weights_data[widx];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

class MKLDNNGraphDeconvolutionalTests: public TestsCommon,
                                     public WithParamInterface<deconv_test_params> {
    std::string model_t_5D = R"V0G0N(
<net name="Deconvolution_Only" version="3" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">__SRC_DIMS__
                </port>
            </output>
        </layer>
        <layer name="deconv1" id="1" type="Deconvolution" precision="FP32">
            <deconvolution _AP_ kernel="_K_"
                         pads_begin="_PB_"  pads_end="_PE_"
                         strides="_KS_"
                         output="_OC_" group="_GC_" PrimitivesPriority="_IMPLS_"/>

            <weights offset="0" size="_S1_" />
            <biases offset="_S1_" size="_S2_" />

            <input>
                <port id="1">__SRC_DIMS__
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>_IN_</dim>
                    <dim>_OC_</dim>__DST_DIMS__
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</net>
)V0G0N";

protected:
    std::string getModel(deconv_test_params p) {
        std::string model = model_t_5D;
        auto dims_size = p.dims.size();
        std::string s_dims;
        for (auto& dim : p.dims) {
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(dim) + "</dim>";
        }
	REPLACE_WITH_STR(model, "__SRC_DIMS__", s_dims);

        s_dims = "";
        int k_len = p.kernel.size();
        for (size_t i = 2; i < p.dims.size(); i++) {
            size_t inx = k_len - i + 1;
            size_t dim = p.strides[inx] * (p.dims[i] - 1) + p.kernel[inx] - 2 * p.pads_begin[inx];
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(dim) + "</dim>";
        }
	REPLACE_WITH_STR(model, "__DST_DIMS__", s_dims);
        REPLACE_WITH_NUM(model, "_IN_", p.dims[0]);

        if (!p.with_bias) REMOVE_LINE(model, "<biases offset=\"_S1_\" size=\"_S2_\" />");

        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_K_", p.kernel);
        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_KS_", p.strides);
        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_PB_", p.pads_begin);
        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_PE_", p.pads_end);
        REPLACE_WITH_NUM(model, "_GC_", p.grp_c);
        REPLACE_WITH_NUM(model, "_OC_", p.out_c);
        string auto_pad;
        if (!p.auto_pad.empty()) auto_pad = string("auto_pad=") + string("\"") + p.auto_pad + string("\"");
        REPLACE_WITH_STR(model, "_AP_", auto_pad);

        size_t blob_size = p.out_c * (p.dims[1] / p.grp_c);
        for (auto k : p.kernel) {
            blob_size *= k;
        }
        size_t w_data_size = blob_size * sizeof(float);
        REPLACE_WITH_NUM(model, "_S1_", w_data_size);

        size_t b_data_size = p.out_c * sizeof(float);
        REPLACE_WITH_NUM(model, "_S2_", b_data_size);

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
            TestsCommon::SetUp();
            deconv_test_params p = ::testing::WithParamInterface<deconv_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            size_t blob_size = p.out_c * (p.dims[1] / p.grp_c);
            for (auto k : p.kernel) {
                blob_size *= k;
            }
            InferenceEngine::SizeVector dims_weights = { blob_size };

            std::vector<InferenceEngine::Blob::Ptr> blob_to_model;
            InferenceEngine::Blob::Ptr weights = InferenceEngine::make_shared_blob<float>(InferenceEngine::Precision::FP32, InferenceEngine::C, dims_weights);
            weights->allocate();
            fill_data(weights->buffer().as<float*>(), weights->size());
            blob_to_model.push_back(weights);

            InferenceEngine::Blob::Ptr bias = InferenceEngine::make_shared_blob<float>(InferenceEngine::Precision::FP32, InferenceEngine::C, {p.out_c});
            bias->allocate();
            fill_data(bias->buffer().as<float*>(), bias->size());
            blob_to_model.push_back(bias);

            size_t total_size_in_bytes = 0;
            for (InferenceEngine::Blob::Ptr blb : blob_to_model) total_size_in_bytes += blb->byteSize();

            InferenceEngine::TBlob<uint8_t>::Ptr model_blob =
                    InferenceEngine::make_shared_blob<uint8_t>(InferenceEngine::Precision::U8, InferenceEngine::C, {total_size_in_bytes});
            model_blob->allocate();
            uint8_t* model_blob_ptr = model_blob->buffer().as<uint8_t*>();
            for (InferenceEngine::Blob::Ptr blb : blob_to_model) {
                memcpy(model_blob_ptr, blb->buffer().as<uint8_t*>(), blb->byteSize());
                model_blob_ptr += blb->byteSize();
            }
            net_reader.SetWeights(model_blob);

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());
            auto& nodes = graph.getNodes();
            for (auto &node : nodes) {
                if (node->getType() == MKLDNNPlugin::Deconvolution) {
                    ASSERT_LE(p.num_prim_desc, node->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(node->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, node->getSelectedPrimitiveDescriptor());
                    bool good_prim = false;
                    for (auto & selected : p.selectedTypes)
                        if (selected == (node->getSelectedPrimitiveDescriptor()->getImplementationType() & selected))
                            good_prim = true;
                    ASSERT_TRUE(good_prim);
                }
            }

            InferenceEngine::SizeVector dims_src = p.dims;

            InferenceEngine::Layout layout = ANY;
            switch (p.dims.size()) {
                case 4:
                    layout = InferenceEngine::NCHW;
                    break;
                case 5:
                    layout = InferenceEngine::NCDHW;
                    break;
            }
            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, layout, dims_src);
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

            ref_deconv(*srcPtr, weights, bias, dst_ref, p);

            compare(*output, dst_ref, 0.0002f);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphDeconvolutionalTests, TestsDeconvolution) {}


INSTANTIATE_TEST_CASE_P(
        TestDeconvolution, MKLDNNGraphDeconvolutionalTests,
        ::testing::Values(
        /*0*/   deconv_test_params{{1, 3, 3, 3}, {3, 3}, {1, 1}, {0, 0}, {0, 0}, 2, 1, false, "", 2, {MKLDNNPlugin::impl_desc_type::jit} },
                deconv_test_params{{3, 3, 3, 3}, {4, 3}, {1, 1}, {0, 0}, {0, 0}, 2, 1, false, "", 2, {MKLDNNPlugin::impl_desc_type::jit} },
                deconv_test_params{{1, 3, 3, 3}, {4, 3}, {1, 2}, {0, 0}, {0, 0}, 2, 1, false, "", 2, {MKLDNNPlugin::impl_desc_type::gemm, MKLDNNPlugin::impl_desc_type::jit} },
                deconv_test_params{{1, 3, 3, 3}, {4, 3}, {2, 2}, {0, 0}, {0, 0}, 2, 1, false, "", 2, {MKLDNNPlugin::impl_desc_type::gemm, MKLDNNPlugin::impl_desc_type::jit} },
                deconv_test_params{{4, 17, 3, 3}, {4, 3}, {2, 2}, {0, 0}, {0, 0}, 2, 1, false, "", 2, {MKLDNNPlugin::impl_desc_type::gemm, MKLDNNPlugin::impl_desc_type::jit} },
                deconv_test_params{{2, 8, 5, 5}, {4, 4}, {2, 2}, {1, 1}, {0, 0}, 8, 8, false, "", 4, {MKLDNNPlugin::impl_desc_type::jit | MKLDNNPlugin::impl_desc_type::_dw}},
                deconv_test_params{{2, 8, 5, 5}, {8, 8}, {4, 4}, {1, 1}, {0, 0}, 8, 8, false, "", 4, {MKLDNNPlugin::impl_desc_type::jit | MKLDNNPlugin::impl_desc_type::_dw}},
                deconv_test_params{{2, 8, 5, 5}, {4, 8}, {2, 4}, {1, 1}, {0, 0}, 8, 8, false, "", 4, {MKLDNNPlugin::impl_desc_type::jit | MKLDNNPlugin::impl_desc_type::_dw}},
        /*8*/   deconv_test_params{{1, 3, 3, 3}, {3, 3}, {1, 1}, {0, 0}, {0, 0}, 2, 1, true, "", 2, {MKLDNNPlugin::impl_desc_type::jit} },
                deconv_test_params{{3, 3, 3, 3}, {4, 3}, {1, 1}, {0, 0}, {0, 0}, 2, 1, true, "", 2, {MKLDNNPlugin::impl_desc_type::jit} },
                deconv_test_params{{1, 3, 3, 3}, {4, 3}, {1, 2}, {0, 0}, {0, 0}, 2, 1, true, "", 2, {MKLDNNPlugin::impl_desc_type::gemm, MKLDNNPlugin::impl_desc_type::jit} },
                deconv_test_params{{1, 3, 3, 3}, {4, 3}, {2, 2}, {0, 0}, {0, 0}, 2, 1, true, "", 2, {MKLDNNPlugin::impl_desc_type::gemm, MKLDNNPlugin::impl_desc_type::jit} },
                deconv_test_params{{4, 17, 3, 3}, {4, 3}, {2, 2}, {0, 0}, {0, 0}, 2, 1, true, "", 2, {MKLDNNPlugin::impl_desc_type::gemm, MKLDNNPlugin::impl_desc_type::jit} },
                deconv_test_params{{2, 8, 5, 5}, {4, 4}, {2, 2}, {1, 1}, {0, 0}, 8, 8, true, "", 4, {MKLDNNPlugin::impl_desc_type::jit | MKLDNNPlugin::impl_desc_type::_dw}},
                deconv_test_params{{2, 8, 5, 5}, {8, 8}, {4, 4}, {1, 1}, {0, 0}, 8, 8, true, "", 4, {MKLDNNPlugin::impl_desc_type::jit | MKLDNNPlugin::impl_desc_type::_dw}},
                deconv_test_params{{2, 8, 5, 5}, {4, 8}, {2, 4}, {1, 1}, {0, 0}, 8, 8, true, "", 4, {MKLDNNPlugin::impl_desc_type::jit | MKLDNNPlugin::impl_desc_type::_dw}},
                deconv_test_params{{1, 3, 3, 3}, {3, 3}, {1, 1}, {0, 0}, {0, 0}, 2, 1, false, "", 2, {MKLDNNPlugin::impl_desc_type::ref_any}, 
                                    {MKLDNNPlugin::impl_desc_type::ref_any}},
        /*17*/  deconv_test_params{{2, 8, 5, 5}, {1, 3}, {1, 1}, {0, 1}, {0, 1}, 8, 8, true, "", 2,
                    {MKLDNNPlugin::impl_desc_type::ref_any}, {MKLDNNPlugin::impl_desc_type::ref_any}},
                deconv_test_params{{1, 6, 6, 5}, {3, 1}, {1, 1}, {1, 0}, {1, 0}, 9, 3, true, "", 2,
                    {MKLDNNPlugin::impl_desc_type::ref_any}, {MKLDNNPlugin::impl_desc_type::ref_any}},
                deconv_test_params{{2, 24, 5, 5}, {4, 4}, {2, 2}, {1, 1}, {0, 0}, 24, 3, true, "", 4, {MKLDNNPlugin::impl_desc_type::jit}},
                deconv_test_params{{2, 24, 5, 5}, {4, 4}, {2, 2}, {1, 1}, {0, 0}, 24, 1, true, "", 3, {MKLDNNPlugin::impl_desc_type::jit}},
                deconv_test_params{{2, 72, 5, 5}, {4, 4}, {2, 2}, {1, 1}, {0, 0}, 72, 3, true, "", 4, {MKLDNNPlugin::impl_desc_type::jit}},
                deconv_test_params{{1, 12, 2, 2}, {4, 4}, {2, 2}, {1, 1}, {1, 1}, 12, 12, true, "", 4, {MKLDNNPlugin::impl_desc_type::jit}},
#ifdef USE_MKL
                deconv_test_params{{2, 8, 5, 5}, {4, 4}, {2, 2}, {1, 1}, {0, 0}, 8, 2, false, "", 3, {MKLDNNPlugin::impl_desc_type::gemm}},
                deconv_test_params{{2, 8, 5, 5}, {4, 4}, {2, 2}, {1, 1}, {0, 0}, 8, 2, true, "", 3, {MKLDNNPlugin::impl_desc_type::gemm}},
                deconv_test_params{{1, 6, 6, 5}, {3, 1}, {1, 1}, {1, 0}, {1, 0}, 9, 3, true, "", 2,
                    {MKLDNNPlugin::impl_desc_type::gemm_blas}},
                deconv_test_params{{1, 64, 12, 12, 2}, {2, 2, 2}, {2, 2, 2}, {0, 0, 0}, {1, 0, 0}, 32, 1, true, "", 4,
                    {MKLDNNPlugin::impl_desc_type::gemm_blas}},
                deconv_test_params{{1, 32, 12, 12, 2}, {2, 2, 2}, {2, 2, 2}, {0, 0, 0}, {1, 0, 0}, 16, 1, true, "", 4, 
                    {MKLDNNPlugin::impl_desc_type::gemm_blas} },
                deconv_test_params{{1, 25, 1, 1, 1}, {4, 4, 4}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, 64, 1, true, "valid", 3,
                    {MKLDNNPlugin::impl_desc_type::gemm_blas} },
                deconv_test_params{{1, 32, 16, 16, 16}, {4, 4, 4}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, 1, 1, true, "same_upper", 3,
                    {MKLDNNPlugin::impl_desc_type::gemm_blas} },
                deconv_test_params{{1, 64, 12, 12, 2}, {2, 2, 2}, {2, 2, 2}, {0, 0, 0}, {1, 0, 0}, 32, 1, true, "same_upper", 3,
                    {MKLDNNPlugin::impl_desc_type::gemm_blas} },
                deconv_test_params{{1, 50, 1, 1, 1}, {4, 4, 4}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, 128, 1, true, "", 3,
                    {MKLDNNPlugin::impl_desc_type::gemm_blas}, {MKLDNNPlugin::impl_desc_type::gemm_blas}},
#endif
                // 5D
                deconv_test_params{{1, 2, 8, 5, 5}, {3, 3, 3}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, 4, 1, true, "", 4,
                    {MKLDNNPlugin::impl_desc_type::ref_any}, {MKLDNNPlugin::impl_desc_type::ref_any} }

                // Blocked, with biases
                // TODO support on jit
//                deconv_test_params{{2, 24, 5, 5, 5}, {4, 4}, {2, 2}, {1, 1}, {0, 0}, 24, 3, true, "", 4, {MKLDNNPlugin::impl_desc_type::jit}},
//                deconv_test_params{{2, 24, 5, 5, 5}, {4, 4}, {2, 2}, {1, 1}, {0, 0}, 24, 1, true, "", 3, {MKLDNNPlugin::impl_desc_type::jit}},
//                deconv_test_params{{2, 72, 5, 5, 5}, {4, 4}, {2, 2}, {1, 1}, {0, 0}, 72, 3, true, "", 4, {MKLDNNPlugin::impl_desc_type::jit}}
        ));

class MKLDNNGraphDynBatchDeconvolutionalTests: public MKLDNNGraphDeconvolutionalTests {
protected:
    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            deconv_test_params p = ::testing::WithParamInterface<deconv_test_params>::GetParam();
            std::string model = getModel(p);
            size_t MB = p.dims[0];
            if (MB < 2)
                MB = 2;

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));
            
            size_t blob_size = 1;
            for (auto k : p.kernel) {
                blob_size *= k;
            }
            InferenceEngine::SizeVector dims_weights = {blob_size * p.out_c * (p.dims[1] / p.grp_c)};

            std::vector<InferenceEngine::Blob::Ptr> blob_to_model;
            InferenceEngine::Blob::Ptr weights = InferenceEngine::make_shared_blob<float>(InferenceEngine::Precision::FP32, InferenceEngine::C, dims_weights);
            weights->allocate();
            fill_data(weights->buffer().as<float*>(), weights->size());
            blob_to_model.push_back(weights);

            InferenceEngine::Blob::Ptr bias = InferenceEngine::make_shared_blob<float>(InferenceEngine::Precision::FP32, InferenceEngine::C, {p.out_c});
            bias->allocate();
            fill_data(bias->buffer().as<float*>(), bias->size());
            blob_to_model.push_back(bias);

            size_t total_size_in_bytes = 0;
            for (InferenceEngine::Blob::Ptr blb : blob_to_model) total_size_in_bytes += blb->byteSize();

            InferenceEngine::TBlob<uint8_t>::Ptr model_blob =
                    InferenceEngine::make_shared_blob<uint8_t>(InferenceEngine::Precision::U8, InferenceEngine::C, {total_size_in_bytes});
            model_blob->allocate();
            uint8_t* model_blob_ptr = model_blob->buffer().as<uint8_t*>();
            for (InferenceEngine::Blob::Ptr blb : blob_to_model) {
                memcpy(model_blob_ptr, blb->buffer().as<uint8_t*>(), blb->byteSize());
                model_blob_ptr += blb->byteSize();
            }
            net_reader.SetWeights(model_blob);

            InferenceEngine::CNNNetwork network = net_reader.getNetwork();
            auto implNet = dynamic_cast<InferenceEngine::details::CNNNetworkImpl *>(&((InferenceEngine::ICNNNetwork&)network));
            ASSERT_NE(nullptr, implNet) << "Failed to cast ICNNNetwork to CNNNetworkImpl";
            InferenceEngine::ResponseDesc resp;
            InferenceEngine::StatusCode sts  = implNet->setBatchSizeReshape(MB, &resp);
            ASSERT_EQ((int)InferenceEngine::StatusCode::OK, sts) << resp.msg;


            MKLDNNGraphTestClass graph;
            graph.setProperty({{InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED, InferenceEngine::PluginConfigParams::YES}});
            graph.CreateGraph(net_reader.getNetwork());

            InferenceEngine::SizeVector dims_src = p.dims;

            InferenceEngine::Layout layout = ANY;
            switch (p.dims.size()) {
                case 4:
                    layout = InferenceEngine::NCHW;
                    break;
                case 5:
                    layout = InferenceEngine::NCDHW;
                    break;
            }
            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, layout, dims_src);
            InferenceEngine::TBlob<float>* srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());
            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            src->allocate();
            fill_data(src->buffer(), src->size());

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

            auto checkDeconvolution = [](const MKLDNNPlugin::MKLDNNNodePtr& node) {
                return node->getType() == MKLDNNPlugin::Deconvolution;
            };

            graph.checkDynBatch(srcs, outputBlobs, MB, MB, checkDeconvolution, MKLDNNGraphTestClass::CheckDynBatchType::Child);
            graph.checkDynBatch(srcs, outputBlobs, 1, MB, checkDeconvolution, MKLDNNGraphTestClass::CheckDynBatchType::Child);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphDynBatchDeconvolutionalTests, TestsDynBatchDeconvolutional) {}

INSTANTIATE_TEST_CASE_P(
        TestsDynBatchDeconvolutional, MKLDNNGraphDynBatchDeconvolutionalTests,
        ::testing::Values(
                deconv_test_params{{1, 3, 3, 3}, {3, 3}, {1, 1}, {0, 0}, {0, 0}, 2, 1, false, "", 5, {MKLDNNPlugin::impl_desc_type::jit} },
                deconv_test_params{{3, 3, 3, 3}, {4, 3}, {1, 1}, {0, 0}, {0, 0}, 2, 1, false, "", 5, {MKLDNNPlugin::impl_desc_type::jit} },
                deconv_test_params{{1, 3, 3, 3}, {4, 3}, {1, 2}, {0, 0}, {0, 0}, 2, 1, false, "", 4, {MKLDNNPlugin::impl_desc_type::gemm, MKLDNNPlugin::impl_desc_type::jit} },
                deconv_test_params{{1, 3, 3, 3}, {4, 3}, {2, 2}, {0, 0}, {0, 0}, 2, 1, false, "", 3, {MKLDNNPlugin::impl_desc_type::gemm, MKLDNNPlugin::impl_desc_type::jit} },
                deconv_test_params{{4, 17, 3, 3}, {4, 3}, {2, 2}, {0, 0}, {0, 0}, 2, 1, false, "", 3, {MKLDNNPlugin::impl_desc_type::gemm, MKLDNNPlugin::impl_desc_type::jit} },
                deconv_test_params{{2, 8, 5, 5}, {4, 4}, {2, 2}, {1, 1}, {0, 0}, 8, 2, false, "", 3, {MKLDNNPlugin::impl_desc_type::gemm}},
                deconv_test_params{{2, 8, 5, 5}, {4, 4}, {2, 2}, {1, 1}, {0, 0}, 8, 8, false, "", 4, {MKLDNNPlugin::impl_desc_type::jit | MKLDNNPlugin::impl_desc_type::_dw}},
                deconv_test_params{{2, 8, 5, 5}, {8, 8}, {4, 4}, {1, 1}, {0, 0}, 8, 8, false, "", 4, {MKLDNNPlugin::impl_desc_type::jit | MKLDNNPlugin::impl_desc_type::_dw}},
                deconv_test_params{{2, 8, 5, 5}, {4, 8}, {2, 4}, {1, 1}, {0, 0}, 8, 8, false, "", 4, {MKLDNNPlugin::impl_desc_type::jit | MKLDNNPlugin::impl_desc_type::_dw}}
        ));
