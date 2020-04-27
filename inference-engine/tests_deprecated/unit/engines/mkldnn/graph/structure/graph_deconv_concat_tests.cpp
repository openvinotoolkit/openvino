// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "mkldnn_graph.h"

#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include <mkldnn_extension_utils.h>
#include "tests_common.hpp"
#include "ir_gen_helper.hpp"
#include <ie_core.hpp>
#include "common_test_utils/common_layers_params.hpp"

using namespace ::testing;
using namespace std;
using namespace mkldnn;
using namespace single_layer_tests;

struct concat_params {
    size_t axis;
};

struct deconv_concat_params {
    // Formats: NCHW, NCDHW
    std::vector<size_t> in;

    CommonTestUtils::conv_common_params deconv;
    concat_params concat;

    std::vector<MKLDNNPlugin::impl_desc_type> preferTypes;
};

void ref_deconv_common(const InferenceEngine::Blob &src,
                       InferenceEngine::Blob &dst,
                       const float *weights_data,
                       size_t weights_size,
                       const float *bias_data,
                       size_t bias_size,
                       const CommonTestUtils::conv_common_params &prm) {
    auto dims_size = src.getTensorDesc().getDims().size();

    size_t G  = prm.group;
    size_t KW = prm.kernel[InferenceEngine::X_AXIS];
    size_t KH = prm.kernel[InferenceEngine::Y_AXIS];
    size_t KD = prm.kernel.size() > InferenceEngine::Z_AXIS ? prm.kernel[InferenceEngine::Z_AXIS] : 1u;

    size_t PW = prm.pads_begin[InferenceEngine::X_AXIS];
    size_t PH = prm.pads_begin[InferenceEngine::Y_AXIS];
    size_t PD = prm.pads_begin.size() > InferenceEngine::Z_AXIS ? prm.pads_begin[InferenceEngine::Z_AXIS] : 0u;

    size_t SW = prm.stride[InferenceEngine::X_AXIS];
    size_t SH = prm.stride[InferenceEngine::Y_AXIS];
    size_t SD = prm.stride.size() > InferenceEngine::Z_AXIS ? prm.stride[InferenceEngine::Z_AXIS] : 1u;

    size_t IW = src.getTensorDesc().getDims()[dims_size - 1];
    size_t IH = src.getTensorDesc().getDims()[dims_size - 2];
    size_t ID = dims_size == 5 ? src.getTensorDesc().getDims()[dims_size - 3] : 1u;
    size_t IC = src.getTensorDesc().getDims()[1];
    size_t MB = src.getTensorDesc().getDims()[0];

    size_t OC = prm.out_c;

    size_t OW = SW * (IW - 1lu) + KW - 2lu * PW;
    size_t OH = SH * (IH - 1lu) + KH - 2lu * PH;
    size_t OD = dims_size == 5 ? (SD * (ID - 1) + KD - 2 * PD) : 1u;

    const float *src_data = src.cbuffer().as<float *>();
    float *dst_data = dst.buffer().as<float *>();

    size_t CS1 = OH * OW;
    size_t CS2 = CS1 * OD;
    size_t CS3 = CS2 * OC;

    size_t CI1 = IH * IW;
    size_t CI2 = CI1 * ID;
    size_t CI3 = CI2 * IC;
    
    size_t OC_G = OC / G;
    size_t IC_G = IC / G;

    size_t CK1 = KH * KW;
    size_t CK2 = CK1 * KD;
    size_t CK3 = CK2 * OC_G;
    size_t CK4 = CK3 * IC_G;

    for (size_t g = 0lu; g < G; ++g) {
        size_t g_OC_G = g * OC_G;
        size_t g_IC_G = g * IC_G;
        size_t g_CK4 = g * CK4;
        for (size_t mb = 0lu; mb < MB; ++mb) {
            size_t mb_CS3 = mb * CS3;
            size_t mb_CI3 = mb * CI3;
            for (size_t oc = 0lu; oc < OC_G; ++oc) {
                size_t g_OC_G_oc = g_OC_G + oc;
                size_t mb_CS3_g_OC_G_oc_CS2 = mb_CS3 + g_OC_G_oc * CS2;
                size_t g_CK4_oc_CK2 = g_CK4 + oc * CK2;
                for (size_t od = 0lu; od < OD; ++od) {
                    size_t mb_CS3_g_OC_G_oc_CS2_od_CS1 = mb_CS3_g_OC_G_oc_CS2 + od * CS1;
                    size_t od_PD = od + PD;
                    for (size_t oh = 0lu; oh < OH; ++oh) {
                        size_t mb_CS3_g_OC_G_oc_CS2_od_CS1_oh_OW = mb_CS3_g_OC_G_oc_CS2_od_CS1 + oh * OW;
                        size_t oh_PH = oh + PH;
                        for (size_t ow = 0lu; ow < OW; ++ow) {
                            size_t didx = mb_CS3_g_OC_G_oc_CS2_od_CS1_oh_OW + ow;
                            size_t ow_PW = ow + PW;

                            dst_data[didx] = float(0);
                            if (prm.with_bias) dst_data[didx] += bias_data[g_OC_G_oc];

                            for (size_t ic = 0lu; ic < IC_G; ic++) {
                                size_t mb_CI3_g_IC_G_ic_CI2 = mb_CI3 + (g_IC_G + ic) * CI2;
                                size_t g_CK4_oc_CK2_ic_CK3 = g_CK4_oc_CK2 + ic * CK3;
                                for (int kd = 0lu; kd < KD; kd++) {
                                    if (od_PD < kd) continue;
                                    size_t id = od_PD - kd;
                                    if (id % SD != 0) continue;
                                    id /= SD;
                                    if (id >= ID) continue;
                                    size_t mb_CI3_g_IC_G_ic_CI2_id_CI1 = mb_CI3_g_IC_G_ic_CI2 + id * CI1;
                                    size_t g_CK4_oc_CK2_ic_CK3_kd_CK1 = g_CK4_oc_CK2_ic_CK3 + kd * CK1;
                                    for (size_t kh = 0lu; kh < KH; kh++) {
                                        if (oh_PH < kh) continue;
                                        size_t ih = oh_PH - kh;
                                        if (ih % SH != 0) continue;
                                        ih /= SH;
                                        if (ih >= IH) continue;
                                        size_t mb_CI3_g_IC_G_ic_CI2_id_CI1_ih_IW = mb_CI3_g_IC_G_ic_CI2_id_CI1 + ih * IW;
                                        size_t g_CK4_oc_CK2_ic_CK3_kd_CK1_kh_KW = g_CK4_oc_CK2_ic_CK3_kd_CK1 + kh * KW;
                                        for (size_t kw = 0lu; kw < KW; kw++) {
                                            if (ow_PW < kw) continue;
                                            size_t iw = ow_PW - kw;
                                            if (iw % SW != 0) continue;
                                            iw /= SW;
                                            if (iw >= IW) continue;

                                            size_t sidx = mb_CI3_g_IC_G_ic_CI2_id_CI1_ih_IW + iw;

                                            size_t widx = g_CK4_oc_CK2_ic_CK3_kd_CK1_kh_KW + kw;

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

class MKLDNNDeconvConcatTests: public TestsCommon,
                                    public WithParamInterface<deconv_concat_params> {
    std::string layers_t = R"V0G0N(
        <layer id="2" name="Deconvolution_1" precision="FP32" type="Deconvolution">
            <data kernel="_K_" strides="_KS_"
             pads_begin="_PB_" pads_end="_PE_"
             dilations="1,1,1" output="_OC_" group="_GC_" PrimitivesPriority="_IMPLS_"/>
            <input>
                <port id="0">
                    __INP_DIMS__
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>_IN_</dim>
                    <dim>_OC_</dim>
                    __DECONV_OUT_DIMS__
                </port>
            </output>
            <blobs>
                <weights offset="0" size="262144"/>
            </blobs>
	</layer>
        <layer id="3" name="concat0" precision="FP32" type="Concat">
            <data axis="__AXIS__"/>
            <input>
                <port id="0">
                    <dim>_IN_</dim>
                    <dim>_OC_</dim>
                    __DECONV_OUT_DIMS__
                </port>
                <port id="1">
                    __INP_DIMS__
                </port>
            </input>
            <output>
                <port id="2">
                    __CONCAT_OUT_DIMS__
                </port>
            </output>
        </layer>
)V0G0N";

    std::string edges_t = R"V0G0N(
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="1" to-layer="3" to-port="0"/>
)V0G0N";

    std::string getModel(deconv_concat_params p) {
        std::string model = layers_t;
        
        std::string s_dims;
        for (auto& dim : p.in) {
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(dim) + "</dim>";
        }
	REPLACE_WITH_STR(model, "__INP_DIMS__", s_dims);

        s_dims = "";
        size_t deconv_axis_val = p.in[p.concat.axis];
        int k_len = p.deconv.kernel.size();
        for (size_t i = 2lu; i < p.in.size(); i++) {
            size_t inx = k_len - i + 1;
            size_t dim = p.deconv.stride[inx] * (p.in[i] - 1) + p.deconv.kernel[inx] - 2 * p.deconv.pads_begin[inx];
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(dim) + "</dim>";
            if (i == p.concat.axis) {
                deconv_axis_val = dim;
            }
        }
	REPLACE_WITH_STR(model, "__DECONV_OUT_DIMS__", s_dims);

        s_dims = "";
        for (size_t i = 0lu; i < p.in.size(); i++) {
            size_t val = p.in[i];
            if (i == p.concat.axis) {
                val += deconv_axis_val;
            }
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(val) + "</dim>";
        }
	REPLACE_WITH_STR(model, "__CONCAT_OUT_DIMS__", s_dims);

        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_K_", p.deconv.kernel);
        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_KS_", p.deconv.stride);
        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_PB_", p.deconv.pads_begin);
        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_PE_", p.deconv.pads_end);
        REPLACE_WITH_NUM(model, "_GC_", p.deconv.group);
        REPLACE_WITH_NUM(model, "_OC_", p.deconv.out_c);
        REPLACE_WITH_NUM(model, "_IN_", p.in[0]);
        REPLACE_WITH_NUM(model, "__AXIS__", p.concat.axis);

        std::string impls;
        for (const auto& preferType : p.preferTypes) {
            if (!impls.empty())
                impls += ",";
            impls += "cpu:" + MKLDNNGraphTestClass::getStrPrimitiveDescriptorType(preferType);
        }
        REPLACE_WITH_STR(model, "_IMPLS_", impls);

        model = IRTemplateGenerator::getIRTemplate("Deconvolution_Concat", p.in, "FP32", model, edges_t);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            deconv_concat_params p = ::testing::WithParamInterface<deconv_concat_params>::GetParam();
            std::string model = getModel(p);

            size_t blob_size = p.deconv.out_c * (p.in[1] / p.deconv.group);
            for (int i = 0 ; i < p.deconv.kernel.size(); i++) {
                blob_size *= p.deconv.kernel[i];
            }
            InferenceEngine::SizeVector dims_weights = { blob_size };

            std::vector<InferenceEngine::Blob::Ptr> blob_to_model;
            InferenceEngine::Blob::Ptr weights = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, dims_weights, InferenceEngine::C });
            weights->allocate();
            fill_data(weights->buffer().as<float*>(), weights->size());
            blob_to_model.push_back(weights);

            InferenceEngine::Blob::Ptr bias = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, {p.deconv.out_c}, InferenceEngine::C });
            bias->allocate();
            fill_data(bias->buffer().as<float*>(), bias->size());
            blob_to_model.push_back(bias);

            size_t total_size_in_bytes = 0;
            for (InferenceEngine::Blob::Ptr blb : blob_to_model) total_size_in_bytes += blb->byteSize();

            InferenceEngine::TBlob<uint8_t>::Ptr model_blob =
                    InferenceEngine::make_shared_blob<uint8_t>({ InferenceEngine::Precision::U8, {total_size_in_bytes}, InferenceEngine::C });
            model_blob->allocate();
            uint8_t* model_blob_ptr = model_blob->buffer().as<uint8_t*>();
            for (InferenceEngine::Blob::Ptr blb : blob_to_model) {
                memcpy(model_blob_ptr, blb->buffer().as<uint8_t*>(), blb->byteSize());
                model_blob_ptr += blb->byteSize();
            }

            InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;
            ASSERT_NO_THROW(network = core.ReadNetwork(model, model_blob));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(network);

            InferenceEngine::SizeVector dims_src = p.in;

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>(
                    {InferenceEngine::Precision::FP32, dims_src, InferenceEngine::TensorDesc::getLayoutByDims(p.in)});
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

            // Compare with reference

            auto deconv = network.getLayerByName("Deconvolution_1");
            InferenceEngine::TBlob<float> deconv_ref(deconv->outData[0]->getTensorDesc());
            deconv_ref.allocate();

            ref_deconv_common(*srcPtr, deconv_ref, weights->buffer().as<float*>(), weights->size(),
                    bias->buffer().as<float*>(), bias->size(), p.deconv);

            float *src1_ptr = deconv_ref.buffer();
            size_t src1_size = deconv_ref.size();
            float *src2_ptr = src->buffer();
            size_t src2_size = src->size();
            float *dst_ptr = output->buffer();
            size_t dst_size = output->size();

            int len1 = 1, len2 = 1;
            for (int dim = p.concat.axis; dim < output->getTensorDesc().getDims().size(); dim++) {
                len1 *= deconv->outData[0]->getTensorDesc().getDims()[dim];
                len2 *= src->getTensorDesc().getDims()[dim];
            }

            size_t index1 = 0, index2 = 0, index = 0;
            float max_diff = 0.0001f;
            for (size_t cycle = 0lu; cycle < p.concat.axis; cycle ++) {
                for (int i1 = 0; i1 < len1; i1++) {
                    if (fabs(src1_ptr[index1] - dst_ptr[index]) > max_diff)
                    {
                        FAIL() << "index: " << index << " src: " << src1_ptr[index1] << ", dst: " << dst_ptr[index];
                    }
                    index1++; index++;
                }
                for (int i2 = 0; i2 < len2; i2++) {
                    if (fabs(src2_ptr[index2] - dst_ptr[index]) > max_diff)
                    {
                        FAIL() << "index: " << index << " src: " << src2_ptr[index2] << ", dst: " << dst_ptr[index];
                    }
                    index2++; index++;
                }
            }

        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNDeconvConcatTests, TestsDwConvFusing) {}

INSTANTIATE_TEST_CASE_P(
        TestsDwConvFusing, MKLDNNDeconvConcatTests,
        ::testing::Values(
                deconv_concat_params{{1, 256, 4, 4}, 
                                     { {1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "", 1, 256, false },
                                     {1}, {MKLDNNPlugin::impl_desc_type::gemm_blas}},
                deconv_concat_params{{2, 256, 4, 4},
                                     { {1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "", 1, 256, false },
                                     {1}, {MKLDNNPlugin::impl_desc_type::gemm_blas}},
                deconv_concat_params{{1, 256, 4, 4, 4},
                                     { {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, "", 1, 256, false },
                                     {1}, {MKLDNNPlugin::impl_desc_type::gemm_blas}},
                deconv_concat_params{{2, 256, 4, 4, 4},
                                     { {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, "", 1, 256, false },
                                     {1}, {MKLDNNPlugin::impl_desc_type::gemm_blas}}
        ));
