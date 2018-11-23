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


struct deconv_test_params {
    struct {
        size_t n;
        size_t c;
        size_t h;
        size_t w;
    } in;

    size_t krn_w;
    size_t krn_h;
    size_t str_w;
    size_t str_h;
    size_t pad_w;
    size_t pad_h;

    size_t out_c;
    size_t grp_c;

    bool with_bias;

    size_t num_prim_desc;

    std::vector<int> selectedTypes;
    std::vector<MKLDNNPlugin::impl_desc_type> preferTypes;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

template <typename data_t>
void ref_deconv(const InferenceEngine::TBlob<data_t> &src, const InferenceEngine::Blob::Ptr &weights, const InferenceEngine::Blob::Ptr &bias,
                InferenceEngine::TBlob<data_t> &dst, deconv_test_params prm) {

    size_t G  = prm.grp_c;
    size_t KW = prm.krn_w;
    size_t KH = prm.krn_h;

    size_t PW = prm.pad_w;
    size_t PH = prm.pad_h;

    size_t SW = prm.str_w;
    size_t SH = prm.str_h;

    size_t IW = src.dims()[3];
    size_t IH = src.dims()[2];
    size_t IC = src.dims()[1];
    size_t MB = src.dims()[0];

    size_t OC = prm.out_c;

    size_t OW = SW * (IW - 1) + KW - 2 * PW;
    size_t OH = SH * (IH - 1) + KH - 2 * PH;

    const data_t *src_data = src.readOnly();
    const data_t *weights_data = weights->buffer().as<data_t*>();
    const data_t *bias_data = bias->buffer().as<data_t*>();

    data_t *dst_data = dst.data();

    for (int g = 0; g < G; ++g) {
        for (int mb = 0; mb < MB; ++mb) {
            for (int oc = 0; oc < OC / G; ++oc) {
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        size_t didx = mb * OC * OH * OW
                                      + (g * OC / G + oc) * OH * OW + oh * OW + ow;

                        dst_data[didx] = data_t(0);
                        if (prm.with_bias) dst_data[didx] += bias_data[oc];

                        for (int ic = 0; ic < IC / G; ic++) {
                            for (int kh = 0; kh < KH; kh++) {
                                for (int kw = 0; kw < KW; kw++) {
                                    if (ow + PW < kw || oh + PH < kh)
                                        continue;

                                    size_t iw = ow - kw + PW;
                                    size_t ih = oh - kh + PH;

                                    if (iw % SW != 0 || ih % SH != 0)
                                        continue;

                                    iw /= SW;
                                    ih /= SH;

                                    if (ih < IH && iw < IW) {
                                        size_t sidx = mb * IC * IH * IW
                                                      + (g * IC / G + ic) * IH * IW + ih * IW
                                                      + iw;

                                        size_t widx = g * (IC / G) * (OC / G) * KH * KW +
                                                      ic * (OC / G) * KH * KW +
                                                      + oc * KH * KW + kh * KW
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

class MKLDNNGraphDeconvolutionalTests: public TestsCommon,
                                     public WithParamInterface<deconv_test_params> {
    std::string model_t = R"V0G0N(
<Net Name="Deconvolution_Only" version="2" precision="FP32" batch="1">
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
        <layer name="deconv1" id="1" type="Deconvolution" precision="FP32">
            <deconvolution stride-x="_SW_" stride-y="_SH_"
                         pad-x="_PW_"    pad-y="_PH_"
                         kernel-x="_KW_" kernel-y="_KH_"
                         output="_OC_"   group="_GC_"/>

            <weights offset="0" size="_S1_" />
            <biases offset="_OFF2_" size="_S2_" />

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
                    <dim>_OC_</dim>
                    <dim>_OH_</dim>
                    <dim>_OW_</dim>
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
    std::string getModel(deconv_test_params p) {
        std::string model = model_t;

        REPLACE_WITH_NUM(model, "_IW_", p.in.w);
        REPLACE_WITH_NUM(model, "_IH_", p.in.h);
        REPLACE_WITH_NUM(model, "_IC_", p.in.c);
        REPLACE_WITH_NUM(model, "_IN_", p.in.n);

        REPLACE_WITH_NUM(model, "_KW_", p.krn_w);
        REPLACE_WITH_NUM(model, "_KH_", p.krn_h);
        REPLACE_WITH_NUM(model, "_SW_", p.str_w);
        REPLACE_WITH_NUM(model, "_SH_", p.str_h);
        REPLACE_WITH_NUM(model, "_PW_", p.pad_w);
        REPLACE_WITH_NUM(model, "_PH_", p.pad_h);

        REPLACE_WITH_NUM(model, "_GC_", p.grp_c);
        REPLACE_WITH_NUM(model, "_OC_", p.out_c);
        REPLACE_WITH_NUM(model, "_OH_", p.str_h * (p.in.h - 1) + p.krn_h - 2 * p.pad_h);
        REPLACE_WITH_NUM(model, "_OW_", p.str_w * (p.in.w - 1) + p.krn_w - 2 * p.pad_w);

        size_t w_data_size = (p.krn_w * p.krn_h * p.out_c * (p.in.c / p.grp_c)) * sizeof(float);
        REPLACE_WITH_NUM(model, "_S1_", w_data_size);

        if (!p.with_bias) REMOVE_LINE(model, "<biases offset=\"_OFF2_\" size=\"_S2_\" />");
        size_t b_data_size = p.out_c * sizeof(float);
        REPLACE_WITH_NUM(model, "_OFF2_", w_data_size);
        REPLACE_WITH_NUM(model, "_S2_", b_data_size);

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

            InferenceEngine::SizeVector dims_weights = {p.krn_w * p.krn_h * p.out_c * (p.in.c / p.grp_c)};

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

            ref_deconv(*srcPtr, weights, bias, dst_ref, p);

            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphDeconvolutionalTests, TestsDeconvolution) {}


INSTANTIATE_TEST_CASE_P(
        TestDeconvolution, MKLDNNGraphDeconvolutionalTests,
        ::testing::Values(
                deconv_test_params{{1, 3, 3, 3}, 3, 3, 1, 1, 0, 0, 2, 1, false, 2, {MKLDNNPlugin::impl_desc_type::jit} },
                deconv_test_params{{3, 3, 3, 3}, 4, 3, 1, 1, 0, 0, 2, 1, false, 2, {MKLDNNPlugin::impl_desc_type::jit} },
                deconv_test_params{{1, 3, 3, 3}, 4, 3, 1, 2, 0, 0, 2, 1, false, 2, {MKLDNNPlugin::impl_desc_type::gemm, MKLDNNPlugin::impl_desc_type::jit} },
                deconv_test_params{{1, 3, 3, 3}, 4, 3, 2, 2, 0, 0, 2, 1, false, 2, {MKLDNNPlugin::impl_desc_type::gemm, MKLDNNPlugin::impl_desc_type::jit} },
                deconv_test_params{{4, 17, 3, 3}, 4, 3, 2, 2, 0, 0, 2, 1, false, 2, {MKLDNNPlugin::impl_desc_type::gemm, MKLDNNPlugin::impl_desc_type::jit} },
                /*deconv_test_params{{2, 8, 5, 5}, 4, 4, 2, 2, 1, 1, 8, 2, false, 3, {MKLDNNPlugin::impl_desc_type::gemm}},*/
                deconv_test_params{{2, 8, 5, 5}, 4, 4, 2, 2, 1, 1, 8, 8, false, 4, {MKLDNNPlugin::impl_desc_type::jit | MKLDNNPlugin::impl_desc_type::_dw}},
                deconv_test_params{{2, 8, 5, 5}, 8, 8, 4, 4, 1, 1, 8, 8, false, 4, {MKLDNNPlugin::impl_desc_type::jit | MKLDNNPlugin::impl_desc_type::_dw}},
                deconv_test_params{{2, 8, 5, 5}, 4, 8, 2, 4, 1, 1, 8, 8, false, 4, {MKLDNNPlugin::impl_desc_type::jit | MKLDNNPlugin::impl_desc_type::_dw}},
                deconv_test_params{{1, 3, 3, 3}, 3, 3, 1, 1, 0, 0, 2, 1, true, 2, {MKLDNNPlugin::impl_desc_type::jit} },
                deconv_test_params{{3, 3, 3, 3}, 4, 3, 1, 1, 0, 0, 2, 1, true, 2, {MKLDNNPlugin::impl_desc_type::jit} },
                deconv_test_params{{1, 3, 3, 3}, 4, 3, 1, 2, 0, 0, 2, 1, true, 2, {MKLDNNPlugin::impl_desc_type::gemm, MKLDNNPlugin::impl_desc_type::jit} },
                deconv_test_params{{1, 3, 3, 3}, 4, 3, 2, 2, 0, 0, 2, 1, true, 2, {MKLDNNPlugin::impl_desc_type::gemm, MKLDNNPlugin::impl_desc_type::jit} },
                deconv_test_params{{4, 17, 3, 3}, 4, 3, 2, 2, 0, 0, 2, 1, true, 2, {MKLDNNPlugin::impl_desc_type::gemm, MKLDNNPlugin::impl_desc_type::jit} },
                /*deconv_test_params{{2, 8, 5, 5}, 4, 4, 2, 2, 1, 1, 8, 2, true, 3, {MKLDNNPlugin::impl_desc_type::gemm}},*/
                deconv_test_params{{2, 8, 5, 5}, 4, 4, 2, 2, 1, 1, 8, 8, true, 4, {MKLDNNPlugin::impl_desc_type::jit | MKLDNNPlugin::impl_desc_type::_dw}},
                deconv_test_params{{2, 8, 5, 5}, 8, 8, 4, 4, 1, 1, 8, 8, true, 4, {MKLDNNPlugin::impl_desc_type::jit | MKLDNNPlugin::impl_desc_type::_dw}},
                deconv_test_params{{2, 8, 5, 5}, 4, 8, 2, 4, 1, 1, 8, 8, true, 4, {MKLDNNPlugin::impl_desc_type::jit | MKLDNNPlugin::impl_desc_type::_dw}}
        ));

class MKLDNNGraphDynBatchDeconvolutionalTests: public MKLDNNGraphDeconvolutionalTests {
protected:
    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            deconv_test_params p = ::testing::WithParamInterface<deconv_test_params>::GetParam();
            std::string model = getModel(p);
            size_t MB = p.in.n;
            if (MB < 2)
                MB = 2;

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            InferenceEngine::SizeVector dims_weights = {p.krn_w * p.krn_h * p.out_c * (p.in.c / p.grp_c)};

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

            InferenceEngine::SizeVector dims_src = {MB, p.in.c, p.in.h, p.in.w};
            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, InferenceEngine::NCHW, dims_src);
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
                deconv_test_params{{1, 3, 3, 3}, 3, 3, 1, 1, 0, 0, 2, 1, false, 5, {MKLDNNPlugin::impl_desc_type::jit} },
                deconv_test_params{{3, 3, 3, 3}, 4, 3, 1, 1, 0, 0, 2, 1, false, 5, {MKLDNNPlugin::impl_desc_type::jit} },
                deconv_test_params{{1, 3, 3, 3}, 4, 3, 1, 2, 0, 0, 2, 1, false, 4, {MKLDNNPlugin::impl_desc_type::gemm, MKLDNNPlugin::impl_desc_type::jit} },
                deconv_test_params{{1, 3, 3, 3}, 4, 3, 2, 2, 0, 0, 2, 1, false, 3, {MKLDNNPlugin::impl_desc_type::gemm, MKLDNNPlugin::impl_desc_type::jit} },
                deconv_test_params{{4, 17, 3, 3}, 4, 3, 2, 2, 0, 0, 2, 1, false, 3, {MKLDNNPlugin::impl_desc_type::gemm, MKLDNNPlugin::impl_desc_type::jit} },
                deconv_test_params{{2, 8, 5, 5}, 4, 4, 2, 2, 1, 1, 8, 2, false, 3, {MKLDNNPlugin::impl_desc_type::gemm}},
                deconv_test_params{{2, 8, 5, 5}, 4, 4, 2, 2, 1, 1, 8, 8, false, 4, {MKLDNNPlugin::impl_desc_type::jit | MKLDNNPlugin::impl_desc_type::_dw}},
                deconv_test_params{{2, 8, 5, 5}, 8, 8, 4, 4, 1, 1, 8, 8, false, 4, {MKLDNNPlugin::impl_desc_type::jit | MKLDNNPlugin::impl_desc_type::_dw}},
                deconv_test_params{{2, 8, 5, 5}, 4, 8, 2, 4, 1, 1, 8, 8, false, 4, {MKLDNNPlugin::impl_desc_type::jit | MKLDNNPlugin::impl_desc_type::_dw}}
        ));
