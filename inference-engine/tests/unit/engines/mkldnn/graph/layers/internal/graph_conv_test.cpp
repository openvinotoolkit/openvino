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
#include <inference_engine/cnn_network_impl.hpp>
#include "tests_common.hpp"


using namespace ::testing;
using namespace std;
using namespace mkldnn;


struct conv_test_params {
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

    size_t num_prim_desc;

    int selectedType;
    std::vector<MKLDNNPlugin::impl_desc_type> preferTypes;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

template <typename data_t>
void ref_conv(const InferenceEngine::TBlob<data_t> &src, const data_t *weights, const size_t weightsSize,
                InferenceEngine::TBlob<data_t> &dst, conv_test_params prm) {
    size_t KW = prm.krn_w;
    size_t KH = prm.krn_h;
    size_t GC = prm.grp_c;

    size_t IC = src.dims()[1];
    size_t IH = src.dims()[2];
    size_t IW = src.dims()[3];

    size_t OW = (IW + 2 * prm.pad_w - prm.krn_w) / prm.str_w + 1;
    size_t OH = (IH + 2 * prm.pad_h - prm.krn_h) / prm.str_h + 1;
    size_t OC = prm.out_c;


    const data_t *src_data = src.readOnly();
    const data_t *weights_data = weights;
    const data_t *bias_data = weights_data + KW * KH * OC * IC / GC;
    data_t *dst_data = dst.data();

    IE_ASSERT(KW * KH * OC * IC / GC + OC == weightsSize);
    IE_ASSERT(OW == dst.dims()[0]);
    IE_ASSERT(OH == dst.dims()[1]);

    for (uint32_t g = 0; g < GC; g++) {
        for (uint32_t oc = 0; oc < OC / GC; oc++) {
            for (uint32_t oh = 0; oh < OH; oh++) {
                for (uint32_t ow = 0; ow < OW; ow++) {
                    size_t oidx = g * OC / GC * OH * OW
                                  + oc * OH * OW + oh * OW + ow;
                    dst_data[oidx] = bias_data[g * OC / GC + oc];

                    for (size_t ic = 0; ic < IC / GC; ic++) {
                        for (size_t kh = 0; kh < KH; kh++) {
                            for (size_t kw = 0; kw < KW; kw++) {
                                int32_t iw = ow * prm.str_w - prm.pad_w + kw;
                                int32_t ih = oh * prm.str_h - prm.pad_h + kh;
                                if (iw < 0 || iw >= (int32_t)IW || ih < 0
                                    || ih >= (int32_t)IH)
                                    continue;
                                size_t iidx = g * IC / GC * IH * IW
                                              + ic * IH * IW + ih * IW + iw;
                                size_t widx = g * OC / GC * IC / GC * KH * KW
                                              + oc * IC / GC * KH * KW
                                              + ic * KH * KW + kh * KW + kw;

                                dst_data[ oidx] += src_data[iidx] * weights_data[widx];
                            }
                        }
                    }
                }
            }
        }
    }
}

class MKLDNNGraphConvolutionTests: public TestsCommon,
                                   public WithParamInterface<conv_test_params> {
    std::string model_t = R"V0G0N(
<Net Name="Convolution_Only" version="2" precision="FP32" batch="1">
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
        <layer name="conv1" id="1" type="Convolution" precision="FP32">
            <convolution stride-x="_SW_" stride-y="_SH_"
                         pad-x="_PW_"    pad-y="_PH_"
                         kernel-x="_KW_" kernel-y="_KH_"
                         output="_OC_"   group="_GC_" PrimitivesPriority="_IMPLS_"/>

            <weights offset="0" size="_S1_" />
            <biases offset="_S1_" size="_S2_" />

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
    std::string getModel(conv_test_params p) {
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
        REPLACE_WITH_NUM(model, "_OH_", (p.in.h + 2 * p.pad_h - p.krn_h) / p.str_h + 1);
        REPLACE_WITH_NUM(model, "_OW_", (p.in.w + 2 * p.pad_w - p.krn_w) / p.str_w + 1);

        size_t w_data_size = (p.krn_w * p.krn_h * p.out_c * p.in.c / p.grp_c) * sizeof(float);
        size_t b_data_size = p.out_c * sizeof(float);
        REPLACE_WITH_NUM(model, "_S1_", w_data_size);
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
            conv_test_params p = ::testing::WithParamInterface<conv_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>(InferenceEngine::Precision::U8, InferenceEngine::C, {(p.krn_w * p.krn_h * p.out_c * p.in.c / p.grp_c + p.out_c)
                                                              * sizeof(float)});
            weights->allocate();
            fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
            InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

            net_reader.SetWeights(weights_ptr);

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());

            auto& nodes = graph.getNodes();
            nodes = graph.getNodes();
            for (auto &node : nodes) {
                if (node->getType() == MKLDNNPlugin::Convolution) {
                    ASSERT_LE(p.num_prim_desc, node->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(node->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, node->getSelectedPrimitiveDescriptor());
                    ASSERT_EQ(p.selectedType,
                              node->getSelectedPrimitiveDescriptor()->getImplementationType() & p.selectedType);
                }
            }

            InferenceEngine::SizeVector dims_src = {p.in.n, p.in.c, p.in.h, p.in.w};

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, InferenceEngine::NCHW, dims_src);
            src->allocate();
            fill_data(src->buffer(), src->size());

            auto * srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());

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
            ref_conv(*srcPtr, (const float *)weights->buffer(), weights->size() / sizeof(float), dst_ref, p);
            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphConvolutionTests, TestsConvolution) {}

INSTANTIATE_TEST_CASE_P(
        TestConvolution, MKLDNNGraphConvolutionTests,
        ::testing::Values(
                conv_test_params{{1, 9, 16, 32},
                                 1, 1, 1, 1, 0, 0, 17, 1, 7, MKLDNNPlugin::impl_desc_type::jit | MKLDNNPlugin::impl_desc_type::_1x1,
                },
                conv_test_params{{1, 9, 32, 16},
                                 2, 4, 1, 1, 0, 0, 17, 1, 5, MKLDNNPlugin::impl_desc_type::jit },
                conv_test_params{{1, 9, 32, 16},
                                 2, 4, 2, 1, 0, 0, 17, 1, 5, MKLDNNPlugin::impl_desc_type::jit },
                conv_test_params{{1, 3, 40, 40},
                                 3, 3, 1, 2, 0, 0, 20, 1, 5, MKLDNNPlugin::impl_desc_type::jit },
                conv_test_params{{1, 1, 40, 40},
                                 3, 3, 1, 2, 0, 0, 20, 1, 5, MKLDNNPlugin::impl_desc_type::jit },
                conv_test_params{{1, 1, 32, 16},
                                 2, 4, 2, 1, 0, 0, 17, 1, 5, MKLDNNPlugin::impl_desc_type::jit },
                conv_test_params{{1, 9, 16, 32},
                                 1, 1, 1, 1, 0, 0, 17, 1, 7, MKLDNNPlugin::impl_desc_type::gemm,
                                 {MKLDNNPlugin::impl_desc_type::gemm_any,
                                  MKLDNNPlugin::impl_desc_type::gemm_blas,
                                  MKLDNNPlugin::impl_desc_type::gemm_avx512,
                                  MKLDNNPlugin::impl_desc_type::gemm_avx2,
                                  MKLDNNPlugin::impl_desc_type::gemm_sse42}
                },
                conv_test_params{{1, 9, 32, 16},
                                 2, 4, 1, 1, 0, 0, 17, 1, 5, MKLDNNPlugin::impl_desc_type::ref_any, {MKLDNNPlugin::impl_desc_type::ref_any} }));

class MKLDNNGraphDynBatchConvolutionTests: public MKLDNNGraphConvolutionTests {
protected:
    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            conv_test_params p = ::testing::WithParamInterface<conv_test_params>::GetParam();
            std::string model = getModel(p);
            size_t MB = p.in.n;
            if (MB < 2)
                MB = 2;

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>(InferenceEngine::Precision::U8, InferenceEngine::C,
                    {(p.krn_w * p.krn_h * p.out_c * p.in.c / p.grp_c + p.out_c) * sizeof(float)});
            weights->allocate();
            fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
            InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

            net_reader.SetWeights(weights_ptr);
            InferenceEngine::CNNNetwork network = net_reader.getNetwork();
            auto implNet = dynamic_cast<InferenceEngine::details::CNNNetworkImpl *>(&((InferenceEngine::ICNNNetwork&)network));
            ASSERT_NE(nullptr, implNet) << "Failed to cast ICNNNetwork to CNNNetworkImpl";
            InferenceEngine::ResponseDesc resp;
            InferenceEngine::StatusCode sts  = implNet->setBatchSizeReshape(MB, &resp);
            ASSERT_EQ((int)InferenceEngine::StatusCode::OK, sts) << resp.msg;

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());

            InferenceEngine::SizeVector dims_src = {MB, p.in.c, p.in.h, p.in.w};

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, InferenceEngine::NCHW, dims_src);
            src->allocate();
            fill_data(src->buffer(), src->size());

            auto * srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());

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

            auto checkConvolution = [](const MKLDNNPlugin::MKLDNNNodePtr& node) {
                return node->getType() == MKLDNNPlugin::Convolution ||
                       node->getType() == MKLDNNPlugin::Convolution_Activation ||
                       node->getType() == MKLDNNPlugin::Convolution_Sum ||
                       node->getType() == MKLDNNPlugin::Convolution_Sum_Activation;
            };

            graph.checkDynBatch(srcs, outputBlobs, MB, MB, checkConvolution, MKLDNNGraphTestClass::CheckDynBatchType::Child);
            graph.checkDynBatch(srcs, outputBlobs, 1, MB, checkConvolution, MKLDNNGraphTestClass::CheckDynBatchType::Child);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphDynBatchConvolutionTests, TestsDynBatchConvolution) {}

INSTANTIATE_TEST_CASE_P(
        TestDynBatchConvolution, MKLDNNGraphDynBatchConvolutionTests,
        ::testing::Values(
                conv_test_params{{1, 8, 16, 32},
                                 1, 1, 1, 1, 0, 0, 17, 1, 7, MKLDNNPlugin::impl_desc_type::jit | MKLDNNPlugin::impl_desc_type::_1x1,
                },
                conv_test_params{{1, 9, 32, 16},
                                 2, 4, 1, 1, 0, 0, 17, 1, 5, MKLDNNPlugin::impl_desc_type::jit },
                conv_test_params{{1, 9, 32, 16},
                                 2, 4, 2, 1, 0, 0, 17, 1, 5, MKLDNNPlugin::impl_desc_type::jit },
                conv_test_params{{1, 3, 40, 40},
                                 3, 3, 1, 2, 0, 0, 20, 1, 5, MKLDNNPlugin::impl_desc_type::jit },
                conv_test_params{{1, 1, 40, 40},
                                 3, 3, 1, 2, 0, 0, 20, 1, 5, MKLDNNPlugin::impl_desc_type::jit },
                conv_test_params{{1, 1, 32, 16},
                                 2, 4, 2, 1, 0, 0, 17, 1, 5, MKLDNNPlugin::impl_desc_type::jit },
                conv_test_params{{1, 9, 16, 32},
                                 1, 1, 1, 1, 0, 0, 17, 1, 7, MKLDNNPlugin::impl_desc_type::gemm,
                                 {MKLDNNPlugin::impl_desc_type::gemm_any,
                                  MKLDNNPlugin::impl_desc_type::gemm_blas,
                                  MKLDNNPlugin::impl_desc_type::gemm_avx512,
                                  MKLDNNPlugin::impl_desc_type::gemm_avx2,
                                  MKLDNNPlugin::impl_desc_type::gemm_sse42}
                },
                conv_test_params{{1, 9, 32, 16},
                                 2, 4, 1, 1, 0, 0, 17, 1, 5, MKLDNNPlugin::impl_desc_type::ref_any, {MKLDNNPlugin::impl_desc_type::ref_any} }));
