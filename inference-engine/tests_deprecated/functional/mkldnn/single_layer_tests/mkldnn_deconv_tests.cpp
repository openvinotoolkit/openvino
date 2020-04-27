// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <string>
#include <ie_core.hpp>

#include "tests_common.hpp"
#include "../common_single_layer_tests/deconv_ref.hpp"
#include "ir_gen_helper.hpp"
#include "common_test_utils/common_layers_params.hpp"

using namespace ::testing;
using namespace InferenceEngine;
using namespace single_layer_tests;

struct deconv_test_params {
    std::string device_name;

    struct {
        size_t w;
        size_t h;
        size_t c;
    } in;

    size_t krn_w;
    size_t krn_h;
    size_t str_w;
    size_t str_h;
    size_t pad_w;
    size_t pad_h;

    size_t out_c;

    bool with_bias;
};

template<typename data_t>
void ref_deconv(const Blob::Ptr &src, const Blob::Ptr &weights, const Blob::Ptr &bias,
                Blob::Ptr &dst_ref, deconv_test_params p) {
    const float *weights_data = (const float *) weights->buffer();
    size_t bias_size = p.out_c;
    size_t weights_size = weights->size() / sizeof(float) - bias_size;
    const float *bias_data = p.with_bias ? (const float *) bias->buffer() : nullptr;
    CommonTestUtils::conv_common_params params;
    params.kernel.insert(X_AXIS, p.krn_w);
    params.kernel.insert(Y_AXIS, p.krn_h);
    params.stride.insert(X_AXIS, p.str_w);
    params.stride.insert(Y_AXIS, p.str_h);
    params.pads_begin.insert(X_AXIS, p.pad_w);
    params.pads_begin.insert(Y_AXIS, p.pad_h);
    params.out_c = p.out_c;
    ref_deconv_common<float>({ src }, *dst_ref.get(), weights_data, weights_size, bias_data, bias_size, params);
}

class smoke_CPUDeconvolutionOnlyTest : public TestsCommon,
                                    public WithParamInterface<deconv_test_params> {
    std::string layers_t = R"V0G0N(
        <layer name="deconv1" id="1" type="Deconvolution" precision="FP32">
            <deconvolution
                kernel="_KH_,_KW_"
                strides="_SH_,_SW_"
                pads_begin="_PH_,_PW_"  pads_end="_PH_,_PW_"
                output="_OC_"/>

            <weights offset="0" size="_S1_" />
            <biases offset="_OFF2_" size="_S2_" />

            <input>
                <port id="0">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>_IN_</dim>
                    <dim>_OC_</dim>
                    <dim>_OH_</dim>
                    <dim>_OW_</dim>
                </port>
            </output>
        </layer>
)V0G0N";
    
    std::string edges_t = R"V0G0N(
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
)V0G0N";

    std::string getModel(deconv_test_params p) {
        std::string model = layers_t;

        REPLACE_WITH_NUM(model, "_IN_", 1);
        REPLACE_WITH_NUM(model, "_IW_", p.in.w);
        REPLACE_WITH_NUM(model, "_IH_", p.in.h);
        REPLACE_WITH_NUM(model, "_IC_", p.in.c);

        REPLACE_WITH_NUM(model, "_KW_", p.krn_w);
        REPLACE_WITH_NUM(model, "_KH_", p.krn_h);
        REPLACE_WITH_NUM(model, "_SW_", p.str_w);
        REPLACE_WITH_NUM(model, "_SH_", p.str_h);
        REPLACE_WITH_NUM(model, "_PW_", p.pad_w);
        REPLACE_WITH_NUM(model, "_PH_", p.pad_h);

        REPLACE_WITH_NUM(model, "_OC_", p.out_c);
        REPLACE_WITH_NUM(model, "_OH_", p.str_h * (p.in.h - 1) + p.krn_h - 2 * p.pad_h);
        REPLACE_WITH_NUM(model, "_OW_", p.str_w * (p.in.w - 1) + p.krn_w - 2 * p.pad_w);

        if (!p.with_bias) REMOVE_LINE(model, "<biases offset=\"_OFF2_\" size=\"_S2_\" />");

        size_t w_data_size = (p.krn_w * p.krn_h * p.out_c * p.in.c) * sizeof(float);
        size_t b_data_size = p.out_c * sizeof(float);
        REPLACE_WITH_NUM(model, "_S1_", w_data_size);
        REPLACE_WITH_NUM(model, "_OFF2_", w_data_size);
        REPLACE_WITH_NUM(model, "_S2_", b_data_size);
        
        model = IRTemplateGenerator::getIRTemplate("Deconvolution_Only", {1lu, p.in.c, p.in.h, p.in.w}, "FP32", model, edges_t);

        return model;
    }

protected:
    virtual void SetUp() {
        try {
            deconv_test_params p = ::testing::WithParamInterface<deconv_test_params>::GetParam();
            std::string model = getModel(p);

            std::vector<Blob::Ptr> blob_to_model;
            Blob::Ptr weights = make_shared_blob<float>(TensorDesc(Precision::FP32,
                                                        {p.krn_w * p.krn_h * p.out_c * p.in.c}, C));
            weights->allocate();
            fill_data(weights->buffer().as<float *>(), weights->size());
            blob_to_model.push_back(weights);

            Blob::Ptr bias = nullptr;
            if (p.with_bias) {
                bias = make_shared_blob<float>(TensorDesc(Precision::FP32,
                                               {p.krn_w * p.krn_h * p.out_c * p.in.c}, C));
                bias->allocate();
                fill_data(bias->buffer().as<float *>(), bias->size());
                blob_to_model.push_back(bias);
            }

            size_t total_size_in_bytes = 0;
            for (Blob::Ptr blb : blob_to_model) total_size_in_bytes += blb->byteSize();

            TBlob<uint8_t>::Ptr model_blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, { total_size_in_bytes }, C));
            model_blob->allocate();
            uint8_t *model_blob_ptr = model_blob->buffer().as<uint8_t *>();
            for (Blob::Ptr blb : blob_to_model) {
                memcpy(model_blob_ptr, blb->buffer().as<uint8_t *>(), blb->byteSize());
                model_blob_ptr += blb->byteSize();
            }

            Core ie;
            CNNNetwork network = ie.ReadNetwork(model, model_blob);

            SizeVector dims_src = {p.in.w, p.in.h, p.in.c, 1};  // 1 is a batch size

            Blob::Ptr src = make_shared_blob<float>(TensorDesc(Precision::FP32, SizeVector(dims_src.rbegin(), dims_src.rend()), NCHW));
            src->allocate();
            fill_data(src->buffer().as<float *>(), src->size());

            size_t OW = p.str_w * (p.in.w - 1) + p.krn_w - 2 * p.pad_w;
            size_t OH = p.str_h * (p.in.h - 1) + p.krn_h - 2 * p.pad_h;

            SizeVector dims_dst = {OW, OH, p.out_c, 1};

            Blob::Ptr dst = make_shared_blob<float>(TensorDesc(Precision::FP32, SizeVector(dims_dst.rbegin(), dims_dst.rend()), NCHW));
            dst->allocate();
            fill_data(dst->buffer().as<float *>(), dst->size());

            ExecutableNetwork exeNetwork = ie.LoadNetwork(network, "CPU");
            InferRequest inferRequest = exeNetwork.CreateInferRequest();
            OutputsDataMap outInfo;
            outInfo = network.getOutputsInfo();
            ASSERT_EQ(outInfo.size(), 1);
            ASSERT_NE(outInfo.begin()->second, nullptr);
            inferRequest.SetBlob(network.getInputsInfo().begin()->first, src);
            inferRequest.SetBlob(outInfo.begin()->first, dst);
            inferRequest.Infer();

            Blob::Ptr dst_ref = make_shared_blob<float>(TensorDesc(Precision::FP32, SizeVector(dims_dst.rbegin(), dims_dst.rend()), NCHW));
            dst_ref->allocate();

            ref_deconv<float>(src, weights, bias, dst_ref, p);

            compare(*dst.get(), *dst_ref.get());
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(smoke_CPUDeconvolutionOnlyTest, TestsDeconvolution) {}

INSTANTIATE_TEST_CASE_P(
        TestDeconvolution, smoke_CPUDeconvolutionOnlyTest,
        ::testing::Values(
                deconv_test_params{"CPU",
                                   {3, 3, 3},
                                   3, 3, 1, 1, 0, 0, 2, true},
                deconv_test_params{"CPU",
                                   {3, 3, 3},
                                   4, 3, 1, 1, 0, 0, 2, true},
                deconv_test_params{"CPU",
                                   {3, 3, 3},
                                   4, 3, 1, 2, 0, 0, 2, true},
                deconv_test_params{"CPU",
                                   {4, 4, 3},
                                   3, 3, 1, 2, 0, 0, 2, true}, // jit impl should work
                deconv_test_params{"CPU",
                                   {4, 4, 3},
                                   3, 3, 1, 2, 0, 0, 2, false}, // jit impl should work
                deconv_test_params{"CPU",
                                   {3, 3, 3},
                                   3, 3, 1, 1, 0, 0, 2, false},
                deconv_test_params{"CPU",
                                   {3, 3, 3},
                                   4, 3, 1, 1, 0, 0, 2, false},
                deconv_test_params{"CPU",
                                   {3, 3, 3},
                                   4, 3, 1, 2, 0, 0, 2, false}));


/*** TBD ***/


