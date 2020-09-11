// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <gtest/gtest.h>
#include <ie_core.hpp>

#include "tests_common.hpp"
#include "single_layer_common.hpp"
#include "ir_gen_helper.hpp"

using namespace ::testing;
using namespace InferenceEngine;
using namespace single_layer_tests;

struct batchnorm4D_test_params {
    std::string device_name;

    struct {
        size_t w;
        size_t h;
        size_t c;
    } in;

    double epsilon;
};

template <typename data_t>
void ref_batchnorm4D(const TBlob<data_t> &src, const data_t *variance, const data_t *mean,
                    TBlob<data_t> &dst, batchnorm4D_test_params prm) {
    size_t IW = src.getTensorDesc().getDims()[3];
    size_t IH = src.getTensorDesc().getDims()[2];
    size_t IC = src.getTensorDesc().getDims()[1];
    size_t MB = src.getTensorDesc().getDims()[0];

    const double eps = prm.epsilon;

    const data_t *src_data = src.readOnly();
    data_t *dst_data = dst.data();

    for (int c = 0; c < IC; ++c) {
        data_t v_mean = mean[c];
        data_t v_variance = variance[c];
        data_t sqrt_variance = 0;

        sqrt_variance = 1. / sqrt(v_variance + eps);

        for (int n = 0; n < MB; ++n)
            for (int h = 0; h < IH; ++h)
                for (int w = 0; w < IW; ++w) {
                    size_t idx = n * IC * IH * IW
                                 + c * IH * IW
                                 + h * IW + w;
                    dst_data[idx] = (src_data[idx] - v_mean) * sqrt_variance;
                }
    }
}

class smoke_CPUBatchNorn4DOnlyTest: public TestsCommon,
                                public WithParamInterface<batchnorm4D_test_params> {
    std::string layers_t = R"V0G0N(
        <layer name="batchNorm" id="1" type="BatchNormalization" precision="FP32">
            <batch_norm_data epsilon="_EPSILON_"/>

            <weights offset="0" size="_S1_" />
            <biases offset="_S1_" size="_S2_" />

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
    std::string getModel(batchnorm4D_test_params p) {
        std::string model = layers_t;

        REPLACE_WITH_NUM(model, "_IN_", 1);
        REPLACE_WITH_NUM(model, "_IW_", p.in.w);
        REPLACE_WITH_NUM(model, "_IH_", p.in.h);
        REPLACE_WITH_NUM(model, "_IC_", p.in.c);
        REPLACE_WITH_NUM(model, "_EPSILON_", p.epsilon);

        REPLACE_WITH_NUM(model, "_OW_", p.in.w);
        REPLACE_WITH_NUM(model, "_OH_", p.in.h);
        REPLACE_WITH_NUM(model, "_OC_", p.in.c);

        size_t w_data_size = p.in.c * sizeof(float);
        size_t b_data_size = p.in.c * sizeof(float);
        REPLACE_WITH_NUM(model, "_S1_", w_data_size);
        REPLACE_WITH_NUM(model, "_S2_", b_data_size);

        model = IRTemplateGenerator::getIRTemplate("BatchNorm4D_Only", {1lu, p.in.c, p.in.h, p.in.w}, "FP32", model, edges_t);

        return model;
    }

protected:
    virtual void SetUp() {
        try {
            batchnorm4D_test_params p = ::testing::WithParamInterface<batchnorm4D_test_params>::GetParam();
            std::string model = getModel(p);

            TBlob<uint8_t> *weights = new TBlob<uint8_t>(TensorDesc(Precision::U8, {p.in.c * 2 * sizeof(float)}, C));
            weights->allocate();
            fill_data(weights->buffer(), weights->size() / sizeof(float));
            float * data = weights->buffer();
            for (size_t i = 0; i < weights->size() / sizeof(float); i++) {
                if (data[i] < 0) {
                    data[i] *= -1;
                }
            }

            TBlob<uint8_t>::Ptr weights_ptr = TBlob<uint8_t>::Ptr(weights);

            Core ie;
            CNNNetwork network = ie.ReadNetwork(model, weights_ptr);

            SizeVector dims_src = {p.in.w,
                                   p.in.h,
                                   p.in.c,
                                   1};          // 1 is a batch size
            Blob::Ptr src = make_shared_blob<float>(TensorDesc(Precision::FP32, SizeVector(dims_src.rbegin(), dims_src.rend()), NCHW));
            src->allocate();
            fill_data(src->buffer().as<float *>(), src->size());

            Blob::Ptr dst = make_shared_blob<float>(TensorDesc(Precision::FP32, SizeVector(dims_src.rbegin(), dims_src.rend()), NCHW));
            dst->allocate();

            ExecutableNetwork exeNetwork = ie.LoadNetwork(network, "CPU");
            InferRequest inferRequest = exeNetwork.CreateInferRequest();
            OutputsDataMap outInfo;
            outInfo = network.getOutputsInfo();
            ASSERT_EQ(outInfo.size(), 1);
            ASSERT_NE(outInfo.begin()->second, nullptr);
            inferRequest.SetBlob(network.getInputsInfo().begin()->first, src);
            inferRequest.SetBlob(outInfo.begin()->first, dst);
            inferRequest.Infer();

            TBlob<float> dst_ref(TensorDesc(Precision::FP32, SizeVector(dims_src.rbegin(), dims_src.rend()), NCHW));
            dst_ref.allocate();

            auto * srcPtr = dynamic_cast<TBlob<float>*>(src.get());
            ref_batchnorm4D(*srcPtr, (const float*) weights->buffer(), ((const float*) weights->buffer() + p.in.c), dst_ref, p);

            compare(*dst, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(smoke_CPUBatchNorn4DOnlyTest, TestsBatchNorm4D) {}

INSTANTIATE_TEST_CASE_P(
        TestBatchNorm4D, smoke_CPUBatchNorn4DOnlyTest,
        ::testing::Values(
                batchnorm4D_test_params{ "CPU",
                                         {256, 128, 32}, 1e-6}));
