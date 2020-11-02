// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_core.hpp>

#include "tests_common.hpp"
#include "single_layer_common.hpp"
#include "ir_gen_helper.hpp"

using namespace ::testing;
using namespace InferenceEngine;
using namespace single_layer_tests;

struct scaleshift_test_params {
    std::string device_name;

    struct {
        size_t w;
        size_t h;
        size_t c;
    } in;

    int broadcast;
};

template <typename data_t>
void ref_scaleshift(const TBlob<data_t> &src, const data_t *weights, const size_t weightsSize,
              TBlob<data_t> &dst, scaleshift_test_params prm) {

    size_t IW = src.getTensorDesc().getDims()[3];
    size_t IH = src.getTensorDesc().getDims()[2];
    size_t IC = src.getTensorDesc().getDims()[1];
    size_t MB = src.getTensorDesc().getDims()[0];

    const data_t *src_data = src.readOnly();
    const data_t *weights_data = weights;
    const data_t *bias_data = weights_data + IC;
    data_t *dst_data = dst.data();

    for(int mb = 0; mb < MB; mb++) {
        for(int c = 0; c < IC; c++) {
            for(int h = 0; h < IH; h++) {
                for(int w = 0; w < IW; w++) {
                    int idx = mb * IC * IH * IW
                        + c * IH * IW
                        + h * IW + w;

                    int widx = c;
                    int bidx = c;

                    dst_data[idx] = src_data[idx] * weights_data[widx] + bias_data[bidx];
                }
            }
        }
    }
}

class smoke_CPUScaleShiftOnlyTest: public TestsCommon,
                           public WithParamInterface<scaleshift_test_params> {
    std::string layers_t = R"V0G0N(
        <layer name="scaleshift" id="1" type="ScaleShift" precision="FP32">
            <data broadcast="_BROADCAST_"/>

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
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
)V0G0N";
    
    std::string edges_t = R"V0G0N(
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
)V0G0N";

    std::string getModel(scaleshift_test_params p) {
        std::string model = layers_t;

        REPLACE_WITH_NUM(model, "_IN_", 1);
        REPLACE_WITH_NUM(model, "_IW_", p.in.w);
        REPLACE_WITH_NUM(model, "_IH_", p.in.h);
        REPLACE_WITH_NUM(model, "_IC_", p.in.c);
        REPLACE_WITH_NUM(model, "_BROADCAST_", p.broadcast);

        size_t w_data_size = p.in.c * sizeof(float);
        size_t b_data_size = p.in.c * sizeof(float);
        REPLACE_WITH_NUM(model, "_S1_", w_data_size);
        REPLACE_WITH_NUM(model, "_S2_", b_data_size);

        model = IRTemplateGenerator::getIRTemplate("ScaleShift_Only", {1lu, p.in.c, p.in.h, p.in.w}, "FP32", model, edges_t);

        return model;
    }

protected:
    virtual void SetUp() {

        try {
            scaleshift_test_params p = ::testing::WithParamInterface<scaleshift_test_params>::GetParam();
            std::string model = getModel(p);

            TBlob<uint8_t> *weights = new TBlob<uint8_t>(TensorDesc(Precision::U8, { p.in.c * 2 * sizeof(float) }, C));
            weights->allocate();
            fill_data( weights->data().as<float*>(), weights->size() / sizeof(float));

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
            ref_scaleshift(*srcPtr, weights->readOnly().as<const float*>(), weights->size() / sizeof(float), dst_ref, p);

            compare(*dst, dst_ref);

        } catch (const details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(smoke_CPUScaleShiftOnlyTest, TestsScaleShift) {}

INSTANTIATE_TEST_CASE_P(
        TestScaleShift, smoke_CPUScaleShiftOnlyTest,
        ::testing::Values(
                scaleshift_test_params{ "CPU",
                                  {256, 128, 32}, 0}));

