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

struct fc_base_params {
    struct {
        size_t w;
        size_t h;
        size_t c;
    } in;

    size_t out_c;
};

struct fc_test_params : fc_base_params {
    std::string device_name;

    fc_test_params(std::string name, fc_base_params params) :
            fc_base_params(params), device_name(name) {}
};

template <typename data_t>
void ref_innerproduct(const TBlob<data_t> &src, const data_t *weights, const size_t weightsSize,
                      TBlob<data_t> &dst, fc_test_params prm)
{
    size_t IW = src.getTensorDesc().getDims()[3];
    size_t IH = src.getTensorDesc().getDims()[2];
    size_t IC = src.getTensorDesc().getDims()[1];

    size_t OC = prm.out_c;

    const data_t *src_data = src.readOnly();
    const data_t *weights_data = weights;
    const data_t *bias_data = weights_data + IW*IH*IC*OC;
    data_t *dst_data = dst.data();

    IE_ASSERT( IW*IH*IC*OC + OC == weightsSize);
    IE_ASSERT( OC == dst.getTensorDesc().getDims()[1]);

    for (size_t oc = 0; oc < OC; oc++) {
        dst_data[oc] = bias_data[oc];
        for (size_t ic = 0; ic < IC; ic++) {
            for (size_t kh = 0; kh < IH; kh++) {
                for (size_t  kw = 0; kw < IW; kw++) {
                    size_t iidx = ic * IH * IW + kh * IW + kw;
                    size_t widx = oc * IC * IH * IW
                                    + ic * IH * IW + kh * IW + kw;

                    dst_data[oc] += src_data[iidx] * weights_data[widx];
                }
            }
        }
    }
}

class smoke_FullyConnectedOnlyTest: public TestsCommon,
                              public WithParamInterface<fc_test_params> {

    std::string layers_t = R"V0G0N(
        <layer name="FullyConnected" id="1" type="InnerProduct" precision="FP32">
            <fc out-size="_OC_" />
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
                </port>
            </output>
        </layer>
)V0G0N";

    std::string edges_t = R"V0G0N(
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
)V0G0N";

    std::string getModel(fc_test_params p) {
        std::string model = layers_t;

        REPLACE_WITH_NUM(model, "_IN_", 1);
        REPLACE_WITH_NUM(model, "_IW_", p.in.w);
        REPLACE_WITH_NUM(model, "_IH_", p.in.h);
        REPLACE_WITH_NUM(model, "_IC_", p.in.c);
        REPLACE_WITH_NUM(model, "_OC_", p.out_c);

        size_t w_data_size = (p.in.w * p.in.h * p.in.c * p.out_c )* sizeof(float);
        size_t b_data_size = p.out_c * sizeof(float);
        REPLACE_WITH_NUM(model, "_S1_", w_data_size);
        REPLACE_WITH_NUM(model, "_S2_", b_data_size);

        model = IRTemplateGenerator::getIRTemplate("FullyConnected_Only", {1lu, p.in.c, p.in.h, p.in.w}, "FP32", model, edges_t);

        return model;
    }

protected:
    virtual void SetUp() {

        try {
            fc_test_params p = ::testing::WithParamInterface<fc_test_params>::GetParam();
            std::string model = getModel(p);

            TBlob<uint8_t> *weights = new TBlob<uint8_t>({Precision::U8, {(p.in.w * p.in.h * p.in.c * p.out_c + p.out_c) * sizeof(float)}, Layout::C});
            weights->allocate();
            fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
            TBlob<uint8_t>::Ptr weights_ptr = TBlob<uint8_t>::Ptr(weights);
 
            Core ie;
            CNNNetwork network = ie.ReadNetwork(model, weights_ptr);

            SizeVector dims_src = {1,
                                   p.in.c,
                                   p.in.h,
                                   p.in.w};
            Blob::Ptr src = make_shared_blob<float>(TensorDesc({ Precision::FP32, dims_src, Layout::NCHW }));
            src->allocate();
            fill_data(src->buffer().as<float *>(), src->size());

            SizeVector dims_dst = {1, p.out_c};
            Blob::Ptr dst = make_shared_blob<float>(TensorDesc({ Precision::FP32, dims_dst, Layout::NC }));
            dst->allocate();

            TBlob<float> dst_ref({Precision::FP32, dims_dst, Layout::NC});
            dst_ref.allocate();

            ExecutableNetwork exeNetwork = ie.LoadNetwork(network, p.device_name);
            InferRequest inferRequest = exeNetwork.CreateInferRequest();
            OutputsDataMap outInfo;
            outInfo = network.getOutputsInfo();
            ASSERT_EQ(outInfo.size(), 1);
            ASSERT_NE(outInfo.begin()->second, nullptr);
            inferRequest.SetBlob(network.getInputsInfo().begin()->first, src);
            inferRequest.SetBlob(outInfo.begin()->first, dst);
            inferRequest.Infer();

            auto * srcPtr = dynamic_cast<TBlob<float>*>(src.get());
            ref_innerproduct(*srcPtr, weights->readOnly().as<const float *>(), weights->size() / sizeof(float), dst_ref, p);
            compare(*dst, dst_ref, 0.9f);

        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

#define case_1 fc_base_params({{227, 227, 3}, 96})
#define case_2 fc_base_params({{227, 227, 4}, 8})

TEST_P(smoke_FullyConnectedOnlyTest, TestsFullyConnected) {}

std::string  getTestCaseName(testing::TestParamInfo<fc_test_params> obj) {
    return  obj.param.device_name +
        "_w" + std::to_string(obj.param.in.w) +
        "_h" + std::to_string(obj.param.in.h) +
        "_c" + std::to_string(obj.param.in.c) +
        "_outc" + std::to_string(obj.param.out_c);
}

fc_test_params fc_only_test_cases[] = {
		fc_test_params("CPU", case_1),
		fc_test_params("CPU", case_2),
};

INSTANTIATE_TEST_CASE_P(
        TestsFullyConnected, smoke_FullyConnectedOnlyTest, ::testing::ValuesIn(fc_only_test_cases), getTestCaseName);
