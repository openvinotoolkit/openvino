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


struct norm_base_params {
    struct {
        size_t w;
        size_t h;
        size_t c;
    } in;

    size_t local_size;
    float alpha;
    float beta;
    size_t k;

};

struct norm_test_params : norm_base_params {
    std::string device_name;

    norm_test_params(std::string name, norm_base_params params) :
            norm_base_params(params), device_name(name) {}
};


template <typename data_t>
void ref_norm(const TBlob<data_t> &src, TBlob<data_t> &dst, norm_test_params prm)
{
    size_t IW = prm.in.w;
    size_t IH = prm.in.h;
    size_t IC = prm.in.c;

    const data_t *src_data = src.readOnly();
    data_t *dst_data = dst.data();

        for (uint32_t c = 0; c < IC; c++) {
            for (uint32_t h = 0; h < IH; h++) {
                for (uint32_t w = 0; w < IW; w++) {
                    uint32_t oidx = c * IH * IW
                                    + h * IW + w;

                    uint32_t sz = prm.local_size;
                    int32_t c_start = c - sz / 2;
                    int32_t c_end = c_start + sz;
                    if (c_start < 0) c_start = 0;
                    if (c_end > (int32_t)IC) c_end = IC;
                    data_t sum = 0.0;
                    for (int32_t c1 = c_start; c1 < c_end; c1++) {
                        uint32_t idx = c1 * IH * IW + h * IW + w;
                        data_t s = src_data[idx];

                        sum += s * s;
                    }

                    data_t norm_coef = powf(1. + prm.alpha * sum / sz, -prm.beta);
                    dst_data[oidx] = norm_coef * src_data[oidx];
                }
            }
        }
}

class smoke_NormOnlyTest: public TestsCommon,
                    public WithParamInterface<norm_test_params> {
    std::string layers_t = R"V0G0N(
        <layer name="norm" id="1" type="LRN" precision="FP32">
            <lrn local_size="_LS_" alpha="_A__" beta="_B__" k="_K__" region="ACROSS" />

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
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
)V0G0N";
    
    std::string edges_t = R"V0G0N(
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
)V0G0N";

    std::string getModel(norm_test_params p) {
        std::string model = layers_t;

        REPLACE_WITH_NUM(model, "_IN_", 1);
        REPLACE_WITH_NUM(model, "_IW_", p.in.w);
        REPLACE_WITH_NUM(model, "_IH_", p.in.h);
        REPLACE_WITH_NUM(model, "_IC_", p.in.c);

        REPLACE_WITH_NUM(model, "_LS_", p.local_size);
        REPLACE_WITH_NUM(model, "_A__", p.alpha);
        REPLACE_WITH_NUM(model, "_B__", p.beta);
        REPLACE_WITH_NUM(model, "_K__", p.k);

        model = IRTemplateGenerator::getIRTemplate("FullyConnected_Only", {1lu, p.in.c, p.in.h, p.in.w}, "FP32", model, edges_t);

        return model;
    }

protected:
    virtual void SetUp() {

        try {
            norm_test_params p = ::testing::WithParamInterface<norm_test_params>::GetParam();
            std::string model = getModel(p);

            Core ie;
            CNNNetwork network = ie.ReadNetwork(model, Blob::CPtr());

            SizeVector dims_src = {1,
                                   p.in.c,
                                   p.in.h,
                                   p.in.w};
            Blob::Ptr src = make_shared_blob<float>(TensorDesc({ Precision::FP32, dims_src, Layout::NCHW }));
            src->allocate();
            fill_data(src->buffer().as<float *>(), src->size());

            SizeVector dims_dst = dims_src;
            Blob::Ptr dst = make_shared_blob<float>(TensorDesc({ Precision::FP32, dims_dst, Layout::NCHW }));
            dst->allocate();

            TBlob<float> dst_ref({Precision::FP32, dims_dst, Layout::NCHW});
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
            ref_norm(*srcPtr, dst_ref, p);
            compare(*dst, dst_ref);

        } catch (const details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

#define case_1 norm_base_params({{228, 228, 3}, 5, 0.0001f, 0.75f, 1})

TEST_P(smoke_NormOnlyTest, TestsNorm) {}

std::string  getTestCaseName(testing::TestParamInfo<norm_test_params> obj) {
    return  obj.param.device_name +
        "_w" + std::to_string(obj.param.in.w) +
        "_h" + std::to_string(obj.param.in.h) +
        "_c" + std::to_string(obj.param.in.c);
}

norm_test_params norm_only_test_cases[] = {
		norm_test_params("CPU", case_1),
};

INSTANTIATE_TEST_CASE_P(
        TestsNorm, smoke_NormOnlyTest, ::testing::ValuesIn(norm_only_test_cases), getTestCaseName);
