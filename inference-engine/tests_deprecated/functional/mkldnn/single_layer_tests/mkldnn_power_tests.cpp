// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_core.hpp>
#include <cmath>

#include "tests_common.hpp"
#include "single_layer_common.hpp"
#include "ir_gen_helper.hpp"

using namespace ::testing;
using namespace InferenceEngine;
using namespace single_layer_tests;

struct power_test_params {
    std::string device_name;

    struct {
        size_t w;
        size_t h;
        size_t c;
    } in;

    float power;
    float scale;
    float shift;
};

template <typename data_t>
void ref_power(const TBlob<data_t> &src, TBlob<data_t> &dst, power_test_params prm) {

    data_t *dst_data = dst.data();
    const data_t *src_data = src.readOnly();

    const double scale = prm.scale;
    const double power = prm.power;
    const double shift = prm.shift;

    for(int i = 0; i < src.size(); i++) {
        dst_data[i] = (float)std::pow(shift + src_data[i] * scale, power);
    }
}

class smoke_CPUPowerOnlyTest: public TestsCommon,
                           public WithParamInterface<power_test_params> {
    std::string layers_t = R"V0G0N(
        <layer name="power" id="1" type="Power" precision="FP32">
            <power_data power="_POWER_" scale="_SCALE_" shift="_SHIFT_"/>
            <input>
                <port id="0">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </input>
            <output>
                <port id="0">
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

    std::string getModel(power_test_params p) {
        std::string model = layers_t;

        REPLACE_WITH_NUM(model, "_IN_", 1);
        REPLACE_WITH_NUM(model, "_IW_", p.in.w);
        REPLACE_WITH_NUM(model, "_IH_", p.in.h);
        REPLACE_WITH_NUM(model, "_IC_", p.in.c);
        REPLACE_WITH_NUM(model, "_POWER_", p.power);
        REPLACE_WITH_NUM(model, "_SCALE_", p.scale);
        REPLACE_WITH_NUM(model, "_SHIFT_", p.shift);

        model = IRTemplateGenerator::getIRTemplate("Power_Only", {1lu, p.in.c, p.in.h, p.in.w}, "FP32", model, edges_t);

        return model;
    }

protected:
    virtual void SetUp() {

        try {
            power_test_params p = ::testing::WithParamInterface<power_test_params>::GetParam();
            std::string model = getModel(p);

            Core ie;
            CNNNetwork network = ie.ReadNetwork(model, Blob::CPtr());

            SizeVector dims_src = {p.in.w,
                                   p.in.h,
                                   p.in.c,
                                   1};

            Blob::Ptr src = make_shared_blob<float>(TensorDesc(Precision::FP32, SizeVector(dims_src.rbegin(), dims_src.rend()), NCHW));
            src->allocate();
            fill_data(src->buffer().as<float *>(), src->size());

            SizeVector dims_dst = dims_src;

            Blob::Ptr dst = make_shared_blob<float>(TensorDesc(Precision::FP32, SizeVector(dims_dst.rbegin(), dims_dst.rend()), NCHW));
            dst->allocate();

            TBlob<float> dst_ref(TensorDesc(Precision::FP32, SizeVector(dims_dst.rbegin(), dims_dst.rend()), NCHW));
            dst_ref.allocate();

            auto * srcPtr = dynamic_cast<TBlob<float>*>(src.get());
            ref_power(*srcPtr, dst_ref, p);

            ExecutableNetwork exeNetwork = ie.LoadNetwork(network, "CPU");
            InferRequest inferRequest = exeNetwork.CreateInferRequest();
            OutputsDataMap outInfo;
            outInfo = network.getOutputsInfo();
            ASSERT_EQ(outInfo.size(), 1);
            ASSERT_NE(outInfo.begin()->second, nullptr);
            inferRequest.SetBlob(network.getInputsInfo().begin()->first, src);
            inferRequest.SetBlob(outInfo.begin()->first, dst);
            inferRequest.Infer();

            compare(*dst, dst_ref);

        } catch (const details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(smoke_CPUPowerOnlyTest, TestsPower) {}

INSTANTIATE_TEST_CASE_P(
        TestPower, smoke_CPUPowerOnlyTest,
        ::testing::Values(
                power_test_params{ "CPU",
                    {13, 13, 3}, 1, 2, 0.5f },
                power_test_params{ "CPU",
                    {23, 23, 1}, 3, 8, 2 },
                power_test_params{ "CPU",
                    {23, 23, 8}, 8, 2, 1 }));

/*** TBD ***/


