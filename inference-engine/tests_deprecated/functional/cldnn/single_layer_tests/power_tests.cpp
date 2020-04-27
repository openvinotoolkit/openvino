// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_core.hpp>
#include <cmath>

#include "tests_common.hpp"
#include "single_layer_common.hpp"



using namespace ::testing;
using namespace InferenceEngine;
using namespace std;


struct test_params {
    std::string device_name;
    std::string inPrecision;
    InferenceEngine::SizeVector in_out_shape;
    float power;
    float scale;
    float shift;
    std::vector<float> reference;
};

template<typename data_t>
void ref_power(InferenceEngine::TBlob<float> &dst, test_params const& prm) {
    data_t *dst_data = dst.data().as<data_t*>();

    for (size_t i = 0; i < prm.in_out_shape.size(); ++i) {
        dst_data[i] = std::pow(prm.shift + i * prm.scale, prm.power);
    }
}

class PowerTests : public TestsCommon, public WithParamInterface<test_params> {
    std::string model_t = R"V0G0N(
<net Name="Power_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="input" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    _IN_OUT_
                </port>
            </output>
        </layer>
        <layer name="output" id="2" type="Power" precision="FP32">
            <data power="_POWER_" scale="_SCALE_" shift="_SHIFT_"/>
            <input>
                <port id="1">
                    _IN_OUT_
                </port>
           </input>
            <output>
                <port id="2">
                    _IN_OUT_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(test_params p) {
        std::string model = model_t;
        std::string in_out_shape;

        for (size_t i = 0; i < p.in_out_shape.size(); i++) {
            in_out_shape += "<dim>";
            in_out_shape += std::to_string(p.in_out_shape[i]) + "</dim>\n";
        }
        REPLACE_WITH_STR(model, "_IN_OUT_", in_out_shape);
        REPLACE_WITH_NUM(model, "_POWER_", p.power);
        REPLACE_WITH_NUM(model, "_SCALE_", p.scale);
        REPLACE_WITH_NUM(model, "_SHIFT_", p.shift);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            test_params p = ::testing::WithParamInterface<test_params>::GetParam();
            std::string model = getModel(p);

            Core ie;
            CNNNetwork net = ie.ReadNetwork(model, Blob::CPtr());
            ExecutableNetwork executable_network = ie.LoadNetwork(net, p.device_name);
            InferRequest inferRequest = executable_network.CreateInferRequest();

            // Input Data
            InputsDataMap inputInfo(net.getInputsInfo());
            Blob::Ptr inputBlob = inferRequest.GetBlob(inputInfo.begin()->first);
            float* inputData = inputBlob->buffer().as<float*>();
            fill_data_dbgval(inputData, inputBlob->size());

            inferRequest.Infer();

            // Output Data
            OutputsDataMap outputInfo(net.getOutputsInfo());
            Blob::Ptr outputBlob = inferRequest.GetBlob(outputInfo.begin()->first);

            // Output Reference
            InferenceEngine::TBlob<float> dst_ref(outputBlob->getTensorDesc());
            dst_ref.allocate();
            ref_power<float>(dst_ref, p);

            //  Check results
            if (memcmp(dst_ref.data(), &p.reference[0], p.reference.size() * sizeof(float)) != 0)
                FAIL() << "Wrong result with compare TF reference!";

            compare(*outputBlob, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(PowerTests, smoke_GPU_TestsPower) {}

std::vector<float> power_ref_0 = { 0.f, 1.f, 4.f, 9.f };
std::vector<float> power_ref_1 = { 0.f, 4.f, 16.f, 36.f };

INSTANTIATE_TEST_CASE_P(
        smoke_TestsPower, PowerTests,
        ::testing::Values(
            test_params{ "GPU", "FP32", { 1, 1, 2, 2 }, 2.f, 1.f, 0.f, power_ref_0 },
            test_params{ "GPU", "FP32", { 1, 1, 2, 2 }, 2.f, 2.f, 0.f, power_ref_1 }
        ));
