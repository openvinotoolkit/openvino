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


struct activation_base_params {
    struct {
        size_t w;
        size_t h;
        size_t c;
    } in;

    float n_clope;
};

struct activation_test_params : activation_base_params {
    std::string device_name;
    std::string activationType;

    activation_test_params(std::string name, activation_base_params params, std::string activationType) :
            activation_base_params(params), device_name(name), activationType(activationType) {}

};

template <typename data_t>
void ref_activation(const data_t *src_data, data_t *dst_data, activation_test_params prm)
{
    size_t IW = prm.in.w;
    size_t IH = prm.in.h;
    size_t IC = prm.in.c;

    for (uint32_t c = 0; c < IC; c++) {
        for (uint32_t h = 0; h < IH; h++) {
            for (uint32_t w = 0; w < IW; w++) {
                uint32_t oidx = c * IH * IW
                                + h * IW + w;

                if (prm.activationType == "exp")
                    dst_data[oidx] = exp(src_data[oidx]);
                else if (prm.activationType == "not")
                    dst_data[oidx] = !(src_data[oidx]);
                else if (prm.activationType == "sin")
                    dst_data[oidx] = sin(src_data[oidx]);
                else if (prm.activationType == "sinh")
                    dst_data[oidx] = sinh(src_data[oidx]);
                else if (prm.activationType == "cos")
                    dst_data[oidx] = cos(src_data[oidx]);
                else if (prm.activationType == "cosh")
                    dst_data[oidx] = cosh(src_data[oidx]);
                else
                    dst_data[oidx] = src_data[oidx] >= 0.0 ?
                                     src_data[oidx] :
                                     src_data[oidx] * prm.n_clope;
            }
        }
    }
}

class ActivationTest: public TestsCommon,
                    public WithParamInterface<activation_test_params> {
    std::string layers_t = R"V0G0N(
        <layer name="_ACTIVATION_TYPE_" id="1" type="_ACTIVATION_TYPE_" precision="FP32">
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

    std::string getModel(activation_test_params p) {
        std::string model = layers_t;

        if (p.activationType == "exp")
            REPLACE_WITH_STR(model, "_ACTIVATION_TYPE_", "Exp");
        else if (p.activationType == "not")
            REPLACE_WITH_STR(model, "_ACTIVATION_TYPE_", "Not");
        else if (p.activationType == "sin")
            REPLACE_WITH_STR(model, "_ACTIVATION_TYPE_", "Sin");
        else if (p.activationType == "sinh")
            REPLACE_WITH_STR(model, "_ACTIVATION_TYPE_", "Sinh");
        else if (p.activationType == "cos")
            REPLACE_WITH_STR(model, "_ACTIVATION_TYPE_", "Cos");
        else if (p.activationType == "cosh")
            REPLACE_WITH_STR(model, "_ACTIVATION_TYPE_", "Cosh");
        else
            REPLACE_WITH_STR(model, "_ACTIVATION_TYPE_", "ReLU"); // Default value

        REPLACE_WITH_NUM(model, "_IN_", 1);
        REPLACE_WITH_NUM(model, "_IW_", p.in.w);
        REPLACE_WITH_NUM(model, "_IH_", p.in.h);
        REPLACE_WITH_NUM(model, "_IC_", p.in.c);

        model = IRTemplateGenerator::getIRTemplate(p.activationType + "_Only", {1lu, p.in.c, p.in.h, p.in.w}, "FP32", model, edges_t);

        return model;
    }

protected:
    virtual void SetUp() {

        try {
            activation_test_params p = ::testing::WithParamInterface<activation_test_params>::GetParam();
            std::string model = getModel(p);

            Core ie;
            CNNNetwork net = ie.ReadNetwork(model, Blob::CPtr());
            InputsDataMap in_info_map = net.getInputsInfo();
            OutputsDataMap out_info_map = net.getOutputsInfo();

            ExecutableNetwork executable_network = ie.LoadNetwork(net, p.device_name);
            InferRequest inferRequest = executable_network.CreateInferRequest();

            SizeVector dims_src = {1,
                                   p.in.c,
                                   p.in.h,
                                   p.in.w};

            Blob::Ptr inputBlob = inferRequest.GetBlob(in_info_map.begin()->first);
            float* src = inputBlob->buffer().as<float*>();
            fill_data(src, inputBlob->size());

            SizeVector dims_dst = dims_src;
            Blob::Ptr outputBlob = inferRequest.GetBlob(out_info_map.begin()->first);

            TBlob<float> dst_ref({ Precision::FP32, dims_dst, Layout::NCHW });
            dst_ref.allocate();

            inferRequest.Infer();

            ref_activation<float>(src, dst_ref.data(), p);

            const float* res = outputBlob->buffer().as<float*>();
            const float* ref = dst_ref.data();
            compare(res, ref, outputBlob->size());

        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

#define case_1 activation_base_params({{228, 228, 3}, 0.0})

TEST_P(ActivationTest, TestsActivationFunctions) {}

std::string  getTestCaseName(testing::TestParamInfo<activation_test_params> obj) {
    return  obj.param.device_name +
            "_w" + std::to_string(obj.param.in.w) +
            "_h" + std::to_string(obj.param.in.h) +
            "_c" + std::to_string(obj.param.in.c) +
            "_" + obj.param.activationType;
}
