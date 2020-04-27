// Copyright (C) 2020 Intel Corporation
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


struct convert_like_test_params {
    std::string device_name;
    std::string inPrecision;
    std::string likePrecision;
    InferenceEngine::SizeVector in_out_shape;
    InferenceEngine::SizeVector like_shape;
};



class ConvertLikeTest : public TestsCommon, public WithParamInterface<convert_like_test_params> {
    std::string model_t = R"V0G0N(
<net Name="ConvertLike_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="input" type="Input" precision="_INP_" id="0">
            <output>
                <port id="0">
                    _IN_OUT_
                </port>
            </output>
        </layer>
        <layer name="like" type="Input" precision="_LKP_" id="1">
            <output>
                <port id="0">
                    _LIKE_
                </port>
            </output>
        </layer>
        <layer name="output" type="ConvertLike" precision="_LKP_" id="2">
            <input>
                <port id="0">
                    _IN_OUT_
                </port>
                <port id="1">
                    _LIKE_
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
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
    </edges>
</net>
)V0G0N";


    std::string getModel(convert_like_test_params p) {
        std::string model = model_t;
        std::string in_out_shape, like_shape;

        for (size_t i = 0; i < p.in_out_shape.size(); i++) {
            in_out_shape += "<dim>";
            in_out_shape += std::to_string(p.in_out_shape[i]);
            in_out_shape += "</dim>\n";
        }

        for (size_t i = 0; i < p.like_shape.size(); i++) {
            like_shape += "<dim>";
            like_shape += std::to_string(p.like_shape[i]);
            like_shape += "</dim>\n";
        }

        REPLACE_WITH_STR(model, "_INP_", p.inPrecision);
        REPLACE_WITH_STR(model, "_LKP_", p.likePrecision);
        REPLACE_WITH_STR(model, "_IN_OUT_", in_out_shape);
        REPLACE_WITH_STR(model, "_LIKE_", like_shape);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try
        {
            convert_like_test_params p = ::testing::WithParamInterface<convert_like_test_params>::GetParam();
            std::string model = getModel(p);

            Core ie;
            CNNNetwork net = ie.ReadNetwork(model, InferenceEngine::Blob::CPtr());

            ExecutableNetwork executable_network = ie.LoadNetwork(net, p.device_name);
            InferRequest inferRequest = executable_network.CreateInferRequest();

            // Input Data
            InputsDataMap inputInfo(net.getInputsInfo());
            Blob::Ptr input1 = inferRequest.GetBlob("input");
            input1->allocate();
            Blob::Ptr input2 = inferRequest.GetBlob("like");
            input2->allocate();

            inferRequest.Infer();

            // Output Data
            OutputsDataMap outputInfo(net.getOutputsInfo());
            Blob::Ptr outputBlob = inferRequest.GetBlob(outputInfo.begin()->first);
            auto outputPrecision = outputBlob->getTensorDesc().getPrecision();
            auto likePrecision = input2->getTensorDesc().getPrecision();

            if (outputPrecision != likePrecision)
            {
                FAIL() << "Different output and like precision!";
            }

        }
        catch (const InferenceEngine::details::InferenceEngineException &e)
        {
            FAIL() << e.what();
        }
    }
};

TEST_P(ConvertLikeTest, smoke_GPU_TestsConvertLike) {}

INSTANTIATE_TEST_CASE_P(
    smoke_TestsConvertLike, ConvertLikeTest,
    ::testing::Values(
        convert_like_test_params{ "GPU", "FP32", "I32", { 3, 5 }, { 2 } },
        convert_like_test_params{ "GPU", "FP32", "I32", { 10, 10, 10, 5 }, { 2 } },
        convert_like_test_params{ "GPU", "FP32", "I32", { 3, 5 }, { 2, 4, 5 } },
        convert_like_test_params{ "GPU", "FP32", "FP16", { 3, 5 }, { 2 } },
        convert_like_test_params{ "GPU", "I32", "FP16", { 3, 5 }, { 2 } },
        convert_like_test_params{ "GPU", "I32", "FP32", { 3, 5 }, { 2 } }
));
