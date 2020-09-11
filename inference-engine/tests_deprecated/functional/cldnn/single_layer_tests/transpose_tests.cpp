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


struct transpose_test_params {
    std::string device_name;
    InferenceEngine::SizeVector in_shape;
    InferenceEngine::SizeVector out_shape;
    bool secondInput;
};


class TransposeTest : public TestsCommon, public WithParamInterface<transpose_test_params> {
    std::string model_t = R"V0G0N(
<net Name="Transpose_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="input" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    _IN_
                </port>
            </output>
        </layer>
        _SND_INP_
        <layer name="output" type="Transpose" precision="_LKP_" id="2">
            <input>
                <port id="0">
                    _IN_
                </port>
                _SND_INPUT_SHAPE_
            </input>
            <output>
                <port id="2">
                    _OUT_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        _SND_EDGE_
    </edges>
</net>
)V0G0N";


    std::string getModel(transpose_test_params p) {
        std::string model = model_t;
        std::string in_shape, out_shape, snd_layer, snd_shape, snd_edge;
        snd_layer = snd_shape = snd_edge = "";

        for (size_t i = 0; i < p.in_shape.size(); i++) {
            in_shape += "<dim>";
            in_shape += std::to_string(p.in_shape[i]);
            in_shape += "</dim>\n";
        }

        for (size_t i = 0; i < p.out_shape.size(); i++) {
            out_shape += "<dim>";
            out_shape += std::to_string(p.out_shape[i]);
            out_shape += "</dim>\n";
        }

        if (p.secondInput)
        {
            snd_shape += "<port id=\"1\">\n";
            snd_shape += std::to_string(p.in_shape.size());
            snd_shape += "\n</port>\n";

            snd_layer += "<layer name=\"order\" type=\"Input\" precision=\"I32\" id=\"1\">\n";
            snd_layer += "<output>\n";
            snd_layer += snd_shape;
            snd_layer += "</output>\n";
            snd_layer += "</layer>\n";

            snd_edge += "<edge from-layer=\"1\" from-port=\"1\" to-layer=\"2\" to-port=\"1\"/>";
        }

        REPLACE_WITH_STR(model, "_IN_", in_shape);
        REPLACE_WITH_STR(model, "_OUT_", out_shape);
        REPLACE_WITH_STR(model, "_SND_INP_", snd_layer);
        REPLACE_WITH_STR(model, "_SND_INPUT_SHAPE_", snd_shape);
        REPLACE_WITH_STR(model, "_SND_EDGE_", snd_edge);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try
        {
            transpose_test_params p = ::testing::WithParamInterface<transpose_test_params>::GetParam();
            std::string model = getModel(p);

            Core ie;
            CNNNetwork net = ie.ReadNetwork(model, Blob::CPtr());
            ExecutableNetwork executable_network = ie.LoadNetwork(net, p.device_name);
            InferRequest inferRequest = executable_network.CreateInferRequest();

            Blob::Ptr src = make_shared_blob<float>({Precision::FP32, p.in_shape,
                TensorDesc::getLayoutByDims(p.in_shape)});
            src->allocate();

            auto* srcPtr = dynamic_cast<TBlob<float>*>(src.get());

            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            inferRequest.SetBlob("input", src);

            inferRequest.Infer();

            OutputsDataMap outputInfo(net.getOutputsInfo());
            Blob::Ptr outputBlob = inferRequest.GetBlob(outputInfo.begin()->first);
            auto outputDims = outputBlob->getTensorDesc().getDims();

            compare(outputDims, p.out_shape);

        }
        catch (const InferenceEngine::details::InferenceEngineException &e)
        {
            FAIL() << e.what();
        }
    }
};

TEST_P(TransposeTest, smoke_GPU_TestsTranspose) {}

INSTANTIATE_TEST_CASE_P(
    smoke_TestsTranspose, TransposeTest,
    ::testing::Values(
        transpose_test_params{ "GPU", { 2, 3, 4 }, { 4, 3, 2 }, false },
        transpose_test_params{ "GPU", { 2, 3, 4, 5 }, { 5, 4, 3, 2 }, false },
        transpose_test_params{ "GPU", { 2, 3, 4 }, { 4, 3, 2 }, true },
        transpose_test_params{ "GPU", { 2, 3, 4 }, { 4, 2, 3 }, true },
        transpose_test_params{ "GPU", { 2, 3, 4, 5 }, { 2, 3, 5, 4 }, true }
));
