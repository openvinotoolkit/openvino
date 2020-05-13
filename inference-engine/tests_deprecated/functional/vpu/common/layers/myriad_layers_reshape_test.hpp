// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "ie_layouts.h"
#include "myriad_layers_tests.hpp"
#include <vpu/private_plugin_config.hpp>
#include "myriad_layers_reference_functions.hpp"

using namespace InferenceEngine;

using myriadEliminateReshapeTests_smoke = myriadLayersTests_nightly;

typedef myriadLayerTestBaseWithParam<std::tuple<InferenceEngine::SizeVector, InferenceEngine::SizeVector>> myriadLayerReshape_smoke;

TEST_P(myriadLayerReshape_smoke, Reshape) {
    auto input_tensor = std::get<0>(GetParam());
    auto output_tensor = std::get<1>(GetParam());

    std::string shape = std::to_string(output_tensor[0]);
    for (size_t i = 1; i < output_tensor.size(); ++i) {
        shape += "," + std::to_string(output_tensor[i]);
    }

    std::map<std::string, std::string> params;
    params["dim"] = shape;

    _testNet.addLayer(LayerInitParams("Reshape")
             .params(params)
             .in({input_tensor})
             .out({output_tensor}),
            ref_reshape_wrap);

    _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);

    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams().useHWOpt(CheckMyriadX()).layoutPreference(vpu::LayoutPreference::ChannelMinor)));
}


typedef myriadLayersTests_nightly myriadLayerReshapeFasterRCNN_smoke;

static std::vector<InferenceEngine::SizeVector> s_reshapeInParams = {
    {{1, 4, 2, 16}},
    {{1, 2, 4, 16}},
    {{1, 4, 16, 2}},
    {{1, 16, 4, 2}},
    {{1, 8,  4,  4}},
};

static std::vector<InferenceEngine::SizeVector> s_reshapeOutParams = {
    {{1, 16, 2, 4}},
    {{1, 4, 16, 2}},
    {{1, 4, 2, 16}},
    {{1, 4, 4,  8}},
    {{1, 4, 8,  4}},
    {{1, 2, 4, 16}},
    {{1, 2, 16, 4}},
    {{1, 64, 2, 1}},
};

std::string MODEL_WITH_FLATTEN = R"V0G0N(
    <net name="MODEL_WITH_FLATTEN" version="2" batch="1">
        <layers>
            <layer id="0" name="input" precision="FP16" type="Input">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>16</dim>
                        <dim>6</dim>
                        <dim>6</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="pool5" precision="FP16" type="Pooling">
                <data exclude-pad="false" kernel-x="2" kernel-y="2" pad-x="0" pad-y="0" pool-method="max" stride="1,1,2,2" stride-x="2" stride-y="2"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>16</dim>
                        <dim>6</dim>
                        <dim>6</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>16</dim>
                        <dim>3</dim>
                        <dim>3</dim>
                    </port>
                </output>
            </layer>
            <layer id="2" name="flatten_0" precision="FP16" type="Reshape">
                <data axis="1" dim="1,144" num_axes="-1" />
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>16</dim>
                        <dim>3</dim>
                        <dim>3</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>144</dim>
                    </port>
                </output>
            </layer>
            <layer id="3" name="fc6" precision="FP16" type="FullyConnected">
                <data out-size="32"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>144</dim>
                    </port>
                </input>
                <output>
                    <port id="3">
                        <dim>1</dim>
                        <dim>32</dim>
                    </port>
                </output>
                <blobs>
                    <weights offset="0" size="9216"/>
                    <biases offset="9216" size="64"/>
                </blobs>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
            <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
            <edge from-layer="2" from-port="1" to-layer="3" to-port="0"/>
        </edges>
    </net>
)V0G0N";

std::string MODEL_WITHOUT_FLATTEN = R"V0G0N(
    <net name="MODEL_WITHOUT_FLATTEN" version="2" batch="1">
        <layers>
            <layer id="0" name="input" precision="FP16" type="Input">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>16</dim>
                        <dim>6</dim>
                        <dim>6</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="pool5" precision="FP16" type="Pooling">
                <data exclude-pad="false" kernel-x="2" kernel-y="2" pad-x="0" pad-y="0" pool-method="max" stride="1,1,2,2" stride-x="2" stride-y="2"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>16</dim>
                        <dim>6</dim>
                        <dim>6</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>16</dim>
                        <dim>3</dim>
                        <dim>3</dim>
                    </port>
                </output>
            </layer>
            <layer id="2" name="fc6" precision="FP16" type="FullyConnected">
                <data out-size="32"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>16</dim>
                        <dim>3</dim>
                        <dim>3</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>32</dim>
                    </port>
                </output>
                <blobs>
                    <weights offset="0" size="9216"/>
                    <biases offset="9216" size="64"/>
                </blobs>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
            <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
        </edges>
    </net>
)V0G0N";


typedef myriadLayerTestBaseWithParam<std::string> myriadLayersTestsReshapeBeforeFC_smoke;

TEST_P(myriadLayersTestsReshapeBeforeFC_smoke, OptimizeReshapeIfItIsPlacedBeforeFC) {
    std::string HWConfigValue = GetParam();
    if (!CheckMyriadX() && HWConfigValue == CONFIG_VALUE(YES)) {
        std::cout << "Disable for non-MyriadX devices" << std::endl;
        return;
    }

    std::string outputName = "fc6";
    StatusCode st = InferenceEngine::OK;
    InferenceEngine::ResponseDesc resp;
    TBlob<uint8_t>::Ptr weights(GenWeights(9280 / sizeof(ie_fp16)));

    Core ie;
    auto network = ie.ReadNetwork(MODEL_WITH_FLATTEN, weights);

    auto inputsInfo = network.getInputsInfo();
    inputsInfo["input"]->setPrecision(Precision::FP16);

    auto outputsInfo = network.getOutputsInfo();
    outputsInfo[outputName]->setPrecision(Precision::FP16);

    InferenceEngine::IExecutableNetwork::Ptr exeNetwork;
    ASSERT_NO_THROW(st = _vpuPluginPtr->LoadNetwork(exeNetwork, network,
                                                      { {VPU_CONFIG_KEY(PERF_REPORT_MODE), VPU_CONFIG_VALUE(PER_STAGE)},
                                                        {VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), HWConfigValue},
                                                        {CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES) }}, &resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    InferenceEngine::IInferRequest::Ptr inferRequest;
    ASSERT_NO_THROW(st = exeNetwork->CreateInferRequest(inferRequest, &resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    Blob::Ptr input;
    ASSERT_NO_THROW(st = inferRequest->GetBlob("input", input, &resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = inferRequest->Infer(&_resp));
    ASSERT_EQ(StatusCode::OK, st) << resp.msg;

    std::map<std::string, InferenceEngineProfileInfo> perfMap;
    ASSERT_NO_THROW(st = inferRequest->GetPerformanceCounts(perfMap, &resp));
    ASSERT_EQ(StatusCode::OK, st) << resp.msg;

    auto layerInfo = perfMap["flatten_0"];
    EXPECT_EQ(InferenceEngineProfileInfo::NOT_RUN, layerInfo.status);
}

class myriadLayersTestsReshapeFasterRCNN_smoke: public ConvolutionTest<>{
};

// FIXME: rewrite the test (it doesn't use Convolution) avoid HWC layout for 3D tensor in reference code
TEST_P(myriadLayersTestsReshapeFasterRCNN_smoke, DISABLED_Convolution) {
    std::map<std::string, std::string> permute_params = {
              {"order", "0,2,3,1"}
    };
    std::map<std::string, std::string> reshape_params = {
                {"axis", "0"}
              , {"dim", "0,-1,2"}
              , {"num_axes", "-1"}
    };
    InferenceEngine::SizeVector perm_out = {1, 14, 14, 24};
    _testNet.addLayer(LayerInitParams("Permute")
             .params(permute_params)
             .in({_output_tensor})
             .out({perm_out}),
             ref_permute_wrap);

    _testNet.addLayer(LayerInitParams("Reshape")
             .params(reshape_params)
             .in({perm_out})
             .out({{1, 2352, 2}}),
             ref_reshape_wrap);

    float maxerr = 0;
    maxerr = 0.00066 * (IC) * kernel.x * kernel.y;
    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams()));
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), maxerr);
}

static const std::vector<InferenceEngine::SizeVector> s_convTensor = {
    {{1, 512, 14, 14}} 
};
