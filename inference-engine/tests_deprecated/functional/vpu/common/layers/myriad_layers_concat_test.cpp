// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_concat_test.hpp"

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsConcat_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_concatCores),
        ::testing::ValuesIn(s_axis),
        ::testing::ValuesIn(s_concatInputs),
        ::testing::ValuesIn(s_dimension),
        ::testing::ValuesIn(s_batch)),
                        getTestCaseName
);

TEST_F(myriadLayersTestsConcat_smoke, ConcatAfterNormalize) {
    const std::string model = R"V0G0N(
        <Net name="ConcatAfterNormalize" version="2" batch="1">
            <layers>
                <layer name="input" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>128</dim>
                            <dim>128</dim>
                        </port>
                    </output>
                </layer>
                <layer name="normalize1" type="Normalize" precision="FP16" id="2">
                    <data across_spatial="0" channel_shared="1" eps="9.99999993922529e-09"/>
                    <input>
                        <port id="2">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>128</dim>
                            <dim>128</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>128</dim>
                            <dim>128</dim>
                        </port>
                    </output>
                    <weights offset="0" size="2"/>
                </layer>
                <layer name="normalize2" type="Normalize" precision="FP16" id="3">
                    <data across_spatial="0" channel_shared="1" eps="9.99999993922529e-09"/>
                    <input>
                        <port id="4">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>128</dim>
                            <dim>128</dim>
                        </port>
                    </input>
                    <output>
                        <port id="5">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>128</dim>
                            <dim>128</dim>
                        </port>
                    </output>
                    <weights offset="2" size="2"/>
                </layer>
                <layer name="copy1" type="Copy" precision="FP16" id="4">
                    <input>
                        <port id="6">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>128</dim>
                            <dim>128</dim>
                        </port>
                    </input>
                    <output>
                        <port id="7">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>128</dim>
                            <dim>128</dim>
                        </port>
                    </output>
                </layer>
                <layer name="copy2" type="Copy" precision="FP16" id="5">
                    <input>
                        <port id="8">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>128</dim>
                            <dim>128</dim>
                        </port>
                    </input>
                    <output>
                        <port id="9">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>128</dim>
                            <dim>128</dim>
                        </port>
                    </output>
                </layer>
                <layer name="concat" type="Concat" precision="FP16" id="6">
                    <concat_data axis="1"/>
                    <input>
                        <port id="10">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>128</dim>
                            <dim>128</dim>
                        </port>
                        <port id="11">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>128</dim>
                            <dim>128</dim>
                        </port>
                    </input>
                    <output>
                        <port id="12">
                            <dim>1</dim>
                            <dim>6</dim>
                            <dim>128</dim>
                            <dim>128</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="1" to-layer="2" to-port="2"/>
                <edge from-layer="1" from-port="1" to-layer="3" to-port="4"/>
                <edge from-layer="2" from-port="3" to-layer="4" to-port="6"/>
                <edge from-layer="3" from-port="5" to-layer="5" to-port="8"/>
                <edge from-layer="2" from-port="3" to-layer="6" to-port="10"/>
                <edge from-layer="3" from-port="5" to-layer="6" to-port="11"/>
            </edges>
        </Net>
    )V0G0N";

    TBlob<uint8_t>::Ptr weights(GenWeights(4 / sizeof(ie_fp16)));

    // Parse model
    InferenceEngine::Core ie;
    auto network = ie.ReadNetwork(model, weights);

    auto inputsInfo = network.getInputsInfo();
    inputsInfo["input"]->setPrecision(Precision::FP16);
    inputsInfo["input"]->setLayout(Layout::NHWC);

    auto outputsInfo = network.getOutputsInfo();
    outputsInfo["concat"]->setPrecision(Precision::FP16);
    outputsInfo["concat"]->setLayout(Layout::NHWC);
    outputsInfo["copy1"]->setPrecision(Precision::FP16);
    outputsInfo["copy1"]->setLayout(Layout::NHWC);
    outputsInfo["copy2"]->setPrecision(Precision::FP16);
    outputsInfo["copy2"]->setLayout(Layout::NHWC);

    // Load network
    ASSERT_NO_THROW(_exeNetwork = _vpuPluginPtr->LoadNetwork(network, {}));

    // Create InferRequest
    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = _exeNetwork.CreateInferRequest());
    
    // Generate input blob
    InferenceEngine::Blob::Ptr inputBlob;
    ASSERT_NO_THROW(inputBlob = inferRequest.GetBlob("input"));
    GenRandomData(inputBlob);

    // Get output blob
    InferenceEngine::Blob::Ptr output;
    ASSERT_NO_THROW(inferRequest.Infer());
    ASSERT_NO_THROW(output = inferRequest.GetBlob("concat"));
    
    // Get blobs which are input to Concat
    InferenceEngine::Blob::Ptr norm1, norm2;
    ASSERT_NO_THROW(norm1 = inferRequest.GetBlob("copy1"));
    ASSERT_NO_THROW(norm2 = inferRequest.GetBlob("copy2"));
    
    InferenceEngine::BlobMap normMap;
    normMap["normalize1"] = norm1;
    normMap["normalize2"] = norm2;
    CheckOutput(normMap, output, 2);
}
