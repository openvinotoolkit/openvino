// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "myriad_hw_tests_base.hpp"

TEST_F(MyriadX_HW_Tests_nightly, SeveralLayers) {
    if (!CheckMyriadX()) {
        GTEST_SKIP() << "Non-MyriadX device";
    }

    tensor_test_params dims1{1, 3, 224, 224};
    tensor_test_params dims2{1, 64, 112, 112};
    tensor_test_params dims3{1, 64, 56, 56};

    param_size kernel1{7, 7};
    param_size stride1{2, 2};
    param_size pad1{3, 3};

    param_size kernel2{3, 3};
    param_size stride2{2, 2};
    param_size pad2{0, 0};

    IN_OUT_desc tensor1, tensor2, tensor3;
    tensor1.push_back({dims1.n, dims1.c, dims1.h, dims1.w});
    tensor2.push_back({dims2.n, dims2.c, dims2.h, dims2.w});
    tensor3.push_back({dims3.n, dims3.c, dims3.h, dims3.w});

    size_t numWeights = kernel1.x * kernel1.y * dims1.c * dims2.c;
    size_t numBiases = dims2.c;

    ParamsStruct convParams = {
              {"kernel-x", std::to_string(kernel1.x)}
            , {"kernel-y", std::to_string(kernel1.y)}
            , {"stride-x", std::to_string(stride1.x)}
            , {"stride-y", std::to_string(stride1.y)}
            , {"pad-x", std::to_string(pad1.x)}
            , {"pad-y", std::to_string(pad1.y)}
            , {"output", std::to_string(dims2.c)}
            , {"group", "1"}
    };
    _testNet.addLayer(LayerInitParams("Convolution")
             .params(convParams)
             .weights(numWeights).fillWeights(defaultWeightsRange)
             .biases(numBiases).fillBiases(defaultWeightsRange)
             .in(tensor1)
             .out(tensor2),
             ref_convolution_wrap);

    ParamsStruct reluParams = {
        {"negative_slope", "0.0"}
    };
    _testNet.addLayer(LayerInitParams("ReLU")
             .params(reluParams)
             .in(tensor2)
             .out(tensor2),
             ref_ReLU_wrap);

    ParamsStruct poolParams = {
              {"kernel-x", std::to_string(kernel2.x)}
            , {"kernel-y", std::to_string(kernel2.y)}
            , {"stride-x", std::to_string(stride2.x)}
            , {"stride-y", std::to_string(stride2.y)}
            , {"pad-x", std::to_string(pad2.x)}
            , {"pad-y", std::to_string(pad2.y)}
            , {"pool-method", "max"}
    };
    _testNet.addLayer(LayerInitParams("Pooling")
             .params(poolParams)
             .in(tensor2)
             .out(tensor3),
             ref_pooling_wrap);

    CompareWithSW(0.1f);
}

TEST_F(MyriadX_HW_Tests_nightly, LargePoolWithConv) {
    if (!CheckMyriadX()) {
        GTEST_SKIP() << "Non-MyriadX device";
    }

    tensor_test_params dims1{1, 16, 448, 448};
    tensor_test_params dims2{1, 16, 224, 224};
    tensor_test_params dims3{1, 32, 224, 224};

    param_size kernel1{2, 2};
    param_size stride1{2, 2};
    param_size pad1{0, 0};

    param_size kernel2{3, 3};
    param_size stride2{1, 1};
    param_size pad2{1, 1};

    IN_OUT_desc tensor1, tensor2, tensor3;
    tensor1.push_back({dims1.n, dims1.c, dims1.h, dims1.w});
    tensor2.push_back({dims2.n, dims2.c, dims2.h, dims2.w});
    tensor3.push_back({dims3.n, dims3.c, dims3.h, dims3.w});

    ParamsStruct poolParams = {
              {"kernel-x", std::to_string(kernel1.x)}
            , {"kernel-y", std::to_string(kernel1.y)}
            , {"stride-x", std::to_string(stride1.x)}
            , {"stride-y", std::to_string(stride1.y)}
            , {"pad-x", std::to_string(pad1.x)}
            , {"pad-y", std::to_string(pad1.y)}
            , {"pool-method", "max"}
    };
    _testNet.addLayer(LayerInitParams("Pooling")
             .params(poolParams)
             .in(tensor1)
             .out(tensor2),
             ref_pooling_wrap);

    size_t numWeights = kernel2.x * kernel2.y * dims2.c * dims3.c;
    size_t numBiases = dims3.c;

    ParamsStruct convParams = {
              {"kernel-x", std::to_string(kernel2.x)}
            , {"kernel-y", std::to_string(kernel2.y)}
            , {"stride-x", std::to_string(stride2.x)}
            , {"stride-y", std::to_string(stride2.y)}
            , {"pad-x", std::to_string(pad2.x)}
            , {"pad-y", std::to_string(pad2.y)}
            , {"output", std::to_string(dims3.c)}
            , {"group", "1"}
    };
    _testNet.addLayer(LayerInitParams("Convolution")
             .params(convParams)
             .weights(numWeights).fillWeights(defaultWeightsRange)
             .biases(numBiases).fillBiases(defaultWeightsRange)
             .in(tensor2)
             .out(tensor3),
             ref_convolution_wrap);

    CompareWithSW(0.095f, vpu::LayoutPreference::ChannelMinor);
}

TEST_F(MyriadX_HW_Tests_nightly, ConvWithPool) {
    if (!CheckMyriadX()) {
        GTEST_SKIP() << "Non-MyriadX device";
    }

    tensor_test_params dims1{1, 16, 4, 4};
    tensor_test_params dims2{1, 64, 4, 4};
    tensor_test_params dims3{1, 64, 2, 2};

    param_size kernel1{3, 3};
    param_size stride1{1, 1};
    param_size pad1{1, 1};

    param_size kernel2{2, 2};
    param_size stride2{2, 2};
    param_size pad2{0, 0};

    IN_OUT_desc tensor1, tensor2, tensor3;
    tensor1.push_back({dims1.n, dims1.c, dims1.h, dims1.w});
    tensor2.push_back({dims2.n, dims2.c, dims2.h, dims2.w});
    tensor3.push_back({dims3.n, dims3.c, dims3.h, dims3.w});

    size_t numWeights = kernel1.x * kernel1.y * dims1.c * dims2.c;
    size_t numBiases = dims2.c;

    ParamsStruct convParams = {
              {"kernel-x", std::to_string(kernel1.x)}
            , {"kernel-y", std::to_string(kernel1.y)}
            , {"stride-x", std::to_string(stride1.x)}
            , {"stride-y", std::to_string(stride1.y)}
            , {"pad-x", std::to_string(pad1.x)}
            , {"pad-y", std::to_string(pad1.y)}
            , {"output", std::to_string(dims2.c)}
            , {"group", "1"}
    };

    _testNet.addLayer(LayerInitParams("Convolution")
             .params(convParams)
             .weights(numWeights).fillWeights(defaultWeightsRange)
             .biases(numBiases).fillBiases(defaultWeightsRange)
             .in(tensor1)
             .out(tensor2),
             ref_convolution_wrap);

    ParamsStruct poolParams = {
              {"kernel-x", std::to_string(kernel2.x)}
            , {"kernel-y", std::to_string(kernel2.y)}
            , {"stride-x", std::to_string(stride2.x)}
            , {"stride-y", std::to_string(stride2.y)}
            , {"pad-x", std::to_string(pad2.x)}
            , {"pad-y", std::to_string(pad2.y)}
            , {"pool-method", "max"}
    };

    _testNet.addLayer(LayerInitParams("Pooling")
             .params(poolParams)
             .in(tensor2)
             .out(tensor3),
             ref_pooling_wrap);

    CompareWithSW(0.08f);
}

TEST_F(MyriadX_HW_Tests_nightly, WithConcat) {
    if (!CheckMyriadX()) {
        GTEST_SKIP() << "Non-MyriadX device";
    }

    const std::string model = R"V0G0N(
        <Net name="WithConcat" version="2" batch="1">
            <layers>
                <layer name="input" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv1" type="Convolution" precision="FP16" id="2">
                    <data stride-x="1" stride-y="1" pad-x="0" pad-y="0" kernel-x="1" kernel-y="1" output="16" group="1"/>
                    <input>
                        <port id="2">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </output>
                    <weights offset="0" size="512"/>
                    <biases offset="512" size="32"/>
                </layer>
                <layer name="conv2" type="Convolution" precision="FP16" id="3">
                    <data stride-x="1" stride-y="1" pad-x="0" pad-y="0" kernel-x="1" kernel-y="1" output="16" group="1"/>
                    <input>
                        <port id="4">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </input>
                    <output>
                        <port id="5">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </output>
                    <weights offset="0" size="512"/>
                    <biases offset="512" size="32"/>
                </layer>
                <layer name="conv3" type="Convolution" precision="FP16" id="4">
                    <data stride-x="1" stride-y="1" pad-x="0" pad-y="0" kernel-x="1" kernel-y="1" output="16" group="1"/>
                    <input>
                        <port id="6">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </input>
                    <output>
                        <port id="7">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </output>
                    <weights offset="0" size="512"/>
                    <biases offset="512" size="32"/>
                </layer>
                <layer name="concat" type="Concat" precision="FP16" id="5">
                    <data axis="1"/>
                    <input>
                        <port id="8">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                        <port id="9">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                        <port id="10">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </input>
                    <output>
                        <port id="11">
                            <dim>1</dim>
                            <dim>48</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </output>
                </layer>
                <layer name="last" type="Convolution" precision="FP16" id="6">
                    <data stride-x="1" stride-y="1" pad-x="0" pad-y="0" kernel-x="1" kernel-y="1" output="48" group="1"/>
                    <input>
                        <port id="12">
                            <dim>1</dim>
                            <dim>48</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </input>
                    <output>
                        <port id="13">
                            <dim>1</dim>
                            <dim>48</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </output>
                    <weights offset="544" size="4608"/>
                    <biases offset="5152" size="96"/>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="1" to-layer="2" to-port="2"/>
                <edge from-layer="1" from-port="1" to-layer="3" to-port="4"/>
                <edge from-layer="1" from-port="1" to-layer="4" to-port="6"/>
                <edge from-layer="2" from-port="3" to-layer="5" to-port="8"/>
                <edge from-layer="3" from-port="5" to-layer="5" to-port="9"/>
                <edge from-layer="4" from-port="7" to-layer="5" to-port="10"/>
                <edge from-layer="5" from-port="11" to-layer="6" to-port="12"/>
            </edges>
        </Net>
    )V0G0N";

    TBlob<uint8_t>::Ptr weights(GenWeights(5248 / sizeof(ie_fp16)));

    ASSERT_NO_THROW(readNetwork(model, weights));

    const auto& network = _cnnNetwork;

    _inputsInfo = network.getInputsInfo();
    auto inputInfo = _inputsInfo["input"];
    inputInfo->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    auto outputInfo = _outputsInfo["last"];
    outputInfo->setPrecision(Precision::FP16);

    auto tensorDesc = InferenceEngine::TensorDesc(Precision::FP16, inputInfo->getTensorDesc().getDims(), Layout::NCHW);
    Blob::Ptr input = make_shared_blob<ie_fp16>(tensorDesc);
    input->allocate();
    GenRandomData(input);

    Blob::Ptr swOutput, hwOutput;
    {
        SCOPED_TRACE("SW");

        RunInfo runInfo;
        runInfo.hwMode = false;

        ASSERT_NO_FATAL_FAILURE(RunNetwork(network, input, swOutput, "input", "last", runInfo));
    }

    {
        SCOPED_TRACE("HW");

        RunInfo runInfo;
        runInfo.hwMode = true;

        ASSERT_NO_FATAL_FAILURE(RunNetwork(network, input, hwOutput, "input", "last", runInfo));

        ASSERT_NO_FATAL_FAILURE(CheckHWRun());
    }

    CompareCommonAbsolute(hwOutput, swOutput, 0.2f);
}

TEST_F(MyriadX_HW_Tests_nightly, WithConcatMisaligned) {
    if (!CheckMyriadX()) {
        GTEST_SKIP() << "Non-MyriadX device";
    }

    const std::string model = R"V0G0N(
        <Net name="WithConcat" version="2" batch="1">
            <layers>
                <layer name="input" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>35</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv1" type="Convolution" precision="FP16" id="2">
                    <data stride-x="1" stride-y="1" pad-x="0" pad-y="0" kernel-x="1" kernel-y="1" output="35" group="1"/>
                    <input>
                        <port id="2">
                            <dim>1</dim>
                            <dim>35</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>35</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </output>
                    <weights offset="0" size="2450"/>
                    <biases offset="2450" size="70"/>
                </layer>
                <layer name="conv2" type="Convolution" precision="FP16" id="3">
                    <data stride-x="1" stride-y="1" pad-x="0" pad-y="0" kernel-x="1" kernel-y="1" output="35" group="1"/>
                    <input>
                        <port id="4">
                            <dim>1</dim>
                            <dim>35</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </input>
                    <output>
                        <port id="5">
                            <dim>1</dim>
                            <dim>35</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </output>
                    <weights offset="0" size="2450"/>
                    <biases offset="2450" size="70"/>
                </layer>
                <layer name="concat" type="Concat" precision="FP16" id="4">
                    <data axis="1"/>
                    <input>
                        <port id="6">
                            <dim>1</dim>
                            <dim>35</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                        <port id="7">
                            <dim>1</dim>
                            <dim>35</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </input>
                    <output>
                        <port id="8">
                            <dim>1</dim>
                            <dim>70</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </output>
                </layer>
                <layer name="last" type="Power" precision="FP16" id="5">
                    <data power="1.0" scale="1.0" shift="0.0"/>
                    <input>
                        <port id="9">
                            <dim>1</dim>
                            <dim>70</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </input>
                    <output>
                        <port id="10">
                            <dim>1</dim>
                            <dim>70</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="1" to-layer="2" to-port="2"/>
                <edge from-layer="1" from-port="1" to-layer="3" to-port="4"/>
                <edge from-layer="2" from-port="3" to-layer="4" to-port="6"/>
                <edge from-layer="3" from-port="5" to-layer="4" to-port="7"/>
                <edge from-layer="4" from-port="8" to-layer="5" to-port="9"/>
            </edges>
        </Net>
    )V0G0N";

    TBlob<uint8_t>::Ptr weights(GenWeights(2520 / sizeof(ie_fp16)));

    ASSERT_NO_THROW(readNetwork(model, weights));

    const auto& network = _cnnNetwork;

    _inputsInfo = network.getInputsInfo();
    auto inputInfo = _inputsInfo["input"];
    inputInfo->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    auto outputInfo = _outputsInfo["last"];
    outputInfo->setPrecision(Precision::FP16);

    Blob::Ptr input = make_shared_blob<ie_fp16>({Precision::FP16, inputInfo->getTensorDesc().getDims(), Layout::NCHW});
    input->allocate();
    GenRandomData(input);

    Blob::Ptr swOutput, hwOutput;
    {
        SCOPED_TRACE("SW");

        RunInfo runInfo;
        runInfo.hwMode = false;

        ASSERT_NO_FATAL_FAILURE(RunNetwork(network, input, swOutput, "input", "last", runInfo));
    }

    {
        SCOPED_TRACE("HW");

        RunInfo runInfo;
        runInfo.hwMode = true;

        ASSERT_NO_FATAL_FAILURE(RunNetwork(network, input, hwOutput, "input", "last", runInfo));

        ASSERT_NO_FATAL_FAILURE(CheckHWRun());
    }

    CompareCommonAbsolute(hwOutput, swOutput, 0.03f);
}

TEST_F(MyriadX_HW_Tests_nightly, With_3_FC_Layers) {
    if (!CheckMyriadX()) {
        GTEST_SKIP() << "Non-MyriadX device";
    }

    const std::string model = R"V0G0N(
        <Net name="WithConcat" version="2" batch="1">
            <layers>
                <layer name="input" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>128</dim>
                            <dim>2</dim>
                            <dim>2</dim>
                        </port>
                    </output>
                </layer>
                <layer id="2" name="angle_y" precision="FP16" type="FullyConnected">
                    <data out-size="1"/>
                    <input>
                        <port id="0">
                            <dim>1</dim>
                            <dim>128</dim>
                            <dim>2</dim>
                            <dim>2</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </output>
                    <blobs>
                        <weights offset="0" size="1024"/>
                        <biases offset="1024" size="2"/>
                    </blobs>
                </layer>
                <layer id="3" name="angle_p" precision="FP16" type="FullyConnected">
                    <data out-size="1"/>
                    <input>
                        <port id="0">
                            <dim>1</dim>
                            <dim>128</dim>
                            <dim>2</dim>
                            <dim>2</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </output>
                    <blobs>
                        <weights offset="0" size="1024"/>
                        <biases offset="1024" size="2"/>
                    </blobs>
                </layer>
                <layer id="4" name="angle_q" precision="FP16" type="FullyConnected">
                    <data out-size="1"/>
                    <input>
                        <port id="0">
                            <dim>1</dim>
                            <dim>128</dim>
                            <dim>2</dim>
                            <dim>2</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </output>
                    <blobs>
                        <weights offset="0" size="1024"/>
                        <biases offset="1024" size="2"/>
                    </blobs>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
                <edge from-layer="1" from-port="1" to-layer="3" to-port="0"/>
                <edge from-layer="1" from-port="1" to-layer="4" to-port="0"/>
            </edges>
        </Net>
    )V0G0N";

    TBlob<uint8_t>::Ptr weights(GenWeights((1024 + 2) / sizeof(ie_fp16)));

    ASSERT_NO_THROW(readNetwork(model, weights));

    const auto& network = _cnnNetwork;

    _inputsInfo = network.getInputsInfo();
    auto inputInfo = _inputsInfo["input"];
    inputInfo->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    const std::string names[] = { "angle_p", "angle_q", "angle_y" };
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
        auto outputInfo = _outputsInfo[names[i]];
        ASSERT_NE(outputInfo, nullptr);
        outputInfo->setPrecision(Precision::FP32);

    }

    Blob::Ptr input = make_shared_blob<ie_fp16>({Precision::FP16, inputInfo->getTensorDesc().getDims(), Layout::NCHW});
    input->allocate();
    GenRandomData(input);

    Blob::Ptr swOutput, hwOutput;
    _inferRequest = {};
    _exeNetwork = {};

    ASSERT_NO_THROW(_exeNetwork = _vpuPluginPtr->LoadNetwork(network,
        {
            {
                InferenceEngine::MYRIAD_PERF_REPORT_MODE,
                InferenceEngine::MYRIAD_PER_STAGE
            },
            {
                InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION,
                CONFIG_VALUE(YES)
            },
        }));

    ASSERT_NO_THROW(_inferRequest = _exeNetwork.CreateInferRequest());
    ASSERT_NO_THROW(_inferRequest.SetBlob("input", input));
    ASSERT_NO_THROW(_inferRequest.Infer());

    std::vector<float> results(sizeof(names) / sizeof(names[0]));
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
        ASSERT_NO_THROW(hwOutput = _inferRequest.GetBlob(names[i].c_str()));
        ASSERT_NE(hwOutput, nullptr);
        BufferWrapper res_ptr(hwOutput);
        results[i] = res_ptr[0];
    }
    for (size_t i = 1; i < results.size(); ++i) {
        ASSERT_NEAR(results[0], results[i], 0.0001f);
    }
}

TEST_F(MyriadX_HW_Tests_nightly, WithEltwise) {
    if (!CheckMyriadX()) {
        GTEST_SKIP() << "Non-MyriadX device";
    }

    const std::string model = R"V0G0N(
        <Net name="WithEltwise" version="2" batch="1">
            <layers>
                <layer name="input" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                </layer>
                <layer name="branch1" type="Convolution" precision="FP16" id="2">
                    <convolution_data
                        stride-x="1" stride-y="1"
                        pad-x="0" pad-y="0"
                        kernel-x="1" kernel-y="1"
                        output="64"
                        group="1"/>
                    <input>
                        <port id="2">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                    <weights offset="0" size="8192"/>
                    <biases offset="8192" size="128"/>
                </layer>
                <layer name="branch2a" type="Convolution" precision="FP16" id="3">
                    <convolution_data
                        stride-x="1" stride-y="1"
                        pad-x="1" pad-y="1"
                        kernel-x="3" kernel-y="3"
                        output="64"
                        group="1"/>
                    <input>
                        <port id="4">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </input>
                    <output>
                        <port id="5">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                    <weights offset="8320" size="73728"/>
                    <biases offset="82048" size="128"/>
                </layer>
                <layer name="branch2a_relu" type="ReLU" precision="FP16" id="4">
                    <input>
                        <port id="6">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </input>
                    <output>
                        <port id="7">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                </layer>
                <layer name="branch2b" type="Convolution" precision="FP16" id="5">
                    <convolution_data
                        stride-x="1" stride-y="1"
                        pad-x="1" pad-y="1"
                        kernel-x="3" kernel-y="3"
                        output="64"
                        group="1"/>
                    <input>
                        <port id="8">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </input>
                    <output>
                        <port id="9">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                    <weights offset="82176" size="73728"/>
                    <biases offset="155904" size="128"/>
                </layer>
                <layer name="sum" type="Eltwise" precision="FP16" id="6">
                    <elementwise_data operation="sum"/>
                    <input>
                        <port id="10">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                        <port id="11">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </input>
                    <output>
                        <port id="12">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                </layer>
                <layer name="last" type="Convolution" precision="FP16" id="7">
                    <convolution_data
                        stride-x="1" stride-y="1"
                        pad-x="0" pad-y="0"
                        kernel-x="1" kernel-y="1"
                        output="64"
                        group="1"/>
                    <input>
                        <port id="13">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </input>
                    <output>
                        <port id="14">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                    <weights offset="156032" size="8192"/>
                    <biases offset="164224" size="128"/>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="1" to-layer="2" to-port="2"/>
                <edge from-layer="1" from-port="1" to-layer="3" to-port="4"/>
                <edge from-layer="3" from-port="5" to-layer="4" to-port="6"/>
                <edge from-layer="4" from-port="7" to-layer="5" to-port="8"/>
                <edge from-layer="2" from-port="3" to-layer="6" to-port="10"/>
                <edge from-layer="5" from-port="9" to-layer="6" to-port="11"/>
                <edge from-layer="6" from-port="12" to-layer="7" to-port="13"/>
            </edges>
        </Net>
    )V0G0N";

    TBlob<uint8_t>::Ptr weights(GenWeights(164352 / sizeof(ie_fp16)));

    ASSERT_NO_THROW(readNetwork(model, weights));

    const auto& network = _cnnNetwork;

    _inputsInfo = network.getInputsInfo();
    auto inputInfo = _inputsInfo["input"];
    inputInfo->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    auto outputInfo = _outputsInfo["last"];
    outputInfo->setPrecision(Precision::FP16);

    Blob::Ptr input = make_shared_blob<ie_fp16>({Precision::FP16, inputInfo->getTensorDesc().getDims(), Layout::NCHW});
    input->allocate();
    GenRandomData(input);

    Blob::Ptr swOutput;
    {
        SCOPED_TRACE("SW");

        RunInfo runInfo;
        runInfo.hwMode = false;

        ASSERT_NO_FATAL_FAILURE(RunNetwork(network, input, swOutput, "input", "last", runInfo));
    }

    Blob::Ptr hwOutput;
    {
        SCOPED_TRACE("HW");

        RunInfo runInfo;
        runInfo.hwMode = true;

        ASSERT_NO_FATAL_FAILURE(RunNetwork(network, input, hwOutput, "input", "last", runInfo));
        ASSERT_NO_FATAL_FAILURE(CheckHWRun());
    }

    CompareCommonAbsolute(hwOutput, swOutput, 30);
}

TEST_F(MyriadX_HW_Tests_nightly, WithEltwiseReLU) {
    if (!CheckMyriadX()) {
        GTEST_SKIP() << "Non-MyriadX device";
    }

    const std::string model = R"V0G0N(
        <Net name="WithEltwise" version="2" batch="1">
            <layers>
                <layer name="input" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                </layer>
                <layer name="branch1" type="Convolution" precision="FP16" id="2">
                    <convolution_data
                        stride-x="1" stride-y="1"
                        pad-x="0" pad-y="0"
                        kernel-x="1" kernel-y="1"
                        output="64"
                        group="1"/>
                    <input>
                        <port id="2">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                    <weights offset="0" size="8192"/>
                    <biases offset="8192" size="128"/>
                </layer>
                <layer name="branch2a" type="Convolution" precision="FP16" id="3">
                    <convolution_data
                        stride-x="1" stride-y="1"
                        pad-x="1" pad-y="1"
                        kernel-x="3" kernel-y="3"
                        output="64"
                        group="1"/>
                    <input>
                        <port id="4">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </input>
                    <output>
                        <port id="5">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                    <weights offset="8320" size="73728"/>
                    <biases offset="82048" size="128"/>
                </layer>
                <layer name="branch2a_relu" type="ReLU" precision="FP16" id="4">
                    <input>
                        <port id="6">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </input>
                    <output>
                        <port id="7">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                </layer>
                <layer name="branch2b" type="Convolution" precision="FP16" id="5">
                    <convolution_data
                        stride-x="1" stride-y="1"
                        pad-x="1" pad-y="1"
                        kernel-x="3" kernel-y="3"
                        output="64"
                        group="1"/>
                    <input>
                        <port id="8">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </input>
                    <output>
                        <port id="9">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                    <weights offset="82176" size="73728"/>
                    <biases offset="155904" size="128"/>
                </layer>
                <layer name="sum" type="Eltwise" precision="FP16" id="6">
                    <elementwise_data operation="sum"/>
                    <input>
                        <port id="10">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                        <port id="11">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </input>
                    <output>
                        <port id="12">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                </layer>
                <layer name="sum_relu" type="ReLU" precision="FP16" id="7">
                    <input>
                        <port id="13">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </input>
                    <output>
                        <port id="14">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                </layer>
                <layer name="last" type="Convolution" precision="FP16" id="8">
                    <convolution_data
                        stride-x="1" stride-y="1"
                        pad-x="0" pad-y="0"
                        kernel-x="1" kernel-y="1"
                        output="64"
                        group="1"/>
                    <input>
                        <port id="15">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </input>
                    <output>
                        <port id="16">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                    <weights offset="156032" size="8192"/>
                    <biases offset="164224" size="128"/>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="1" to-layer="2" to-port="2"/>
                <edge from-layer="1" from-port="1" to-layer="3" to-port="4"/>
                <edge from-layer="3" from-port="5" to-layer="4" to-port="6"/>
                <edge from-layer="4" from-port="7" to-layer="5" to-port="8"/>
                <edge from-layer="2" from-port="3" to-layer="6" to-port="10"/>
                <edge from-layer="5" from-port="9" to-layer="6" to-port="11"/>
                <edge from-layer="6" from-port="12" to-layer="7" to-port="13"/>
                <edge from-layer="7" from-port="14" to-layer="8" to-port="15"/>
            </edges>
        </Net>
    )V0G0N";

    TBlob<uint8_t>::Ptr weights(GenWeights(164352 / sizeof(ie_fp16)));

    ASSERT_NO_THROW(readNetwork(model, weights));

    const auto& network = _cnnNetwork;

    _inputsInfo = network.getInputsInfo();
    auto inputInfo = _inputsInfo["input"];
    inputInfo->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    auto outputInfo = _outputsInfo["last"];
    outputInfo->setPrecision(Precision::FP16);

    Blob::Ptr input = make_shared_blob<ie_fp16>(TensorDesc(Precision::FP16, inputInfo->getTensorDesc().getDims() , Layout::NCHW));
    input->allocate();
    GenRandomData(input);

    Blob::Ptr swOutput;
    {
        SCOPED_TRACE("SW");

        RunInfo runInfo;
        runInfo.hwMode = false;

        ASSERT_NO_FATAL_FAILURE(RunNetwork(network, input, swOutput, "input", "last", runInfo));
    }

    Blob::Ptr hwOutput;
    {
        SCOPED_TRACE("HW");

        RunInfo runInfo;
        runInfo.hwMode = true;

        ASSERT_NO_FATAL_FAILURE(RunNetwork(network, input, hwOutput, "input", "last", runInfo));
        ASSERT_NO_FATAL_FAILURE(CheckHWRun());
    }

    CompareCommonAbsolute(hwOutput, swOutput, 18.f);
}

TEST_F(MyriadX_HW_Tests_nightly, PermuteFlattenConcat) {
    if (!CheckMyriadX()) {
        GTEST_SKIP() << "Non-MyriadX device";
    }

    const std::string model = R"V0G0N(
        <Net name="WithPermuteFlattenConcat" version="2" batch="1">
            <layers>
                <layer name="input" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>23</dim>
                            <dim>23</dim>
                        </port>
                    </output>
                </layer>

                <layer name="conv1" type="Convolution" precision="FP16" id="2">
                    <convolution_data
                         stride-x="1"
                         stride-y="1"
                         pad-x="1"
                         pad-y="1"
                         kernel-x="3"
                         kernel-y="3"
                         output="54"
                         group="1" />
                    <input>
                        <port id="2">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>23</dim>
                            <dim>23</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>54</dim>
                            <dim>23</dim>
                            <dim>23</dim>
                        </port>
                    </output>
                    <weights offset="0" size="248832"/>
                    <biases offset="248832" size="108"/>
                </layer>
                <layer name="perm1" type="Permute" precision="FP16" id="3">
                    <data order="0,2,3,1"/>
                    <input>
                        <port id="4">
                            <dim>1</dim>
                            <dim>54</dim>
                            <dim>23</dim>
                            <dim>23</dim>
                        </port>
                    </input>
                    <output>
                        <port id="5">
                            <dim>1</dim>
                            <dim>23</dim>
                            <dim>23</dim>
                            <dim>54</dim>
                        </port>
                    </output>
                </layer>
                <layer name="flat1" type="Flatten" precision="FP16" id="4">
                    <data axis="1" end_axis="-1"/>
                    <input>
                        <port id="6">
                            <dim>1</dim>
                            <dim>23</dim>
                            <dim>23</dim>
                            <dim>54</dim>
                        </port>
                    </input>
                    <output>
                        <port id="7">
                            <dim>1</dim>
                            <dim>28566</dim>
                        </port>
                    </output>
                </layer>

                <layer name="conv2" type="Convolution" precision="FP16" id="5">
                    <convolution_data
                         stride-x="1"
                         stride-y="1"
                         pad-x="1"
                         pad-y="1"
                         kernel-x="3"
                         kernel-y="3"
                         output="54"
                         group="1" />
                    <input>
                        <port id="8">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>23</dim>
                            <dim>23</dim>
                        </port>
                    </input>
                    <output>
                        <port id="9">
                            <dim>1</dim>
                            <dim>54</dim>
                            <dim>23</dim>
                            <dim>23</dim>
                        </port>
                    </output>
                    <weights offset="0" size="248832"/>
                    <biases offset="248832" size="108"/>
                </layer>
                <layer name="perm2" type="Permute" precision="FP16" id="6">
                    <data order="0,2,3,1"/>
                    <input>
                        <port id="10">
                            <dim>1</dim>
                            <dim>54</dim>
                            <dim>23</dim>
                            <dim>23</dim>
                        </port>
                    </input>
                    <output>
                        <port id="11">
                            <dim>1</dim>
                            <dim>23</dim>
                            <dim>23</dim>
                            <dim>54</dim>
                        </port>
                    </output>
                </layer>
                <layer name="flat2" type="Flatten" precision="FP16" id="7">
                    <data axis="1" end_axis="-1"/>
                    <input>
                        <port id="12">
                            <dim>1</dim>
                            <dim>23</dim>
                            <dim>23</dim>
                            <dim>54</dim>
                        </port>
                    </input>
                    <output>
                        <port id="13">
                            <dim>1</dim>
                            <dim>28566</dim>
                        </port>
                    </output>
                </layer>

                <layer name="result" type="Concat" precision="FP16" id="8">
                    <concat_data axis="1"/>
                    <input>
                        <port id="14">
                            <dim>1</dim>
                            <dim>28566</dim>
                        </port>
                        <port id="15">
                            <dim>1</dim>
                            <dim>28566</dim>
                        </port>
                    </input>
                    <output>
                        <port id="16">
                            <dim>1</dim>
                            <dim>57132</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="1" to-layer="2" to-port="2"/>
                <edge from-layer="2" from-port="3" to-layer="3" to-port="4"/>
                <edge from-layer="3" from-port="5" to-layer="4" to-port="6"/>
                <edge from-layer="4" from-port="7" to-layer="8" to-port="14"/>
                <edge from-layer="1" from-port="1" to-layer="5" to-port="8"/>
                <edge from-layer="5" from-port="9" to-layer="6" to-port="10"/>
                <edge from-layer="6" from-port="11" to-layer="7" to-port="12"/>
                <edge from-layer="7" from-port="13" to-layer="8" to-port="15"/>
            </edges>
        </Net>
    )V0G0N";

    TBlob<uint8_t>::Ptr weights(GenWeights(248940 / sizeof(ie_fp16)));

    ASSERT_NO_THROW(readNetwork(model, weights));

    const auto& network = _cnnNetwork;

    _inputsInfo = network.getInputsInfo();
    auto inputInfo = _inputsInfo["input"];
    inputInfo->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    auto outputInfo = _outputsInfo["result"];
    outputInfo->setPrecision(Precision::FP16);

    Blob::Ptr input = make_shared_blob<ie_fp16>({Precision::FP16, inputInfo->getTensorDesc().getDims(), Layout::NCHW});
    input->allocate();
    GenRandomData(input);

    Blob::Ptr swOutput;
    {
        SCOPED_TRACE("SW");

        RunInfo runInfo;
        runInfo.hwMode = false;

        ASSERT_NO_FATAL_FAILURE(RunNetwork(network, input, swOutput, "input", "result", runInfo));
    }

    Blob::Ptr hwOutput;
    {
        SCOPED_TRACE("HW");

        RunInfo runInfo;
        runInfo.hwMode = true;

        ASSERT_NO_FATAL_FAILURE(RunNetwork(network, input, hwOutput, "input", "result", runInfo));
        ASSERT_NO_FATAL_FAILURE(CheckHWRun());
    }

    CompareCommonAbsolute(hwOutput, swOutput, 1.3f);
}

TEST_F(MyriadX_HW_Tests_nightly, VGG_FirstTwoConvs) {
    if (!CheckMyriadX()) {
        GTEST_SKIP() << "Non-MyriadX device";
    }

    IN_OUT_desc in_tensor, out_tensor;
    in_tensor.push_back({1, 3, 224, 224});
    out_tensor.push_back({1, 64, 224, 224});

    ParamsStruct conv1_params = {
        {"kernel-x", "3"},
        {"kernel-y", "3"},
        {"stride-x", "1"},
        {"stride-y", "1"},
        {"pad-x", "1"},
        {"pad-y", "1"},
        {"output", "64"},
        {"group", "1"}
    };
    _testNet.addLayer(LayerInitParams("Convolution")
             .params(conv1_params)
             .weights(1728).fillWeights(defaultWeightsRange)
             .biases(64).fillBiases(defaultWeightsRange)
             .in(in_tensor)
             .out(out_tensor),
             ref_convolution_wrap);

    _testNet.addLayer(LayerInitParams("ReLU")
             .in(out_tensor)
             .out(out_tensor),
             ref_ReLU_wrap);

    ParamsStruct conv2_params = {
        {"kernel-x", "3"},
        {"kernel-y", "3"},
        {"stride-x", "1"},
        {"stride-y", "1"},
        {"pad-x", "1"},
        {"pad-y", "1"},
        {"output", "64"},
        {"group", "1"}
    };
    _testNet.addLayer(LayerInitParams("Convolution")
             .params(conv2_params)
             .weights(36864).fillWeights(defaultWeightsRange)
             .biases(64).fillBiases(defaultWeightsRange)
             .in(out_tensor)
             .out(out_tensor),
             ref_convolution_wrap);

    _testNet.addLayer(LayerInitParams("ReLU")
             .in(out_tensor)
             .out(out_tensor),
             ref_ReLU_wrap);

    CompareWithSW(0.85f);
}
