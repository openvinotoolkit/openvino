// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_eltwise_test.hpp"

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsEltwiseMax_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_eltwiseTensors),
        ::testing::ValuesIn(s_eltwiseInputs),
        ::testing::ValuesIn(s_eltwiseDims))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsEltwiseSum_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_eltwiseTensors),
        ::testing::ValuesIn(s_eltwiseInputs),
        ::testing::ValuesIn(s_eltwiseDims))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsEltwiseSub_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_eltwiseTensors),
        ::testing::ValuesIn(s_eltwiseOnlyTwoInputs),
        ::testing::ValuesIn(s_eltwiseDims))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsEltwiseMul_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_eltwiseTensors),
        ::testing::ValuesIn(s_eltwiseInputs),
        ::testing::ValuesIn(s_eltwiseDims))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsEltwiseSumWithCoeff_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_eltwiseTensors),
        ::testing::ValuesIn(s_eltwiseInputs),
        ::testing::ValuesIn(s_eltwiseDims))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsEltwiseSumWithBroadcast_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_eltwiseTensors),
        ::testing::ValuesIn(s_eltwiseInputs),
        ::testing::Values<int>(4))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsEltwiseSubWithCoeff_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_eltwiseTensors),
        ::testing::ValuesIn(s_eltwiseOnlyTwoInputs),
        ::testing::ValuesIn(s_eltwiseDims))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsEltwiseSubWithBroadcast_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_eltwiseTensors),
        ::testing::ValuesIn(s_eltwiseOnlyTwoInputs),
        ::testing::Values<int>(4))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsEltwiseDiv_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_eltwiseTensors),
        ::testing::ValuesIn(s_eltwiseOnlyTwoInputs),
        ::testing::ValuesIn(s_eltwiseDims))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsEltwiseMin_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_eltwiseTensors),
        ::testing::ValuesIn(s_eltwiseInputs),
        ::testing::ValuesIn(s_eltwiseDims))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsEltwiseSqDiff_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_eltwiseTensors),
        ::testing::ValuesIn(s_eltwiseOnlyTwoInputs),
        ::testing::ValuesIn(s_eltwiseDims))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsEltwisePow_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_eltwiseTensors),
        ::testing::ValuesIn(s_eltwiseOnlyTwoInputs),
        ::testing::ValuesIn(s_eltwiseDims))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsEltwiseFloorMod_smoke,
        ::testing::Combine(
        ::testing::ValuesIn(s_eltwiseTensors),
        ::testing::ValuesIn(s_eltwiseOnlyTwoInputs),
        ::testing::ValuesIn(s_eltwiseDims))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsEltwiseEqual_smoke,
        ::testing::Combine(
        ::testing::ValuesIn(s_eltwiseTensors),
        ::testing::ValuesIn(s_eltwiseOnlyTwoInputs),
        ::testing::ValuesIn(s_eltwiseDims))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsEltwiseNotEqual_smoke,
        ::testing::Combine(
        ::testing::ValuesIn(s_eltwiseTensors),
        ::testing::ValuesIn(s_eltwiseOnlyTwoInputs),
        ::testing::ValuesIn(s_eltwiseDims))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsEltwiseGreater_smoke,
        ::testing::Combine(
        ::testing::ValuesIn(s_eltwiseTensors),
        ::testing::ValuesIn(s_eltwiseOnlyTwoInputs),
        ::testing::ValuesIn(s_eltwiseDims))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsEltwiseGreaterEqual_smoke,
        ::testing::Combine(
        ::testing::ValuesIn(s_eltwiseTensors),
        ::testing::ValuesIn(s_eltwiseOnlyTwoInputs),
        ::testing::ValuesIn(s_eltwiseDims))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsEltwiseLess_smoke,
        ::testing::Combine(
        ::testing::ValuesIn(s_eltwiseTensors),
        ::testing::ValuesIn(s_eltwiseOnlyTwoInputs),
        ::testing::ValuesIn(s_eltwiseDims))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsEltwiseLessEqual_smoke,
        ::testing::Combine(
        ::testing::ValuesIn(s_eltwiseTensors),
        ::testing::ValuesIn(s_eltwiseOnlyTwoInputs),
        ::testing::ValuesIn(s_eltwiseDims))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsEltwiseLogicalNot_smoke,
        ::testing::Combine(
        ::testing::ValuesIn(s_eltwiseTensors),
        ::testing::ValuesIn(s_eltwiseOnlyOneInput),
        ::testing::ValuesIn(s_eltwiseDims))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsEltwiseLogicalAnd_smoke,
        ::testing::Combine(
        ::testing::ValuesIn(s_eltwiseTensors),
        ::testing::ValuesIn(s_eltwiseInputs),
        ::testing::ValuesIn(s_eltwiseDims))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsEltwiseLogicalOr_smoke,
        ::testing::Combine(
        ::testing::ValuesIn(s_eltwiseTensors),
        ::testing::ValuesIn(s_eltwiseInputs),
        ::testing::ValuesIn(s_eltwiseDims))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsEltwiseLogicalXor_smoke,
        ::testing::Combine(
        ::testing::ValuesIn(s_eltwiseTensors),
        ::testing::ValuesIn(s_eltwiseInputs),
        ::testing::ValuesIn(s_eltwiseDims))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsEltwiseMean_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_eltwiseTensors),
        ::testing::ValuesIn(s_eltwiseOnlyTwoInputs),
        ::testing::ValuesIn(s_eltwiseDims))
);

TEST_F(myriadLayersTestsEltwiseBase, EltwiseWithSameInputs) {

    const std::string model = R"V0G0N(
<net batch="1" name="VNect: Test" version="2">
	<layers>
		<layer id="0" name="data" precision="FP16" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="pow1" precision="FP16" type="Power">
			<data power="1.0" scale="1.0" shift="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>2</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="eltwise" precision="FP16" type="Eltwise">
			<data operation="mul"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>2</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>2</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
	</edges>
</net>
            )V0G0N";

    InferenceEngine::Core ie;
    auto network = ie.ReadNetwork(model, InferenceEngine::Blob::CPtr());

    InferenceEngine::InputsDataMap networkInputs;
    ASSERT_NO_THROW(networkInputs = network.getInputsInfo());
    InferenceEngine::OutputsDataMap networkOutputs;
    ASSERT_NO_THROW(networkOutputs = network.getOutputsInfo());

    networkInputs.begin()->second->setPrecision(InferenceEngine::Precision::FP16);
    networkOutputs.begin()->second->setPrecision(InferenceEngine::Precision::FP16);

    InferenceEngine::Blob::Ptr inputBlob;
    InferenceEngine::ExecutableNetwork exeNetwork;

    std::map<std::string, std::string> networkConfig = {{InferenceEngine::MYRIAD_PERF_REPORT_MODE, InferenceEngine::MYRIAD_PER_STAGE}};
    ASSERT_NO_THROW(exeNetwork = _vpuPluginPtr->LoadNetwork(network, networkConfig));

    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = exeNetwork.CreateInferRequest());
    ASSERT_NO_THROW(inputBlob = inferRequest.GetBlob(networkInputs.begin()->first.c_str()));

    GenRandomData(inputBlob);

    InferenceEngine::Blob::Ptr output;
    ASSERT_NO_THROW(inferRequest.Infer());
    ASSERT_NO_THROW(output = inferRequest.GetBlob(networkOutputs.begin()->first.c_str()));

    _refBlob = make_shared_blob<ie_fp16>({Precision::FP16, output->getTensorDesc().getDims(), output->getTensorDesc().getLayout()});
    _refBlob->allocate();
    ref_eltwise(inputBlob, inputBlob, inputBlob, _refBlob, refMul, std::vector<float>({1.0f, 1.0f, 1.0f}));

    CompareCommonAbsolute(_refBlob, output, 0.1f);
};

TEST_F(myriadLayersTests_nightly, MergeEltwiseWithReLU) {
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

    ASSERT_NO_THROW(_exeNetwork = _vpuPluginPtr->LoadNetwork(network,
            { {InferenceEngine::MYRIAD_PERF_REPORT_MODE, InferenceEngine::MYRIAD_PER_STAGE},
              {CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES)},
              {InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, CONFIG_VALUE(NO)} }));

    ASSERT_NO_THROW(_inferRequest = _exeNetwork.CreateInferRequest());
    ASSERT_NO_THROW(_inferRequest.Infer());

    std::map<std::string, InferenceEngineProfileInfo> perfMap;
    ASSERT_NO_THROW(perfMap = _inferRequest.GetPerformanceCounts());

    auto sumAndReLULayerIt = perfMap.find("sum + sum_relu");
    ASSERT_TRUE(sumAndReLULayerIt != perfMap.end());
    EXPECT_EQ(InferenceEngineProfileInfo::EXECUTED, sumAndReLULayerIt->second.status);
}

TEST_F(myriadLayersTests_nightly, MergeEltwiseWithLeakyReLU) {
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
                <layer name="sum_leaky_relu" type="ReLU" precision="FP16" id="7">
                    <data negative_slope="3.0"/>
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

    ASSERT_NO_THROW(_exeNetwork = _vpuPluginPtr->LoadNetwork(network,
            { {InferenceEngine::MYRIAD_PERF_REPORT_MODE, InferenceEngine::MYRIAD_PER_STAGE},
              {CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES)},
              {InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, CONFIG_VALUE(NO)} }));

    ASSERT_NO_THROW(_inferRequest = _exeNetwork.CreateInferRequest());
    ASSERT_NO_THROW(_inferRequest.Infer());

    std::map<std::string, InferenceEngineProfileInfo> perfMap;
    ASSERT_NO_THROW(perfMap = _inferRequest.GetPerformanceCounts());

    auto sumAndReLULayerIt = perfMap.find("sum + sum_leaky_relu");
    ASSERT_TRUE(sumAndReLULayerIt != perfMap.end());
    EXPECT_EQ(InferenceEngineProfileInfo::EXECUTED, sumAndReLULayerIt->second.status);
}

TEST_F(myriadLayersTests_nightly, MergeEltwiseWithClamp) {
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
                <layer name="sum_clamp" type="Clamp" precision="FP16" id="7">
                    <data max="10" min="-10" />
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

    ASSERT_NO_THROW(_exeNetwork = _vpuPluginPtr->LoadNetwork(
            network,
            { {InferenceEngine::MYRIAD_PERF_REPORT_MODE, InferenceEngine::MYRIAD_PER_STAGE},
              {CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES)},
              {InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, CONFIG_VALUE(NO)} }));

    ASSERT_NO_THROW(_inferRequest = _exeNetwork.CreateInferRequest());
    ASSERT_NO_THROW(_inferRequest.Infer());

    std::map<std::string, InferenceEngineProfileInfo> perfMap;
    ASSERT_NO_THROW(perfMap = _inferRequest.GetPerformanceCounts());

    auto sumAndReLULayerIt = perfMap.find("sum + sum_clamp");
    ASSERT_TRUE(sumAndReLULayerIt != perfMap.end());
    EXPECT_EQ(InferenceEngineProfileInfo::EXECUTED, sumAndReLULayerIt->second.status);
}
