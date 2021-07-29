// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_mvn_test.hpp"

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsMVN_smoke,
                        ::testing::Combine(
                                ::testing::ValuesIn(s_MVNTensors),
                                ::testing::ValuesIn(s_MVN_acrossChannels),
                                ::testing::ValuesIn(s_MVN_normalize),
                                ::testing::ValuesIn(s_MVN_epsilon),
                                ::testing::Values(IRVersion::v7, IRVersion::v10),
                                ::testing::ValuesIn(s_MVNCustomConfig)));

TEST_F(myriadLayersTests_nightly, DISABLED_MVN_CHW_Input)
{
    std::string model = R"V0G0N(
        <net name="MVN" version="2" batch="1">
            <layers>
                <layer name="data" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>512</dim>
                            <dim>896</dim>
                        </port>
                    </output>
                </layer>
                <layer name="mvn" type="MVN" precision="FP16" id="2">
                    <data across_channels="1" eps="9.999999717180685e-10" normalize_variance="1"/>
                    <input>
                        <port id="2">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>512</dim>
                            <dim>896</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>512</dim>
                            <dim>896</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="1" to-layer="2" to-port="2"/>
            </edges>
        </net>
    )V0G0N";

    ASSERT_NO_THROW(readNetwork(model));

    const auto& network = _cnnNetwork;

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["data"]->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["mvn"]->setPrecision(Precision::FP16);

    ASSERT_NO_THROW(_exeNetwork = _vpuPluginPtr->LoadNetwork(network,
            {{InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, CONFIG_VALUE(YES)}}));

    ASSERT_NO_THROW(_inferRequest = _exeNetwork.CreateInferRequest());
    
    auto tensorDesc = TensorDesc(Precision::FP16, _inputsInfo["data"]->getTensorDesc().getDims(), Layout::NCHW);
    auto inputNCHW = make_shared_blob<ie_fp16>(tensorDesc);
    ASSERT_NO_THROW(inputNCHW->allocate());

    auto outputNCHW = make_shared_blob<ie_fp16>(tensorDesc);
    ASSERT_NO_THROW(outputNCHW->allocate());

    auto output_ref = make_shared_blob<ie_fp16>(tensorDesc);
    ASSERT_NO_THROW(output_ref->allocate());

    ASSERT_NO_THROW(GenRandomData(inputNCHW));

    ASSERT_NO_THROW(_inferRequest.SetBlob("data", inputNCHW));
    ASSERT_NO_THROW(_inferRequest.SetBlob("mvn", outputNCHW));
    ASSERT_NO_THROW(_inferRequest.Infer());

    ASSERT_NO_FATAL_FAILURE(refMVN(inputNCHW, output_ref, 1, 1, 9.999999717180685e-10, true));

    CompareCommonAbsolute(outputNCHW, output_ref, 0.003);
}
