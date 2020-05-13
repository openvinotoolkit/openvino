// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_mvn_test.hpp"

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsMVN_smoke,
                        ::testing::Combine(
                                ::testing::ValuesIn(s_MVNTensors),
                                ::testing::ValuesIn(s_MVN_acrossChannels),
                                ::testing::ValuesIn(s_MVN_normalize),
                                ::testing::ValuesIn(s_MVN_epsilon),
                                ::testing::Values(IRVersion::v7, IRVersion::v10),
                                ::testing::ValuesIn(s_MVNCustomConfig)));

TEST_F(myriadLayersTests_nightly, MVN_CHW_Input)
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

    StatusCode st;

    ASSERT_NO_THROW(readNetwork(model));

    const auto& network = _cnnNetwork;

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["data"]->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["mvn"]->setPrecision(Precision::FP16);

    ASSERT_NO_THROW(st = _vpuPluginPtr->LoadNetwork(_exeNetwork, network,
                                                      {{VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), CONFIG_VALUE(YES)}}, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    ASSERT_NE(_exeNetwork, nullptr) << _resp.msg;

    ASSERT_NO_THROW(st = _exeNetwork->CreateInferRequest(_inferRequest, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    auto tensorDesc = TensorDesc(Precision::FP16, _inputsInfo["data"]->getTensorDesc().getDims(), Layout::NCHW);
    auto inputNCHW = make_shared_blob<ie_fp16>(tensorDesc);
    ASSERT_NO_THROW(inputNCHW->allocate());

    auto outputNCHW = make_shared_blob<ie_fp16>(tensorDesc);
    ASSERT_NO_THROW(outputNCHW->allocate());

    auto output_ref = make_shared_blob<ie_fp16>(tensorDesc);
    ASSERT_NO_THROW(output_ref->allocate());

    ASSERT_NO_THROW(GenRandomData(inputNCHW));

    ASSERT_NO_THROW(st = _inferRequest->SetBlob("data", inputNCHW, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = _inferRequest->SetBlob("mvn", outputNCHW, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = _inferRequest->Infer(&_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_FATAL_FAILURE(refMVN(inputNCHW, output_ref, 1, 1, 9.999999717180685e-10, true));

    CompareCommonAbsolute(outputNCHW, output_ref, 0.003);
}
