// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_relu_test.hpp"

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayerReLU_smoke,
                        ::testing::Combine(
                                ::testing::ValuesIn(s_copyTensors),
                                ::testing::ValuesIn(s_reluLayerParams)
                        )
);

INSTANTIATE_TEST_SUITE_P(
        accuracy, myriadLayerFullyConnectedWithReLU_smoke,
        ::testing::Combine(
                ::testing::ValuesIn(g_fcTestParamsSubset),
                ::testing::Values(g_dimensionsFC[0]),
                ::testing::ValuesIn(g_addBiasFC),
                ::testing::ValuesIn(s_reluLayerParams)
        )
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsMaxPoolingWithReLU_smoke,
                        ::testing::Combine(
                                ::testing::ValuesIn(g_poolingInput),
                                ::testing::ValuesIn(g_poolingLayerParamsLite),
                                ::testing::ValuesIn(g_poolingLayout),
                                ::testing::ValuesIn(s_reluLayerParams))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsAvgPoolingWithReLU_smoke,
                        ::testing::Combine(
                                ::testing::ValuesIn(g_poolingInput),
                                ::testing::ValuesIn(g_poolingLayerParamsLite),
                                ::testing::ValuesIn(g_poolingLayout),
                                ::testing::ValuesIn(s_reluLayerParams))
);

INSTANTIATE_TEST_SUITE_P(accuracy_postop, myriadLayersTestsMaxPoolingWithReLU_smoke,
                        ::testing::Combine(
                                ::testing::ValuesIn(g_poolingInput_postOp),
                                ::testing::Values<pooling_layer_params>(MAKE_STRUCT(pooling_layer_params, {3, 3}, {1, 1}, {1, 1})),
                                ::testing::ValuesIn(g_poolingLayout),
                                ::testing::Values<ReLULayerDef>(MAKE_STRUCT(ReLULayerDef, {{{"negative_slope", "0.0"}}})))
);

INSTANTIATE_TEST_SUITE_P(accuracy_postop, myriadLayersTestsAvgPoolingWithReLU_smoke,
                        ::testing::Combine(
                                ::testing::ValuesIn(g_poolingInput_postOp),
                                ::testing::Values<pooling_layer_params>(MAKE_STRUCT(pooling_layer_params, {3, 3}, {1, 1}, {1, 1})),
                                ::testing::ValuesIn(g_poolingLayout),
                                ::testing::Values<ReLULayerDef>(MAKE_STRUCT(ReLULayerDef, {{{"negative_slope", "0.0"}}})))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayerConvolutionWithReLU_smoke,
                        ::testing::Combine(
                                ::testing::ValuesIn(g_convolutionTensors)
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<uint32_t>(16)
                                , ::testing::Values<uint32_t>(1)
                                , ::testing::ValuesIn(s_reluLayerParams)
                        )
);

INSTANTIATE_TEST_SUITE_P(accuracy_postop, myriadLayerConvolutionWithReLU_smoke,
                        ::testing::Combine(
                                ::testing::ValuesIn(g_poolingInput_postOp)
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1)/*, MAKE_STRUCT(param_size, 2, 2)*/)
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<uint32_t>(32)
                                , ::testing::Values<uint32_t>(32)
                                , ::testing::Values<ReLULayerDef>(MAKE_STRUCT(ReLULayerDef, {{{"negative_slope", "0.0"}}}))
                        )
);

TEST_F(myriadLayersTests_nightly, graphTransformerNotThrowExceptionIfConvOutputIsInputForReLUAndGroupDeconv) {
    const std::string model = R"V0G0N(
    <net name="multi_hcp01" version="2" batch="1">
            <layers>
                <layer name="input" type="Input" precision="FP16" id="0">
                    <output>
                        <port id="0">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>23</dim>
                            <dim>40</dim>
                        </port>
                    </output>
                </layer>
               <layer name="conv1" type="Convolution" precision="FP16" id="1">
                    <convolution_data stride-x="1" stride-y="1" pad-x="0" pad-y="0" kernel-x="1" kernel-y="1" output="3" group="1"/>
                    <input>
                        <port id="1">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>23</dim>
                            <dim>40</dim>
                        </port>
                    </input>
                    <output>
                        <port id="2">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>23</dim>
                            <dim>40</dim>
                        </port>
                    </output>
                    <weights offset="0" size="18"/>
                    <biases offset="18" size="6"/>
                </layer>
                <layer name="conv1/relu" type="ReLU" precision="FP16" id="2">
                    <data negative_slope="0.000000" engine="caffe.ReLUParameter.DEFAULT"/>
                    <input>
                        <port id="3">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>23</dim>
                            <dim>40</dim>
                        </port>
                    </input>
                    <output>
                        <port id="4">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>23</dim>
                            <dim>40</dim>
                        </port>
                    </output>
                </layer>
            <layer name="deconv" type="Deconvolution" precision="FP16" id="3">
                <deconvolution_data stride-x="2" stride-y="2" pad-x="1" pad-y="1" kernel-x="4" kernel-y="4" output="3" group="3"/>
                <input>
                    <port id="5">
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>23</dim>
                        <dim>40</dim>
                    </port>
                </input>
                <output>
                    <port id="6">
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>46</dim>
                        <dim>80</dim>
                    </port>
                </output>
                <weights offset="24" size="96"/>
                <biases offset="120" size="0"/>
            </layer>
        </layers>
            <edges>
                <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
                <edge from-layer="1" from-port="2" to-layer="2" to-port="3"/>
                <edge from-layer="1" from-port="2" to-layer="3" to-port="5"/>
            </edges>
        </net>
        )V0G0N";

    TBlob<uint8_t>::Ptr weightsBlob(GenWeights(120));

    ASSERT_NO_THROW(readNetwork(model, weightsBlob));

    const auto& network = _cnnNetwork;

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["conv1/relu"]->setPrecision(Precision::FP16);
    _outputsInfo["deconv"]->setPrecision(Precision::FP16);

    ASSERT_NO_THROW(_exeNetwork = _vpuPluginPtr->LoadNetwork(network, {}));
    }

TEST_F(myriadLayersTests_nightly, ReLU_PostOp_Conflict) {
    const std::string model = R"V0G0N(
        <Net name="ReLU_PostOp_Conflict" version="2" batch="1">
            <layers>
                <layer name="input" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>64</dim>
                            <dim>64</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv" type="Convolution" precision="FP16" id="2">
                    <convolution_data
                        stride-x="1"
                        stride-y="1"
                        pad-x="1"
                        pad-y="1"
                        kernel-x="3"
                        kernel-y="3"
                        output="16"
                        group="1"/>
                    <input>
                        <port id="2">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>64</dim>
                            <dim>64</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>64</dim>
                            <dim>64</dim>
                        </port>
                    </output>
                    <weights offset="0" size="864"/>
                    <biases offset="864" size="32"/>
                </layer>
                <layer name="relu" type="ReLU" precision="FP16" id="3">
                    <data negative_slope="0.0" engine="caffe.ReLUParameter.DEFAULT"/>
                    <input>
                        <port id="4">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>64</dim>
                            <dim>64</dim>
                        </port>
                    </input>
                    <output>
                        <port id="5">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>64</dim>
                            <dim>64</dim>
                        </port>
                    </output>
                </layer>
                <layer name="power" type="Power" precision="FP16" id="4">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="6">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>64</dim>
                            <dim>64</dim>
                        </port>
                    </input>
                    <output>
                        <port id="7">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>64</dim>
                            <dim>64</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="1" to-layer="2" to-port="2"/>
                <edge from-layer="2" from-port="3" to-layer="3" to-port="4"/>
                <edge from-layer="2" from-port="3" to-layer="4" to-port="6"/>
            </edges>
        </Net>
    )V0G0N";

    size_t num_weights = 432;
    size_t num_bias = 16;

    TBlob<uint8_t>::Ptr weights(GenWeights(num_weights + num_bias));

    ASSERT_NO_THROW(readNetwork(model, weights));

    const auto& network = _cnnNetwork;

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["relu"]->setPrecision(Precision::FP16);
    _outputsInfo["power"]->setPrecision(Precision::FP16);

    ASSERT_NO_THROW(_exeNetwork = _vpuPluginPtr->LoadNetwork(network, {}));
}
