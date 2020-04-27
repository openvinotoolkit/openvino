// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_test_data.hpp"
#include "single_layer_common.hpp"
#include "common_test_utils/common_layers_params.hpp"
#include "functional_test_utils/test_model/test_model.hpp"

using TestModel = FuncTestUtils::TestModel::TestModel;

std::string getRawCnnModel() {
    return (R"V0G0N(
<net name="_NAME_" version="_VER_" batch="1">
	<layers>
		<layer name="input_1" type="input" id="1" precision="_PRC_">
			<output>
				<port id="1">
					<!--connected to , Reshape_2-->
					<dim>1</dim>
					<dim>1056</dim>
				</port>
			</output>
		</layer>
		<layer name="Reshape_2" type="Reshape" id="2" precision="_PRC_">
			<input>
				<port id="2">
					<!--connected to input_1-->
					<dim>1</dim>
					<dim>1056</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<!--connected to , Convolution_3-->
					<dim>1</dim>
					<dim>33</dim>
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer name="Convolution_3" type="Convolution" id="3" precision="_PRC_">
			<convolution_data kernel-x="9" kernel-y="1" output="128" pad-x="0" pad-y="0" stride-x="1" stride-y="1" />
			<input>
				<port id="4">
					<!--connected to Reshape_2-->
					<dim>1</dim>
					<dim>33</dim>
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="5">
					<!--connected to , Pooling_5-->
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>24</dim>
				</port>
			</output>
			<weights offset="0" size="_CONV_WS_" precision="_PRC_" />
			<biases offset="_CONV_WS_" size="_CONV_BS_" precision="_PRC_" />
		</layer>
		<layer name="Pooling_5" type="Pooling" id="5" precision="_PRC_">
			<data kernel-x="3" kernel-y="1" pad-x="0" pad-y="0" pool-method="max" stride-x="3" stride-y="1" />
			<input>
				<port id="8">
					<!--connected to Convolution_3-->
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>24</dim>
				</port>
			</input>
			<output>
				<port id="9">
					<!--connected to , Reshape_6-->
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer name="Reshape_6" type="Reshape" id="6" precision="_PRC_">
			<input>
				<port id="10">
					<!--connected to Pooling_5-->
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="11">
					<!--connected to , ScaleShift_7-->
					<dim>1</dim>
					<dim>1024</dim>
				</port>
			</output>
		</layer>
		<layer name="ScaleShift_7" type="ScaleShift" id="7" precision="_PRC_">
			<input>
				<port id="13">
					<!--connected to Reshape_6-->
					<dim>1</dim>
					<dim>1024</dim>
				</port>
			</input>
			<output>
				<port id="12">
					<!--connected to , Activation_8-->
					<dim>1</dim>
					<dim>1024</dim>
				</port>
			</output>
			<weights offset="_SS_W_OFFS_" size="_SS_WS_" precision="_PRC_" />
			<biases offset="_SS_B_OFFS_" size="_SS_BS_" precision="_PRC_" />
		</layer>
		<layer name="Activation_8" type="Activation" id="8" precision="_PRC_">
			<data type="sigmoid" />
			<input>
				<port id="14">
					<!--connected to ScaleShift_7-->
					<dim>1</dim>
					<dim>1024</dim>
				</port>
			</input>
			<output>
				<port id="15">
					<dim>1</dim>
					<dim>1024</dim>
				</port>
			</output>
		</layer>
    </layers>
    <edges>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="2" />
		<edge from-layer="2" from-port="3" to-layer="3" to-port="4" />
		<edge from-layer="5" from-port="9" to-layer="6" to-port="10" />
		<edge from-layer="6" from-port="11" to-layer="7" to-port="13" />
		<edge from-layer="7" from-port="12" to-layer="8" to-port="14" />
    </edges>
</net>
)V0G0N");
}

TestModel getGnaCnnModel(InferenceEngine::Precision netPrc) {
    std::string model_str = getRawCnnModel();
    /* Default values for FP32 are used */
    size_t convWeigthsLen = 38016;  // kernel_x * kernel_y * in_channels * out_channels
    size_t convWeigthsSize = convWeigthsLen * 4;  // 4 (bytes in FP32)
    size_t convBiasesLen = 128;  // out_channels
    size_t convBiasesSize = convBiasesLen * 4;  // 4 (bytes in FP32)
    size_t scaleShiftWeigthsLen = 1024;  // out_channels
    size_t scaleShiftWeigthsSize = scaleShiftWeigthsLen * 4;  // 4 (bytes in FP32)
    size_t scaleShiftBiasesLen = 1024;  // out_channels
    size_t scaleShiftBiasesSize = scaleShiftBiasesLen * 4;  // 4 (bytes in FP32)
    switch (netPrc) {
        case InferenceEngine::Precision::FP32:
            break;
        default:
            std::string err = "GnaCnnModel can not be constructed with precision ";
            err += netPrc.name();
            throw std::runtime_error(err);
    }
    std::string irName = std::string("GnaCnnModel") + netPrc.name();
    REPLACE_WITH_STR(model_str, "_NAME_", irName);
    REPLACE_WITH_NUM(model_str, "_VER_", 2);
    REPLACE_WITH_STR(model_str, "_PRC_", netPrc.name());
    REPLACE_WITH_NUM(model_str, "_CONV_WS_", convWeigthsSize);
    REPLACE_WITH_NUM(model_str, "_CONV_BS_", convBiasesSize);
    REPLACE_WITH_NUM(model_str, "_SS_W_OFFS_", convWeigthsSize + convBiasesSize);
    REPLACE_WITH_NUM(model_str, "_SS_WS_", scaleShiftWeigthsSize);
    REPLACE_WITH_NUM(model_str, "_SS_B_OFFS_", convWeigthsSize + convBiasesSize + scaleShiftWeigthsSize);
    REPLACE_WITH_NUM(model_str, "_SS_BS_", scaleShiftBiasesSize);
    return TestModel(model_str, CommonTestUtils::getWeightsBlob(
            convWeigthsSize + convBiasesSize + scaleShiftWeigthsSize + scaleShiftBiasesSize));
}

std::string getRawMemoryModel() {
    return (R"V0G0N(
<net Name="activationAfterSplit" version="_VER_" precision="_PRC_" batch="1">
    <layers>
        <layer name="input_1" type="input" id="0" precision="_PRC_">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer name="Memory_27" type="Memory" id="27" precision="_PRC_">
            <data id="r_27-28" index="0" size="2" />
            <input>
                <port id="60">
                    <!--connected to Activation_38-->
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </input>
        </layer>
        <layer name="Memory_28" type="Memory" id="28" precision="_PRC_">
            <data id="r_27-28" index="1" size="2" />
            <output>
                <port id="59">
                    <!--connected to , Eltwise_8-->
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer name="FullyConnected" id="2" type="InnerProduct" precision="_PRC_">
            <fc out-size="10" />
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </output>
            <biases offset="0" size="_FC_BS_" />
            <weights offset="_FC_BS_" size="_FC_WS_" />
        </layer>
        <layer name="Activation_38" type="Activation" id="38" precision="_PRC_">
            <data type="tanh" />
            <input>
                <port id="82">
                    <!--connected to Eltwise_37-->
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </input>
            <output>
                <port id="83">
                    <!--connected to , Eltwise_41-->
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer name="Eltwise_8" type="Eltwise" id="11" precision="_PRC_">
            <data operation="sum" />
            <input>
                <port id="0">
                    <!--connected to FC1-->
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
                <port id="1">
                    <!--connected to FC2-->
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0" />
        <edge from-layer="2" from-port="1" to-layer="38" to-port="82" />
        <edge from-layer="38" from-port="83" to-layer="11" to-port="0" />
        <edge from-layer="28" from-port="59" to-layer="11" to-port="1" />
        <edge from-layer="38" from-port="83" to-layer="27" to-port="60" />
    </edges>
</net>
)V0G0N");
}

TestModel getGnaMemoryModel(InferenceEngine::Precision netPrc) {
    std::string model_str = getRawMemoryModel();
    /* Default values for FP32 are used */
    size_t fcBiasesLen = 10;  // num of fc_out_channels
    size_t fcWeigthsLen = 100;  // fc_in_channels * fc_out_channels
    size_t fcBiasesSize = fcBiasesLen * 4;  // 4 bytes for FP32
    size_t fcWeigthsSize = fcWeigthsLen * 4;  // 4 bytes for FP32
    switch (netPrc) {
        case InferenceEngine::Precision::FP32:
            break;
        default:
            std::string err = "getGnaMemoryModel can not be constructed with precision ";
            err += netPrc.name();
            throw std::runtime_error(err);
    }
    std::string irName = std::string("GnaMemoryModel") + netPrc.name();
    REPLACE_WITH_STR(model_str, "_NAME_", irName);
    REPLACE_WITH_NUM(model_str, "_VER_", 2);
    REPLACE_WITH_STR(model_str, "_PRC_", netPrc.name());
    REPLACE_WITH_NUM(model_str, "_FC_BS_", fcBiasesSize);
    REPLACE_WITH_NUM(model_str, "_FC_WS_", fcWeigthsSize);
    return TestModel(model_str, CommonTestUtils::getWeightsBlob(fcBiasesSize + fcWeigthsSize));
}
