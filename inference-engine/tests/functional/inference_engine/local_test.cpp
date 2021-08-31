// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <legacy/net_pass.h>
#include "common_test_utils/common_utils.hpp"

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;

class LocaleTests : public ::testing::Test {
    std::string originalLocale;
    std::string _model = R"V0G0N(
<net name="model" version="10">
	<layers>
		<layer id="0" name="input" type="Parameter" version="opset1">
			<data shape="1,256,200,272" element_type="f16"/>
			<output>
				<port id="0" precision="FP16" names="input">
					<dim>1</dim>
					<dim>256</dim>
					<dim>200</dim>
					<dim>272</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="rois" type="Parameter" version="opset1">
			<data shape="1000,4" element_type="f16"/>
			<output>
				<port id="0" precision="FP16" names="rois">
					<dim>1000</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="indices" type="Parameter" version="opset1">
			<data shape="1000" element_type="f16"/>
			<output>
				<port id="0" precision="FP16" names="indices">
					<dim>1000</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="output" type="ROIAlign" version="opset3">
			<data mode="avg" pooled_h="7" pooled_w="7" sampling_ratio="2" spatial_scale="0.25"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>200</dim>
					<dim>272</dim>
				</port>
				<port id="1">
					<dim>1000</dim>
					<dim>4</dim>
				</port>
				<port id="2">
					<dim>1000</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP16" names="output">
					<dim>1000</dim>
					<dim>256</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="output/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1000</dim>
					<dim>256</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
		<edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
		<edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
		<edge from-layer="3" from-port="3" to-layer="4" to-port="0"/>
	</edges>
</net>
)V0G0N";

    std::string _model_LSTM = R"V0G0N(
<net name="model" version="10">
	<layers>
		<layer id="0" name="X" type="Parameter" version="opset1">
			<data element_type="f32" shape="10,4,64"/>
			<output>
				<port id="0" names="X" precision="FP32">
					<dim>10</dim>
					<dim>4</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="767658" type="Const" version="opset1">
			<data element_type="f32" offset="0" shape="512,192" size="393216"/>
			<output>
				<port id="0" precision="FP32">
					<dim>512</dim>
					<dim>192</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="768661" type="Const" version="opset1">
			<data element_type="f32" offset="393216" shape="512" size="2048"/>
			<output>
				<port id="0" precision="FP32">
					<dim>512</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="init_h" type="Const" version="opset1">
			<data element_type="f32" offset="395264" shape="1,4,128" size="2048"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="7/ShapeOf" type="ShapeOf" version="opset3">
			<data output_type="i64"/>
			<input>
				<port id="0">
					<dim>10</dim>
					<dim>4</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="7/ShapeOf/Indices86664" type="Const" version="opset1">
			<data element_type="i32" offset="397312" shape="1" size="4"/>
			<output>
				<port id="0" precision="I32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="7/ShapeOf/Axis87670" type="Const" version="opset1">
			<data element_type="i64" offset="397316" shape="" size="8"/>
			<output>
				<port id="0" precision="I64"/>
			</output>
		</layer>
		<layer id="7" name="7/ShapeOf/Gather" type="Gather" version="opset8">
			<data batch_dims="0"/>
			<input>
				<port id="0">
					<dim>3</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
				<port id="2"/>
			</input>
			<output>
				<port id="3" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="8" name="7/HiddenStateResizeDim/value91676" type="Const" version="opset1">
			<data element_type="i64" offset="397324" shape="1" size="8"/>
			<output>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="9" name="7/HiddenStateResizeDim" type="Concat" version="opset1">
			<data axis="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="10" name="7/HiddenStateResize" type="Reshape" version="opset1">
			<data special_zero="true"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>4</dim>
					<dim>128</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" names="init_h" precision="FP32">
					<dim>4</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="11" name="init_c" type="Const" version="opset1">
			<data element_type="f32" offset="395264" shape="1,4,128" size="2048"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="12" name="7/CellStateResize" type="Reshape" version="opset1">
			<data special_zero="true"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>4</dim>
					<dim>128</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" names="init_c" precision="FP32">
					<dim>4</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="13" name="7" type="LSTMSequence" version="opset1">
			<data activation_alpha="[0.3]" direction="forward" hidden_size="128"/>
			<input>
				<port id="0">
					<dim>10</dim>
					<dim>4</dim>
					<dim>64</dim>
				</port>
				<port id="1">
					<dim>512</dim>
					<dim>192</dim>
				</port>
				<port id="2">
					<dim>512</dim>
				</port>
				<port id="3">
					<dim>4</dim>
					<dim>128</dim>
				</port>
				<port id="4">
					<dim>4</dim>
					<dim>128</dim>
				</port>
			</input>
			<output>
				<port id="5" precision="FP32">
					<dim>10</dim>
					<dim>4</dim>
					<dim>128</dim>
				</port>
				<port id="6" precision="FP32">
					<dim>4</dim>
					<dim>128</dim>
				</port>
				<port id="7" precision="FP32">
					<dim>4</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="14" name="71655" type="Const" version="opset1">
			<data element_type="i64" offset="397332" shape="1" size="8"/>
			<output>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="15" name="7/UnsqueezeNumDirections/0" type="Unsqueeze" version="opset1">
			<input>
				<port id="0">
					<dim>10</dim>
					<dim>4</dim>
					<dim>128</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>10</dim>
					<dim>1</dim>
					<dim>4</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="16" name="7/UnsqueezeNumDirections/0/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>10</dim>
					<dim>1</dim>
					<dim>4</dim>
					<dim>128</dim>
				</port>
			</input>
		</layer>
		<layer id="17" name="79667" type="Const" version="opset1">
			<data element_type="i64" offset="397316" shape="1" size="8"/>
			<output>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="18" name="7/UnsqueezeNumDirections/2" type="Unsqueeze" version="opset1">
			<input>
				<port id="0">
					<dim>4</dim>
					<dim>128</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="19" name="7/UnsqueezeNumDirections/2/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>4</dim>
					<dim>128</dim>
				</port>
			</input>
		</layer>
		<layer id="20" name="75673" type="Const" version="opset1">
			<data element_type="i64" offset="397316" shape="1" size="8"/>
			<output>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="21" name="Y" type="Unsqueeze" version="opset1">
			<input>
				<port id="0">
					<dim>4</dim>
					<dim>128</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" names="Y" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="22" name="Y/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>4</dim>
					<dim>128</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
		<edge from-layer="4" from-port="1" to-layer="7" to-port="0"/>
		<edge from-layer="5" from-port="0" to-layer="7" to-port="1"/>
		<edge from-layer="6" from-port="0" to-layer="7" to-port="2"/>
		<edge from-layer="7" from-port="3" to-layer="9" to-port="0"/>
		<edge from-layer="8" from-port="0" to-layer="9" to-port="1"/>
		<edge from-layer="3" from-port="0" to-layer="10" to-port="0"/>
		<edge from-layer="9" from-port="2" to-layer="10" to-port="1"/>
		<edge from-layer="11" from-port="0" to-layer="12" to-port="0"/>
		<edge from-layer="9" from-port="2" to-layer="12" to-port="1"/>
		<edge from-layer="0" from-port="0" to-layer="13" to-port="0"/>
		<edge from-layer="1" from-port="0" to-layer="13" to-port="1"/>
		<edge from-layer="2" from-port="0" to-layer="13" to-port="2"/>
		<edge from-layer="10" from-port="2" to-layer="13" to-port="3"/>
		<edge from-layer="12" from-port="2" to-layer="13" to-port="4"/>
		<edge from-layer="13" from-port="5" to-layer="15" to-port="0"/>
		<edge from-layer="14" from-port="0" to-layer="15" to-port="1"/>
		<edge from-layer="15" from-port="2" to-layer="16" to-port="0"/>
		<edge from-layer="13" from-port="7" to-layer="18" to-port="0"/>
		<edge from-layer="17" from-port="0" to-layer="18" to-port="1"/>
		<edge from-layer="18" from-port="2" to-layer="19" to-port="0"/>
		<edge from-layer="13" from-port="6" to-layer="21" to-port="0"/>
		<edge from-layer="20" from-port="0" to-layer="21" to-port="1"/>
		<edge from-layer="21" from-port="2" to-layer="22" to-port="0"/>
	</edges>
</net>
)V0G0N";

protected:
    void SetUp() override {
        originalLocale  = setlocale(LC_ALL, nullptr);
    }
    void TearDown() override  {
        setlocale(LC_ALL, originalLocale.c_str());
    }

    void testBody(bool isLSTM = false) const {
        InferenceEngine::Core core;

        // This model contains layers with float attributes.
        // Conversion from string may be affected by locale.
        std::string model = isLSTM ? _model_LSTM : _model;
        auto blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {3360}, Layout::C));
        blob->allocate();
        auto net = core.ReadNetwork(model, blob);

        if (!isLSTM) {
            auto power_layer = dynamic_pointer_cast<PowerLayer>(CommonTestUtils::getLayerByName(net, "power"));
            ASSERT_EQ(power_layer->scale, 0.75f);
            ASSERT_EQ(power_layer->offset, 0.35f);
            ASSERT_EQ(power_layer->power, 0.5f);

            auto sum_layer = dynamic_pointer_cast<EltwiseLayer>(CommonTestUtils::getLayerByName(net, "sum"));
            std::vector<float> ref_coeff{0.77f, 0.33f};
            ASSERT_EQ(sum_layer->coeff, ref_coeff);

            auto info = net.getInputsInfo();
            auto preproc = info.begin()->second->getPreProcess();
            ASSERT_EQ(preproc[0]->stdScale, 0.1f);
            ASSERT_EQ(preproc[0]->meanValue, 104.006f);
        } else {
            InferenceEngine::NetPass::UnrollRNN_if(net, [] (const RNNCellBase& rnn) -> bool { return true; });
            auto lstmcell_layer = dynamic_pointer_cast<ClampLayer>(CommonTestUtils::getLayerByName(net, "LSTMCell:split_clip"));

            float ref_coeff = 0.2f;
            ASSERT_EQ(lstmcell_layer->min_value, -ref_coeff);
            ASSERT_EQ(lstmcell_layer->max_value,  ref_coeff);

            ASSERT_EQ(lstmcell_layer->GetParamAsFloat("min"), -ref_coeff);
            ASSERT_EQ(lstmcell_layer->GetParamAsFloat("max"),  ref_coeff);
        }
    }
};

TEST_F(LocaleTests, WithRULocale) {
    setlocale(LC_ALL, "ru_RU.UTF-8");
    testBody();
}

TEST_F(LocaleTests, WithUSLocale) {
    setlocale(LC_ALL, "en_US.UTF-8");
    testBody();
}

TEST_F(LocaleTests, WithRULocaleOnLSTM) {
    setlocale(LC_ALL, "ru_RU.UTF-8");
    testBody(true);
}

TEST_F(LocaleTests, WithUSLocaleOnLSTM) {
    setlocale(LC_ALL, "en_US.UTF-8");
    testBody(true);
}

TEST_F(LocaleTests, DISABLED_WithRULocaleCPP) {
    auto prev = std::locale();
    std::locale::global(std::locale("ru_RU.UTF-8"));
    testBody();
    std::locale::global(prev);
}

TEST_F(LocaleTests, DISABLED_WithUSLocaleCPP) {
    auto prev = std::locale();
    std::locale::global(std::locale("en_US.UTF-8"));
    testBody();
    std::locale::global(prev);
}
