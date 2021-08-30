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
			<data shape="10, 12" element_type="f16"/>
			<output>
				<port id="0" precision="FP16" names="input">
					<dim>10</dim>
					<dim>12</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="output" type="Abs" version="opset1">
			<input>
				<port id="0" precision="FP16">
					<dim>10</dim>
					<dim>12</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP16" names="output">
					<dim>10</dim>
					<dim>12</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="output/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0" precision="FP16">
					<dim>10</dim>
					<dim>12</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
	</edges>
	<meta_data>
		<MO_version value="2021.4.0-3839-cd81789d294-releases/2021/4"/>
		<cli_parameters>
			<caffe_parser_path value="DIR"/>
			<data_type value="FP16"/>
			<disable_nhwc_to_nchw value="False"/>
			<disable_omitting_optional value="False"/>
			<disable_resnet_optimization value="False"/>
			<disable_weights_compression value="False"/>
			<enable_concat_optimization value="False"/>
			<enable_flattening_nested_params value="False"/>
			<enable_ssd_gluoncv value="False"/>
			<extensions value="DIR"/>
			<framework value="onnx"/>
			<freeze_placeholder_with_value value="{}"/>
			<generate_deprecated_IR_V7 value="False"/>
			<input_model value="DIR\model.onnx"/>
			<input_model_is_text value="False"/>
			<k value="DIR\CustomLayersMapping.xml"/>
			<keep_shape_ops value="True"/>
			<legacy_ir_generation value="False"/>
			<legacy_mxnet_model value="False"/>
			<log_level value="ERROR"/>
			<mean_scale_values value="{}"/>
			<mean_values value="()"/>
			<model_name value="model"/>
			<output_dir value="DIR"/>
			<placeholder_data_types value="{}"/>
			<progress value="False"/>
			<remove_memory value="False"/>
			<remove_output_softmax value="False"/>
			<reverse_input_channels value="False"/>
			<save_params_from_nd value="False"/>
			<scale_values value="()"/>
			<silent value="False"/>
			<static_shape value="False"/>
			<stream_output value="False"/>
			<transform value=""/>
			<unset unset_cli_parameters="batch, counts, disable_fusing, disable_gfusing, finegrain_fusing, input, input_checkpoint, input_meta_graph, input_proto, input_shape, input_symbol, mean_file, mean_file_offsets, move_to_preprocess, nd_prefix_name, output, placeholder_shapes, pretrained_model_name, saved_model_dir, saved_model_tags, scale, tensorboard_logdir, tensorflow_custom_layer_libraries, tensorflow_custom_operations_config_update, tensorflow_object_detection_api_pipeline_config, tensorflow_use_custom_operations_config, transformations_config"/>
		</cli_parameters>
	</meta_data>
</net>
)V0G0N";

    std::string _model_LSTM = R"V0G0N(
<net name="model" version="10">
	<layers>
		<layer id="0" name="X" type="Parameter" version="opset1">
			<data shape="10, 4, 64" element_type="f16"/>
			<output>
				<port id="0" precision="FP16" names="X">
					<dim>10</dim>
					<dim>4</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="init_h" type="Const" version="opset1">
			<data element_type="f16" shape="1, 4, 128" offset="0" size="1024"/>
			<output>
				<port id="0" precision="FP16">
					<dim>1</dim>
					<dim>4</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="7/ShapeOf" type="ShapeOf" version="opset3">
			<data output_type="i64"/>
			<input>
				<port id="0" precision="FP16">
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
		<layer id="3" name="7/ShapeOf/Indices86572" type="Const" version="opset1">
			<data element_type="i32" shape="1" offset="1024" size="4"/>
			<output>
				<port id="0" precision="I32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="7/ShapeOf/Axis87575" type="Const" version="opset1">
			<data element_type="i64" shape="" offset="1028" size="8"/>
			<output>
				<port id="0" precision="I64"/>
			</output>
		</layer>
		<layer id="5" name="7/ShapeOf/Gather" type="Gather" version="opset7">
			<data batch_dims="0"/>
			<input>
				<port id="0" precision="I64">
					<dim>3</dim>
				</port>
				<port id="1" precision="I32">
					<dim>1</dim>
				</port>
				<port id="2" precision="I64"/>
			</input>
			<output>
				<port id="3" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="7/HiddenStateResizeDim/value91566" type="Const" version="opset1">
			<data element_type="i64" shape="1" offset="1036" size="8"/>
			<output>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="7" name="7/HiddenStateResizeDim" type="Concat" version="opset1">
			<data axis="0"/>
			<input>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="8" name="7/HiddenStateResize" type="Reshape" version="opset1">
			<data special_zero="true"/>
			<input>
				<port id="0" precision="FP16">
					<dim>1</dim>
					<dim>4</dim>
					<dim>128</dim>
				</port>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP16" names="init_h">
					<dim>4</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="9" name="init_c" type="Const" version="opset1">
			<data element_type="f16" shape="1, 4, 128" offset="0" size="1024"/>
			<output>
				<port id="0" precision="FP16">
					<dim>1</dim>
					<dim>4</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="10" name="7/CellStateResize" type="Reshape" version="opset1">
			<data special_zero="true"/>
			<input>
				<port id="0" precision="FP16">
					<dim>1</dim>
					<dim>4</dim>
					<dim>128</dim>
				</port>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP16" names="init_c">
					<dim>4</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="11" name="7/TensorIterator" type="TensorIterator" version="opset1">
			<port_map>
				<input axis="0" external_port_id="0" internal_layer_id="2" start="0" end="-1" stride="1" part_size="1"/>
				<input external_port_id="1" internal_layer_id="1"/>
				<input external_port_id="2" internal_layer_id="0"/>
				<output axis="0" external_port_id="3" internal_layer_id="11" start="0" end="-1" stride="1" part_size="1"/>
				<output external_port_id="4" internal_layer_id="13"/>
				<output external_port_id="5" internal_layer_id="12"/>
			</port_map>
			<back_edges>
				<edge from-layer="13" to-layer="1"/>
				<edge from-layer="12" to-layer="0"/>
			</back_edges>
			<input>
				<port id="0" precision="FP16">
					<dim>10</dim>
					<dim>4</dim>
					<dim>64</dim>
				</port>
				<port id="1" precision="FP16">
					<dim>4</dim>
					<dim>128</dim>
				</port>
				<port id="2" precision="FP16">
					<dim>4</dim>
					<dim>128</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP16">
					<dim>10</dim>
					<dim>4</dim>
					<dim>128</dim>
				</port>
				<port id="4" precision="FP16">
					<dim>4</dim>
					<dim>128</dim>
				</port>
				<port id="5" precision="FP16">
					<dim>4</dim>
					<dim>128</dim>
				</port>
			</output>
			<body>
				<layers>
					<layer id="0" name="24" type="Parameter" version="opset1">
						<data shape="4, 128" element_type="f16"/>
						<output>
							<port id="0" precision="FP16">
								<dim>4</dim>
								<dim>128</dim>
							</port>
						</output>
					</layer>
					<layer id="1" name="22" type="Parameter" version="opset1">
						<data shape="4, 128" element_type="f16"/>
						<output>
							<port id="0" precision="FP16">
								<dim>4</dim>
								<dim>128</dim>
							</port>
						</output>
					</layer>
					<layer id="2" name="20" type="Parameter" version="opset1">
						<data shape="1, 4, 64" element_type="f16"/>
						<output>
							<port id="0" precision="FP16">
								<dim>1</dim>
								<dim>4</dim>
								<dim>64</dim>
							</port>
						</output>
					</layer>
					<layer id="3" name="7253" type="Const" version="opset1">
						<data element_type="i64" shape="1" offset="1028" size="8"/>
						<output>
							<port id="0" precision="I64">
								<dim>1</dim>
							</port>
						</output>
					</layer>
					<layer id="4" name="7/input_squeeze" type="Squeeze" version="opset1">
						<input>
							<port id="0" precision="FP16">
								<dim>1</dim>
								<dim>4</dim>
								<dim>64</dim>
							</port>
							<port id="1" precision="I64">
								<dim>1</dim>
							</port>
						</input>
						<output>
							<port id="2" precision="FP16">
								<dim>4</dim>
								<dim>64</dim>
							</port>
						</output>
					</layer>
					<layer id="5" name="7/LSTMCell/Split197244" type="Const" version="opset1">
						<data element_type="f16" shape="512, 64" offset="1044" size="65536"/>
						<output>
							<port id="0" precision="FP16">
								<dim>512</dim>
								<dim>64</dim>
							</port>
						</output>
					</layer>
					<layer id="6" name="7/LSTMCell/Split198247" type="Const" version="opset1">
						<data element_type="f16" shape="512, 128" offset="66580" size="131072"/>
						<output>
							<port id="0" precision="FP16">
								<dim>512</dim>
								<dim>128</dim>
							</port>
						</output>
					</layer>
					<layer id="7" name="7/inport/2250" type="Const" version="opset1">
						<data element_type="f16" shape="512" offset="197652" size="1024"/>
						<output>
							<port id="0" precision="FP16">
								<dim>512</dim>
							</port>
						</output>
					</layer>
					<layer id="8" name="7/LSTMCell" type="LSTMCell" version="opset4">
						<data hidden_size="128" activations="sigmoid, tanh, tanh" activations_alpha="" activations_beta="" clip="0"/>
						<input>
							<port id="0" precision="FP16">
								<dim>4</dim>
								<dim>64</dim>
							</port>
							<port id="1" precision="FP16">
								<dim>4</dim>
								<dim>128</dim>
							</port>
							<port id="2" precision="FP16">
								<dim>4</dim>
								<dim>128</dim>
							</port>
							<port id="3" precision="FP16">
								<dim>512</dim>
								<dim>64</dim>
							</port>
							<port id="4" precision="FP16">
								<dim>512</dim>
								<dim>128</dim>
							</port>
							<port id="5" precision="FP16">
								<dim>512</dim>
							</port>
						</input>
						<output>
							<port id="6" precision="FP16">
								<dim>4</dim>
								<dim>128</dim>
							</port>
							<port id="7" precision="FP16">
								<dim>4</dim>
								<dim>128</dim>
							</port>
						</output>
					</layer>
					<layer id="9" name="15256" type="Const" version="opset1">
						<data element_type="i64" shape="1" offset="1028" size="8"/>
						<output>
							<port id="0" precision="I64">
								<dim>1</dim>
							</port>
						</output>
					</layer>
					<layer id="10" name="7output_unsqueeze" type="Unsqueeze" version="opset1">
						<input>
							<port id="0" precision="FP16">
								<dim>4</dim>
								<dim>128</dim>
							</port>
							<port id="1" precision="I64">
								<dim>1</dim>
							</port>
						</input>
						<output>
							<port id="2" precision="FP16">
								<dim>1</dim>
								<dim>4</dim>
								<dim>128</dim>
							</port>
						</output>
					</layer>
					<layer id="11" name="18/sink_port_0" type="Result" version="opset1">
						<input>
							<port id="0" precision="FP16">
								<dim>1</dim>
								<dim>4</dim>
								<dim>128</dim>
							</port>
						</input>
					</layer>
					<layer id="12" name="7/outport/1/sink_port_0" type="Result" version="opset1">
						<input>
							<port id="0" precision="FP16">
								<dim>4</dim>
								<dim>128</dim>
							</port>
						</input>
					</layer>
					<layer id="13" name="7/outport/0/sink_port_0" type="Result" version="opset1">
						<input>
							<port id="0" precision="FP16">
								<dim>4</dim>
								<dim>128</dim>
							</port>
						</input>
					</layer>
				</layers>
				<edges>
					<edge from-layer="0" from-port="0" to-layer="8" to-port="2"/>
					<edge from-layer="1" from-port="0" to-layer="8" to-port="1"/>
					<edge from-layer="2" from-port="0" to-layer="4" to-port="0"/>
					<edge from-layer="3" from-port="0" to-layer="4" to-port="1"/>
					<edge from-layer="4" from-port="2" to-layer="8" to-port="0"/>
					<edge from-layer="5" from-port="0" to-layer="8" to-port="3"/>
					<edge from-layer="6" from-port="0" to-layer="8" to-port="4"/>
					<edge from-layer="7" from-port="0" to-layer="8" to-port="5"/>
					<edge from-layer="8" from-port="6" to-layer="10" to-port="0"/>
					<edge from-layer="8" from-port="7" to-layer="12" to-port="0"/>
					<edge from-layer="8" from-port="6" to-layer="13" to-port="0"/>
					<edge from-layer="9" from-port="0" to-layer="10" to-port="1"/>
					<edge from-layer="10" from-port="2" to-layer="11" to-port="0"/>
				</edges>
			</body>
		</layer>
		<layer id="12" name="75578" type="Const" version="opset1">
			<data element_type="i64" shape="1" offset="1028" size="8"/>
			<output>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="13" name="Y" type="Unsqueeze" version="opset1">
			<input>
				<port id="0" precision="FP16">
					<dim>4</dim>
					<dim>128</dim>
				</port>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP16" names="Y">
					<dim>1</dim>
					<dim>4</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="14" name="Y/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0" precision="FP16">
					<dim>1</dim>
					<dim>4</dim>
					<dim>128</dim>
				</port>
			</input>
		</layer>
		<layer id="15" name="79581" type="Const" version="opset1">
			<data element_type="i64" shape="1" offset="1028" size="8"/>
			<output>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="16" name="7/UnsqueezeNumDirections/2" type="Unsqueeze" version="opset1">
			<input>
				<port id="0" precision="FP16">
					<dim>4</dim>
					<dim>128</dim>
				</port>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP16">
					<dim>1</dim>
					<dim>4</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="17" name="7/UnsqueezeNumDirections/2/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0" precision="FP16">
					<dim>1</dim>
					<dim>4</dim>
					<dim>128</dim>
				</port>
			</input>
		</layer>
		<layer id="18" name="71560" type="Const" version="opset1">
			<data element_type="i64" shape="1" offset="198676" size="8"/>
			<output>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="19" name="7/UnsqueezeNumDirections/0" type="Unsqueeze" version="opset1">
			<input>
				<port id="0" precision="FP16">
					<dim>10</dim>
					<dim>4</dim>
					<dim>128</dim>
				</port>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP16">
					<dim>10</dim>
					<dim>1</dim>
					<dim>4</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="20" name="7/UnsqueezeNumDirections/0/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0" precision="FP16">
					<dim>10</dim>
					<dim>1</dim>
					<dim>4</dim>
					<dim>128</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="0" from-port="0" to-layer="11" to-port="0"/>
		<edge from-layer="1" from-port="0" to-layer="8" to-port="0"/>
		<edge from-layer="2" from-port="1" to-layer="5" to-port="0"/>
		<edge from-layer="3" from-port="0" to-layer="5" to-port="1"/>
		<edge from-layer="4" from-port="0" to-layer="5" to-port="2"/>
		<edge from-layer="5" from-port="3" to-layer="7" to-port="0"/>
		<edge from-layer="6" from-port="0" to-layer="7" to-port="1"/>
		<edge from-layer="7" from-port="2" to-layer="8" to-port="1"/>
		<edge from-layer="7" from-port="2" to-layer="10" to-port="1"/>
		<edge from-layer="8" from-port="2" to-layer="11" to-port="1"/>
		<edge from-layer="9" from-port="0" to-layer="10" to-port="0"/>
		<edge from-layer="10" from-port="2" to-layer="11" to-port="2"/>
		<edge from-layer="11" from-port="4" to-layer="13" to-port="0"/>
		<edge from-layer="11" from-port="5" to-layer="16" to-port="0"/>
		<edge from-layer="11" from-port="3" to-layer="19" to-port="0"/>
		<edge from-layer="12" from-port="0" to-layer="13" to-port="1"/>
		<edge from-layer="13" from-port="2" to-layer="14" to-port="0"/>
		<edge from-layer="15" from-port="0" to-layer="16" to-port="1"/>
		<edge from-layer="16" from-port="2" to-layer="17" to-port="0"/>
		<edge from-layer="18" from-port="0" to-layer="19" to-port="1"/>
		<edge from-layer="19" from-port="2" to-layer="20" to-port="0"/>
	</edges>
	<meta_data>
		<MO_version value="2021.4.0-3839-cd81789d294-releases/2021/4"/>
		<cli_parameters>
			<caffe_parser_path value="DIR"/>
			<data_type value="FP16"/>
			<disable_nhwc_to_nchw value="False"/>
			<disable_omitting_optional value="False"/>
			<disable_resnet_optimization value="False"/>
			<disable_weights_compression value="False"/>
			<enable_concat_optimization value="False"/>
			<enable_flattening_nested_params value="False"/>
			<enable_ssd_gluoncv value="False"/>
			<extensions value="DIR"/>
			<framework value="onnx"/>
			<freeze_placeholder_with_value value="{}"/>
			<generate_deprecated_IR_V7 value="False"/>
			<input_model value="DIR\model.onnx"/>
			<input_model_is_text value="False"/>
			<k value="DIR\CustomLayersMapping.xml"/>
			<keep_shape_ops value="True"/>
			<legacy_ir_generation value="False"/>
			<legacy_mxnet_model value="False"/>
			<log_level value="ERROR"/>
			<mean_scale_values value="{}"/>
			<mean_values value="()"/>
			<model_name value="model"/>
			<output_dir value="DIR"/>
			<placeholder_data_types value="{}"/>
			<progress value="False"/>
			<remove_memory value="False"/>
			<remove_output_softmax value="False"/>
			<reverse_input_channels value="False"/>
			<save_params_from_nd value="False"/>
			<scale_values value="()"/>
			<silent value="False"/>
			<static_shape value="False"/>
			<stream_output value="False"/>
			<transform value=""/>
			<unset unset_cli_parameters="batch, counts, disable_fusing, disable_gfusing, finegrain_fusing, input, input_checkpoint, input_meta_graph, input_proto, input_shape, input_symbol, mean_file, mean_file_offsets, move_to_preprocess, nd_prefix_name, output, placeholder_shapes, pretrained_model_name, saved_model_dir, saved_model_tags, scale, tensorboard_logdir, tensorflow_custom_layer_libraries, tensorflow_custom_operations_config_update, tensorflow_object_detection_api_pipeline_config, tensorflow_use_custom_operations_config, transformations_config"/>
		</cli_parameters>
	</meta_data>
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
