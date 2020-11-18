// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

namespace FuncTestUtils {

const char serialize_test_model[] = R"V0G0N(<?xml version="1.0" ?>
<?xml version="1.0" ?>
<net name="addmul_abc" version="10">
	<layers>
		<layer id="0" name="A" type="Parameter" version="opset1">
			<data element_type="f32" shape="1"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="B" type="Parameter" version="opset1">
			<data element_type="f32" shape="1"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="add_node1" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="add_node2" type="Multiply" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="add_node3" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="C" type="Parameter" version="opset1">
			<data element_type="f32" shape="1"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="add_node4" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="7" name="Y" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="8" name="Y/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
		<edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
		<edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
		<edge from-layer="2" from-port="2" to-layer="4" to-port="0"/>
		<edge from-layer="3" from-port="2" to-layer="4" to-port="1"/>
		<edge from-layer="4" from-port="2" to-layer="6" to-port="0"/>
		<edge from-layer="5" from-port="0" to-layer="6" to-port="1"/>
		<edge from-layer="6" from-port="2" to-layer="7" to-port="0"/>
		<edge from-layer="5" from-port="0" to-layer="7" to-port="1"/>
		<edge from-layer="7" from-port="2" to-layer="8" to-port="0"/>
	</edges>
	<meta_data>
		<MO_version value="unknown version"/>
		<cli_parameters>
			<caffe_parser_path value="DIR"/>
			<data_type value="float"/>
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
			<input_model value="DIR/addmul_abc.onnx"/>
			<input_model_is_text value="False"/>
			<k value="DIR/CustomLayersMapping.xml"/>
			<keep_shape_ops value="True"/>
			<legacy_mxnet_model value="False"/>
			<log_level value="ERROR"/>
			<mean_scale_values value="{}"/>
			<mean_values value="()"/>
			<model_name value="addmul_abc"/>
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
			<unset unset_cli_parameters="batch, counts, disable_fusing, disable_gfusing, finegrain_fusing, input, input_checkpoint, input_meta_graph, input_proto, input_shape, input_symbol, mean_file, mean_file_offsets, move_to_preprocess, nd_prefix_name, output, placeholder_shapes, pretrained_model_name, saved_model_dir, saved_model_tags, scale, tensorboard_logdir, tensorflow_custom_layer_libraries, tensorflow_custom_operations_config_update, tensorflow_object_detection_api_pipeline_config, tensorflow_use_custom_operations_config, transformations_config"/>
		</cli_parameters>
	</meta_data>
</net>
)V0G0N";

const char expected_serialized_model[] = R"V0G0N(
<?xml version="1.0"?>
<net name="addmul_abc" version="10">
	<layers>
		<layer id="0" name="C" type="Input">
			<data execOrder="3" execTimeMcs="not_executed" originalLayersNames="C" outputLayouts="x" outputPrecisions="FP32" primitiveType="unknown_FP32" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="B" type="Input">
			<data execOrder="1" execTimeMcs="not_executed" originalLayersNames="B" outputLayouts="x" outputPrecisions="FP32" primitiveType="unknown_FP32" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="A" type="Input">
			<data execOrder="0" execTimeMcs="not_executed" originalLayersNames="A" outputLayouts="x" outputPrecisions="FP32" primitiveType="unknown_FP32" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="add_node2" type="Eltwise">
			<data execOrder="2" execTimeMcs="not_executed" originalLayersNames="add_node2" outputLayouts="x" outputPrecisions="FP32" primitiveType="jit_avx512_FP32" />
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="add_node1" type="Eltwise">
			<data execOrder="4" execTimeMcs="not_executed" originalLayersNames="add_node1,add_node3,add_node4" outputLayouts="x" outputPrecisions="FP32" primitiveType="jit_avx512_FP32" />
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
				<port id="2">
					<dim>1</dim>
				</port>
				<port id="3">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="4" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="Y" type="Eltwise">
			<data execOrder="5" execTimeMcs="not_executed" originalLayersNames="Y" outputLayouts="x" outputPrecisions="FP32" primitiveType="jit_avx512_FP32" />
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="out_Y" type="Output">
			<data execOrder="6" execTimeMcs="not_executed" originalLayersNames="" outputLayouts="undef" outputPrecisions="FP32" primitiveType="unknown_FP32" />
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="4" to-port="3" />
		<edge from-layer="0" from-port="0" to-layer="5" to-port="1" />
		<edge from-layer="1" from-port="0" to-layer="3" to-port="1" />
		<edge from-layer="1" from-port="0" to-layer="4" to-port="1" />
		<edge from-layer="2" from-port="0" to-layer="3" to-port="0" />
		<edge from-layer="2" from-port="0" to-layer="4" to-port="0" />
		<edge from-layer="3" from-port="2" to-layer="4" to-port="2" />
		<edge from-layer="4" from-port="4" to-layer="5" to-port="0" />
		<edge from-layer="5" from-port="2" to-layer="6" to-port="0" />
	</edges>
</net>
)V0G0N";


} // namespace FuncTestUtils