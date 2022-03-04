// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"

TEST_F(NGraphReaderTests, ReadDeformablePSROIPoolingNetwork_incorrect_mode) {
    std::string model = R"V0G0N(
<net name="DeformablePSROIPooling" version="10">
	<layers>
		<layer id="0" name="detector/bbox/ps_roi_pooling/placeholder_port_0" type="Parameter" version="opset1">
			<data shape="1,3240,38,38" element_type="f32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>3240</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="detector/bbox/ps_roi_pooling/placeholder_port_1" type="Parameter" version="opset1">
			<data shape="100,5" element_type="f32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>100</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="DeformablePSROIPooling" type="DeformablePSROIPooling" version="opset2">
			<data group_size="6" mode="bilinear" no_trans="1" output_dim="360" spatial_bins_x="3" spatial_bins_y="3" spatial_scale="1"/>
			<input>                
				<port id="0">
					<dim>1</dim>
					<dim>3240</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
				<port id="1">
					<dim>100</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>100</dim>
					<dim>360</dim>
					<dim>6</dim>
					<dim>6</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="detector/bbox/ps_roi_pooling/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>100</dim>
					<dim>360</dim>
					<dim>6</dim>
					<dim>6</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
		<edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
	</edges>
</net>
)V0G0N";
    std::string modelV7 = R"V0G0N(
<net name="DeformablePSROIPooling" version="7">
	<layers>
		<layer id="0" name="detector/bbox/ps_roi_pooling/placeholder_port_0" type="Input" version="opset1">
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>3240</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="detector/bbox/ps_roi_pooling/placeholder_port_1" type="Input" version="opset1">
			<output>
				<port id="0" precision="FP32">
					<dim>100</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="DeformablePSROIPooling" type="PSROIPooling" version="opset2">
			<data group_size="6" mode="bilinear" no_trans="1" part_size="1" pooled_height="6" pooled_width="6" trans_std="1" output_dim="360" spatial_bins_x="3" spatial_bins_y="3" spatial_scale="1"/>
			<input>                
				<port id="0">
					<dim>1</dim>
					<dim>3240</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
				<port id="1">
					<dim>100</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>100</dim>
					<dim>360</dim>
					<dim>6</dim>
					<dim>6</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
	</edges>
</net>
)V0G0N";
    compareIRs(model, modelV7);
}

TEST_F(NGraphReaderTests, ReadDeformablePSROIPoolingNetwork) {
    std::string model = R"V0G0N(
<net name="DeformablePSROIPooling" version="10">
	<layers>
		<layer id="0" name="port_0" type="Parameter" version="opset1">
			<data shape="1,3240,38,38" element_type="f32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>3240</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="port_1" type="Parameter" version="opset1">
			<data shape="100,5" element_type="f32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>100</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="DeformablePSROIPooling" type="DeformablePSROIPooling" version="opset1">
		<data group_size="3" mode="bilinear_deformable" no_trans="1" part_size="1" pooled_height="3" pooled_width="3" trans_std="1" output_dim="360" spatial_bins_x="3" spatial_bins_y="3" spatial_scale="1"/>
			<input>                
				<port id="0">
					<dim>1</dim>
					<dim>3240</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
				<port id="1">
					<dim>100</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>100</dim>
					<dim>360</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>100</dim>
					<dim>360</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
		<edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
	</edges>
</net>
)V0G0N";
    std::string modelV7 = R"V0G0N(
<net name="DeformablePSROIPooling" version="7">
	<layers>
		<layer id="0" name="port_0" type="Input" version="opset1">
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>3240</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="port_1" type="Input" version="opset1">
			<output>
				<port id="0" precision="FP32">
					<dim>100</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="DeformablePSROIPooling" type="PSROIPooling" version="opset2">
		<data group_size="3" mode="bilinear_deformable" no_trans="1" part_size="1" pooled_height="3" pooled_width="3" trans_std="1" output_dim="360" spatial_bins_x="3" spatial_bins_y="3" spatial_scale="1"/>
			<input>                
				<port id="0">
					<dim>1</dim>
					<dim>3240</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
				<port id="1">
					<dim>100</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>100</dim>
					<dim>360</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
	</edges>
</net>
)V0G0N";
    compareIRs(model, modelV7);
}
