// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_reader_tests.hpp"
#include <string>

// MKLDNN: "Crop supports only 2d, 4d and 5d blobs."
// This test should pass after deleting
// "input_shape.size() != 2 && input_shape.size() != 4 && input_shape.size() != 5" condition in
// strided_slice_to_crop transformation
TEST_F(NGraphReaderTests, DISABLED_ConvertStridedSliceToCrop) {
    std::string model_version10 = R"V0G0N(
<net name="Reshape" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter" version="opset1">
            <data element_type="f32" shape="300,90,1,4"/>
            <output>
                <port id="0">
                    <dim>300</dim>
                    <dim>90</dim>
                    <dim>1</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
		<layer id="1" name="Begin" precision="I64" type="Const" version="opset1">
			<data offset="0" size="32"/>
			<output>
				<port id="0">
					<dim>4</dim>
				</port>
			</output>
		</layer>
        <layer id="2" name="End" precision="I64" type="Const" version="opset1">
			<data offset="32" size="32"/>
			<output>
				<port id="0">
					<dim>4</dim>
				</port>
			</output>
		</layer>
        <layer id="3" name="Strides" precision="I64" type="Const" version="opset1">
			<data offset="64" size="32"/>
			<output>
				<port id="0">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="Crop_" precision="FP32" type="StridedSlice" version="opset1">
			<data begin_mask="1,0,1,1" ellipsis_mask="0,0,0,0" end_mask="1,0,1,1" new_axis_mask="0,0,0,0" shrink_axis_mask="0,0,0,0"/>
			<input>
				<port id="0">
					<dim>300</dim>
					<dim>90</dim>
					<dim>1</dim>
					<dim>4</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
				<port id="2">
					<dim>4</dim>
				</port>
				<port id="3">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="4">
					<dim>300</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
        <layer id="5" name="output" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>300</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>4</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
    <edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
    <edge from-layer="1" from-port="0" to-layer="4" to-port="1"/>
    <edge from-layer="2" from-port="0" to-layer="4" to-port="2"/>
    <edge from-layer="3" from-port="0" to-layer="4" to-port="3"/>
    <edge from-layer="4" from-port="4" to-layer="5" to-port="0"/>
    </edges>
    </net>
    )V0G0N";
    std::string model_version6 = R"V0G0N(
<net name="Reshape" version="6" batch="300">
    <layers>
        <layer name="data" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>300</dim>
                    <dim>90</dim>
                    <dim>1</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer name="Crop_" type="Crop" precision="FP32" id="1">
            <data axis="0,1,2,3" dim="300,1,1,4" offset="0,1,0,0" />
            <input>
                <port id="0">
                    <dim>300</dim>
                    <dim>90</dim>
                    <dim>1</dim>
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>300</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
    </edges>
    <statistics />
</net>
)V0G0N";

    compareIRs(model_version10, model_version6, 96, [](Blob::Ptr& weights) {
                auto *data = weights->buffer().as<int64_t *>();

                // According to begin (0,1,0,0) and end masks (0,1,0,0)
                // and input and result shapes (300, 90, 1, 4) -> (300, 1, 1, 4)
                // begin
                data[0] = 0;
                data[1] = 1;
                data[2] = 0;
                data[3] = 0;

                // end
                data[4] = 0;
                data[5] = 2;
                data[6] = 0;
                data[7] = 0;
                // Set "1" into each stride to apply "StrideSliceToCrop" transformation
                for (int stride_node_idx = 8; stride_node_idx < 12; ++stride_node_idx) {
                    data[stride_node_idx] = 1;
                }
            });
}

// MKLDNN: "Crop supports only 2d, 4d and 5d blobs."
// This test should pass after deleting
// "input_shape.size() != 2 && input_shape.size() != 4 && input_shape.size() != 5" condition in
// strided_slice_to_crop transformation
TEST_F(NGraphReaderTests, DISABLED_ConvertStridedSliceToCropMultipleMasks) {

    // c = np.zeros((9, 9, 9, 9, 9, 9, 9))
    // const_use_axis_mask = tf.constant(c)
    // strided_slice_with_mask = tf.strided_slice(const_use_axis_mask,
    //                                           name="OurStridedSlice",
    //                                           begin=[0, 0, 0, 0, 0, 0], end=[ 2, 2, 2, 2, 2, 2],
    // new_axis_mask = 9,
    //        ellipsis_mask= 2,
    //        shrink_axis_mask=4)
    // # (1, 9, 9, 9, 9, 2, 1, 2, 2) without shrink
    // # (1, 9, 9, 9, 9, 1, 2, 2) with shrink
    std::string model_version10 = R"V0G0N(
<net name="Reshape" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter" version="opset1">
            <data element_type="f32" shape="9,9,9,9,9,9,9"/>
            <output>
                <port id="0">
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
            </output>
        </layer>
		<layer id="1" name="Begin" precision="I64" type="Const" version="opset1">
			<data offset="0" size="48"/>
			<output>
				<port id="0">
					<dim>6</dim>
				</port>
			</output>
		</layer>
        <layer id="2" name="End" precision="I64" type="Const" version="opset1">
			<data offset="48" size="48"/>
			<output>
				<port id="0">
					<dim>6</dim>
				</port>
			</output>
		</layer>
        <layer id="3" name="Strides" precision="I64" type="Const" version="opset1">
			<data offset="96" size="48"/>
			<output>
				<port id="0">
					<dim>6</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="Crop_" precision="FP32" type="StridedSlice" version="opset1">
			<data begin_mask="0,0,0,0,0,0,0" ellipsis_mask="0,1,0,0,0,0,0" end_mask="0,0,0,0,0,0,0" new_axis_mask="1,0,0,1,0,0,0" shrink_axis_mask="0,0,1,0,0,0,0"/>
			<input>
				<port id="0">
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
				<port id="1">
					<dim>6</dim>
				</port>
				<port id="2">
					<dim>6</dim>
				</port>
				<port id="3">
					<dim>6</dim>
				</port>
			</input>
			<output>
				<port id="4">
	                <dim>1</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
        <layer id="5" name="output" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
	                <dim>1</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>2</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="4" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="4" to-port="2"/>
        <edge from-layer="3" from-port="0" to-layer="4" to-port="3"/>
        <edge from-layer="4" from-port="4" to-layer="5" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string model_version6 = R"V0G0N(
<net name="Reshape" version="6" batch="9">
	<layers>
		<layer name="data" type="Input" precision="FP32" id="0">
			<output>
				<port id="0">
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</output>
		</layer>
		<layer name="Constant_98" type="Const" precision="I64" id="1">
			<output>
				<port id="0">
					<dim>8</dim>
				</port>
			</output>
			<blobs>
				<custom offset="0" size="64" />
			</blobs>
		</layer>
		<layer name="Constant_95" type="Const" precision="I64" id="2">
			<output>
				<port id="0">
					<dim>9</dim>
				</port>
			</output>
			<blobs>
				<custom offset="64" size="72" />
			</blobs>
		</layer>
		<layer name="slice/DynReshape_before" type="Reshape" precision="FP32" id="3">
			<data dim="" />
			<input>
				<port id="0">
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
				<port id="1">
					<dim>9</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</output>
		</layer>
		<layer name="Crop_" type="Crop" precision="FP32" id="4">
			<data axis="0,1,2,3,4,5,6,7,8" dim="1,9,9,9,9,1,1,2,2" offset="0,0,0,0,0,0,0,0,0" />
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer name="slice/DynReshape_after" type="Reshape" precision="FP32" id="5">
			<data dim="" />
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="3" to-port="0" />
		<edge from-layer="1" from-port="0" to-layer="5" to-port="1" />
		<edge from-layer="2" from-port="0" to-layer="3" to-port="1" />
		<edge from-layer="3" from-port="2" to-layer="4" to-port="0" />
		<edge from-layer="4" from-port="1" to-layer="5" to-port="0" />
	</edges>
	<statistics />
</net>
)V0G0N";


    compareIRs(model_version10, model_version6, 144, [](Blob::Ptr& weights) {
        auto *data = weights->buffer().as<int64_t *>();

        for (int begin_node = 0; begin_node < 6; ++begin_node)
            data[begin_node] = 0;
        for (int end_node = 6; end_node < 12; ++end_node)
            data[end_node] = 2;

        // Set "1" into each stride to apply "StrideSliceToCrop" transformation
        for (int stride_node_idx = 12; stride_node_idx < 18; ++stride_node_idx) {
            data[stride_node_idx] = 1;
        }
    });
}

// TODO delete this check in ngraph "Check 'static_cast<size_t>(data_rank) == mask_size'
TEST_F(NGraphReaderTests, DISABLED_ConvertStridedSliceToCropMultipleMasks_2) {
    std::string model_version10 = R"V0G0N(
<net name="Reshape" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter" version="opset1">
            <data element_type="f32" shape="9,9,9,9,9,9,9"/>
            <output>
                <port id="0">
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
            </output>
        </layer>
		<layer id="1" name="Begin" precision="I64" type="Const" version="opset1">
			<data offset="0" size="64"/>
			<output>
				<port id="0">
					<dim>8</dim>
				</port>
			</output>
		</layer>
        <layer id="2" name="End" precision="I64" type="Const" version="opset1">
			<data offset="64" size="64"/>
			<output>
				<port id="0">
					<dim>8</dim>
				</port>
			</output>
		</layer>
        <layer id="3" name="Strides" precision="I64" type="Const" version="opset1">
			<data offset="128" size="64"/>
			<output>
				<port id="0">
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="Crop_" precision="FP32" type="StridedSlice" version="opset1">
			<data begin_mask="0,0,0,0,0,0,0,0" ellipsis_mask="0,0,0,1,0,0,0,0" end_mask="0,0,0,0,0,0,0,0" new_axis_mask="1,0,1,0,0,1,0,1" shrink_axis_mask="0,0,0,0,0,0,0,0"/>
			<input>
				<port id="0">
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
				<port id="1">
					<dim>8</dim>
				</port>
				<port id="2">
					<dim>8</dim>
				</port>
				<port id="3">
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="4">
	                <dim>1</dim>
					<dim>1</dim>
                    <dim>1</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>2</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
        <layer id="5" name="output" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
	                <dim>1</dim>
					<dim>1</dim>
                    <dim>1</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>2</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>1</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="4" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="4" to-port="2"/>
        <edge from-layer="3" from-port="0" to-layer="4" to-port="3"/>
        <edge from-layer="4" from-port="4" to-layer="5" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string model_version6 = R"V0G0N(
<net name="Reshape" version="6" batch="9">
	<layers>
		<layer name="data" type="Input" precision="FP32" id="0">
			<output>
				<port id="0">
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</output>
		</layer>
		<layer name="Constant_95" type="Const" precision="I64" id="1">
			<output>
				<port id="0">
					<dim>11</dim>
				</port>
			</output>
			<blobs>
				<custom offset="0" size="88" />
			</blobs>
		</layer>
		<layer name="slice/DynReshape_before" type="Reshape" precision="FP32" id="2">
			<data dim="" />
			<input>
				<port id="0">
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
				<port id="1">
					<dim>11</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>9</dim>
					<dim>1</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>1</dim>
					<dim>9</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer name="Crop_" type="Crop" precision="FP32" id="3">
			<data axis="0,1,2,3,4,5,6,7,8,9,10" dim="1,1,1,9,9,9,9,2,1,2,1" offset="0,2,0,0,0,0,0,2,0,2,0" />
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>9</dim>
					<dim>1</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>1</dim>
					<dim>9</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>9</dim>
					<dim>2</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0" />
		<edge from-layer="1" from-port="0" to-layer="2" to-port="1" />
		<edge from-layer="2" from-port="2" to-layer="3" to-port="0" />
	</edges>
	<statistics />
</net>
)V0G0N";

    compareIRs(model_version10, model_version6, 192, [](Blob::Ptr& weights) {
        auto *data = weights->buffer().as<int64_t *>();

        // begin node
        data[0] = 0;
        data[1] = 2;
        data[2] = 0;
        data[3] = 0;
        data[4] = 2;
        data[5] = 0;
        data[6] = 2;
        data[7] = 0;

        // end node
        data[8] = 0;
        data[9] = 3;
        data[10] = 0;
        data[11] = 0;
        data[12] = 4;
        data[13] = 0;
        data[14] = 4;
        data[15] = 0;

        // Set "1" into each stride to apply "StrideSliceToCrop" transformation
        for (int stride_node_idx = 16; stride_node_idx < 24; ++stride_node_idx){
            data[stride_node_idx] = 1;
        }
    });
}
