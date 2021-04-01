// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <single_layer_common.hpp>
#include "test_irs.hpp"

namespace GNATestIRs {

std::string FCOnlyModel() {
    return R"V0G0N(
<Net Name="FullyConnected_Only" version="2" precision="FP32" batch="1">
	<layers>
		<layer name="input_1" type="input" id="0" precision="FP32">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>


        <layer name="FullyConnected" id="1" type="InnerProduct" precision="FP32">

            <fc out-size="10" />

            <biases offset="0" size="40" />
            <weights offset="40" size="400" />

            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>10</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
    </edges>
</Net>
)V0G0N";
}

std::string Fc2DOutputModel() {
    return R"V0G0N(
<Net Name="FullyConnected_Only" version="2" precision="FP32" batch="1">
	<layers>
		<layer name="input_1" type="input" id="0" precision="FP32">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>


        <layer name="FullyConnected" id="1" type="InnerProduct" precision="FP32">

            <fc out-size="10" />

            <biases offset="0" size="40" />
            <weights offset="40" size="400" />

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
        </layer>
    </layers>
    <edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
    </edges>
</Net>
)V0G0N";
}

std::string affineToMemoryModel() {
    return R"V0G0N(
<Net Name="FullyConnected_ToMemory" version="2" precision="FP32" batch="1">
	<layers>
		<layer name="input_1" type="input" id="0" precision="FP32">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>

        <layer name="FullyConnected" id="1" type="InnerProduct" precision="FP32">

            <fc out-size="10" />

            <biases offset="0" size="40" />
            <weights offset="40" size="400" />

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
        </layer>

        <layer name="Eltwise_8" type="Eltwise" id="11" precision="FP32">
			<data operation="sum" />
			<input>
				<port id="0">
					<!--connected to FullyConnected-->
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<!--connected to Memory_28-->
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

        <layer name="Memory_27" type="Memory" id="27" precision="FP32">
			<data id="r_27-28" index="0" size="2" />
			<input>
				<port id="60">
					<!--connected to FullyConnected-->
                    <dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
		</layer>

		<layer name="Memory_28" type="Memory" id="28" precision="FP32">
			<data id="r_27-28" index="1" size="2" />
			<output>
				<port id="59">
					<!--connected to , Eltwise_8-->
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>

    </layers>
    <edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
        <edge from-layer="1" from-port="1" to-layer="27" to-port="60" />
        <edge from-layer="1" from-port="1" to-layer="11" to-port="1" />
        <edge from-layer="28" from-port="59" to-layer="11" to-port="0" />
    </edges>
</Net>
)V0G0N";
}


std::string MemoryAndConcatAfterOneNode() {
    return R"V0G0N(
<net Name="FullyConnected_ToMemory" version="2" precision="FP32" batch="1">
	<layers>
		<layer name="input_1" type="input" id="0" precision="FP32">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>

        <layer name="ReLU1" id="1" type="Activation" precision="FP32">
                <data type="ReLU" negative_slope="0.000000" />
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>32</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>32</dim>
                    </port>
                </output>
            </layer>

		<layer name="input_2" type="memory" id="3" precision="FP32">
			<data id="r_27-28" index="1" size="2" />
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>

        <layer name="ReLU2" id="11" type="Activation" precision="FP32">
                <data type="ReLU" negative_slope="0.000000" />
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>32</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>32</dim>
                    </port>
                </output>
            </layer>

        <layer name="Memory_27" type="Memory" id="2" precision="FP32">
			<data id="r_27-28" index="0" size="2" />
			<input>
				<port id="0">
                    <dim>1</dim>
					<dim>32</dim>
				</port>
			</input>
		</layer>

        <layer name="concat_1" type="Concat" id="4" precision="FP32">
			<input>
				<port id="0">
					<!--connected to FullyConnected-->
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<!--connected to FullyConnected2-->
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>

    </layers>
    <edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0" />
        <edge from-layer="1" from-port="1" to-layer="4" to-port="0" />
        <edge from-layer="3" from-port="0" to-layer="11" to-port="0" />
        <edge from-layer="11" from-port="1" to-layer="4" to-port="1" />
    </edges>
</net>
)V0G0N";
}

std::string MemoryAfterConcatModel() {
    return R"V0G0N(
<net Name="FullyConnected_ToMemory" version="2" precision="FP32" batch="1">
	<layers>
		<layer name="input_1" type="input" id="0" precision="FP32">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>

		<layer name="input_2" type="memory" id="1" precision="FP32">
			<data id="r_27-28" index="1" size="2" />
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>20</dim>
				</port>
			</output>
		</layer>
        <layer name="Crop1" type="Crop" id="11" precision="FP32">
            <data axis="1" dim="10" offset="0"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>20</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>

        <layer name="FullyConnected" id="2" type="InnerProduct" precision="FP32">

            <fc out-size="10" />

            <biases offset="0" size="40" />
            <weights offset="40" size="400" />

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
        </layer>


        <layer name="FullyConnected2" id="3" type="InnerProduct" precision="FP32">

            <fc out-size="10" />

            <biases offset="440" size="40" />
            <weights offset="480" size="400" />

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
        </layer>

        <layer name="Eltwise_8" type="Concat" id="4" precision="FP32">
			<input>
				<port id="0">
					<!--connected to FullyConnected-->
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<!--connected to FullyConnected2-->
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>20</dim>
				</port>
			</output>
		</layer>

        <layer name="reshape_1" type="Reshape" id="5" precision="FP32">
			<input>
				<port id="0">
					<!--connected to concat-->
                    <dim>1</dim>
					<dim>20</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<!--connected to memory-->
                    <dim>1</dim>
					<dim>20</dim>
				</port>
			</output>
		</layer>

        <layer name="reshape_2" type="Reshape" id="7" precision="FP32">
			<input>
				<port id="0">
					<!--connected to concat-->
                    <dim>1</dim>
					<dim>20</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<!--connected to memory-->
                    <dim>1</dim>
					<dim>20</dim>
				</port>
			</output>
		</layer>

        <layer name="Memory_27" type="Memory" id="6" precision="FP32">
			<data id="r_27-28" index="0" size="2" />
			<input>
				<port id="0">
					<!--connected to concat-->
                    <dim>1</dim>
					<dim>20</dim>
				</port>
			</input>
		</layer>
    </layers>
    <edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0" />
        <edge from-layer="1" from-port="0" to-layer="11" to-port="0" />
        <edge from-layer="11" from-port="1" to-layer="3" to-port="0" />
        <edge from-layer="2" from-port="1" to-layer="4" to-port="0" />
        <edge from-layer="3" from-port="1" to-layer="4" to-port="1" />
        <edge from-layer="4" from-port="2" to-layer="5" to-port="0" />
        <edge from-layer="5" from-port="1" to-layer="6" to-port="0" />
        <edge from-layer="5" from-port="1" to-layer="7" to-port="0" />
    </edges>
</net>
)V0G0N";
}

std::string eltwiseToMemoryModelNoOutput() {
    return R"V0G0N(
<Net Name="FullyConnected_ToMemory" version="2" precision="FP32" batch="1">
	<layers>
		<layer name="input_1" type="input" id="0" precision="FP32">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>


        <layer name="Eltwise_8" type="Eltwise" id="11" precision="FP32">
			<data operation="sum" />
			<input>
				<port id="0">
					<!--connected to FullyConnected-->
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<!--connected to Memory_28-->
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


        <layer name="Memory_27" type="Memory" id="27" precision="FP32">
			<data id="r_27-28" index="0" size="2" />
			<input>
				<port id="60">
					<!--connected to Eltwise_8-->
                    <dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
		</layer>

		<layer name="Memory_28" type="Memory" id="28" precision="FP32">
			<data id="r_27-28" index="1" size="2" />
			<output>
				<port id="59">
					<!--connected to , Eltwise_8-->
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>

    </layers>
    <edges>
		<edge from-layer="0" from-port="0" to-layer="11" to-port="1" />
        <edge from-layer="11" from-port="2" to-layer="27" to-port="60" />
        <edge from-layer="28" from-port="59" to-layer="11" to-port="0" />
    </edges>
</Net>
)V0G0N";
}
std::string eltwiseToMemoryModel() {
    return R"V0G0N(
<Net Name="FullyConnected_ToMemory" version="2" precision="FP32" batch="1">
	<layers>
		<layer name="input_1" type="input" id="0" precision="FP32">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
        <layer name="Eltwise_8" type="Eltwise" id="11" precision="FP32">
			<data operation="sum" />
			<input>
				<port id="0">
					<!--connected to Memory_28-->
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<!--connected to input-->
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

        <layer name="Eltwise_9" type="Eltwise" id="12" precision="FP32">
			<data operation="sum" />
			<input>
				<port id="0">
					<!--connected Memory_28 to -->
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<!--connected to Elwise_8-->
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

        <layer name="Memory_27" type="Memory" id="27" precision="FP32">
			<data id="r_27-28" index="0" size="2" />
			<input>
				<port id="60">
					<!--connected to Eltwise_8-->
                    <dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
		</layer>

		<layer name="Memory_28" type="Memory" id="28" precision="FP32">
			<data id="r_27-28" index="1" size="2" />
			<output>
				<port id="59">
					<!--connected to , Eltwise_8-->
					<dim>1</dim>
					<dim>10</dim>
				</port>
                <port id="5010">
					<!--connected to , Eltwise_9-->
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>

    </layers>
    <edges>
		<edge from-layer="0" from-port="0" to-layer="11" to-port="1" />
        <edge from-layer="11" from-port="2" to-layer="27" to-port="60" />
        <edge from-layer="11" from-port="2" to-layer="12" to-port="1" />
        <edge from-layer="28" from-port="59" to-layer="11" to-port="0" />
        <edge from-layer="28" from-port="5010" to-layer="12" to-port="0" />
    </edges>
</Net>
)V0G0N";
}

std::string activationAfterSplitModel() {
    return R"V0G0N(
    <net Name="activationAfterSplit" version="2" precision="FP32" batch="1">
        <layers>
            <layer name="input_1" type="input" id="0" precision="FP32">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </output>
            </layer>
            <layer name="Eltwise_8" type="Eltwise" id="11" precision="FP32">
                <data operation="sum" />
                <input>
                    <port id="0">
                        <!--connected to split-->
                        <dim>1</dim>
                        <dim>10</dim>
                    </port>
                    <port id="1">
                        <!--connected to tanh_28-->
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

            <layer name="Split_1" type="Split" id="12" precision="FP32">
                <data axis="1" />
                <input>
                    <port id="0">
                        <!--connected to input-->
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <!--connected to tanh-->
                        <dim>1</dim>
                        <dim>10</dim>
                    </port>
                    <port id="2">
                        <!--connected to eltwise-->
                        <dim>1</dim>
                        <dim>10</dim>
                    </port>
                </output>
            </layer>
            <layer name="Activation_38" type="Activation" id="38" precision="FP32">
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
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="12" to-port="0" />
            <edge from-layer="12" from-port="1" to-layer="11" to-port="0" />
            <edge from-layer="12" from-port="2" to-layer="38" to-port="82" />
            <edge from-layer="38" from-port="83" to-layer="11" to-port="1" />
        </edges>
    </net>
    )V0G0N";
}

std::string FCWithPaddingAfterSplitModel() {
    return R"V0G0N(
    <Net Name="FCWithPaddingAfterSplitModel" version="2" precision="FP32" batch="1">
        <layers>
            <layer name="input_1" type="input" id="0" precision="FP32">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </output>
            </layer>
            <layer name="Split_1" type="Split" id="1" precision="FP32">
                <data axis="1" />
                <input>
                    <port id="0">
                        <!--connected to input-->
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <!--connected to eltwise-->
                        <dim>1</dim>
                        <dim>10</dim>
                    </port>
                    <port id="2">
                        <!--connected to fc-->
                        <dim>1</dim>
                        <dim>10</dim>
                    </port>
                </output>
            </layer>
            <layer name="FullyConnected" id="11" type="InnerProduct" precision="FP32">
                <fc out-size="10" />
                <biases offset="0" size="40" />
                <weights offset="40" size="400" />
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
            </layer>
            <layer name="Eltwise_8" type="Eltwise" id="21" precision="FP32">
                <data operation="sum" />
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>10</dim>
                    </port>
                    <port id="1">
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
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
            <edge from-layer="1" from-port="1" to-layer="21" to-port="0" />
            <edge from-layer="1" from-port="2" to-layer="11" to-port="0" />
            <edge from-layer="11" from-port="1" to-layer="21" to-port="1" />
        </edges>
    </Net>
    )V0G0N";
}

std::string FCBeforeSplitModel() {
    return R"V0G0N(
    <Net Name="FCBeforeSplitModel" version="2" precision="FP32" batch="1">
        <layers>
            <layer name="input_1" type="input" id="0" precision="FP32">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </output>
            </layer>
            <layer name="FullyConnected_1" id="1" type="InnerProduct" precision="FP32">
                <fc out-size="20" />
                <biases offset="0" size="80" />
                <weights offset="80" size="1600" />
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </output>
            </layer>
            <layer name="Split_1" type="Split" id="2" precision="FP32">
                <data axis="1" />
                <input>
                    <port id="0">
                        <!--connected to input-->
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <!--connected to eltwise-->
                        <dim>1</dim>
                        <dim>10</dim>
                    </port>
                    <port id="2">
                        <!--connected to fc-->
                        <dim>1</dim>
                        <dim>10</dim>
                    </port>
                </output>
            </layer>
            <layer name="FullyConnected_2" id="11" type="InnerProduct" precision="FP32">
                <fc out-size="10" />
                <biases offset="1600" size="40" />
                <weights offset="1640" size="400" />
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
            </layer>
            <layer name="Eltwise_8" type="Eltwise" id="21" precision="FP32">
                <data operation="sum" />
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>10</dim>
                    </port>
                    <port id="1">
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
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
            <edge from-layer="1" from-port="1" to-layer="2" to-port="0" />
            <edge from-layer="2" from-port="1" to-layer="21" to-port="0" />
            <edge from-layer="2" from-port="2" to-layer="11" to-port="0" />
            <edge from-layer="11" from-port="1" to-layer="21" to-port="1" />
        </edges>
    </Net>
    )V0G0N";
}
std::string twoFCWithPaddingAfterSliceModel() {
    return R"V0G0N(
    <Net Name="twoFCWithPaddingAfterSliceModel" version="2" precision="FP32" batch="1">
        <layers>
            <layer name="input_1" type="input" id="0" precision="FP32">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </output>
            </layer>
            <layer name="Slice_1" type="Slice" id="1" precision="FP32">
                <data axis="1" slice_point="8" slice_dim="1"/>
                <input>
                    <port id="0">
                        <!--connected to input-->
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <!--connected to eltwise-->
                        <dim>1</dim>
                        <dim>8</dim>
                    </port>
                    <port id="2">
                        <!--connected to fc-->
                        <dim>1</dim>
                        <dim>12</dim>
                    </port>
                </output>
            </layer>
            <layer name="FullyConnected1" id="11" type="InnerProduct" precision="FP32">
                <fc out-size="8" />
                <biases offset="0" size="32" />
                <weights offset="32" size="384" />
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>12</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>8</dim>
                    </port>
                </output>
            </layer>
            <layer name="FullyConnected2" id="12" type="InnerProduct" precision="FP32">
                <fc out-size="8" />
                <biases offset="0" size="32" />
                <weights offset="32" size="384" />
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>12</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>8</dim>
                    </port>
                </output>
            </layer>
            <layer name="Eltwise_1" type="Eltwise" id="21" precision="FP32">
                <data operation="sum" />
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>8</dim>
                    </port>
                    <port id="1">
                        <dim>1</dim>
                        <dim>8</dim>
                    </port>
                </input>
                <output>
                    <port id="2">
                        <dim>1</dim>
                        <dim>8</dim>
                    </port>
                </output>
            </layer>
            <layer name="Eltwise_2" type="Eltwise" id="22" precision="FP32">
                <data operation="sum" />
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>8</dim>
                    </port>
                    <port id="1">
                        <dim>1</dim>
                        <dim>8</dim>
                    </port>
                </input>
                <output>
                    <port id="2">
                        <dim>1</dim>
                        <dim>8</dim>
                    </port>
                </output>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
            <edge from-layer="1" from-port="1" to-layer="21" to-port="0" />
            <edge from-layer="1" from-port="2" to-layer="11" to-port="0" />
            <edge from-layer="1" from-port="2" to-layer="12" to-port="0" />
            <edge from-layer="11" from-port="1" to-layer="21" to-port="1" />
            <edge from-layer="21" from-port="2" to-layer="22" to-port="0" />
            <edge from-layer="12" from-port="1" to-layer="22" to-port="1" />
        </edges>
    </Net>
    )V0G0N";
}

std::string FCWithPaddingAfterSliceModel() {
    return R"V0G0N(
    <Net Name="FCWithPaddingAfterSliceModel" version="2" precision="FP32" batch="1">
        <layers>
            <layer name="input_1" type="input" id="0" precision="FP32">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </output>
            </layer>
            <layer name="Slice_1" type="Slice" id="1" precision="FP32">
                <data axis="1" slice_point="8" slice_dim="1"/>
                <input>
                    <port id="0">
                        <!--connected to input-->
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <!--connected to eltwise-->
                        <dim>1</dim>
                        <dim>8</dim>
                    </port>
                    <port id="2">
                        <!--connected to fc-->
                        <dim>1</dim>
                        <dim>12</dim>
                    </port>
                </output>
            </layer>
            <layer name="FullyConnected" id="11" type="InnerProduct" precision="FP32">
                <fc out-size="8" />
                <biases offset="0" size="32" />
                <weights offset="32" size="384" />
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>12</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>8</dim>
                    </port>
                </output>
            </layer>
            <layer name="Eltwise_8" type="Eltwise" id="21" precision="FP32">
                <data operation="sum" />
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>8</dim>
                    </port>
                    <port id="1">
                        <dim>1</dim>
                        <dim>8</dim>
                    </port>
                </input>
                <output>
                    <port id="2">
                        <dim>1</dim>
                        <dim>8</dim>
                    </port>
                </output>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
            <edge from-layer="1" from-port="1" to-layer="21" to-port="0" />
            <edge from-layer="1" from-port="2" to-layer="11" to-port="0" />
            <edge from-layer="11" from-port="1" to-layer="21" to-port="1" />
        </edges>
    </Net>
    )V0G0N";
}

std::string SliceModelWithAlignedOutputs() {
    return R"V0G0N(
    <Net Name="SliceModelWithAlignedOutputs" version="2" precision="FP32" batch="1">
        <layers>
            <layer name="input_1" type="input" id="0" precision="FP32">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </output>
            </layer>
            <layer name="Slice_1" type="Slice" id="1" precision="FP32">
                <data axis="1" slice_point="8" slice_dim="1"/>
                <input>
                    <port id="0">
                        <!--connected to input-->
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <!--connected to fc-->
                        <dim>1</dim>
                        <dim>16</dim>
                    </port>
                    <port id="2">
                        <!--connected to eltwise-->
                        <dim>1</dim>
                        <dim>4</dim>
                    </port>
                </output>
            </layer>
            <layer name="FullyConnected" id="11" type="InnerProduct" precision="FP32">
                <fc out-size="4" />
                <biases offset="0" size="16" />
                <weights offset="16" size="512" />
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>16</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>4</dim>
                    </port>
                </output>
            </layer>
            <layer name="Eltwise_8" type="Eltwise" id="21" precision="FP32">
                <data operation="sum" />
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>4</dim>
                    </port>
                    <port id="1">
                        <dim>1</dim>
                        <dim>4</dim>
                    </port>
                </input>
                <output>
                    <port id="2">
                        <dim>1</dim>
                        <dim>4</dim>
                    </port>
                </output>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
            <edge from-layer="1" from-port="1" to-layer="11" to-port="0" />
            <edge from-layer="1" from-port="2" to-layer="21" to-port="0" />
            <edge from-layer="11" from-port="1" to-layer="21" to-port="1" />
        </edges>
    </Net>
    )V0G0N";
}

std::string eltwiseSummModel()  {
    return R"V0G0N(
    <Net Name="activationAfterSplit" version="2" precision="FP32" batch="1">
        <layers>
            <layer name="input_1" type="input" id="0" precision="FP32">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>10</dim>
                    </port>
                </output>
            </layer>
            <layer name="FullyConnected" id="2" type="InnerProduct" precision="FP32">

                <fc out-size="10" />

                <biases offset="0" size="40" />
                <weights offset="40" size="400" />

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
            </layer>

        <layer name="FullyConnected_1" id="3" type="InnerProduct" precision="FP32">

            <fc out-size="10" />

            <biases offset="0" size="40" />
            <weights offset="40" size="400" />

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
        </layer>

            <layer name="Eltwise_8" type="Eltwise" id="11" precision="FP32">
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
            <edge from-layer="0" from-port="0" to-layer="3" to-port="0" />
            <edge from-layer="2" from-port="1" to-layer="11" to-port="0" />
            <edge from-layer="3" from-port="1" to-layer="11" to-port="1" />
        </edges>
    </Net>
    )V0G0N";
}


std::string eltwiseMulModel()  {
    return R"V0G0N(
    <Net Name="eltwiseMul" version="2" precision="FP32" batch="1">
        <layers>
            <layer name="input_1" type="input" id="0" precision="FP32">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>10</dim>
                    </port>
                </output>
            </layer>
            <layer name="FullyConnected" id="2" type="InnerProduct" precision="FP32">

                <fc out-size="10" />

                <biases offset="0" size="40" />
                <weights offset="40" size="400" />

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
            </layer>

        <layer name="FullyConnected_1" id="3" type="InnerProduct" precision="FP32">

            <fc out-size="10" />

            <biases offset="0" size="40" />
            <weights offset="40" size="400" />

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
        </layer>

            <layer name="Eltwise_8" type="Eltwise" id="11" precision="FP32">
                <data operation="mul" />
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
            <edge from-layer="0" from-port="0" to-layer="3" to-port="0" />
            <edge from-layer="2" from-port="1" to-layer="11" to-port="0" />
            <edge from-layer="3" from-port="1" to-layer="11" to-port="1" />
        </edges>
    </Net>
    )V0G0N";
}

std::string scaleShiftAffineModel() {
    return R"V0G0N(
<Net Name="FullyConnected_Only" version="2" precision="FP32" batch="1">
	<layers>
		<layer name="input_1" type="input" id="0" precision="FP32">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>

        <layer name="ScaleShift_21" type="ScaleShift" id="21" precision="FP32">
			<input>
				<port id="46">
					<!--connected to input-->
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="45">
					<!--connected to , FullyConnected-->
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
			<weights offset="0" size="40" precision="FP32" />
		</layer>

        <layer name="FullyConnected" id="1" type="InnerProduct" precision="FP32">

            <fc out-size="10" />

            <biases offset="0" size="40" />
            <weights offset="40" size="400" />

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
        </layer>
    </layers>
    <edges>
		<edge from-layer="0" from-port="0" to-layer="21" to-port="46" />
        <edge from-layer="21" from-port="45" to-layer="1" to-port="0" />
    </edges>
</Net>
)V0G0N";

}

std::string clampFollowedByTanhModel() {
    return R"V0G0N(
<Net Name="clampFollowedByTanhModel" version="2" precision="FP32" batch="1">
	<layers>
		<layer name="input_1" type="input" id="0" precision="FP32">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>

        <layer name="Clamp_20" type="Clamp" id="20" precision="FP32">
			<data max="50" min="-50" />
			<input>
				<port id="43">
					<!--connected to Eltwise_19-->
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="44">
					<!--connected to , ScaleShift_21, Activation_24, Memory_4-->
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>

        <layer name="Activation_38" type="Activation" id="38" precision="FP32">
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


    </layers>
    <edges>
		<edge from-layer="0" from-port="0" to-layer="20" to-port="43" />
        <edge from-layer="20" from-port="44" to-layer="38" to-port="82" />
    </edges>
</Net>
)V0G0N";

}

std::string eltwiseWithMemoryAndActivationInputModel() {
    return R"V0G0N(
    <Net Name="activationAfterSplit" version="2" precision="FP32" batch="1">
        <layers>
            <layer name="input_1" type="input" id="0" precision="FP32">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>10</dim>
                    </port>
                </output>
            </layer>

        <layer name="Memory_27" type="Memory" id="27" precision="FP32">
			<data id="r_27-28" index="0" size="2" />
			<input>
				<port id="60">
					<!--connected to Activation_38-->
                    <dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
		</layer>

		<layer name="Memory_28" type="Memory" id="28" precision="FP32">
			<data id="r_27-28" index="1" size="2" />
			<output>
				<port id="59">
					<!--connected to , Eltwise_8-->
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>

            <layer name="FullyConnected" id="2" type="InnerProduct" precision="FP32">

                <fc out-size="10" />

                <biases offset="0" size="40" />
                <weights offset="40" size="400" />

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
            </layer>

            <layer name="Activation_38" type="Activation" id="38" precision="FP32">
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

            <layer name="Eltwise_8" type="Eltwise" id="11" precision="FP32">
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
    </Net>
    )V0G0N";

}
std::string AffineWith2AffineOutputsModel() {
    return R"V0G0N(
    <Net Name="eltwiseMul" version="2" precision="FP32" batch="1">
        <layers>
            <layer name="input_1" type="input" id="0" precision="FP32">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>10</dim>
                    </port>
                </output>
            </layer>
            <layer name="FullyConnected" id="2" type="InnerProduct" precision="FP32">

                <fc out-size="10" />

                <biases offset="0" size="40" />
                <weights offset="40" size="400" />

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
            </layer>

        <layer name="FullyConnected_1" id="3" type="InnerProduct" precision="FP32">

            <fc out-size="10" />

            <biases offset="0" size="40" />
            <weights offset="40" size="400" />

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
        </layer>

        <layer name="FullyConnected_5" id="4" type="InnerProduct" precision="FP32">

            <fc out-size="10" />

            <biases offset="0" size="40" />
            <weights offset="40" size="400" />

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
        </layer>
        <layer name="Eltwise_8" type="Eltwise" id="11" precision="FP32">
			<data operation="sum" />
			<input>
				<port id="0">
					<!--connected to FullyConnected-->
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<!--connected to Memory_28-->
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
            <edge from-layer="2" from-port="1" to-layer="3" to-port="0" />
            <edge from-layer="2" from-port="1" to-layer="4" to-port="0" />
            <edge from-layer="4" from-port="1" to-layer="11" to-port="0" />
            <edge from-layer="3" from-port="1" to-layer="11" to-port="1" />
        </edges>
    </Net>
    )V0G0N";

}

std::string SigmoidActivationModel() {
    return R"V0G0N(
<Net Name="InputLayerWithSigmoidActivation" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="input_1" type="input" id="0" precision="FP32">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer name="Sig_Activation" id="2" type="Activation" precision="FP32">
            <data type="sigmoid" />
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
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0" />
    </edges>
</Net>
)V0G0N";
}

std::string TanhActivationModel() {
    return R"V0G0N(
<Net Name="InputLayerWithTanhActivation" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="input_1" type="input" id="0" precision="FP32">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer name="Tanh_Activation" id="2" type="Activation" precision="FP32">
            <data type="tanh" />
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
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0" />
    </edges>
</Net>
)V0G0N";
}

std::string ReLUActivationModel() {
    return R"V0G0N(
<Net Name="InputLayerWithReLUActivation" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="input_1" type="input" id="0" precision="FP32">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>10</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer name="ReLU_Activation" type="Activation" id="2" precision="FP32">
            <data type="ReLU" negative_slope="0.000000" />
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>10</dim>
                    <dim>10</dim>
                </port>
            </input>
            <output>
                <port id="4">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>10</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0" />
    </edges>
</Net>
)V0G0N";
}

std::string LeakyReLUActivationModel() {
    return R"V0G0N(
<Net Name="InputLayerWithLeakyReLUActivation" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="input_1" type="input" id="0" precision="FP32">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>10</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer name="LeakyReLU_Activation" type="Activation" id="2" precision="FP32">
            <data type="ReLU" negative_slope="0.010000" />
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>10</dim>
                    <dim>10</dim>
                </port>
            </input>
            <output>
                <port id="4">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>10</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0" />
    </edges>
</Net>
)V0G0N";
}

std::string ClampActivationModel() {
    return R"V0G0N(
<Net Name="InputLayerWithClippingActivation" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="input_1" type="input" id="0" precision="FP32">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer name="Clamp_Activation" id="2" type="Activation" precision="FP32">
            <data type="clamp" min="-50" max="50" />
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
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0" />
    </edges>
</Net>
)V0G0N";
}

std::string IdentityActivationModel() {
    return R"V0G0N(
<Net Name="InputLayerWithIdentityActivation" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="input_1" type="input" id="0" precision="FP32">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer name="Identity_Activation" id="2" type="Activation" precision="FP32">
            <data type="identity" />
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
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0" />
    </edges>
</Net>
)V0G0N";
}

std::string concatModel()  {
    return R"V0G0N(
    <Net Name="concatinationModel" version="2" precision="FP32" batch="1">
        <layers>
            <layer name="input1" type="input" id="0" precision="FP32">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </output>
            </layer>
            <layer name="Split1" type="Split" id="1" precision="FP32">
                <data axis="1" />
                <input>
                    <port id="0">
                        <!--connected to input-->
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <!--connected to eltwise-->
                        <dim>1</dim>
                        <dim>10</dim>
                    </port>
                    <port id="2">
                        <!--connected to fc-->
                        <dim>1</dim>
                        <dim>10</dim>
                    </port>
                </output>
            </layer>
           <layer name="ReLU1" id="11" type="Activation" precision="FP32">
                <data type="ReLU" negative_slope="0.000000" />
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
            </layer>
            <layer name="FullyConnected1" id="12" type="InnerProduct" precision="FP32">
                <fc out-size="10" />
                <biases offset="0" size="40" />
                <weights offset="40" size="400" />
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
            </layer>
            <layer name="concat1" id="21"  precision="FP32" type="Concat">
                    <data axis="1"/>
                    <input>
                            <port id="0">
                                    <dim>1</dim>
                                    <dim>10</dim>
                            </port>
                            <port id="1">
                                    <dim>1</dim>
                                    <dim>10</dim>
                            </port>
                    </input>
                    <output>
                            <port id="2">
                                    <dim>1</dim>
                                    <dim>20</dim>
                            </port>
                    </output>
            </layer>
            <layer name="FullyConnected2" id="31" type="InnerProduct" precision="FP32">
                <fc out-size="20" />
                <biases offset="0" size="80" />
                <weights offset="80" size="1600" />
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </output>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
            <edge from-layer="1" from-port="1" to-layer="11" to-port="0" />
            <edge from-layer="1" from-port="2" to-layer="12" to-port="0" />
            <edge from-layer="11" from-port="1" to-layer="21" to-port="0" />
            <edge from-layer="12" from-port="1" to-layer="21" to-port="1" />
            <edge from-layer="21" from-port="2" to-layer="31" to-port="0" />
        </edges>
    </Net>
    )V0G0N";
}
std::string TFLeakyReluModel() {
    return R"V0G0N(
    <?xml version="1.0" ?>
    <net batch="1" name="model" version="2">
        <layers>
            <layer id="0" name="Placeholder" precision="FP32" type="Input">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>128</dim>
                        <dim>1</dim>
                        <dim>126</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="conv1_node/Conv2D" precision="FP32" type="Convolution">
                <data dilation-x="1" dilation-y="1" group="1" kernel-x="5" kernel-y="1" output="128" pad-x="0" pad-y="0" stride="1,1,1,1" stride-x="1" stride-y="1"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>128</dim>
                        <dim>1</dim>
                        <dim>126</dim>
                    </port>
                </input>
                <output>
                    <port id="3">
                        <dim>1</dim>
                        <dim>128</dim>
                        <dim>1</dim>
                        <dim>122</dim>
                    </port>
                </output>
                <blobs>
                    <weights offset="0" size="327680"/>
                    <biases offset="327680" size="512"/>
                </blobs>
            </layer>
            <layer id="2" name="conv1_node/Relu" precision="FP32" type="ReLU">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>128</dim>
                        <dim>1</dim>
                        <dim>122</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>128</dim>
                        <dim>1</dim>
                        <dim>122</dim>
                    </port>
                </output>
            </layer>
            <layer id="3" name="conv1_node/Neg" precision="FP32" type="Power">
                <data power="1" scale="-1" shift="0"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>128</dim>
                        <dim>1</dim>
                        <dim>122</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>128</dim>
                        <dim>1</dim>
                        <dim>122</dim>
                    </port>
                </output>
            </layer>
            <layer id="4" name="conv1_node/Relu_1" precision="FP32" type="ReLU">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>128</dim>
                        <dim>1</dim>
                        <dim>122</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>128</dim>
                        <dim>1</dim>
                        <dim>122</dim>
                    </port>
                </output>
            </layer>
            <layer id="5" name="conv1_node/mul" precision="FP32" type="Power">
                <data power="1" scale="0.20000000298023224" shift="0"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>128</dim>
                        <dim>1</dim>
                        <dim>122</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>128</dim>
                        <dim>1</dim>
                        <dim>122</dim>
                    </port>
                </output>
            </layer>
            <layer id="47" name="conv1_node/sub/negate_86" precision="FP32" type="Power">
                <data power="1" scale="-1" shift="0"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>128</dim>
                        <dim>1</dim>
                        <dim>122</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>128</dim>
                        <dim>1</dim>
                        <dim>122</dim>
                    </port>
                </output>
            </layer>
            <layer id="48" name="conv1_node/sub/add_87" precision="FP32" type="Eltwise">
                <data operation="sum"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>128</dim>
                        <dim>1</dim>
                        <dim>122</dim>
                    </port>
                    <port id="1">
                        <dim>1</dim>
                        <dim>128</dim>
                        <dim>1</dim>
                        <dim>122</dim>
                    </port>
                </input>
                <output>
                    <port id="2">
                        <dim>1</dim>
                        <dim>128</dim>
                        <dim>1</dim>
                        <dim>122</dim>
                    </port>
                </output>
            </layer>
        </layers>
        <edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="3" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="3" to-layer="3" to-port="0"/>
		<edge from-layer="3" from-port="1" to-layer="4" to-port="0"/>
		<edge from-layer="4" from-port="1" to-layer="5" to-port="0"/>
		<edge from-layer="5" from-port="1" to-layer="47" to-port="0"/>
		<edge from-layer="2" from-port="1" to-layer="48" to-port="0"/>
		<edge from-layer="47" from-port="1" to-layer="48" to-port="1"/>
        </edges>
    </net>
    )V0G0N";
}

std::string TFSoftsignUnfoldedModel() {
    return R"V0G0N(
<?xml version="1.0" ?>
<net name="LSTM_statics_1000_1frame_test_unpack" version="7">
<layers>
    <layer id="0" name="input_batch" type="Input">
        <output>
            <port id="0" precision="FP32">
                <dim>1</dim>
                <dim>64</dim>
            </port>
        </output>
    </layer>
	<layer id="42" name="FC_layer0/packed/ExpandDims_/Dims/Output_0/Data__const" type="Const">
		<output>
			<port id="1" precision="I32">
				<dim>1</dim>
			</port>
		</output>
		<blobs>
			<custom offset="0" precision="I32" size="4"/>
		</blobs>
	</layer>
    <layer id="43" name="FC_layer0/packed/ExpandDims_131" type="Unsqueeze">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>64</dim>
                </port>
            </output>
        </layer>
        <layer id="44" name="abs_FC_layer0/Softsign" type="Abs">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>64</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>64</dim>
                </port>
            </output>
        </layer>
        <layer id="45" name="FC_layer0/Softsign_plus_1/fused_power" type="Power">
            <data power="-1.0" scale="1" shift="1.0"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>64</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>64</dim>
                </port>
            </output>
        </layer>
        <layer id="46" name="div_FC_layer0/Softsign/mul_" type="Eltwise">
            <data operation="mul"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>64</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>64</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>64</dim>
                </port>
            </output>
        </layer>
        <layer id="47" name="unstack_6/Squeeze_/value/Output_0/Data__const" type="Const">
            <output>
                <port id="1" precision="I32">
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="4" precision="I32" size="4"/>
            </blobs>
        </layer>
        <layer id="48" name="unstack_6/Squeeze_" type="Squeeze">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>64</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
	<edge from-layer="0" from-port="0" to-layer="43" to-port="0"/>
	<edge from-layer="42" from-port="1" to-layer="43" to-port="1"/>
	<edge from-layer="43" from-port="2" to-layer="44" to-port="0"/>
	<edge from-layer="44" from-port="1" to-layer="45" to-port="0"/>
	<edge from-layer="45" from-port="1" to-layer="46" to-port="0"/>
	<edge from-layer="43" from-port="2" to-layer="46" to-port="1"/>
	<edge from-layer="46" from-port="2" to-layer="48" to-port="0"/>
    <edge from-layer="47" from-port="1" to-layer="48" to-port="1"/>
    </edges>
</net>

  )V0G0N";
}

std::string maxpoolAfterRelu() {
    return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model" version="2">
	<layers>
		<layer id="0" name="Placeholder" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>126</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="conv1_node/Conv2D" precision="FP32" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="5" kernel-y="1" output="128" pad-x="0" pad-y="0" stride="1,1,1,1" stride-x="1" stride-y="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>126</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>122</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="327680"/>
				<biases offset="327680" size="512"/>
			</blobs>
		</layer>
		<layer id="2" name="conv1_node/Relu" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>122</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>122</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="conv1_node/Neg" precision="FP32" type="Power">
			<data power="1" scale="-1" shift="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>122</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>122</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="conv1_node/Relu_1" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>122</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>122</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="conv1_node/mul" precision="FP32" type="Power">
			<data power="1" scale="0.20000000298023224" shift="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>122</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>122</dim>
				</port>
			</output>
		</layer>
		<layer id="47" name="conv1_node/sub/negate_86" precision="FP32" type="Power">
			<data power="1" scale="-1" shift="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>122</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>122</dim>
				</port>
			</output>
		</layer>
		<layer id="48" name="conv1_node/sub/add_87" precision="FP32" type="Eltwise">
			<data operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>122</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>122</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>122</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="conv2_node/Conv2D" precision="FP32" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="5" kernel-y="1" output="128" pad-x="0" pad-y="0" stride="1,1,1,1" stride-x="1" stride-y="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>122</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>118</dim>
				</port>
			</output>
			<blobs>
				<weights offset="328192" size="327680"/>
				<biases offset="655872" size="512"/>
			</blobs>
		</layer>
		<layer id="7" name="conv2_node/Relu" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>118</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>118</dim>
				</port>
			</output>
		</layer>
		<layer id="8" name="conv2_node/Neg" precision="FP32" type="Power">
			<data power="1" scale="-1" shift="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>118</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>118</dim>
				</port>
			</output>
		</layer>
		<layer id="9" name="conv2_node/Relu_1" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>118</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>118</dim>
				</port>
			</output>
		</layer>
		<layer id="10" name="conv2_node/mul" precision="FP32" type="Power">
			<data power="1" scale="0.20000000298023224" shift="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>118</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>118</dim>
				</port>
			</output>
		</layer>
		<layer id="53" name="conv2_node/sub/negate_92" precision="FP32" type="Power">
			<data power="1" scale="-1" shift="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>118</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>118</dim>
				</port>
			</output>
		</layer>
		<layer id="54" name="conv2_node/sub/add_93" precision="FP32" type="Eltwise">
			<data operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>118</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>118</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>118</dim>
				</port>
			</output>
		</layer>
		<layer id="11" name="pool1_node/MaxPool" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel-x="2" kernel-y="1" pad-x="0" pad-y="0" pool-method="max" stride="1,1,1,2" stride-x="2" stride-y="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>118</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>59</dim>
				</port>
			</output>
		</layer>
        </layers>
        <edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="3" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="3" to-layer="3" to-port="0"/>
		<edge from-layer="3" from-port="1" to-layer="4" to-port="0"/>
		<edge from-layer="4" from-port="1" to-layer="5" to-port="0"/>
		<edge from-layer="5" from-port="1" to-layer="47" to-port="0"/>
		<edge from-layer="2" from-port="1" to-layer="48" to-port="0"/>
		<edge from-layer="47" from-port="1" to-layer="48" to-port="1"/>
		<edge from-layer="48" from-port="2" to-layer="6" to-port="0"/>
		<edge from-layer="6" from-port="3" to-layer="7" to-port="0"/>
		<edge from-layer="6" from-port="3" to-layer="8" to-port="0"/>
		<edge from-layer="8" from-port="1" to-layer="9" to-port="0"/>
		<edge from-layer="9" from-port="1" to-layer="10" to-port="0"/>
		<edge from-layer="10" from-port="1" to-layer="53" to-port="0"/>
		<edge from-layer="7" from-port="1" to-layer="54" to-port="0"/>
		<edge from-layer="53" from-port="1" to-layer="54" to-port="1"/>
		<edge from-layer="54" from-port="2" to-layer="11" to-port="0"/>
        </edges>
    </net>

    )V0G0N";
}

std::string doubleConcatModel() {
    return R"V0G0N(
    <Net Name="concatinationModel" version="2" precision="FP32" batch="1">
        <layers>
            <layer name="input_1" type="input" id="0" precision="FP32">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>40</dim>
                    </port>
                </output>
            </layer>
            <layer name="Split1" type="Split" id="1" precision="FP32">
                <data axis="1" />
                <input>
                    <port id="0">
                        <!--connected to input-->
                        <dim>1</dim>
                        <dim>40</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <!--connected to relu-->
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                    <port id="2">
                        <!--connected to split-->
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </output>
            </layer>
           <layer name="ReLU1" id="11" type="Activation" precision="FP32">
                <data type="ReLU" negative_slope="0.000000" />
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </output>
            </layer>
            <layer name="Split2" type="Split" id="12" precision="FP32">
                <data axis="1" />
                <input>
                    <port id="0">
                        <!--connected to split-->
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <!--connected to relu-->
                        <dim>1</dim>
                        <dim>10</dim>
                    </port>
                    <port id="2">
                        <!--connected to fc-->
                        <dim>1</dim>
                        <dim>10</dim>
                    </port>
                </output>
            </layer>
           <layer name="ReLU2" id="21" type="Activation" precision="FP32">
                <data type="ReLU" negative_slope="0.000000" />
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
            </layer>
            <layer name="FullyConnected1" id="22" type="InnerProduct" precision="FP32">
                <fc out-size="10" />
                <biases offset="0" size="40" />
                <weights offset="40" size="400" />
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
            </layer>
            <layer name="concat1" id="31"  precision="FP32" type="Concat">
                    <data axis="1"/>
                    <input>
                            <port id="0">
                                    <dim>1</dim>
                                    <dim>10</dim>
                            </port>
                            <port id="1">
                                    <dim>1</dim>
                                    <dim>10</dim>
                            </port>
                    </input>
                    <output>
                            <port id="2">
                                    <dim>1</dim>
                                    <dim>20</dim>
                            </port>
                    </output>
            </layer>
             <layer name="concat2" id="41"  precision="FP32" type="Concat">
                    <data axis="1"/>
                    <input>
                            <port id="0">
                                    <dim>1</dim>
                                    <dim>20</dim>
                            </port>
                            <port id="1">
                                    <dim>1</dim>
                                    <dim>20</dim>
                            </port>
                    </input>
                    <output>
                            <port id="2">
                                    <dim>1</dim>
                                    <dim>40</dim>
                            </port>
                    </output>
            </layer>
            <layer name="FullyConnected2" id="51" type="InnerProduct" precision="FP32">
                <fc out-size="40" />
                <biases offset="400" size="160" />
                <weights offset="560" size="6960" />
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>40</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>40</dim>
                    </port>
                </output>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
            <edge from-layer="1" from-port="1" to-layer="11" to-port="0" />
            <edge from-layer="1" from-port="2" to-layer="12" to-port="0" />
            <edge from-layer="11" from-port="1" to-layer="41" to-port="0" />
            <edge from-layer="12" from-port="1" to-layer="21" to-port="0" />
            <edge from-layer="12" from-port="2" to-layer="22" to-port="0" />
            <edge from-layer="21" from-port="1" to-layer="31" to-port="0" />
            <edge from-layer="22" from-port="1" to-layer="31" to-port="1" />
            <edge from-layer="31" from-port="2" to-layer="41" to-port="1" />
            <edge from-layer="41" from-port="2" to-layer="51" to-port="0" />
        </edges>
    </Net>
    )V0G0N";
}


std::string cropWithoutOffsetModel() {
    return R"V0G0N(
    <Net Name="cropWithoutOffsetModel" version="2" precision="FP32" batch="1">
        <layers>
            <layer name="input_1" type="input" id="0" precision="FP32">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </output>
            </layer>
            <layer name="Crop1" type="Crop" id="1" precision="FP32">
                <data axis="1" dim="10" offset="0"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>10</dim>
                    </port>
                </output>
            </layer>
            <layer name="FullyConnected1" id="2" type="InnerProduct" precision="FP32">
                <fc out-size="10" />
                <biases offset="0" size="40" />
                <weights offset="40" size="400" />
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
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
            <edge from-layer="1" from-port="1" to-layer="2" to-port="0" />
        </edges>
    </Net>
    )V0G0N";
}

std::string cropWithAlignedOffsetModel() {
    return R"V0G0N(
    <Net Name="cropWithAlignedOffsetModel" version="2" precision="FP32" batch="1">
        <layers>
            <layer name="input_1" type="input" id="0" precision="FP32">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </output>
            </layer>
            <layer name="Crop1" type="Crop" id="1" precision="FP32">
                <data axis="1" dim="10" offset="8"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>10</dim>
                    </port>
                </output>
            </layer>
            <layer name="FullyConnected1" id="2" type="InnerProduct" precision="FP32">
                <fc out-size="12" />
                <biases offset="0" size="40" />
                <weights offset="40" size="400" />
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
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
            <edge from-layer="1" from-port="1" to-layer="2" to-port="0" />
        </edges>
    </Net>
    )V0G0N";
}

std::string cropWithOffsetModel() {
    return R"V0G0N(
    <Net Name="cropWithOffsetModel" version="2" precision="FP32" batch="1">
        <layers>
            <layer name="input_1" type="input" id="0" precision="FP32">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </output>
            </layer>
            <layer name="Crop1" type="Crop" id="1" precision="FP32">
                <data axis="1" dim="10" offset="5"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>10</dim>
                    </port>
                </output>
            </layer>
            <layer name="FullyConnected1" id="2" type="InnerProduct" precision="FP32">
                <fc out-size="10" />
                <biases offset="0" size="40" />
                <weights offset="40" size="400" />
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
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
            <edge from-layer="1" from-port="1" to-layer="2" to-port="0" />
        </edges>
    </Net>
    )V0G0N";
}

std::string cropWithOffsetAndSecondDimModel() {
    return R"V0G0N(
<Net Name="cropWithOffsetModel" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="input_1" type="input" id="0" precision="FP32">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>20</dim>
                </port>
            </output>
        </layer>
        <layer name="Crop1" type="Crop" id="1" precision="FP32">
            <data axis="0,1" dim="1,10" offset="0,5"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>20</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer name="FullyConnected1" id="2" type="InnerProduct" precision="FP32">
            <fc out-size="10" />
            <biases offset="0" size="40" />
            <weights offset="40" size="400" />
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
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0" />
    </edges>
</Net>
)V0G0N";
}


std::string cropWithMaxOffsetModel() {
    return R"V0G0N(
    <Net Name="cropWithOffsetModel" version="2" precision="FP32" batch="1">
        <layers>
            <layer name="input_1" type="input" id="0" precision="FP32">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </output>
            </layer>
            <layer name="Crop1" type="Crop" id="1" precision="FP32">
                <data axis="1" dim="10" offset="10"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>10</dim>
                    </port>
                </output>
            </layer>
            <layer name="FullyConnected1" id="2" type="InnerProduct" precision="FP32">
                <fc out-size="10" />
                <biases offset="0" size="40" />
                <weights offset="40" size="400" />
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
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
            <edge from-layer="1" from-port="1" to-layer="2" to-port="0" />
        </edges>
    </Net>
    )V0G0N";
}

std::string cropWithOffsetExtendedModel() {
    return R"V0G0N(
    <Net Name="cropWithOffsetExtendedModel" version="2" precision="FP32" batch="1">
        <layers>
            <layer name="input_1" type="input" id="0" precision="FP32">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </output>
            </layer>
            <layer name="FullyConnected1" id="1" type="InnerProduct" precision="FP32">
                <fc out-size="20" />
                <biases offset="0" size="80" />
                <weights offset="80" size="1920" />
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </output>
            </layer>
            <layer name="Crop1" type="Crop" id="11" precision="FP32">
                <data axis="1" dim="10" offset="5"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>10</dim>
                    </port>
                </output>
            </layer>
            <layer name="FullyConnected2" id="12" type="InnerProduct" precision="FP32">
                <fc out-size="10" />
                <biases offset="1920" size="40" />
                <weights offset="1960" size="640" />
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
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
            <edge from-layer="1" from-port="1" to-layer="11" to-port="0" />
            <edge from-layer="11" from-port="1" to-layer="12" to-port="0" />
        </edges>
    </Net>
    )V0G0N";
}

std::string copyModel() {
    return R"V0G0N(
    <Net Name="cropWithOffsetExtendedModel" version="2" precision="FP32" batch="1">
        <layers>
            <layer name="input_1" type="input" id="0" precision="FP32">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </output>
            </layer>
            <layer name="FullyConnected1" id="1" type="InnerProduct" precision="FP32">
                <fc out-size="20" />
                <biases offset="0" size="80" />
                <weights offset="80" size="1920" />
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </output>
            </layer>
            <layer name="Copy1" id="2" type="Copy" precision="FP32">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </output>
            </layer>
            <layer name="Eltwise_1" type="Eltwise" id="11" precision="FP32">
                <data operation="sum" />
                <input>
                    <port id="0">
                        <!--connected to FullyConnected-->
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                    <port id="1">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </input>
                <output>
                    <port id="2">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </output>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
            <edge from-layer="0" from-port="0" to-layer="2" to-port="0" />
            <edge from-layer="1" from-port="1" to-layer="11" to-port="0" />
            <edge from-layer="2" from-port="1" to-layer="11" to-port="1" />
        </edges>
    </Net>
    )V0G0N";
}

std::string two_inputs_to_concat() {
    return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="N" version="2">
	<layers>
		<layer id="0" name="input_1" precision="FP32" type="input">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>600</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="input_2" precision="FP32" type="input">
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>600</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="concat" precision="FP32" type="Concat">
			<data out-size="600"/>
			<input>
				<port id="3">
					<dim>1</dim>
					<dim>600</dim>
				</port>
			</input>
			<input>
				<port id="4">
					<dim>1</dim>
					<dim>600</dim>
				</port>
			</input>
			<output>
				<port id="5">
					<dim>1</dim>
					<dim>1200</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="tanh_6" precision="FP32" type="Activation">
			<data type="tanh"/>
			<input>
				<port id="10">
					<dim>1</dim>
					<dim>600</dim>
				</port>
			</input>
			<output>
				<port id="11">
					<dim>1</dim>
					<dim>600</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="1" to-layer="2" to-port="3"/>
		<edge from-layer="1" from-port="2" to-layer="2" to-port="4"/>
		<edge from-layer="2" from-port="5" to-layer="5" to-port="10"/>
	</edges>
</net>
    )V0G0N";

}

std::string two_inputs_to_affine() {
    return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="" version="2">
	<layers>
		<layer id="0" name="input_1" precision="FP32" type="input">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="input_2" precision="FP32" type="input">
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="affinetransform_3" precision="FP32" type="FullyConnected">
			<data out-size="10"/>
			<input>
				<port id="3">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="4">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="400"/>
			</blobs>
		</layer>
		<layer id="3" name="affinetransform_4" precision="FP32" type="FullyConnected">
			<data out-size="600"/>
			<input>
				<port id="5">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="6">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
			<blobs>
				<weights offset="400" size="400"/>
			</blobs>
		</layer>
		<layer id="4" name="add_5" precision="FP32" type="Eltwise">
			<data operation="sum"/>
			<input>
				<port id="7">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="8">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="9">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="1" to-layer="2" to-port="3"/>
		<edge from-layer="1" from-port="2" to-layer="3" to-port="5"/>
		<edge from-layer="2" from-port="4" to-layer="4" to-port="7"/>
		<edge from-layer="3" from-port="6" to-layer="4" to-port="8"/>
	</edges>
</net>
    )V0G0N";

}


std::string affineAfterConvNoPermute() {
    return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model" version="2">
	<layers>
		<layer id="0" name="Placeholder" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>126</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="conv1" precision="FP32" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="5" kernel-y="1" output="128" pad-x="0" pad-y="0" stride="1,1,1,1" stride-x="1" stride-y="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>126</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>122</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="327680"/>
				<biases offset="327680" size="512"/>
			</blobs>
		</layer>
		<layer id="2" name="conv1_node/Relu" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>122</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>122</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="pool1_node/MaxPool" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel-x="2" kernel-y="1" pad-x="0" pad-y="0" pool-method="max" stride="1,1,1,2" stride-x="2" stride-y="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>122</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>61</dim>
				</port>
			</output>
		</layer>

        <layer id="4" name="Reshape_3" precision="FP32" type="Reshape">
            <data axis="0" dim="1,7808" num_axes="-1"/>
            <input>
            <port id="0">
                <dim>1</dim>
                <dim>128</dim>
                <dim>1</dim>
                <dim>61</dim>
            </port>
            </input>
            <output>
            <port id="1">
                <dim>1</dim>
                <dim>7808</dim>
            </port>
            </output>
        </layer>

        <layer name="FullyConnected" id="5" type="InnerProduct" precision="FP32">

            <fc out-size="10" />

            <biases offset="328192" size="40" />
            <weights offset="328232" size="312320" />

            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>7808</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        </layers>
        <edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
		<edge from-layer="2" from-port="1" to-layer="3" to-port="0"/>
        <edge from-layer="3" from-port="1" to-layer="4" to-port="0"/>
        <edge from-layer="4" from-port="1" to-layer="5" to-port="0"/>
        </edges>
    </net>

    )V0G0N";
}

std::string affineAfterConvWithPermute() {
    return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model" version="2">
	<layers>
		<layer id="0" name="Placeholder" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>126</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="conv1" precision="FP32" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="5" kernel-y="1" output="128" pad-x="0" pad-y="0" stride="1,1,1,1" stride-x="1" stride-y="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>126</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>122</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="327680"/>
				<biases offset="327680" size="512"/>
			</blobs>
		</layer>
		<layer id="2" name="conv1_node/Relu" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>122</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>122</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="pool1_node/MaxPool" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel-x="2" kernel-y="1" pad-x="0" pad-y="0" pool-method="max" stride="1,1,1,2" stride-x="2" stride-y="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>122</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>61</dim>
				</port>
			</output>
		</layer>

		<layer id="4" name="maxpoolingcomponent32/Permute" precision="FP32" type="Permute">
			<data order="0,3,2,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>61</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>61</dim>
					<dim>1</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>

        <layer id="5" name="Reshape_3" precision="FP32" type="Reshape">
            <data axis="0" dim="1,7808" num_axes="-1"/>
            <input>
            <port id="0">
                <dim>1</dim>
                <dim>61</dim>
                <dim>1</dim>
                <dim>128</dim>
            </port>
            </input>
            <output>
            <port id="1">
                <dim>1</dim>
                <dim>7808</dim>
            </port>
            </output>
        </layer>

        <layer name="FullyConnected" id="6" type="InnerProduct" precision="FP32">

            <fc out-size="10" />

            <biases offset="328192" size="40" />
            <weights offset="328232" size="312320" />

            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>7808</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        </layers>
        <edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
		<edge from-layer="2" from-port="1" to-layer="3" to-port="0"/>
        <edge from-layer="3" from-port="1" to-layer="4" to-port="0"/>
        <edge from-layer="4" from-port="1" to-layer="5" to-port="0"/>
        <edge from-layer="5" from-port="1" to-layer="6" to-port="0"/>
        </edges>
    </net>

    )V0G0N";
}



std::string ScaleShift3DModel() {
    return R"V0G0N(
   <?xml version="1.0" ?>
<net batch="1" name="frozen_model" version="4">
	<layers>
		<layer id="0" name="reshape_1_input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>40</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="conv1d_1/convolution/Squeeze" precision="FP32" type="Reshape">
			<data dim="1,5,8"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>40</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>5</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="conv1d_1/add" precision="FP32" type="ScaleShift">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>5</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>5</dim>
					<dim>8</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="32"/>
				<biases offset="32" size="32"/>
			</blobs>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
	</edges>
</net>

    )V0G0N";
}

std::string FCOnlyModelFP16() {
    return R"V0G0N(
    <Net Name="FullyConnected_Only" version="2" precision="FP16" batch="1">
	<layers>
		<layer name="input_1" type="input" id="0" precision="FP16">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>


        <layer name="FullyConnected" id="1" type="InnerProduct" precision="FP16">
            <fc out-size="10" />
            <biases offset="0" size="20" />
            <weights offset="20" size="200" />

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
        </layer>
    </layers>
    <edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
    </edges>
    </Net>
    )V0G0N";
}

std::string AffineWithReluSigmoidAndIdentity() {
    return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model" version="2">
	<layers>
		<layer id="0" name="Placeholder" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>

        <layer name="FullyConnected1" id="1" type="InnerProduct" precision="FP32">
            <fc out-size="10" />
            <weights offset="0" size="400" />
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
        </layer>

        <layer name="FullyConnected2" id="2" type="InnerProduct" precision="FP32">
            <fc out-size="10" />
            <weights offset="0" size="400" />
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
        </layer>

         <layer name="Eltwise_5" type="Eltwise" id="5" precision="FP32">
			<data operation="mul" />
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="1">
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


         <layer name="Eltwise_6" type="Eltwise" id="6" precision="FP32">
			<data operation="sum" />
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="1">
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

        <layer name="Sig_Activation" id="3" type="Activation" precision="FP32">
            <data type="sigmoid" />
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
        </layer>


        <layer name="Relu_Activation" id="4" type="Activation" precision="FP32">
            <data type="relu" />
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
        </layer>



         <layer name="Eltwise_7" type="Eltwise" id="7" precision="FP32">
			<data operation="sum" />
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="1">
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
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>

		<edge from-layer="1" from-port="1" to-layer="6" to-port="1"/>
		<edge from-layer="2" from-port="1" to-layer="6" to-port="0"/>
        <edge from-layer="2" from-port="1" to-layer="3" to-port="0"/>
        <edge from-layer="2" from-port="1" to-layer="4" to-port="0"/>

        <edge from-layer="3" from-port="1" to-layer="5" to-port="0"/>
        <edge from-layer="4" from-port="1" to-layer="5" to-port="1"/>

        <edge from-layer="6" from-port="2" to-layer="7" to-port="0"/>
        <edge from-layer="5" from-port="2" to-layer="7" to-port="1"/>

        </edges>
    </net>

    )V0G0N";
}

std::string AffineWithReluSigmoid() {

    return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model" version="2">
	<layers>
		<layer id="0" name="Placeholder" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>

        <layer name="FullyConnected1" id="1" type="InnerProduct" precision="FP32">
            <fc out-size="10" />
            <weights offset="0" size="400" />
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
        </layer>

        <layer name="Eltwise_4" type="Eltwise" id="4" precision="FP32">
			<data operation="mul" />
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="1">
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

        <layer name="Sig_Activation" id="2" type="Activation" precision="FP32">
            <data type="sigmoid" />
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
        </layer>


        <layer name="Relu_Activation" id="3" type="Activation" precision="FP32">
            <data type="relu" />
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
        </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
            <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
            <edge from-layer="1" from-port="1" to-layer="3" to-port="0"/>
            <edge from-layer="2" from-port="1" to-layer="4" to-port="0"/>
            <edge from-layer="3" from-port="1" to-layer="4" to-port="1"/>
        </edges>
    </net>

    )V0G0N";
}

std::string concatModelWithConstLayer() {
        return R"V0G0N(
    <net Name="concatinationModel" version="2" precision="FP32" batch="1">
        <layers>
            <layer name="input1" type="input" id="0" precision="FP32">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>10</dim>
                    </port>
                </output>
            </layer>
            <layer name="FullyConnected1" id="12" type="FullyConnected" precision="FP32">
                <fc out-size="10" />
                <biases offset="0" size="40" />
                <weights offset="0" size="400" />
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
            </layer>
            <layer name="input2" type="Const" id="2" precision="FP32">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>10</dim>
                    </port>
                </output>
                <blobs>
                    <custom offset="0" size="40"/>
                </blobs>
            </layer>

            <layer name="concat1" id="21"  precision="FP32" type="Concat">
                    <data axis="1"/>
                    <input>
                            <port id="0">
                                    <dim>1</dim>
                                    <dim>10</dim>
                            </port>
                            <port id="1">
                                    <dim>1</dim>
                                    <dim>10</dim>
                            </port>
                    </input>
                    <output>
                            <port id="2">
                                    <dim>1</dim>
                                    <dim>20</dim>
                            </port>
                    </output>
            </layer>
            <layer name="FullyConnected2" id="31" type="FullyConnected" precision="FP32">
                <fc out-size="20" />
                <biases offset="0" size="80" />
                <weights offset="0" size="1600" />
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>20</dim>
                    </port>
                </output>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="12" to-port="0" />
            <edge from-layer="12" from-port="1" to-layer="21" to-port="0" />
            <edge from-layer="2" from-port="0" to-layer="21" to-port="1" />
            <edge from-layer="21" from-port="2" to-layer="31" to-port="0" />
        </edges>
    </net>
    )V0G0N";
}

std::string LSTMCellOnlyModel() {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model" version="2">
    <layers>
        <layer id="0" name="Input" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>96</dim>
                </port>
            </output>
        </layer>
        <layer name="input-to-split-broken" id="31" type="Scaleshift" precision="FP32">
                <weights offset="0" size="384" />
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>96</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>96</dim>
                    </port>
                </output>
            </layer>

        <layer id="1" name="Split" precision="FP32" type="Split">
            <data axis="1" />
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>96</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>32</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>32</dim>
                </port>
                <port id="3">
                    <dim>1</dim>
                    <dim>32</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="LSTMCell" precision="FP32" type="LSTMCell">
            <data hidden_size="32"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>32</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>32</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>32</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>32</dim>
                </port>
                <port id="4">
                    <dim>1</dim>
                    <dim>32</dim>
                </port>
            </output>
            <blobs>
                <weights offset="384" size="32768"/>
                <biases offset="33152" size="512"/>
            </blobs>
        </layer>
        <layer name="Eltwise" type="Eltwise" id="3" precision="FP32">
            <data operation="sum" />
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>32</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>32</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>32</dim>
                </port>
            </output>
        </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="31" to-port="0"/>
    		<edge from-layer="31" from-port="1" to-layer="1" to-port="0"/>
            <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
            <edge from-layer="1" from-port="2" to-layer="2" to-port="1"/>
            <edge from-layer="1" from-port="3" to-layer="2" to-port="2"/>
            <edge from-layer="2" from-port="3" to-layer="3" to-port="0"/>
            <edge from-layer="2" from-port="4" to-layer="3" to-port="1"/>
        </edges>
    </net>
    )V0G0N";
};



std::string TIModelWithLSTMCell1() {
    return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="Basic_LSTM_S" version="5">
	<layers>
		<layer id="0" name="Reshape/placeholder_port_0" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
        <layer name="ScaleShift_1" type="ScaleShift" id="31" precision="FP32">
			<input>
				<port id="0">
					<!--connected to input-->
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<!--connected to , FullyConnected-->
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
			<weights offset="0" size="40" precision="FP32" />
		</layer>
		<layer id="1" name="Reshape_1/shape/Output_0/Data__const" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>3</dim>
				</port>
			</output>
			<blobs>
				<custom offset="40" size="12"/>
			</blobs>
		</layer>
		<layer id="2" name="Reshape_1" precision="FP32" type="Reshape">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="LSTM-Layer/lstm/rnn/while/Enter_3/Output_0/Data__const" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
			<blobs>
				<custom offset="52" size="40"/>
			</blobs>
		</layer>
		<layer id="5" name="LSTM-Layer/lstm/rnn/while/Enter_2/Output_0/Data__const" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
			<blobs>
				<custom offset="52" size="40"/>
			</blobs>
		</layer>
		<layer id="6" name="LSTM-Layer/lstm/rnn/while/LoopCond/TensorIteratorCondition_/TensorIterator" precision="FP32" type="TensorIterator">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
			<port_map>
				<input axis="0" external_port_id="0" internal_layer_id="0" internal_port_id="0" start="0"/>
				<input external_port_id="1" internal_layer_id="1" internal_port_id="1"/>
				<input external_port_id="2" internal_layer_id="1" internal_port_id="2"/>
				<output external_port_id="3" internal_layer_id="1" internal_port_id="5"/>
				<output external_port_id="4" internal_layer_id="1" internal_port_id="6"/>
			</port_map>
			<back_edges>
				<edge from-layer="1" from-port="5" to-layer="1" to-port="1"/>
				<edge from-layer="1" from-port="6" to-layer="1" to-port="2"/>
			</back_edges>
			<body>
				<layers>
					<layer id="0" name="LSTM-Layer/lstm/rnn/while/TensorArrayReadV3/Output_0/Data_/InputSqueeze" precision="FP32" type="Reshape">
						<data dim="-1,10"/>
						<input>
							<port id="0">
								<dim>1</dim>
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
					</layer>
					<layer id="1" name="LSTM-Layer/lstm/rnn/while/rnn/basic_lstm_cell/concat/LSTMCell" precision="FP32" type="LSTMCell">
						<data hidden_size="10"/>
						<input>
							<port id="0">
								<dim>1</dim>
								<dim>10</dim>
							</port>
							<port id="1">
								<dim>1</dim>
								<dim>10</dim>
							</port>
							<port id="2">
								<dim>1</dim>
								<dim>10</dim>
							</port>
						</input>
						<output>
							<port id="5">
								<dim>1</dim>
								<dim>10</dim>
							</port>
							<port id="6">
								<dim>1</dim>
								<dim>10</dim>
							</port>
						</output>
						<blobs>
							<weights offset="620" size="3200"/>
							<biases offset="3820" size="160"/>
						</blobs>
					</layer>
				</layers>
				<edges>
					<edge from-layer="0" from-port="1" to-layer="1" to-port="0"/>
				</edges>
			</body>
		</layer>
		<layer id="7" name="Output-Layer/MatMul" precision="FP32" type="FullyConnected">
			<data out-size="12"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>12</dim>
				</port>
			</output>
			<blobs>
				<weights offset="92" size="480"/>
				<biases offset="572" size="48"/>
			</blobs>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="31" to-port="0"/>
        <edge from-layer="31" from-port="1" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="2" from-port="2" to-layer="6" to-port="0"/>
		<edge from-layer="4" from-port="1" to-layer="6" to-port="1"/>
		<edge from-layer="5" from-port="1" to-layer="6" to-port="2"/>
		<edge from-layer="6" from-port="3" to-layer="7" to-port="0"/>
	</edges>
</net>
     )V0G0N";
}

std::string TIModelWithLSTMCell1WithoutScaleShift() {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="Basic_LSTM_S" version="5">
	<layers>
		<layer id="0" name="Reshape/placeholder_port_0" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="Reshape_1/shape/Output_0/Data__const" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>3</dim>
				</port>
			</output>
			<blobs>
				<custom offset="40" size="12"/>
			</blobs>
		</layer>
		<layer id="2" name="Reshape_1" precision="FP32" type="Reshape">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="LSTM-Layer/lstm/rnn/while/Enter_3/Output_0/Data__const" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
			<blobs>
				<custom offset="52" size="40"/>
			</blobs>
		</layer>
		<layer id="5" name="LSTM-Layer/lstm/rnn/while/Enter_2/Output_0/Data__const" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
			<blobs>
				<custom offset="52" size="40"/>
			</blobs>
		</layer>
		<layer id="6" name="LSTM-Layer/lstm/rnn/while/LoopCond/TensorIteratorCondition_/TensorIterator" precision="FP32" type="TensorIterator">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
			<port_map>
				<input axis="0" external_port_id="0" internal_layer_id="0" internal_port_id="0" start="0"/>
				<input external_port_id="1" internal_layer_id="1" internal_port_id="1"/>
				<input external_port_id="2" internal_layer_id="1" internal_port_id="2"/>
				<output external_port_id="3" internal_layer_id="1" internal_port_id="5"/>
				<output external_port_id="4" internal_layer_id="1" internal_port_id="6"/>
			</port_map>
			<back_edges>
				<edge from-layer="1" from-port="5" to-layer="1" to-port="1"/>
				<edge from-layer="1" from-port="6" to-layer="1" to-port="2"/>
			</back_edges>
			<body>
				<layers>
					<layer id="0" name="LSTM-Layer/lstm/rnn/while/TensorArrayReadV3/Output_0/Data_/InputSqueeze" precision="FP32" type="Reshape">
						<data dim="-1,10"/>
						<input>
							<port id="0">
								<dim>1</dim>
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
					</layer>
					<layer id="1" name="LSTM-Layer/lstm/rnn/while/rnn/basic_lstm_cell/concat/LSTMCell" precision="FP32" type="LSTMCell">
						<data hidden_size="10"/>
						<input>
							<port id="0">
								<dim>1</dim>
								<dim>10</dim>
							</port>
							<port id="1">
								<dim>1</dim>
								<dim>10</dim>
							</port>
							<port id="2">
								<dim>1</dim>
								<dim>10</dim>
							</port>
						</input>
						<output>
							<port id="5">
								<dim>1</dim>
								<dim>10</dim>
							</port>
							<port id="6">
								<dim>1</dim>
								<dim>10</dim>
							</port>
						</output>
						<blobs>
							<weights offset="620" size="3200"/>
							<biases offset="3820" size="160"/>
						</blobs>
					</layer>
				</layers>
				<edges>
					<edge from-layer="0" from-port="1" to-layer="1" to-port="0"/>
				</edges>
			</body>
		</layer>
		<layer id="7" name="Output-Layer/MatMul" precision="FP32" type="FullyConnected">
			<data out-size="12"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>12</dim>
				</port>
			</output>
			<blobs>
				<weights offset="92" size="480"/>
				<biases offset="572" size="48"/>
			</blobs>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="2" from-port="2" to-layer="6" to-port="0"/>
		<edge from-layer="4" from-port="1" to-layer="6" to-port="1"/>
		<edge from-layer="5" from-port="1" to-layer="6" to-port="2"/>
		<edge from-layer="6" from-port="3" to-layer="7" to-port="0"/>
	</edges>
</net>
     )V0G0N";
    }

    std::string TIModelWithLSTMCell1Aligned() {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="Basic_LSTM_S" version="5">
	<layers>
		<layer id="0" name="Reshape/placeholder_port_0" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
        <layer name="ScaleShift_1" type="ScaleShift" id="31" precision="FP32">
			<input>
				<port id="0">
					<!--connected to input-->
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<!--connected to , FullyConnected-->
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
			<weights offset="38588" size="128" precision="FP32" />
		</layer>
		<layer id="1" name="Reshape_1/shape/Output_0/Data__const" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>3</dim>
				</port>
			</output>
			<blobs>
				<custom offset="0" size="12"/>
			</blobs>
		</layer>
		<layer id="2" name="Reshape_1" precision="FP32" type="Reshape">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="LSTM-Layer/lstm/rnn/while/Enter_3/Output_0/Data__const" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<custom offset="12" size="128"/>
			</blobs>
		</layer>
		<layer id="5" name="LSTM-Layer/lstm/rnn/while/Enter_2/Output_0/Data__const" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<custom offset="12" size="128"/>
			</blobs>
		</layer>
		<layer id="6" name="LSTM-Layer/lstm/rnn/while/LoopCond/TensorIteratorCondition_/TensorIterator" precision="FP32" type="TensorIterator">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
			<port_map>
				<input axis="0" external_port_id="0" internal_layer_id="0" internal_port_id="0" start="0"/>
				<input external_port_id="1" internal_layer_id="1" internal_port_id="1"/>
				<input external_port_id="2" internal_layer_id="1" internal_port_id="2"/>
				<output external_port_id="3" internal_layer_id="1" internal_port_id="5"/>
				<output external_port_id="4" internal_layer_id="1" internal_port_id="6"/>
			</port_map>
			<back_edges>
				<edge from-layer="1" from-port="5" to-layer="1" to-port="1"/>
				<edge from-layer="1" from-port="6" to-layer="1" to-port="2"/>
			</back_edges>
			<body>
				<layers>
					<layer id="0" name="LSTM-Layer/lstm/rnn/while/TensorArrayReadV3/Output_0/Data_/InputSqueeze" precision="FP32" type="Reshape">
						<data dim="-1,32"/>
						<input>
							<port id="0">
								<dim>1</dim>
								<dim>1</dim>
								<dim>32</dim>
							</port>
						</input>
						<output>
							<port id="1">
								<dim>1</dim>
								<dim>32</dim>
							</port>
						</output>
					</layer>
					<layer id="1" name="LSTM-Layer/lstm/rnn/while/rnn/basic_lstm_cell/concat/LSTMCell" precision="FP32" type="LSTMCell">
						<data hidden_size="32"/>
						<input>
							<port id="0">
								<dim>1</dim>
								<dim>32</dim>
							</port>
							<port id="1">
								<dim>1</dim>
								<dim>32</dim>
							</port>
							<port id="2">
								<dim>1</dim>
								<dim>32</dim>
							</port>
						</input>
						<output>
							<port id="5">
								<dim>1</dim>
								<dim>32</dim>
							</port>
							<port id="6">
								<dim>1</dim>
								<dim>32</dim>
							</port>
						</output>
                        <blobs>
                            <weights offset="1724" size="32768"/>
                            <biases offset="34492" size="512"/>
                        </blobs>
					</layer>
				</layers>
				<edges>
					<edge from-layer="0" from-port="1" to-layer="1" to-port="0"/>
				</edges>
			</body>
		</layer>
		<layer id="7" name="Output-Layer/MatMul" precision="FP32" type="FullyConnected">
			<data out-size="12"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>12</dim>
				</port>
			</output>
			<blobs>
				<weights offset="140" size="1536"/>
				<biases offset="1676" size="48"/>
			</blobs>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="31" to-port="0"/>
        <edge from-layer="31" from-port="1" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="2" from-port="2" to-layer="6" to-port="0"/>
		<edge from-layer="4" from-port="1" to-layer="6" to-port="1"/>
		<edge from-layer="5" from-port="1" to-layer="6" to-port="2"/>
		<edge from-layer="6" from-port="3" to-layer="7" to-port="0"/>
	</edges>
</net>
     )V0G0N";
}

    std::string TIModelWithLSTMCell3Aligned() {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="Basic_LSTM_S" version="5">
	<layers>
		<layer id="0" name="Reshape/placeholder_port_0" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="Reshape_1/shape/Output_0/Data__const" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>3</dim>
				</port>
			</output>
			<blobs>
				<custom offset="0" size="12"/>
			</blobs>
		</layer>
		<layer id="2" name="Reshape_1" precision="FP32" type="Reshape">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
				</port>
				<port id="1">
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>3</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="LSTM-Layer/lstm/rnn/while/Enter_3/Output_0/Data__const" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<custom offset="12" size="128"/>
			</blobs>
		</layer>
		<layer id="5" name="LSTM-Layer/lstm/rnn/while/Enter_2/Output_0/Data__const" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<custom offset="12" size="128"/>
			</blobs>
		</layer>
		<layer id="6" name="LSTM-Layer/lstm/rnn/while/LoopCond/TensorIteratorCondition_/TensorIterator" precision="FP32" type="TensorIterator">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
			<port_map>
				<input axis="1" external_port_id="0" internal_layer_id="0" internal_port_id="0" start="0"/>
				<input external_port_id="1" internal_layer_id="1" internal_port_id="1"/>
				<input external_port_id="2" internal_layer_id="1" internal_port_id="2"/>
				<output external_port_id="3" internal_layer_id="1" internal_port_id="5"/>
				<output external_port_id="4" internal_layer_id="1" internal_port_id="6"/>
			</port_map>
			<back_edges>
				<edge from-layer="1" from-port="5" to-layer="1" to-port="1"/>
				<edge from-layer="1" from-port="6" to-layer="1" to-port="2"/>
			</back_edges>
			<body>
				<layers>
					<layer id="0" name="LSTM-Layer/lstm/rnn/while/TensorArrayReadV3/Output_0/Data_/InputSqueeze" precision="FP32" type="Reshape">
						<data dim="-1,32"/>
						<input>
							<port id="0">
								<dim>1</dim>
								<dim>1</dim>
								<dim>32</dim>
							</port>
						</input>
						<output>
							<port id="1">
								<dim>1</dim>
								<dim>32</dim>
							</port>
						</output>
					</layer>
					<layer id="1" name="LSTM-Layer/lstm/rnn/while/rnn/basic_lstm_cell/concat/LSTMCell" precision="FP32" type="LSTMCell">
						<data hidden_size="32"/>
						<input>
							<port id="0">
								<dim>1</dim>
								<dim>32</dim>
							</port>
							<port id="1">
								<dim>1</dim>
								<dim>32</dim>
							</port>
							<port id="2">
								<dim>1</dim>
								<dim>32</dim>
							</port>
						</input>
						<output>
							<port id="5">
								<dim>1</dim>
								<dim>32</dim>
							</port>
							<port id="6">
								<dim>1</dim>
								<dim>32</dim>
							</port>
						</output>
						<blobs>
							<weights offset="1724" size="32768"/>
							<biases offset="34492" size="512"/>
						</blobs>
					</layer>
				</layers>
				<edges>
					<edge from-layer="0" from-port="1" to-layer="1" to-port="0"/>
				</edges>
			</body>
		</layer>
		<layer id="7" name="Output-Layer/MatMul" precision="FP32" type="FullyConnected">
			<data out-size="12"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>12</dim>
				</port>
			</output>
			<blobs>
				<weights offset="140" size="1536"/>
				<biases offset="1676" size="48"/>
			</blobs>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="2" from-port="2" to-layer="6" to-port="0"/>
		<edge from-layer="4" from-port="1" to-layer="6" to-port="1"/>
		<edge from-layer="5" from-port="1" to-layer="6" to-port="2"/>
		<edge from-layer="6" from-port="3" to-layer="7" to-port="0"/>
	</edges>
</net>
    )V0G0N";
    };

std::string TIModelWithLSTMCell2() {
    return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="Basic_LSTM_S" version="5">
	<layers>
		<layer id="0" name="Reshape/placeholder_port_0" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>20</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="Reshape_1/shape/Output_0/Data__const" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>3</dim>
				</port>
			</output>
			<blobs>
				<custom offset="0" size="12"/>
			</blobs>
		</layer>
		<layer id="2" name="Reshape_1" precision="FP32" type="Reshape">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>20</dim>
				</port>
				<port id="1">
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>2</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="LSTM-Layer/lstm/rnn/while/Enter_3/Output_0/Data__const" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
			<blobs>
				<custom offset="12" size="40"/>
			</blobs>
		</layer>
		<layer id="5" name="LSTM-Layer/lstm/rnn/while/Enter_2/Output_0/Data__const" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
			<blobs>
				<custom offset="12" size="40"/>
			</blobs>
		</layer>
		<layer id="6" name="LSTM-Layer/lstm/rnn/while/LoopCond/TensorIteratorCondition_/TensorIterator" precision="FP32" type="TensorIterator">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
			<port_map>
				<input axis="0" external_port_id="0" internal_layer_id="0" internal_port_id="0" start="0"/>
				<input external_port_id="1" internal_layer_id="1" internal_port_id="1"/>
				<input external_port_id="2" internal_layer_id="1" internal_port_id="2"/>
				<output external_port_id="3" internal_layer_id="1" internal_port_id="5"/>
				<output external_port_id="4" internal_layer_id="1" internal_port_id="6"/>
			</port_map>
			<back_edges>
				<edge from-layer="1" from-port="5" to-layer="1" to-port="1"/>
				<edge from-layer="1" from-port="6" to-layer="1" to-port="2"/>
			</back_edges>
			<body>
				<layers>
					<layer id="0" name="LSTM-Layer/lstm/rnn/while/TensorArrayReadV3/Output_0/Data_/InputSqueeze" precision="FP32" type="Reshape">
						<data dim="-1,10"/>
						<input>
							<port id="0">
								<dim>1</dim>
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
					</layer>
					<layer id="1" name="LSTM-Layer/lstm/rnn/while/rnn/basic_lstm_cell/concat/LSTMCell" precision="FP32" type="LSTMCell">
						<data hidden_size="10"/>
						<input>
							<port id="0">
								<dim>1</dim>
								<dim>10</dim>
							</port>
							<port id="1">
								<dim>1</dim>
								<dim>10</dim>
							</port>
							<port id="2">
								<dim>1</dim>
								<dim>10</dim>
							</port>
						</input>
						<output>
							<port id="5">
								<dim>1</dim>
								<dim>10</dim>
							</port>
							<port id="6">
								<dim>1</dim>
								<dim>10</dim>
							</port>
						</output>
						<blobs>
							<weights offset="580" size="3200"/>
							<biases offset="3780" size="160"/>
						</blobs>
					</layer>
				</layers>
				<edges>
					<edge from-layer="0" from-port="1" to-layer="1" to-port="0"/>
				</edges>
			</body>
		</layer>
		<layer id="7" name="Output-Layer/MatMul" precision="FP32" type="FullyConnected">
			<data out-size="12"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>12</dim>
				</port>
			</output>
			<blobs>
				<weights offset="52" size="480"/>
				<biases offset="532" size="48"/>
			</blobs>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="2" from-port="2" to-layer="6" to-port="0"/>
		<edge from-layer="4" from-port="1" to-layer="6" to-port="1"/>
		<edge from-layer="5" from-port="1" to-layer="6" to-port="2"/>
		<edge from-layer="6" from-port="3" to-layer="7" to-port="0"/>
	</edges>
</net>
    )V0G0N";
};

std::string InputSplitConcatModel() {
    return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="InputSplitConcatModel" version="5">
	<layers>
		<layer id="0" name="input_1" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
        <layer name="ScaleShift_1" type="ScaleShift" id="4" precision="FP32">
			<input>
				<port id="0">
					<!--connected to input-->
					<dim>1</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<!--connected to , FullyConnected-->
					<dim>1</dim>
					<dim>64</dim>
				</port>
			</output>
			<weights offset="0" size="256" precision="FP32" />
		</layer>
		<layer id="1" name="split_1" precision="FP32" type="Split">
            <input>
                <port id="0">
					<dim>1</dim>
					<dim>64</dim>
				</port>
			</input>
            <output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
                <port id="2">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
        <layer id="2" name="concat_1" precision="FP32" type="Concat">
            <input>
                <port id="0">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
            </input>
            <output>
                <port id="2">
					<dim>1</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="affinetransform_1" precision="FP32" type="FullyConnected">
			<data out-size="10"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
			<blobs>
				<weights offset="256" size="2560"/>
			</blobs>
		</layer>
    </layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
        <edge from-layer="4" from-port="1" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
	</edges>
</net>
    )V0G0N";
}

std::string InputSplitConcatModelUnaligned() {
    return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="InputSplitConcatModel" version="5">
	<layers>
		<layer id="0" name="input_1" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>20</dim>
				</port>
			</output>
		</layer>
        <layer name="ScaleShift_1" type="ScaleShift" id="4" precision="FP32">
			<input>
				<port id="0">
					<!--connected to input-->
					<dim>1</dim>
					<dim>20</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<!--connected to , FullyConnected-->
					<dim>1</dim>
					<dim>20</dim>
				</port>
			</output>
			<weights offset="0" size="80" precision="FP32" />
		</layer>
		<layer id="1" name="split_1" precision="FP32" type="Split">
            <input>
                <port id="0">
					<dim>1</dim>
					<dim>20</dim>
				</port>
			</input>
            <output>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
                <port id="2">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
        <layer id="2" name="concat_1" precision="FP32" type="Concat">
            <input>
                <port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
            </input>
            <output>
                <port id="2">
					<dim>1</dim>
					<dim>20</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="affinetransform_1" precision="FP32" type="FullyConnected">
			<data out-size="10"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>20</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
			<blobs>
				<weights offset="80" size="800"/>
			</blobs>
		</layer>
    </layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
        <edge from-layer="4" from-port="1" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
	</edges>
</net>
    )V0G0N";
}

std::string InputSplitConcatReshapeModelUnaligned() {
    return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="InputSplitConcatModel" version="5">
	<layers>
		<layer id="0" name="input_1" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>20</dim>
				</port>
			</output>
		</layer>
        <layer name="ScaleShift_1" type="ScaleShift" id="4" precision="FP32">
			<input>
				<port id="0">
					<!--connected to input-->
					<dim>1</dim>
					<dim>20</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<!--connected to , FullyConnected-->
					<dim>1</dim>
					<dim>20</dim>
				</port>
			</output>
			<weights offset="0" size="80" precision="FP32" />
		</layer>
		<layer id="1" name="split_1" precision="FP32" type="Split">
            <input>
                <port id="0">
					<dim>1</dim>
					<dim>20</dim>
				</port>
			</input>
            <output>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
                <port id="2">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>

        <layer name="Reshape_1" id="41" precision="FP32" type="Reshape">
            <data axis="0" dim="1,10" num_axes="-1"/>
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
        </layer>


        <layer id="2" name="concat_1" precision="FP32" type="Concat">
            <input>
                <port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
            </input>
            <output>
                <port id="2">
					<dim>1</dim>
					<dim>20</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="affinetransform_1" precision="FP32" type="FullyConnected">
			<data out-size="10"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>20</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
			<blobs>
				<weights offset="80" size="800"/>
			</blobs>
		</layer>
    </layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
        <edge from-layer="4" from-port="1" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="41" to-port="0"/>
        <edge from-layer="41" from-port="1" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
	</edges>
</net>
    )V0G0N";
}


std::string eltwiseSumModelWithConstLayer2() {
    return R"V0G0N(
    <net Name="concatinationModel" version="2" precision="FP32" batch="1">
        <layers>
            <layer name="input1" type="input" id="0" precision="FP32">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>10</dim>
                    </port>
                </output>
            </layer>
            <layer name="input2" type="Const" id="2" precision="FP32">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>10</dim>
                    </port>
                </output>
                <blobs>
                    <custom offset="0" size="40"/>
                </blobs>
            </layer>

            <layer name="add" id="21"  precision="FP32" type="Eltwise">
			<data operation="sum" />
                    <data axis="1"/>
                    <input>
                            <port id="0">
                                    <dim>1</dim>
                                    <dim>10</dim>
                            </port>
                            <port id="1">
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
            <edge from-layer="0" from-port="0" to-layer="21" to-port="1" />
            <edge from-layer="2" from-port="0" to-layer="21" to-port="0" />
        </edges>
    </net>
    )V0G0N";
}

std::string eltwiseSumModelWithConstLayer() {
    return R"V0G0N(
    <net Name="concatinationModel" version="2" precision="FP32" batch="1">
        <layers>
            <layer name="input1" type="input" id="0" precision="FP32">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>10</dim>
                    </port>
                </output>
            </layer>
            <layer name="input2" type="Const" id="2" precision="FP32">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>10</dim>
                    </port>
                </output>
                <blobs>
                    <custom offset="0" size="40"/>
                </blobs>
            </layer>

            <layer name="add" id="21"  precision="FP32" type="Eltwise">
			<data operation="sum" />
                    <data axis="1"/>
                    <input>
                            <port id="0">
                                    <dim>1</dim>
                                    <dim>10</dim>
                            </port>
                            <port id="1">
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
            <edge from-layer="0" from-port="0" to-layer="21" to-port="0" />
            <edge from-layer="2" from-port="0" to-layer="21" to-port="1" />
        </edges>
    </net>
    )V0G0N";
}

std::string eltwiseMulModelWithConstLayer() {
    return R"V0G0N(
    <net Name="concatinationModel" version="2" precision="FP32" batch="1">
        <layers>
            <layer name="input1" type="input" id="0" precision="FP32">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>10</dim>
                    </port>
                </output>
            </layer>
            <layer name="input2" type="Const" id="2" precision="FP32">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>10</dim>
                    </port>
                </output>
                <blobs>
                    <custom offset="0" size="40"/>
                </blobs>
            </layer>

            <layer name="add" id="21"  precision="FP32" type="Eltwise">
			<data operation="mul" />
                    <data axis="1"/>
                    <input>
                            <port id="0">
                                    <dim>1</dim>
                                    <dim>10</dim>
                            </port>
                            <port id="1">
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
            <edge from-layer="0" from-port="0" to-layer="21" to-port="0" />
            <edge from-layer="2" from-port="0" to-layer="21" to-port="1" />
        </edges>
    </net>
    )V0G0N";
}

std::string LSTMCellOnlyModelUnaligned() {
    return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model" version="2">
    <layers>
        <layer id="0" name="Input" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>30</dim>
                </port>
            </output>
        </layer>
        <layer name="input-to-split-broken" id="31" type="Scaleshift" precision="FP32">
                <weights offset="0" size="120" />
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>30</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>30</dim>
                    </port>
                </output>
            </layer>

        <layer id="1" name="Split" precision="FP32" type="Split">
            <data axis="1" />
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>30</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
                <port id="3">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="LSTMCell" precision="FP32" type="LSTMCell">
            <data hidden_size="10"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
                <port id="4">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </output>
            <blobs>
                <weights offset="120" size="3200"/>
                <biases offset="3320" size="160"/>
            </blobs>
        </layer>
        <layer name="Eltwise" type="Eltwise" id="3" precision="FP32">
            <data operation="sum" />
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
                <port id="1">
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
            <edge from-layer="0" from-port="0" to-layer="31" to-port="0"/>
    		<edge from-layer="31" from-port="1" to-layer="1" to-port="0"/>
            <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
            <edge from-layer="1" from-port="2" to-layer="2" to-port="1"/>
            <edge from-layer="1" from-port="3" to-layer="2" to-port="2"/>
            <edge from-layer="2" from-port="3" to-layer="3" to-port="0"/>
            <edge from-layer="2" from-port="4" to-layer="3" to-port="1"/>
        </edges>
    </net>
)V0G0N";
};

std::string twoCropsModel() {
    return R"V0G0N(
    <Net Name="cropWithoutOffsetModel" version="2" precision="FP32" batch="1">
        <layers>
            <layer name="input_1" type="input" id="0" precision="FP32">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>48</dim>
                    </port>
                </output>
            </layer>

            <layer name="InputDiagonal" id="1" type="ScaleShift" precision="FP32">
                <weights offset="0" size="192" />
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>48</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>48</dim>
                    </port>
                </output>
            </layer>

            <layer name="Crop1" type="Crop" id="2" precision="FP32">
                <data axis="1" dim="16" offset="0"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>48</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>16</dim>
                    </port>
                </output>
            </layer>
            <layer name="Crop2" type="Crop" id="3" precision="FP32">
                <data axis="1" dim="16" offset="32"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>48</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>16</dim>
                    </port>
                </output>
            </layer>

            <layer name="concat1" id="4"  precision="FP32" type="Concat">
                    <data axis="1"/>
                    <input>
                            <port id="0">
                                    <dim>1</dim>
                                    <dim>16</dim>
                            </port>
                            <port id="1">
                                    <dim>1</dim>
                                    <dim>16</dim>
                            </port>
                    </input>
                    <output>
                            <port id="2">
                                    <dim>1</dim>
                                    <dim>32</dim>
                            </port>
                    </output>
            </layer>
            <layer name="Diagonal" id="5" type="ScaleShift" precision="FP32">
                <weights offset="192" size="128" />
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>32</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>32</dim>
                    </port>
                </output>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
            <edge from-layer="1" from-port="1" to-layer="2" to-port="0" />
            <edge from-layer="1" from-port="1" to-layer="3" to-port="0" />
            <edge from-layer="2" from-port="1" to-layer="4" to-port="0" />
            <edge from-layer="3" from-port="1" to-layer="4" to-port="1" />
            <edge from-layer="4" from-port="2" to-layer="5" to-port="0" />
        </edges>
    </Net>
    )V0G0N";
}

std::string threeCropsModel() {
        return R"V0G0N(
    <Net Name="cropWithoutOffsetModel" version="2" precision="FP32" batch="1">
        <layers>
            <layer name="input_1" type="input" id="0" precision="FP32">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>40</dim>
                    </port>
                </output>
            </layer>

            <layer name="InputDiagonal" id="1" type="ScaleShift" precision="FP32">
                <weights offset="0" size="160" />
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>40</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>40</dim>
                    </port>
                </output>
            </layer>

            <layer name="Crop1" type="Crop" id="2" precision="FP32">
                <data axis="1" dim="8" offset="0"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>40</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>8</dim>
                    </port>
                </output>
            </layer>
            <layer name="Crop2" type="Crop" id="3" precision="FP32">
                <data axis="1" dim="8" offset="16"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>40</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>8</dim>
                    </port>
                </output>
            </layer>

            <layer name="Crop3" type="Crop" id="4" precision="FP32">
                <data axis="1" dim="8" offset="32"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>40</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>8</dim>
                    </port>
                </output>
            </layer>

            <layer name="concat1" id="5"  precision="FP32" type="Concat">
                    <data axis="1"/>
                    <input>
                            <port id="0">
                                    <dim>1</dim>
                                    <dim>8</dim>
                            </port>
                            <port id="1">
                                    <dim>1</dim>
                                    <dim>16</dim>
                            </port>
                    </input>
                    <output>
                            <port id="2">
                                    <dim>1</dim>
                                    <dim>24</dim>
                            </port>
                    </output>
            </layer>
            <layer name="concat2" id="6"  precision="FP32" type="Concat">
                    <data axis="1"/>
                    <input>
                            <port id="0">
                                    <dim>1</dim>
                                    <dim>8</dim>
                            </port>
                            <port id="1">
                                    <dim>1</dim>
                                    <dim>8</dim>
                            </port>
                    </input>
                    <output>
                            <port id="2">
                                    <dim>1</dim>
                                    <dim>16</dim>
                            </port>
                    </output>
            </layer>
            <layer name="Diagonal" id="7" type="ScaleShift" precision="FP32">
                <weights offset="160" size="96" />
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>24</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>24</dim>
                    </port>
                </output>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
            <edge from-layer="1" from-port="1" to-layer="2" to-port="0" />
            <edge from-layer="1" from-port="1" to-layer="3" to-port="0" />
            <edge from-layer="1" from-port="1" to-layer="4" to-port="0" />

            <edge from-layer="2" from-port="1" to-layer="5" to-port="0" />
            <edge from-layer="3" from-port="1" to-layer="6" to-port="0" />
            <edge from-layer="4" from-port="1" to-layer="6" to-port="1" />

            <edge from-layer="6" from-port="2" to-layer="5" to-port="1" />
            <edge from-layer="5" from-port="2" to-layer="7" to-port="0" />
        </edges>
    </Net>
    )V0G0N";
}


std::string threeCropsWithReshapeModel() {
    return R"V0G0N(
    <Net Name="cropWithoutOffsetModel" version="2" precision="FP32" batch="1">
        <layers>
            <layer name="input_1" type="input" id="0" precision="FP32">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>40</dim>
                    </port>
                </output>
            </layer>

            <layer name="InputDiagonal" id="1" type="ScaleShift" precision="FP32">
                <weights offset="0" size="160" />
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>40</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>40</dim>
                    </port>
                </output>
            </layer>

            <layer name="Crop1" type="Crop" id="2" precision="FP32">
                <data axis="1" dim="8" offset="0"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>40</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>8</dim>
                    </port>
                </output>
            </layer>
            <layer name="Crop2" type="Crop" id="3" precision="FP32">
                <data axis="1" dim="8" offset="16"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>40</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>8</dim>
                    </port>
                </output>
            </layer>

            <layer name="Crop3" type="Crop" id="4" precision="FP32">
                <data axis="1" dim="8" offset="32"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>40</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>8</dim>
                    </port>
                </output>
            </layer>

        <layer name="Reshape_1" id="41" precision="FP32" type="Reshape">
            <data axis="0" dim="1,8" num_axes="-1"/>
            <input>
            <port id="0">
                <dim>1</dim>
                <dim>8</dim>
            </port>
            </input>
            <output>
            <port id="1">
                <dim>1</dim>
                <dim>8</dim>
            </port>
            </output>
        </layer>

        <layer name="Reshape_2" id="42" precision="FP32" type="Reshape">
            <data axis="0" dim="1,8" num_axes="-1"/>
            <input>
            <port id="0">
                <dim>1</dim>
                <dim>8</dim>
            </port>
            </input>
            <output>
            <port id="1">
                <dim>1</dim>
                <dim>8</dim>
            </port>
            </output>
        </layer>

        <layer name="Reshape_3" id="43" precision="FP32" type="Reshape">
            <data axis="0" dim="1,8" num_axes="-1"/>
            <input>
            <port id="0">
                <dim>1</dim>
                <dim>8</dim>
            </port>
            </input>
            <output>
            <port id="1">
                <dim>1</dim>
                <dim>8</dim>
            </port>
            </output>
        </layer>

            <layer name="concat1" id="5"  precision="FP32" type="Concat">
                    <data axis="1"/>
                    <input>
                            <port id="0">
                                    <dim>1</dim>
                                    <dim>8</dim>
                            </port>
                            <port id="1">
                                    <dim>1</dim>
                                    <dim>16</dim>
                            </port>
                    </input>
                    <output>
                            <port id="2">
                                    <dim>1</dim>
                                    <dim>24</dim>
                            </port>
                    </output>
            </layer>
            <layer name="concat2" id="6"  precision="FP32" type="Concat">
                    <data axis="1"/>
                    <input>
                            <port id="0">
                                    <dim>1</dim>
                                    <dim>8</dim>
                            </port>
                            <port id="1">
                                    <dim>1</dim>
                                    <dim>8</dim>
                            </port>
                    </input>
                    <output>
                            <port id="2">
                                    <dim>1</dim>
                                    <dim>16</dim>
                            </port>
                    </output>
            </layer>
            <layer name="Diagonal" id="7" type="ScaleShift" precision="FP32">
                <weights offset="160" size="96" />
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>24</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>24</dim>
                    </port>
                </output>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
            <edge from-layer="1" from-port="1" to-layer="2" to-port="0" />
            <edge from-layer="1" from-port="1" to-layer="3" to-port="0" />
            <edge from-layer="1" from-port="1" to-layer="4" to-port="0" />

            <edge from-layer="2" from-port="1" to-layer="41" to-port="0" />
            <edge from-layer="3" from-port="1" to-layer="42" to-port="0" />
            <edge from-layer="4" from-port="1" to-layer="43" to-port="0" />


            <edge from-layer="41" from-port="1" to-layer="5" to-port="0" />
            <edge from-layer="42" from-port="1" to-layer="6" to-port="0" />
            <edge from-layer="43" from-port="1" to-layer="6" to-port="1" />

            <edge from-layer="6" from-port="2" to-layer="5" to-port="1" />
            <edge from-layer="5" from-port="2" to-layer="7" to-port="0" />
        </edges>
    </Net>
    )V0G0N";
}

std::string PowerWithScaleFactor1() {
    return R"V0G0N(
    <?xml version="1.0" ?>
<net batch="1" name="PowerWithScaleShift" version="5">
	<layers>
		<layer id="0" name="Reshape/placeholder_port_0" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
        <layer id="1" name="conv2_node/Neg" precision="FP32" type="Power">
			<data power="1" scale="1" shift="0"/>
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
		</layer>
		<layer id="2" name="fc" precision="FP32" type="FullyConnected">
			<data out-size="12"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>12</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="480"/>
				<biases offset="480" size="48"/>
			</blobs>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
	</edges>
</net>
    )V0G0N";
}

std::string SplitToConcatThroughScaleShift() {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model_split_merge" version="2">
	<layers>
		<layer id="0" name="Placeholder" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>30</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="strided_slice/Split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>30</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>20</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="add" precision="FP32" type="ScaleShift">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="40"/>
				<biases offset="40" size="40"/>
			</blobs>
		</layer>
		<layer id="3" name="concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>20</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>30</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="MatMul_1" precision="FP32" type="FullyConnected">
			<data out-size="20"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>30</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>20</dim>
				</port>
			</output>
			<blobs>
				<weights offset="80" size="2400"/>
				<biases offset="2480" size="80"/>
			</blobs>
		</layer>
		<layer id="5" name="add_1" precision="FP32" type="ScaleShift">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>20</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>20</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2560" size="80"/>
				<biases offset="2640" size="80"/>
			</blobs>
		</layer>
		<layer id="6" name="concat_1" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>20</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>20</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>40</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="3" to-port="0"/>
		<edge from-layer="2" from-port="3" to-layer="3" to-port="1"/>
		<edge from-layer="3" from-port="2" to-layer="4" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="5" to-port="0"/>
		<edge from-layer="4" from-port="3" to-layer="6" to-port="0"/>
		<edge from-layer="5" from-port="3" to-layer="6" to-port="1"/>
	</edges>
</net>
    )V0G0N";
    }

std::string ConcatWithDiffScaleFactor() {
        return R"V0G0N(
<net Name="concatinationWithDiffScaleFactor" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="input1" type="input" id="0" precision="FP32">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>20</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="Split" precision="FP32" type="Split">
            <data axis="1" />
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>20</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer name="identity_activation" id="2" type="Activation" precision="FP32">
            <data type="sigmoid" />
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
        </layer>
        <layer name="tanh_activation" id="3" type="Activation" precision="FP32">
            <data type="tanh" />
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
        </layer>
        <layer id="4" name="concat" precision="FP32" type="Concat">
            <input>
                <port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
            </input>
            <output>
                <port id="2">
					<dim>1</dim>
					<dim>20</dim>
				</port>
			</output>
		</layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0" />
        <edge from-layer="1" from-port="2" to-layer="3" to-port="0" />
        <edge from-layer="2" from-port="1" to-layer="4" to-port="0" />
        <edge from-layer="3" from-port="1" to-layer="4" to-port="1" />
    </edges>
</net>
)V0G0N";
    }

std::string TwoOutputs() {
return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model_split_merge" version="2">
	<layers>
		<layer id="0" name="Placeholder" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="affinetransform_0" precision="FP32" type="FullyConnected">
			<data out-size="20"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>20</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="800"/>
                <biases offset="800" size="80"/>
			</blobs>
		</layer>
		<layer id="2" name="affinetransform_1" precision="FP32" type="FullyConnected">
			<data out-size="10"/>
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
			<blobs>
				<weights offset="880" size="400"/>
                <biases offset="1280" size="40"/>
			</blobs>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
	</edges>
</net>
    )V0G0N";
}

std::string TwoOutputsDiffPrecision() {
    return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model_split_merge" version="2">
	<layers>
		<layer id="0" name="Placeholder" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="affinetransform_0" precision="FP32" type="FullyConnected">
			<data out-size="10"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>20</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="800"/>
                <biases offset="800" size="80"/>
			</blobs>
		</layer>
		<layer id="2" name="affinetransform_1" precision="FP32" type="FullyConnected">
			<data out-size="10"/>
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
			<blobs>
				<weights offset="880" size="400"/>
                <biases offset="1280" size="40"/>
			</blobs>
		</layer>
        <layer name="ReLU_Activation" type="Activation" id="3" precision="FP32">
            <data type="ReLU" negative_slope="0.000000" />
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
        </layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="2" from-port="1" to-layer="3" to-port="0"/>
	</edges>
</net>
    )V0G0N";
}


std::string SplitToConcatWith2InputsNotAlignedNoFC() {
    return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model_split_to_concat_with_2_inputs" version="2">
	<layers>
		<layer id="0" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>20</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>20</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="4">
					<dim>1</dim>
					<dim>20</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>

		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="1" from-port="2" to-layer="2" to-port="2"/>

	</edges>
</net>
        )V0G0N";
    }

    std::string SplitToConcatWith2Inputs1360NotAlignedNoFC () {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model_split_to_concat_with_2_inputs" version="2">
	<layers>
		<layer id="0" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>1360</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1360</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>880</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>480</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="1">
					<dim>1</dim>
					<dim>880</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>480</dim>
				</port>
			</input>
			<output>
				<port id="4">
					<dim>1</dim>
					<dim>1360</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>

		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="1" from-port="2" to-layer="2" to-port="2"/>

	</edges>
</net>
        )V0G0N";
    }

    std::string SplitToConcatWith2By50InputsNotAlignedNoFC () {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model_split_to_concat_with_2_inputs" version="2">
	<layers>
		<layer id="0" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>100</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>100</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>50</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>50</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="1">
					<dim>1</dim>
					<dim>50</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>50</dim>
				</port>
			</input>
			<output>
				<port id="4">
					<dim>1</dim>
					<dim>100</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>

		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="1" from-port="2" to-layer="2" to-port="2"/>

	</edges>
</net>
        )V0G0N";
    }

    std::string SplitToConcatWith2By50InputsNotAlignedNoFCWithInCopyWithOutCopy  () {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model_split_to_concat_with_2_inputs" version="2">
	<layers>
		<layer id="0" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>100</dim>
				</port>
			</output>
		</layer>
        <layer name="input_copy" id="4" type="Copy" precision="FP32">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>100</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>100</dim>
                    </port>
                </output>
        </layer>
		<layer id="1" name="split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>100</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>50</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>50</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="1">
					<dim>1</dim>
					<dim>50</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>50</dim>
				</port>
			</input>
			<output>
				<port id="4">
					<dim>1</dim>
					<dim>100</dim>
				</port>
			</output>
		</layer>
        <layer name="output_copy" id="3" type="Copy" precision="FP32">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>100</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>100</dim>
                    </port>
                </output>
        </layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>

        <edge from-layer="4" from-port="1" to-layer="1" to-port="0"/>

		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="1" from-port="2" to-layer="2" to-port="2"/>

        <edge from-layer="2" from-port="4" to-layer="3" to-port="0"/>
	</edges>
</net>
        )V0G0N";
    }

    std::string SplitToConcatWith2By64InputsAlignedNoFC () {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model_split_to_concat_with_2_inputs" version="2">
	<layers>
		<layer id="0" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="4">
					<dim>1</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>

		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="1" from-port="2" to-layer="2" to-port="2"/>

	</edges>
</net>
        )V0G0N";
    }

    std::string SplitToConcatWith2By64InputsAlignedNoFCWithOutCopy () {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model_split_to_concat_with_2_inputs" version="2">
	<layers>
		<layer id="0" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="4">
					<dim>1</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
        <layer name="output_copy" id="3" type="Copy" precision="FP32">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>128</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>128</dim>
                    </port>
                </output>
        </layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>

		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="1" from-port="2" to-layer="2" to-port="2"/>

        <edge from-layer="2" from-port="4" to-layer="3" to-port="0"/>
	</edges>
</net>
        )V0G0N";
    }

    std::string SplitToConcatWith2InputsAlignedNoFC () {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model_split_to_concat_with_2_inputs" version="2">
	<layers>
		<layer id="0" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="4">
					<dim>1</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
    </layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>

		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="1" from-port="2" to-layer="2" to-port="2"/>
	</edges>
</net>
        )V0G0N";
    }

    std::string SplitToConcatWith2InputsAlignedNoFCWithInCopyWithOutCopy () {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model_split_to_concat_with_2_inputs" version="2">
	<layers>
		<layer id="0" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
        <layer name="input_copy" id="4" type="Copy" precision="FP32">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>64</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>64</dim>
                    </port>
                </output>
        </layer>
		<layer id="1" name="split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="4">
					<dim>1</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
        <layer name="output_copy" id="3" type="Copy" precision="FP32">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>64</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>64</dim>
                    </port>
                </output>
        </layer>
    </layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>

        <edge from-layer="4" from-port="1" to-layer="1" to-port="0"/>

		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="1" from-port="2" to-layer="2" to-port="2"/>

        <edge from-layer="2" from-port="4" to-layer="3" to-port="0"/>
	</edges>
</net>
        )V0G0N";
    }

    std::string SplitToConcatWith2InputsNotAlignedWithFC () {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model_split_to_concat_with_2_inputs" version="2">
	<layers>
		<layer id="0" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>20</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>20</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="4">
					<dim>1</dim>
					<dim>20</dim>
				</port>
			</output>
		</layer>
        <layer id="3" name="fc" precision="FP32" type="FullyConnected">
			<data out-size="10"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>20</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="840"/>
				<biases offset="800" size="40"/>
			</blobs>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>

		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="1" from-port="2" to-layer="2" to-port="2"/>

    <edge from-layer="2" from-port="4" to-layer="3" to-port="0"/>
	</edges>
</net>
        )V0G0N";
    }

    std::string SplitToConcatWith2InputsAlignedWithFC () {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model_split_to_concat_with_2_inputs" version="2">
	<layers>
		<layer id="0" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="4">
					<dim>1</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
        <layer id="3" name="fc" precision="FP32" type="FullyConnected">
			<data out-size="32"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="8324"/>
				<biases offset="8196" size="128"/>
			</blobs>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>

		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="1" from-port="2" to-layer="2" to-port="2"/>

        <edge from-layer="2" from-port="4" to-layer="3" to-port="0"/>
	</edges>
</net>
        )V0G0N";
    }

    std::string SplitToConcatWith2InputsAlignedWithFCWithInCopy () {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model_split_to_concat_with_2_inputs" version="2">
	<layers>
		<layer id="0" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
        <layer name="input_copy" id="4" type="Copy" precision="FP32">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>64</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>64</dim>
                    </port>
                </output>
        </layer>
		<layer id="1" name="split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="4">
					<dim>1</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
        <layer id="3" name="fc" precision="FP32" type="FullyConnected">
			<data out-size="32"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="8324"/>
				<biases offset="8196" size="128"/>
			</blobs>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>

        <edge from-layer="4" from-port="1" to-layer="1" to-port="0"/>

		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="1" from-port="2" to-layer="2" to-port="2"/>

        <edge from-layer="2" from-port="4" to-layer="3" to-port="0"/>
	</edges>
</net>
        )V0G0N";
    }

std::string SplitToConcatWith3InputsNotAlignedNoFC () {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model_split_to_concat_with_3_inputs" version="2">
	<layers>
		<layer id="0" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>30</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>30</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="4">
					<dim>1</dim>
					<dim>30</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>

		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="1" from-port="2" to-layer="2" to-port="2"/>
		<edge from-layer="1" from-port="3" to-layer="2" to-port="3"/>

	</edges>
</net>
        )V0G0N";
}

std::string SplitToConcatWith3InputsNotAlignedWithFC () {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model_split_to_concat_with_3_inputs" version="2">
	<layers>
		<layer id="0" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>30</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>30</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="4">
					<dim>1</dim>
					<dim>30</dim>
				</port>
			</output>
		</layer>
        <layer id="3" name="fc" precision="FP32" type="FullyConnected">
			<data out-size="10"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>30</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="1240"/>
				<biases offset="1200" size="40"/>
			</blobs>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>

		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="1" from-port="2" to-layer="2" to-port="2"/>
		<edge from-layer="1" from-port="3" to-layer="2" to-port="3"/>

        <edge from-layer="2" from-port="4" to-layer="3" to-port="0"/>
	</edges>
</net>
        )V0G0N";
    }

    std::string SplitToConcatWith3InputsAlignedNoFC () {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model_split_to_concat_with_3_inputs_align" version="2">
	<layers>
		<layer id="0" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="4">
					<dim>1</dim>
					<dim>96</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>

		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="1" from-port="2" to-layer="2" to-port="2"/>
		<edge from-layer="1" from-port="3" to-layer="2" to-port="3"/>

	</edges>
</net>
        )V0G0N";
    }

    std::string SplitToConcatWith3InputsAlignedNoFCWithInCopyWithOutCopy () {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model_split_to_concat_with_3_inputs_align" version="2">
	<layers>
		<layer id="0" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
				</port>
			</output>
		</layer>
        <layer name="input_copy" id="4" type="Copy" precision="FP32">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>96</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>96</dim>
                    </port>
                </output>
        </layer>
		<layer id="1" name="split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="4">
					<dim>1</dim>
					<dim>96</dim>
				</port>
			</output>
		</layer>
        <layer name="output_copy" id="3" type="Copy" precision="FP32">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>96</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>96</dim>
                    </port>
                </output>
        </layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>

        <edge from-layer="4" from-port="1" to-layer="1" to-port="0"/>

		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="1" from-port="2" to-layer="2" to-port="2"/>
		<edge from-layer="1" from-port="3" to-layer="2" to-port="3"/>

        <edge from-layer="2" from-port="4" to-layer="3" to-port="0"/>

	</edges>
</net>
        )V0G0N";
}

    std::string SplitToConcatWith3InputsAlignedWithFC () {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model_split_to_concat_with_3_inputs_align" version="2">
	<layers>
		<layer id="0" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="4">
					<dim>1</dim>
					<dim>96</dim>
				</port>
			</output>
		</layer>
        <layer id="3" name="fc" precision="FP32" type="FullyConnected">
			<data out-size="10"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="12416"/>
				<biases offset="12288" size="128"/>
			</blobs>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>

		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="1" from-port="2" to-layer="2" to-port="2"/>
		<edge from-layer="1" from-port="3" to-layer="2" to-port="3"/>

        <edge from-layer="2" from-port="4" to-layer="3" to-port="0"/>
	</edges>
</net>
        )V0G0N";
    }

    std::string SplitToConcatWith3InputsAlignedWithFCWithInCopy () {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model_split_to_concat_with_3_inputs_align" version="2">
	<layers>
		<layer id="0" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
				</port>
			</output>
		</layer>
        <layer name="input_copy" id="4" type="Copy" precision="FP32">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>96</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>96</dim>
                    </port>
                </output>
        </layer>
		<layer id="1" name="split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="4">
					<dim>1</dim>
					<dim>96</dim>
				</port>
			</output>
		</layer>
        <layer id="3" name="fc" precision="FP32" type="FullyConnected">
			<data out-size="10"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="12416"/>
				<biases offset="12288" size="128"/>
			</blobs>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>

        <edge from-layer="4" from-port="1" to-layer="1" to-port="0"/>

		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="1" from-port="2" to-layer="2" to-port="2"/>
		<edge from-layer="1" from-port="3" to-layer="2" to-port="3"/>

        <edge from-layer="2" from-port="4" to-layer="3" to-port="0"/>
	</edges>
</net>
        )V0G0N";
    }

    std::string SplitToConcatWith4InputsNotAlignedNoFC () {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model_split_to_concat_with_4_inputs" version="2">
	<layers>
		<layer id="0" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>40</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>40</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>10</dim>
				</port>
                <port id="4">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>10</dim>
				</port>
                <port id="4">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="5">
					<dim>1</dim>
					<dim>40</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>

		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="1" from-port="2" to-layer="2" to-port="2"/>
		<edge from-layer="1" from-port="3" to-layer="2" to-port="3"/>
        <edge from-layer="1" from-port="4" to-layer="2" to-port="4"/>
	</edges>
</net>
        )V0G0N";
    }

    std::string SplitToConcatWith4InputsNotAlignedNoFCWithOutCopy () {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model_split_to_concat_with_4_inputs" version="2">
	<layers>
		<layer id="0" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>40</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>40</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>10</dim>
				</port>
                <port id="4">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>10</dim>
				</port>
                <port id="4">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="5">
					<dim>1</dim>
					<dim>40</dim>
				</port>
			</output>
		</layer>
        <layer name="output_copy" id="3" type="Copy" precision="FP32">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>40</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>40</dim>
                    </port>
                </output>
        </layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>

		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="1" from-port="2" to-layer="2" to-port="2"/>
		<edge from-layer="1" from-port="3" to-layer="2" to-port="3"/>
        <edge from-layer="1" from-port="4" to-layer="2" to-port="4"/>

        <edge from-layer="2" from-port="5" to-layer="3" to-port="0"/>
	</edges>
</net>
        )V0G0N";
    }

    std::string SplitToConcatWith10InputsNotAlignedNoFC () {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model_split_to_concat_with_10_inputs" version="2">
	<layers>
		<layer id="0" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>100</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>100</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="5">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="6">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="7">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="8">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="9">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="10">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="5">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="6">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="7">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="8">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="9">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="10">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="11">
					<dim>1</dim>
					<dim>100</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>

		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="1" from-port="2" to-layer="2" to-port="2"/>
		<edge from-layer="1" from-port="3" to-layer="2" to-port="3"/>
		<edge from-layer="1" from-port="4" to-layer="2" to-port="4"/>
		<edge from-layer="1" from-port="5" to-layer="2" to-port="5"/>
		<edge from-layer="1" from-port="6" to-layer="2" to-port="6"/>
		<edge from-layer="1" from-port="7" to-layer="2" to-port="7"/>
		<edge from-layer="1" from-port="8" to-layer="2" to-port="8"/>
		<edge from-layer="1" from-port="9" to-layer="2" to-port="9"/>
		<edge from-layer="1" from-port="10" to-layer="2" to-port="10"/>
	</edges>
</net>
        )V0G0N";
    }

    std::string SplitToConcatWith10InputsNotAlignedNoFCWithOutCopy () {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model_split_to_concat_with_10_inputs" version="2">
	<layers>
		<layer id="0" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>100</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>100</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="5">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="6">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="7">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="8">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="9">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="10">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="5">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="6">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="7">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="8">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="9">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="10">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="11">
					<dim>1</dim>
					<dim>100</dim>
				</port>
			</output>
		</layer>
        <layer name="output_copy" id="3" type="Copy" precision="FP32">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>100</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>100</dim>
                    </port>
                </output>
        </layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>

		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="1" from-port="2" to-layer="2" to-port="2"/>
		<edge from-layer="1" from-port="3" to-layer="2" to-port="3"/>
		<edge from-layer="1" from-port="4" to-layer="2" to-port="4"/>
		<edge from-layer="1" from-port="5" to-layer="2" to-port="5"/>
		<edge from-layer="1" from-port="6" to-layer="2" to-port="6"/>
		<edge from-layer="1" from-port="7" to-layer="2" to-port="7"/>
		<edge from-layer="1" from-port="8" to-layer="2" to-port="8"/>
		<edge from-layer="1" from-port="9" to-layer="2" to-port="9"/>
		<edge from-layer="1" from-port="10" to-layer="2" to-port="10"/>

        <edge from-layer="2" from-port="11" to-layer="3" to-port="0"/>
	</edges>
</net>
        )V0G0N";
    }

    std::string SplitToConcatWith10By1InputsNotAlignedNoFCWithOutCopy () {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model_split_to_concat_with_10_inputs" version="2">
	<layers>
		<layer id="0" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="5">
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="6">
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="7">
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="8">
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="9">
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="10">
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="5">
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="6">
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="7">
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="8">
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="9">
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="10">
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="11">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
        <layer name="output_copy" id="3" type="Copy" precision="FP32">
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
        </layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>

		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="1" from-port="2" to-layer="2" to-port="2"/>
		<edge from-layer="1" from-port="3" to-layer="2" to-port="3"/>
		<edge from-layer="1" from-port="4" to-layer="2" to-port="4"/>
		<edge from-layer="1" from-port="5" to-layer="2" to-port="5"/>
		<edge from-layer="1" from-port="6" to-layer="2" to-port="6"/>
		<edge from-layer="1" from-port="7" to-layer="2" to-port="7"/>
		<edge from-layer="1" from-port="8" to-layer="2" to-port="8"/>
		<edge from-layer="1" from-port="9" to-layer="2" to-port="9"/>
		<edge from-layer="1" from-port="10" to-layer="2" to-port="10"/>

        <edge from-layer="2" from-port="11" to-layer="3" to-port="0"/>
	</edges>
</net>
        )V0G0N";
    }

    std::string SplitToConcatWith10InputsAlignedNoFC () {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model_split_to_concat_with_10_inputs" version="2">
	<layers>
		<layer id="0" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>320</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>320</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="5">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="6">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="7">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="8">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="9">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="10">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="5">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="6">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="7">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="8">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="9">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="10">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="11">
					<dim>1</dim>
					<dim>320</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>

		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="1" from-port="2" to-layer="2" to-port="2"/>
		<edge from-layer="1" from-port="3" to-layer="2" to-port="3"/>
		<edge from-layer="1" from-port="4" to-layer="2" to-port="4"/>
		<edge from-layer="1" from-port="5" to-layer="2" to-port="5"/>
		<edge from-layer="1" from-port="6" to-layer="2" to-port="6"/>
		<edge from-layer="1" from-port="7" to-layer="2" to-port="7"/>
		<edge from-layer="1" from-port="8" to-layer="2" to-port="8"/>
		<edge from-layer="1" from-port="9" to-layer="2" to-port="9"/>
		<edge from-layer="1" from-port="10" to-layer="2" to-port="10"/>
	</edges>
</net>
        )V0G0N";
    }

    std::string SplitToConcatWith10InputsAlignedNoFCWithInCopyWithOutCopy () {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model_split_to_concat_with_10_inputs" version="2">
	<layers>
		<layer id="0" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>320</dim>
				</port>
			</output>
		</layer>
        <layer name="input_copy" id="4" type="Copy" precision="FP32">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>320</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>320</dim>
                    </port>
                </output>
        </layer>
		<layer id="1" name="split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>320</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="5">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="6">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="7">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="8">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="9">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="10">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="5">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="6">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="7">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="8">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="9">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="10">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="11">
					<dim>1</dim>
					<dim>320</dim>
				</port>
			</output>
		</layer>
        <layer name="output_copy" id="3" type="Copy" precision="FP32">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>320</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>320</dim>
                    </port>
                </output>
        </layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>

        <edge from-layer="4" from-port="1" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="1" from-port="2" to-layer="2" to-port="2"/>
		<edge from-layer="1" from-port="3" to-layer="2" to-port="3"/>
		<edge from-layer="1" from-port="4" to-layer="2" to-port="4"/>
		<edge from-layer="1" from-port="5" to-layer="2" to-port="5"/>
		<edge from-layer="1" from-port="6" to-layer="2" to-port="6"/>
		<edge from-layer="1" from-port="7" to-layer="2" to-port="7"/>
		<edge from-layer="1" from-port="8" to-layer="2" to-port="8"/>
		<edge from-layer="1" from-port="9" to-layer="2" to-port="9"/>
		<edge from-layer="1" from-port="10" to-layer="2" to-port="10"/>

        <edge from-layer="2" from-port="11" to-layer="3" to-port="0"/>
	</edges>
</net>
        )V0G0N";
    }

    std::string SplitToConcatWith10InputsNotAlignedWithFC () {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model_split_to_concat_with_10_inputs" version="2">
	<layers>
		<layer id="0" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>100</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>100</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="5">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="6">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="7">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="8">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="9">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="10">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="5">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="6">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="7">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="8">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="9">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="10">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="11">
					<dim>1</dim>
					<dim>100</dim>
				</port>
			</output>
		</layer>
        <layer id="3" name="fc" precision="FP32" type="FullyConnected">
			<data out-size="10"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>100</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="4040"/>
				<biases offset="4000" size="40"/>
			</blobs>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>

		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="1" from-port="2" to-layer="2" to-port="2"/>
		<edge from-layer="1" from-port="3" to-layer="2" to-port="3"/>
		<edge from-layer="1" from-port="4" to-layer="2" to-port="4"/>
		<edge from-layer="1" from-port="5" to-layer="2" to-port="5"/>
		<edge from-layer="1" from-port="6" to-layer="2" to-port="6"/>
		<edge from-layer="1" from-port="7" to-layer="2" to-port="7"/>
		<edge from-layer="1" from-port="8" to-layer="2" to-port="8"/>
		<edge from-layer="1" from-port="9" to-layer="2" to-port="9"/>
		<edge from-layer="1" from-port="10" to-layer="2" to-port="10"/>

        <edge from-layer="2" from-port="11" to-layer="3" to-port="0"/>
	</edges>
</net>
        )V0G0N";
    }

    std::string SplitToConcatWith10InputsAlignedWithFC () {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model_split_to_concat_with_10_inputs" version="2">
	<layers>
		<layer id="0" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>320</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>320</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="5">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="6">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="7">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="8">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="9">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="10">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="5">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="6">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="7">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="8">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="9">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="10">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="11">
					<dim>1</dim>
					<dim>320</dim>
				</port>
			</output>
		</layer>
       <layer id="3" name="fc" precision="FP32" type="FullyConnected">
			<data out-size="32"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>320</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="41088"/>
				<biases offset="40960" size="128"/>
			</blobs>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>

		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="1" from-port="2" to-layer="2" to-port="2"/>
		<edge from-layer="1" from-port="3" to-layer="2" to-port="3"/>
		<edge from-layer="1" from-port="4" to-layer="2" to-port="4"/>
		<edge from-layer="1" from-port="5" to-layer="2" to-port="5"/>
		<edge from-layer="1" from-port="6" to-layer="2" to-port="6"/>
		<edge from-layer="1" from-port="7" to-layer="2" to-port="7"/>
		<edge from-layer="1" from-port="8" to-layer="2" to-port="8"/>
		<edge from-layer="1" from-port="9" to-layer="2" to-port="9"/>
		<edge from-layer="1" from-port="10" to-layer="2" to-port="10"/>

        <edge from-layer="2" from-port="11" to-layer="3" to-port="0"/>
	</edges>
</net>
        )V0G0N";
    }

    std::string SplitToConcatWith10InputsAlignedWithFCWithInCopy () {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model_split_to_concat_with_10_inputs" version="2">
	<layers>
		<layer id="0" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>320</dim>
				</port>
			</output>
		</layer>
        <layer name="input_copy" id="4" type="Copy" precision="FP32">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>320</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>320</dim>
                    </port>
                </output>
        </layer>
		<layer id="1" name="split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>320</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="5">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="6">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="7">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="8">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="9">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="10">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="5">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="6">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="7">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="8">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="9">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="10">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="11">
					<dim>1</dim>
					<dim>320</dim>
				</port>
			</output>
		</layer>
       <layer id="3" name="fc" precision="FP32" type="FullyConnected">
			<data out-size="32"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>320</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="41088"/>
				<biases offset="40960" size="128"/>
			</blobs>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>

        <edge from-layer="4" from-port="1" to-layer="1" to-port="0"/>

		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="1" from-port="2" to-layer="2" to-port="2"/>
		<edge from-layer="1" from-port="3" to-layer="2" to-port="3"/>
		<edge from-layer="1" from-port="4" to-layer="2" to-port="4"/>
		<edge from-layer="1" from-port="5" to-layer="2" to-port="5"/>
		<edge from-layer="1" from-port="6" to-layer="2" to-port="6"/>
		<edge from-layer="1" from-port="7" to-layer="2" to-port="7"/>
		<edge from-layer="1" from-port="8" to-layer="2" to-port="8"/>
		<edge from-layer="1" from-port="9" to-layer="2" to-port="9"/>
		<edge from-layer="1" from-port="10" to-layer="2" to-port="10"/>

        <edge from-layer="2" from-port="11" to-layer="3" to-port="0"/>
	</edges>
</net>
        )V0G0N";
    }

    std::string SplitToConcatWith3By512InputsWithOutCopy () {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model_split_to_concat_with_3_inputs_align" version="2">
	<layers>
		<layer id="0" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>1536</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1536</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>512</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>512</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>512</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="1">
					<dim>1</dim>
					<dim>512</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>512</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>512</dim>
				</port>
			</input>
			<output>
				<port id="4">
					<dim>1</dim>
					<dim>1536</dim>
				</port>
			</output>
		</layer>
        <layer name="output_copy" id="3" type="Copy" precision="FP32">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>1536</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>1536</dim>
                    </port>
                </output>
        </layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>

		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="1" from-port="2" to-layer="2" to-port="2"/>
		<edge from-layer="1" from-port="3" to-layer="2" to-port="3"/>

        <edge from-layer="2" from-port="4" to-layer="3" to-port="0"/>
	</edges>
</net>
        )V0G0N";
    }

    std::string ReshapeConvolutionLessThan48Filters() {
        return R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="frozen_model" version="4">
	<layers>
		<layer id="0" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>800</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="conv1d_1/convolution/ExpandDims" precision="FP32" type="Reshape">
			<data dim="1,4,1,200"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>800</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>4</dim>
					<dim>1</dim>
					<dim>200</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="conv1d_1/convolution/Conv1D" precision="FP32" type="Convolution">
			<data auto_pad="valid" dilations="1,1" group="1" kernel="1,2" output="16" pads_begin="0,0" pads_end="0,0" strides="1,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>4</dim>
					<dim>1</dim>
					<dim>200</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>100</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="512"/>
			</blobs>
		</layer>
		<layer id="3" name="conv1d_1/convolution/RevertDims" precision="FP32" type="Reshape">
			<data dim="1,1600"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>100</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1600</dim>
				</port>
			</output>
		</layer>
		<layer name="output_copy" id="4" type="Copy" precision="FP32">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>1600</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>1600</dim>
                    </port>
                </output>
            	</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
		<edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
		<edge from-layer="3" from-port="1" to-layer="4" to-port="0"/>
	</edges>
</net>
    )V0G0N";
    }

std::string EltwiseAfterSplitModel(int tensor_size, bool bMul) {
    std::string ir = R"V0G0N(
    <Net Name="FCWithPaddingAfterSplitModel" version="2" precision="FP32" batch="1">
        <layers>
            <layer name="input_1" type="input" id="0" precision="FP32">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>__TZ2__</dim>
                    </port>
                </output>
            </layer>
            <layer name="Split_1" type="Split" id="1" precision="FP32">
                <data axis="1" />
                <input>
                    <port id="0">
                        <!--connected to input-->
                        <dim>1</dim>
                        <dim>__TZ2__</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <!--connected to eltwise-->
                        <dim>1</dim>
                        <dim>__TZ__</dim>
                    </port>
                    <port id="2">
                        <!--connected to fc-->
                        <dim>1</dim>
                        <dim>__TZ__</dim>
                    </port>
                </output>
            </layer>
            <layer name="Eltwise_8" type="Eltwise" id="21" precision="FP32">
                <data operation="sum" />
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>__TZ__</dim>
                    </port>
                    <port id="1">
                        <dim>1</dim>
                        <dim>__TZ__</dim>
                    </port>
                </input>
                <output>
                    <port id="2">
                        <dim>1</dim>
                        <dim>__TZ__</dim>
                    </port>
                </output>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
            <edge from-layer="1" from-port="1" to-layer="21" to-port="0" />
            <edge from-layer="1" from-port="2" to-layer="21" to-port="1" />
        </edges>
    </Net>
    )V0G0N";
    if (bMul) {
        REPLACE_WITH_STR(ir, "sum", "mul");
    }
    REPLACE_WITH_NUM(ir, "__TZ__", tensor_size);
    REPLACE_WITH_NUM(ir, "__TZ2__", 2 * tensor_size);


    return ir;
}

std::string TwoInputsModelForIO() {
    return R"V0G0N(
<?xml version="1.0" ?>
<net name="multiInputs2" version="7">
	<layers>
		<layer id="0" name="Placeholder" type="Input">
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="Placeholder_1" type="Input">
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="Add" type="Eltwise">
			<data operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="Layer_output" type="Activation">
			<data type="tanh"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
		<edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
	</edges>
</net>
    )V0G0N";
}

std::string PermuteModelForIO() {
    return R"V0G0N(
<?xml version="1.0" ?>
<net name="permute" version="7">
	<layers>
		<layer id="0" name="Placeholder" type="Input" version="opset1">
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>640</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="Reshape/Cast_1238_const" type="Const" version="opset1">
			<output>
				<port id="1" precision="I32">
					<dim>3</dim>
				</port>
			</output>
			<blobs>
				<custom offset="0" precision="I32" size="12"/>
			</blobs>
		</layer>
		<layer id="2" name="Reshape" type="Reshape" version="opset1">
			<data special_zero="False"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>640</dim>
				</port>
				<port id="1">
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>160</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="transpose" type="Permute" version="opset1">
			<data order="0,2,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>160</dim>
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
					<dim>160</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="Layer_output" type="Reshape" version="opset1">
			<data special_zero="False"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>4</dim>
					<dim>160</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>640</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
		<edge from-layer="3" from-port="1" to-layer="5" to-port="0"/>
	</edges>
</net>
    )V0G0N";
}

}  // namespace GNATestIRs
