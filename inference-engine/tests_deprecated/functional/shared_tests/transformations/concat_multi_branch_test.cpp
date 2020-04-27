// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"

std::string ConcatMultiBranchTestModel::getModel(SingleLayerTransformationsTestParams& p) const {
    std::string layers = layersTemplate;
    // TODO: hard-coded values

    size_t totalOffset = 0;

    REPLACE_WITH_NUM(layers, "DATA_CONST_INPUT_LOW_OFFSET_1", totalOffset);
    totalOffset += 4;
    REPLACE_WITH_NUM(layers, "DATA_CONST_INPUT_HIGHT_OFFSET_1", totalOffset);
    totalOffset += 4;
    REPLACE_WITH_NUM(layers, "DATA_CONST_OUTPUT_LOW_OFFSET_1", totalOffset);
    totalOffset += 4;
    REPLACE_WITH_NUM(layers, "DATA_CONST_OUTPUT_HIGH_OFFSET_1", totalOffset);
    totalOffset += 4;

    REPLACE_WITH_NUM(layers, "DATA_CONST_INPUT_LOW_OFFSET_2", totalOffset);
    totalOffset += 4;
    REPLACE_WITH_NUM(layers, "DATA_CONST_INPUT_HIGHT_OFFSET_2", totalOffset);
    totalOffset += 4;
    REPLACE_WITH_NUM(layers, "DATA_CONST_OUTPUT_LOW_OFFSET_2", totalOffset);
    totalOffset += 4;
    REPLACE_WITH_NUM(layers, "DATA_CONST_OUTPUT_HIGH_OFFSET_2", totalOffset);
    totalOffset += 4;

    REPLACE_WITH_NUM(layers, "DATA_CONST_INPUT_LOW_OFFSET_3", totalOffset);
    totalOffset += 4;
    REPLACE_WITH_NUM(layers, "DATA_CONST_INPUT_HIGHT_OFFSET_3", totalOffset);
    totalOffset += 4;
    REPLACE_WITH_NUM(layers, "DATA_CONST_OUTPUT_LOW_OFFSET_3", totalOffset);
    totalOffset += 4;
    REPLACE_WITH_NUM(layers, "DATA_CONST_OUTPUT_HIGH_OFFSET_3", totalOffset);
    totalOffset += 4;

    REPLACE_WITH_NUM(layers, "WEIGHTS_CONST_INPUT_OFFSET", totalOffset);
    totalOffset += 6 * 6 * 3 * 3 * 4;
    REPLACE_WITH_NUM(layers, "WEIGHTS_CONST_INPUT_SIZE", 6 * 6 * 3 * 3 * 4);

    REPLACE_WITH_NUM(layers, "WEIGHTS_CONST_INPUT_LOW_OFFSET", totalOffset);
    totalOffset += 4;
    REPLACE_WITH_NUM(layers, "WEIGHTS_CONST_INPUT_HIGHT_OFFSET", totalOffset);
    totalOffset += 4;
    REPLACE_WITH_NUM(layers, "WEIGHTS_CONST_OUTPUT_LOW_OFFSET", totalOffset);
    totalOffset += 4;
    REPLACE_WITH_NUM(layers, "WEIGHTS_CONST_OUTPUT_HIGH_OFFSET", totalOffset);
    totalOffset += 4;

    REPLACE_WITH_NUM(layers, "BIASES_CONST_OFFSET", totalOffset);
    totalOffset += 6 * 4;
    REPLACE_WITH_NUM(layers, "BIASES_CONST_SIZE", 6 * 4);

    REPLACE_WITH_NUM(layers, "DATA_CONST_INPUT_LOW_OFFSET_4", totalOffset);
    totalOffset += 4;
    REPLACE_WITH_NUM(layers, "DATA_CONST_INPUT_HIGHT_OFFSET_4", totalOffset);
    totalOffset += 4;
    REPLACE_WITH_NUM(layers, "DATA_CONST_OUTPUT_LOW_OFFSET_4", totalOffset);
    totalOffset += 4;
    REPLACE_WITH_NUM(layers, "DATA_CONST_OUTPUT_HIGH_OFFSET_4", totalOffset);
    totalOffset += 4;

    REPLACE_WITH_NUM(layers, "DEQUANTIZE_SCALESHIFT_WEIGHTS_OFFSET", totalOffset);
    totalOffset += 24;
    REPLACE_WITH_NUM(layers, "DEQUANTIZE_SCALESHIFT_BIASES_OFFSET", totalOffset);
    totalOffset += 24;

    REPLACE_WITH_STR(layers, "_PR_", p._network_precision);

    const std::string model = IRTemplateGenerator::getIRTemplate(
        "TransformationsTest",
        { { 1lu, 3, 299, 299 }, { 1lu, 3, 299, 299 } },
        p._network_precision,
        layers,
        edgesTemplate,
        6);

    return model;
}

std::string ConcatMultiBranchTestModel::getName() const {
    return "ConcatMultiBranchTestModel";
}

bool ConcatMultiBranchTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    LowPrecisionTransformer transformer(LowPrecisionTransformer::getAllTransformations(params));
    transformer.transform(network);
    return true;
}

void ConcatMultiBranchTestModel::resetTransformation(CNNNetwork& network) const {
    fillData(getLayer(network, "branch1/dataConstInputLow1"), 255.0 / 200.0, "custom");
    fillData(getLayer(network, "branch1/dataConstInputHigh1"), 255.0 / 100.0, "custom");
    fillData(getLayer(network, "branch1/dataConstOutputLow1"), 255.0 / 200.0, "custom");
    fillData(getLayer(network, "branch1/dataConstOutputHigh1"), 255.0 / 100.0, "custom");

    fillData(getLayer(network, "branch1/dataConstInputLow2"), 255.0 / 400.0, "custom");
    fillData(getLayer(network, "branch1/dataConstInputHigh2"), 255.0 / 200.0, "custom");
    fillData(getLayer(network, "branch1/dataConstOutputLow2"), 255.0 / 400.0, "custom");
    fillData(getLayer(network, "branch1/dataConstOutputHigh2"), 255.0 / 200.0, "custom");

    fillData(getLayer(network, "branch2/dataConstInputLow3"), 255.0 / 200.0, "custom");
    fillData(getLayer(network, "branch2/dataConstInputHigh3"), 255.0 / 100.0, "custom");
    fillData(getLayer(network, "branch2/dataConstOutputLow3"), 255.0 / 200.0, "custom");
    fillData(getLayer(network, "branch2/dataConstOutputHigh3"), 255.0 / 100.0, "custom");

    fillData(getLayer(network, "branch2/weightsConstInput"), 0.0, "custom");
    fillData(getLayer(network, "branch2/weightsConstInputLow"), 0.0, "custom");
    fillData(getLayer(network, "branch2/weightsConstInputHigh"), 255.0 / 200.0, "custom");
    fillData(getLayer(network, "branch2/weightsConstOutputLow"), 0.0, "custom");
    fillData(getLayer(network, "branch2/weightsConstOutputHigh"), 255.0 / 200.0, "custom");

    fillData(getLayer(network, "branch2/biasesConst"), { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });

    fillData(getLayer(network, "branch2/dataConstInputLow4"), 255.0 / 800.0, "custom");
    fillData(getLayer(network, "branch2/dataConstInputHigh4"), 255.0 / 400.0, "custom");
    fillData(getLayer(network, "branch2/dataConstOutputLow4"), 255.0 / 800.0, "custom");
    fillData(getLayer(network, "branch2/dataConstOutputHigh4"), 255.0 / 400.0, "custom");
}

const std::string ConcatMultiBranchTestModel::layersTemplate = R"V0G0N(
<layer name="branch1/dataConstInputLow1" type="Const" precision="_PR_" id="102">
    <output>
        <port id="0">
            <dim>1</dim>
        </port>
    </output>
    <blobs>
        <custom offset="DATA_CONST_INPUT_LOW_OFFSET_1" size="4"/>
    </blobs>
</layer>
<layer name="branch1/dataConstInputHigh1" type="Const" precision="_PR_" id="103">
    <output>
        <port id="0">
            <dim>1</dim>
        </port>
    </output>
    <blobs>
        <custom offset="DATA_CONST_INPUT_HIGHT_OFFSET_1" size="4"/>
    </blobs>
</layer>

<layer name="branch1/dataConstOutputLow1" type="Const" precision="_PR_" id="104">
    <output>
        <port id="0">
            <dim>1</dim>
        </port>
    </output>
    <blobs>
        <custom offset="DATA_CONST_OUTPUT_LOW_OFFSET_1" size="4"/>
    </blobs>
</layer>
<layer name="branch1/dataConstOutputHigh1" type="Const" precision="_PR_" id="105">
    <output>
        <port id="0">
            <dim>1</dim>
        </port>
    </output>
    <blobs>
        <custom offset="DATA_CONST_OUTPUT_HIGH_OFFSET_1" size="4"/>
    </blobs>
</layer>

<layer name="branch1/dataFakeQuantize1" type="FakeQuantize" precision="_PR_" id="106">
    <data levels="256" />
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>3</dim>
            <dim>299</dim>
            <dim>299</dim>
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
        <port id="4">
            <dim>1</dim>
        </port>
    </input>
    <output>
        <port id="5">
            <dim>1</dim>
            <dim>3</dim>
            <dim>299</dim>
            <dim>299</dim>
        </port>
    </output>
</layer>

<layer name="branch1/dataConstInputLow2" type="Const" precision="_PR_" id="107">
    <output>
        <port id="0">
            <dim>1</dim>
        </port>
    </output>
    <blobs>
        <custom offset="DATA_CONST_INPUT_LOW_OFFSET_2" size="4"/>
    </blobs>
</layer>
<layer name="branch1/dataConstInputHigh2" type="Const" precision="_PR_" id="108">
    <output>
        <port id="0">
            <dim>1</dim>
        </port>
    </output>
    <blobs>
        <custom offset="DATA_CONST_INPUT_HIGHT_OFFSET_2" size="4"/>
    </blobs>
</layer>

<layer name="branch1/dataConstOutputLow2" type="Const" precision="_PR_" id="109">
        <output>
            <port id="0">
                <dim>1</dim>
            </port>
        </output>
    <blobs>
        <custom offset="DATA_CONST_OUTPUT_LOW_OFFSET_2" size="4"/>
    </blobs>
</layer>
<layer name="branch1/dataConstOutputHigh2" type="Const" precision="_PR_" id="110">
        <output>
            <port id="0">
                <dim>1</dim>
            </port>
        </output>
    <blobs>
        <custom offset="DATA_CONST_OUTPUT_HIGH_OFFSET_2" size="4"/>
    </blobs>
</layer>


<layer name="branch1/dataFakeQuantize2" type="FakeQuantize" precision="_PR_" id="111">
    <data levels="256" />
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>3</dim>
            <dim>299</dim>
            <dim>299</dim>
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
        <port id="4">
            <dim>1</dim>
        </port>
    </input>
    <output>
        <port id="5">
            <dim>1</dim>
            <dim>3</dim>
            <dim>299</dim>
            <dim>299</dim>
        </port>
    </output>
</layer>

<layer name="branch1/concat" type="Concat" precision="_PR_" id="113">
    <data axis="1" />
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>3</dim>
            <dim>299</dim>
            <dim>299</dim>
        </port>
        <port id="1">
            <dim>1</dim>
            <dim>3</dim>
            <dim>299</dim>
            <dim>299</dim>
        </port>

    </input>
    <output>
        <port id="2">
            <dim>1</dim>
            <dim>6</dim>
            <dim>299</dim>
            <dim>299</dim>
        </port>
    </output>
</layer>

<layer name="branch2/dataConstInputLow3" type="Const" precision="_PR_" id="207">
        <output>
            <port id="0">
                <dim>1</dim>
            </port>
        </output>
    <blobs>
        <custom offset="DATA_CONST_INPUT_LOW_OFFSET_3" size="4"/>
    </blobs>
</layer>
<layer name="branch2/dataConstInputHigh3" type="Const" precision="_PR_" id="208">
        <output>
            <port id="0">
                <dim>1</dim>
            </port>
        </output>
    <blobs>
        <custom offset="DATA_CONST_INPUT_HIGHT_OFFSET_3" size="4"/>
    </blobs>
</layer>

<layer name="branch2/dataConstOutputLow3" type="Const" precision="_PR_" id="209">
        <output>
            <port id="0">
                <dim>1</dim>
            </port>
        </output>
    <blobs>
        <custom offset="DATA_CONST_OUTPUT_LOW_OFFSET_3" size="4"/>
    </blobs>
</layer>
<layer name="branch2/dataConstOutputHigh3" type="Const" precision="_PR_" id="210">
        <output>
            <port id="0">
                <dim>1</dim>
            </port>
        </output>
    <blobs>
        <custom offset="DATA_CONST_OUTPUT_HIGH_OFFSET_3" size="4"/>
    </blobs>
</layer>


<layer name="branch2/dataFakeQuantize3" type="FakeQuantize" precision="_PR_" id="211">
    <data levels="256" />
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>6</dim>
            <dim>299</dim>
            <dim>299</dim>
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
        <port id="4">
            <dim>1</dim>
        </port>
    </input>
    <output>
        <port id="5">
            <dim>1</dim>
            <dim>6</dim>
            <dim>299</dim>
            <dim>299</dim>
        </port>
    </output>
</layer>


<layer name="branch2/weightsConstInput" type="Const" precision="_PR_" id="212">
    <output>
        <port id="0">
            <dim>6</dim>
            <dim>6</dim>
            <dim>3</dim>
            <dim>3</dim>
        </port>
    </output>
    <blobs>
        <custom offset="WEIGHTS_CONST_INPUT_OFFSET" size="WEIGHTS_CONST_INPUT_SIZE"/>
    </blobs>
</layer>
<layer name="branch2/weightsConstInputLow" type="Const" precision="_PR_" id="213">
    <output>
        <port id="0">
            <dim>1</dim>
        </port>
    </output>
    <blobs>
        <custom offset="WEIGHTS_CONST_INPUT_LOW_OFFSET" size="4"/>
    </blobs>
</layer>
<layer name="branch2/weightsConstInputHigh" type="Const" precision="_PR_" id="214">
    <output>
        <port id="0">
            <dim>1</dim>
        </port>
    </output>
    <blobs>
        <custom offset="WEIGHTS_CONST_INPUT_HIGHT_OFFSET" size="4"/>
    </blobs>
</layer>

<layer name="branch2/weightsConstOutputLow" type="Const" precision="_PR_" id="215">
    <output>
    <port id="0">
        <dim>1</dim>
    </port>
    </output>
    <blobs>
        <custom offset="WEIGHTS_CONST_OUTPUT_LOW_OFFSET" size="4"/>
    </blobs>
</layer>
<layer name="branch2/weightsConstOutputHigh" type="Const" precision="_PR_" id="216">
    <output>
        <port id="0">
            <dim>1</dim>
        </port>
    </output>
    <blobs>
        <custom offset="WEIGHTS_CONST_OUTPUT_HIGH_OFFSET" size="4"/>
    </blobs>
</layer>


<layer name="branch2/weightsFakeQuantize" type="FakeQuantize" precision="_PR_" id="218">
    <data levels="256" />
    <input>
        <port id="0">
            <dim>6</dim>
            <dim>6</dim>
            <dim>3</dim>
            <dim>3</dim>
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
        <port id="4">
            <dim>1</dim>
        </port>
    </input>
    <output>
        <port id="5">
            <dim>6</dim>
            <dim>6</dim>
            <dim>3</dim>
            <dim>3</dim>
        </port>
    </output>
</layer>

<layer name="branch2/biasesConst" type="Const" precision="_PR_" id="219">
    <output>
        <port id="0">
            <dim>6</dim>
        </port>
    </output>
    <blobs>
        <custom offset="BIASES_CONST_OFFSET" size="BIASES_CONST_SIZE"/>
    </blobs>
</layer>


<layer name="branch2/convolution" precision="_PR_" type="Convolution" id="220">
                <data auto_pad="valid" dilations="1,1" group="1" kernel="3,3" output="6" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
                <input>
                        <port id="0">
                                <dim>1</dim>
                                <dim>6</dim>
                                <dim>299</dim>
                                <dim>299</dim>
                        </port>
                        <port id="1">
                                <dim>6</dim>
                                <dim>6</dim>
                                <dim>3</dim>
                                <dim>3</dim>
                        </port>
                        <port id="2">
                                <dim>6</dim>
                        </port>
                </input>
                <output>
                        <port id="3">
                                <dim>1</dim>
                                <dim>6</dim>
                                <dim>299</dim>
                                <dim>299</dim>
                        </port>
                </output>
        </layer>

<layer name="branch2/dataConstInputLow4" type="Const" precision="_PR_" id="222">
    <output>
        <port id="0">
            <dim>1</dim>
        </port>
    </output>
    <blobs>
        <custom offset="DATA_CONST_INPUT_LOW_OFFSET_4" size="4"/>
    </blobs>
</layer>
<layer name="branch2/dataConstInputHigh4" type="Const" precision="_PR_" id="223">
    <output>
        <port id="0">
            <dim>1</dim>
        </port>
    </output>
    <blobs>
        <custom offset="DATA_CONST_INPUT_HIGHT_OFFSET_4" size="4"/>
    </blobs>
</layer>

<layer name="branch2/dataConstOutputLow4" type="Const" precision="_PR_" id="224">
    <output>
        <port id="0">
            <dim>1</dim>
        </port>
    </output>
    <blobs>
        <custom offset="DATA_CONST_OUTPUT_LOW_OFFSET_4" size="4"/>
    </blobs>
</layer>
<layer name="branch2/dataConstOutputHigh4" type="Const" precision="_PR_" id="225">
    <output>
        <port id="0">
            <dim>1</dim>
        </port>
    </output>
    <blobs>
        <custom offset="DATA_CONST_OUTPUT_HIGH_OFFSET_4" size="4"/>
    </blobs>
</layer>

<layer name="branch2/dataFakeQuantize4" type="FakeQuantize" precision="_PR_" id="226">
    <data levels="256" />
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>6</dim>
            <dim>299</dim>
            <dim>299</dim>
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
        <port id="4">
            <dim>1</dim>
        </port>
    </input>
    <output>
        <port id="5">
            <dim>1</dim>
            <dim>6</dim>
            <dim>299</dim>
            <dim>299</dim>
        </port>
    </output>
</layer>

<layer name="branch2/concat" type="Concat" precision="_PR_" id="227">
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>6</dim>
            <dim>299</dim>
            <dim>299</dim>
        </port>
        <port id="1">
            <dim>1</dim>
            <dim>6</dim>
            <dim>299</dim>
            <dim>299</dim>
        </port>

    </input>
    <output>
        <port id="2">
            <dim>1</dim>
            <dim>12</dim>
            <dim>299</dim>
            <dim>299</dim>
        </port>
    </output>
</layer>


<layer name="outputPower" type="Power" precision="_PR_" id="300">
    <power_data power="1" scale="1" shift="0"/>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>12</dim>
            <dim>299</dim>
            <dim>299</dim>
        </port>
    </input>
    <output>
        <port id="1">
            <dim>1</dim>
            <dim>12</dim>
            <dim>299</dim>
            <dim>299</dim>
        </port>
    </output>
</layer>

)V0G0N";