// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <string>
#include <unordered_map>

#include <ie_core.hpp>
#include "cpp_interfaces/impl/ie_plugin_internal.hpp"

#include "common/low_precision_tests_utils.hpp"
#include "low_precision_transformations/transformer.hpp"
#include "low_precision_transformations/convolution.hpp"
#include "low_precision_transformations/network_helper.hpp"
#include "low_precision_transformations/eltwise.hpp"

#include "tests_common.hpp"
#include "ir_gen_helper.hpp"

using namespace ::testing;
using namespace InferenceEngine;
using namespace InferenceEngine::details;
using namespace single_layer_tests;

inline void fillDataMy(CNNLayerPtr layer, std::vector<int> values, const std::string& blobName = "") {
    if (layer == nullptr) {
        THROW_IE_EXCEPTION << "layer is nullable";
    }
    if (blobName.empty() && (layer->blobs.size() != 1)) {
        THROW_IE_EXCEPTION << "several blobs";
    }

    Blob::Ptr blob = blobName.empty() ? layer->blobs.begin()->second : layer->blobs[blobName];
    if (blob->size() != values.size()) {
        THROW_IE_EXCEPTION << "values size is not correct";
    }

    int* buffer = blob->buffer().as<int*>();
    for (size_t i = 0; i < blob->size(); i++) {
        buffer[i] = values[i];
    }
}

/**
 * @brief base class for test model.
  */
class SingleLayerTransformationsTestParams;

class SingleLayerTestModel {
public:
    typedef std::shared_ptr<SingleLayerTestModel> Ptr;

    LowPrecisionTransformations getLowPrecisionTransformations(const LayerTransformation::Params& params) const;
    LowPrecisionTransformer getLowPrecisionTransformer(const LayerTransformation::Params& params) const;

    virtual std::string getModel(SingleLayerTransformationsTestParams& p) const = 0;
    virtual std::string getName() const = 0;

    virtual void initInput(Blob::Ptr input) const {}
    virtual float getZeroThreshold() const {
        return 1e-7;
    }
    virtual bool transform(CNNNetwork& network, LayerTransformation::Params& params) const = 0;
    virtual void resetTransformation(CNNNetwork& network) const = 0;
    virtual std::unordered_set<std::string> getNotTransformedLayers() const {
        return {};
    }

    virtual float getThreshold(const std::string& device_name, const Precision precision, LayerTransformation::Params& params) const {
        return precision == Precision::FP16 ? 0.0005f : 0.0003f;
    }

protected:
    // TODO: pass as parameter: 22403
    const std::string device_name = "CPU";
};

class SingleLayerTransformationsTestParams {
public:
    SingleLayerTransformationsTestParams(
        const std::string& name,
        SingleLayerTestModel::Ptr model,
        const std::vector<std::vector<size_t>>& inputDimensions,
        const std::vector<std::vector<size_t>>& outputDimensions,
        const std::string& network_precision = "FP32") :
        device_name(name),
        model(model),
        inputDimensions(inputDimensions),
        outputDimensions(outputDimensions),
        _network_precision(network_precision) {}

    const std::string device_name;
    SingleLayerTestModel::Ptr model;
    const std::vector<std::vector<size_t>> inputDimensions;
    const std::vector<std::vector<size_t>> outputDimensions;
    std::string _network_precision;


    static std::string getLowPrecisionTransformerSingleLayerTestName(testing::TestParamInfo<SingleLayerTransformationsTestParams> p) {
        return p.param.model->getName();
    }
};

class FullyConnectedAndScaleShiftsOnActivationsTestModel : public SingleLayerTestModel {
public:
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    void resetTransformation(CNNNetwork& network) const override;
};

class ResampleTestModel : public SingleLayerTestModel {
public:
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    void resetTransformation(CNNNetwork& network) const override;
};


class ConvolutionAndQuantizeOnActivationsAndWeightsBaseTestModel : public SingleLayerTestModel {
public:
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
};

class ConvolutionAndQuantizeOnSignedActivationsAndWeightsPositiveTestModel : public ConvolutionAndQuantizeOnActivationsAndWeightsBaseTestModel {
public:
    void resetTransformation(CNNNetwork& network) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
};

class ConvolutionAndQuantizeOnSignedActivationsAndWeightsNegativeTestModel : public ConvolutionAndQuantizeOnActivationsAndWeightsBaseTestModel {
public:
    void resetTransformation(CNNNetwork& network) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
};

class ConvolutionAndQuantizeOnUnsignedActivationsAndWeightsTestModel : public ConvolutionAndQuantizeOnActivationsAndWeightsBaseTestModel {
public:
    void resetTransformation(CNNNetwork& network) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
};

class ConvolutionAndQuantizeOnSignedActivationsAndInvertedWeightsTestModel : public ConvolutionAndQuantizeOnActivationsAndWeightsBaseTestModel {
public:
    void resetTransformation(CNNNetwork& network) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
};

class FakeQuantizeReshapePoolingTestModelWithConstants : public SingleLayerTestModel {
public:
    void resetTransformation(CNNNetwork& network) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
};

class FakeQuantizeReshapePoolingTestModelWithoutConstants : public SingleLayerTestModel {
public:
    void resetTransformation(CNNNetwork& network) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
};

class FakeQuantizeReshapeTestModelWithConstants : public SingleLayerTestModel {
public:
    void resetTransformation(CNNNetwork& network) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
};

class ScaleShiftToConvolutionTestModel : public SingleLayerTestModel {
public:
    void resetTransformation(CNNNetwork& network) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
};

class ScaleShiftToConvolutionAfterNotConcatIgnoreTestModel : public SingleLayerTestModel {
public:
    void resetTransformation(CNNNetwork& network) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
};

class ScaleShiftToConvolutionAfterFakeQuantizeIgnoreTestModel : public SingleLayerTestModel {
public:
    void resetTransformation(CNNNetwork& network) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
};

class ScaleShiftToConvolutionAfterConcatTestModel : public SingleLayerTestModel {
public:
    ScaleShiftToConvolutionAfterConcatTestModel(const bool scaleShiftIsOutput);
    void resetTransformation(CNNNetwork& network) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;

private:
    const bool scaleShiftIsOutput;
};

class FullyConnectedAndQuantizeTestModel : public SingleLayerTestModel {
public:
    void resetTransformation(CNNNetwork& network) const override {
        fillData(getLayer(network, "dataConstInputLow"), 63.5, "custom");
        fillData(getLayer(network, "dataConstInputHigh"), 127.0, "custom");
        fillData(getLayer(network, "dataConstOutputLow"), 63.5, "custom");
        fillData(getLayer(network, "dataConstOutputHigh"), 127.0, "custom");

        //fillData(getLayer(network, "weightsConstInput"), 3.0, "custom");
        fillDataWithInitValue(getLayer(network, "weightsConstInput"), "custom", 1.234);

        fillData(getLayer(network, "weightsConstInputLow"), -1.275 / 2.0, "custom");
        fillData(getLayer(network, "weightsConstInputHigh"), 1.275, "custom");
        fillData(getLayer(network, "weightsConstOutputLow"), -1.275 / 2.0, "custom");
        fillData(getLayer(network, "weightsConstOutputHigh"), 1.275, "custom");

        //fillData(getLayer(network, "biasesConvolutionConst"), 5.0, "custom");
        fillDataWithInitValue(getLayer(network, "biasesConvolutionConst"), "custom", 2.123);

        fillDataMy(getLayer(network, "reshapeConst"), { 1, -1 });
    }

    std::string getName() const override {
        return "FullyConnectedAndQuantizeTestModel";
    }

    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override {
        LowPrecisionTransformer transformer(LowPrecisionTransformer::getAllTransformations(params));
        transformer.transform(network);

        const std::vector<CNNLayerPtr> layers = CNNNetSortTopologically(network);

        const CNNLayerPtr convolution = layers[layers.size() - 2];
        if ((convolution->type != "FullyConnected") || (convolution->name != "fullyconnected_original")) {
            THROW_IE_EXCEPTION << "unexpected layer type '" << convolution->type << "' or name '" << convolution->name << "'";
        }

        const CNNLayerPtr dequantizationScaleShift = layers[layers.size() - 1];
        if ((dequantizationScaleShift->type != "ScaleShift") || (dequantizationScaleShift->name != "fullyconnected")) {
            THROW_IE_EXCEPTION << "unexpected layer type '" << dequantizationScaleShift->type << "' or name '" << dequantizationScaleShift->name << "'";
        }

        return true;
    }

    std::string getModel(SingleLayerTransformationsTestParams& p) const override {
        std::string layers = layersTemplate;
        auto inputSizes = p.inputDimensions.at(0);
        auto inBatch = inputSizes.at(0);
        auto inChannel = inputSizes.at(1);
        auto inX = inputSizes.at(2);
        auto inY = inputSizes.at(3);

        REPLACE_WITH_NUM(layers, "IN_BATCH", inBatch);
        REPLACE_WITH_NUM(layers, "IN_CHANNEL", inChannel);
        REPLACE_WITH_NUM(layers, "IN_X", inX);
        REPLACE_WITH_NUM(layers, "IN_Y", inY);
        REPLACE_WITH_NUM(layers, "RESHAPED_CH_X_Y", inChannel * inX * inY);

        auto outputSizes = p.outputDimensions.at(0);
        auto outBatch = outputSizes.at(0);
        auto outChannel = outputSizes.at(1);
        REPLACE_WITH_NUM(layers, "OUT_BATCH", outBatch);
        REPLACE_WITH_NUM(layers, "OUT_CHANNEL", outChannel);

        size_t totalOffset = 0;

        REPLACE_WITH_NUM(layers, "DATA_CONST_INPUT_LOW_OFFSET", totalOffset);
        totalOffset += 4;
        REPLACE_WITH_NUM(layers, "DATA_CONST_INPUT_HIGH_OFFSET", totalOffset);
        totalOffset += 4;
        REPLACE_WITH_NUM(layers, "DATA_CONST_OUTPUT_LOW_OFFSET", totalOffset);
        totalOffset += 4;
        REPLACE_WITH_NUM(layers, "DATA_CONST_OUTPUT_HIGH_OFFSET", totalOffset);
        totalOffset += 4;

        REPLACE_WITH_NUM(layers, "WEIGHTS_CONST_INPUT_OFFSET", totalOffset);
        totalOffset += inChannel * outChannel * 4;
        REPLACE_WITH_NUM(layers, "WEIGHTS_CONST_INPUT_LOW_OFFSET", totalOffset);
        totalOffset += 4;
        REPLACE_WITH_NUM(layers, "WEIGHTS_CONST_INPUT_HIGH_OFFSET", totalOffset);
        totalOffset += 4;
        REPLACE_WITH_NUM(layers, "WEIGHTS_CONST_OUTPUT_LOW_OFFSET", totalOffset);
        totalOffset += 4;
        REPLACE_WITH_NUM(layers, "WEIGHTS_CONST_OUTPUT_HIGH_OFFSET", totalOffset);
        totalOffset += 4;
        REPLACE_WITH_NUM(layers, "RESHAPE_CONST_OFFSET", totalOffset);
        totalOffset += 8;
        REPLACE_WITH_NUM(layers, "FULLYCONNECTED_BIASES_CONST_OFFSET", totalOffset);
        totalOffset += 128;


        const std::string model = IRTemplateGenerator::getIRTemplate(
            "TransformationsTest",
            p.inputDimensions,
            "FP32",
            layers,
            edgesTemplate,
            6);

        return model;
    }

private:
    const std::string layersTemplate = R"V0G0N(
        <layer name="inputPower" type="Power" precision="FP32" id="1">
            <power_data power="1" scale="1" shift="0"/>
            <input>
				<port id="0">
					<dim>IN_BATCH</dim>
					<dim>IN_CHANNEL</dim>
					<dim>IN_X</dim>
					<dim>IN_Y</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>IN_BATCH</dim>
					<dim>IN_CHANNEL</dim>
					<dim>IN_X</dim>
					<dim>IN_Y</dim>
				</port>
			</output>
        </layer>


        <layer id="9" name="dataConstInputLow" precision="FP32" type="Const">
			<output>
				<port id="1"/>
			</output>
			<blobs>
				<custom offset="DATA_CONST_INPUT_LOW_OFFSET" size="4"/>
			</blobs>
		</layer>
		<layer id="10" name="dataConstInputHigh" precision="FP32" type="Const">
			<output>
				<port id="1"/>
			</output>
			<blobs>
				<custom offset="DATA_CONST_INPUT_HIGH_OFFSET" size="4"/>
			</blobs>
		</layer>
		<layer id="11" name="dataConstOutputLow" precision="FP32" type="Const">
			<output>
				<port id="1"/>
			</output>
			<blobs>
				<custom offset="DATA_CONST_OUTPUT_LOW_OFFSET" size="4"/>
			</blobs>
		</layer>
		<layer id="12" name="dataConstOutputHigh" precision="FP32" type="Const">
			<output>
				<port id="1"/>
			</output>
			<blobs>
				<custom offset="DATA_CONST_OUTPUT_HIGH_OFFSET" size="4"/>
			</blobs>
		</layer>
		<layer id="13" name="dataFakeQuantize" precision="FP32" type="FakeQuantize">
			<data levels="256"/>
			<input>
				<port id="0">
					<dim>IN_BATCH</dim>
					<dim>IN_CHANNEL</dim>
					<dim>IN_X</dim>
					<dim>IN_Y</dim>
				</port>
				<port id="1"/>
				<port id="2"/>
				<port id="3"/>
				<port id="4"/>
			</input>
			<output>
				<port id="5">
					<dim>IN_BATCH</dim>
					<dim>IN_CHANNEL</dim>
					<dim>IN_X</dim>
					<dim>IN_Y</dim>
				</port>
			</output>
		</layer>
		<layer id="14" name="weightsConstInput" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>OUT_CHANNEL</dim>
					<dim>IN_CHANNEL</dim>
				</port>
			</output>
			<blobs>
				<custom offset="WEIGHTS_CONST_INPUT_OFFSET" size="4096"/>
			</blobs>
		</layer>
		<layer id="15" name="weightsConstInputLow" precision="FP32" type="Const">
			<output>
				<port id="1"/>
			</output>
			<blobs>
				<custom offset="WEIGHTS_CONST_INPUT_LOW_OFFSET" size="4"/>
			</blobs>
		</layer>
		<layer id="16" name="weightsConstInputHigh" precision="FP32" type="Const">
			<output>
				<port id="1"/>
			</output>
			<blobs>
				<custom offset="WEIGHTS_CONST_INPUT_HIGH_OFFSET" size="4"/>
			</blobs>
		</layer>
		<layer id="17" name="weightsConstOutputLow" precision="FP32" type="Const">
			<output>
				<port id="1"/>
			</output>
			<blobs>
				<custom offset="WEIGHTS_CONST_OUTPUT_LOW_OFFSET" size="4"/>
			</blobs>
		</layer>
		<layer id="18" name="weightsConstOutputHigh" precision="FP32" type="Const">
			<output>
				<port id="1"/>
			</output>
			<blobs>
				<custom offset="WEIGHTS_CONST_OUTPUT_HIGH_OFFSET" size="4"/>
			</blobs>
		</layer>
		<layer id="19" name="weightsFakeQuantize" precision="FP32" type="FakeQuantize">
			<data levels="256"/>
			<input>
				<port id="0">
					<dim>OUT_CHANNEL</dim>
					<dim>IN_CHANNEL</dim>
				</port>
				<port id="1"/>
				<port id="2"/>
				<port id="3"/>
				<port id="4"/>
			</input>
			<output>
				<port id="5">
					<dim>OUT_CHANNEL</dim>
					<dim>IN_CHANNEL</dim>
				</port>
			</output>
		</layer>
		<layer id="20" name="biasesConvolutionConst" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>OUT_CHANNEL</dim>
				</port>
			</output>
			<blobs>
				<custom offset="FULLYCONNECTED_BIASES_CONST_OFFSET" size="128"/>
			</blobs>
		</layer>
        <layer id="211" name="reshapeConst" precision="I32" type="Const">
			<output>
				<port id="1">
					<dim>2</dim>
				</port>
			</output>
            <blobs>
				<custom offset="RESHAPE_CONST_OFFSET" size="8"/>
			</blobs>
		</layer>
        <layer id="21" name="reshape" precision="FP32" type="Reshape">
			<input>
				<port id="0">
					<dim>IN_BATCH</dim>
					<dim>IN_CHANNEL</dim>
					<dim>IN_X</dim>
					<dim>IN_Y</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>IN_BATCH</dim>
					<dim>RESHAPED_CH_X_Y</dim>
				</port>
			</output>
		</layer>
		<layer id="22" name="fullyconnected" precision="FP32" type="FullyConnected">
			<data out-size="OUT_CHANNEL"/>
			<input>
				<port id="0">
					<dim>IN_BATCH</dim>
					<dim>RESHAPED_CH_X_Y</dim>
				</port>
				<port id="1">
					<dim>OUT_CHANNEL</dim>
					<dim>IN_CHANNEL</dim>
				</port>
				<port id="2">
					<dim>OUT_CHANNEL</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>OUT_BATCH</dim>
					<dim>OUT_CHANNEL</dim>
				</port>
			</output>
		</layer>
        )V0G0N";

    const std::string edgesTemplate = R"V0G0N(
        <edge from-layer="0"  from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="1"  from-port="1" to-layer="13" to-port="0"/>

        <!-- data FakeQuantize -->
        <edge from-layer="9"  from-port="1" to-layer="13" to-port="1"/>
        <edge from-layer="10"  from-port="1" to-layer="13" to-port="2"/>
        <edge from-layer="11"  from-port="1" to-layer="13" to-port="3"/>
        <edge from-layer="12"  from-port="1" to-layer="13" to-port="4"/>

        <!-- weights FakeQuantize -->
        <edge from-layer="14"  from-port="1" to-layer="19" to-port="0"/>
        <edge from-layer="15"  from-port="1" to-layer="19" to-port="1"/>
        <edge from-layer="16"  from-port="1" to-layer="19" to-port="2"/>
        <edge from-layer="17" from-port="1" to-layer="19" to-port="3"/>
        <edge from-layer="18" from-port="1" to-layer="19" to-port="4"/>

        <edge from-layer="13" from-port="5" to-layer="21" to-port="0"/>
        <edge from-layer="211" from-port="1" to-layer="21" to-port="1"/>
        <edge from-layer="21" from-port="2" to-layer="22" to-port="0"/>

        <!-- FullyConnected -->
        <edge from-layer="21" from-port="2" to-layer="22" to-port="0"/>
        <edge from-layer="19" from-port="5" to-layer="22" to-port="1"/>
        <edge from-layer="20" from-port="1" to-layer="22" to-port="2"/>
        )V0G0N";
};

class GemmAndQuantizeTestModel : public SingleLayerTestModel {
public:
    void resetTransformation(CNNNetwork& network) const override {
        fillData(getLayer(network, "dataConstInputLow"), 63.5, "custom");
        fillData(getLayer(network, "dataConstInputHigh"), 127.0, "custom");
        fillData(getLayer(network, "dataConstOutputLow"), 63.5, "custom");
        fillData(getLayer(network, "dataConstOutputHigh"), 127.0, "custom");

        //fillData(getLayer(network, "weightsConstInput"), 3.0, "custom");
        fillDataWithInitValue(getLayer(network, "weightsConstInput"), "custom", 1.234);

        fillData(getLayer(network, "weightsConstInputLow"), -1.275 / 2.0, "custom");
        fillData(getLayer(network, "weightsConstInputHigh"), 1.275, "custom");
        fillData(getLayer(network, "weightsConstOutputLow"), -1.275 / 2.0, "custom");
        fillData(getLayer(network, "weightsConstOutputHigh"), 1.275, "custom");

        fillDataMy(getLayer(network, "reshapeConst"), { 1, -1 });
    }

    std::string getName() const override {
        return "GemmAndQuantizeTestModel";
    }

    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override {
        LowPrecisionTransformer transformer(LowPrecisionTransformer::getAllTransformations(params));
        transformer.transform(network);

        const std::vector<CNNLayerPtr> layers = CNNNetSortTopologically(network);

        const CNNLayerPtr convolution = layers[layers.size() - 2];
        if ((convolution->type != "GEMM") || (convolution->name != "gemm_original")) {
            THROW_IE_EXCEPTION << "unexpected layer type '" << convolution->type << "' or name '" << convolution->name << "'";
        }

        const CNNLayerPtr dequantizationScaleShift = layers[layers.size() - 1];
        if ((dequantizationScaleShift->type != "ScaleShift") || (dequantizationScaleShift->name != "gemm")) {
            THROW_IE_EXCEPTION << "unexpected layer type '" << dequantizationScaleShift->type << "' or name '" << dequantizationScaleShift->name << "'";
        }

        return true;
    }

    std::string getModel(SingleLayerTransformationsTestParams& p) const override {
        std::string layers = layersTemplate;
        size_t totalOffset = 0;

        REPLACE_WITH_NUM(layers, "DATA_CONST_INPUT_LOW_OFFSET", totalOffset);
        totalOffset += 4;
        REPLACE_WITH_NUM(layers, "DATA_CONST_INPUT_HIGH_OFFSET", totalOffset);
        totalOffset += 4;
        REPLACE_WITH_NUM(layers, "DATA_CONST_OUTPUT_LOW_OFFSET", totalOffset);
        totalOffset += 4;
        REPLACE_WITH_NUM(layers, "DATA_CONST_OUTPUT_HIGH_OFFSET", totalOffset);
        totalOffset += 4;

        REPLACE_WITH_NUM(layers, "WEIGHTS_CONST_INPUT_OFFSET", totalOffset);
        totalOffset += 32 * 32 * 4;
        REPLACE_WITH_NUM(layers, "WEIGHTS_CONST_INPUT_LOW_OFFSET", totalOffset);
        totalOffset += 4;
        REPLACE_WITH_NUM(layers, "WEIGHTS_CONST_INPUT_HIGH_OFFSET", totalOffset);
        totalOffset += 4;
        REPLACE_WITH_NUM(layers, "WEIGHTS_CONST_OUTPUT_LOW_OFFSET", totalOffset);
        totalOffset += 4;
        REPLACE_WITH_NUM(layers, "WEIGHTS_CONST_OUTPUT_HIGH_OFFSET", totalOffset);
        totalOffset += 4;
        REPLACE_WITH_NUM(layers, "RESHAPE_CONST_OFFSET", totalOffset);
        totalOffset += 8;

        const std::string model = IRTemplateGenerator::getIRTemplate(
            "TransformationsTest",
            { 1, 32, 149, 149 },
            "FP32",
            layers,
            edgesTemplate,
            6);

        return model;
    }

private:
    const std::string layersTemplate = R"V0G0N(
        <layer name="inputPower" type="Power" precision="FP32" id="1">
            <power_data power="1" scale="1" shift="0"/>
            <input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>149</dim>
					<dim>149</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
					<dim>149</dim>
					<dim>149</dim>
				</port>
			</output>
        </layer>


        <layer id="9" name="dataConstInputLow" precision="FP32" type="Const">
			<output>
				<port id="1"/>
			</output>
			<blobs>
				<custom offset="DATA_CONST_INPUT_LOW_OFFSET" size="4"/>
			</blobs>
		</layer>
		<layer id="10" name="dataConstInputHigh" precision="FP32" type="Const">
			<output>
				<port id="1"/>
			</output>
			<blobs>
				<custom offset="DATA_CONST_INPUT_HIGH_OFFSET" size="4"/>
			</blobs>
		</layer>
		<layer id="11" name="dataConstOutputLow" precision="FP32" type="Const">
			<output>
				<port id="1"/>
			</output>
			<blobs>
				<custom offset="DATA_CONST_OUTPUT_LOW_OFFSET" size="4"/>
			</blobs>
		</layer>
		<layer id="12" name="dataConstOutputHigh" precision="FP32" type="Const">
			<output>
				<port id="1"/>
			</output>
			<blobs>
				<custom offset="DATA_CONST_OUTPUT_HIGH_OFFSET" size="4"/>
			</blobs>
		</layer>
		<layer id="13" name="dataFakeQuantize" precision="FP32" type="FakeQuantize">
			<data levels="256"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>149</dim>
					<dim>149</dim>
				</port>
				<port id="1"/>
				<port id="2"/>
				<port id="3"/>
				<port id="4"/>
			</input>
			<output>
				<port id="5">
					<dim>1</dim>
					<dim>32</dim>
					<dim>149</dim>
					<dim>149</dim>
				</port>
			</output>
		</layer>
		<layer id="14" name="weightsConstInput" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<custom offset="WEIGHTS_CONST_INPUT_OFFSET" size="4096"/>
			</blobs>
		</layer>
		<layer id="15" name="weightsConstInputLow" precision="FP32" type="Const">
			<output>
				<port id="1"/>
			</output>
			<blobs>
				<custom offset="WEIGHTS_CONST_INPUT_LOW_OFFSET" size="4"/>
			</blobs>
		</layer>
		<layer id="16" name="weightsConstInputHigh" precision="FP32" type="Const">
			<output>
				<port id="1"/>
			</output>
			<blobs>
				<custom offset="WEIGHTS_CONST_INPUT_HIGH_OFFSET" size="4"/>
			</blobs>
		</layer>
		<layer id="17" name="weightsConstOutputLow" precision="FP32" type="Const">
			<output>
				<port id="1"/>
			</output>
			<blobs>
				<custom offset="WEIGHTS_CONST_OUTPUT_LOW_OFFSET" size="4"/>
			</blobs>
		</layer>
		<layer id="18" name="weightsConstOutputHigh" precision="FP32" type="Const">
			<output>
				<port id="1"/>
			</output>
			<blobs>
				<custom offset="WEIGHTS_CONST_OUTPUT_HIGH_OFFSET" size="4"/>
			</blobs>
		</layer>
		<layer id="19" name="weightsFakeQuantize" precision="FP32" type="FakeQuantize">
			<data levels="256"/>
			<input>
				<port id="0">
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1"/>
				<port id="2"/>
				<port id="3"/>
				<port id="4"/>
			</input>
			<output>
				<port id="5">
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="211" name="reshapeConst" precision="I32" type="Const">
			<output>
				<port id="1">
					<dim>2</dim>
				</port>
			</output>
            <blobs>
				<custom offset="RESHAPE_CONST_OFFSET" size="8"/>
			</blobs>
		</layer>
        <layer id="21" name="reshape" precision="FP32" type="Reshape">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>149</dim>
					<dim>149</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="22" name="gemm" precision="FP32" type="GEMM">
			<data transpose_a="0" transpose_b="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
        )V0G0N";

    const std::string edgesTemplate = R"V0G0N(
        <edge from-layer="0"  from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="1"  from-port="1" to-layer="13" to-port="0"/>

        <!-- data FakeQuantize -->
        <edge from-layer="9"  from-port="1" to-layer="13" to-port="1"/>
        <edge from-layer="10"  from-port="1" to-layer="13" to-port="2"/>
        <edge from-layer="11"  from-port="1" to-layer="13" to-port="3"/>
        <edge from-layer="12"  from-port="1" to-layer="13" to-port="4"/>

        <!-- weights FakeQuantize -->
        <edge from-layer="14"  from-port="1" to-layer="19" to-port="0"/>
        <edge from-layer="15"  from-port="1" to-layer="19" to-port="1"/>
        <edge from-layer="16"  from-port="1" to-layer="19" to-port="2"/>
        <edge from-layer="17" from-port="1" to-layer="19" to-port="3"/>
        <edge from-layer="18" from-port="1" to-layer="19" to-port="4"/>

        <edge from-layer="13" from-port="5" to-layer="21" to-port="0"/>
        <edge from-layer="211" from-port="1" to-layer="21" to-port="1"/>
        <edge from-layer="21" from-port="2" to-layer="22" to-port="0"/>

        <!-- FullyConnected -->
        <edge from-layer="21" from-port="2" to-layer="22" to-port="0"/>
        <edge from-layer="19" from-port="5" to-layer="22" to-port="1"/>
        )V0G0N";
};

class PoolingTestModel : public SingleLayerTestModel {
public:
    void resetTransformation(CNNNetwork& network) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
};

class PowerTestModel : public SingleLayerTestModel {
public:
    PowerTestModel(const float& power, const float& scale, const float& shift) : power(power), scale(scale), shift(shift) {}
    void resetTransformation(CNNNetwork& network) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;

private:
    const float power;
    const float scale;
    const float shift;
};

class ConvolutionAndQuantizeOnWeightsWithMultiOutputIntervalsTestModel : public SingleLayerTestModel {
public:
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    void resetTransformation(CNNNetwork& network) const override;
};

class ConvolutionAndQuantizeOnWeightsWithoutConstTransformationTestModel : public SingleLayerTestModel {
public:
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    void resetTransformation(CNNNetwork& network) const override;
};

// Base test class to manually quantize weights and biases
class QuantizationOnWeightsTestModel : public SingleLayerTestModel {
public:
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    virtual std::unordered_set<std::string> getNotTransformedLayers() const override;
    void resetTransformation(CNNNetwork& network) const override;
};

class QuantizationOnInvertedWeightsTestModel : public SingleLayerTestModel {
public:
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    virtual std::unordered_set<std::string> getNotTransformedLayers() const override;
    void resetTransformation(CNNNetwork& network) const override;
};

class FakeQuantizeAsOutputTest : public QuantizationOnWeightsTestModel {
public:
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    virtual std::unordered_set<std::string> getNotTransformedLayers() const override;
};

class FakeQuantizeWithMultiOutputsTest : public SingleLayerTestModel {
public:
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    virtual std::unordered_set<std::string> getNotTransformedLayers() const override;
    void resetTransformation(CNNNetwork& network) const override;
};

class FakeQuantizeWithTwoScaleShiftsAsOutput : public SingleLayerTestModel {
public:
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    void resetTransformation(CNNNetwork& network) const override;
};

class ConvolutionAndPoolingAndQuantizeOnActivationsTestModel : public SingleLayerTestModel {
public:
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    void resetTransformation(CNNNetwork& network) const override;
};

class ConvolutionAndQuantizeOnActivationsTestModel : public SingleLayerTestModel {
public:
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    void resetTransformation(CNNNetwork& network) const override;
};

class ConvolutionAndDequantizationScaleShiftsOnActivationsTestModel : public SingleLayerTestModel {
public:
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    void resetTransformation(CNNNetwork& network) const override;
};

// base test type for FullyConnected test
class FullyConnectedBaseTestModel : public SingleLayerTestModel {
public:
    FullyConnectedBaseTestModel(const bool addBiasesLayer = true);
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    void resetTransformation(CNNNetwork& network) const override;
protected:
    virtual bool areScalesOnActivationsDifferent() const;
    const bool addBiasesLayer;
};

// base test type for convolution test
class ConvolutionBaseTestModel : public SingleLayerTestModel {
public:
    ConvolutionBaseTestModel(const bool addBiasesLayer = true);
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    void resetTransformation(CNNNetwork& network) const override;
protected:
    virtual size_t getGroupsCount(SingleLayerTransformationsTestParams& p) const;
    virtual bool areScalesOnActivationsDifferent() const;
    const bool addBiasesLayer;
};

class ConvolutionDepthwiseTestModel : public ConvolutionBaseTestModel {
public:
    std::string getName() const override;
protected:
    size_t getGroupsCount(SingleLayerTransformationsTestParams& p) const override;
    bool areScalesOnActivationsDifferent() const override;
};

class ConvolutionGroupedTestModel : public ConvolutionBaseTestModel {
public:
    std::string getName() const override;
    void initInput(Blob::Ptr input) const override;
protected:
    size_t getGroupsCount(SingleLayerTransformationsTestParams& p) const override;
    bool areScalesOnActivationsDifferent() const override;
};

class UpdateBiasesConvolutionTestModel : public ConvolutionBaseTestModel {
public:
    UpdateBiasesConvolutionTestModel(const bool addBiasesLayer = false);
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    void initInput(Blob::Ptr input) const override;
};

class UpdateBiasesFullyConnectedTestModel : public FullyConnectedBaseTestModel {
public:
    UpdateBiasesFullyConnectedTestModel(const bool addBiasesLayer = false);
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    void initInput(Blob::Ptr input) const override;
};

class FullyConnectedTestModel : public SingleLayerTestModel {
public:
    FullyConnectedTestModel(const std::vector<size_t>& inputDimentions, const std::vector<size_t>& outputDimentions);
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    void initInput(Blob::Ptr input) const override;
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
    void resetTransformation(CNNNetwork& network) const override;
protected:
    virtual bool areScalesOnActivationsDifferent() const;
    const bool addBiasesLayer;

private:
    const std::vector<size_t> inputDimentions;
    const std::vector<size_t> outputDimentions;
};

class EltwiseTestModel : public SingleLayerTestModel {
public:
    EltwiseTestModel(
        const bool cpuSpecific,
        const std::string& operation,
        const bool signedIntervals,
        const size_t minLevels = 2ul,
        const bool addPooling = true) :
        SingleLayerTestModel(),
        cpuSpecific(cpuSpecific),
        operation(operation),
        signedIntervals(signedIntervals),
        minLevels(minLevels),
        addPooling(addPooling) {}

    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    void resetTransformation(CNNNetwork& network) const override;

private:
    const bool cpuSpecific;
    const std::string operation;
    const bool signedIntervals;
    const size_t minLevels;
    const bool addPooling;
};

class EltwiseFqWithChildrenTestModel : public SingleLayerTestModel {
public:
    EltwiseFqWithChildrenTestModel(
        const bool cpuSpecific,
        const std::string& operation,
        const bool signedIntervals,
        const size_t minLevels = 2ul,
        const bool addPooling = true) :
        SingleLayerTestModel(),
        cpuSpecific(cpuSpecific),
        operation(operation),
        signedIntervals(signedIntervals),
        minLevels(minLevels),
        addPooling(addPooling) {}

    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    void resetTransformation(CNNNetwork& network) const override;

private:
    const bool cpuSpecific;
    const std::string operation;
    const bool signedIntervals;
    const size_t minLevels;
    const bool addPooling;
};


class EltwiseWithPoolingTestModel : public SingleLayerTestModel {
public:
    EltwiseWithPoolingTestModel(
        const bool cpuSpecific,
        const std::string& operation,
        const bool signedIntervals,
        const size_t minLevels = 2ul) :
        SingleLayerTestModel(),
        cpuSpecific(cpuSpecific),
        operation(operation),
        signedIntervals(signedIntervals),
        minLevels(minLevels) {}

    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    void resetTransformation(CNNNetwork& network) const override;

private:
    const bool cpuSpecific;
    const std::string operation;
    const bool signedIntervals;
    const size_t minLevels;
};

class EltwiseBroadcastTestModel : public SingleLayerTestModel {
public:
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    void resetTransformation(CNNNetwork& network) const override;
};

class EltwiseCpuTestModel : public SingleLayerTestModel {
public:
    std::string getModel(SingleLayerTransformationsTestParams& p) const override {

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

        REPLACE_WITH_NUM(layers, "DATA_CONST_INPUT_LOW_OFFSET_3", totalOffset);
        totalOffset += 4;
        REPLACE_WITH_NUM(layers, "DATA_CONST_INPUT_HIGHT_OFFSET_3", totalOffset);
        totalOffset += 4;
        REPLACE_WITH_NUM(layers, "DATA_CONST_OUTPUT_LOW_OFFSET_3", totalOffset);
        totalOffset += 4;
        REPLACE_WITH_NUM(layers, "DATA_CONST_OUTPUT_HIGH_OFFSET_3", totalOffset);
        totalOffset += 4;

        REPLACE_WITH_NUM(layers, "WEIGHTS_CONST_INPUT_OFFSET", totalOffset);
        totalOffset += 3 * 3 * 3 * 3 * 4;
        REPLACE_WITH_NUM(layers, "WEIGHTS_CONST_INPUT_SIZE", 3 * 3 * 3 * 3 * 4);

        REPLACE_WITH_NUM(layers, "WEIGHTS_CONST_INPUT_LOW_OFFSET", totalOffset);
        totalOffset += 4;
        REPLACE_WITH_NUM(layers, "WEIGHTS_CONST_INPUT_HIGHT_OFFSET", totalOffset);
        totalOffset += 4;
        REPLACE_WITH_NUM(layers, "WEIGHTS_CONST_OUTPUT_LOW_OFFSET", totalOffset);
        totalOffset += 4;
        REPLACE_WITH_NUM(layers, "WEIGHTS_CONST_OUTPUT_HIGH_OFFSET", totalOffset);
        totalOffset += 4;

        REPLACE_WITH_NUM(layers, "BIASES_CONST_OFFSET", totalOffset);
        totalOffset += 3 * 4;
        REPLACE_WITH_NUM(layers, "BIASES_CONST_SIZE", 3 * 4);

        REPLACE_WITH_NUM(layers, "DATA_CONST_INPUT_LOW_OFFSET_4", totalOffset);
        totalOffset += 4;
        REPLACE_WITH_NUM(layers, "DATA_CONST_INPUT_HIGHT_OFFSET_4", totalOffset);
        totalOffset += 4;
        REPLACE_WITH_NUM(layers, "DATA_CONST_OUTPUT_LOW_OFFSET_4", totalOffset);
        totalOffset += 4;
        REPLACE_WITH_NUM(layers, "DATA_CONST_OUTPUT_HIGH_OFFSET_4", totalOffset);
        totalOffset += 4;

        REPLACE_WITH_NUM(layers, "DEQUANTIZE_SCALESHIFT_WEIGHTS_OFFSET", totalOffset);
        totalOffset += 12;
        REPLACE_WITH_NUM(layers, "DEQUANTIZE_SCALESHIFT_BIASES_OFFSET", totalOffset);
        totalOffset += 12;

        const std::string model = IRTemplateGenerator::getIRTemplate(
            "TransformationsTest",
            { 1, 3, 299, 299 },
            "FP32",
            layers,
            edgesTemplate,
            6);

        return model;
    }

    std::string getName() const override {
        return "EltwiseCpuTestModel";
    }

    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override {
        LowPrecisionTransformer transformer = getLowPrecisionTransformer(params);
        transformer.transform(network);

        // TODO: skip interval validation - not completed
        return false;
    }

    void resetTransformation(CNNNetwork& network) const override {
        fillData(getLayer(network, "branch1/dataConstInputLow1"), 255.0 / 200.0, "custom");
        fillData(getLayer(network, "branch1/dataConstInputHigh1"), 255.0 / 100.0, "custom");
        fillData(getLayer(network, "branch1/dataConstOutputLow1"), 255.0 / 200.0, "custom");
        fillData(getLayer(network, "branch1/dataConstOutputHigh1"), 255.0 / 100.0, "custom");

        fillData(getLayer(network, "branch2/dataConstInputLow3"), 255.0 / 200.0, "custom");
        fillData(getLayer(network, "branch2/dataConstInputHigh3"), 255.0 / 100.0, "custom");
        fillData(getLayer(network, "branch2/dataConstOutputLow3"), 255.0 / 200.0, "custom");
        fillData(getLayer(network, "branch2/dataConstOutputHigh3"), 255.0 / 100.0, "custom");

        fillData(getLayer(network, "branch2/weightsConstInput"), 0.0, "custom");
        fillData(getLayer(network, "branch2/weightsConstInputLow"), 0.0, "custom");
        fillData(getLayer(network, "branch2/weightsConstInputHigh"), 255.0 / 200.0, "custom");
        fillData(getLayer(network, "branch2/weightsConstOutputLow"), 0.0, "custom");
        fillData(getLayer(network, "branch2/weightsConstOutputHigh"), 255.0 / 200.0, "custom");

        fillData(getLayer(network, "branch2/biasesConst"), { 1.0, 2.0, 3.0 });

        fillData(getLayer(network, "branch2/dataConstInputLow4"), 255.0 / 800.0, "custom");
        fillData(getLayer(network, "branch2/dataConstInputHigh4"), 255.0 / 400.0, "custom");
        fillData(getLayer(network, "branch2/dataConstOutputLow4"), 255.0 / 800.0, "custom");
        fillData(getLayer(network, "branch2/dataConstOutputHigh4"), 255.0 / 400.0, "custom");
    }

private:
    const std::string layersTemplate = R"V0G0N(
        <layer name="branch1/dataConstInputLow1" type="Const" precision="FP32" id="102">
            <output>
                <port id="0">
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="DATA_CONST_INPUT_LOW_OFFSET_1" size="4"/>
            </blobs>
        </layer>
        <layer name="branch1/dataConstInputHigh1" type="Const" precision="FP32" id="103">
            <output>
                <port id="0">
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="DATA_CONST_INPUT_HIGHT_OFFSET_1" size="4"/>
            </blobs>
        </layer>

        <layer name="branch1/dataConstOutputLow1" type="Const" precision="FP32" id="104">
            <output>
                <port id="0">
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="DATA_CONST_OUTPUT_LOW_OFFSET_1" size="4"/>
            </blobs>
        </layer>
        <layer name="branch1/dataConstOutputHigh1" type="Const" precision="FP32" id="105">
            <output>
                <port id="0">
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="DATA_CONST_OUTPUT_HIGH_OFFSET_1" size="4"/>
            </blobs>
        </layer>

        <layer name="branch1/dataFakeQuantize1" type="FakeQuantize" precision="FP32" id="106">
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

        <layer name="branch2/dataConstInputLow3" type="Const" precision="FP32" id="207">
            <output>
                <port id="0">
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="DATA_CONST_INPUT_LOW_OFFSET_3" size="4"/>
            </blobs>
        </layer>
        <layer name="branch2/dataConstInputHigh3" type="Const" precision="FP32" id="208">
            <output>
                <port id="0">
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="DATA_CONST_INPUT_HIGHT_OFFSET_3" size="4"/>
            </blobs>
        </layer>

        <layer name="branch2/dataConstOutputLow3" type="Const" precision="FP32" id="209">
            <output>
                <port id="0">
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="DATA_CONST_OUTPUT_LOW_OFFSET_3" size="4"/>
            </blobs>
        </layer>
        <layer name="branch2/dataConstOutputHigh3" type="Const" precision="FP32" id="210">
            <output>
                <port id="0">
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="DATA_CONST_OUTPUT_HIGH_OFFSET_3" size="4"/>
            </blobs>
        </layer>


        <layer name="branch2/dataFakeQuantize3" type="FakeQuantize" precision="FP32" id="211">
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


        <layer name="branch2/weightsConstInput" type="Const" precision="FP32" id="212">
            <output>
                <port id="0">
                    <dim>3</dim>
                    <dim>3</dim>
                    <dim>3</dim>
                    <dim>3</dim>
                </port>
            </output>
            <blobs>
                <custom offset="WEIGHTS_CONST_INPUT_OFFSET" size="WEIGHTS_CONST_INPUT_SIZE"/>
            </blobs>
        </layer>
        <layer name="branch2/weightsConstInputLow" type="Const" precision="FP32" id="213">
            <output>
                <port id="0">
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="WEIGHTS_CONST_INPUT_LOW_OFFSET" size="4"/>
            </blobs>
        </layer>
        <layer name="branch2/weightsConstInputHigh" type="Const" precision="FP32" id="214">
            <output>
                <port id="0">
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="WEIGHTS_CONST_INPUT_HIGHT_OFFSET" size="4"/>
            </blobs>
        </layer>

        <layer name="branch2/weightsConstOutputLow" type="Const" precision="FP32" id="215">
            <output>
            <port id="0">
                <dim>1</dim>
            </port>
            </output>
            <blobs>
                <custom offset="WEIGHTS_CONST_OUTPUT_LOW_OFFSET" size="4"/>
            </blobs>
        </layer>
        <layer name="branch2/weightsConstOutputHigh" type="Const" precision="FP32" id="216">
            <output>
                <port id="0">
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="WEIGHTS_CONST_OUTPUT_HIGH_OFFSET" size="4"/>
            </blobs>
        </layer>


        <layer name="branch2/weightsFakeQuantize" type="FakeQuantize" precision="FP32" id="218">
            <data levels="256" />
            <input>
                <port id="0">
                    <dim>3</dim>
                    <dim>3</dim>
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
                    <dim>3</dim>
                    <dim>3</dim>
                    <dim>3</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>

        <layer name="branch2/biasesConst" type="Const" precision="FP32" id="219">
            <output>
                <port id="0">
                    <dim>3</dim>
                </port>
            </output>
            <blobs>
                <custom offset="BIASES_CONST_OFFSET" size="BIASES_CONST_SIZE"/>
            </blobs>
        </layer>


        <layer name="branch2/convolution" precision="FP32" type="Convolution" id="220">
			<data dilations="1,1" group="1" kernel="3,3" output="3" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>299</dim>
					<dim>299</dim>
				</port>
				<port id="1">
					<dim>3</dim>
					<dim>3</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
				<port id="2">
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>3</dim>
					<dim>299</dim>
					<dim>299</dim>
				</port>
			</output>
		</layer>

        <layer name="branch2/dataConstInputLow4" type="Const" precision="FP32" id="222">
            <output>
                <port id="0">
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="DATA_CONST_INPUT_LOW_OFFSET_4" size="4"/>
            </blobs>
        </layer>
        <layer name="branch2/dataConstInputHigh4" type="Const" precision="FP32" id="223">
            <output>
                <port id="0">
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="DATA_CONST_INPUT_HIGHT_OFFSET_4" size="4"/>
            </blobs>
        </layer>

        <layer name="branch2/dataConstOutputLow4" type="Const" precision="FP32" id="224">
            <output>
                <port id="0">
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="DATA_CONST_OUTPUT_LOW_OFFSET_4" size="4"/>
            </blobs>
        </layer>
        <layer name="branch2/dataConstOutputHigh4" type="Const" precision="FP32" id="225">
            <output>
                <port id="0">
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="DATA_CONST_OUTPUT_HIGH_OFFSET_4" size="4"/>
            </blobs>
        </layer>

        <layer name="branch2/dataFakeQuantize4" type="FakeQuantize" precision="FP32" id="226">
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

        <layer name="branch2/eltwise" type="Eltwise" precision="FP32" id="227">
            <data operation="sum"/>
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
                    <dim>3</dim>
                    <dim>299</dim>
                    <dim>299</dim>
                </port>
            </output>
        </layer>


        <layer name="outputPower" type="Power" precision="FP32" id="300">
            <power_data power="1" scale="1" shift="0"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>299</dim>
                    <dim>299</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>299</dim>
                    <dim>299</dim>
                </port>
            </output>
        </layer>

        )V0G0N";

    const std::string edgesTemplate = R"V0G0N(
        <!-- branch 1 -->

        <edge from-layer="0" from-port="0" to-layer="106" to-port="0"/>
        <edge from-layer="102" from-port="0" to-layer="106" to-port="1"/>
        <edge from-layer="103" from-port="0" to-layer="106" to-port="2"/>
        <edge from-layer="104" from-port="0" to-layer="106" to-port="3"/>
        <edge from-layer="105" from-port="0" to-layer="106" to-port="4"/>
        <edge from-layer="106" from-port="5" to-layer="211" to-port="0"/>
        <edge from-layer="106" from-port="5" to-layer="227" to-port="0"/>

        <!-- branch 2 -->

        <!-- FakeQuantize on activations -->
        <edge from-layer="207" from-port="0" to-layer="211" to-port="1"/>
        <edge from-layer="208" from-port="0" to-layer="211" to-port="2"/>
        <edge from-layer="209" from-port="0" to-layer="211" to-port="3"/>
        <edge from-layer="210" from-port="0" to-layer="211" to-port="4"/>
        <edge from-layer="211" from-port="5" to-layer="220" to-port="0"/>

        <!-- FakeQuantize on weights -->
        <edge from-layer="212" from-port="0" to-layer="218" to-port="0"/>
        <edge from-layer="213" from-port="0" to-layer="218" to-port="1"/>
        <edge from-layer="214" from-port="0" to-layer="218" to-port="2"/>
        <edge from-layer="215" from-port="0" to-layer="218" to-port="3"/>
        <edge from-layer="216" from-port="0" to-layer="218" to-port="4"/>
        <edge from-layer="218" from-port="5" to-layer="220" to-port="1"/>

        <!-- Const on biases -->
        <edge from-layer="219" from-port="0" to-layer="220" to-port="2"/>

        <!-- Convolution -->
        <edge from-layer="220" from-port="3" to-layer="226" to-port="0"/>

        <!-- FakeQuantize on activations -->
        <edge from-layer="222" from-port="0" to-layer="226" to-port="1"/>
        <edge from-layer="223" from-port="0" to-layer="226" to-port="2"/>
        <edge from-layer="224" from-port="0" to-layer="226" to-port="3"/>
        <edge from-layer="225" from-port="0" to-layer="226" to-port="4"/>
        <edge from-layer="226" from-port="5" to-layer="227" to-port="1"/>

        <!-- Eltwise -->
        <edge from-layer="227" from-port="2" to-layer="300" to-port="0"/>
        )V0G0N";

    const std::map<std::string, std::vector<size_t>> dimensions = {
        {{ "in1", { 299, 299, 3, 1 } },
        { "in2", { 299, 299, 3, 1 } } }
    };
};

class ConcatTestModel : public SingleLayerTestModel {
public:
    ConcatTestModel(
        const bool signedIntervals,
        const bool symmetricInterval = true,
        const bool multiChannel = true,
        const std::vector<size_t>& constInputDimentions = { 1 });

    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    void resetTransformation(CNNNetwork& network) const override;
    float getThreshold(const std::string& device_name, const Precision precision, LayerTransformation::Params& params) const override;
private:
    const bool signedIntervals;
    const bool symmetricInterval;
    const bool multiChannel;
    const std::vector<size_t> constInputDimentions;
};

class ConcatWithPoolingTestModel : public SingleLayerTestModel {
public:
    ConcatWithPoolingTestModel(
        const bool multiChannel,
        const bool signedIntervals,
        const bool shift,
        const float dequantizationIntervalsDifference) :
        SingleLayerTestModel(),
        multiChannel(multiChannel),
        signedIntervals(signedIntervals),
        shift(shift),
        dequantizationIntervalsDifference(dequantizationIntervalsDifference) {}

    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    void resetTransformation(CNNNetwork& network) const override;
    float getThreshold(const std::string& pluginName, const Precision precision, LayerTransformation::Params& params) const override;

private:
    const bool multiChannel;
    const bool signedIntervals;
    const bool shift;
    const float dequantizationIntervalsDifference;
};

class ConcatMultiChannelTestModel : public SingleLayerTestModel {
public:
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    void resetTransformation(CNNNetwork& network) const override;
};

// TODO: remove, not used
class ConcatMultiBranchTestModel : public SingleLayerTestModel {
public:
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    void resetTransformation(CNNNetwork& network) const override;

    const static std::string layersTemplate;
private:

    const std::string edgesTemplate = R"V0G0N(
        <!-- branch 1 -->

        <edge from-layer="0" from-port="0" to-layer="106" to-port="0"/>
        <edge from-layer="102" from-port="0" to-layer="106" to-port="1"/>
        <edge from-layer="103" from-port="0" to-layer="106" to-port="2"/>
        <edge from-layer="104" from-port="0" to-layer="106" to-port="3"/>
        <edge from-layer="105" from-port="0" to-layer="106" to-port="4"/>
        <edge from-layer="106" from-port="5" to-layer="113" to-port="0"/>

        <edge from-layer="1" from-port="0" to-layer="111" to-port="0"/>
        <edge from-layer="107" from-port="0" to-layer="111" to-port="1"/>
        <edge from-layer="108" from-port="0" to-layer="111" to-port="2"/>
        <edge from-layer="109" from-port="0" to-layer="111" to-port="3"/>
        <edge from-layer="110" from-port="0" to-layer="111" to-port="4"/>
        <edge from-layer="111" from-port="5" to-layer="113" to-port="1"/>

        <edge from-layer="113" from-port="2" to-layer="227" to-port="0"/>

        <!-- branch 2 -->

        <!-- FakeQuantize on activations -->
        <edge from-layer="113" from-port="2" to-layer="211" to-port="0"/>
        <edge from-layer="207" from-port="0" to-layer="211" to-port="1"/>
        <edge from-layer="208" from-port="0" to-layer="211" to-port="2"/>
        <edge from-layer="209" from-port="0" to-layer="211" to-port="3"/>
        <edge from-layer="210" from-port="0" to-layer="211" to-port="4"/>
        <edge from-layer="211" from-port="5" to-layer="220" to-port="0"/>

        <!-- FakeQuantize on weights -->
        <edge from-layer="212" from-port="0" to-layer="218" to-port="0"/>
        <edge from-layer="213" from-port="0" to-layer="218" to-port="1"/>
        <edge from-layer="214" from-port="0" to-layer="218" to-port="2"/>
        <edge from-layer="215" from-port="0" to-layer="218" to-port="3"/>
        <edge from-layer="216" from-port="0" to-layer="218" to-port="4"/>
        <edge from-layer="218" from-port="5" to-layer="220" to-port="1"/>

        <!-- Const on biases -->
        <edge from-layer="219" from-port="0" to-layer="220" to-port="2"/>

        <!-- Convolution -->
        <edge from-layer="220" from-port="3" to-layer="226" to-port="0"/>

        <!-- FakeQuantize on activations -->
        <edge from-layer="222" from-port="0" to-layer="226" to-port="1"/>
        <edge from-layer="223" from-port="0" to-layer="226" to-port="2"/>
        <edge from-layer="224" from-port="0" to-layer="226" to-port="3"/>
        <edge from-layer="225" from-port="0" to-layer="226" to-port="4"/>
        <edge from-layer="226" from-port="5" to-layer="227" to-port="1"/>

        <!-- Concat -->
        <edge from-layer="227" from-port="2" to-layer="300" to-port="0"/>
        )V0G0N";

    const std::map<std::string, std::vector<size_t>> dimensions = {
        {{ "in1", { 299, 299, 3, 1 } },
        { "in2", { 299, 299, 3, 1 } } }
    };
};

class FakeQuantizeAndScaleShiftTestModel : public SingleLayerTestModel {
public:
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    void resetTransformation(CNNNetwork& network) const override;
};

class FakeQuantizeAndActivationTestModel : public SingleLayerTestModel {
public:
    FakeQuantizeAndActivationTestModel(const std::vector<std::pair<float, float>>& intervals);
    void initInput(Blob::Ptr input) const override;
    float getZeroThreshold() const override;
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    void resetTransformation(CNNNetwork& network) const override;

private:
    const std::vector<std::pair<float, float>> intervals;
};

class ScaleShiftAndFakeQuantizeTestModel : public SingleLayerTestModel {
public:
    void initInput(Blob::Ptr input) const override;
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    void resetTransformation(CNNNetwork& network) const override;
};

class FakeQuantizeAndActivationWithNegativeScalesTestModel : public SingleLayerTestModel {
public:
    void initInput(Blob::Ptr input) const override;
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    void resetTransformation(CNNNetwork& network) const override;
};

class FakeQuantizeAndActivationWithNegativeSlopeTestModel : public SingleLayerTestModel {
public:
    void initInput(Blob::Ptr input) const override;
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    void resetTransformation(CNNNetwork& network) const override;
};

class ConvolutionAndDequantizationScaleShiftAndQuantizeOnActivationsTestModel : public SingleLayerTestModel {
public:
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    void resetTransformation(CNNNetwork& network) const override;
};

class MvnTestModel : public SingleLayerTestModel {
public:
    MvnTestModel(const size_t acrossChannels, const size_t normalizeVariance);
    void initInput(Blob::Ptr input) const override;
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    void resetTransformation(CNNNetwork& network) const override;

private:
    const size_t acrossChannels;
    const size_t normalizeVariance;
};

class PrecisionSelectionMultibranchPreservedTestModel : public SingleLayerTestModel {
public:
    PrecisionSelectionMultibranchPreservedTestModel(const bool signedIntervalOnActivation);
    void initInput(Blob::Ptr input) const override;
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    void resetTransformation(CNNNetwork& network) const override;

private:
    const size_t acrossChannels;
    const size_t normalizeVariance;
    const bool signedIntervalOnActivation;
};

class PrecisionSelectionMultibranchNotPreservedTestModel : public SingleLayerTestModel {
public:
    PrecisionSelectionMultibranchNotPreservedTestModel(const bool signedIntervalOnActivation);
    void initInput(Blob::Ptr input) const override;
    std::string getModel(SingleLayerTransformationsTestParams& p) const override;
    std::string getName() const override;
    bool transform(CNNNetwork& network, LayerTransformation::Params& params) const override;
    void resetTransformation(CNNNetwork& network) const override;

private:
    const size_t acrossChannels;
    const size_t normalizeVariance;
    const bool signedIntervalOnActivation;
};

class SingleLayerTransformationsTest : public TestsCommon, public WithParamInterface<SingleLayerTransformationsTestParams> {
    TBlob<uint8_t>::Ptr generateWeights(const CNNNetwork& network);
    void checkNetworkWithFakeQuantize(const CNNNetwork& network);
    void checkNetworkWithQuantize(const CNNNetwork& network);
    //void sortBlobs(CNNLayer& layer);
    CNNNetwork createNetwork();
    std::unordered_map<std::string, InferenceEngine::Blob::Ptr> infer(
            CNNNetwork& network,
            std::unordered_map<std::string, Blob::Ptr>& inputBlobs,
            Core & plugin, const std::string & device_name,
            ExecutableNetwork & executableNetwork,
            InferRequest & inferRequest);

protected:
    static void compareInDetails(
        InferenceEngine::Blob &res,
        InferenceEngine::Blob &ref,
        const size_t maxDifferenceCounts,
        float max_diff = 0.01f);
    virtual void SetUp();
};
