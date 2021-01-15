// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_layer_validators.hpp"

#include <ie_iextension.h>

#include <cmath>
#include <details/ie_exception.hpp>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "debug.h"
#include <legacy/ie_layers.h>
#include "xml_parse_utils.h"

#ifdef __clang__
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

namespace InferenceEngine {

using namespace details;
using std::map;
using std::string;
using std::vector;

template <typename T, typename P>
inline bool one_of(T val, P item) {
    return val == item;
}
template <typename T, typename P, typename... Args>
inline bool one_of(T val, P item, Args... item_others) {
    return val == item || one_of(val, item_others...);
}

void details::validateLayer(const CNNLayer * layer) {
    try {
        LayerValidator::Ptr validator = LayerValidators::getInstance()->getValidator(layer->type);
        validator->checkParams(layer);
        InOutDims shapes;
        getInOutShapes(layer, shapes);
        validator->checkShapes(layer, shapes.inDims);
    } catch (const InferenceEngineException& ie_e) {
        THROW_IE_EXCEPTION << "Error of validate layer: " << layer->name << " with type: " << layer->type << ". "
                           << ie_e.what();
    }
}

struct WeightableParams {
    std::vector<size_t> _kernel;
    size_t _outputs = 0lu;
    size_t _groups = 1lu;
    bool _isKernelFromInput = false;

    WeightableParams(size_t outputs, bool isKernelFromInput, size_t groups = 0, const std::vector<size_t>& kernel = {})
        : _kernel(kernel), _outputs(outputs), _groups(groups), _isKernelFromInput(isKernelFromInput) {}
};

void checkWeightable(const std::map<std::string, Blob::Ptr>& blobs, const vector<SizeVector>& inShapes,
                     WeightableParams params, const SizeVector& numDims) {
    std::vector<size_t> expected_num_of_shapes = {1, 2, 3};
    bool shape_was_found = false;
    for (const auto& i : expected_num_of_shapes) {
        if (inShapes.size() == i) {
            shape_was_found = true;
            break;
        }
    }
    if (!shape_was_found)
        THROW_IE_EXCEPTION << "Number of inputs (" << inShapes.size() << ") is not equal to expected ones (1)";
    SizeVector firstInputShape = inShapes[0];
    size_t inputSize = firstInputShape.size();

    bool isOK = false;
    for (auto dim : numDims) {
        if (inputSize == dim) {
            isOK = true;
            break;
        }
    }
    if (!isOK) {
        THROW_IE_EXCEPTION << "Input shape " << details::dumpVec(firstInputShape)
                           << " has unexpected size, supported sizes: " << details::dumpVec(numDims);
    }

    if (firstInputShape.empty()) THROW_IE_EXCEPTION << "Input shape can't be empty";

    size_t IC, OC;
    std::vector<size_t> kernel;
    IC = firstInputShape[1];
    if (params._isKernelFromInput) {
        for (size_t i = 1; i <= inputSize - 2; i++)
            kernel.push_back(firstInputShape[inputSize - i]);
    } else {
        for (auto k : params._kernel) {
            kernel.push_back(k);
        }
    }
    OC = params._outputs;

    auto it = blobs.find("weights");
    if (it !=
        blobs
            .end()) {  // TODO: return with fixing shape infer tests: THROW_IE_EXCEPTION << "Invalid blobs: no weights";
        auto weights = it->second;
        if (weights == nullptr || weights->getTensorDesc().getDims().empty())
            THROW_IE_EXCEPTION << "Weights can't be empty";

        auto weightsSize = details::product(weights->getTensorDesc().getDims());
        size_t expectedWeightsSize = OC * IC;
        for (auto k : kernel) {
            expectedWeightsSize *= k;
        }
        if (params._groups) expectedWeightsSize /= params._groups;
        if (expectedWeightsSize != weightsSize) {
            std::string ker_str;
            for (size_t i = 0; i < params._kernel.size(); i++) {
                if (!ker_str.empty()) ker_str += "x";
                ker_str += std::to_string(kernel[i]);
            }
            THROW_IE_EXCEPTION << "New shapes " << details::dumpVec(firstInputShape) << " make Kernels(" << ker_str
                               << "), Channels(" << IC << "), Output depth(" << OC << "), Groups(" << params._groups
                               << ") not matching weights size: " << expectedWeightsSize << " vs " << weightsSize;
        }
    }

    it = blobs.find("biases");
    if (it != blobs.end()) {
        auto biases = it->second;
        if (biases == nullptr || biases->getTensorDesc().getDims().empty())
            THROW_IE_EXCEPTION << "Biases can't be empty";
        auto biasesSize = details::product(biases->getTensorDesc().getDims());
        if (OC != biasesSize) {
            THROW_IE_EXCEPTION << "Number of outputs (" << OC << ") don't match biases size: " << biasesSize;
        }
    }
}

void checkDeformableConv(const DeformableConvolutionLayer* deformableConvLayer,
                         const std::map<std::string, Blob::Ptr>& blobs, const vector<SizeVector>& inShapes) {
    std::vector<size_t> krn;
    for (size_t i = 0; i < deformableConvLayer->_kernel.size(); i++)
        krn.push_back(deformableConvLayer->_kernel[i]);
    checkWeightable(blobs, {inShapes[0]}, {deformableConvLayer->_out_depth, false, deformableConvLayer->_group, krn},
                    {4});

    SizeVector transInputShape = inShapes[1];

    if (transInputShape.empty()) THROW_IE_EXCEPTION << "Trans input shape can't be empty";
    size_t TC = transInputShape[1];

    if (TC != deformableConvLayer->_deformable_group * 2 * krn[0] * krn[1])
        THROW_IE_EXCEPTION << "Failed with invalid shapes: trans input channel dimension is invalid. Actual value is: "
                           << TC
                           << ". Expected value: " << deformableConvLayer->_deformable_group * 2 * krn[0] * krn[1];
}

void checkDims(const std::vector<SizeVector>& shapes, const vector<int>& expected_shape_size) {
    for (auto i : shapes) {
        if (i.empty()) {
            THROW_IE_EXCEPTION << " Failed with invalid shapes: dimension is empty";
        }
        auto iter = std::find(expected_shape_size.begin(), expected_shape_size.end(), i.size());
        if (iter == expected_shape_size.end()) {
            THROW_IE_EXCEPTION << " Failed with invalid shapes: dimension is invalid";
        }
    }
}

void checkNumOfInput(const std::vector<SizeVector>& inShapes, const vector<size_t>& expected_num_of_shapes) {
    bool shape_was_found = false;
    for (const auto& i : expected_num_of_shapes) {
        if (inShapes.size() == i) {
            shape_was_found = true;
            break;
        }
    }
    if (!shape_was_found) {
        THROW_IE_EXCEPTION << "Number of inputs (" << inShapes.size()
                           << ") is not equal to expected ones: " << expected_num_of_shapes.size();
    }
}

LayerValidators* LayerValidators::getInstance() {
    static LayerValidators instance;
    return &instance;
}

LayerValidator::Ptr LayerValidators::getValidator(const std::string& type) {
    if (_validators.find(type) == _validators.end()) {
        return std::make_shared<GeneralValidator>(type);
    }
    return _validators[type];
}

GeneralValidator::GeneralValidator(const std::string& _type): LayerValidator(_type) {}

void FullyConnectedValidator::checkParams(const CNNLayer* layer) {
    auto casted = dynamic_cast<const FullyConnectedLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of FullyConnectedLayer class";
    }
    unsigned int _out_num = casted->GetParamAsUInt("out-size");
    (void)_out_num;
}

void FullyConnectedValidator::checkCorrespondence(const CNNLayer* layer, const std::map<std::string, Blob::Ptr>& blobs,
                                                  const vector<SizeVector>& inShapes) const {
    const auto casted = dynamic_cast<const FullyConnectedLayer*>(layer);
    if (!casted) THROW_IE_EXCEPTION << "Layer is not instance of FullyConnected layer class";
    checkWeightable(blobs, inShapes, {casted->_out_num, true, 1}, {2, 4, 5});
}

FullyConnectedValidator::FullyConnectedValidator(const std::string& _type): LayerValidator(_type) {}

void FullyConnectedValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1, 2, 3});
}

void CropValidator::checkParams(const CNNLayer* layer) {
    auto casted = dynamic_cast<const CropLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of CropLayer class";
    }
    if (casted->axis.size() != casted->offset.size()) {
        THROW_IE_EXCEPTION << "Incorrect format of the Crop layer: number of axis doesn't match number of offset - ("
                           << casted->axis.size() << " vs. " << casted->offset.size() << ")";
    }
}

CropValidator::CropValidator(const std::string& _type): LayerValidator(_type) {}

void CropValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const CropLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of CropLayer class";
    }
    size_t numInputs = inShapes.size();
    checkNumOfInput(inShapes, {1, 2});

    auto firstShape = inShapes[0];
    size_t shapeSize = firstShape.size();
    for (size_t i = 0; i < casted->axis.size(); i++) {
        int axis = casted->axis[i];
        int offset = casted->offset[i];
        if (static_cast<int>(shapeSize) <= axis)
            THROW_IE_EXCEPTION << "Crop axis(" << casted->axis[i]
                               << ") should be less the number of dimensions of first input (" << firstShape.size()
                               << ")";
        if (numInputs == 2) {
            if (casted->params.find("crop_begin") != casted->params.end()) {
                THROW_IE_EXCEPTION << "Incorrect format of the Crop layer: `crop_begin` and `crop_end` attributes are "
                                      "valid for single input only";
            }
            auto secondShape = inShapes[1];
            if (static_cast<int>(secondShape.size()) <= axis)
                THROW_IE_EXCEPTION << "Crop axis(" << axis
                                   << ") should be less the number of dimensions of second input ("
                                   << secondShape.size() << ")";
            size_t newSize = secondShape[axis];
            if (firstShape[axis] < static_cast<size_t>(offset + newSize)) {
                THROW_IE_EXCEPTION << "Incorrect crop data! Offset(" << offset << ") + result size of output("
                                   << newSize << ") should be less then input size(" << firstShape[axis]
                                   << ") for axis(" << axis << ")";
            }
        } else if (!casted->dim.empty()) {
            int dim = casted->dim[i];
            if (firstShape[axis] < (static_cast<size_t>(offset) + dim)) {
                THROW_IE_EXCEPTION << "Incorrect crop data! Offset(" << offset << ") + result size of output(" << dim
                                   << ") should be less then input size(" << firstShape[axis] << ") for axis(" << axis
                                   << ")";
            }
        }
    }
}

ConvolutionValidator::ConvolutionValidator(const std::string& _type): LayerValidator(_type) {}

void ConvolutionValidator::checkParams(const CNNLayer* layer) {
    auto casted = dynamic_cast<const ConvolutionLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of ConvolutionLayer class";
    }
    casted->GetParamAsUInt("output");

    vector<unsigned int> kernels = casted->GetParamAsUInts("kernel", {});
    if (kernels.empty()) {
        // IR_v == 2
        casted->GetParamAsUInt("kernel-x");
        casted->GetParamAsUInt("kernel-y");
        casted->GetParamAsUInt("stride-x", 1u);
        casted->GetParamAsUInt("stride-y", 1u);
        casted->GetParamAsUInt("pad-x", 0u);
        casted->GetParamAsUInt("pad-y", 0u);
        casted->GetParamAsUInt("pad-r", casted->_padding[X_AXIS]);
        casted->GetParamAsUInt("pad-b", casted->_padding[Y_AXIS]);
        casted->GetParamAsUInt("dilation-x", 1u);
        casted->GetParamAsUInt("dilation-y", 1u);
    } else {
        // IR_v > 2
        vector<unsigned int> default_0 = vector<unsigned int>(casted->_kernel.size(), 0u);
        vector<unsigned int> default_1 = vector<unsigned int>(casted->_kernel.size(), 1u);
        casted->GetParamAsUInts("strides", default_1);
        casted->GetParamAsUInts("pads_begin", default_0);
        casted->GetParamAsUInts("pads_end", default_0);
        casted->GetParamAsUInts("dilations", default_1);
    }
    casted->GetParamAsString("auto_pad", "");
    casted->GetParamAsUInt("group", 1);
}

void ConvolutionValidator::checkCorrespondence(const CNNLayer* layer, const std::map<std::string, Blob::Ptr>& blobs,
                                               const vector<SizeVector>& inShapes) const {
    auto convLayer = dynamic_cast<const ConvolutionLayer*>(layer);
    if (!convLayer) THROW_IE_EXCEPTION << "Layer is not instance of Convolution layer class";

    std::vector<size_t> krn;
    for (size_t i = 0; i < convLayer->_kernel.size(); i++)
        krn.push_back(convLayer->_kernel[i]);
    checkWeightable(blobs, inShapes, {convLayer->_out_depth, false, convLayer->_group, krn}, {4, 5});
}

void ConvolutionValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1, 2, 3});
}

void DeconvolutionValidator::checkParams(const CNNLayer* layer) {
    auto casted = dynamic_cast<const ConvolutionLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of ConvolutionLayer class";
    }
    casted->GetParamAsUInt("output");

    vector<unsigned int> kernels = casted->GetParamAsUInts("kernel", {});
    if (kernels.empty()) {
        // IR_v == 2
        casted->GetParamAsUInt("kernel-x");
        casted->GetParamAsUInt("kernel-y");
        casted->GetParamAsUInt("stride-x", 1u);
        casted->GetParamAsUInt("stride-y", 1u);
        casted->GetParamAsUInt("pad-x", 0u);
        casted->GetParamAsUInt("pad-y", 0u);
        casted->GetParamAsUInt("pad-r", casted->_padding[X_AXIS]);
        casted->GetParamAsUInt("pad-b", casted->_padding[Y_AXIS]);
        casted->GetParamAsUInt("dilation-x", 1u);
        casted->GetParamAsUInt("dilation-y", 1u);
    } else {
        // IR_v > 2
        vector<unsigned int> default_0 = vector<unsigned int>(casted->_kernel.size(), 0u);
        vector<unsigned int> default_1 = vector<unsigned int>(casted->_kernel.size(), 1u);
        casted->GetParamAsUInts("strides", default_1);
        casted->GetParamAsUInts("pads_begin", default_0);
        casted->GetParamAsUInts("pads_end", default_0);
        casted->GetParamAsUInts("dilations", default_1);
    }
    casted->GetParamAsString("auto_pad", "");
    casted->GetParamAsUInt("group", 1);
}

DeconvolutionValidator::DeconvolutionValidator(const std::string& _type): ConvolutionValidator(_type) {}

void DeconvolutionValidator::checkCorrespondence(const CNNLayer* layer, const std::map<std::string, Blob::Ptr>& blobs,
                                                 const vector<SizeVector>& inShapes) const {
    auto deconv_layer = dynamic_cast<const DeconvolutionLayer*>(layer);
    if (!deconv_layer) THROW_IE_EXCEPTION << "Layer is not instance of Deconvolution layer class";

    std::vector<size_t> krn;
    for (size_t i = 0; i < deconv_layer->_kernel.size(); i++)
        krn.push_back(deconv_layer->_kernel[i]);
    checkWeightable(blobs, inShapes, {deconv_layer->_out_depth, false, deconv_layer->_group, krn}, {4, 5});
}

void DeconvolutionValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1, 2, 3});
}

void DeformableConvolutionValidator::checkParams(const CNNLayer* layer) {
    auto casted = dynamic_cast<const ConvolutionLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of DeformableConvolutionLayer class";
    }
    casted->GetParamAsUInt("output");

    vector<unsigned int> kernels = casted->GetParamAsUInts("kernel", {});
    // IR_v > 2
    vector<unsigned int> default_0 = vector<unsigned int>(casted->_kernel.size(), 0u);
    vector<unsigned int> default_1 = vector<unsigned int>(casted->_kernel.size(), 1u);
    casted->GetParamAsUInts("strides", default_1);
    casted->GetParamAsUInts("pads_begin", default_0);
    casted->GetParamAsUInts("pads_end", default_0);
    casted->GetParamAsUInts("dilations", default_1);
    casted->GetParamAsString("auto_pad", "");
    casted->GetParamAsUInt("group", 1);
    casted->GetParamAsUInt("deformable_group", 1);
}

DeformableConvolutionValidator::DeformableConvolutionValidator(const std::string& _type): ConvolutionValidator(_type) {}

void DeformableConvolutionValidator::checkCorrespondence(const CNNLayer* layer,
                                                         const std::map<std::string, Blob::Ptr>& blobs,
                                                         const vector<SizeVector>& inShapes) const {
    auto deformable_conv_layer = dynamic_cast<const DeformableConvolutionLayer*>(layer);
    if (!deformable_conv_layer) THROW_IE_EXCEPTION << "Layer is not instance of DeformableConvolutionLayer class";

    checkDeformableConv(deformable_conv_layer, blobs, inShapes);
}

void DeformableConvolutionValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {2, 3, 4});
}

PoolingValidator::PoolingValidator(const std::string& _type): LayerValidator(_type) {}

void PoolingValidator::checkParams(const CNNLayer* layer) {
    // TODO: check that values belong to the scope of the definition according to spec
}

void PoolingValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1, 2});
}

void BatchNormalizationValidator::checkParams(const CNNLayer* layer) {
    auto casted = dynamic_cast<const BatchNormalizationLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of BatchNormalizationLayer class";
    }
    float epsilon = casted->GetParamAsFloat("epsilon");
    if (epsilon < 0) {
        THROW_IE_EXCEPTION << "The value of BatchNormalization layer epsilon parameter is invalid";
    }
}

BatchNormalizationValidator::BatchNormalizationValidator(const std::string& _type): LayerValidator(_type) {}

void BatchNormalizationValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1});
}

void PowerValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

PowerValidator::PowerValidator(const std::string& _type): LayerValidator(_type) {}

void PowerValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1});
}

void PReLUValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

PReLUValidator::PReLUValidator(const std::string& _type): LayerValidator(_type) {}

void PReLUValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1});
}

void ScaleShiftValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

ScaleShiftValidator::ScaleShiftValidator(const std::string& _type): LayerValidator(_type) {}

void ScaleShiftValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1});
}

void TileValidator::checkParams(const CNNLayer* layer) {
    auto casted = dynamic_cast<const TileLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of TileLayer class";
    }
    int axis = casted->GetParamAsInt("axis", -1);
    int tiles = casted->GetParamAsInt("tiles", -1);
    if (axis < 0 && tiles < 0) {
        THROW_IE_EXCEPTION << "The value of Tile layer parameters is invalid";
    }
}

TileValidator::TileValidator(const std::string& _type): LayerValidator(_type) {}

void TileValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1});
}

ReshapeValidator::ReshapeValidator(const std::string& _type): LayerValidator(_type) {}

void ReshapeValidator::checkParams(const CNNLayer* layer) {
    auto casted = dynamic_cast<const ReshapeLayer*>(layer);
    if (!casted) THROW_IE_EXCEPTION << "Layer is not instance of ReshapeLayer class";
    size_t num = 0;
    for (int dim : casted->shape) {
        if (dim < -1)
            THROW_IE_EXCEPTION << "Invalid value of Reshape mask (dim attribute):" << dim
                               << ". Supported values: 0, -1, >0";
        if (dim == -1) num++;
    }
    if (num > 1) THROW_IE_EXCEPTION << "Invalid Reshape mask (dim attribute): at most one dimension can be `-1`";
}

void EltwiseValidator::checkParams(const CNNLayer* layer) {
    auto casted = dynamic_cast<const EltwiseLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of EltwiseLayer class";
    }
}

void EltwiseValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    if (inShapes.empty()) {
        THROW_IE_EXCEPTION << "Number of inputs (" << inShapes.size() << ") of Eltwise layer is zero";
    }
}

EltwiseValidator::EltwiseValidator(const std::string& _type): LayerValidator(_type) {}

ClampValidator::ClampValidator(const std::string& _type): LayerValidator(_type) {}

void ClampValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1});
}

void ReLUValidator::checkParams(const CNNLayer* layer) {
    auto casted = dynamic_cast<const ReLULayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of ReLULayer class";
    }
    if (casted->params.count("negative_slope")) {
        float negative_slope = casted->GetParamAsFloat("negative_slope");
        (void)negative_slope;
    }
}

ReLUValidator::ReLUValidator(const std::string& _type): LayerValidator(_type) {}

void ReLUValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1, 2});
}

void MVNValidator::checkParams(const CNNLayer* layer) {}

MVNValidator::MVNValidator(const std::string& _type): LayerValidator(_type) {}

void MVNValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1});
}

void GRNValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

GRNValidator::GRNValidator(const std::string& _type): LayerValidator(_type) {}

void GRNValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1});
}

void SoftMaxValidator::checkParams(const CNNLayer* layer) {
    auto casted = dynamic_cast<const SoftMaxLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of SoftMaxLayer class";
    }
    int axis = casted->GetParamAsInt("axis", 1);
    if (axis < 0) {
        THROW_IE_EXCEPTION << "The value of SoftMax layer axis parameter is invalid";
    }
}

SoftMaxValidator::SoftMaxValidator(const std::string& _type): LayerValidator(_type) {}

void SoftMaxValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1});
}

void NormValidator::checkParams(const CNNLayer* layer) {
    auto casted = dynamic_cast<const NormLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of NormLayer class";
    }
    float _alpha = casted->GetParamAsFloat("alpha");
    float _beta = casted->GetParamAsFloat("beta");
    if (_alpha < 0 && _beta < 0) {
        THROW_IE_EXCEPTION << "The value of Norm layer alpha or beta parameters is invalid";
    }
}

NormValidator::NormValidator(const std::string& _type): LayerValidator(_type) {}

void NormValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1});
}

SplitValidator::SplitValidator(const std::string& _type): LayerValidator(_type) {}

void SplitValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
    std::vector<int> out_sizes = layer->GetParamAsInts("out_sizes", {});
    if (out_sizes.empty()) {
        THROW_IE_EXCEPTION << "Value of out_sizes attribute is empty";
    }
}

void SplitValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const SplitLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of SplitLayer class";
    }
    checkNumOfInput(inShapes, {1});
    std::vector<int> out_sizes = layer->GetParamAsInts("out_sizes", {});
    size_t sum(0);
    for (const auto& size : out_sizes) sum += size;
    if (inShapes.empty() || inShapes[0].size() <= casted->_axis)
        THROW_IE_EXCEPTION << "Layer has incorrect input shapes!";
    if (sum != inShapes[0][casted->_axis]) {
        THROW_IE_EXCEPTION << "The sum of the dimensions on the axis(" << casted->_axis
                           << ") is not equal out_sizes: " << details::dumpVec(out_sizes);
    }
}

ConcatValidator::ConcatValidator(const std::string& _type): LayerValidator(_type) {}

void ConcatValidator::checkParams(const CNNLayer* layer) {}

void ConcatValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    if (inShapes.empty()) THROW_IE_EXCEPTION << "Inputs are empty";

    auto casted = dynamic_cast<const ConcatLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Invalid Concat layer.";
    }

    auto firstShape = inShapes[0];
    size_t firstShapeSize = firstShape.size();
    size_t axis = casted->_axis;
    if (axis >= firstShapeSize)
        THROW_IE_EXCEPTION << "Concat axis(" << axis << ") should be less the number of current input dimensions ("
                           << firstShapeSize << ")";

    for (size_t i = 1; i < inShapes.size(); i++) {
        auto shape = inShapes[i];
        if (shape.size() != firstShapeSize)
            THROW_IE_EXCEPTION << "Invalid inputs for Concat layer: number of dimensions must match: " << firstShapeSize
                               << " vs " << shape.size();
        bool eq_part1 = std::equal(firstShape.begin(), firstShape.begin() + axis, shape.begin());
        bool eq_part2 = std::equal(firstShape.begin() + axis + 1, firstShape.end(), shape.begin() + axis + 1);
        if (!(eq_part1 && eq_part2))
            THROW_IE_EXCEPTION << "Invalid inputs for Concat layer: dimensions should match in all "
                               << "positions except axis (" << axis << ") : [" << dumpVec(firstShape) << "] vs ["
                               << dumpVec(shape) << "]";
    }
}

GemmValidator::GemmValidator(const std::string& _type): LayerValidator(_type) {}

void GemmValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void GemmValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const GemmLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of GemmLayer class";
    }

    checkNumOfInput(inShapes, {2, 3});

    auto dims0 = inShapes[0];
    auto dims1 = inShapes[1];
    if (dims0.size() < 2 || dims1.size() < 2) {
        THROW_IE_EXCEPTION << "Gemm input shapes must have at least 2 dimensions";
    }

    unsigned long xAxis0 = static_cast<unsigned long>(dims0.size() - 1);
    unsigned long yAxis0 = static_cast<unsigned long>(dims0.size() - 2);

    if (casted->transpose_a) std::swap(xAxis0, yAxis0);

    unsigned long xAxis1 = static_cast<unsigned long>(dims1.size() - 1);
    unsigned long yAxis1 = static_cast<unsigned long>(dims1.size() - 2);

    if (casted->transpose_b) std::swap(xAxis1, yAxis1);

    if (dims0[xAxis0] != dims1[yAxis1])
        THROW_IE_EXCEPTION << "Gemm input0 x dimension must be equal to input1 y dimension (" << dims0[xAxis0] << " vs "
                           << dims1[yAxis1] << ")";

    if (inShapes.size() == 3) {
        auto dims2 = inShapes[2];
        if (dims2.size() < 2) {
            THROW_IE_EXCEPTION << "Gemm input shapes must have at least 2 dimensions";
        }

        unsigned long xAxis2 = static_cast<unsigned long>(dims2.size() - 1);
        unsigned long yAxis2 = static_cast<unsigned long>(dims2.size() - 2);

        if (dims2[xAxis2] != dims1[xAxis1])
            THROW_IE_EXCEPTION << "Gemm input2 x dimension must be equal to input1 x dimension (" << dims2[xAxis2]
                               << " vs " << dims1[xAxis1] << ")";

        if (dims2[yAxis2] != dims0[yAxis0])
            THROW_IE_EXCEPTION << "Gemm input2 y dimension must be equal to input0 y dimension (" << dims2[yAxis2]
                               << " vs " << dims0[yAxis0] << ")";
    }
}

PadValidator::PadValidator(const std::string& _type): LayerValidator(_type) {}

void PadValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void PadValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const PadLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of PadLayer class";
    }

    checkNumOfInput(inShapes, {1});

    if (inShapes[0].size() != casted->pads_begin.size())
        THROW_IE_EXCEPTION << layer->name << " Dimensions count mismatch in layer " << layer->name
                           << ". Expected: " << casted->pads_begin.size() << " Got: " << inShapes[0].size();

    if (inShapes[0].size() != casted->pads_end.size())
        THROW_IE_EXCEPTION << layer->name << " Dimensions count mismatch in layer " << layer->name
                           << ". Expected: " << casted->pads_end.size() << " Got: " << inShapes[0].size();

    if (casted->pad_mode == PadLayer::Symmetric || casted->pad_mode == PadLayer::Reflect) {
        for (size_t i = 0; i < inShapes[0].size(); i++) {
            if (inShapes[0][i] < casted->pads_begin[i]) {
                THROW_IE_EXCEPTION << layer->name
                                   << " Pad can't be grater than input shape in symmetric and reflect modes."
                                   << " For dimension " << i << " pad_begin=" << casted->pads_begin[i]
                                   << " in_shape=" << inShapes[0][i];
            }
            if (inShapes[0][i] < casted->pads_end[i]) {
                THROW_IE_EXCEPTION << layer->name
                                   << " Pad can't be grater than input shape in symmetric and reflect modes."
                                   << " For dimension " << i << " pad_end=" << casted->pads_end[i]
                                   << " in_shape=" << inShapes[0][i];
            }
        }
    }
}

GatherValidator::GatherValidator(const std::string& _type): LayerValidator(_type) {}

void GatherValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void GatherValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const GatherLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of GatherLayer class";
    }

    size_t numInputs = inShapes.size();
    if (numInputs != 2)
        THROW_IE_EXCEPTION << layer->name << " Gather can take only 2 inputs, but actually it has: " << numInputs;

    if (casted->axis > 0 && static_cast<int>(inShapes[0].size()) < (1 + casted->axis))
        THROW_IE_EXCEPTION << layer->name << " Incorrect input dictionary dimensions " << inShapes[0].size()
                           << " and axis number " << casted->axis;
    else if (casted->axis < 0 && (static_cast<int>(inShapes[0].size()) + casted->axis) < 0)
        THROW_IE_EXCEPTION << layer->name << " Incorrect input dictionary dimensions " << inShapes[0].size()
                           << " and axis number " << casted->axis;
}

StridedSliceValidator::StridedSliceValidator(const std::string& _type): LayerValidator(_type) {}

void StridedSliceValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void StridedSliceValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const StridedSliceLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of StridedSliceLayer class";
    }

    size_t numInputs = inShapes.size();
    if (numInputs > 4)
        THROW_IE_EXCEPTION << layer->name
                           << " StridedSlice can take up to 4 inputs, but actually it has: " << numInputs;

    size_t ellipsis_mask_counter = 0;
    for (size_t i = 0; i < casted->ellipsis_mask.size(); ++i) {
        if (casted->ellipsis_mask[i] == '1') ellipsis_mask_counter++;
    }
    if (ellipsis_mask_counter > 1)
        THROW_IE_EXCEPTION << layer->name << " 'Ellipsis_mask' must be a power of two (only one ellipsis)!";
}

ShuffleChannelsValidator::ShuffleChannelsValidator(const std::string& _type): LayerValidator(_type) {}

void ShuffleChannelsValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void ShuffleChannelsValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const ShuffleChannelsLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of ShuffleChannels class";
    }

    size_t numInputs = inShapes.size();
    if (numInputs != 1)
        THROW_IE_EXCEPTION << layer->name
                           << " ShuffleChannels can take only 1 input, but actually it has: " << numInputs;

    if (casted->axis > 0 && static_cast<int>(inShapes[0].size()) < (1 + casted->axis))
        THROW_IE_EXCEPTION << layer->name << "I ncorrect input tensor dimensions " << inShapes[0].size()
                           << " and axis number " << casted->axis;
    else if (casted->axis < 0 && (static_cast<int>(inShapes[0].size()) + casted->axis) < 0)
        THROW_IE_EXCEPTION << layer->name << " Incorrect input dictionary dimensions " << inShapes[0].size()
                           << " and axis number " << casted->axis;

    int axis = casted->axis;
    if (axis < 0) axis += static_cast<int>(inShapes[0].size());

    if (inShapes[0][axis] % casted->group)
        THROW_IE_EXCEPTION << layer->name << " Group parameter must evenly divide the channel dimension!";

    size_t dataLength = 1;
    for (size_t i = axis + 1; i < inShapes[0].size(); i++) dataLength *= inShapes[0][i];

    if (dataLength == 0) THROW_IE_EXCEPTION << layer->name << " Incorrect input parameters dimension!";
}

DepthToSpaceValidator::DepthToSpaceValidator(const std::string& _type): LayerValidator(_type) {}

void DepthToSpaceValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void DepthToSpaceValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const DepthToSpaceLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of DepthToSpace class";
    }

    size_t numInputs = inShapes.size();
    if (numInputs != 1)
        THROW_IE_EXCEPTION << layer->name << " DepthToSpace can take only 1 input, but actually it has: " << numInputs;

    if (inShapes[0].size() < 3) THROW_IE_EXCEPTION << layer->name << " Incorrect number of input dimensions!";

    if (casted->block_size == 0) THROW_IE_EXCEPTION << layer->name << " Incorrect block_size parameter is zero!";

    size_t numSpatialDims = inShapes[0].size() - 2;
    if (inShapes[0][1] % static_cast<size_t>(std::pow(casted->block_size, numSpatialDims)))
        THROW_IE_EXCEPTION << layer->name << " block_size parameter is incompatible with input tensor Color dimension size!";
}

SpaceToDepthValidator::SpaceToDepthValidator(const std::string& _type): LayerValidator(_type) {}

void SpaceToDepthValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void SpaceToDepthValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const SpaceToDepthLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of SpaceToDepth class";
    }

    size_t numInputs = inShapes.size();
    if (numInputs != 1)
        THROW_IE_EXCEPTION << layer->name << " SpaceToDepth can take only 1 input, but actually it has: " << numInputs;

    if (inShapes[0].size() < 2) THROW_IE_EXCEPTION << layer->name << " Incorrect number of input dimensions!";

    if (casted->block_size == 0) THROW_IE_EXCEPTION << layer->name << " Incorrect block_size parameter is zero!";

    if (inShapes[0][inShapes[0].size() - 1] % casted->block_size)
        THROW_IE_EXCEPTION << layer->name
                           << " block_size parameter is incompatible with input tensor With dimension size!";

    if (inShapes[0][inShapes[0].size() - 2] % casted->block_size)
        THROW_IE_EXCEPTION << layer->name
                           << " block_size parameter is incompatible with input tensor Height dimension size!";
}

/*--- SpaceToBatchValidator ---*/
SpaceToBatchValidator::SpaceToBatchValidator(const std::string& _type): LayerValidator(_type) {}

void SpaceToBatchValidator::checkParams(const CNNLayer* layer) {
    const auto spaceToBatchLayer = dynamic_cast<const SpaceToBatchLayer*>(layer);
    if (!spaceToBatchLayer)
        THROW_IE_EXCEPTION << "'" << layer->name << "' layer is not instance of SpaceToBatchLayer class";
    LayerValidator::checkParams(layer);

    for (auto& bs : spaceToBatchLayer->_block_shape) {
        if (bs == 0lu)
            THROW_IE_EXCEPTION << "'" << layer->name << "' layer has nullable block shape.";
    }
}

void SpaceToBatchValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto spaceToBatchLayer = dynamic_cast<const SpaceToBatchLayer*>(layer);
    if (!spaceToBatchLayer) {
        THROW_IE_EXCEPTION << "'" << layer->name << "' layer is not instance of SpaceToBatchLayer class";
    }
    // TODO: finish
}

/*--- BatchToSpaceValidator ---*/
BatchToSpaceValidator::BatchToSpaceValidator(const std::string& _type): LayerValidator(_type) {}

void BatchToSpaceValidator::checkParams(const CNNLayer* layer) {
    auto batchToSpaceLayer = dynamic_cast<const BatchToSpaceLayer*>(layer);
    if (!batchToSpaceLayer) {
        THROW_IE_EXCEPTION << "'" << layer->name << "' layer is not instance of BatchToSpaceLayer class";
    }
    LayerValidator::checkParams(layer);
}

void BatchToSpaceValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto batchToSpaceLayer = dynamic_cast<const BatchToSpaceLayer*>(layer);
    if (!batchToSpaceLayer) {
        THROW_IE_EXCEPTION << "'" << layer->name << "' layer is not instance of BatchToSpaceLayer class";
    }
    // TODO: finish
}

/*--- SparseFillEmptyRowsValidator ---*/
SparseFillEmptyRowsValidator::SparseFillEmptyRowsValidator(const std::string& _type): LayerValidator(_type) {}

void SparseFillEmptyRowsValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void SparseFillEmptyRowsValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const SparseFillEmptyRowsLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of SparseFillEmptyRows class";
    }

    size_t numInputs = inShapes.size();
    if (numInputs != 4)
        THROW_IE_EXCEPTION << layer->name
                           << " SparseFillEmptyRows must have 4 inputs, but actually it has: " << numInputs;

    // Check dimensions of a tensor with input indices
    if (inShapes[0].size() != 2)
        THROW_IE_EXCEPTION << layer->name << " Input indices of SparseFillEmptyRows must be 2-D tensor";
    if (inShapes[0][1] != 2) THROW_IE_EXCEPTION << layer->name << " Input indices must be two-dimensional";

    // Check dimensions of a tensor with input values
    if (inShapes[1].size() != 1)
        THROW_IE_EXCEPTION << layer->name << " Input values of SparseFillEmptyRows must be 1-D tensor";
    if (inShapes[1][0] != inShapes[0][0])
        THROW_IE_EXCEPTION << layer->name << " Number of input indices and values must match";

    // Check dimensions of a tensor with a dense shape
    if (inShapes[2].size() != 1)
        THROW_IE_EXCEPTION << layer->name << " Dense shape of SparseFillEmptyRows must be 1-D tensor";
    // TODO: check that dense shape value is set

    // Check dimensions of a tensor with default value
    if (inShapes[3].size() != 1)
        THROW_IE_EXCEPTION << layer->name << " Default value of SparseFillEmptyRows must be 1-D tensor";
}

SparseSegmentReduceValidator::SparseSegmentReduceValidator(const std::string& _type): LayerValidator(_type) {}

void SparseSegmentReduceValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void SparseSegmentReduceValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const SparseSegmentReduceLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of SparseSegmentReduce class";
    }

    size_t numInputs = inShapes.size();
    if (numInputs != 3)
        THROW_IE_EXCEPTION << layer->name
                           << " SparseSegmentReduce must take three inputs, but actually it has: " << numInputs;

    // check that the second and the third inputs are one-dimensional
    if (inShapes[1].size() != 1) {
        THROW_IE_EXCEPTION << layer->name << " The second input of SparseSegmentReduce must be one-dimensional";
    }
    if (inShapes[2].size() != 1) {
        THROW_IE_EXCEPTION << layer->name << " The third input of SparseSegmentReduce must be one-dimensional";
    }
}

ExperimentalSparseWeightedReduceValidator::ExperimentalSparseWeightedReduceValidator(const std::string& _type) : LayerValidator(_type) {}

void ExperimentalSparseWeightedReduceValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void ExperimentalSparseWeightedReduceValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const ExperimentalSparseWeightedReduceLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of ExperimentalSparseWeightedReduce class";
    }

    size_t numInputs = inShapes.size();
    if (numInputs != 5 && numInputs != 6)
        THROW_IE_EXCEPTION << layer->name
                           << " ExperimentalSparseWeightedReduce must take five or six inputs, but actually it has: "
                           << numInputs;

    // check shapes of inputs
    if (inShapes[0].size() != 2 || inShapes[0][1] != 2) {
        THROW_IE_EXCEPTION << layer->name
                           << " The first input with indices of ExperimentalSparseSegmentReduce must be two-dimensional";
    }
    if (inShapes[1].size() != 1) {
        THROW_IE_EXCEPTION << layer->name
                           << " The second input with values of ExperimentalSparseSegmentReduce must be one-dimensional";
    }
    if (inShapes[1][0] != inShapes[0][0]) {
        THROW_IE_EXCEPTION << layer->name
            << "  The first dimension value of the first and second inputs of ExperimentalSparseSegmentReduce must be the same";
    }
    if (inShapes[2].size() != 1 || inShapes[2][0] != 2) {
        THROW_IE_EXCEPTION << layer->name
            << " The third input with a dense shape of ExperimentalSparseSegmentReduce must be one-dimensional";
    }
    if (inShapes[3].size() < 2) {
        THROW_IE_EXCEPTION << layer->name
            << " The fourth input with parameters table of ExperimentalSparseSegmentReduce must have two dimensions at least";
    }
    if (inShapes[4].size() != 0) {
        THROW_IE_EXCEPTION << layer->name
            << " The fifth input with a default value of ExperimentalSparseSegmentReduce must be a scalar";
    }
    if (numInputs == 6 && (inShapes[5].size() != 1 || inShapes[5][0] != inShapes[1][0])) {
        THROW_IE_EXCEPTION << layer->name
            << " The second and sixth inputs of ExperimentalSparseSegmentReduce must have the same shapes";
    }

    // check precisions of inputs
    const size_t INPUT_INDICES_PORT = 0;
    const size_t INPUT_VALUES_PORT = 1;
    const size_t INPUT_DENSE_SHAPE = 2;
    const size_t INPUT_PARAMS_TABLE = 3;
    const size_t INPUT_DEFAULT_VALUE = 4;
    const size_t INPUT_WEIGHTS = 5;

    Precision input_indices_precision = layer->insData[INPUT_INDICES_PORT].lock()->getTensorDesc().getPrecision();
    if (input_indices_precision != Precision::I32)
        THROW_IE_EXCEPTION << layer->name << " Incorrect input indices precision. Only I32 are supported!";
    Precision input_values_precision = layer->insData[INPUT_VALUES_PORT].lock()->getTensorDesc().getPrecision();
    if (input_values_precision != Precision::I32)
        THROW_IE_EXCEPTION << layer->name << " Incorrect input values precision. Only I32 are supported!";
    Precision input_dense_shape_precision = layer->insData[INPUT_DENSE_SHAPE].lock()->getTensorDesc().getPrecision();
    if (input_dense_shape_precision != Precision::I32)
        THROW_IE_EXCEPTION << layer->name << " Incorrect input dense shape precision. Only I32 are supported!";
    Precision input_params_table_precision = layer->insData[INPUT_PARAMS_TABLE].lock()->getTensorDesc().getPrecision();
    if (input_params_table_precision != Precision::FP32)
        THROW_IE_EXCEPTION << layer->name << " Incorrect input parameters table precision. Only FP32 are supported!";
    Precision input_default_value_precision = layer->insData[INPUT_DEFAULT_VALUE].lock()->getTensorDesc().getPrecision();
    if (input_default_value_precision != Precision::I32)
        THROW_IE_EXCEPTION << layer->name << " Incorrect input default value precision. Only I32 are supported!";
    if (numInputs == 6) {
        Precision input_weights_precision = layer->insData[INPUT_WEIGHTS].lock()->getTensorDesc().getPrecision();
        if (input_weights_precision != Precision::FP32)
            THROW_IE_EXCEPTION << layer->name << " Incorrect input weights precision. Only FP32 are supported!";
    }
}


SparseToDenseValidator::SparseToDenseValidator(const std::string& _type) : LayerValidator(_type) {}

void SparseToDenseValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void SparseToDenseValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const SparseToDenseLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of SparseToDense class";
    }

    size_t numInputs = inShapes.size();
    if (numInputs != 3 && numInputs != 4)
        THROW_IE_EXCEPTION << layer->name
                           << " SparseToDense must take three or four inputs, but actually it has: "
                           << numInputs;

    // check shapes of inputs
    if (inShapes[0].size() != 2) {
        THROW_IE_EXCEPTION << layer->name
                           << " The first input with indices of SparseToDense must be two-dimensional";
    }
    if (inShapes[1].size() != 1 || inShapes[1][0] != inShapes[0][1]) {
        THROW_IE_EXCEPTION << layer->name
                           << " The second input with a dense shape of SparseToDense must be one-dimensional";
    }
    if (inShapes[2].size() != 1 || inShapes[2][0] != inShapes[0][0]) {
        THROW_IE_EXCEPTION << layer->name
                           << " The third input with values of SparseToDense must be one-dimensional";
    }
    if (numInputs == 4 && inShapes[3].size() != 0) {
        THROW_IE_EXCEPTION << layer->name
            << " The fourth input with default value of SparseToDense must be a scalar";
    }

    // check precisions of inputs
    const size_t INPUT_INDICES_PORT = 0;
    const size_t INPUT_DENSE_SHAPE = 1;
    const size_t INPUT_VALUES_PORT = 2;
    const size_t INPUT_DEFAULT_VALUE = 3;

    Precision input_indices_precision = layer->insData[INPUT_INDICES_PORT].lock()->getTensorDesc().getPrecision();
    if (input_indices_precision != Precision::I32)
        THROW_IE_EXCEPTION << layer->name << " Incorrect input indices precision. Only I32 are supported!";
    Precision input_dense_shape_precision = layer->insData[INPUT_DENSE_SHAPE].lock()->getTensorDesc().getPrecision();
    if (input_dense_shape_precision != Precision::I32)
        THROW_IE_EXCEPTION << layer->name << " Incorrect input dense shape precision. Only I32 are supported!";
    Precision input_values_precision = layer->insData[INPUT_VALUES_PORT].lock()->getTensorDesc().getPrecision();
    if (input_values_precision != Precision::I32)
        THROW_IE_EXCEPTION << layer->name << " Incorrect input values precision. Only I32 are supported!";
    if (numInputs == 4) {
        Precision input_default_value_precision = layer->insData[INPUT_DEFAULT_VALUE].lock()->getTensorDesc().getPrecision();
        if (input_default_value_precision != Precision::I32)
            THROW_IE_EXCEPTION << layer->name << " Incorrect input default value precision. Only I32 are supported!";
    }
}


BucketizeValidator::BucketizeValidator(const std::string& _type) : LayerValidator(_type) {}

void BucketizeValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void BucketizeValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const BucketizeLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of Bucketize class";
    }

    size_t numInputs = inShapes.size();
    if (numInputs != 2)
        THROW_IE_EXCEPTION << layer->name
                           << " Bucketize must take two inputs, but actually it has: "
                           << numInputs;

    // check shapes of inputs
    if (inShapes[1].size() != 1) {
        THROW_IE_EXCEPTION << layer->name
                           << " The second input of Bucketize must be one-dimensional";
    }

    // check precisions of inputs
    const size_t INPUT_TENSOR_PORT = 0;
    const size_t INPUT_BOUNDARIES_PORT = 1;

    Precision input_tensor_precision = layer->insData[INPUT_TENSOR_PORT].lock()->getTensorDesc().getPrecision();
    if (input_tensor_precision != Precision::I32 || input_tensor_precision != Precision::FP32)
        THROW_IE_EXCEPTION << layer->name << " Incorrect input tensor precision. Only I32 and FP32 are supported!";
    Precision input_boundaries_precision = layer->insData[INPUT_BOUNDARIES_PORT].lock()->getTensorDesc().getPrecision();
    if (input_boundaries_precision != Precision::FP32)
        THROW_IE_EXCEPTION << layer->name << " Incorrect input boundaries precision. Only FP32 are supported!";
}

ReverseSequenceValidator::ReverseSequenceValidator(const std::string& _type): LayerValidator(_type) {}

void ReverseSequenceValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void ReverseSequenceValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const ReverseSequenceLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of ReverseSequence class";
    }

    size_t numInputs = inShapes.size();
    if (numInputs != 2)
        THROW_IE_EXCEPTION << layer->name << " ReverseSequence can take 2 inputs, but actually it has: " << numInputs;

    if (inShapes[1].size() != 1)
        THROW_IE_EXCEPTION << layer->name << " Incorrect number of 'seq_lengths' input dimensions!";

    if (casted->seq_axis > 0 && static_cast<int>(inShapes[0].size()) < (1 + casted->seq_axis))
        THROW_IE_EXCEPTION << layer->name << "Incorrect input tensor dimensions " << inShapes[0].size()
                           << " and seq_axis number " << casted->seq_axis;
    else if (casted->seq_axis < 0 && (static_cast<int>(inShapes[0].size()) + casted->seq_axis) < 0)
        THROW_IE_EXCEPTION << layer->name << " Incorrect input dictionary dimensions " << inShapes[0].size()
                           << " and seq_axis number " << casted->seq_axis;

    if (casted->batch_axis > 0 && static_cast<int>(inShapes[0].size()) < (1 + casted->batch_axis))
        THROW_IE_EXCEPTION << layer->name << "Incorrect input tensor dimensions " << inShapes[0].size()
                           << " and batch_axis number " << casted->batch_axis;
    else if (casted->batch_axis < 0 && (static_cast<int>(inShapes[0].size()) + casted->batch_axis) < 0)
        THROW_IE_EXCEPTION << layer->name << " Incorrect input dictionary dimensions " << inShapes[0].size()
                           << " and batch_axis number " << casted->batch_axis;

    int batch_axis = casted->batch_axis;
    if (batch_axis < 0) batch_axis += static_cast<int>(inShapes[0].size());
    if (inShapes[1][0] != inShapes[0][batch_axis])
        THROW_IE_EXCEPTION << layer->name << " Incorrect 'seq_lengths_dims' parameter dimensions!";
}

SqueezeValidator::SqueezeValidator(const std::string& _type): LayerValidator(_type) {}

void SqueezeValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void SqueezeValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const CNNLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of Squeeze class";
    }

    size_t numInputs = inShapes.size();
    if (numInputs != 2)
        THROW_IE_EXCEPTION << layer->name << " Squeeze can take 2 inputs, but actually it has: " << numInputs;

    if (inShapes[1].size() != 1)
        THROW_IE_EXCEPTION << layer->name << " Incorrect number of 'indices_to_squeeze' input dimensions!";
}

UnsqueezeValidator::UnsqueezeValidator(const std::string& _type): LayerValidator(_type) {}

void UnsqueezeValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void UnsqueezeValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const CNNLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of Unsqueeze class";
    }

    size_t numInputs = inShapes.size();
    if (numInputs != 2)
        THROW_IE_EXCEPTION << layer->name << " Unsqueeze can take 2 inputs, but actually it has: " << numInputs;

    if (inShapes[1].size() != 1)
        THROW_IE_EXCEPTION << layer->name << " Incorrect number of 'indices_to_set' input dimensions!";
}

RangeValidator::RangeValidator(const std::string& _type): LayerValidator(_type) {}

void RangeValidator::checkParams(const CNNLayer* layer) {}

void RangeValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const RangeLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of Range class";
    }

    size_t numInputs = inShapes.size();
    if (numInputs != 3)
        THROW_IE_EXCEPTION << layer->name << " Range can take 3 inputs, but actually it has: " << numInputs;

    if (inShapes[0].size() != 1) THROW_IE_EXCEPTION << layer->name << " Incorrect number of 'start' input dimensions!";

    if (inShapes[1].size() != 1) THROW_IE_EXCEPTION << layer->name << " Incorrect number of 'limit' input dimensions!";

    if (inShapes[2].size() != 1) THROW_IE_EXCEPTION << layer->name << " Incorrect number of 'delta' input dimensions!";
}

FillValidator::FillValidator(const std::string& _type): LayerValidator(_type) {}

void FillValidator::checkParams(const CNNLayer* layer) {}

void FillValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    size_t numInputs = inShapes.size();
    if (numInputs != 2)
        THROW_IE_EXCEPTION << layer->name << " Fill can take 2 inputs, but actually it has: " << numInputs;

    if (inShapes[0].size() != 1)
        THROW_IE_EXCEPTION << layer->name << " Incorrect number of 'fill_dims' input dimensions!";

    if (inShapes[1].size() != 1)
        THROW_IE_EXCEPTION << layer->name << " Incorrect number of 'fill_value' input dimensions!";
}

BroadcastValidator::BroadcastValidator(const std::string& _type): LayerValidator(_type) {}

void BroadcastValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void BroadcastValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const BroadcastLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of Broadcast class";
    }

    size_t numInputs = inShapes.size();
    if (numInputs != 2)
        THROW_IE_EXCEPTION << layer->name << " Broadcast can take 2 inputs, but actually it has: " << numInputs;

    if (inShapes[1].size() != 1) THROW_IE_EXCEPTION << layer->name << " Incorrect number of 'shape' input dimensions!";
}

/****************************************/
/*** RNN specific validators ************/
/****************************************/

RNNBaseValidator::RNNBaseValidator(const std::string& _type, RNNSequenceLayer::CellType CELL): LayerValidator(_type) {
    if (RNNSequenceLayer::LSTM == CELL) {
        def_acts = {"sigmoid", "tanh", "tanh"};
        def_alpha = {0, 0, 0};
        def_beta = {0, 0, 0};
        G = 4;
        NS = 2;
    } else if (RNNSequenceLayer::GRU == CELL) {
        def_acts = {"sigmoid", "tanh"};
        def_alpha = {0, 0};
        def_beta = {0, 0};
        G = 3;
        NS = 1;
    } else if (RNNSequenceLayer::RNN == CELL) {
        def_acts = {"tanh"};
        def_alpha = {0};
        def_beta = {0};
        G = 1;
        NS = 1;
    } else {
        IE_ASSERT(false);
    }
}

void RNNBaseValidator::checkParams(const InferenceEngine::CNNLayer* layer) {
    auto rnn = dynamic_cast<const RNNCellBase*>(layer);
    if (!rnn) THROW_IE_EXCEPTION << "Layer is not instance of RNNLayer class";

    if (rnn->clip < 0.0f) THROW_IE_EXCEPTION << "Clip parameter should be positive";

    for (auto& act : rnn->activations)
        if (!one_of(act, "sigmoid", "tanh", "relu"))
            THROW_IE_EXCEPTION << "Unsupported activation function (" << act << ") for RNN layer.";

    size_t act_num_required = def_acts.size();
    if (rnn->activations.size() != act_num_required)
        THROW_IE_EXCEPTION << "Expected " << act_num_required << " activations, but provided "
                           << rnn->activations.size();

    if (rnn->activation_alpha.size() != act_num_required)
        THROW_IE_EXCEPTION << "Expected " << act_num_required << " activation alpha parameters, "
                           << "but provided " << rnn->activation_alpha.size();
    if (rnn->activation_beta.size() != act_num_required)
        THROW_IE_EXCEPTION << "Expected " << act_num_required << " activation beta parameters, "
                           << "but provided " << rnn->activation_beta.size();
}

void RNNBaseValidator::checkCorrespondence(const CNNLayer* layer, const map<string, Blob::Ptr>& blobs,
                                           const vector<SizeVector>& inShapes) const {
    auto rnn = dynamic_cast<const RNNCellBase*>(layer);
    if (!rnn) THROW_IE_EXCEPTION << "Layer is not instance of RNNLayer class";

    if (blobs.size() != 2)
        THROW_IE_EXCEPTION << "Expected only 2 blobs with trained parameters (weights and biases), "
                           << "but provided only " << blobs.size();
    if (inShapes.empty()) THROW_IE_EXCEPTION << "No input tensors.";

    size_t D = inShapes[0].back();
    size_t S = rnn->hidden_size;
    size_t expectetd_w_size = G * S * (D + S);
    size_t expectetd_b_size = G * S;

    if (rnn->cellType == RNNCellBase::GRU_LBR) expectetd_b_size = (G + 1) * S;

    auto w = blobs.find("weights");
    if (w == blobs.end()) THROW_IE_EXCEPTION << "Weights blob is not provided";

    if (w->second->size() != expectetd_w_size)
        THROW_IE_EXCEPTION << "Weights blob has wrang size. Expected " << expectetd_w_size;

    auto b = blobs.find("biases");
    if (b == blobs.end()) THROW_IE_EXCEPTION << "Biases blob is not provided";

    if (b->second->size() != expectetd_b_size)
        THROW_IE_EXCEPTION << "Biases blob has wrang size. Expected " << expectetd_b_size;
}

template <RNNSequenceLayer::CellType CELL>
RNNSequenceValidator<CELL>::RNNSequenceValidator(const std::string& _type): RNNBaseValidator(_type, CELL) {}

template <RNNSequenceLayer::CellType CELL>
void RNNSequenceValidator<CELL>::checkParams(const InferenceEngine::CNNLayer* layer) {
    RNNBaseValidator::checkParams(layer);

    auto casted = dynamic_cast<const RNNSequenceLayer*>(layer);
    if (!casted) THROW_IE_EXCEPTION << "Layer is not instance of RNNLayer class";

    if (!one_of(casted->axis, 1u, 0u))
        THROW_IE_EXCEPTION << "Unsupported iteration axis for RNNSequense layer. Only 0 or 1 axis are supported.";
}

template <RNNSequenceLayer::CellType CELL>
void RNNSequenceValidator<CELL>::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto rnn = dynamic_cast<const RNNSequenceLayer*>(layer);
    if (!rnn) THROW_IE_EXCEPTION << "Layer is not instance of RNNSequenceLayer class";

    if (inShapes.empty()) THROW_IE_EXCEPTION << "No input tensors.";

    if (inShapes[0].size() != 3) THROW_IE_EXCEPTION << "First input data tensor should be 3D";

    size_t T_axis = rnn->axis;
    size_t N_axis = (T_axis + 1) % 2;
    size_t N = inShapes[0][N_axis];
    size_t S = rnn->hidden_size;
    size_t NS = RNNSequenceValidator<CELL>::NS;

    SizeVector expected_state_shape {N, S};
    SizeVector expected_seq_l_shape {N};

    if (inShapes.size() > 1) {  // has an initial state blobs
        if (inShapes.size() != 1 + NS && inShapes.size() != 2 + NS)
            THROW_IE_EXCEPTION << "Wrong number of input tensors. Expected 1 (data) or " << 1 + NS
                               << " (data and states) or " << 2 + NS << " (data, states and seq_length).";
        if (inShapes[1] != expected_state_shape) THROW_IE_EXCEPTION << "Wrong shape of first initial state tensors.";
        //                             << " Expected " << expected_state_shape << " but provided " << inShapes[1];

        if (NS == 2 && inShapes[2] != expected_state_shape)
            THROW_IE_EXCEPTION << "Wrong shape of second initial state tensors.";
        //                             << " Expected " << expected_state_shape << " but provided " << inShapes[2];
        if (inShapes.size() == 2 + NS && inShapes[NS + 1] != expected_seq_l_shape)
            THROW_IE_EXCEPTION << "Wrong shape of last input tensor with sequance length data.";
        //                             << " Expected " << expected_seq_l_shape << " but provided " << inShapes[NS + 1];
    }
}

template class details::RNNSequenceValidator<RNNSequenceLayer::RNN>;
template class details::RNNSequenceValidator<RNNSequenceLayer::GRU>;
template class details::RNNSequenceValidator<RNNSequenceLayer::LSTM>;

template <RNNSequenceLayer::CellType CELL>
RNNCellValidator<CELL>::RNNCellValidator(const std::string& _type): RNNBaseValidator(_type, CELL) {}

template <RNNSequenceLayer::CellType CELL>
void RNNCellValidator<CELL>::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto rnn = dynamic_cast<const RNNCellBase*>(layer);
    if (!rnn) THROW_IE_EXCEPTION << "Layer is not instance of RNNSequenceLayer class";

    const size_t& NS = RNNCellValidator<CELL>::NS;

    if (inShapes.size() != NS + 1) THROW_IE_EXCEPTION << "Wrong number of input tensors. Expected " << NS + 1;

    if (inShapes[0].size() != 2) THROW_IE_EXCEPTION << "First input data tensor should be 2D";

    size_t N = inShapes[0][0];
    size_t S = rnn->hidden_size;

    SizeVector expected_state_shape {N, S};

    if (inShapes[1] != expected_state_shape) THROW_IE_EXCEPTION << "Wrong shape of first initial state tensors.";
    //                         << " Expected " << expected_state_shape << " but provided " << inShapes[1];

    if (NS == 2 && inShapes[2] != expected_state_shape)
        THROW_IE_EXCEPTION << "Wrong shape of second initial state tensors.";
    //                         << " Expected " << expected_state_shape << " but provided " << inShapes[2];
}

template class details::RNNCellValidator<RNNSequenceLayer::RNN>;
template class details::RNNCellValidator<RNNSequenceLayer::GRU>;
template class details::RNNCellValidator<RNNSequenceLayer::LSTM>;

void ArgMaxValidator::checkParams(const CNNLayer* layer) {
    unsigned int top_k_ = layer->GetParamAsUInt("top_k");
    (void)top_k_;
}

void ArgMaxValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1});
}

ArgMaxValidator::ArgMaxValidator(const std::string& _type): LayerValidator(_type) {}

void CTCGreedyDecoderValidator::checkParams(const CNNLayer* layer) {
    int flag = layer->GetParamAsInt("ctc_merge_repeated", 0);
    if (flag != 0 && flag != 1) {
        THROW_IE_EXCEPTION << "CTCGreedyDecoder layer parameter ctc_merge_repeated is invalid";
    }
}

void CTCGreedyDecoderValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1, 2});
}

CTCGreedyDecoderValidator::CTCGreedyDecoderValidator(const std::string& _type): LayerValidator(_type) {}

void DetectionOutputValidator::checkParams(const CNNLayer* layer) {
    unsigned int num_classes = layer->GetParamAsUInt("num_classes");
    if (num_classes == 0) {
        THROW_IE_EXCEPTION << "num_classes parameter of DetectionOutput layer can't be equal to zero";
    }
    float _nms_threshold = layer->GetParamAsFloat("nms_threshold");
    if (_nms_threshold < 0) {
        THROW_IE_EXCEPTION << "nms_threshold parameter of DetectionOutput layer can't be less then zero";
    }
    int _keep_top_k = layer->GetParamAsUInt("keep_top_k", -1);
    (void)_keep_top_k;

    if (layer->CheckParamPresence("background_label_id")) {
        int _background_label_id = layer->GetParamAsUInt("background_label_id", -1);
        (void)_background_label_id;
    }
    if (layer->CheckParamPresence("top_k")) {
        int _top_k = layer->GetParamAsUInt("top_k", -1);
        (void)_top_k;
    }
    if (layer->CheckParamPresence("variance_encoded_in_target")) {
        bool _variance_encoded_in_target = static_cast<bool>(layer->GetParamAsUInt("variance_encoded_in_target"));
        (void)_variance_encoded_in_target;
    }
    if (layer->CheckParamPresence("num_orient_classes")) {
        int _num_orient_classes = layer->GetParamAsUInt("num_orient_classes");
        (void)_num_orient_classes;
    }
    if (layer->CheckParamPresence("share_location")) {
        bool _share_location = static_cast<bool>(layer->GetParamAsUInt("share_location"));
        (void)_share_location;
    }
    if (layer->CheckParamPresence("interpolate_orientation")) {
        int _interpolate_orientation = layer->GetParamAsInt("interpolate_orientation");
        (void)_interpolate_orientation;
    }
    if (layer->CheckParamPresence("confidence_threshold")) {
        float _confidence_threshold = layer->GetParamAsFloat("confidence_threshold");
        if (_confidence_threshold < 0) {
            THROW_IE_EXCEPTION << "_nms_threshold parameter of DetectionOutput layer can't be less then zero";
        }
    }
    if (layer->CheckParamPresence("code_type")) {
        std::string _code_type = layer->GetParamAsString("code_type");
        std::vector<std::string> code_types = {"caffe.PriorBoxParameter.CENTER_SIZE", "caffe.PriorBoxParameter.CORNER"};
        auto it = std::find(code_types.begin(), code_types.end(), _code_type);
        if (it == code_types.end()) {
            THROW_IE_EXCEPTION << "Parameter code_type of DetectionOutput layer ";
        }
    }
}

void DetectionOutputValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {3, 5});
}

DetectionOutputValidator::DetectionOutputValidator(const std::string& _type): LayerValidator(_type) {}

void InterpValidator::checkParams(const CNNLayer* layer) {}

void InterpValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1, 2});
    auto IS_ZERO = [](float value) {
        return std::fabs(value) < std::numeric_limits<float>::epsilon();
    };
    if (inShapes.size() != 2) {
        float factor = layer->GetParamAsFloat("factor", 0);
        if (factor < 0) THROW_IE_EXCEPTION << "factor parameter of Interp layer can't be less then zero";
        float shrink_factor = layer->GetParamAsFloat("shrink_factor", 0);
        if (shrink_factor < 0) THROW_IE_EXCEPTION << "shrink_factor parameter of Interp layer can't be less then zero";
        float zoom_factor = (layer->GetParamAsFloat("zoom_factor", 0));
        if (zoom_factor < 0) THROW_IE_EXCEPTION << "zoom_factor parameter of Interp layer can't be less then zero";
        bool noFactor = IS_ZERO(factor) && IS_ZERO(shrink_factor) && IS_ZERO(zoom_factor);

        auto height = layer->GetParamAsUInt("height", 0);
        auto width = layer->GetParamAsUInt("width", 0);

        if (noFactor && (height == 0 || width == 0)) {
            THROW_IE_EXCEPTION << "Can't reshape without factor, or target resolution. "
                               << "Supported attributes: factor, shrink_factor, zoom_factor, height, width";
        }
    }
}

InterpValidator::InterpValidator(const std::string& _type): LayerValidator(_type) {}

void PermuteValidator::checkParams(const CNNLayer* layer) {
    std::vector<unsigned int> layerOrder = layer->GetParamAsUInts("order");
}

void PermuteValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1});
}

PermuteValidator::PermuteValidator(const std::string& _type): LayerValidator(_type) {}

void PriorBoxValidator::checkParams(const CNNLayer* layer) {
    std::vector<unsigned int> min_sizes = layer->GetParamAsUInts("min_size", {});
    std::vector<unsigned int> max_sizes = layer->GetParamAsUInts("max_size", {});
    bool flip = static_cast<bool>(layer->GetParamAsInt("flip"));
    (void)flip;
    if (layer->CheckParamPresence("aspect_ratio"))
        const std::vector<unsigned int> aspect_ratios = layer->GetParamAsUInts("aspect_ratio", {});
    bool clip_ = static_cast<bool>(layer->GetParamAsInt("clip"));
    (void)clip_;
    std::vector<float> variance_ = layer->GetParamAsFloats("variance", {});
    for (float i : variance_) {
        if (i < 0) {
            THROW_IE_EXCEPTION << "The value of PriorBox layer variance parameter is invalid. "
                                  "Positive value is expected";
        }
    }
    float step_ = layer->GetParamAsFloat("step", 0);
    if (step_ < 0) {
        THROW_IE_EXCEPTION << "The value of PriorBox layer step_ parameter is invalid";
    }
    float offset_ = layer->GetParamAsFloat("offset");
    if (offset_ < 0) {
        THROW_IE_EXCEPTION << "The value of PriorBox layer offset_ parameter is invalid";
    }
}

void PriorBoxValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {2});
}

PriorBoxValidator::PriorBoxValidator(const std::string& _type): LayerValidator(_type) {}

void PriorBoxClusteredValidator::checkParams(const CNNLayer* layer) {
    std::vector<float> widths = layer->GetParamAsFloats("width", {});
    for (auto i : widths) {
        if (i < 0) {
            THROW_IE_EXCEPTION << "The value of PriorBoxClustered layer width parameter is invalid";
        }
    }
    std::vector<float> heights = layer->GetParamAsFloats("height", {});
    for (auto i : heights) {
        if (i < 0) {
            THROW_IE_EXCEPTION << "The value of PriorBoxClustered layer heights parameter is invalid";
        }
    }
    bool flip = static_cast<bool>(layer->GetParamAsInt("flip"));
    (void)flip;
    bool clip_ = static_cast<bool>(layer->GetParamAsInt("clip"));
    (void)clip_;
    float offset_ = layer->GetParamAsFloat("offset");
    if (offset_ < 0) {
        THROW_IE_EXCEPTION << "The value of PriorBox layer offset_ parameter is invalid";
    }

    std::vector<float> variance_ = layer->GetParamAsFloats("variance", {});
    for (float i : variance_) {
        if (i < 0) {
            THROW_IE_EXCEPTION << "The value of PriorBox layer variance parameter is invalid. "
                                  "Positive value is expected";
        }
    }
    float step_h_ = layer->GetParamAsFloat("step_h", 0);
    if (step_h_ < 0) {
        THROW_IE_EXCEPTION << "The value of PriorBox layer step_h_ parameter is invalid";
    }
    float step_w_ = layer->GetParamAsFloat("step_w", 0);
    if (step_w_ < 0) {
        THROW_IE_EXCEPTION << "The value of PriorBox layer step_w_ parameter is invalid";
    }
    float img_h_ = layer->GetParamAsFloat("img_h", 0);
    if (img_h_ < 0) {
        THROW_IE_EXCEPTION << "The value of PriorBox layer img_h_ parameter is invalid";
    }
    float img_w_ = layer->GetParamAsFloat("img_w", 0);
    if (img_w_ < 0) {
        THROW_IE_EXCEPTION << "The value of PriorBox layer img_w_ parameter is invalid";
    }
}

void PriorBoxClusteredValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {2});
}

PriorBoxClusteredValidator::PriorBoxClusteredValidator(const std::string& _type): LayerValidator(_type) {}

void ProposalValidator::checkParams(const CNNLayer* layer) {
    unsigned int post_nms_topn_ = layer->GetParamAsUInt("post_nms_topn");
    (void)post_nms_topn_;

    if (layer->CheckParamPresence("feat_stride")) {
        unsigned int feat_stride_ = layer->GetParamAsUInt("feat_stride");
        (void)feat_stride_;
    }
    if (layer->CheckParamPresence("base_size")) {
        unsigned int base_size_ = layer->GetParamAsUInt("base_size");
        (void)base_size_;
    }
    if (layer->CheckParamPresence("min_size")) {
        unsigned int min_size_ = layer->GetParamAsUInt("min_size");
        (void)min_size_;
    }
    if (layer->CheckParamPresence("pre_nms_topn")) {
        unsigned int pre_nms_topn_ = layer->GetParamAsUInt("pre_nms_topn");
        (void)pre_nms_topn_;
    }
    if (layer->CheckParamPresence("nms_thresh")) {
        float nms_thresh_ = layer->GetParamAsFloat("nms_thresh");
        if (nms_thresh_ < 0) {
            THROW_IE_EXCEPTION << "The value of Proposal layer nms_thresh_ parameter is invalid";
        }
    }
}

void ProposalValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {3});
}

ProposalValidator::ProposalValidator(const std::string& _type): LayerValidator(_type) {}

void PSROIPoolingValidator::checkParams(const CNNLayer* layer) {
    unsigned int output_dim = layer->GetParamAsUInt("output_dim");
    (void)output_dim;
    unsigned int group_size = layer->GetParamAsUInt("group_size");
    (void)group_size;
    if (layer->CheckParamPresence("spatial_scale")) {
        float spatial_scale_ = layer->GetParamAsFloat("spatial_scale");
        if (spatial_scale_ < 0) {
            THROW_IE_EXCEPTION << "The value of PSROIPooling layer spatial_scale_ parameter is invalid";
        }
    }
}

void PSROIPoolingValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1, 2, 3});
}

PSROIPoolingValidator::PSROIPoolingValidator(const std::string& _type): LayerValidator(_type) {}

void RegionYoloValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void RegionYoloValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1});
}

RegionYoloValidator::RegionYoloValidator(const std::string& _type): LayerValidator(_type) {}

void ReorgYoloValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void ReorgYoloValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1});
}

ReorgYoloValidator::ReorgYoloValidator(const std::string& _type): LayerValidator(_type) {}

void ResampleValidator::checkParams(const CNNLayer* layer) {
    if (layer->CheckParamPresence("antialias")) {
        auto antialias = static_cast<size_t>(layer->GetParamAsInt("antialias"));

        if (antialias != 0 && antialias != 1) {
            THROW_IE_EXCEPTION << "The value of resample layer antialias parameter is invalid";
        }
    }
    if (layer->CheckParamPresence("type")) {
        std::string type = layer->GetParamAsString("type");
        if (type != "caffe.ResampleParameter.NEAREST" && type != "caffe.ResampleParameter.CUBIC" &&
            type != "caffe.ResampleParameter.LINEAR") {
            THROW_IE_EXCEPTION << "The value of resample layer type parameter is invalid";
        }
    }
}

void ResampleValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1, 2});
}

ResampleValidator::ResampleValidator(const std::string& _type): LayerValidator(_type) {}

void ROIPoolingValidator::checkParams(const CNNLayer* layer) {
    unsigned int pooled_h = layer->GetParamAsUInt("pooled_h");
    (void)pooled_h;
    unsigned int pooled_w = layer->GetParamAsUInt("pooled_w");
    (void)pooled_w;
    float spatial_scale = layer->GetParamAsFloat("spatial_scale");
    if (spatial_scale < 0) {
        THROW_IE_EXCEPTION << "The value of ROIPooling layer spatial_scale parameter is invalid";
    }
}

void ROIPoolingValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1, 2});
}

ROIPoolingValidator::ROIPoolingValidator(const std::string& _type): LayerValidator(_type) {}

void SimplerNMSValidator::checkParams(const CNNLayer* layer) {
    unsigned int post_nms_topn_ = layer->GetParamAsUInt("post_nms_topn");
    (void)post_nms_topn_;

    if (layer->CheckParamPresence("min_bbox_size")) {
        unsigned int min_box_size_ = layer->GetParamAsUInt("min_bbox_size");
        (void)min_box_size_;
    }
    if (layer->CheckParamPresence("feat_stride")) {
        unsigned int feat_stride_ = layer->GetParamAsUInt("feat_stride");
        (void)feat_stride_;
    }
    if (layer->CheckParamPresence("pre_nms_topn")) {
        unsigned int pre_nms_topn_ = layer->GetParamAsUInt("pre_nms_topn");
        (void)pre_nms_topn_;
    }
    if (layer->CheckParamPresence("iou_threshold")) {
        float iou_threshold_ = layer->GetParamAsFloat("iou_threshold");
        if (iou_threshold_ < 0) {
            THROW_IE_EXCEPTION << "The value of SimplerNMS layer iou_threshold_ parameter is invalid";
        }
    }
    if (layer->CheckParamPresence("scale")) {
        std::vector<unsigned int> scale = layer->GetParamAsUInts("scale", {});
        (void)scale;
    }
    if (layer->CheckParamPresence("cls_threshold")) {
        float cls_threshold = layer->GetParamAsFloat("cls_threshold");
        if (cls_threshold < 0) {
            THROW_IE_EXCEPTION << "The value of SimplerNMS layer cls_threshold parameter is invalid";
        }
    }
}

void SimplerNMSValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {3});
}

SimplerNMSValidator::SimplerNMSValidator(const std::string& _type): LayerValidator(_type) {}

void SpatialTransformerValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void SpatialTransformerValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {2});
}

SpatialTransformerValidator::SpatialTransformerValidator(const std::string& _type): LayerValidator(_type) {}

void UpsamplingValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void UpsamplingValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1});
}

UpsamplingValidator::UpsamplingValidator(const std::string& _type): LayerValidator(_type) {}

void UnpoolingValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void UnpoolingValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1});
}

UnpoolingValidator::UnpoolingValidator(const std::string& _type): LayerValidator(_type) {}

ActivationValidator::ActivationValidator(const std::string& _type): LayerValidator(_type) {}

void ActivationValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void ActivationValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1});
}

ConstValidator::ConstValidator(const std::string& _type): LayerValidator(_type) {}

void ConstValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void ConstValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {0, 1});
}

CopyValidator::CopyValidator(const std::string& _type): LayerValidator(_type) {}

void CopyValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void CopyValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1});
}

ELUValidator::ELUValidator(const std::string& _type): LayerValidator(_type) {}

void ELUValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void ELUValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1});
}

InputValidator::InputValidator(const std::string& _type): LayerValidator(_type) {}

void InputValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void InputValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {0});
}

MemoryValidator::MemoryValidator(const std::string& _type): LayerValidator(_type) {}

void MemoryValidator::checkParams(const CNNLayer* layer) {
    int size = layer->GetParamAsInt("size");
    if (size != 2) {
        THROW_IE_EXCEPTION << "The value of Memory layer size parameter is invalid";
    }
}

void MemoryValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1, 0});
}

NormalizeValidator::NormalizeValidator(const std::string& _type): LayerValidator(_type) {}

void NormalizeValidator::checkParams(const CNNLayer* layer) {
    if (layer->CheckParamPresence("eps")) {
        float eps = layer->GetParamAsFloat("eps");
        if (eps < 0) {
            THROW_IE_EXCEPTION << "The value of Normalize layer eps parameter is invalid";
        }
    }
}

SelectValidator::SelectValidator(const std::string& _type): LayerValidator(_type) {}

void SelectValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    enum { CONDITION, THEN, ELSE, numOfInputs };

    size_t numInputs = inShapes.size();
    if (numOfInputs != numInputs) THROW_IE_EXCEPTION << "Select layer with name '" << layer->name << "' take 3 inputs, but actually it has: " << numInputs;

    size_t new_rank = inShapes[ELSE].size();
    new_rank = std::max(new_rank, inShapes[THEN].size());

    if (inShapes[CONDITION].size() > new_rank)
        THROW_IE_EXCEPTION << "Select layer with name '" << layer->name << "' has 'Mask' input's rank more than broadcasted 'Then' and 'Else' inputs' ranks";

    for (size_t i = 0; i < new_rank; i++) {
        auto in1 = i < (new_rank - inShapes[THEN].size()) ? 1 : inShapes[THEN][i - (new_rank - inShapes[THEN].size())];
        auto in2 = i < (new_rank - inShapes[ELSE].size()) ? 1 : inShapes[ELSE][i - (new_rank - inShapes[ELSE].size())];

        size_t tmp = 0;
        if (in1 == in2 || in1 == 1 || in2 == 1)
            tmp = std::max(in1, in2);
        else
            THROW_IE_EXCEPTION << "Select layer with name '" << layer->name << "' has incompatible 'Then' and 'Else' inputs' shapes";

        auto in0 = i < (new_rank - inShapes[CONDITION].size()) ? 1 : inShapes[CONDITION][i - (new_rank - inShapes[CONDITION].size())];

        if (tmp == in0 || in0 == 1)
            tmp = std::max(tmp, in0);
        else
            THROW_IE_EXCEPTION << "Select layer with name '" << layer->name
                                                        << "' has incompatible 'Mask' input's shapes and broadcasted 'Then' and 'Else' inputs' shapes";
    }
}

void NormalizeValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1});
}

PowerFileValidator::PowerFileValidator(const std::string& _type): LayerValidator(_type) {}

void PowerFileValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void PowerFileValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1});
}

ReLU6Validator::ReLU6Validator(const std::string& _type): LayerValidator(_type) {}

void ReLU6Validator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void ReLU6Validator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1});
}

SigmoidValidator::SigmoidValidator(const std::string& _type): LayerValidator(_type) {}

void SigmoidValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void SigmoidValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1});
}

TanHValidator::TanHValidator(const std::string& _type): LayerValidator(_type) {}

void TanHValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1});
}

OneHotValidator::OneHotValidator(const std::string& _type): LayerValidator(_type) {}

void OneHotValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void OneHotValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1});
}

QuantizeValidator::QuantizeValidator(const std::string& _type): LayerValidator(_type) {}

void QuantizeValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void QuantizeValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const QuantizeLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of QuantizeLayer class";
    }

    size_t numInputs = inShapes.size();
    if (numInputs != 5) THROW_IE_EXCEPTION << "Quantize can take only 5 inputs, but actually it has: " << numInputs;

    auto dims0 = inShapes[0];
    if (dims0.size() < 1) {
        THROW_IE_EXCEPTION << "Quantize input0 shape must have at least 1 dimension";
    }
}

BinaryConvolutionValidator::BinaryConvolutionValidator(const std::string& _type): LayerValidator(_type) {}

void BinaryConvolutionValidator::checkParams(const CNNLayer* layer) {
    auto casted = dynamic_cast<const BinaryConvolutionLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of BinaryConvolutionLayer class";
    }
}

void BinaryConvolutionValidator::checkCorrespondence(const CNNLayer* layer,
                                                     const std::map<std::string, Blob::Ptr>& blobs,
                                                     const vector<SizeVector>& inShapes) const {
    auto binConvLayer = dynamic_cast<const BinaryConvolutionLayer*>(layer);
    if (!binConvLayer) THROW_IE_EXCEPTION << "Layer is not instance of BinaryConvolutionLayer class";
}

void BinaryConvolutionValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1});
}

MathValidator::MathValidator(const std::string& _type): LayerValidator(_type) {}

void MathValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {1});
}

ReduceValidator::ReduceValidator(const std::string& _type): LayerValidator(_type) {}

void ReduceValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void ReduceValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    size_t numInputs = inShapes.size();
    if (numInputs > 2)
        THROW_IE_EXCEPTION << layer->name << " Reduce can take up to 2 inputs, but actually it has: " << numInputs;
}

GatherTreeValidator::GatherTreeValidator(const std::string& _type): LayerValidator(_type) {}

void GatherTreeValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void GatherTreeValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {4});
    if (inShapes[0].size() != 3)
        THROW_IE_EXCEPTION << layer->name << " step_idx tensor should be 3 dimension " << inShapes[0].size();
    if (inShapes[1].size() != 3)
        THROW_IE_EXCEPTION << layer->name << " parent_idx tensor should be 3 dimension " << inShapes[1].size();
    if (inShapes[2].size() != 1)
        THROW_IE_EXCEPTION << layer->name << " max_seq_len vector should be 1 dimension " << inShapes[2].size();
    if (inShapes[3].size() != 1)
        THROW_IE_EXCEPTION << layer->name << " end_token vector should be 1 dimension " << inShapes[3].size();
    if (inShapes[0] != inShapes[1] || inShapes[0][1] != inShapes[2][0]) {
        THROW_IE_EXCEPTION << layer->name << " Input tensors dimensions mismatch";
    }
}

TopKValidator::TopKValidator(const std::string& _type): LayerValidator(_type) {}

void TopKValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    size_t numInputs = inShapes.size();
    if (numInputs != 2)
        THROW_IE_EXCEPTION << layer->name << " TopK can take only 2 inputs, but actually it has: " << numInputs;
}

UniqueValidator::UniqueValidator(const std::string& _type): LayerValidator(_type) {}

void UniqueValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    size_t numInputs = inShapes.size();
    if (numInputs != 1)
        THROW_IE_EXCEPTION << layer->name << " Unique can take only 1 input, but actually it has: " << numInputs;
}

NMSValidator::NMSValidator(const std::string& _type): LayerValidator(_type) {}

void NMSValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void NMSValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    size_t numInputs = inShapes.size();
    if (numInputs < 2 || numInputs > 6)
        THROW_IE_EXCEPTION << layer->name
                           << " NonMaxSuppression can take 2 - 6 inputs, but actually it has: " << numInputs;

    if (inShapes[0].size() != 3 || inShapes[0][2] != 4)
        THROW_IE_EXCEPTION << layer->name << " 'boxes' should be with shape [num_batches, spatial_dimension, 4]";

    if (inShapes[1].size() != 3)
        THROW_IE_EXCEPTION << layer->name
                           << " 'scores' should be with shape [num_batches, num_classes, spatial_dimension]";

    if (inShapes[0][0] != inShapes[1][0])
        THROW_IE_EXCEPTION << layer->name << " num_batches is different in 'boxes' and 'scores' tensors";

    if (inShapes[0][1] != inShapes[1][2])
        THROW_IE_EXCEPTION << layer->name << " spatial_dimension is different in 'boxes' and 'scores' tensors";

    if (numInputs > 2 && inShapes[2].size() && inShapes[2][0] != 1)
        THROW_IE_EXCEPTION << layer->name << " 'max_output_boxes_per_class' should be scalar";

    if (numInputs > 3 && inShapes[3].size() && inShapes[3][0] != 1)
        THROW_IE_EXCEPTION << layer->name << " 'iou_threshold' should be scalar";

    if (numInputs > 4 && inShapes[4].size() && inShapes[4][0] != 1)
        THROW_IE_EXCEPTION << layer->name << " 'score_threshold' should be scalar";
}

ScatterUpdateValidator::ScatterUpdateValidator(const std::string& _type): LayerValidator(_type) {}

void ScatterUpdateValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const ScatterUpdateLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of ScatterUpdateLayer class";
    }

    size_t numInputs = inShapes.size();
    if (numInputs != 4)
        THROW_IE_EXCEPTION << layer->name << " Scatter can take only 4 inputs, but actually it has: " << numInputs;

    static constexpr int DATA = 0;
    static constexpr int INDICES = 1;
    static constexpr int UPDATES = 2;
    static constexpr int AXIS = 3;

    if (inShapes[DATA].size() < 1)
        THROW_IE_EXCEPTION << layer->name << " 'Data' tensor rank must be >= 1";

    if (inShapes[INDICES].size() < 1)
        THROW_IE_EXCEPTION << layer->name << " 'Indices' tensor rank must be >= 1";

    if (inShapes[UPDATES].size() < 1)
        THROW_IE_EXCEPTION << layer->name << " 'Updates' tensor rank must be >= 1";

    if (!(inShapes[AXIS].size() == 0 ||
         (inShapes[AXIS].size() == 1 && inShapes[AXIS][0] == 1)))
        THROW_IE_EXCEPTION << layer->name << " 'Axis' tensor must be scalar, or 1D array of 1 element";

    if (inShapes[UPDATES].size() != inShapes[INDICES].size() + inShapes[DATA].size() - 1)
        THROW_IE_EXCEPTION << layer->name << " Incorrect number of 'indexes' and 'updates' tensors dimension";

    Precision inIdxPrecision = layer->insData[INDICES].lock()->getTensorDesc().getPrecision();
    if (inIdxPrecision != Precision::I32 && inIdxPrecision != Precision::I64)
        THROW_IE_EXCEPTION << layer->name << " Incorrect input 'Indices' precision. Only I32 or I64 are supported!";

    Precision inAxisPrecision = layer->insData[AXIS].lock()->getTensorDesc().getPrecision();
    if (inAxisPrecision != Precision::I32 && inAxisPrecision != Precision::I64)
        THROW_IE_EXCEPTION << layer->name << " Incorrect input 'Axis' precision. Only I32 or I64 are supported!";

    if (layer->insData[DATA].lock()->getTensorDesc().getPrecision() !=
        layer->insData[UPDATES].lock()->getTensorDesc().getPrecision())
        THROW_IE_EXCEPTION << layer->name << " Precision should be equal for input tensors 'Data' and 'Updates'";
}

ScatterElementsUpdateValidator::ScatterElementsUpdateValidator(const std::string& _type): LayerValidator(_type) {}

void ScatterElementsUpdateValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const ScatterElementsUpdateLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of ScatterElementsUpdateLayer class";
    }

    size_t numInputs = inShapes.size();
    if (numInputs != 4)
        THROW_IE_EXCEPTION << layer->name << " Scatter can take only 4 inputs, but actually it has: " << numInputs;

    static constexpr int DATA = 0;
    static constexpr int INDICES = 1;
    static constexpr int UPDATES = 2;
    static constexpr int AXIS = 3;

    if (inShapes[DATA].size() < 1)
        THROW_IE_EXCEPTION << layer->name << " 'Data' tensor rank must be >= 1";

    if (inShapes[INDICES].size() < 1)
        THROW_IE_EXCEPTION << layer->name << " 'Indices' tensor rank must be >= 1";

    if (inShapes[UPDATES].size() < 1)
        THROW_IE_EXCEPTION << layer->name << " 'Updates' tensor rank must be >= 1";

    if (!(inShapes[AXIS].size() == 0 ||
         (inShapes[AXIS].size() == 1 && inShapes[AXIS][0] == 1)))
        THROW_IE_EXCEPTION << layer->name << " 'Axis' tensor must be scalar, or 1D array of 1 element";

    if (inShapes[INDICES].size() != inShapes[DATA].size())
        THROW_IE_EXCEPTION << layer->name << " Incorrect number of 'indexes' tensors dimension";

    if (inShapes[UPDATES].size() != inShapes[DATA].size())
        THROW_IE_EXCEPTION << layer->name << " Incorrect number of 'updates' tensors dimension";

    Precision inIdxPrecision = layer->insData[INDICES].lock()->getTensorDesc().getPrecision();
    if (inIdxPrecision != Precision::I32 && inIdxPrecision != Precision::I64)
        THROW_IE_EXCEPTION << layer->name << " Incorrect input 'Indices' precision. Only I32 or I64 are supported!";

    Precision inAxisPrecision = layer->insData[AXIS].lock()->getTensorDesc().getPrecision();
    if (inAxisPrecision != Precision::I32 && inIdxPrecision != Precision::I64)
        THROW_IE_EXCEPTION << layer->name << " Incorrect input 'Axis' precision. Only I32 or I64 are supported!";

    if (layer->insData[DATA].lock()->getTensorDesc().getPrecision() !=
        layer->insData[UPDATES].lock()->getTensorDesc().getPrecision())
        THROW_IE_EXCEPTION << layer->name << " Precision should be equal for input tensors 'Data' and 'Updates'";
}

#define REG_LAYER_VALIDATOR_FOR_TYPE(__validator, __type) _validators[#__type] = std::make_shared<__validator>(#__type)

LayerValidators::LayerValidators() : _validators() {
    REG_LAYER_VALIDATOR_FOR_TYPE(ActivationValidator, Activation);
    REG_LAYER_VALIDATOR_FOR_TYPE(ArgMaxValidator, ArgMax);
    REG_LAYER_VALIDATOR_FOR_TYPE(BatchNormalizationValidator, BatchNormalization);
    REG_LAYER_VALIDATOR_FOR_TYPE(CTCGreedyDecoderValidator, CTCGreedyDecoder);
    REG_LAYER_VALIDATOR_FOR_TYPE(ClampValidator, Clamp);
    REG_LAYER_VALIDATOR_FOR_TYPE(ConcatValidator, Concat);
    REG_LAYER_VALIDATOR_FOR_TYPE(ConstValidator, Const);
    REG_LAYER_VALIDATOR_FOR_TYPE(ConvolutionValidator, Convolution);
    REG_LAYER_VALIDATOR_FOR_TYPE(CopyValidator, Copy);
    REG_LAYER_VALIDATOR_FOR_TYPE(CropValidator, Crop);
    REG_LAYER_VALIDATOR_FOR_TYPE(DeconvolutionValidator, Deconvolution);
    REG_LAYER_VALIDATOR_FOR_TYPE(DeformableConvolutionValidator, DeformableConvolution);
    REG_LAYER_VALIDATOR_FOR_TYPE(DetectionOutputValidator, DetectionOutput);
    REG_LAYER_VALIDATOR_FOR_TYPE(ELUValidator, ELU);
    REG_LAYER_VALIDATOR_FOR_TYPE(EltwiseValidator, Eltwise);
    REG_LAYER_VALIDATOR_FOR_TYPE(FullyConnectedValidator, InnerProduct);
    REG_LAYER_VALIDATOR_FOR_TYPE(FullyConnectedValidator, FullyConnected);
    REG_LAYER_VALIDATOR_FOR_TYPE(GRNValidator, GRN);
    REG_LAYER_VALIDATOR_FOR_TYPE(InputValidator, Input);
    REG_LAYER_VALIDATOR_FOR_TYPE(InterpValidator, Interp);
    REG_LAYER_VALIDATOR_FOR_TYPE(MVNValidator, MVN);
    REG_LAYER_VALIDATOR_FOR_TYPE(MemoryValidator, Memory);
    REG_LAYER_VALIDATOR_FOR_TYPE(NormValidator, Norm);
    REG_LAYER_VALIDATOR_FOR_TYPE(NormValidator, LRN);
    REG_LAYER_VALIDATOR_FOR_TYPE(NormalizeValidator, Normalize);
    REG_LAYER_VALIDATOR_FOR_TYPE(PReLUValidator, PReLU);
    REG_LAYER_VALIDATOR_FOR_TYPE(PSROIPoolingValidator, PSROIPooling);
    REG_LAYER_VALIDATOR_FOR_TYPE(PermuteValidator, Permute);
    REG_LAYER_VALIDATOR_FOR_TYPE(PoolingValidator, Pooling);
    REG_LAYER_VALIDATOR_FOR_TYPE(PowerValidator, Power);
    REG_LAYER_VALIDATOR_FOR_TYPE(PowerFileValidator, PowerFile);
    REG_LAYER_VALIDATOR_FOR_TYPE(PriorBoxClusteredValidator, PriorBoxClustered);
    REG_LAYER_VALIDATOR_FOR_TYPE(PriorBoxValidator, PriorBox);
    REG_LAYER_VALIDATOR_FOR_TYPE(ProposalValidator, Proposal);
    REG_LAYER_VALIDATOR_FOR_TYPE(ROIPoolingValidator, ROIPooling);
    REG_LAYER_VALIDATOR_FOR_TYPE(ReLUValidator, ReLU);
    REG_LAYER_VALIDATOR_FOR_TYPE(ReLU6Validator, ReLU6);
    REG_LAYER_VALIDATOR_FOR_TYPE(RegionYoloValidator, RegionYolo);
    REG_LAYER_VALIDATOR_FOR_TYPE(ReorgYoloValidator, ReorgYolo);
    REG_LAYER_VALIDATOR_FOR_TYPE(ResampleValidator, Resample);
    REG_LAYER_VALIDATOR_FOR_TYPE(ReshapeValidator, Reshape);
    REG_LAYER_VALIDATOR_FOR_TYPE(ReshapeValidator, Flatten);
    REG_LAYER_VALIDATOR_FOR_TYPE(ScaleShiftValidator, ScaleShift);
    REG_LAYER_VALIDATOR_FOR_TYPE(SigmoidValidator, Sigmoid);
    REG_LAYER_VALIDATOR_FOR_TYPE(SigmoidValidator, Logistic);
    REG_LAYER_VALIDATOR_FOR_TYPE(SimplerNMSValidator, SimplerNMS);
    REG_LAYER_VALIDATOR_FOR_TYPE(SoftMaxValidator, SoftMax);
    REG_LAYER_VALIDATOR_FOR_TYPE(SpatialTransformerValidator, SpatialTransformer);
    REG_LAYER_VALIDATOR_FOR_TYPE(SplitValidator, Split);
    REG_LAYER_VALIDATOR_FOR_TYPE(SplitValidator, Slice);
    REG_LAYER_VALIDATOR_FOR_TYPE(GemmValidator, Gemm);
    REG_LAYER_VALIDATOR_FOR_TYPE(PadValidator, Pad);
    REG_LAYER_VALIDATOR_FOR_TYPE(GatherValidator, Gather);
    REG_LAYER_VALIDATOR_FOR_TYPE(StridedSliceValidator, StridedSlice);
    REG_LAYER_VALIDATOR_FOR_TYPE(ShuffleChannelsValidator, ShuffleChannels);
    REG_LAYER_VALIDATOR_FOR_TYPE(DepthToSpaceValidator, DepthToSpace);
    REG_LAYER_VALIDATOR_FOR_TYPE(SpaceToDepthValidator, SpaceToDepth);
    REG_LAYER_VALIDATOR_FOR_TYPE(SpaceToBatchValidator, SpaceToBatch);
    REG_LAYER_VALIDATOR_FOR_TYPE(BatchToSpaceValidator, BatchToSpace);
    REG_LAYER_VALIDATOR_FOR_TYPE(SparseFillEmptyRowsValidator, SparseFillEmptyRows);
    REG_LAYER_VALIDATOR_FOR_TYPE(SparseSegmentReduceValidator, SparseSegmentMean);
    REG_LAYER_VALIDATOR_FOR_TYPE(SparseSegmentReduceValidator, SparseSegmentSqrtN);
    REG_LAYER_VALIDATOR_FOR_TYPE(SparseSegmentReduceValidator, SparseSegmentSum);
    REG_LAYER_VALIDATOR_FOR_TYPE(ReverseSequenceValidator, ReverseSequence);
    REG_LAYER_VALIDATOR_FOR_TYPE(RNNCellValidator<RNNSequenceLayer::RNN>, RNNCell);
    REG_LAYER_VALIDATOR_FOR_TYPE(RNNCellValidator<RNNSequenceLayer::GRU>, GRUCell);
    REG_LAYER_VALIDATOR_FOR_TYPE(RNNCellValidator<RNNSequenceLayer::LSTM>, LSTMCell);
    REG_LAYER_VALIDATOR_FOR_TYPE(RNNSequenceValidator<RNNSequenceLayer::RNN>, RNNSequence);
    REG_LAYER_VALIDATOR_FOR_TYPE(RNNSequenceValidator<RNNSequenceLayer::GRU>, GRUSequence);
    REG_LAYER_VALIDATOR_FOR_TYPE(RNNSequenceValidator<RNNSequenceLayer::LSTM>, LSTMSequence);
    REG_LAYER_VALIDATOR_FOR_TYPE(SelectValidator, Select);
    REG_LAYER_VALIDATOR_FOR_TYPE(SqueezeValidator, Squeeze);
    REG_LAYER_VALIDATOR_FOR_TYPE(UnsqueezeValidator, Unsqueeze);
    REG_LAYER_VALIDATOR_FOR_TYPE(RangeValidator, Range);
    REG_LAYER_VALIDATOR_FOR_TYPE(FillValidator, Fill);
    REG_LAYER_VALIDATOR_FOR_TYPE(BroadcastValidator, Broadcast);
    REG_LAYER_VALIDATOR_FOR_TYPE(TanHValidator, TanH);
    REG_LAYER_VALIDATOR_FOR_TYPE(TileValidator, Tile);
    REG_LAYER_VALIDATOR_FOR_TYPE(UnpoolingValidator, Unpooling);
    REG_LAYER_VALIDATOR_FOR_TYPE(UpsamplingValidator, Upsampling);
    REG_LAYER_VALIDATOR_FOR_TYPE(OneHotValidator, OneHot);
    REG_LAYER_VALIDATOR_FOR_TYPE(QuantizeValidator, FakeQuantize);
    REG_LAYER_VALIDATOR_FOR_TYPE(BinaryConvolutionValidator, BinaryConvolution);
    REG_LAYER_VALIDATOR_FOR_TYPE(MathValidator, Abs);
    REG_LAYER_VALIDATOR_FOR_TYPE(MathValidator, Acos);
    REG_LAYER_VALIDATOR_FOR_TYPE(MathValidator, Acosh);
    REG_LAYER_VALIDATOR_FOR_TYPE(MathValidator, Asin);
    REG_LAYER_VALIDATOR_FOR_TYPE(MathValidator, Asinh);
    REG_LAYER_VALIDATOR_FOR_TYPE(MathValidator, Atan);
    REG_LAYER_VALIDATOR_FOR_TYPE(MathValidator, Atanh);
    REG_LAYER_VALIDATOR_FOR_TYPE(MathValidator, Ceil);
    REG_LAYER_VALIDATOR_FOR_TYPE(MathValidator, Cos);
    REG_LAYER_VALIDATOR_FOR_TYPE(MathValidator, Cosh);
    REG_LAYER_VALIDATOR_FOR_TYPE(MathValidator, Erf);
    REG_LAYER_VALIDATOR_FOR_TYPE(MathValidator, Floor);
    REG_LAYER_VALIDATOR_FOR_TYPE(MathValidator, HardSigmoid);
    REG_LAYER_VALIDATOR_FOR_TYPE(MathValidator, Log);
    REG_LAYER_VALIDATOR_FOR_TYPE(MathValidator, Neg);
    REG_LAYER_VALIDATOR_FOR_TYPE(MathValidator, Reciprocal);
    REG_LAYER_VALIDATOR_FOR_TYPE(MathValidator, Selu);
    REG_LAYER_VALIDATOR_FOR_TYPE(MathValidator, Sign);
    REG_LAYER_VALIDATOR_FOR_TYPE(MathValidator, Sin);
    REG_LAYER_VALIDATOR_FOR_TYPE(MathValidator, Sinh);
    REG_LAYER_VALIDATOR_FOR_TYPE(MathValidator, Softplus);
    REG_LAYER_VALIDATOR_FOR_TYPE(MathValidator, Softsign);
    REG_LAYER_VALIDATOR_FOR_TYPE(MathValidator, Tan);
    REG_LAYER_VALIDATOR_FOR_TYPE(ReduceValidator, ReduceAnd);
    REG_LAYER_VALIDATOR_FOR_TYPE(ReduceValidator, ReduceL1);
    REG_LAYER_VALIDATOR_FOR_TYPE(ReduceValidator, ReduceL2);
    REG_LAYER_VALIDATOR_FOR_TYPE(ReduceValidator, ReduceLogSum);
    REG_LAYER_VALIDATOR_FOR_TYPE(ReduceValidator, ReduceLogSumExp);
    REG_LAYER_VALIDATOR_FOR_TYPE(ReduceValidator, ReduceMax);
    REG_LAYER_VALIDATOR_FOR_TYPE(ReduceValidator, ReduceMean);
    REG_LAYER_VALIDATOR_FOR_TYPE(ReduceValidator, ReduceMin);
    REG_LAYER_VALIDATOR_FOR_TYPE(ReduceValidator, ReduceOr);
    REG_LAYER_VALIDATOR_FOR_TYPE(ReduceValidator, ReduceProd);
    REG_LAYER_VALIDATOR_FOR_TYPE(ReduceValidator, ReduceSum);
    REG_LAYER_VALIDATOR_FOR_TYPE(ReduceValidator, ReduceSumSquare);
    REG_LAYER_VALIDATOR_FOR_TYPE(GatherTreeValidator, GatherTree);
    REG_LAYER_VALIDATOR_FOR_TYPE(TopKValidator, TopK);
    REG_LAYER_VALIDATOR_FOR_TYPE(UniqueValidator, Unique);
    REG_LAYER_VALIDATOR_FOR_TYPE(NMSValidator, NonMaxSuppression);
    REG_LAYER_VALIDATOR_FOR_TYPE(ScatterUpdateValidator, ScatterUpdate);
    REG_LAYER_VALIDATOR_FOR_TYPE(ScatterElementsUpdateValidator, ScatterElementsUpdate);
}

}  // namespace InferenceEngine
