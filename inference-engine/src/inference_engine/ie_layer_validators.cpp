// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_layers.h"
#include "ie_layer_validators.hpp"
#include "debug.h"
#include "xml_parse_utils.h"
#include <memory>
#include <string>
#include <map>
#include <vector>
#include <ie_iextension.h>

using namespace InferenceEngine;
using namespace details;

void CNNLayer::validateLayer() {
    LayerValidator::Ptr validator = LayerValidators::getInstance()->getValidator(type);
    validator->parseParams(this);
    validator->checkParams(this);
    InOutDims shapes;
    getInOutShapes(this, shapes);
    validator->checkShapes(this, shapes.inDims);
}

struct WeightableParams {
    size_t kernel_w, kernel_h, outputs, groups;
    bool isKernelFromInput;

    WeightableParams(size_t _outputs, bool _isKernelFromInput, size_t _groups = 0, size_t _kernel_h = 0,
                     size_t _kernel_w = 0) : outputs(_outputs), isKernelFromInput(_isKernelFromInput),
                                             kernel_h(_kernel_h), kernel_w(_kernel_w),
                                             groups(_groups) {}
};

void checkWeightable(const std::map<std::string, Blob::Ptr>& blobs,
                     const std::vector<SizeVector>& inShapes, WeightableParams params,
                     const SizeVector& numDims) {
    if (inShapes.size() != 1)
        THROW_IE_EXCEPTION << "Number of inputs (" << inShapes.size() << ") is not equal to expected ones (1)";
    SizeVector firstInputShape = inShapes[0];
    size_t inputSize = firstInputShape.size();

    bool isOK = false;
    for (auto dim : numDims) {
        if (inputSize == dim) isOK = true;
    }
    if (!isOK) {
        return THROW_IE_EXCEPTION << "Input shape " << details::dumpVec(firstInputShape)
                                  << " has unexpected size, supported sizes: " << details::dumpVec(numDims);
    }

    if (firstInputShape.empty()) THROW_IE_EXCEPTION << "Input shape can't be empty";

    size_t KW = 1, KH = 1, IC, OC;
    IC = firstInputShape[1];
    if (params.isKernelFromInput) {
        if (firstInputShape.size() == 4) {
            KH = firstInputShape[2];
            KW = firstInputShape[3];
        }
    } else {
        KH = params.kernel_h;
        KW = params.kernel_w;
    }
    OC = params.outputs;

    auto it = blobs.find("weights");
    if (it !=
        blobs.end()) {  // TODO: return with fixing shape infer tests: THROW_IE_EXCEPTION << "Invalid blobs: no weights";
        auto weights = it->second;
        if (weights == nullptr || weights->dims().empty()) THROW_IE_EXCEPTION << "Weights can't be empty";

        auto weightsSize = details::product(weights->dims());
        size_t expectedWeightsSize = OC * KW * KH * IC;
        if (params.groups) expectedWeightsSize /= params.groups;
        if (expectedWeightsSize != weightsSize) {
            THROW_IE_EXCEPTION << "New shapes " << details::dumpVec(firstInputShape) << " make Kernels(" << KH << "x"
                               << KW << "), Channels(" << IC << "), Output depth(" << OC << "), Groups("
                               << params.groups << ") not matching weights size: " << weightsSize;
        }
    }

    it = blobs.find("biases");
    if (it != blobs.end()) {
        auto biases = it->second;
        if (biases == nullptr || biases->dims().empty()) THROW_IE_EXCEPTION << "Biases can't be empty";
        auto biasesSize = details::product(biases->dims());
        if (OC != biasesSize) {
            THROW_IE_EXCEPTION << "Number of outputs (" << OC << ") don't match biases size: " << biasesSize;
        }
    }
}

LayerValidators* LayerValidators::getInstance() {
    if (!_instance) {
        _instance = new LayerValidators();
    }
    return _instance;
}

LayerValidator::Ptr LayerValidators::getValidator(const std::string& type) {
    if (_validators.find(type) == _validators.end()) {
        return std::make_shared<GeneralValidator>(type);
    }
    return _validators[type];
}

void LayerValidators::addImpl(const std::string& type, const LayerValidator::Ptr& validator) {
    _validators[type] = validator;
}

LayerValidators* LayerValidators::_instance = nullptr;

GeneralValidator::GeneralValidator(const std::string& _type) : LayerValidator(_type) {}

void FullyConnectedValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<FullyConnectedLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of FullyConnectedLayer class";
    }
    casted->_out_num = casted->GetParamAsUInt("out-size");
}

void FullyConnectedValidator::checkParams(const CNNLayer* layer) {
    // TODO: check that values belong to the scope of the definition according to spec
}

void FullyConnectedValidator::checkCorrespondence(const CNNLayer* layer,
                                                  const std::map<std::string, Blob::Ptr>& blobs,
                                                  const std::vector<SizeVector>& inShapes) const {
    const auto casted = dynamic_cast<const FullyConnectedLayer*>(layer);
    if (!casted) THROW_IE_EXCEPTION << "Layer is not instance of FullyConnectedLayer class";
    checkWeightable(blobs, inShapes, {casted->_out_num, true, 1}, {4, 2});
}

FullyConnectedValidator::FullyConnectedValidator(const std::string& _type) : LayerValidator(_type) {}

void CropValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<CropLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of CropLayer class";
    }
    if (casted->axis.empty()) {
        auto getArray = [](std::string param, std::vector<int>& array) {
            std::istringstream stream(param);
            std::string str;
            while (getline(stream, str, ',')) {
                int val = std::stoi(str);
                array.push_back(val);
            }
        };
        getArray(layer->GetParamAsString("axis"), casted->axis);
        getArray(layer->GetParamAsString("offset"), casted->offset);
    }
}

void CropValidator::checkParams(const CNNLayer* layer) {
    auto casted = dynamic_cast<const CropLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of CropLayer class";
    }
    if (casted->axis.size() != casted->offset.size()) {
        THROW_IE_EXCEPTION << "Incorrect format of the Crop layer.";
    }
}

CropValidator::CropValidator(const std::string& _type) : LayerValidator(_type) {}

void CropValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const CropLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of CropLayer class";
    }
    size_t numInputs = inShapes.size();
    if (numInputs != 1 && numInputs != 2)
        THROW_IE_EXCEPTION << "Crop can take only 1 or 2 inputs, but actually it has: " << numInputs;
    auto firstShape = inShapes[0];
    size_t shapeSize = firstShape.size();
    for (size_t i = 0; i < casted->axis.size(); i++) {
        int axis = casted->axis[i];
        int offset = casted->offset[i];
        if (shapeSize <= axis)
            THROW_IE_EXCEPTION << "Crop axis(" << casted->axis[i]
                               << ") should be less the number of dimensions of first input ("
                               << firstShape.size() << ")";
        if (numInputs == 2) {
            auto secondShape = inShapes[1];
            if (secondShape.size() <= axis)
                THROW_IE_EXCEPTION << "Crop axis(" << axis
                                   << ") should be less the number of dimensions of second input ("
                                   << secondShape.size() << ")";
            size_t newSize = secondShape[axis];
            if (firstShape[axis] < static_cast<size_t>(offset + newSize)) {
                THROW_IE_EXCEPTION << "Incorrect crop data! Offset(" << offset << ") + result size of output("
                                   << newSize << ") should be less then input size(" << firstShape[axis]
                                   << ") for axis(" << axis << ")";
            }
        }
    }
}

ConvolutionValidator::ConvolutionValidator(const std::string& _type) : LayerValidator(_type) {}

void ConvolutionValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<ConvolutionLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of ConvolutionLayer class";
    }
    casted->_out_depth = casted->GetParamAsUInt("output");
    casted->_kernel_x = casted->GetParamAsUInt("kernel-x");
    casted->_kernel_y = casted->GetParamAsUInt("kernel-y");
    casted->_stride_x = casted->GetParamAsUInt("stride-x", 1);
    casted->_stride_y = casted->GetParamAsUInt("stride-y", 1);
    casted->_padding_x = casted->GetParamAsUInt("pad-x", 0);
    casted->_padding_y = casted->GetParamAsUInt("pad-y", 0);
    casted->_dilation_x = casted->GetParamAsUInt("dilation-x", 1);
    casted->_dilation_y = casted->GetParamAsUInt("dilation-y", 1);
    casted->_group = casted->GetParamAsUInt("group", 1);
    // TODO: checks for presence of all required attributes, and that there's no extraneous parameters only.

    // TODO: maybe just throw exception, why do we change IR?
    if (0 == casted->_stride_x) {
        casted->_stride_x = 1;
        LogError("Warning! in layer %s: Stride x is 0, setting to 1 ", casted->name.c_str());
    }
    if (0 == casted->_stride_y) {
        casted->_stride_y = 1;
        LogError("Warning! in layer %s: Stride y is 0, setting to 1", casted->name.c_str());
    }
}

void ConvolutionValidator::checkParams(const CNNLayer* layer) {
    auto casted = dynamic_cast<const ConvolutionLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of ConvolutionLayer class";
    }
    // TODO: check that values belong to the scope of the definition according to spec
}

void ConvolutionValidator::checkCorrespondence(const CNNLayer* layer,
                                               const std::map<std::string, Blob::Ptr>& blobs,
                                               const std::vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const ConvolutionLayer*>(layer);
    if (!casted) THROW_IE_EXCEPTION << "Layer is not instance of ConvolutionLayer class";
    checkWeightable(blobs, inShapes, {casted->_out_depth, false, casted->_group, casted->_kernel_y, casted->_kernel_x},
                    {4});
}

void DeconvolutionValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<DeconvolutionLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of DeconvolutionLayer class";
    }
    casted->_out_depth = casted->GetParamAsUInt("output");
    casted->_kernel_x = casted->GetParamAsUInt("kernel-x");
    casted->_kernel_y = casted->GetParamAsUInt("kernel-y");
    casted->_stride_x = casted->GetParamAsUInt("stride-x", 1);
    casted->_stride_y = casted->GetParamAsUInt("stride-y", 1);
    casted->_padding_x = casted->GetParamAsUInt("pad-x", 0);
    casted->_padding_y = casted->GetParamAsUInt("pad-y", 0);
    casted->_dilation_x = casted->GetParamAsUInt("dilation-x", 1);
    casted->_dilation_y = casted->GetParamAsUInt("dilation-y", 1);
    casted->_group = casted->GetParamAsUInt("group", 1);

    if (0 == casted->_stride_x) {
        casted->_stride_x = 1;
        LogError("Warning! in layer %s: Stride x is 0, setting to 1 ", casted->name.c_str());
    }
    if (0 == casted->_stride_y) {
        casted->_stride_y = 1;
        LogError("Warning! in layer %s: Stride y is 0, setting to 1", casted->name.c_str());
    }
}

void DeconvolutionValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

DeconvolutionValidator::DeconvolutionValidator(const std::string& _type) : LayerValidator(_type) {}


void DeconvolutionValidator::checkCorrespondence(const CNNLayer* layer,
                                                 const std::map<std::string, Blob::Ptr>& blobs,
                                                 const std::vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const DeconvolutionLayer*>(layer);
    if (!casted) THROW_IE_EXCEPTION << "Layer is not instance of ConvolutionLayer class";
    checkWeightable(blobs, inShapes, {casted->_out_depth, false, casted->_group, casted->_kernel_y, casted->_kernel_x},
                    {4});
}

PoolingValidator::PoolingValidator(const std::string& _type) : LayerValidator(_type) {}

void PoolingValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<PoolingLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of PoolingLayer class";
    }
    int kernel_x = casted->GetParamAsInt("kernel-x", -1);
    /** Pooling as custom layer */
    if (kernel_x == -1) {
        unsigned int kernel_size = casted->GetParamAsUInt("kernel_size");
        unsigned int kernel_w = casted->GetParamAsUInt("kernel_w", 0);
        unsigned int kernel_h = casted->GetParamAsUInt("kernel_h", 0);
        casted->_kernel_x = kernel_w == 0 ? kernel_size : kernel_w;
        casted->_kernel_y = kernel_h == 0 ? kernel_size : kernel_h;

        unsigned int stride = casted->GetParamAsUInt("stride", 1);
        unsigned int stride_w = casted->GetParamAsUInt("stride_w", 0);
        unsigned int stride_h = casted->GetParamAsUInt("stride_h", 0);
        casted->_stride_x = stride_w == 0 ? stride : stride_w;
        casted->_stride_y = stride_h == 0 ? stride : stride_h;

        unsigned int pad = casted->GetParamAsUInt("pad", 0);
        unsigned int pad_w = casted->GetParamAsUInt("pad_w", 0);
        unsigned int pad_h = casted->GetParamAsUInt("pad_h", 0);
        casted->_padding_x = pad_w == 0 ? pad : pad_w;
        casted->_padding_y = pad_h == 0 ? pad : pad_h;

        std::string alg = casted->GetParamAsString("pool", "caffe.PoolingParameter.MAX");
        casted->_type = alg == "caffe.PoolingParameter.MAX" ? PoolingLayer::MAX : PoolingLayer::AVG;
    } else  /** Default behaviour */ {
        casted->_kernel_x = casted->GetParamAsUInt("kernel-x");
        casted->_kernel_y = casted->GetParamAsUInt("kernel-y");
        casted->_stride_x = casted->GetParamAsUInt("stride-x", 1);
        casted->_stride_y = casted->GetParamAsUInt("stride-y", 1);
        casted->_padding_x = casted->GetParamAsUInt("pad-x", 0);
        casted->_padding_y = casted->GetParamAsUInt("pad-y", 0);

        // TODO: All kind of pool methods
        casted->_exclude_pad = casted->GetParamsAsBool("exclude-pad", false);
        std::string alg = casted->GetParamAsString("pool-method", "max");
        casted->_type = alg == "avg" ? PoolingLayer::AVG : PoolingLayer::MAX;
        if (alg != "max" && alg != "avg") {
            THROW_IE_EXCEPTION << "Layer with type `" << _type << "` has incorrect pad-type!";
        }
    }
    // TODO: checks for presence of all required attributes, and that there's no extraneous parameters only.
}

void PoolingValidator::checkParams(const CNNLayer* layer) {
    // TODO: check that values belong to the scope of the definition according to spec
}

void BatchNormalizationValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<BatchNormalizationLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of BatchNormalizationLayer class";
    }
    casted->epsilon = casted->GetParamAsFloat("epsilon");
}

void BatchNormalizationValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

BatchNormalizationValidator::BatchNormalizationValidator(const std::string& _type) : LayerValidator(_type) {}

void PowerValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<PowerLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of PowerLayer class";
    }
    casted->offset = casted->GetParamAsFloat("shift");
    casted->power = casted->GetParamAsFloat("power");
    casted->scale = casted->GetParamAsFloat("scale");
}

void PowerValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

PowerValidator::PowerValidator(const std::string& _type) : LayerValidator(_type) {}

void PReLUValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<PReLULayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of PReLULayer class";
    }
    casted->_channel_shared = casted->GetParamsAsBool("channel_shared", false);
}

void PReLUValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

PReLUValidator::PReLUValidator(const std::string& _type) : LayerValidator(_type) {}

void ScaleShiftValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<ScaleShiftLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of ScaleShiftLayer class";
    }
    if (!casted->params.empty()) {
        casted->_broadcast = casted->GetParamAsUInt("broadcast", 2);
    }
}

void ScaleShiftValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

ScaleShiftValidator::ScaleShiftValidator(const std::string& _type) : LayerValidator(_type) {}

void TileValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<TileLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of TileLayer class";
    }
    casted->axis = casted->GetParamAsInt("axis", -1);
    casted->tiles = casted->GetParamAsInt("tiles", -1);
}

void TileValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

TileValidator::TileValidator(const std::string& _type) : LayerValidator(_type) {}

ReshapeValidator::ReshapeValidator(const std::string& _type) : LayerValidator(_type) {}

void ReshapeValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<ReshapeLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of ReshapeLayer class";
    }
    try {
        if (!casted->params.empty()) {
            casted->num_axes = casted->GetParamAsInt(casted->type == "Flatten" ? "end_axis" : "num_axes", -1);
            casted->axis = casted->GetParamAsInt("axis", 1);
            casted->shape = casted->GetParamAsInts("dim", {});
            calculateIn2Out(casted);
        }
    } catch (...) {}
}

void ReshapeValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void ReshapeValidator::calculateIn2Out(ReshapeLayer* layer) {
    if (layer->outData.empty() || layer->insData.empty())
        return;

    if (!layer->shape.empty() && std::find(layer->shape.begin(), layer->shape.end(), 0) != layer->shape.end())
        return;

    SizeVector inDims = layer->input()->getTensorDesc().getDims();
    SizeVector outDims = layer->outData[0]->getTensorDesc().getDims();

    std::vector<size_t> inMapped;
    std::vector<size_t> outMapped;
    for (size_t i = 0; i < inDims.size(); i++) {
        bool mapped = false;
        inMapped.push_back(i);
        for (size_t j = 0; !mapped && j < outDims.size(); j++) {
            if (outDims[j] == inDims[i] && std::find(outMapped.begin(), outMapped.end(), j) == outMapped.end()) {
                outMapped.push_back(j);
                mapped = true;
            }
        }

        for (size_t j = 1; !mapped && j <= outDims.size(); j++) {
            if (outDims[outDims.size() - j] != inDims[i] && (outDims[outDims.size() - j] % inDims[i] == 0)) {
                outMapped.push_back(outDims.size() - j);
                mapped = true;
            }
        }
        if (!mapped) {
            size_t outIndex = outDims.size() - 1;
            for (size_t k = 0; k < layer->shape.size(); k++) {
                if (layer->shape[k] < 0) {
                    outIndex = k;
                    break;
                }
            }
            outMapped.push_back(outIndex);
        }
    }
    std::string mapped_params;
    for (size_t i = 0; i < inMapped.size(); i++) {
        if (!mapped_params.empty())
            mapped_params += ",";
        mapped_params += std::to_string(inMapped[i]) + "-" + std::to_string(outMapped[i]);
    }

    layer->params["in2out"] = mapped_params;
}

void EltwiseValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<EltwiseLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of EltwiseLayer class";
    }
    // TODO: fix this onece we switched to IR v2.x also enable dedicated unit tests
    // @details: need to remove sum
    std::string op = casted->GetParamAsString("operation", "sum");
    // TODO: remove empty value case in IRv2.x
    if (op == "sum" || op == "") {
        casted->_operation = EltwiseLayer::Sum;
    } else if (op == "mul" || op == "prod") {
        casted->_operation = EltwiseLayer::Prod;
    } else if (op == "max") {
        casted->_operation = EltwiseLayer::Max;
    } else {
        THROW_IE_EXCEPTION << "Unsupported element wise operation: " << op;
    }

    auto getArray = [](std::string param, std::vector<float>& array) {
        std::istringstream stream(param);
        std::string str;
        while (getline(stream, str, ',')) {
            float val = std::stof(str);
            array.push_back(val);
        }
    };
    getArray(casted->GetParamAsString("coeff", ""), casted->coeff);
}

void EltwiseValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

EltwiseValidator::EltwiseValidator(const std::string& _type) : LayerValidator(_type) {}

void ClampValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<ClampLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of ClampLayer class";
    }
    casted->min_value = casted->GetParamAsFloat("min");
    casted->max_value = casted->GetParamAsFloat("max");
}

void ClampValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

ClampValidator::ClampValidator(const std::string& _type) : LayerValidator(_type) {}

void ReLUValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<ReLULayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of ReLULayer class";
    }
    if (!casted->params.empty()) {
        casted->negative_slope = casted->GetParamAsFloat("negative_slope");
    }
}

void ReLUValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

ReLUValidator::ReLUValidator(const std::string& _type) : LayerValidator(_type) {}

void MVNValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<MVNLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of MVNLayer class";
    }
    casted->across_channels = casted->GetParamAsInt("across_channels", 0);
    casted->normalize = casted->GetParamAsInt("normalize_variance", 1);
}

void MVNValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

MVNValidator::MVNValidator(const std::string& _type) : LayerValidator(_type) {}

void GRNValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<GRNLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of GRNLayer class";
    }
    casted->bias = casted->GetParamAsFloat("bias", 0.f);
}

void GRNValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

GRNValidator::GRNValidator(const std::string& _type) : LayerValidator(_type) {}

void SoftMaxValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<SoftMaxLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of SoftMaxLayer class";
    }
    casted->axis = casted->GetParamAsInt("axis", 1);
}

void SoftMaxValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

SoftMaxValidator::SoftMaxValidator(const std::string& _type) : LayerValidator(_type) {}

void NormValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<NormLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of NormLayer class";
    }
    casted->_size = casted->GetParamAsUInt("local_size", 0);
    casted->_size += casted->GetParamAsUInt("local-size", 0);
    casted->_k = casted->GetParamAsUInt("k", 1);
    casted->_alpha = casted->GetParamAsFloat("alpha");
    casted->_beta = casted->GetParamAsFloat("beta");
    casted->_isAcrossMaps = casted->GetParamsAsBool("region", false);
}

void NormValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

NormValidator::NormValidator(const std::string& _type) : LayerValidator(_type) {}

SplitValidator::SplitValidator(const std::string& _type) : LayerValidator(_type) {}

void SplitValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<SplitLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of SplitLayer class";
    }
    casted->_axis = casted->GetParamAsUInt("axis", 1);

    std::string out_sizes;
    for (auto& i : layer->outData) {
        if (!out_sizes.empty())
            out_sizes += ",";
        out_sizes += std::to_string(i->getTensorDesc().getDims()[casted->_axis]);
    }
    if (!out_sizes.empty())
        casted->params["out_sizes"] = out_sizes;
}

void SplitValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

ConcatValidator::ConcatValidator(const std::string& _type) : LayerValidator(_type) {}

void ConcatValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<ConcatLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of ConcatLayer class";
    }
    casted->_axis = casted->GetParamAsUInt("axis", 1);
}

void ConcatValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}
