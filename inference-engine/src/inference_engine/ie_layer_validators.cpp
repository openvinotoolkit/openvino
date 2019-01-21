// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_layers.h"
#include "ie_layers_prv.h"
#include "ie_layer_validators.hpp"
#include "debug.h"
#include "xml_parse_utils.h"
#include <memory>
#include <string>
#include <map>
#include <vector>
#include <ie_iextension.h>
#include <ie_format_parser.h>

#include <details/ie_exception.hpp>

namespace InferenceEngine {

using namespace details;
using std::vector;

template <typename T, typename P>
inline bool one_of(T val, P item) { return val == item; }
template <typename T, typename P, typename... Args>
inline bool one_of(T val, P item, Args... item_others) {
    return val == item || one_of(val, item_others...);
}

void CNNLayer::validateLayer() {
    try {
        LayerValidator::Ptr validator = LayerValidators::getInstance()->getValidator(type);
        validator->parseParams(this);
        validator->checkParams(this);
        InOutDims shapes;
        getInOutShapes(this, shapes);
        validator->checkShapes(this, shapes.inDims);
    } catch(InferenceEngineException ie_e) {
        THROW_IE_EXCEPTION << "Error of validate layer: " << this->name
                           << " with type: " << this->type << ". "
                           << ie_e.what();
    }
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
                     const vector<SizeVector>& inShapes, WeightableParams params,
                     const SizeVector& numDims) {
    if (inShapes.size() != 1)
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
    if (it != blobs.end()) {  // TODO: return with fixing shape infer tests: THROW_IE_EXCEPTION << "Invalid blobs: no weights";
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
                                                  const vector<SizeVector>& inShapes) const {
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
        auto getArray = [](std::string param, vector<int>& array) {
            std::istringstream stream(param);
            std::string str;
            while (getline(stream, str, ',')) {
                int val = std::stoi(str);
                array.push_back(val);
            }
        };
        getArray(layer->GetParamAsString("axis"), casted->axis);
        if (casted->params.find("offset") != casted->params.end()) {
            getArray(layer->GetParamAsString("offset"), casted->offset);
        }
        if (casted->params.find("dim") != casted->params.end()) {
            getArray(layer->GetParamAsString("dim"), casted->dim);
        }
        if (casted->params.find("crop_begin") != casted->params.end()) {
            getArray(layer->GetParamAsString("crop_begin"), casted->offset);
        }
    }
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

CropValidator::CropValidator(const std::string& _type) : LayerValidator(_type) {}

void CropValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const CropLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of CropLayer class";
    }
    size_t numInputs = inShapes.size();
    if (numInputs != 1 && numInputs != 2) {
        THROW_IE_EXCEPTION << "Crop can take only 1 or 2 inputs, but actually it has: " << numInputs;
    }
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
            if (casted->params.find("crop_begin") != casted->params.end()) {
                THROW_IE_EXCEPTION
                        << "Incorrect format of the Crop layer: `crop_begin` and `crop_end` attributes are valid for single input only";
            }
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
        } else if (!casted->dim.empty()) {
            int dim = casted->dim[i];
            if (firstShape[axis] < static_cast<size_t>(offset + dim)) {
                THROW_IE_EXCEPTION << "Incorrect crop data! Offset(" << offset << ") + result size of output("
                                   << dim << ") should be less then input size(" << firstShape[axis]
                                   << ") for axis(" << axis << ")";
            }
        }
    }
}

ConvolutionValidator::ConvolutionValidator(const std::string& _type) : LayerValidator(_type) {}

void ConvolutionValidator::parseParams(CNNLayer* layer) {
    auto convLayer = dynamic_cast<ConvolutionLayer*>(layer);
    if (!convLayer) {
        THROW_IE_EXCEPTION << "Layer is not instance of ConvolutionLayer class";
    }
    convLayer->_out_depth = convLayer->GetParamAsUInt("output");

    convLayer->_kernel.clear();
    convLayer->_stride.clear();
    convLayer->_padding.clear();
    convLayer->_pads_end.clear();
    convLayer->_dilation.clear();

    vector<unsigned int> kernels = convLayer->GetParamAsUInts("kernel", {});
    if (kernels.empty()) {
        // IR_v == 2
        convLayer->_kernel.insert(X_AXIS, convLayer->GetParamAsUInt("kernel-x"));
        convLayer->_kernel.insert(Y_AXIS, convLayer->GetParamAsUInt("kernel-y"));

        convLayer->_stride.insert(X_AXIS, convLayer->GetParamAsUInt("stride-x", 1u));
        convLayer->_stride.insert(Y_AXIS, convLayer->GetParamAsUInt("stride-y", 1u));
        // TODO: maybe just throw exception, why do we change IR?
        if (0 == convLayer->_stride[X_AXIS]) {
            convLayer->_stride[X_AXIS] = 1u;
            LogError("Warning! in layer %s: Stride x is 0, setting to 1 ", convLayer->name.c_str());
        }
        if (0 == convLayer->_stride[Y_AXIS]) {
            convLayer->_stride[Y_AXIS] = 1u;
            LogError("Warning! in layer %s: Stride y is 0, setting to 1", convLayer->name.c_str());
        }

        convLayer->_padding.insert(X_AXIS, convLayer->GetParamAsUInt("pad-x", 0u));
        convLayer->_padding.insert(Y_AXIS, convLayer->GetParamAsUInt("pad-y", 0u));

        convLayer->_pads_end.insert(X_AXIS, convLayer->GetParamAsUInt("pad-r", convLayer->_padding[X_AXIS]));
        convLayer->_pads_end.insert(Y_AXIS, convLayer->GetParamAsUInt("pad-b", convLayer->_padding[Y_AXIS]));

        convLayer->_dilation.insert(X_AXIS, convLayer->GetParamAsUInt("dilation-x", 1u));
        convLayer->_dilation.insert(Y_AXIS, convLayer->GetParamAsUInt("dilation-y", 1u));
    } else {
        // IR_v > 2
        for (int i = 1; i <= kernels.size(); i++) {
            convLayer->_kernel.insert(i - 1, kernels[kernels.size() - i]);
        }

        vector<unsigned int> default_0 = vector<unsigned int> (convLayer->_kernel.size(), 0u);
        vector<unsigned int> default_1 = vector<unsigned int> (convLayer->_kernel.size(), 1u);

        vector<unsigned int> strides = convLayer->GetParamAsUInts("strides", default_1);
        for (int i = 1; i <= strides.size(); i++) {
            if (strides[strides.size() - i] == 0) {
                THROW_IE_EXCEPTION << "Stride could not be 0.\nIn layer " << convLayer->name;
            }
            convLayer->_stride.insert(i - 1, strides[strides.size() - i]);
        }

        vector<unsigned int> pads_begin = convLayer->GetParamAsUInts("pads_begin", default_0);
        for (int i = 1; i <= pads_begin.size(); i++) {
            convLayer->_padding.insert(i - 1, pads_begin[pads_begin.size() - i]);
        }

        vector<unsigned int> pads_end = convLayer->GetParamAsUInts("pads_end", pads_begin);
        for (int i = 1; i <= pads_end.size(); i++) {
            convLayer->_pads_end.insert(i - 1, pads_end[pads_end.size() - i]);
        }

        vector<unsigned int> dilations = convLayer->GetParamAsUInts("dilations", default_1);
        for (int i = 1; i <= dilations.size(); i++) {
            convLayer->_dilation.insert(i - 1, dilations[dilations.size() - i]);
        }
    }

    convLayer->_auto_pad = convLayer->GetParamAsString("auto_pad", "");
    convLayer->_group = convLayer->GetParamAsUInt("group", 1u);
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
                                               const vector<SizeVector>& inShapes) const {
    auto convLayer = dynamic_cast<const ConvolutionLayer*>(layer);
    if (!convLayer) THROW_IE_EXCEPTION << "Layer is not instance of ConvolutionLayer class";
    auto version = BaseCreator::version_;
    if (version < 3) {
        checkWeightable(blobs, inShapes, {convLayer->_out_depth, false, convLayer->_group, convLayer->_kernel[Y_AXIS], convLayer->_kernel[X_AXIS]},
                        {4});
    } else if (version == 3) {
        // TODO: implement v2 convolution valitation
    }
}

void DeconvolutionValidator::parseParams(CNNLayer* layer) {
    auto deconvLayer = dynamic_cast<DeconvolutionLayer*>(layer);
    if (!deconvLayer) {
        THROW_IE_EXCEPTION << "Layer is not instance of DeconvolutionLayer class";
    }
    ConvolutionValidator::parseParams(layer);
}

void DeconvolutionValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

DeconvolutionValidator::DeconvolutionValidator(const std::string& _type) : ConvolutionValidator(_type) {}

void DeconvolutionValidator::checkCorrespondence(const CNNLayer* layer,
                                                 const std::map<std::string, Blob::Ptr>& blobs,
                                                 const vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const DeconvolutionLayer*>(layer);
    if (!casted) THROW_IE_EXCEPTION << "Layer is not instance of ConvolutionLayer class";
    checkWeightable(blobs, inShapes, {casted->_out_depth, false, casted->_group, casted->_kernel[Y_AXIS], casted->_kernel[X_AXIS]},
                    {4});
}

PoolingValidator::PoolingValidator(const std::string& _type) : LayerValidator(_type) {}

void PoolingValidator::parseParams(CNNLayer* layer) {
    auto poolLayer = dynamic_cast<PoolingLayer*>(layer);
    if (!poolLayer) {
        THROW_IE_EXCEPTION << "Layer is not instance of PoolingLayer class";
    }

    poolLayer->_kernel.clear();
    poolLayer->_stride.clear();
    poolLayer->_padding.clear();
    poolLayer->_pads_end.clear();

    poolLayer->_auto_pad = poolLayer->GetParamAsString("auto_pad", "");

    vector<unsigned int> kernels = poolLayer->GetParamAsUInts("kernel", {});
    if (kernels.empty()) {
        int kernel_x = poolLayer->GetParamAsInt("kernel-x", -1);
        /** Pooling as custom layer */
        if (kernel_x == -1) {
            try {
                unsigned int kernel_size = poolLayer->GetParamAsUInt("kernel_size");
                unsigned int kernel_w = poolLayer->GetParamAsUInt("kernel_w", 0u);
                unsigned int kernel_h = poolLayer->GetParamAsUInt("kernel_h", 0u);
                poolLayer->_kernel.insert(X_AXIS, kernel_w == 0u ? kernel_size : kernel_w);
                poolLayer->_kernel.insert(Y_AXIS, kernel_h == 0u ? kernel_size : kernel_h);

                unsigned int stride = poolLayer->GetParamAsUInt("stride", 1u);
                unsigned int stride_w = poolLayer->GetParamAsUInt("stride_w", 0u);
                unsigned int stride_h = poolLayer->GetParamAsUInt("stride_h", 0u);
                poolLayer->_stride.insert(X_AXIS, stride_w == 0u ? stride : stride_w);
                poolLayer->_stride.insert(Y_AXIS, stride_h == 0u ? stride : stride_h);

                unsigned int pad = poolLayer->GetParamAsUInt("pad", 0u);
                unsigned int pad_w = poolLayer->GetParamAsUInt("pad_w", 0u);
                unsigned int pad_h = poolLayer->GetParamAsUInt("pad_h", 0u);

                poolLayer->_padding.insert(X_AXIS, pad_w == 0u ? pad : pad_w);
                poolLayer->_padding.insert(Y_AXIS, pad_h == 0u ? pad : pad_h);

                poolLayer->_pads_end.insert(X_AXIS, 0u);
                poolLayer->_pads_end.insert(Y_AXIS, 0u);
            } catch (...) {
            }

            std::string alg = poolLayer->GetParamAsString("pool", "caffe.PoolingParameter.MAX");
            poolLayer->_type = alg == "caffe.PoolingParameter.MAX" ? PoolingLayer::MAX : PoolingLayer::AVG;
        } else  /** Default behavior */ {
            poolLayer->_kernel.insert(X_AXIS, poolLayer->GetParamAsUInt("kernel-x"));
            poolLayer->_kernel.insert(Y_AXIS, poolLayer->GetParamAsUInt("kernel-y"));

            poolLayer->_stride.insert(X_AXIS, poolLayer->GetParamAsUInt("stride-x", 1u));
            poolLayer->_stride.insert(Y_AXIS, poolLayer->GetParamAsUInt("stride-y", 1u));
            // TODO: maybe just throw exception, why do we change IR?
            if (0 == poolLayer->_stride[X_AXIS]) {
                poolLayer->_stride[X_AXIS] = 1u;
                LogError("Warning! in layer %s: Stride x is 0, setting to 1 ", poolLayer->name.c_str());
            }
            if (0 == poolLayer->_stride[Y_AXIS]) {
                poolLayer->_stride[Y_AXIS] = 1u;
                LogError("Warning! in layer %s: Stride y is 0, setting to 1", poolLayer->name.c_str());
            }

            poolLayer->_padding.insert(X_AXIS, poolLayer->GetParamAsUInt("pad-x", 0u));
            poolLayer->_padding.insert(Y_AXIS, poolLayer->GetParamAsUInt("pad-y", 0u));

            poolLayer->_pads_end.insert(X_AXIS, poolLayer->GetParamAsUInt("pad-r", poolLayer->_padding[X_AXIS]));
            poolLayer->_pads_end.insert(Y_AXIS, poolLayer->GetParamAsUInt("pad-b", poolLayer->_padding[Y_AXIS]));

            // TODO: All kind of pool methods
            poolLayer->_exclude_pad = poolLayer->GetParamsAsBool("exclude-pad", false);
            std::string alg = poolLayer->GetParamAsString("pool-method", "max");
            poolLayer->_type = alg == "avg" ? PoolingLayer::AVG : PoolingLayer::MAX;
            if (alg != "max" && alg != "avg") {
                THROW_IE_EXCEPTION << "Layer with type `" << _type << "` has incorrect pool-type!";
            }
        }
    } else {
        for (int i = 1; i <= kernels.size(); i++) {
            poolLayer->_kernel.insert(i - 1, kernels[kernels.size() - i]);
        }

        vector<unsigned int> default_0 = vector<unsigned int> (poolLayer->_kernel.size(), 0u);
        vector<unsigned int> default_1 = vector<unsigned int> (poolLayer->_kernel.size(), 1u);

        vector<unsigned int> strides = poolLayer->GetParamAsUInts("strides", default_1);
        for (int i = 1; i <= strides.size(); i++) {
            if (strides[strides.size() - i] == 0) {
                THROW_IE_EXCEPTION << "Stride could not be 0.\nIn layer " << poolLayer->name;
            }
            poolLayer->_stride.insert(i - 1, strides[strides.size() - i]);
        }

        vector<unsigned int> pads_begin = poolLayer->GetParamAsUInts("pads_begin", default_0);
        for (int i = 1; i <= pads_begin.size(); i++) {
            poolLayer->_padding.insert(i - 1, pads_begin[pads_begin.size() - i]);
        }

        vector<unsigned int> pads_end = poolLayer->GetParamAsUInts("pads_end", pads_begin);
        for (int i = 1; i <= pads_end.size(); i++) {
            poolLayer->_pads_end.insert(i - 1, pads_end[pads_end.size() - i]);
        }

        poolLayer->_exclude_pad = poolLayer->GetParamsAsBool("exclude-pad", false);
        std::string alg = poolLayer->GetParamAsString("pool-method", "max");
        poolLayer->_type = alg == "avg" ? PoolingLayer::AVG : PoolingLayer::MAX;
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

void ReshapeValidator::parseParams(CNNLayer *layer) {
    auto casted = dynamic_cast<ReshapeLayer *>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of ReshapeLayer class";
    }
    casted->shape.clear();
    if (!casted->params.empty()) {
        if (casted->type == "Flatten") {
            casted->num_axes = casted->GetParamAsInt("end_axis", -1);
            casted->axis = casted->axis = casted->GetParamAsInt("axis", 0);
        } else {
            casted->shape = casted->GetParamAsInts("dim", {});
        }
    }
}

void ReshapeValidator::checkParams(const CNNLayer *layer) {
    auto casted = dynamic_cast<const ReshapeLayer *>(layer);
    if (!casted)
        THROW_IE_EXCEPTION << "Layer is not instance of ReshapeLayer class";
    size_t num = 0;
    for (int dim : casted->shape) {
        if (dim < -1)
            THROW_IE_EXCEPTION << "Invalid value of Reshape mask (dim attribute):" << dim
                               << ". Supported values: 0, -1, >0";
        if (dim == -1) num++;
    }
    if (num > 1) THROW_IE_EXCEPTION << "Invalid Reshape mask (dim attribute): at most one dimension can be `-1`";
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

    auto getArray = [](std::string param, vector<float>& array) {
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
    casted->_isAcrossMaps = CaselessEq<std::string>()(casted->GetParamAsString("region"), "across");
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
        if (static_cast<int>(i->getTensorDesc().getDims().size()) <= casted->_axis) {
            THROW_IE_EXCEPTION << "Internal error - dimensions are emtpy";
        }
        out_sizes += std::to_string(i->getTensorDesc().getDims()[casted->_axis]);
    }
    if (!out_sizes.empty())
        casted->params["out_sizes"] = out_sizes;
}

void checkNumOfInput(const std::vector<SizeVector>& inShapes, const vector<int>& expected_num_of_shapes) {
    bool shape_was_found = false;
    for (const auto& i : expected_num_of_shapes) {
        if (inShapes.size() == i) {
            shape_was_found = true;
        }
    }
    if (!shape_was_found) {
        THROW_IE_EXCEPTION << "Number of inputs (" << inShapes.size() << ") is not equal to expected ones";
    }
}


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

void ConcatValidator::checkShapes(const CNNLayer* layer,
                                  const std::vector<SizeVector>& inShapes) const {
    if (inShapes.empty())
        THROW_IE_EXCEPTION << "Inputs are empty";

    auto casted = dynamic_cast<const ConcatLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Invalid Concat layer.";
    }

    auto firstShape = inShapes[0];
    size_t firstShapeSize = firstShape.size();
    size_t axis = casted->_axis;
    if (axis >= firstShapeSize)
        THROW_IE_EXCEPTION << "Concat axis(" << axis
                           << ") should be less the number of current input dimensions ("
                           << firstShapeSize << ")";

    for (size_t i = 1; i < inShapes.size(); i++) {
        auto shape = inShapes[i];
        if (shape.size() != firstShapeSize)
            THROW_IE_EXCEPTION << "Invalid inputs for Concat layer: number of dimensions must match: "
                               << firstShapeSize << " vs " << shape.size();
        bool eq_part1 = std::equal(firstShape.begin(), firstShape.begin() + axis,
                                   shape.begin());
        bool eq_part2 = std::equal(firstShape.begin() + axis + 1, firstShape.end(),
                                   shape.begin() + axis + 1);
        if (!(eq_part1 && eq_part2))
            THROW_IE_EXCEPTION << "Invalid inputs for Concat layer: dimensions should match in all"
                               << "positions except axis (" << axis << ") one"
                               << ") should match : [" << dumpVec(firstShape) << "] vs ["
                               << dumpVec(shape) <<"]";
    }
}

GemmValidator::GemmValidator(const std::string& _type) : LayerValidator(_type) {}

void GemmValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<GemmLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of GemmLayer class";
    }
    casted->alpha = casted->GetParamAsFloat("alpha", 1);
    casted->beta = casted->GetParamAsFloat("beta", 1);
    casted->transpose_a = casted->GetParamsAsBool("transpose_a", false);
    casted->transpose_b = casted->GetParamsAsBool("transpose_b", false);
}

void GemmValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void GemmValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const GemmLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of GemmLayer class";
    }

    size_t numInputs = inShapes.size();
    if (numInputs != 2 && numInputs != 3)
        THROW_IE_EXCEPTION << "Gemm can take only 2 or 3 inputs, but actually it has: " << numInputs;

    auto dims0 = inShapes[0];
    auto dims1 = inShapes[1];
    if (dims0.size() < 2 || dims1.size() < 2) {
        THROW_IE_EXCEPTION << "Gemm input shapes must have at least 2 dimensions";
    }

    unsigned long xAxis = dims0.size() - 1;
    unsigned long yAxis = dims0.size() - 2;
    if (dims0[xAxis] != dims1[yAxis])
        THROW_IE_EXCEPTION << "Gemm input0 x dimension must be equal to input1 y dimension ("
                           << dims0[xAxis] << " vs " << dims1[yAxis] << ")";

    if (inShapes.size() == 3) {
        auto dims2 = inShapes[2];
        if (dims2.size() < 2) {
            THROW_IE_EXCEPTION << "Gemm input shapes must have at least 2 dimensions";
        }

        if (dims2[xAxis] != dims1[xAxis])
            THROW_IE_EXCEPTION << "Gemm input2 x dimension must be equal to input1 x dimension ("
                               << dims2[xAxis] << " vs " << dims1[xAxis] << ")";

        if (dims2[yAxis] != dims0[yAxis])
            THROW_IE_EXCEPTION << "Gemm input2 y dimension must be equal to input0 y dimension ("
                               << dims2[yAxis] << " vs " << dims0[yAxis] << ")";
    }
}

PadValidator::PadValidator(const std::string& _type) : LayerValidator(_type) {}

void PadValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<PadLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of PadLayer class";
    }
    std::vector<uint32_t> pads_begin = casted->GetParamAsUInts("pads_begin");
    std::vector<uint32_t> pads_end = casted->GetParamAsUInts("pads_end");

    casted->pads_begin.clear();
    for (size_t i = 0; i < pads_begin.size(); i++) {
        casted->pads_begin.insert(i, pads_begin[i]);
    }

    casted->pads_end.clear();
    for (size_t i = 0; i < pads_end.size(); i++) {
        casted->pads_end.insert(i, pads_end[i]);
    }

    casted->pad_value = casted->GetParamAsFloat("pad_value", 0.0f);

    std::string mode = casted->GetParamAsString("pad_mode", "constant");
    if (mode == "constant") {
        casted->pad_mode = PadLayer::Constant;
    } else if (mode == "edge") {
        casted->pad_mode = PadLayer::Edge;
    } else if (mode == "reflect") {
        casted->pad_mode = PadLayer::Reflect;
    } else if (mode == "symmetric") {
        casted->pad_mode = PadLayer::Symmetric;
    } else {
        THROW_IE_EXCEPTION << "Unsupported pad mode operation: " << mode;
    }
}

void PadValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void PadValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const PadLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of PadLayer class";
    }

    size_t numInputs = inShapes.size();
    if (numInputs != 1)
        THROW_IE_EXCEPTION << "Pad can take only 1 input, but actually it has: " << numInputs;

    if (inShapes[0].size() != casted->pads_begin.size())
        THROW_IE_EXCEPTION << "Dimensions count mismatch in layer " << layer->name
                           << ". Expected: " << casted->pads_begin.size() << " Got: " << inShapes[0].size();

    if (inShapes[0].size() != casted->pads_end.size())
        THROW_IE_EXCEPTION << "Dimensions count mismatch in layer " << layer->name
                           << ". Expected: " << casted->pads_end.size() << " Got: " << inShapes[0].size();

    if (casted->pad_mode == PadLayer::Symmetric || casted->pad_mode == PadLayer::Reflect) {
        for (size_t i = 0; i < inShapes[0].size(); i++) {
            if (inShapes[0][i] < casted->pads_begin[i]) {
                THROW_IE_EXCEPTION << "Pad can't be grater than input shape in symmetric and reflect modes."
                                   << " For dimension " << i << " pad_begin=" << casted->pads_begin[i]
                                   << " in_shape="<< inShapes[0][i];
            }
            if (inShapes[0][i] < casted->pads_end[i]) {
                THROW_IE_EXCEPTION << "Pad can't be grater than input shape in symmetric and reflect modes."
                                   << " For dimension " << i << " pad_end=" << casted->pads_end[i]
                                   << " in_shape="<< inShapes[0][i];
            }
        }
    }
}

GatherValidator::GatherValidator(const std::string& _type) : LayerValidator(_type) {}

void GatherValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<GatherLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of GatherLayer class";
    }

    casted->axis = casted->GetParamAsInt("axis", 0);
}

void GatherValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void GatherValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const GatherLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of GatherLayer class";
    }

    size_t numInputs = inShapes.size();
    if (numInputs != 2)
        THROW_IE_EXCEPTION << "Gather can take only 2 inputs, but actually it has: " << numInputs;

    if (casted->axis > 0 && (inShapes[0].size() - casted->axis) < 1)
        THROW_IE_EXCEPTION << "Incorrect input dictionary dimensions " << inShapes[0].size()
                           << " and axis number " << casted->axis;
    else if (casted->axis < 0 && (static_cast<int>(inShapes[0].size()) + casted->axis) < 0)
        THROW_IE_EXCEPTION << "Incorrect input dictionary dimensions " << inShapes[0].size()
                           << " and axis number " << casted->axis;
}

RNNValidator::RNNValidator(const std::string& _type) : LayerValidator(_type) {}

void RNNValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<RNNLayer*>(layer);
    if (!casted)
        THROW_IE_EXCEPTION << "Layer is not instance of RNNLayer class";

    std::string cell = layer->GetParamAsString("cell_type");
    std::string direction = layer->GetParamAsString("direction", "Forward");
    int axis = layer->GetParamAsInt("axis", 1);

    if (!one_of(cell, "LSTM", "RNN", "GRU"))
        THROW_IE_EXCEPTION << "Unknown RNN cell type " << cell << ". "
                           << "Expected one of [ LSTM | RNN | GRU ].";

    if (!one_of(direction, "Forward", "Backward", "Bidirectional"))
        THROW_IE_EXCEPTION << "Unknown RNN direction type " << direction << ". "
                           << "Expected one of [ Forward | Backward | Bidirectional ].";

    casted->axis = axis;
    casted->cellType = cell;
    casted->direction = direction == "Forward"  ? RNNLayer::RNN_FWD :
                        direction == "Backward" ? RNNLayer::RNN_BWD :
                                                  RNNLayer::RNN_BDR;
}

void RNNValidator::checkParams(const InferenceEngine::CNNLayer *layer) {
    auto casted = dynamic_cast<const RNNLayer*>(layer);
    if (!casted)
        THROW_IE_EXCEPTION << "Layer is not instance of RNNLayer class";

    if (!one_of(casted->axis, 1, 0))
        THROW_IE_EXCEPTION << "Unsupported axis for RNN layer iterator. Only 0 and 1 axis are supported.";

    // TODO: Add more RNN verification..
}

void RNNValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {}

}  // namespace InferenceEngine
