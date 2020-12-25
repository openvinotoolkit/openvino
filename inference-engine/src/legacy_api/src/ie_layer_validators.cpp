// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ie_iextension.h>
#include "debug.h"

#include <legacy/ie_layers.h>
#include "ie_layer_validators.hpp"

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

void CNNLayer::parseParams() {
    try {
        LayerValidator::Ptr validator = LayerValidators::getInstance()->getValidator(type);
        validator->parseParams(this);
    } catch (const InferenceEngineException& ie_e) {
        THROW_IE_EXCEPTION << "Error of validate layer: " << this->name << " with type: " << this->type << ". "
                           << ie_e.what();
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

//

void FullyConnectedValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<FullyConnectedLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of FullyConnectedLayer class";
    }
    casted->_out_num = casted->GetParamAsUInt("out-size");
}

FullyConnectedValidator::FullyConnectedValidator(const std::string& _type): LayerValidator(_type) {}

//

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

CropValidator::CropValidator(const std::string& _type): LayerValidator(_type) {}

//

ConvolutionValidator::ConvolutionValidator(const std::string& _type): LayerValidator(_type) {}

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
        }
        if (0 == convLayer->_stride[Y_AXIS]) {
            convLayer->_stride[Y_AXIS] = 1u;
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

        vector<unsigned int> default_0 = vector<unsigned int>(convLayer->_kernel.size(), 0u);
        vector<unsigned int> default_1 = vector<unsigned int>(convLayer->_kernel.size(), 1u);

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

//

void DeconvolutionValidator::parseParams(CNNLayer* layer) {
    auto deconvLayer = dynamic_cast<DeconvolutionLayer*>(layer);
    if (!deconvLayer) {
        THROW_IE_EXCEPTION << "Layer is not instance of DeconvolutionLayer class";
    }
    ConvolutionValidator::parseParams(layer);
}

DeconvolutionValidator::DeconvolutionValidator(const std::string& _type): ConvolutionValidator(_type) {}

//

DeformableConvolutionValidator::DeformableConvolutionValidator(const std::string& _type): ConvolutionValidator(_type) {}

void DeformableConvolutionValidator::parseParams(CNNLayer* layer) {
    auto deformable_conv_layer = dynamic_cast<DeformableConvolutionLayer*>(layer);
    if (!deformable_conv_layer) {
        THROW_IE_EXCEPTION << "Layer is not instance of DeformableConvolutionLayer class";
    }
    deformable_conv_layer->_deformable_group = deformable_conv_layer->GetParamAsUInt("deformable_group", 1u);
    ConvolutionValidator::parseParams(layer);
}

//

PoolingValidator::PoolingValidator(const std::string& _type): LayerValidator(_type) {}

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
        } else /** Default behavior */ {
            poolLayer->_kernel.insert(X_AXIS, poolLayer->GetParamAsUInt("kernel-x"));
            poolLayer->_kernel.insert(Y_AXIS, poolLayer->GetParamAsUInt("kernel-y"));

            poolLayer->_stride.insert(X_AXIS, poolLayer->GetParamAsUInt("stride-x", 1u));
            poolLayer->_stride.insert(Y_AXIS, poolLayer->GetParamAsUInt("stride-y", 1u));
            // TODO: maybe just throw exception, why do we change IR?
            if (0 == poolLayer->_stride[X_AXIS]) {
                poolLayer->_stride[X_AXIS] = 1u;
            }
            if (0 == poolLayer->_stride[Y_AXIS]) {
                poolLayer->_stride[Y_AXIS] = 1u;
            }

            poolLayer->_padding.insert(X_AXIS, poolLayer->GetParamAsUInt("pad-x", 0u));
            poolLayer->_padding.insert(Y_AXIS, poolLayer->GetParamAsUInt("pad-y", 0u));

            poolLayer->_pads_end.insert(X_AXIS, poolLayer->GetParamAsUInt("pad-r", poolLayer->_padding[X_AXIS]));
            poolLayer->_pads_end.insert(Y_AXIS, poolLayer->GetParamAsUInt("pad-b", poolLayer->_padding[Y_AXIS]));

            // TODO: All kind of pool methods
            poolLayer->_exclude_pad = poolLayer->GetParamAsBool("exclude-pad", false);
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

        vector<unsigned int> default_0 = vector<unsigned int>(poolLayer->_kernel.size(), 0u);
        vector<unsigned int> default_1 = vector<unsigned int>(poolLayer->_kernel.size(), 1u);

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

        poolLayer->_exclude_pad = poolLayer->GetParamAsBool("exclude-pad", false);
        std::string alg = poolLayer->GetParamAsString("pool-method", "max");
        poolLayer->_type = alg == "avg" ? PoolingLayer::AVG : PoolingLayer::MAX;
        if (alg != "max" && alg != "avg") {
            THROW_IE_EXCEPTION << "Layer with type `" << _type << "` has incorrect pad-type!";
        }
    }
    // TODO: checks for presence of all required attributes, and that there's no extraneous parameters only.
}

//

void BatchNormalizationValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<BatchNormalizationLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of BatchNormalizationLayer class";
    }
    casted->epsilon = casted->GetParamAsFloat("epsilon");
}

BatchNormalizationValidator::BatchNormalizationValidator(const std::string& _type): LayerValidator(_type) {}

//

void PowerValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<PowerLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of PowerLayer class";
    }
    casted->offset = casted->GetParamAsFloat("shift");
    casted->power = casted->GetParamAsFloat("power");
    casted->scale = casted->GetParamAsFloat("scale");
}

PowerValidator::PowerValidator(const std::string& _type): LayerValidator(_type) {}

//

void PReLUValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<PReLULayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of PReLULayer class";
    }
    casted->_channel_shared = casted->GetParamAsBool("channel_shared", false);
}

PReLUValidator::PReLUValidator(const std::string& _type): LayerValidator(_type) {}

//

void ScaleShiftValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<ScaleShiftLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of ScaleShiftLayer class";
    }
    if (casted->params.count("broadcast")) {
        casted->_broadcast = casted->GetParamAsUInt("broadcast", 2);
    }
}

ScaleShiftValidator::ScaleShiftValidator(const std::string& _type): LayerValidator(_type) {}

//

void TileValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<TileLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of TileLayer class";
    }
    casted->axis = casted->GetParamAsInt("axis", -1);
    casted->tiles = casted->GetParamAsInt("tiles", -1);
}

TileValidator::TileValidator(const std::string& _type): LayerValidator(_type) {}

//

ReshapeValidator::ReshapeValidator(const std::string& _type): LayerValidator(_type) {}

void ReshapeValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<ReshapeLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of ReshapeLayer class";
    }
    casted->shape.clear();
    if (casted->type == "Flatten" && casted->params.count("end_axis") && casted->params.count("axis")) {
        casted->num_axes = casted->GetParamAsInt("end_axis", -1);
        casted->axis = casted->GetParamAsInt("axis", 0);
    } else if (casted->params.count("dim")) {
        casted->shape = casted->GetParamAsInts("dim", {});
    }
}

//

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
    } else if (op == "sub") {
        casted->_operation = EltwiseLayer::Sub;
    } else if (op == "div") {
        casted->_operation = EltwiseLayer::Div;
    } else if (op == "min") {
        casted->_operation = EltwiseLayer::Min;
    } else if (op == "squared_diff") {
        casted->_operation = EltwiseLayer::Squared_diff;
    } else if (op == "equal") {
        casted->_operation = EltwiseLayer::Equal;
    } else if (op == "not_equal") {
        casted->_operation = EltwiseLayer::Not_equal;
    } else if (op == "less") {
        casted->_operation = EltwiseLayer::Less;
    } else if (op == "less_equal") {
        casted->_operation = EltwiseLayer::Less_equal;
    } else if (op == "greater") {
        casted->_operation = EltwiseLayer::Greater;
    } else if (op == "greater_equal") {
        casted->_operation = EltwiseLayer::Greater_equal;
    } else if (op == "logical_not") {
        casted->_operation = EltwiseLayer::Logical_NOT;
    } else if (op == "logical_and") {
        casted->_operation = EltwiseLayer::Logical_AND;
    } else if (op == "logical_or") {
        casted->_operation = EltwiseLayer::Logical_OR;
    } else if (op == "logical_xor") {
        casted->_operation = EltwiseLayer::Logical_XOR;
    } else if (op == "floor_mod") {
        casted->_operation = EltwiseLayer::Floor_mod;
    } else if (op == "pow") {
        casted->_operation = EltwiseLayer::Pow;
    } else if (op == "mean") {
        casted->_operation = EltwiseLayer::Mean;
    } else {
        THROW_IE_EXCEPTION << "Unsupported element wise operation: " << op;
    }

    casted->coeff = casted->GetParamAsFloats("coeff", {});
}

EltwiseValidator::EltwiseValidator(const std::string& _type): LayerValidator(_type) {}

//

void ClampValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<ClampLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of ClampLayer class";
    }
    casted->min_value = casted->GetParamAsFloat("min");
    casted->max_value = casted->GetParamAsFloat("max");
}

ClampValidator::ClampValidator(const std::string& _type): LayerValidator(_type) {}

//

void ReLUValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<ReLULayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of ReLULayer class";
    }
    if (casted->params.count("negative_slope")) {
        casted->negative_slope = casted->GetParamAsFloat("negative_slope");
    }
}

ReLUValidator::ReLUValidator(const std::string& _type): LayerValidator(_type) {}

//

void MVNValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<MVNLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of MVNLayer class";
    }
    casted->across_channels = casted->GetParamAsInt("across_channels", 0);
    casted->normalize = casted->GetParamAsInt("normalize_variance", 1);
}

MVNValidator::MVNValidator(const std::string& _type): LayerValidator(_type) {}

//

void GRNValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<GRNLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of GRNLayer class";
    }
    casted->bias = casted->GetParamAsFloat("bias", 0.f);
}

GRNValidator::GRNValidator(const std::string& _type): LayerValidator(_type) {}

//

void SoftMaxValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<SoftMaxLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of SoftMaxLayer class";
    }
    casted->axis = casted->GetParamAsInt("axis", 1);
}

SoftMaxValidator::SoftMaxValidator(const std::string& _type): LayerValidator(_type) {}

//

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

NormValidator::NormValidator(const std::string& _type): LayerValidator(_type) {}

//

SplitValidator::SplitValidator(const std::string& _type): LayerValidator(_type) {}

void SplitValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<SplitLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of SplitLayer class";
    }
    casted->_axis = casted->GetParamAsUInt("axis", 1);

    std::string out_sizes;
    for (auto& i : layer->outData) {
        if (!out_sizes.empty()) out_sizes += ",";
        if (static_cast<int>(i->getTensorDesc().getDims().size()) <= casted->_axis) {
            THROW_IE_EXCEPTION << "Internal error - dimensions are empty";
        }
        out_sizes += std::to_string(i->getTensorDesc().getDims()[casted->_axis]);
    }
    if (!out_sizes.empty()) casted->params["out_sizes"] = out_sizes;
}

//

ConcatValidator::ConcatValidator(const std::string& _type): LayerValidator(_type) {}

void ConcatValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<ConcatLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of ConcatLayer class";
    }
    casted->_axis = casted->GetParamAsUInt("axis", 1);
}

//

GemmValidator::GemmValidator(const std::string& _type): LayerValidator(_type) {}

void GemmValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<GemmLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of GemmLayer class";
    }
    casted->alpha = casted->GetParamAsFloat("alpha", 1);
    casted->beta = casted->GetParamAsFloat("beta", 1);
    casted->transpose_a = casted->GetParamAsBool("transpose_a", false);
    casted->transpose_b = casted->GetParamAsBool("transpose_b", false);
}

//

PadValidator::PadValidator(const std::string& _type): LayerValidator(_type) {}

void PadValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<PadLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of PadLayer class";
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
        THROW_IE_EXCEPTION << layer->name << " Unsupported pad mode operation: " << mode;
    }
}

//

GatherValidator::GatherValidator(const std::string& _type): LayerValidator(_type) {}

void GatherValidator::parseParams(CNNLayer* layer) {
    if (auto casted = dynamic_cast<GatherLayer*>(layer)) {
        casted->axis = casted->GetParamAsInt("axis", 0);
    } else if (layer->insData.size() != 3) {
        THROW_IE_EXCEPTION << layer->name << " Gather layer is expected to have 3 inputs";
    }
}

//

StridedSliceValidator::StridedSliceValidator(const std::string& _type): LayerValidator(_type) {}

void StridedSliceValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<StridedSliceLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of StridedSlice class";
    }

    casted->begin_mask = layer->GetParamAsString("begin_mask", "");
    casted->end_mask = layer->GetParamAsString("end_mask", "");
    casted->ellipsis_mask = layer->GetParamAsString("ellipsis_mask", "");
    casted->new_axis_mask = layer->GetParamAsString("new_axis_mask", "");
    casted->shrink_axis_mask = layer->GetParamAsString("shrink_axis_mask", "");
}

//

ShuffleChannelsValidator::ShuffleChannelsValidator(const std::string& _type): LayerValidator(_type) {}

void ShuffleChannelsValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<ShuffleChannelsLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of ShuffleChannels class";
    }

    casted->axis = casted->GetParamAsInt("axis", 1);
    casted->group = casted->GetParamAsUInt("group", 1);
}

//

DepthToSpaceValidator::DepthToSpaceValidator(const std::string& _type): LayerValidator(_type) {}

void DepthToSpaceValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<DepthToSpaceLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of DepthToSpace class";
    }

    casted->block_size = casted->GetParamAsUInt("block_size", 1);
}

//

SpaceToDepthValidator::SpaceToDepthValidator(const std::string& _type): LayerValidator(_type) {}

void SpaceToDepthValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<SpaceToDepthLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of SpaceToDepth class";
    }

    casted->block_size = casted->GetParamAsUInt("block_size", 1);
}

/*--- SpaceToBatchValidator ---*/

SpaceToBatchValidator::SpaceToBatchValidator(const std::string& _type): LayerValidator(_type) {}

void SpaceToBatchValidator::parseParams(CNNLayer* layer) {
    auto spaceToBatchLayer = dynamic_cast<SpaceToBatchLayer*>(layer);
    if (!spaceToBatchLayer)
        THROW_IE_EXCEPTION << "'" << layer->name << "' layer is not instance of SpaceToBatchLayer class";

    if (spaceToBatchLayer->insData.size() != 4 || spaceToBatchLayer->outData.size() != 1)
        THROW_IE_EXCEPTION << "'" << spaceToBatchLayer->name << "' layer has incorrect number of inputs or outputs edges!";

    auto getParams = [](const DataPtr& dataPtr, std::vector<size_t>& dst, std::string& layerName) {
        if (dataPtr == nullptr)
            THROW_IE_EXCEPTION << "'" << layerName << "' layer has nullable input data";
        if (dataPtr->getTensorDesc().getPrecision() != Precision::I32
                && dataPtr->getTensorDesc().getPrecision() != Precision::I64)
            THROW_IE_EXCEPTION << "'" << layerName << "' layer has invalid input precision";
        auto creator = getCreatorLayer(dataPtr).lock();
        if (creator == nullptr)
            THROW_IE_EXCEPTION << "'" << layerName << "' layer has nullable input layer";

        const auto& blob = creator->blobs.begin()->second;
        dst.resize(blob->size());
        if (dataPtr->getTensorDesc().getPrecision() == Precision::I32) {
            int* data = blob->buffer().as<int*>();
            for (int i = 0; i < blob->size(); i++)
                dst[i] = data[i];
        } else if (dataPtr->getTensorDesc().getPrecision() == Precision::I64) {
            int64_t* data = blob->buffer().as<int64_t*>();
            for (int i = 0; i < blob->size(); i++)
                dst[i] = data[i];
        }
    };

    if (spaceToBatchLayer->insData[0].lock() == nullptr)
        THROW_IE_EXCEPTION << "'" << layer->name << "' layer has nullable input data";
    getParams(spaceToBatchLayer->insData[1].lock(), spaceToBatchLayer->_block_shape, layer->name);
    getParams(spaceToBatchLayer->insData[2].lock(), spaceToBatchLayer->_pads_begin, layer->name);
    getParams(spaceToBatchLayer->insData[3].lock(), spaceToBatchLayer->_pads_end, layer->name);
}

/*--- BatchToSpaceValidator ---*/

BatchToSpaceValidator::BatchToSpaceValidator(const std::string& _type): LayerValidator(_type) {}

void BatchToSpaceValidator::parseParams(CNNLayer* layer) {
    auto batchToSpaceLayer = dynamic_cast<BatchToSpaceLayer*>(layer);
    if (!batchToSpaceLayer) {
        THROW_IE_EXCEPTION << "'" << layer->name << "' layer is not instance of BatchToSpaceLayer class";
    }

    if (batchToSpaceLayer->insData.empty())
        THROW_IE_EXCEPTION << "'" << layer->name << "' layer does not have any input data";

    auto inData = batchToSpaceLayer->insData[0].lock();
    if (inData == nullptr)
        THROW_IE_EXCEPTION << "'" << layer->name << "' layer has nullable input data";

    auto getParams = [](const DataPtr& dataPtr, std::vector<size_t>& dst, std::string& layerName) {
        if (dataPtr == nullptr)
            THROW_IE_EXCEPTION << "'" << layerName << "' layer has nullable input data";
        if (dataPtr->getTensorDesc().getPrecision() != Precision::I32
                && dataPtr->getTensorDesc().getPrecision() != Precision::I64)
            THROW_IE_EXCEPTION << "'" << layerName << "' layer has invalid input precision";
        auto creator = getCreatorLayer(dataPtr).lock();
        if (creator == nullptr)
            THROW_IE_EXCEPTION << "'" << layerName << "' layer has nullable input layer";

        const auto& blob = creator->blobs.begin()->second;
        dst.resize(blob->size());
        if (dataPtr->getTensorDesc().getPrecision() == Precision::I32) {
            int* data = blob->buffer().as<int*>();
            for (int i = 0; i < blob->size(); i++)
                dst[i] = data[i];
        } else if (dataPtr->getTensorDesc().getPrecision() == Precision::I64) {
            int64_t* data = blob->buffer().as<int64_t*>();
            for (int i = 0; i < blob->size(); i++)
                dst[i] = data[i];
        }
    };

    if (batchToSpaceLayer->insData[0].lock() == nullptr)
        THROW_IE_EXCEPTION << "'" << layer->name << "' layer has nullable input data";
    getParams(batchToSpaceLayer->insData[1].lock(), batchToSpaceLayer->_block_shape, layer->name);
    getParams(batchToSpaceLayer->insData[2].lock(), batchToSpaceLayer->_crops_begin, layer->name);
    getParams(batchToSpaceLayer->insData[3].lock(), batchToSpaceLayer->_crops_end, layer->name);
}

//

BucketizeValidator::BucketizeValidator(const std::string& _type) : LayerValidator(_type) {}

void BucketizeValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<BucketizeLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of Bucketize class";
    }

    casted->with_right_bound = casted->GetParamAsBool("with_right_bound", true);
}

//

ReverseSequenceValidator::ReverseSequenceValidator(const std::string& _type): LayerValidator(_type) {}

void ReverseSequenceValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<ReverseSequenceLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of ReverseSequence class";
    }

    casted->seq_axis = casted->GetParamAsInt("seq_axis", 1);
    casted->batch_axis = casted->GetParamAsInt("batch_axis", 0);
}

/****************************************/
/*** RNN specific validators ************/
/****************************************/

static RNNCellBase::CellType cell_type_from(string type_name) {
    const vector<string> to_remove {"Cell", "Sequence"};
    for (auto& sub : to_remove) {
        auto idx = type_name.find(sub);
        if (idx != string::npos) type_name.erase(idx);
    }

    if (!one_of(type_name, "LSTM", "RNN", "GRU"))
        THROW_IE_EXCEPTION << "Unknown RNN cell type " << type_name << ". "
                           << "Expected one of [ LSTM | RNN | GRU ].";

    return type_name == "LSTM"
               ? RNNSequenceLayer::LSTM
               : type_name == "GRU" ? RNNSequenceLayer::GRU
                                    : type_name == "RNN" ? RNNSequenceLayer::RNN : RNNSequenceLayer::LSTM;
}

static RNNSequenceLayer::Direction direction_from(string direction_name) {
    if (!one_of(direction_name, "Forward", "Backward", "Bidirectional"))
        THROW_IE_EXCEPTION << "Unknown RNN direction type " << direction_name << ". "
                           << "Expected one of [ Forward | Backward | Bidirectional ].";

    return direction_name == "Forward"
               ? RNNSequenceLayer::FWD
               : direction_name == "Backward"
                     ? RNNSequenceLayer::BWD
                     : direction_name == "Bidirecttional" ? RNNSequenceLayer::BDR : RNNSequenceLayer::FWD;
}

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

void RNNBaseValidator::parseParams(CNNLayer* layer) {
    auto rnn = dynamic_cast<RNNCellBase*>(layer);
    if (!rnn) THROW_IE_EXCEPTION << "Layer is not instance of RNNLayer class";

    rnn->cellType = cell_type_from(layer->type);
    rnn->hidden_size = rnn->GetParamAsInt("hidden_size");
    rnn->clip = rnn->GetParamAsFloat("clip", 0.0f);
    rnn->activations = rnn->GetParamAsStrings("activations", def_acts);
    rnn->activation_alpha = rnn->GetParamAsFloats("activation_alpha", def_alpha);
    rnn->activation_beta = rnn->GetParamAsFloats("activation_beta", def_beta);

    if (rnn->cellType == RNNCellBase::GRU) {
        auto lbr = rnn->GetParamAsBool("linear_before_reset", false);
        if (lbr) rnn->cellType = RNNCellBase::GRU_LBR;
    }
}

//

template <RNNSequenceLayer::CellType CELL>
RNNSequenceValidator<CELL>::RNNSequenceValidator(const std::string& _type): RNNBaseValidator(_type, CELL) {}

template <RNNSequenceLayer::CellType CELL>
void RNNSequenceValidator<CELL>::parseParams(CNNLayer* layer) {
    RNNBaseValidator::parseParams(layer);

    auto casted = dynamic_cast<RNNSequenceLayer*>(layer);
    if (!casted) THROW_IE_EXCEPTION << "Layer is not instance of RNNLayer class";

    std::string direction = layer->GetParamAsString("direction");

    casted->axis = layer->GetParamAsUInt("axis", 1);
    casted->direction = direction_from(direction);
}

template class details::RNNSequenceValidator<RNNSequenceLayer::RNN>;
template class details::RNNSequenceValidator<RNNSequenceLayer::GRU>;
template class details::RNNSequenceValidator<RNNSequenceLayer::LSTM>;

//

template <RNNSequenceLayer::CellType CELL>
RNNCellValidator<CELL>::RNNCellValidator(const std::string& _type): RNNBaseValidator(_type, CELL) {}

template class details::RNNCellValidator<RNNSequenceLayer::RNN>;
template class details::RNNCellValidator<RNNSequenceLayer::GRU>;
template class details::RNNCellValidator<RNNSequenceLayer::LSTM>;

//

void DetectionOutputValidator::parseParams(CNNLayer* layer) {
    unsigned int num_classes = layer->GetParamAsUInt("num_classes");
    if (num_classes == 0) {
        THROW_IE_EXCEPTION << "num_classes parameter of DetectionOutput layer can't be equal to zero";
    }
    float _nms_threshold = layer->GetParamAsFloat("nms_threshold");
    if (_nms_threshold < 0) {
        THROW_IE_EXCEPTION << "nms_threshold parameter of DetectionOutput layer can't be less then zero";
    }
    int _keep_top_k = layer->GetParamAsInt("keep_top_k", -1);

    if (layer->CheckParamPresence("background_label_id"))
        int _background_label_id = layer->GetParamAsInt("background_label_id", -1);
    if (layer->CheckParamPresence("top_k"))
        int _top_k = layer->GetParamAsInt("top_k", -1);
    if (layer->CheckParamPresence("variance_encoded_in_target"))
        bool _variance_encoded_in_target = static_cast<bool>(layer->GetParamAsUInt("variance_encoded_in_target", 0));
    if (layer->CheckParamPresence("num_orient_classes"))
        int _num_orient_classes = layer->GetParamAsUInt("num_orient_classes");
    if (layer->CheckParamPresence("share_location"))
        bool _share_location = static_cast<bool>(layer->GetParamAsUInt("share_location", 1));
    if (layer->CheckParamPresence("interpolate_orientation"))
        int _interpolate_orientation = layer->GetParamAsInt("interpolate_orientation");
    if (layer->CheckParamPresence("confidence_threshold")) {
        float _confidence_threshold = layer->GetParamAsFloat("confidence_threshold");
        if (_confidence_threshold < 0) {
            THROW_IE_EXCEPTION << "_confidence_threshold parameter of DetectionOutput layer can't be less then zero";
        }
    }

    if (layer->CheckParamPresence("code_type")) {
        std::string _code_type = layer->GetParamAsString("code_type");
        for (auto& c: _code_type)
        {
            c = std::tolower(c);
        }
        std::vector<std::string> code_types = {"caffe.priorboxparameter.center_size", "caffe.priorboxparameter.corner"};
        auto it = std::find(code_types.begin(), code_types.end(), _code_type);
        if (it == code_types.end()) {
            THROW_IE_EXCEPTION << "Parameter code_type of DetectionOutput layer ";
        }
    }
}

DetectionOutputValidator::DetectionOutputValidator(const std::string& _type): LayerValidator(_type) {}

//

void ProposalValidator::parseParams(CNNLayer* layer) {
    if (layer->params.find("num_outputs") == layer->params.end()) {
        layer->params["num_outputs"] = std::to_string(layer->outData.size());
    }
}

ProposalValidator::ProposalValidator(const std::string& _type): LayerValidator(_type) {}

//

OneHotValidator::OneHotValidator(const std::string& _type): LayerValidator(_type) {}

void OneHotValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<OneHotLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not an instance of the OneHot class";
    }

    if (layer->CheckParamPresence("depth")) {
        casted->depth = layer->GetParamAsUInt("depth");
    } else {
        THROW_IE_EXCEPTION << "The required depth parameter of OneHot layer is missing";
    }

    auto on_value_str = layer->GetParamAsString("on_value", "1.0");
    auto off_value_str = layer->GetParamAsString("off_value", "0.0");

    // there are some key words to represent reserved values
    auto universal_read = [] (std::string str) {
        float res;
        if (str == "True")
            res = 1.0f;
        else if (str == "False")
            res = 0.0f;
        else
            res = CNNLayer::ie_parse_float(str);
        return res;
    };

    casted->on_value = universal_read(on_value_str);
    casted->off_value = universal_read(off_value_str);

    casted->axis = layer->GetParamAsInt("axis", -1);
}

//

QuantizeValidator::QuantizeValidator(const std::string& _type): LayerValidator(_type) {}

void QuantizeValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<QuantizeLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << "Layer is not instance of QuantizeLayer class";
    }

    casted->levels = casted->GetParamAsInt("levels", 1);

    if (casted->levels <= 1) {
        THROW_IE_EXCEPTION << layer->name << ": Incorrect value for parameter levels = " << casted->levels
                           << ". Expected to be > 1.";
    }
}

//

BinaryConvolutionValidator::BinaryConvolutionValidator(const std::string& _type): LayerValidator(_type) {}

void BinaryConvolutionValidator::parseParams(CNNLayer* layer) {
    auto binConvLayer = dynamic_cast<BinaryConvolutionLayer*>(layer);
    if (!binConvLayer) {
        THROW_IE_EXCEPTION << "Layer is not instance of BinaryConvolutionLayer class";
    }

    binConvLayer->_pad_value = binConvLayer->GetParamAsFloat("pad_value", 0.f);
    binConvLayer->_in_depth = binConvLayer->GetParamAsUInt("input");
    binConvLayer->_mode = BinaryConvolutionLayer::xnor_popcount;
    std::string mode = binConvLayer->GetParamAsString("mode", "xnor-popcount");
    if (mode != "xnor-popcount") THROW_IE_EXCEPTION << "Layer with type `" << _type << "` has incorrect mode!";

    binConvLayer->_out_depth = binConvLayer->GetParamAsUInt("output");

    binConvLayer->_kernel.clear();
    binConvLayer->_stride.clear();
    binConvLayer->_padding.clear();
    binConvLayer->_pads_end.clear();
    binConvLayer->_dilation.clear();

    vector<unsigned int> kernels = binConvLayer->GetParamAsUInts("kernel", {});
    if (kernels.empty()) {
        // IR_v == 2
        binConvLayer->_kernel.insert(X_AXIS, binConvLayer->GetParamAsUInt("kernel-x"));
        binConvLayer->_kernel.insert(Y_AXIS, binConvLayer->GetParamAsUInt("kernel-y"));

        binConvLayer->_stride.insert(X_AXIS, binConvLayer->GetParamAsUInt("stride-x", 1u));
        binConvLayer->_stride.insert(Y_AXIS, binConvLayer->GetParamAsUInt("stride-y", 1u));
        // TODO: maybe just throw exception, why do we change IR?
        if (0 == binConvLayer->_stride[X_AXIS]) {
            binConvLayer->_stride[X_AXIS] = 1u;
        }
        if (0 == binConvLayer->_stride[Y_AXIS]) {
            binConvLayer->_stride[Y_AXIS] = 1u;
        }

        binConvLayer->_padding.insert(X_AXIS, binConvLayer->GetParamAsUInt("pad-x", 0u));
        binConvLayer->_padding.insert(Y_AXIS, binConvLayer->GetParamAsUInt("pad-y", 0u));

        binConvLayer->_pads_end.insert(X_AXIS, binConvLayer->GetParamAsUInt("pad-r", binConvLayer->_padding[X_AXIS]));
        binConvLayer->_pads_end.insert(Y_AXIS, binConvLayer->GetParamAsUInt("pad-b", binConvLayer->_padding[Y_AXIS]));

        binConvLayer->_dilation.insert(X_AXIS, binConvLayer->GetParamAsUInt("dilation-x", 1u));
        binConvLayer->_dilation.insert(Y_AXIS, binConvLayer->GetParamAsUInt("dilation-y", 1u));
    } else {
        // IR_v > 2
        for (int i = 1; i <= kernels.size(); i++) {
            binConvLayer->_kernel.insert(i - 1, kernels[kernels.size() - i]);
        }

        vector<unsigned int> default_0 = vector<unsigned int>(binConvLayer->_kernel.size(), 0u);
        vector<unsigned int> default_1 = vector<unsigned int>(binConvLayer->_kernel.size(), 1u);

        vector<unsigned int> strides = binConvLayer->GetParamAsUInts("strides", default_1);
        for (int i = 1; i <= strides.size(); i++) {
            if (strides[strides.size() - i] == 0) {
                THROW_IE_EXCEPTION << "Stride could not be 0.\nIn layer " << binConvLayer->name;
            }
            binConvLayer->_stride.insert(i - 1, strides[strides.size() - i]);
        }

        vector<unsigned int> pads_begin = binConvLayer->GetParamAsUInts("pads_begin", default_0);
        for (int i = 1; i <= pads_begin.size(); i++) {
            binConvLayer->_padding.insert(i - 1, pads_begin[pads_begin.size() - i]);
        }

        vector<unsigned int> pads_end = binConvLayer->GetParamAsUInts("pads_end", pads_begin);
        for (int i = 1; i <= pads_end.size(); i++) {
            binConvLayer->_pads_end.insert(i - 1, pads_end[pads_end.size() - i]);
        }

        vector<unsigned int> dilations = binConvLayer->GetParamAsUInts("dilations", default_1);
        for (int i = 1; i <= dilations.size(); i++) {
            binConvLayer->_dilation.insert(i - 1, dilations[dilations.size() - i]);
        }
    }

    binConvLayer->_auto_pad = binConvLayer->GetParamAsString("auto_pad", "");
    binConvLayer->_group = binConvLayer->GetParamAsUInt("group", 1u);
}

//

ReduceValidator::ReduceValidator(const std::string& _type): LayerValidator(_type) {}

void ReduceValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<ReduceLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of Reduce class";
    }

    casted->keep_dims = layer->GetParamAsBool("keep_dims", "true");
}

//

TopKValidator::TopKValidator(const std::string& _type): LayerValidator(_type) {}

void TopKValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<TopKLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of TopK class";
    }

    casted->mode = layer->GetParamAsString("mode", "max");
    if (casted->mode != "max" && casted->mode != "min")
        THROW_IE_EXCEPTION << layer->name
                           << " TopK can take only 'max' or 'min' for mode, but actually it has: " << casted->mode;
    casted->sort = layer->GetParamAsString("sort", "index");
    if (casted->sort != "value" && casted->sort != "index" && casted->sort != "none")
        THROW_IE_EXCEPTION << layer->name
                           << " TopK can take only 'value', 'index' or 'none' for sort, but actually it has: "
                           << casted->sort;
    casted->axis = layer->GetParamAsInt("axis", -1);
}

//

UniqueValidator::UniqueValidator(const std::string& _type): LayerValidator(_type) {}

void UniqueValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<UniqueLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of Unique class";
    }

    casted->sorted = layer->GetParamAsBool("sorted");
    casted->return_inverse = layer->GetParamAsBool("return_inverse");
    casted->return_counts = layer->GetParamAsBool("return_counts");
}

//

NMSValidator::NMSValidator(const std::string& _type): LayerValidator(_type) {}

void NMSValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<NonMaxSuppressionLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of NonMaxSuppression class";
    }

    casted->center_point_box = layer->GetParamAsBool("center_point_box", false);
    casted->sort_result_descending = layer->GetParamAsBool("sort_result_descending", true);
    casted->output_type = layer->GetParamAsString("output_type", "I64");
}

#define REG_LAYER_VALIDATOR_FOR_TYPE(__validator, __type) _validators[#__type] = std::make_shared<__validator>(#__type)

LayerValidators::LayerValidators() : _validators() {
    REG_LAYER_VALIDATOR_FOR_TYPE(BatchNormalizationValidator, BatchNormalization);
    REG_LAYER_VALIDATOR_FOR_TYPE(ClampValidator, Clamp);
    REG_LAYER_VALIDATOR_FOR_TYPE(ConcatValidator, Concat);
    REG_LAYER_VALIDATOR_FOR_TYPE(ConvolutionValidator, Convolution);
    REG_LAYER_VALIDATOR_FOR_TYPE(CropValidator, Crop);
    REG_LAYER_VALIDATOR_FOR_TYPE(DeconvolutionValidator, Deconvolution);
    REG_LAYER_VALIDATOR_FOR_TYPE(DeformableConvolutionValidator, DeformableConvolution);
    REG_LAYER_VALIDATOR_FOR_TYPE(DetectionOutputValidator, DetectionOutput);
    REG_LAYER_VALIDATOR_FOR_TYPE(EltwiseValidator, Eltwise);
    REG_LAYER_VALIDATOR_FOR_TYPE(FullyConnectedValidator, InnerProduct);
    REG_LAYER_VALIDATOR_FOR_TYPE(FullyConnectedValidator, FullyConnected);
    REG_LAYER_VALIDATOR_FOR_TYPE(GRNValidator, GRN);
    REG_LAYER_VALIDATOR_FOR_TYPE(MVNValidator, MVN);
    REG_LAYER_VALIDATOR_FOR_TYPE(NormValidator, Norm);
    REG_LAYER_VALIDATOR_FOR_TYPE(NormValidator, LRN);
    REG_LAYER_VALIDATOR_FOR_TYPE(PReLUValidator, PReLU);
    REG_LAYER_VALIDATOR_FOR_TYPE(PoolingValidator, Pooling);
    REG_LAYER_VALIDATOR_FOR_TYPE(PowerValidator, Power);
    REG_LAYER_VALIDATOR_FOR_TYPE(ProposalValidator, Proposal);
    REG_LAYER_VALIDATOR_FOR_TYPE(ReLUValidator, ReLU);
    REG_LAYER_VALIDATOR_FOR_TYPE(ReshapeValidator, Reshape);
    REG_LAYER_VALIDATOR_FOR_TYPE(ReshapeValidator, Flatten);
    REG_LAYER_VALIDATOR_FOR_TYPE(ScaleShiftValidator, ScaleShift);
    REG_LAYER_VALIDATOR_FOR_TYPE(SoftMaxValidator, SoftMax);
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
    REG_LAYER_VALIDATOR_FOR_TYPE(ReverseSequenceValidator, ReverseSequence);
    REG_LAYER_VALIDATOR_FOR_TYPE(RNNCellValidator<RNNSequenceLayer::RNN>, RNNCell);
    REG_LAYER_VALIDATOR_FOR_TYPE(RNNCellValidator<RNNSequenceLayer::GRU>, GRUCell);
    REG_LAYER_VALIDATOR_FOR_TYPE(RNNCellValidator<RNNSequenceLayer::LSTM>, LSTMCell);
    REG_LAYER_VALIDATOR_FOR_TYPE(RNNSequenceValidator<RNNSequenceLayer::RNN>, RNNSequence);
    REG_LAYER_VALIDATOR_FOR_TYPE(RNNSequenceValidator<RNNSequenceLayer::GRU>, GRUSequence);
    REG_LAYER_VALIDATOR_FOR_TYPE(RNNSequenceValidator<RNNSequenceLayer::LSTM>, LSTMSequence);
    REG_LAYER_VALIDATOR_FOR_TYPE(TileValidator, Tile);
    REG_LAYER_VALIDATOR_FOR_TYPE(OneHotValidator, OneHot);
    REG_LAYER_VALIDATOR_FOR_TYPE(QuantizeValidator, FakeQuantize);
    REG_LAYER_VALIDATOR_FOR_TYPE(BinaryConvolutionValidator, BinaryConvolution);
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
    REG_LAYER_VALIDATOR_FOR_TYPE(TopKValidator, TopK);
    REG_LAYER_VALIDATOR_FOR_TYPE(UniqueValidator, Unique);
    REG_LAYER_VALIDATOR_FOR_TYPE(NMSValidator, NonMaxSuppression);
}

}  // namespace InferenceEngine
