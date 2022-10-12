// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_limitations.hpp"

#include "gna/gna_config.hpp"

#include <cstdint>
#include <unordered_set>
#include <legacy/ie_layers.h>
#include <legacy/graph_tools.hpp>
#include <layers/gna_layer_type.hpp>
#include <layers/gna_layer_info.hpp>
#include "gna_graph_tools.hpp"
#include "gna_lib_ver_selector.hpp"
#include "common/gna_target.hpp"

namespace GNAPluginNS {
namespace GNALimitations {
namespace Cnn2D {

bool RangeLimit::isValid(const uint32_t val) const {
    return val >= min && val <= max;
}

std::string RangeLimit::GetErrorOrEmpty(const uint32_t val) const {
    std::ostringstream out;
    if (!isValid(val)) {
        out << "Unsupported " << what << ", actual value: " << val << ", valid range [" << min << ", " << max << "]\n";
    }
    return out.str();
}

bool RangeLimit2D::isValid(const uint32_t h, const uint32_t w) const {
    return hLimit.isValid(h) && wLimit.isValid(w);
}

std::string RangeLimit2D::GetErrorOrEmpty(const uint32_t h, const uint32_t w) const {
    return hLimit.GetErrorOrEmpty(h) + wLimit.GetErrorOrEmpty(w);
}

RangeMultipleLimit::RangeMultipleLimit(RangeLimit rlIn, uint32_t multiplierIn) : RangeLimit(rlIn), multiplier(multiplierIn) {
}

bool RangeMultipleLimit::isValid(const uint32_t val) const {
    return RangeLimit::isValid(val) && (val % multiplier == 0);
}

std::string RangeMultipleLimit::GetErrorOrEmpty(const uint32_t val) const {
    auto e = RangeLimit::GetErrorOrEmpty(val);
    std::ostringstream out;
    if (val % multiplier != 0) {
        out << "Unsupported " << what << ": " << val << ", must be multiple of " << multiplier << "\n";
    }
    return e + out.str();
}

bool VectorOrSquareLimit::isValid(const uint32_t h, const uint32_t w) const {
    if (w == 1 && h >= 1 && h <= maxVectorHeight) return true;
    if (h == 1 && w >= 1 && w <= maxVectorWidth) return true;
    if (h == w && h <= maxSquare && h >= 1) return true;
    return false;
}

std::string VectorOrSquareLimit::GetErrorOrEmpty(const uint32_t h, const uint32_t w, std::string what) const {
    std::ostringstream out;
    if (!isValid(h, w)) {
        out << "Unsupported " << what << " shape, actual HxW: " << h << "x" << w <<
            ", only vertical vector up to " << maxVectorHeight << "x1, horizontal up to 1x" << maxVectorWidth <<
            " or square up to " << maxSquare << "x" << maxSquare << " are valid\n";
    }
    return out.str();
}


bool RectLimit::isValid(const uint32_t h, const uint32_t w) const {
    if (h >= 1 && h <= maxVectorHeight && w >= 1 && w <= maxVectorWidth) return true;
    return false;
}

std::string RectLimit::GetErrorOrEmpty(const uint32_t h, const uint32_t w, std::string what) const {
    std::ostringstream out;
    if (!isValid(h, w)) {
        out << "Unsupported " << what << " shape, actual HxW: " << h << "x" << w <<
            ", only rectangular shapes up to " << maxVectorHeight << "x" << maxVectorWidth << " are valid\n";
    }
    return out.str();
}

RectLimit RectLimitByChannels::GetByChannels(const uint32_t channels) const {
    for (auto&& limit : limitPerChannel) {
        if (limit.first >= channels) {
            return limit.second;
        }
    }
    return RectLimit{ 0, 0 };
}

bool RectLimitByChannels::isValid(const uint32_t h, const uint32_t w, const uint32_t channels) const {
    return GetByChannels(channels).isValid(h, w);
}

std::string RectLimitByChannels::GetErrorOrEmpty(const uint32_t h, const uint32_t w,
    const uint32_t channels, std::string what) const {
    return GetByChannels(channels).GetErrorOrEmpty(h, w, what);
}

RectLimitByChannels RectLimitByChannelsAndPrecision::GetByPrecision(const OvGnaType precision) const {
    return precision == OvGnaTypeInt8 ? lowPrecision : defaultPrecision;
}

bool RectLimitByChannelsAndPrecision::isValid(const uint32_t h, const uint32_t w, const OvGnaType precision, const uint32_t channels) const {
    return GetByPrecision(precision).isValid(h, w, channels);
}

std::string RectLimitByChannelsAndPrecision::GetErrorOrEmpty(const uint32_t h, const uint32_t w,
    const OvGnaType precision, const uint32_t channels, std::string what) const {
    return GetByPrecision(precision).GetErrorOrEmpty(h, w, channels, what);
}

const RangeLimit2D Validator_30::kInputHWLimit{{16, 384, "input height"}, {16, 240, "input width"}};
const RangeMultipleLimit Validator_30::kInputChannelsNumberLimit{{8, 384, "number of input channels"}, 8};

const RangeMultipleLimit Validator_30::kKernelNumberLimit{{8, 1024, "number of kernels"}, 8};
const RectLimitByChannelsAndPrecision Validator_30::kKernelLimit{
    {{{96, {7, 7}}, {136, {7, 5}}, {168, {7, 4}}, {240, {7, 3}}, {384, {7, 2}}}},
    {{{48, {7, 7}}, {64, {7, 5}}, {80, {7, 4}}, {120, {7, 3}}, {384, {7, 1}}}},
};

const RangeLimit2D Validator_30::kDilationLimit{{convDilationHeight, convDilationHeight, "dilation height"},
                                                {convDilationWidth, convDilationWidth, "dilation width"}};

bool Validator_30::ValidateCnn2D(const std::string &name, const uint32_t inHeight, const uint32_t inWidth,
    const uint32_t inChannels, const uint32_t kernelH, const uint32_t kernelW, const uint32_t kernelN,
    const uint32_t strideH, const uint32_t strideW, const uint32_t dilationH, const uint32_t dilationW,
    const OvGnaType inPrecision, const bool throwOnError) const {
    const auto& kStrideLimit = kKernelLimit;
    auto error = kInputHWLimit.GetErrorOrEmpty(inHeight, inWidth);

    error += kKernelNumberLimit.GetErrorOrEmpty(kernelN);
    error += kInputChannelsNumberLimit.GetErrorOrEmpty(inChannels);
    error += kKernelLimit.GetErrorOrEmpty(kernelH, kernelW, inPrecision, inChannels, "kernel");
    error += kStrideLimit.GetErrorOrEmpty(strideH, strideW, inPrecision, inChannels, "convolution stride");

    const RangeLimit kKernelStrideHLimit{1, kernelH, "kernel stride height (must be up to kernel height)"};
    const RangeLimit kKernelStrideWLimit{1, kernelW, "kernel stride width (must be up to kernel width)"};

    error += kKernelStrideHLimit.GetErrorOrEmpty(strideH);
    error += kKernelStrideWLimit.GetErrorOrEmpty(strideW);

    error += kDilationLimit.GetErrorOrEmpty(dilationH, dilationW);

    return ValidationSuccesful(throwOnError, error, name, "Convolution2D");
}

const VectorOrSquareLimit Validator_30::kPoolingWindowLimit{3, 1, 1};

bool Validator_30::ValidatePooling2D(const std::string& name,
    const uint32_t windowH, const uint32_t windowW,
    const uint32_t strideH, const uint32_t strideW,
    const bool throwOnError) const {
    auto error = kPoolingWindowLimit.GetErrorOrEmpty(windowH, windowW, "pooling window");
    const RangeLimit poolingStrideHLimit{ 1, windowH, "pooling stride height (must be up to pooling window height)" };
    const RangeLimit poolingStrideWLimit{ 1, windowW, "pooling stride width (must be up to pooling window width)" };

    error += poolingStrideHLimit.GetErrorOrEmpty(strideH);
    error += poolingStrideWLimit.GetErrorOrEmpty(strideW);

    return ValidationSuccesful(throwOnError, error, name, "Pooling2D");
}

bool Validator_30::IsPaddingSupported() const {
    return false;
}

const RangeLimit2D Validator_35::kInputHWLimit{{1, 65535, "input height"}, {1, 65535, "input width"}};
const RangeLimit Validator_35::kInputChannelsNumberLimit1B{1, 2048, "number of input channels"};
const RangeLimit Validator_35::kInputChannelsNumberLimit2B{1, 1024, "number of input channels"};

const RangeLimit Validator_35::kKernelNumberLimit{1, 8192, "number of kernels"};
const RangeLimit2D Validator_35::kKerneHWlLimit{{1, 255, "kernel height"}, {1, 256, "kernel width"}};
const RangeLimit2D Validator_35::kStrideHWLimit{{1, 255, "convolution stride height"},
                                                {1, 256, "convolution stride width"}};
const RangeLimit2D Validator_35::kDilationLimit{{convDilationHeight, convDilationHeight, "dilation height"},
                                                {convDilationWidth, convDilationWidth, "dilation width"}};

bool Validator_35::ValidateCnn2D(const std::string& name, const uint32_t inHeight, const uint32_t inWidth,
    const uint32_t inChannels, const uint32_t kernelH, const uint32_t kernelW, const uint32_t kernelN,
    const uint32_t strideH, const uint32_t strideW, const uint32_t dilationH, const uint32_t dilationW,
    const OvGnaType inPrecision, const bool throwOnError) const {
    auto error = kInputHWLimit.GetErrorOrEmpty(inHeight, inWidth);

    error += kKernelNumberLimit.GetErrorOrEmpty(kernelN);
    auto& inputChannelsNumberLimit = inPrecision == OvGnaTypeInt8 ? kInputChannelsNumberLimit1B : kInputChannelsNumberLimit2B;
    error += inputChannelsNumberLimit.GetErrorOrEmpty(inChannels);
    error += kKerneHWlLimit.GetErrorOrEmpty(kernelH, kernelW);
    error += kStrideHWLimit.GetErrorOrEmpty(strideH, strideW);

    error += kDilationLimit.GetErrorOrEmpty(dilationH, dilationW);

    return ValidationSuccesful(throwOnError, error, name, "Convolution2D");
}

const RangeLimit2D Validator_35::kPoolingWindowHWLimit{{1, 255, "pooling window height"},
                                                       {1, 255, "pooling window width"}};
const RangeLimit2D Validator_35::kPoolingStrideHWLimit{{1, 255, "pooling stride height"},
                                                       {1, 255, "pooling stride width"}};

bool Validator_35::ValidatePooling2D(const std::string& name,
    const uint32_t windowH, const uint32_t windowW,
    const uint32_t strideH, const uint32_t strideW,
    const bool throwOnError) const {
    auto error = kPoolingWindowHWLimit.GetErrorOrEmpty(windowH, windowW);
    error += kPoolingStrideHWLimit.GetErrorOrEmpty(strideH, strideW);

    return ValidationSuccesful(throwOnError, error, name, "Pooling2D");
}

bool Validator_35::IsPaddingSupported() const {
    return true;
}

std::unique_ptr<AbstractValidator> AbstractValidator::Create(const std::string& target) {
    if (target == common::kGnaTarget3_0) {
        return tools::make_unique<Validator_30>();
    } else if (target == common::kGnaTarget3_5) {
        return tools::make_unique<Validator_35>();
    }
    return nullptr;
}

void AbstractValidator::ThrowIfNotEmpty(const std::string& prefix, const std::string& error) {
    if (!error.empty()) {
        THROW_GNA_EXCEPTION << prefix << error;
    }
}

bool AbstractValidator::ValidationSuccesful(const bool throwOnError,
    const std::string& error,
    const std::string& operationName,
    const std::string& type) {
    if (throwOnError) {
        const std::string prefix = "Layer " + type + ": " + operationName + ":";
        ThrowIfNotEmpty(prefix, error);
    }

    return error.empty();
}

} // namespace Cnn2D

IE_SUPPRESS_DEPRECATED_START
static bool ValidateConcatAxis(const InferenceEngine::CNNLayerPtr layer, std::string& errMessage) {
    LayerInfo info(layer);
    auto concat_layer = info.as<InferenceEngine::ConcatLayer*>();
    IE_ASSERT(concat_layer);
    auto dims_size = concat_layer->insData[0].lock()->getDims().size();
    auto in_dims = concat_layer->insData[0].lock()->getDims();
    auto concat_axis = concat_layer->_axis;

    if (dims_size >= 2) {
        InferenceEngine::CNNLayerPtr prev_layer, pre_prev_layer;
        // Skip all convolutions in this check, they will be handled during concat primitive creation
        auto isFusableWithConv = [](InferenceEngine::CNNLayerPtr ptr) {
            return (LayerInfo(ptr).isFusableWithConv() || LayerInfo(ptr).isNonFunctional() ||
                (LayerInfo(ptr).isPermute() && ((ptr->input()->getLayout() == InferenceEngine::Layout::NCHW &&
                    ptr->GetParamAsInts("order") == GetPermuteOrder(InferenceEngine::Layout::NCHW, InferenceEngine::Layout::NHWC)) ||
                    (ptr->input()->getLayout() == InferenceEngine::Layout::CHW &&
                        ptr->GetParamAsInts("order") == std::vector<int32_t>{0, 2, 1} /* NCW to NWC */))));
        };

        for (auto input_idx = 0; input_idx != concat_layer->insData.size(); input_idx++) {
            prev_layer = InferenceEngine::CNNNetPrevLayerSkipCertain(layer, input_idx, isFusableWithConv);
            if (prev_layer && LayerInfo(prev_layer).isConvolution())
                return true;
        }

        // Look for trivial cases which will be flattened later
        // for explanation of what is meant by trivial case,
        // look to FlattenTrivialConcatPass comments
        // TODO: detection of trivial cases could be moved to one common place
        // when all transformations are migrated to ngraph
        bool is_not_trivial_concat = false;

        // Concatentaion of consts and input parameters only is supported, even if first dimentsion of input parameter > 1
        bool concat_all_const_or_inputs = false;

        // If concat axis > 0, detect any dimension > 1 before the concat axis
        if (concat_axis > 0) {
            for (unsigned int axis = 0; axis < concat_axis; axis++) {
                if (in_dims[axis] > 1) {
                    is_not_trivial_concat = true;
                    break;
                }
            }
        // If concat axis == 0, detect any preceding functional layer's input
        // with 0'th dimension > 1, but take into account that some layers need to be skipped
        } else {
            concat_all_const_or_inputs = true;

            for (auto input_idx = 0; input_idx != concat_layer->insData.size(); input_idx++) {
                if (concat_layer->insData[input_idx].lock()->getDims()[0] != 1) {
                    // First we're checking concat input layers
                    prev_layer = InferenceEngine::CNNNetPrevLayerSkipCertain(concat_layer, input_idx,
                        [](InferenceEngine::CNNLayerPtr ptr) {
                            return LayerInfo(ptr).isNonFunctional() || LayerInfo(ptr).isFakeQuantize();
                        });

                    IE_ASSERT(prev_layer);

                    if (LayerInfo(prev_layer).isInput() && prev_layer->outData[0]->getDims()[0] == 1 ||
                        LayerInfo(prev_layer).isConst()) {
                        continue;
                    } else if ((LayerInfo(prev_layer).isInput() && prev_layer->outData[0]->getDims()[0] != 1)) {
                        is_not_trivial_concat = true;
                        break;
                    }

                    // If it's not clear still if concat is supported,
                    // we're moving one more layer back to see the dimensions
                    pre_prev_layer = InferenceEngine::CNNNetPrevLayerSkipCertain(prev_layer, 0,
                        [](InferenceEngine::CNNLayerPtr ptr) {
                            return LayerInfo(ptr).isNonFunctional() || LayerInfo(ptr).isFakeQuantize() ||
                                LayerInfo(ptr).isSplit();
                        });

                    IE_ASSERT(pre_prev_layer);

                    if (LayerInfo(pre_prev_layer).isConst()) {
                        continue;
                    }

                    concat_all_const_or_inputs = false;

                    if (LayerInfo(pre_prev_layer).isInput() && pre_prev_layer->outData[0]->getDims()[0] == 1)
                        continue;

                    if (pre_prev_layer->outData[0]->getDims()[0] != 1) {
                        is_not_trivial_concat = true;
                        break;
                    }
                }
            }
        }

        // This is a trivial concat or it isn't a 'not trivial one' :-)
        // it can be flattened and we're allowing it
        if (!is_not_trivial_concat || concat_all_const_or_inputs)
            return true;

        // For interleaved inputs start checking from axis 1
        // and allow concatenation on axis 0 only when all other dimesions = 1
        std::rotate(in_dims.begin(), in_dims.begin() + 1, in_dims.end());
        concat_axis == 0 ? concat_axis = static_cast<unsigned int>(dims_size - 1) : concat_axis--;

        // Looking for any axis with dimension > 1 before concatentaion axis;
        // in general such concatenation is unsupported
        auto end_dim = in_dims.begin() + concat_axis;
        auto unsupported_concat_axis = std::find_if(in_dims.begin(), end_dim, [](const size_t& in_dim) {
            return (in_dim > 1);
            });

        if (unsupported_concat_axis != end_dim) {
            auto dims = concat_layer->insData[0].lock()->getDims();
            std::ostringstream in_dims_oss;
            std::copy(dims.begin(), dims.end(), std::ostream_iterator<size_t>(in_dims_oss, ","));
            errMessage = "[ WARNING ] Topology with layer: " + layer->name + ", type: " + layer->type +
                ", and concatenation axis(" + std::to_string(concat_layer->_axis) +
                ") for input dimensions(" + in_dims_oss.str() + ") not supported\n";
            return false;
        }
    }
    return true;
}

bool ValidateConvConcatAxis(const InferenceEngine::ConcatLayer* concat_layer) {
    IE_ASSERT(concat_layer);
    auto dims_size = concat_layer->insData[0].lock()->getDims().size();

    if (dims_size >= 2) {
        InferenceEngine::CNNLayerPtr prev_layer;

        // Skipping here all layers which would disappear or otherwise fuse with convolution in the final GNA graph
        auto isFusableWithConv = [](InferenceEngine::CNNLayerPtr ptr) {
            return (LayerInfo(ptr).isFusableWithConv() || LayerInfo(ptr).isNonFunctional());
        };

        auto in_dims = concat_layer->insData[0].lock()->getDims();
        auto concat_axis = concat_layer->_axis;
        auto concat_layout = concat_layer->input()->getLayout();

        for (auto input_idx = 0; input_idx != concat_layer->insData.size(); input_idx++) {
            // Supported cases for concatenation of a convolution
            prev_layer = InferenceEngine::CNNNetPrevLayerSkipCertain(concat_layer, input_idx, isFusableWithConv);
            if (prev_layer && LayerInfo(prev_layer).isConvolution()) {
                // Allow concatenation along N axis for non-interleaved primitives
                // (currently only convolution)
                if (concat_layer->_axis == 0)
                    break;

                // Convert dims to NHWC layout to allow later verification
                auto new_order = GetPermuteOrder(concat_layout, InferenceEngine::Layout::NHWC);
                InferenceEngine::SizeVector new_dims;
                for (size_t i = 0; i < dims_size; ++i) {
                    new_dims.push_back(in_dims[new_order[i]]);
                }
                concat_axis = GetPermuteOrder(InferenceEngine::Layout::NHWC, concat_layout)[concat_axis];

                // Looking for any axis with dimension > 1 before concatentaion axis;
                // in general such concatenation is unsupported
                auto end_dim = new_dims.begin() + concat_axis;
                auto unsupportedconcat_axis = std::find_if(new_dims.begin(), end_dim, [](const size_t& inDim) {
                    return (inDim > 1);
                    });

                if (unsupportedconcat_axis != end_dim) {
                    return false;
                }

                break;
            }
        }
    }
    return true;
}

bool AreLayersSupported(InferenceEngine::CNNNetwork& network, std::string& errMessage, bool userWarning) {
    IE_SUPPRESS_DEPRECATED_START
    InferenceEngine::InputsDataMap inputs = network.getInputsInfo();
    std::unordered_set<InferenceEngine::CNNLayer *> allLayers;
    InferenceEngine::CNNLayerPtr startLayer;
    if (inputs.empty()) {
        auto outputs = network.getOutputsInfo();
        IE_ASSERT(!outputs.empty());
        // If there are no inputs start search from an output
        startLayer = getCreatorLayer(outputs.begin()->second).lock();
    } else {
        auto network_input_precision = inputs.begin()->second->getPrecision();

        if (network_input_precision != InferenceEngine::Precision::FP32 &&
            network_input_precision != InferenceEngine::Precision::I16 &&
            network_input_precision != InferenceEngine::Precision::U8) {
            errMessage = "The plugin does not support input precision with " +
                         std::string(network_input_precision.name()) +
                         " format. Supported  input precisions FP32, I16, U8\n";
            return false;
        }

        auto & secondLayers = getInputTo(inputs.begin()->second->getInputData());
        if (secondLayers.empty()) {
            errMessage = "Network consists of input layer only (GNA)\n";
            return false;
        }
        startLayer = secondLayers.begin()->second;
    }
    auto batch_size = network.getBatchSize();
    bool check_result = true;
    InferenceEngine::details::UnorderedDFS(allLayers,
                                           startLayer,
                                           [&](const InferenceEngine::CNNLayerPtr layer) {
                                               LayerInfo info(layer);
                                               if (GNAPluginNS::LayerTypeFromStr(layer->type) == GNAPluginNS::LayerType::NO_TYPE) {
                                                   errMessage = "The plugin does not support layer: " + layer->name + ":" + layer->type + "\n";
                                                   check_result =  false;
                                               }
                                               if (batch_size != 1 && info.isBatchSizeConstrained()) {
                                                   errMessage = "topology with layer: " + layer->name + ", type: " + layer->type +
                                                                ", and batch size(" + std::to_string(batch_size) + ") != 1 not supported";
                                                   check_result =  false;
                                               }
                                               if (info.isFullyConnected()) {
                                                   size_t output_batch_size = info.getOutputBatchSize();
                                                   if (output_batch_size > 8) {
                                                       errMessage = "topology with layer: " + layer->name + ", type: " + layer->type +
                                                           ", and batch size(" + std::to_string(output_batch_size) + ") not supported";
                                                       check_result = false;
                                                   }
                                               } else if (info.isConcat()) {
                                                   if (userWarning && !ValidateConcatAxis(layer, errMessage)) {
                                                       std::cout << errMessage;
                                                   }
                                               }
                                           }, false);
    return check_result;
}
IE_SUPPRESS_DEPRECATED_END

} // namespace GNALimitations
} // namespace GNAPluginNS
