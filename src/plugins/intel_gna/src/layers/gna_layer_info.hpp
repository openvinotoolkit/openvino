// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>
#include <vector>
#include <legacy/ie_layers.h>
#include "caseless.hpp"
#include "ie_algorithm.hpp"
#include "backend/gna_types.h"
#include "gna_permute.hpp"
#include "gna_lib_ver_selector.hpp"
#include "gna_copy_layer.hpp"
#include <legacy/ngraph_ops/power.hpp>
#include <ngraph/opsets/opset8.hpp>
#include "ops/pwl.hpp"
#include "layers/gna_crop_layer.hpp"
#include "backend/gna_limitations.hpp"
#include "transformations/rt_info/gna_transpose_fusable.hpp"

namespace GNAPluginNS {

/**
 * @brief detecting of const pointer for dynamic cast operations
 * @tparam T
 */
template <class T>
struct is_const_pointer : public std::false_type{
};

template <class T>
struct is_const_pointer<const T *> : public std::true_type{
};


/**
 * similar to type traits determined in standard library this trait provides details per layer type, with some attributes specific for GNA
 * we don't need to have compile time performance for this yet
 */
class LayerInfo {
    InferenceEngine::CNNLayer * layer;

#define IS_VALID() if (nullptr == layer) return false

 public:
    explicit LayerInfo(InferenceEngine::CNNLayer & layer)
        : LayerInfo(&layer) {
    }
    explicit LayerInfo(const InferenceEngine::CNNLayerPtr & layer)
        : LayerInfo(layer.get()) {
    }
    explicit LayerInfo(InferenceEngine::CNNLayer * layer)
        : layer(layer) {
    }
    bool hasMultipleInputs() const noexcept {
        IS_VALID();
        return layer->insData.size() > 1;
    }
    // The name of the funciton may be somehwat misleading
    // Explanation: when in low precision mode the listed layers have 8-bit outputs
    // and when in 16-bit input mode, they have 16-bit outputs
    bool has8BOr16BOutput() const noexcept {
        IS_VALID();
        static InferenceEngine::details::caseless_set<std::string> layersWith8BOr16BOutputs = {"memory", "input", "split", "slice", "concat", "copy", "const"};
        return layersWith8BOr16BOutputs.find(layer->type) != layersWith8BOr16BOutputs.end() ||
               isActivation() ||
               (isCrop() && !isCropAffined()) ||
               isPermute();
    }
    bool has32BOutput() const noexcept {
        IS_VALID();
        std::vector<std::function<bool()>> has32BOutputsProbes = {
            [this]() { return isFullyConnected(); },
            [this]() { return isAffineFilter(); },
            [this]() { return isConcatAlignFilter(); },
            [this]() { return isConvolutionFilter(); },
            [this]() { return isEltwise(); },
            [this]() { return isScaleShift(); },
            [this]() { return isConvolution(); },
            [this]() { return isPooling(); },
            [this]() { return isPower(); },
            [this]() { return isCropAffined(); },
            [this]() { return isGemm(); },
        };

        for (auto && has32BOutputs : has32BOutputsProbes) {
            if (has32BOutputs()) {
                return true;
            }
        }
        return false;
    }
    bool isBatchSizeConstrained() {
        static InferenceEngine::details::caseless_set<std::string> layersWithConstrains = {"memory", "convolution"};
        return layersWithConstrains.find(layer->name) != layersWithConstrains.end();
    }
    size_t getOutputBatchSize() const {
        if (!layer) {
            THROW_GNA_EXCEPTION << "layer is null";
        }
        if (!layer->outData[0]) {
            THROW_GNA_EXCEPTION << "output data of layer '" << layer->name << "' is null";
        }
        auto& dims = layer->outData[0]->getDims();
        auto layout = layer->outData[0]->getLayout();
        switch (dims.size()) {
        case 1:
            return 1;
        case 2:
            if (layout == InferenceEngine::Layout::NC) {
                return dims[0];
            } else if (layout == InferenceEngine::Layout::CN) {
                return dims[1];
            } else {
                THROW_GNA_EXCEPTION << "batch size is not define in layer '" << layer->name << "'";
            }
        case 4:
            return dims[0];
        default:
            THROW_GNA_EXCEPTION << "batch size is not define in layer '" << layer->name << "'";
        }
    }

    bool isActivation() const noexcept {
        IS_VALID();
        static InferenceEngine::details::caseless_set<std::string> activations =
            {"clamp",
             "sigmoid",
             "identity",
             "relu",
             "leakyrelu",
             "tanh",
             "prelu",
             "exp",
             "log",
             "sign",
             "abs",
             "neglog",
             "neghalflog",
             "softsign",
             "power",
             "fakequantize",
             "pwl"};

        if (isOfType("power")) {
            auto powerLayer = as<const InferenceEngine::PowerLayer*>();
            return powerLayer != nullptr && powerLayer->power != 1.0f;
        }

        return activations.find(layer->type) != activations.end();
    }

    bool isWeightable() const noexcept {
        auto weigtable_ptr = as<const InferenceEngine::WeightableLayer*>();
        return weigtable_ptr != nullptr;
    }
    bool isConcatAlignFilter() const noexcept {
        return isOfType("ConcatAlignFilter");
    }
    bool isLink() const noexcept {
        return isOfType("Link");
    }
    bool isAffineFilter() const noexcept {
        return isOfType("AffineFilter");
    }
    bool isConvolutionFilter() const noexcept {
        return isOfType("ConvolutionFilter");
    }
    bool isRelu() const noexcept {
        return isOfType("relu");
    }
    bool isConvolution() const noexcept {
        return isOfType("convolution");
    }
    bool isPower() const noexcept {
        if (isOfType("power")) {
            return true;
        }
        std::shared_ptr<ov::intel_gna::op::Pwl> pwl_node;
        if (!layer->getNode() || !(pwl_node = std::dynamic_pointer_cast<ov::intel_gna::op::Pwl>(layer->getNode()))) {
            return false;
        }
        return std::dynamic_pointer_cast<ngraph::op::PowerIE>(pwl_node->get_base_node()) ||
               std::dynamic_pointer_cast<ngraph::opset8::Power>(pwl_node->get_base_node());
    }
    bool has32BInput() const noexcept {
        IS_VALID();
        return isActivation() || isPooling();
    }
    bool isInput() const noexcept {
        return isOfType("input");
    }
    bool isOutput() const noexcept {
        for (auto& out : layer->outData) {
            if (getInputTo(out).empty()) {
                return true;
            }
        }
        return false;
    }
    bool isConst() const noexcept {
        return isOfType("const");
    }
    bool isScaleShift() const noexcept {
        IS_VALID();
        return nullptr != as<const InferenceEngine::ScaleShiftLayer*>();
    }
    bool isSyntheticScaleShift() const noexcept {
        IS_VALID();
        return layer->name.find("SyntheticScaleShift") != std::string::npos;
    }
    bool isEltwise() const noexcept {
        IS_VALID();
        return nullptr != as<const InferenceEngine::EltwiseLayer*>();
    }
    bool isEltwiseSum() const noexcept {
        IS_VALID();
        if (!isEltwise()) return false;
        // dynamic_cast<const InferenceEngine::EltwiseLayer *>(layer) is validated in isEltwise function
        // coverity[var_deref_op]
        return dynamic_cast<const InferenceEngine::EltwiseLayer *>(layer)->_operation ==
               InferenceEngine::EltwiseLayer::Sum;
    }
    bool isEltwiseSub() const noexcept {
        IS_VALID();
        if (!isEltwise()) return false;
        // dynamic_cast<const InferenceEngine::EltwiseLayer *>(layer) is validated in isEltwise function
        // coverity[var_deref_op]
        return dynamic_cast<const InferenceEngine::EltwiseLayer *>(layer)->_operation ==
            InferenceEngine::EltwiseLayer::Sub;
    }

    bool isEltwiseMul() const noexcept {
        IS_VALID();
        if (!isEltwise()) return false;
        // dynamic_cast<const InferenceEngine::EltwiseLayer *>(layer) is validated in isEltwise function
        // coverity[var_deref_op]
        return dynamic_cast<const InferenceEngine::EltwiseLayer*>(layer)->_operation ==
            InferenceEngine::EltwiseLayer::Prod;
    }
    bool isAbs() const noexcept {
        return isOfType("abs");
    }
    bool isIdentity() const noexcept {
        return isOfType("identity");
    }
    bool isTanh() const noexcept {
        return isOfType("tanh");
    }
    bool isSigmoid() const noexcept {
        return isOfType("sigmoid");
    }
    bool isSoftSign() const noexcept {
        return isOfType("softsign");
    }
    bool isClamp() const noexcept {
        return isOfType("clamp");
    }
    bool isFullyConnected() const noexcept {
        return isOfType("FullyConnected") || isOfType("InnerProduct");
    }
    bool isGemm() const noexcept {
        return isOfType("Gemm");
    }
    bool isSplit() const noexcept {
        return isOfType("split");
    }
    bool isSlice() const noexcept {
        return isOfType("slice");
    }
    bool isConcat() const noexcept {
        return isOfType("concat");
    }
    bool isFakeQuantize() const noexcept {
        return isOfType("FakeQuantize");
    }
    bool isNonFunctional() const {
        return isOfType("reshape") || isOfType("squeeze") || isOfType("unsqueeze") || isTrivialPermute();
    }
    bool isReshape() const noexcept {
        return isOfType("reshape");
    }
    bool isPermute() const noexcept {
        return isOfType("permute");
    }
    bool isPermuteFusable() const noexcept {
        return isPermute() && (layer->params.count(ov::intel_gna::rt_info::GNATransposeFusable::get_type_info_static()) > 0);
    }
    bool isPermuteViaReshape() const {
        if (!isOfType("reshape")) return false;

        auto input_dims = layer->insData[0].lock()->getDims();
        auto output_dims = layer->outData[0]->getDims();

        if (input_dims.size() != output_dims.size()) {
            return false;
        }

        input_dims.erase(std::remove(input_dims.begin(), input_dims.end(), 1), input_dims.end());
        output_dims.erase(std::remove(output_dims.begin(), output_dims.end(), 1), output_dims.end());

        if (input_dims != output_dims) {
            return false;
        }
        return true;
    }

    // @brief this not only mathematically trivial, has some WA for kaldi case
    bool isTrivialPermute() const {
        if (!isPermute()) return false;

        if (isPermuteFusable()) return true;

        auto layerOrder = layer->GetParamAsInts("order");
        if (layer->insData.empty()) {
            return false;  // unsupported case
        }
        auto inputs = layer->insData.begin()->lock();
        auto inputsOrder = inputs->getTensorDesc().getDims();

        return GNAPluginNS::isTrivialPermute(std::vector<int64_t>{begin(layerOrder), end(layerOrder)},
            inputsOrder);
    }
    bool isNonValuesChangable() const {
        return isNonFunctional() || isSplit() || isSlice() || isConcat();
    }
    bool isPooling() const noexcept {
        return isOfType("pooling");
    }
    bool isMaxPooling() const noexcept {
        IS_VALID();
        if (!isPooling()) return false;
        return as<const InferenceEngine::PoolingLayer*>()->_type == InferenceEngine::PoolingLayer::MAX;
    }
    bool isMemory() const noexcept {
        return isOfType("memory");
    }
    bool isCrop() const noexcept {
        return isOfType("crop");
    }
    bool isCropAffined() const noexcept {
        auto cropLayer = dynamic_cast<InferenceEngine::CropLayer *> (layer);
        if (cropLayer != nullptr && !cropLayer->offset.empty()) {
            size_t offset;
            std::tie(offset, std::ignore, std::ignore) = GetCropParams(cropLayer);
            return GNAPluginNS::GNALimitations::isCropAffinedOffset(offset);
        }
        return false;
    }
    bool isCopy() const noexcept {
        return isOfType(CopyLayerName) || isOfType(DelayedCopyLayerName);
    }
    bool isCopyDelayed() const noexcept {
        return isOfType(DelayedCopyLayerName);
    }
    bool isWeightableIdentity() const noexcept {
        return isConcatAlignFilter() || isSyntheticScaleShift() || isCropAffined();
    }
    bool isFusableWithConv() const noexcept {
        return isActivation() || isMaxPooling();
    }

    bool isSynthetic() const noexcept {
        return isConcatAlignFilter() || isSyntheticScaleShift() || isConvolutionFilter() || isAffineFilter();
    }

    size_t paddingSize() const {
        static InferenceEngine::details::caseless_set<std::string> layersWithPossiblePadding = {"FullyConnected",
                                                                        "InnerProduct",
                                                                             "Pooling",
                                                                         "Convolution"};
        if (layersWithPossiblePadding.find(layer->type) != layersWithPossiblePadding.end()) {
            size_t size_without_padding = 0;
            IE_ASSERT(!layer->insData.empty());
            auto inputs = layer->insData.begin()->lock();
            if (inputs) {
                size_without_padding = InferenceEngine::details::product(begin(inputs->getTensorDesc().getDims()),
                                                                         end(inputs->getTensorDesc().getDims()));
            }
            return ALIGN(size_without_padding, 8) - size_without_padding;
        }
        return 0;
    }
    template <class T>
    typename std::enable_if<!is_const_pointer<T>::value, T>::type as() noexcept {
        return dynamic_cast<T>(layer);
    }
    template <class T>
    typename std::enable_if<is_const_pointer<T>::value, T>::type as() const noexcept {
        return dynamic_cast<T>(layer);
    }
    operator InferenceEngine::CNNLayer *() noexcept {
        return layer;
    }
    operator const InferenceEngine::CNNLayer *() const noexcept {
        return layer;
    }
    operator InferenceEngine::CNNLayerPtr () const noexcept {
        return std::shared_ptr<InferenceEngine::CNNLayer>(layer, [] (InferenceEngine::CNNLayer * p) {});
    }

 protected:
    bool isOfType(const std::string & type) const noexcept {
        IS_VALID();
        return InferenceEngine::details::CaselessEq<std::string>()(layer->type, type);
    }
    #undef IS_VALID
};

inline std::ostream & operator <<(std::ostream &os, const LayerInfo & info) {
    os << static_cast<const InferenceEngine::CNNLayer*>(info)->name;
    return os;
}

}  // namespace GNAPluginNS
