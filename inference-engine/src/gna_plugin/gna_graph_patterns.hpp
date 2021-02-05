// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <legacy/graph_tools.hpp>
#include <legacy/details/ie_cnn_network_tools.h>
#include "gna_data_types.hpp"
#include "gna_graph_tools.hpp"
#include "gna_plugin_log.hpp"
#include "gna_upstream_iterator.hpp"
#include "layers/gna_layer_info.hpp"

namespace GNAPluginNS {

/**
 * @brief searchs for a pattern: Permute(0,3,1,2) -> ... -> Convolution -> ... -> Permute(0,2,3,1) or
 *        Reshape -> ... -> Convolution -> ... -> Permute(0,2,3,1) if Convolution has only one input dimension not equal to 1
 * @param layer convolution layer
 * @return the found permutations before and after convolution
 */
inline std::pair<InferenceEngine::CNNLayerPtr, InferenceEngine::CNNLayerPtr> FindPermutationsAroundConvolutionInNHWCModel(
    InferenceEngine::CNNLayerPtr layer) {
    // Skip a convolution which doesn't have previous or next layers
    if (layer->outData.size() != 1) {
        return std::make_pair(nullptr, nullptr);
    }

    if (getInputTo(layer->outData.front()).empty()) {
        return std::make_pair(nullptr, nullptr);
    }

    if (!InferenceEngine::CNNNetHasPrevLayer(layer.get())) {
        return std::make_pair(nullptr, nullptr);
    }

    auto next = getInputTo(layer->outData.front()).begin()->second;
    // Permute is inserted before Reshape by MO in NHWC models, so we need to find either permute, or reshape, or output
    while (!LayerInfo(next).isPermute() && !LayerInfo(next).isNonFunctional() && !LayerInfo(next).isOutput() &&
           next->outData.size() == 1) {
        auto input_to = getInputTo(next->outData.front());
        if (input_to.size() != 1) break;
        next = input_to.begin()->second;
    }

    // Check if the found layer is NCHW to NHWC permute, if it's not just skip this convolution
    if (!LayerInfo(next).isPermute() || next->input()->getLayout() != InferenceEngine::Layout::NCHW ||
        next->GetParamAsInts("order") != GetPermuteOrder(InferenceEngine::Layout::NCHW, InferenceEngine::Layout::NHWC)) {
        return std::make_pair(nullptr, nullptr);
    }

    // Permute is inserted after Reshape by MO in NHWC models, so we need to find either permute, or reshape, or input
    auto parent = InferenceEngine::CNNNetPrevLayer(layer);
    auto prev = parent;
    while (!LayerInfo(prev).isPermute() && !LayerInfo(prev).isNonFunctional() && !LayerInfo(prev).isInput() &&
           InferenceEngine::CNNNetHasPrevLayer(prev.get())) {
        prev = InferenceEngine::CNNNetPrevLayer(prev);
    }
    // Check if the found layer is NHWC to NCHW permute or have 1D data, if it's not just skip this convolution
    if (LayerInfo(prev).isPermute()) {
        if (prev->outData[0]->getLayout() != InferenceEngine::Layout::NCHW ||
            prev->GetParamAsInts("order") != GetPermuteOrder(InferenceEngine::Layout::NHWC, InferenceEngine::Layout::NCHW)) {
            return std::make_pair(nullptr, nullptr);
        }
    } else  {
        if (parent->outData.size() != 1 || InferenceEngine::getInputTo(parent->outData[0]).size() != 1) {
            return std::make_pair(nullptr, nullptr);
        }
        auto parent_dims = parent->outData[0]->getDims();
        // Check if the previous layer has all dimensions except one to be equal to 1
        if (std::count_if(std::begin(parent_dims), std::end(parent_dims), [](size_t dim) { return dim != 1; }) > 1) {
            return std::make_pair(nullptr, nullptr);
        }
    }
    return std::make_pair(prev, next);
}

/**
 * @brief searches for a pattern Convolution -> ... -> Permute(0,3,2,1) -> ... -> ScaleShift | FullyConnected
 * @param layer convolution layer
 * @return the found permutation layer
 */
inline InferenceEngine::CNNLayerPtr FindPermutationAfterConvolutionInKaldiModel(InferenceEngine::CNNLayerPtr layer) {
    // Skip a convolution which doesn't have next layers
    if (layer->outData.size() != 1) {
        return nullptr;
    }

    if (getInputTo(layer->outData.front()).empty()) {
        return nullptr;
    }

    /* Permute is inserted between a convolution and a scaleshift|fullyconnected layer by MO in Kaldi models,
     * so we need to fing either permute, or fullyconnected, or scaleshift, or output, or reshape to 2D
     */
    auto next = getInputTo(layer->outData.front()).begin()->second;
    while (!LayerInfo(next).isPermute() && !LayerInfo(next).isFullyConnected() && !LayerInfo(next).isScaleShift() &&
           !LayerInfo(next).isOutput() &&
           (!LayerInfo(next).isNonFunctional() || next->outData[0]->getDims().size() == next->input()->getDims().size())) {
        next = getInputTo(next->outData.front()).begin()->second;
    }

    // Check if the found layer is NCHW to NWHC permute
    if (!LayerInfo(next).isPermute() || next->input()->getLayout() != InferenceEngine::Layout::NCHW ||
        next->GetParamAsInts("order") != std::vector<int>{0, 3, 2, 1}) {
        return nullptr;
    }

    return next;
}

/**
 * @brief identifies if a model must be converted to NHWC, it must not be neither NHWC, nor Kaldi
 * @param layers model sorted layers
 */
inline bool MustBeConvertedFromNCHWToNHWC(const std::vector<InferenceEngine::CNNLayerPtr> &layers) {
    for (auto& l : layers) {
        if (!LayerInfo(l).isConvolution()) continue;

        InferenceEngine::CNNLayerPtr next;
        std::tie(std::ignore, next) = FindPermutationsAroundConvolutionInNHWCModel(l);
        if (next != nullptr) return false;
        // If a convolution has only 1-dimension input and output we should skip it
        auto in_dims = l->insData.begin()->lock()->getDims();
        auto out_dims = l->outData.front()->getDims();
        if (std::count_if(std::begin(in_dims), std::end(in_dims), [](size_t dim) { return dim != 1; }) <= 1 &&
            std::count_if(std::begin(out_dims), std::end(out_dims), [](size_t dim) { return dim != 1; }) <= 1) {
            continue;
        }

        return FindPermutationAfterConvolutionInKaldiModel(l) == nullptr;
    }
    return false;
}

/**
 * @brief returns rotation information for a layer based on the previous convolution or pooling dimensions order
 * @param layer layer from which rotation info search must be started
 * @return bool value which identifies if rotation info is found and rotation information
 */
inline std::vector<TranspositionInfo> FindTranspositionInfoFromPrevLayers(InferenceEngine::CNNLayerPtr layer) {
    std::function<std::vector<TranspositionInfo>(InferenceEngine::CNNLayerPtr)> findTranspositionInfoRecursive =
        [&findTranspositionInfoRecursive](InferenceEngine::CNNLayerPtr layer) -> std::vector<TranspositionInfo> {
        if (LayerInfo(layer).isSplit()) {
            THROW_GNA_EXCEPTION << layer->name << " Failed to find transposition info";
        }

        if (LayerInfo(layer).isConvolution() || LayerInfo(layer).isPooling()) {
            auto out_dims = layer->outData[0]->getDims();
            return {{true, out_dims[1], out_dims[2] * out_dims[3]}};
        }

        /* If a fullyconnected or input layers are reached, it means that transposition isn't needed, but we should keep
         * its output size to skip this part during transposition if transposed layer is a result of concatination */
        if (LayerInfo(layer).isFullyConnected() || LayerInfo(layer).isInput()) {
            auto out_dims = layer->outData[0]->getDims();
            return {{false, 1, InferenceEngine::details::product(std::begin(out_dims) + 1, std::end(out_dims))}};
        }

        // If an eltwise is reached we should follow only one not-const direction
        if (LayerInfo(layer).isEltwise()) {
            auto input1 = InferenceEngine::CNNNetPrevLayer(layer, 0);
            auto input2 = InferenceEngine::CNNNetPrevLayer(layer, 1);
            if (LayerInfo(input1).isConst()) return findTranspositionInfoRecursive(input2);
            return findTranspositionInfoRecursive(input1);
        }

        std::vector<TranspositionInfo> transpositionInfo;
        for (int idx = 0; idx < layer->insData.size(); ++idx) {
            if (!InferenceEngine::CNNNetHasPrevLayer(layer.get(), idx)) continue;
            auto inputLayer = InferenceEngine::CNNNetPrevLayer(layer, idx);
            // If a concat input is a const we should keep its size to skip this part during transposition
            if (LayerInfo(layer).isConcat() && LayerInfo(inputLayer).isConst()) {
                auto in_dims = layer->insData[idx].lock()->getDims();
                auto data_size = InferenceEngine::details::product(std::begin(in_dims) + 1, std::end(in_dims));
                transpositionInfo.push_back({false, 1, data_size});
            } else {
                std::vector<TranspositionInfo> results = findTranspositionInfoRecursive(inputLayer);
                transpositionInfo.insert(std::end(transpositionInfo), std::begin(results), std::end(results));
            }
        }
        return transpositionInfo;
    };
    return findTranspositionInfoRecursive(layer);
}

/**
 * @brief returns rotation information for a layer based on the next convolution layer dimensions order
 * @param layer layer from which rotation info search must be started
 * @return bool value which identifies if rotation info is found and rotation information
 */
inline std::vector<TranspositionInfo> FindTranspositionInfoFromNextLayers(InferenceEngine::CNNLayerPtr layer) {
    std::function<std::vector<TranspositionInfo>(InferenceEngine::CNNLayerPtr)> findTranspositionInfoRecursive =
        [&findTranspositionInfoRecursive](InferenceEngine::CNNLayerPtr layer) -> std::vector<TranspositionInfo> {
        if (LayerInfo(layer).isConcat()) return {};

        if (LayerInfo(layer).isConvolution()) {
            auto in_dims = layer->input()->getDims();
            return {{true, in_dims[1], in_dims[2] * in_dims[3]}};
        }

        /* If a fullyconnected or output layers are reached, it means that transposition isn't needed, but we should keep
         * its input size to skip this part during transposition if transposed layer is splitting */
        if (LayerInfo(layer).isFullyConnected() || LayerInfo(layer).isOutput()) {
            auto in_dims = layer->input()->getDims();
            return {{false, 1, InferenceEngine::details::product(std::begin(in_dims) + 1, std::end(in_dims))}};
        }

        std::vector<TranspositionInfo> transpositionInfo;
        for (const auto &output : layer->outData) {
            if (getInputTo(output).empty()) continue;
            std::vector<TranspositionInfo> results;
            // Return transposition info from the first branch where convolution is found
            for (const auto &inputTo : getInputTo(output)) {
                results = findTranspositionInfoRecursive(inputTo.second);
                auto found = std::find_if(std::begin(results), std::end(results), [](const TranspositionInfo & result) {
                    return result.transpose;
                });
                if (found != std::end(results)) break;
            }
            if (results.empty()) {
                THROW_GNA_EXCEPTION << layer->name << " Failed to find transposition info";
            }
            transpositionInfo.insert(std::end(transpositionInfo), std::begin(results), std::end(results));
        }
        return transpositionInfo;
    };

    return findTranspositionInfoRecursive(layer);
}

} // namespace GNAPluginNS