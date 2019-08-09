// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_plugin_policy.hpp"
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <algorithm>
#include <list>
#include <unordered_set>
#include <set>
#include <fstream>

#include <quantization/quantized_layer_params.hpp>
#include <graph_tools.hpp>
#include <gna-api.h>
#include <blob_factory.hpp>
#include <ie_memcpy.h>
#include <ie_algorithm.hpp>
#include <details/ie_cnn_network_tools.h>
#include <ie_util_internal.hpp>
#include <iomanip>

#include "gna_pass_manager.hpp"
#include "gna_layer_info.hpp"
#include "gna_plugin_log.hpp"
#include "gna_upstream_iterator.hpp"
#include "net_pass.h"

using namespace InferenceEngine;
using namespace InferenceEngine::details;
using namespace GNAPluginNS;

std::shared_ptr<IPassManager> BasePass::getPassManager() {
    auto sharedMgr = mgr.lock();
    if (!sharedMgr) {
        THROW_GNA_EXCEPTION << getName() << ": cannot get PassManager object";
    }
    return sharedMgr;
}

// indexes stored in pass manager
static const char diagonalLayersCounterName[] = "diagonalLayerCounter";
static const char copyLayersCounter[] = "numCopyLayers";

/**
 * @brief helper injections of diagonal layer with certain value
 */
static void insertDiagonalLayerBetween(InferenceEngine::CNNLayerPtr prevLayer,
                                       InferenceEngine::CNNLayerPtr nextLayer,
                                       std::shared_ptr<IPassManager> passmanager,
                                       float fillValue) {
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(prevLayer);
    auto diagName = std::string("SyntheticScaleShift_") + std::to_string(passmanager->getIntVar(diagonalLayersCounterName)++);
    gnalog() << "Inserted Diagonal Layer " << diagName <<" between: " << prevLayer->name << " and " << nextLayer->name << "\n" << std::flush;

    auto diagLayer = std::make_shared<ScaleShiftLayer>(LayerParams({diagName, "ScaleShift", Precision::FP32}));

    // TODO: diagonal size
    std::vector<float> weightsValues(nextLayer->outData[0]->dims[0], fillValue);
    diagLayer->_weights = make_shared_blob<float>(nextLayer->outData[0]->precision, Layout::C, weightsValues);
    auto newDims = nextLayer->outData[0]->getDims();
    auto dataPtr = std::make_shared<Data>(diagName,
                                          TensorDesc(nextLayer->outData[0]->precision,
                                                     newDims,
                                                     nextLayer->outData[0]->layout));
    auto diagonalWithQuant = quantized ?
                             InferenceEngine::injectData<QuantizedLayerParams>(diagLayer) : diagLayer;

    dataPtr->creatorLayer = diagonalWithQuant;
    diagonalWithQuant->outData.push_back(dataPtr);

    // actual insertion
    CNNNetworkInsertLayer(prevLayer, nextLayer, diagonalWithQuant);
}

/**
 * @brief copy layer inserted by several passes
 * @returns pointer to newly created COPYLayer
 */
static CNNLayerPtr InsertCopyLayer(CNNLayerPtr prevLayer, CNNLayerPtr nextLayer, int beforeIdx, std::shared_ptr<IPassManager> passmanager) {
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(prevLayer);
    std::string copyName = std::string("copy_") + std::to_string(passmanager->getIntVar(copyLayersCounter)++);
    gnalog() << "Inserted " << copyName << " between: " << prevLayer->name << " and " << nextLayer->name << std::endl;

    CNNLayerPtr copyLayer = std::make_shared<GenericLayer>(LayerParams({copyName, "Copy", Precision::FP32}));

    auto inputData = nextLayer->insData[beforeIdx].lock();
    auto newDims = inputData->getDims();
    auto dataPtr = std::make_shared<Data>(copyName,
                                          TensorDesc(inputData->precision,
                                                     inputData->getDims(),
                                                     inputData->layout));

    auto copyWithQuant = quantized ?
                         InferenceEngine::injectData<QuantizedLayerParams>(copyLayer) :
                         copyLayer;
    dataPtr->creatorLayer = copyWithQuant;
    copyWithQuant->outData.push_back(dataPtr);
    CNNNetworkInsertLayer(prevLayer, nextLayer, copyWithQuant);
    return copyWithQuant;
}

static std::vector<CNNLayerPtr> getCandidatesForIdentityInsertion(const CNNLayerPtr l) {
    std::vector<CNNLayerPtr> prevLayers;

    // skipping memory inputs and true inputs layers
    if (l->insData.empty()) return {};

    auto eltwise = dynamic_cast<InferenceEngine::EltwiseLayer *>(l.get());
    auto concat = dynamic_cast<InferenceEngine::ConcatLayer *>(l.get());

    // eltwise
    if (eltwise != nullptr) {
        // eltwise layer has 2 inputs, so depends on situation identity should or should not be inserted

        // for  sum if we have 4-4 inputs we will handle that by inserting identity activation case (1)
        // for  sum if we have 4-2 - OK
        // for  sum if we have 2-2 inputs we need to insert diagonal

        // for  mul if we have 2-2 - OK
        // for  mul if we have 2-4 - inputs we need to insert identity activation to make 2 bytes input
        // for  mul if we have 4-4 - there 2 options
        //          option 1 both inputs came from single outdata  - we will insert 1 identity  to just convert single input into 2 bytes
        //          option 2 each input came from it's own outdata - we need to insert 2 identities activations to convert both and feed weights and inputs

        auto prev0 = CNNNetPrevLayer(l, 0);
        auto prev1 = CNNNetPrevLayer(l, 1);
        switch (eltwise->_operation) {
            case EltwiseLayer::Sum:
                if (!LayerInfo(prev0).has32BOutput() || !LayerInfo(prev1).has32BOutput()) {
                    return prevLayers;
                }
                // TODO: whether there are possibility to select after what layer identity gets inserted
                prevLayers.push_back(prev0);
                break;
            case EltwiseLayer::Prod: {
                if (LayerInfo(prev0).has16BOutput() && LayerInfo(prev1).has16BOutput()) {
                    return prevLayers;
                }

                if (LayerInfo(prev0).has32BOutput()) {
                    prevLayers.push_back(prev0);
                }

                // if layers of outdata are different
                auto prevData0 = l->insData[0].lock();
                auto prevData1 = l->insData[1].lock();

                if ((prev0 != prev1 || prevData0 != prevData1) && LayerInfo(prev1).has32BOutput()) {
                        prevLayers.push_back(prev1);
                }

                break;
            }
            default :
                THROW_GNA_EXCEPTION << "Eltwise Layer of type: " << eltwise->_operation << " not supported";
        }
    } else if (concat != nullptr) {
        for (int i = 0; CNNNetHasPrevLayer(l.get(), i); ++i) {
            auto prev = CNNNetPrevLayer(l, i);
            if (LayerInfo(prev).has32BOutput()) {
                prevLayers.push_back(prev);
            }
        }
    } else {
        // not eltwise or concat
        // other layers has 1 inputs - situation is easier
        // ex. activation or pooling - no need to insert identity activation.
        if (LayerInfo(l).has32BInput())
            return prevLayers;

        auto prevLayer = CNNNetPrevLayer(l);
        if (!LayerInfo(prevLayer).has32BOutput())
            return prevLayers;

        prevLayers.push_back(prevLayer);
    }
    return prevLayers;
}

void InsertDiagonalLayerPass::run() {
    int numOfDiagLayers = 0;
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(pLayers->front());
    for (auto & l : *pLayers) {
        if (l->insData.empty()) continue;
        auto prevLayer = CNNNetPrevLayer(l);
        if (LayerInfo(l).isActivation()) {
            if (LayerInfo(prevLayer).has32BOutput()) {
                continue;
            }
        } else {
            auto eltwise = dynamic_cast<InferenceEngine::EltwiseLayer *>(l.get());
            if (!eltwise) {
                continue;
            }
            // in case of eltwise sum one of input would be 4 bytes one - 2
            // in case of eltwise mull one of input would be 2 bytes one - 2
            // for e sum if we have 4-4 inputs we will handle that by inserting identity activation
            // for e sum if we have 4-2 - OK
            // for e sum if we have 2-2 inputs we need to insert diagonal -- handling here
            // for e mul if we have 2-2 - OK
            // for e mul if we have 2-4 - inputs we need to insert identity to put 4 bytes input into weights
            // for e mul if we have 4-4 - inputs we need to insert 2 identities to put both 4 bytes input into weights

            if (eltwise->_operation != EltwiseLayer::Sum)
                continue;

            auto prevLayer1 = CNNNetPrevLayer(l, 1);
            if (!LayerInfo(prevLayer).has16BOutput() || !LayerInfo(prevLayer1).has16BOutput())
                continue;
        }
        insertDiagonalLayerBetween(prevLayer, l, getPassManager(), 1.f);
    }
}

void HandleMultipleActivationsForTheLayerPass::run() {
    // found layer followed by with multiple activations
    for (auto & l : *pLayers) {
        std::set<CNNLayerPtr> activations;
        std::set<CNNLayerPtr> identities;

        for (auto && odata : l->outData) {
            for (auto && inputTo : odata->getInputTo()) {
                LayerInfo info(inputTo.second);

                if (info.isIdentity()) {
                    identities.insert(inputTo.second);
                } else if (info.isActivation()) {
                    activations.insert(inputTo.second);
                }
            }
        }
        // single or not activations case
        if (activations.size() + identities.size() < 2) continue;

        // insert diagonals, but not for identity activations
        for (auto && activation : activations) {
            insertDiagonalLayerBetween(l, activation, getPassManager(), 0.0f);
        }
    }
}

void ReorderMaxPoolPass::run() {
    // detecting following pattern
    // conv->relu->maxpooling
    // changing it to conv->maxpooling->relu
    for (auto & l : *pLayers) {
        auto pool = LayerInfo(l);
        if (!pool.isMaxPooling()) continue;

        // checking prev layer type
        auto activation = LayerInfo(CNNNetPrevLayer(l));
        if (!activation.isActivation()) continue;

        // if activation came from convolution
        auto convolution = LayerInfo(CNNNetPrevLayer(static_cast<InferenceEngine::CNNLayer*>(activation)));
        if (!convolution.isConvolution()) continue;

        gnalog() << "MaxPooling: " << pool << ", reordered with activation: " << activation << "\n";

        CNNNetSwapLayers(activation, pool);
    }
}

void SubstitutePReluPass::run() {
    auto getScale = [](CNNLayer* layer) {
        auto powerCandidate = LayerInfo(layer);
        if (!powerCandidate.isPower()) return 0.0f;
        auto power = powerCandidate.as<PowerLayer*>();

        return power->power == 1 && power->offset == 0.0f ? power->scale : 0.0f;
    };

    auto isScale = [getScale](CNNLayer* layer) {
        return getScale(layer) != 0.0f;
    };

    auto isNegate = [getScale](CNNLayer* layer) {
        return getScale(layer) == -1.0f;
    };

    auto getNext = [](CNNLayer* layer) {
        CNNLayer* next = nullptr;
        if (layer == nullptr) return next;
        if (layer->outData.size() != 1) return next;
        return layer->outData[0]->inputTo.begin()->second.get();
    };

    // TODO: unit tests for bad cases
    for (auto & l : *pLayers) {
        // assume l is starting layer, that is followed by eltwise_sum(relu, negate/relu/scale/negate)
        if (l->outData.size() != 1) continue;
        auto &outputLayers = l->outData[0]->inputTo;
        if (outputLayers.size() != 2) continue;

        // one of followed layers need to be generic relu
        auto first = LayerInfo(outputLayers.begin()->second);
        auto second = LayerInfo((++outputLayers.begin())->second);

        auto relu1 = outputLayers.begin()->second;
        auto neg1 = (++outputLayers.begin())->second;
        if (second.isRelu()) {
            std::swap(first, second);
            std::swap(relu1, neg1);
        }
        if (!first.isRelu()) continue;
        // now we have relu as first layer, lets check second
        // negate
        if (!isNegate(neg1.get())) continue;

        // relu
        auto relu2 = getNext(second);
        if (!LayerInfo(relu2).isRelu()) continue;

        // scale
        auto scale = getNext(relu2);
        if (!isScale(scale)) continue;

        // negate2
        auto negate = getNext(scale);
        if (!isNegate(negate)) continue;

        // sum
        auto sum = getNext(negate);
        if (!LayerInfo(sum).isEltwiseSum()) continue;
        if (sum->insData.size() != 2) continue;

        auto s1 = sum->insData[0].lock()->creatorLayer.lock().get();
        auto s2 = sum->insData[1].lock()->creatorLayer.lock().get();

        if (s1 != static_cast<InferenceEngine::CNNLayer *>(first) &&
            s2 != static_cast<InferenceEngine::CNNLayer *>(first)) {
            continue;
        }

        // hurray we found parametric relu group - dont know what to do with it though
        gnalog() << "PRelu with negative slope of " << -LayerInfo(scale).as<PowerLayer*>()->scale << " found" << std::endl;

        // removing all layers references except of relu layer
        outputLayers.clear();
        outputLayers[relu1->name] = relu1;
        // pointing relu to output of eltwise_summ
        relu1->outData = sum->outData;
        // changing creator layer
        relu1->outData[0]->creatorLayer = relu1;
        // pointing back to relu if any
        if (!relu1->outData[0]->inputTo.empty()) {
            auto summOutputLayer = relu1->outData[0]->inputTo.begin()->second;
            summOutputLayer->insData.clear();
            summOutputLayer->insData.push_back(relu1->outData[0]);
        }

        // changing negative slope
        first.as<ReLULayer*>()->negative_slope = LayerInfo(scale).as<PowerLayer*>()->scale;
    }
}

void ReversePermutationsPass::run() {
    std::function<CNNLayerPtr(CNNLayerPtr, std::function<bool(CNNLayerPtr)>)> prevLayerSkipCertain
        = [&prevLayerSkipCertain](CNNLayerPtr layer, std::function<bool(CNNLayerPtr)> shouldSkip) -> CNNLayerPtr {
        if (CNNNetHasPrevLayer(layer.get())) {
            return nullptr;
        }
        auto prev = CNNNetPrevLayer(layer);

        if (!shouldSkip(prev)) return prevLayerSkipCertain(prev, shouldSkip);

        return prev;
    };

    auto prevLayerSkipReshape = [&prevLayerSkipCertain](CNNLayerPtr layer) -> CNNLayerPtr {
        return prevLayerSkipCertain(layer, [] (CNNLayerPtr l2) {
            return LayerInfo(l2).isReshape();
        });
    };


    std::function<CNNLayerPtr(CNNLayerPtr)> nextLayerSkipReshape = [&nextLayerSkipReshape](CNNLayerPtr layer) -> CNNLayerPtr {
        if (layer->outData.empty()) {
            return nullptr;
        }
        if (layer->outData.front()->inputTo.size() != 1) {
            return nullptr;
        }
        auto next = layer->outData.front()->inputTo.begin()->second;

        if (LayerInfo(next).isReshape()) return nextLayerSkipReshape(next);

        return next;
    };

    auto prevConv = [&prevLayerSkipCertain](CNNLayerPtr layer) -> CNNLayerPtr {
        return prevLayerSkipCertain(layer, [] (CNNLayerPtr l2) {
            return
                LayerInfo(l2).isReshape() ||
                LayerInfo(l2).isPooling() ||
                LayerInfo(l2).isActivation();
        });
    };

    std::unordered_set<std::string> affineWithPermutedWeights;
    std::list<CNNLayerPtr> permutationstoRemove;

    for (auto & l : *pLayers) {
        if (!LayerInfo(l).isPermute()) {
            continue;
        }

        auto layerOrder = l->GetParamAsInts("order");

        if (layerOrder != std::vector<int>({0, 3, 2, 1})) {
            THROW_GNA_EXCEPTION << "Unsupported permute layer: " << l->name << ", order: was " << l->GetParamAsString("order") <<
                               ", but support order is 0,3,2,1";
        }

        // search for it's input convolution
        auto prev = prevConv(l);

        // pooling no used in speech models without convolution
        if (!prev) {
            THROW_GNA_EXCEPTION << "Unsupported permute layer: " << l->name << " no valid input to that layer";
        }

        // we can remove that permutation if it is input to ScaleShift or FC layer
        auto next = nextLayerSkipReshape(l);
        if (!next || !LayerInfo(next).isFullyConnected()) {
            THROW_GNA_EXCEPTION << "Unsupported permute layer: " << l->name << " no valid output of that layer";
        }

        permutationstoRemove.push_back(l);

        // removing that permutation layer and saving information about affine
        affineWithPermutedWeights.insert(next->name);
    }

    for (auto && toRemove : permutationstoRemove) {
        CNNNetworkRemoveLayer(toRemove);
    }

    // search for conv->affine sequences
    for (auto & l : *pLayers) {
        if (!LayerInfo(l).isFullyConnected() || 0 != affineWithPermutedWeights.count(l->name)) {
            continue;
        }
        // found an affine layer that not involved in permutations removing
        // searching whether it has direct input from convolution
        auto prevConvLayer = prevConv(l);
        if (!prevConvLayer) continue;

        auto directPrev = CNNNetPrevLayer(l);

        // TODO : make new permute
        CNNNetworkInsertLayer(l, directPrev, CNNLayerPtr(nullptr));
    }
}

void InsertIdentityLayerPass::run() {
    int numOfIdentityLayers = 0;
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(pLayers->front());
    for (auto & l : *pLayers) {
        for (auto && prev : getCandidatesForIdentityInsertion(l)) {
            // actual insertion
            auto activationName = std::string("identity_") + std::to_string(++numOfIdentityLayers);

            gnalog() << "Inserted "<< activationName << " between: " << prev->name << " and " << l->name << "\n" << std::flush;

            CNNLayerPtr activationLayer =
                std::make_shared<GenericLayer>(LayerParams({activationName, "identity", Precision::FP32}));
            auto inputData = l->insData[0].lock();
            auto newDims = inputData->dims;
            std::reverse(begin(newDims), end(newDims));

            auto dataPtr = std::make_shared<Data>("identity_data_" + std::to_string(numOfIdentityLayers),
                                                  TensorDesc(inputData->precision,
                                                             newDims,
                                                             inputData->layout));
            auto activationLayerWithQuant = quantized ?
                                            InferenceEngine::injectData<QuantizedLayerParams>(activationLayer) :
                                            activationLayer;
            dataPtr->creatorLayer = activationLayerWithQuant;
            activationLayerWithQuant->outData.push_back(dataPtr);
            // wether 1 identity or all outputs TODO possible grouping here, need to implement special groupped inserter
            bool notAll = false;
            for (auto && nextData  : prev->outData) {
                for (auto && nextLayer : nextData->inputTo) {
                    if (nextLayer.second.get() == l.get())
                        continue;
                    if (getCandidatesForIdentityInsertion(nextLayer.second).empty()) {
                        notAll = true;
                    }
                }
            }
            // copy offset - to be used while connecting outputs
            if (prev->params.find("output_offset") != prev->params.end()) {
                activationLayerWithQuant->params["output_offset"] = prev->params["output_offset"];
            }
            // copy offset - to be used while connecting outputs
            if (prev->params.find("original_num_rows") != prev->params.end()) {
                activationLayerWithQuant->params["original_num_rows"] = prev->params["original_num_rows"];
            }

            CNNNetworkInsertLayer(prev, notAll ? l : CNNLayerPtr(nullptr), activationLayerWithQuant);
        }
    }
}

// give previous layers while skipping certain layer according to expression
template <class T>
std::vector<CNNLayerPtr> CNNNetGetPrevLayersSkip(CNNLayerPtr origin, const T &acceptanceCriteria, int idx = -1) {
    std::vector<CNNLayerPtr> prevLayers;
    for (int i = idx == -1 ? 0 : idx; CNNNetHasPrevLayer(origin.get(), i) && (idx == -1 || i == idx); i++) {
        auto prevLayer = CNNNetPrevLayer(origin, i);
        if (acceptanceCriteria(prevLayer)) {
            prevLayers.push_back(prevLayer);
        } else {
            // if for some input we need to look in upper layers - original index not used here intentionally
            auto prevPrevLayers = CNNNetGetPrevLayersSkip(prevLayer, acceptanceCriteria);
            prevLayers.insert(prevLayers.end(), prevPrevLayers.begin(), prevPrevLayers.end());
        }
    }

    return prevLayers;
}

void InsertCopyLayerPass::run() {
    for (auto & l : *pLayers) {
        if (l->insData.empty()) continue;
        auto prevLayers = CNNNetGetPrevLayersSkip(l, [](CNNLayerPtr origin){
            return !LayerInfo(origin).isReshape();
        });

        for (int i=0; i != prevLayers.size(); i++) {
            auto & prevIndirectLayer = prevLayers[i];
            if ((LayerInfo(l).isMemory() && LayerInfo(prevIndirectLayer).isConcat()) ||
                (LayerInfo(l).isConcat() && LayerInfo(prevIndirectLayer).isCrop())) {
                if (LayerInfo(prevIndirectLayer).isCrop()) {
                    auto cropLayer = LayerInfo(prevIndirectLayer).as<CropLayer *>();
                    size_t cropOffset = cropLayer->offset.back() * cropLayer->precision.size();
                    if (ALIGN(cropOffset, 8) != cropOffset) {
                        // The crop will be replaced by affine.
                        // Copy layer insertion is not required
                        continue;
                    }
                }
                auto prevLayer = CNNNetPrevLayer(l, i);
                InsertCopyLayer(prevLayer, l, i, getPassManager());
            }
        }
    }
}

void InsertConcatAligningFilterPass::run() {
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(pLayers->front());
    // aligning specific not required in fp32 mode
    if (getPassManager()->getPolicy().ConcatAlignmentPolicy == Policy::ConcatAlignment::DISABLED_FOR_FP32 && !quantized) {
        return;
    }
    // currently concat layer only supports 2 bytes in int16 and int8 mode. In fp32 mode this no necessary but usefull for testing
    const int bytesPerConcatElement = 2;

    int numOfFilterLayers = 0;

    for (auto & l : *pLayers) {
        LayerInfo info(l);
        if (!info.isConcat()) continue;
        uint32_t offset = 0;
        auto concatLayer = info.as<ConcatLayer*>();

        for (auto && input : concatLayer->insData) {
            auto concatInput = input.lock();
            if (!concatInput) {
                THROW_GNA_EXCEPTION << "cannot get insdata for layer: " << l->name;
            }
            auto dims = concatInput->getDims();
            auto outputSize = details::product(++dims.begin(), dims.end());

            // correcting offset by copy layer insertion. This can be improved by collapsing copy and affine or diagonal later-on
            if (ALIGN64(offset) != offset) {
                auto prevLayer = concatInput->getCreatorLayer().lock();
                // input layer parameters are copied not using GNA-primitives - so nothing to allign here.
                if (LayerInfo(prevLayer).isInput()) continue;

                gnalog() << "Inserted Concat Aligning Layer between: " << prevLayer->name << " and " << l->name << std::endl;

                // insert the filter
                auto filterName = std::string("ConcatAlignFilter_") + std::to_string(numOfFilterLayers++);
                auto concatAligningFilter =
                    std::make_shared<WeightableLayer>(LayerParams({filterName, "ConcatAlignFilter", Precision::FP32}));

                if (dims.size() != 2) {
                    THROW_GNA_EXCEPTION << "unsupported concat input of dims.size()=" << dims.size() << ", layer=" << prevLayer->name;
                }

                auto num_rows_in = dims[1];
                size_t aligned64_offset = std::max(0, static_cast<int>(ALIGN64(offset) - 64));
                size_t num_rows_padded = (offset - aligned64_offset) / bytesPerConcatElement;
                size_t num_rows_out = num_rows_padded + num_rows_in;

                // encodes offset to beginning of split layer input
                concatAligningFilter->params["output_offset"] =
                    std::to_string((aligned64_offset / bytesPerConcatElement) * (quantized ? bytesPerConcatElement : 4));
                // encodes original output size
                concatAligningFilter->params["original_num_rows"] = std::to_string(num_rows_in);

                std::vector<float> filterWeights(num_rows_out * num_rows_in, 0.f);

                auto identityIdx = num_rows_padded * num_rows_in;
                for (int i = 0; i != num_rows_in; i++) {
                    filterWeights[identityIdx] = 1.0f;
                    identityIdx += num_rows_in + 1;
                }

                concatAligningFilter->_weights = make_shared_blob<float>(concatInput->precision, Layout::C, filterWeights);

                // modifying output rows to be used - to avoid modification to original concat we are store num of elements in params
                dims[1] = num_rows_out;

                auto outData = std::make_shared<Data>(filterName,
                                                      TensorDesc(concatInput->precision,
                                                                 dims,
                                                                 concatInput->layout));

                auto filterWithQuant = quantized ?
                                       InferenceEngine::injectData<QuantizedLayerParams>(concatAligningFilter) :
                                       concatAligningFilter;
                outData->creatorLayer = filterWithQuant;
                filterWithQuant->outData.push_back(outData);

                CNNNetworkInsertLayer(prevLayer, l, filterWithQuant);
            }
            offset += outputSize * bytesPerConcatElement;
        }
    }
}



void ReorderConcatInputsPass::run() {
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(pLayers->front());
    // aligning specific not required in fp32 mode
    if (getPassManager()->getPolicy().ConcatAlignmentPolicy == Policy::ConcatAlignment::DISABLED_FOR_FP32 && !quantized) {
        return;
    }
    int numOfLinkLayers = 0;

    for (auto & l : *pLayers) {
        // 1st stage locate concat align filter
        LayerInfo info(l);
        if (!info.isConcatAlignFilter()) continue;

        // 2rd locating concat
        if (l->outData.size() != 1) {
            THROW_GNA_EXCEPTION << "no concat layer after concat aligning layer" << l->name;
        }
        auto nextLayers = l->outData.front()->getInputTo();

        if (nextLayers.size() != 1) {
            THROW_GNA_EXCEPTION << "Invalid concat connection in align filter : " << l->name;
        }
        auto concat = nextLayers.begin()->second;
        if (!LayerInfo(concat).isConcat()) {
            THROW_GNA_EXCEPTION << "no concat layer after concat-aligning layer" << l->name << ", but was: " << concat->type;
        }
        // 3stage locate first input in concat
        if (concat->insData.size() != 2) {
            THROW_GNA_EXCEPTION << "unsupported concat layer: " << concat->name;
        }
        auto inputsToConcatFirst = CNNNetGetPrevLayersSkip(concat, [](CNNLayerPtr origin){
            return !LayerInfo(origin).isReshape();
        }, 0);

        if (inputsToConcatFirst.empty()) {
            THROW_GNA_EXCEPTION << "cannot locate first input into concat layer: " << l;
        }

        auto firstInputToConcat = inputsToConcatFirst.front();

        bool needExtraCopy = false;
        // making a link activation possible without extra layer if first input to concat not a parent / indirect parent of second input
        // using ufs - upper first search
        gnalog() << "[UFS] searching for: "<< firstInputToConcat->name << "\n";

        CNNNetDFS(l, [&l, &firstInputToConcat, &needExtraCopy](CNNLayerPtr layer) {
            gnalog() << "[UFS] from : "<< l->name <<" reached: " << layer->name << "\n";
            // found that direct input to concat is a indirect parent of align filter - so no link required
            if (layer.get() == firstInputToConcat.get()) {
                gnalog() << "[UFS] copy layer insertion needed\n";
                needExtraCopy = true;
            }
        }, true, [&needExtraCopy](InferenceEngine::CNNLayer* from) {
            // aborting UFS once link not need
            return make_upstream_order(!needExtraCopy ? from : nullptr);
        });

        // need physically copy data one more time into concat first element
        if (needExtraCopy) {
            auto firstDirectInputToConcat = CNNNetPrevLayer(concat, 0);
            firstInputToConcat = InsertCopyLayer(firstDirectInputToConcat, concat, 0, getPassManager());
        }

        auto linkName = std::string("link_") + std::to_string(numOfLinkLayers++);

        auto linkWithoutQuant = std::make_shared<CNNLayer>(LayerParams({linkName, "link", Precision::FP32}));

        auto link = quantized ?
                    InferenceEngine::injectData<QuantizedLayerParams>(linkWithoutQuant) :
                    linkWithoutQuant;


        auto linkOutData = std::make_shared<Data>(linkName,
                                              TensorDesc(Precision::FP32,
                                                         {1},
                                                         Layout::C));
        linkOutData->getCreatorLayer() = link;

        link->outData.push_back(linkOutData);
        link->insData.push_back(l->outData.front());

        linkOutData->getInputTo()[firstInputToConcat->name + ".via.link"] = firstInputToConcat;
        firstInputToConcat->insData.push_back(linkOutData);

        l->outData.front()->getInputTo()[linkName] = link;
    }
}

void InsertSplitAligningFilterPass::run() {
    // currently split layer only supports 2 bytes in int16 and int8 mode. In fp32 mode this no necessary but usefull for testing
    const int bytesPerSplitElement = 2;
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(pLayers->front());

    int numOfFilterLayers = 0;
    for (auto &l : *pLayers) {
        auto info = LayerInfo(l);
        if (!info.isSplit() && !info.isSlice()) {
            continue;
        }

        size_t currentOffset = 0;
        int splitOutIndex = 0;
        for (auto &&splitOutput  : l->outData) {
            auto outputSize = product(++begin(splitOutput->getDims()), end(splitOutput->getDims()));

            if (currentOffset != ALIGN64(currentOffset)) {
                // this split output not beginning from 64 bytes aligned boundary - need to correct by aligning filter layer
#ifdef PLOT
                // getting list of layers attached to current split output
                gnalog() << "Inserted Affine Filter Layer between: " << l->name << " and ";
                for (auto &&followingLayers : splitOutput->getInputTo()) {
                    if (splitOutput->getInputTo().size() != 1) {
                        gnalog() << "\n    ";
                    }
                    gnalog() << followingLayers.second->name;
                }
                gnalog() << std::endl;
#endif
                // insert the filter
                auto filterName = std::string("AlignFilter_") + std::to_string(numOfFilterLayers++);
                auto filterLayer =
                    std::make_shared<WeightableLayer>(LayerParams({filterName, "AffineFilter", Precision::FP32}));


                auto inputData = splitOutput;
                auto newDims = splitOutput->dims;

                size_t aligned64_offset = std::max(0, static_cast<int>(ALIGN64(currentOffset) - 64));
                size_t newOutputSize = (currentOffset + ALIGN(outputSize, 8) * bytesPerSplitElement - aligned64_offset)
                    / bytesPerSplitElement;

                // encodes offset to beginning of split layer input
                filterLayer->params["offset"] = std::to_string(aligned64_offset);

                auto dims = splitOutput->getDims();
                if (dims.size() > 3) {
                    THROW_GNA_EXCEPTION << "unsupported split layer dims size: " << dims.size();
                }
                auto num_rows_out = dims[1]  * (dims.size() != 2 ? dims[2] : 1);

                std::vector<float> filterWeights(newOutputSize * num_rows_out, 0.f);

                auto offset = (currentOffset - aligned64_offset) / bytesPerSplitElement;

                for (int i = 0; i != outputSize; i++) {
                    filterWeights[offset] = 1.0f;
                    offset += newOutputSize + 1;
                }

                filterLayer->_weights = make_shared_blob<float>(inputData->precision, Layout::C, filterWeights);

                std::reverse(begin(newDims), end(newDims));

                auto outData = std::make_shared<Data>(filterName,
                                                      TensorDesc(splitOutput->precision,
                                                                 newDims,
                                                                 inputData->layout));

                auto filterWithQuant = quantized ?
                                       InferenceEngine::injectData<QuantizedLayerParams>(filterLayer) :
                                       filterLayer;
                outData->creatorLayer = filterWithQuant;
                filterWithQuant->outData.push_back(outData);
                CNNNetworkInsertLayer(l, nullptr, filterWithQuant, splitOutIndex);
            }


            // search data that starts from unaligned location
            currentOffset += outputSize * bytesPerSplitElement;
            splitOutIndex++;
        }
    }
}

void SubstituteScaleShiftBroadCastPass::run() {
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(pLayers->front());
    for (auto & l : *pLayers) {
        LayerInfo layerInfo(l);

        if (!layerInfo.isScaleShift()) {
            continue;
        }

        auto scaleShift = layerInfo.as<ScaleShiftLayer*>();

        auto insData = scaleShift->insData.front().lock();
        if (!insData) {
            THROW_GNA_EXCEPTION << "Cannot get inputs data for layer: " << l->name;
        }

        if (insData->getDims().size() <= 2) {
            // NC or C cannot do broadcast
            continue;
        }
        auto batchSize = insData->getDims()[0];
        auto nElements = product(begin(insData->getDims()), end(insData->getDims())) / batchSize;
        auto weightsElements = scaleShift->_weights->size();
        auto weightsBytes = scaleShift->_weights->byteSize();

        if (nElements == weightsElements) {
            continue;
        }

        // only 3d scaleshift supported where number of c is arbitrary
        auto lastD = insData->getDims()[insData->getDims().size() - 1];
        if (lastD != weightsElements) {
            THROW_GNA_EXCEPTION << "Unsupported layer: " << l->name
                                << " should have last dim(" << lastD << ") equal to weights(" << weightsElements << ") length";
        }
        if (insData->getDims().size() == 2) {
            THROW_GNA_EXCEPTION << "For layer: " << l->name
                                << " weights size(" << weightsElements<< ") invalid: should match input size of(" << lastD << ")";
        }

        gnalog() << "Substitution ScaleShift broadcast for layer: " << l->name << "\n";
        // approach 1 - weights tiling
        if (getPassManager()->getPolicy().ScaleShiftPolicy == Policy::ScaleShift::WEIGHTS_TILING) {
            auto tileBlob = [](Blob::Ptr &blob, size_t TileTo){
                auto weightsElements = blob->size();
                auto weightsBytes = blob->byteSize();
                if (weightsElements == 0) {
                    THROW_IE_EXCEPTION << "Blob size is 0";
                }
                if (TileTo % weightsElements) {
                    return false;
                }

                auto tiledBlob = make_plain_blob(blob->getTensorDesc().getPrecision(), {TileTo});
                tiledBlob->allocate();


                for (int i=0; i != TileTo / weightsElements; i++) {
                    ie_memcpy(tiledBlob->buffer().as<uint8_t*>() + i * weightsBytes, weightsBytes, blob->cbuffer(), weightsBytes);
                }
                blob = tiledBlob;
                return true;
            };

            if (!tileBlob(scaleShift->_weights, nElements)) {
                THROW_GNA_EXCEPTION << "Cannot tile weights for layer: " << l->name << ", due to weights size not GCD of dims product";
            }
            if (scaleShift->_biases) {
                if (!tileBlob(scaleShift->_biases, nElements)) {
                    THROW_GNA_EXCEPTION << "Cannot tile biases for layer: " << l->name << ", due to biases size not GCD of dims product";
                }
            }

            // currently data type no providing reshape method of tensor desc
            scaleShift->outData.front()->reshape({batchSize, nElements}, Layout::NC);
            insData->reshape({batchSize, nElements}, Layout::NC);
        } else {
            THROW_GNA_EXCEPTION << "Not implemented substitution of scaleshift broadcast policy of "
                                << getPassManager()->getPolicy().ScaleShiftPolicy <<  "using layers tiling, layer: " << l->name;
        }
    }
}

void UnrollLSTMCellPass::run() {
    // TODO: iefode: refactor this code
    InferenceEngine::NetPass::UnrollRNN_if(*getPassManager()->getNetwork(), [] (const RNNCellBase& rnn) -> bool {
        if (rnn.clip != 0.0f)
            return true;
        if (rnn.type == "GRUCell" ||
            rnn.type == "GRUSequence" ||
            rnn.type == "RNNCell" ||
            rnn.type == "RNNSequence")
            return true;
        if (!(rnn.type == "LSTMCell" || rnn.type == "LSTMSequence") ||
            rnn.activations == std::vector<std::string>{"relu"})
            return false;
        return true;
    });
}

void UnrollTIPass::run() {
    auto sts = InferenceEngine::NetPass::UnrollTI(*getPassManager()->getNetwork());
    if (!sts) {
        THROW_GNA_EXCEPTION << "TensorIterator layer cannot be unrolled!";
    }
}

void PassManager::run() {
    int index = 0;
#ifdef PLOT
    auto dumpNetworkAfterPass = [&index, this] (std::shared_ptr<Pass> pass) {
        std::string name = std::string("gna_passes_") + (index < 10 ? "0" : "") + std::to_string(index) + "_" + pass->getName() + ".dot";
        std::ofstream out(name);
        saveGraphToDot(*network.get(), out, [](const CNNLayerPtr layer,
                                               ordered_properties &printed_properties,
                                               ordered_properties &node_properties) {});
    };
#else
    auto dumpNetworkAfterPass = [] (std::shared_ptr<Pass> ) {};
#endif

    for (auto && pass : passes) {
        auto layers = CNNNetSortTopologically(*network.get());
        pass->attach(layers);
        gnalog() << "PASS: " << ++index << "/" << passes.size() << ":" << pass->getName() << "\n";
        pass->run();
        dumpNetworkAfterPass(pass);
    }
}
