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

#include <quantization/quantized_layer_params.hpp>
#include "gna_plugin.hpp"
#include "gna_layer_info.hpp"


using namespace InferenceEngine;
using namespace InferenceEngine::details;
using namespace GNAPluginNS;

void GNAPlugin::insertDiagonalLayer(std::vector<CNNLayerPtr> & layers) {
    int numOfDiagLayers = 0;
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layers.front());
    for (auto & l : layers) {
        if (l->insData.empty()) continue;
        auto prevLayer = CNNNetPrevLayer(l);
        if (LayerInfo(l).isActivation()) {
            if (LayerInfo(prevLayer).has32BOutput())
                continue;
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

#ifdef PLOT
        std::cout << "Inserted Diagonal Layer between: " << prevLayer->name << " and " << l->name << "\n" << std::flush;
#endif
        // actual insertion
        auto diagName = std::string("SyntheticScaleShift_") + std::to_string(numOfDiagLayers++);
        auto diagLayer = std::make_shared<ScaleShiftLayer>(LayerParams({diagName, "ScaleShift", Precision::FP32}));

        // TODO: diagonal size
        std::vector<float> arrayOf1(l->outData[0]->dims[0], 1.f);
        diagLayer->_weights = make_shared_blob<float>(l->outData[0]->precision, Layout::C, arrayOf1);
        auto newDims = l->outData[0]->dims;
        auto dataPtr = std::make_shared<Data>(diagName,
                                              newDims,
                                              l->outData[0]->precision,
                                              l->outData[0]->layout);

        auto diagonalWithQuant = quantized ?
                            InferenceEngine::injectData<QuantizedLayerParams>(diagLayer) :
                                                                                    diagLayer;

        dataPtr->creatorLayer = diagonalWithQuant;
        diagonalWithQuant->outData.push_back(dataPtr);
        CNNNetworkInsertLayer(prevLayer, l, diagonalWithQuant);
    }
}

void GNAPlugin::reorderMaxPool(std::vector<InferenceEngine::CNNLayerPtr> & layers) {
    // detecting following pattern
    // conv->relu->maxpooling
    // changing it to conv->mxpooling->relu
    for (auto & l : layers) {
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

std::vector<CNNLayerPtr> GNAPlugin::getCandidatesForIdentityInsertion(const CNNLayerPtr l) {
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
        // for  mul if we have 4-4 - inputs we need to insert 2 identities activations  to put 2 bytes input and weights
        auto prev0 = CNNNetPrevLayer(l, 0);
        auto prev1 = CNNNetPrevLayer(l, 1);
        switch (eltwise->_operation) {
            case EltwiseLayer::Sum:
                if (!LayerInfo(prev0).has32BOutput() || !LayerInfo(prev1).has32BOutput()) {
                    return prevLayers;
                }
                // TODO: wether there - are possibility to select what layer to quantize
                prevLayers.push_back(prev0);
                break;
            case EltwiseLayer::Prod:
                if (LayerInfo(prev0).has16BOutput() && LayerInfo(prev1).has16BOutput()) {
                    return prevLayers;
                }

                if (LayerInfo(prev0).has32BOutput()) {
                    prevLayers.push_back(prev0);
                }

                if (LayerInfo(prev1).has32BOutput()) {
                    prevLayers.push_back(prev1);
                }

                break;
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
    } else {  // not eltwise or concat
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

void GNAPlugin::substitutePRelu(std::vector<InferenceEngine::CNNLayerPtr> &layers) {
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
    for (auto & l : layers) {
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

void GNAPlugin::reversePermutations(std::vector<CNNLayerPtr> &layers) {
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

    for (auto & l : layers) {
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
    for (auto & l : layers) {
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

void GNAPlugin::insertIdentityLayer(std::vector<CNNLayerPtr> &layers) {
    int numOfIdentityLayers = 0;
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layers.front());
    for (auto & l : layers) {
        for (auto && prev : getCandidatesForIdentityInsertion(l)) {
            // actual insertion
            auto activationName = std::string("identity_") + std::to_string(numOfIdentityLayers++);

            gnalog() << "Inserted "<< activationName << " between: " << prev->name << " and " << l->name << "\n" << std::flush;

            CNNLayerPtr activationLayer =
                std::make_shared<GenericLayer>(LayerParams({activationName, "identity", Precision::FP32}));
            auto inputData = l->insData[0].lock();
            auto newDims = inputData->dims;
            std::reverse(begin(newDims), end(newDims));

            auto dataPtr = std::make_shared<Data>("FullyConnected",
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

            CNNNetworkInsertLayer(prev, notAll ? l : CNNLayerPtr(nullptr), activationLayerWithQuant);
        }
    }
}

void GNAPlugin::insertCopyLayer(std::vector<InferenceEngine::CNNLayerPtr> & layers) {
    int numCopyLayers = 0;
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layers.front());
    for (auto & l : layers) {
        if (l->insData.empty()) continue;
        auto prevLayer = CNNNetPrevLayer(l);
        if ((LayerInfo(l).isMemory() && LayerInfo(prevLayer).isConcat()) ||
            (LayerInfo(l).isConcat() && LayerInfo(prevLayer).isCrop())) {
            if (LayerInfo(prevLayer).isCrop()) {
                auto cropLayer =  LayerInfo(prevLayer).as<CropLayer*>();
                size_t cropOffset = cropLayer->offset.back() * cropLayer->precision.size();
                if (ALIGN(cropOffset, 8) != cropOffset) {
                    // The crop will be replced by affine.
                    // Copy layer insertion is not required
                    continue;
                }
            }
            std::string copyName = std::string("copy_") + std::to_string(numCopyLayers++);
            gnalog() << "Inserted "<< copyName << " between: " << l->name << " and " << prevLayer->name << "\n" << std::flush;

            CNNLayerPtr copyLayer =
            std::make_shared<GenericLayer>(LayerParams({copyName, "Copy", Precision::FP32}));

            auto inputData = l->insData[0].lock();
            auto newDims = inputData->dims;

            std::reverse(begin(newDims), end(newDims));

            auto dataPtr = std::make_shared<Data>(copyName,
                                                  TensorDesc(inputData->precision,
                                                             newDims,
                                                             inputData->layout));

            auto copyWithQuant = quantized ?
                                    InferenceEngine::injectData<QuantizedLayerParams>(copyLayer) :
                                                                                            copyLayer;
            dataPtr->creatorLayer = copyWithQuant;
            copyWithQuant->outData.push_back(dataPtr);
            CNNNetworkInsertLayer(prevLayer, l, copyWithQuant);
        }
    }
}

void GNAPlugin::insertAligningFilterLayer(std::vector<InferenceEngine::CNNLayerPtr> & layers) {
    // currently split layer only supports 2 bytes in int16 and int8 mode. In fp32 mode this no necessary but usefull for testing
    const int bytesPerSplitElement = 2;
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layers.front());

    int numOfFilterLayers = 0;
    for (auto &l : layers) {
        auto info = LayerInfo(l);
        if (!info.isSplit() && !info.isSlice()) {
            continue;
        }

        size_t currentOffset = 0;
        int splitOutIndex = 0;
        for (auto &&splitOutput  : l->outData) {
            auto outputSize = product(++begin(splitOutput->getDims()), end(splitOutput->getDims()));

            if (currentOffset != ALIGN64(currentOffset)) {
                // this split output not beginning from 64 bytes aligned boundary - need to correct by alligning filter layer
#ifdef PLOT
                // getting list of layers attached to current split output
                gnalog() << "Inserted Affine Filter Layer between: " << l->name << " and : \n";
                for (auto &&followingLayers : splitOutput->getInputTo()) {
                    gnalog() << "    " << followingLayers.second->name << "\n";
                }
                gnalog() << std::flush;
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

                auto &num_rows_out = splitOutput->dims[0];

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

void GNAPlugin::substituteScaleShiftBroadCast(std::vector<InferenceEngine::CNNLayerPtr> &layers) {
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layers.front());
    for (auto & l : layers) {
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
        auto nElements = details::product(insData->getDims()) / batchSize;
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
        if (policy.ScaleShiftPolicy == Policy::WEIGHTS_TILING) {
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
                                << policy.ScaleShiftPolicy <<  "using layers tiling, layer: " << l->name;
        }
    }
}