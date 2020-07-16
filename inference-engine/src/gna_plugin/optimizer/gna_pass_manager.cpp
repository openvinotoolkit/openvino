// Copyright (C) 2018-2020 Intel Corporation
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
#include <limits>
#include <iomanip>

#include <graph_transformer.h>
#include <gna-api.h>
#include <blob_factory.hpp>
#include <ie_memcpy.h>
#include <ie_algorithm.hpp>
#include <details/ie_cnn_network_tools.h>
#include <ie_util_internal.hpp>
#include <graph_tools.hpp>
#include <net_pass.h>
#include <cmath>

#include "gna_plugin_log.hpp"
#include "frontend/quantized_layer_params.hpp"
#include "gna_graph_tools.hpp"
#include "gna_pass_manager.hpp"
#include "layers/gna_layer_info.hpp"
#include "gna_upstream_iterator.hpp"


using namespace InferenceEngine;
using namespace InferenceEngine::details;
using namespace GNAPluginNS;

#define pass_trace() gnalog() << "[" << getName() << "]"

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
static const char memoryLayersCounter[] = "numMemoryLayers";
static const char softSignLayersCounter[] = "numSoftSignLayers";

/**
 * @brief helper injections of diagonal layer with certain value
 */

static const char diagonalLayerCounterName[] = "diagonalLayerCounter";

static void insertDiagonalLayerBetween(InferenceEngine::CNNLayerPtr prevLayer,
                                       InferenceEngine::CNNLayerPtr nextLayer,
                                       std::shared_ptr<IPassManager> passmanager,
                                       float fillValue) {
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(prevLayer);
    auto diagName = std::string("SyntheticScaleShift_") + std::to_string(passmanager->getIntVar(diagonalLayersCounterName)++);
    gnalog() << "Inserted Diagonal Layer " << diagName <<" between: " << prevLayer->name << " and " << nextLayer->name << "\n" << std::flush;

    auto diagLayer = std::make_shared<ScaleShiftLayer>(LayerParams({diagName, "ScaleShift", Precision::FP32}));
    IE_ASSERT(diagLayer != nullptr);

    // TODO: diagonal size
    auto dimsIndex = nextLayer->outData[0]->getTensorDesc().getDims().size() - 1;
    std::vector<float> weightsValues(nextLayer->outData[0]->getTensorDesc().getDims()[dimsIndex], fillValue);
    IE_ASSERT(diagLayer != nullptr);
    diagLayer->_weights = make_shared_blob<float>(
            TensorDesc(
                nextLayer->outData[0]->getTensorDesc().getPrecision(),
                SizeVector({weightsValues.size()}),
                Layout::C));
    diagLayer->_weights->allocate();
    CopyVectorToBlob(diagLayer->_weights, weightsValues);
    auto dataPtr = std::make_shared<Data>(diagName, nextLayer->outData[0]->getTensorDesc());

    auto diagonalWithQuant = quantized ?
                             InferenceEngine::injectData<QuantizedLayerParams>(diagLayer) : diagLayer;

    getCreatorLayer(dataPtr) = diagonalWithQuant;
    diagonalWithQuant->outData.push_back(dataPtr);

    // actual insertion
    CNNNetworkInsertLayer(prevLayer, nextLayer, diagonalWithQuant);
}

static void InsertMemoryLayer(InferenceEngine::CNNLayerPtr prevLayer,
                              size_t portId,
                              std::shared_ptr<IPassManager> passmanager,
                              std::string id) {
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(prevLayer);
    auto memoryName = std::string("SyntheticMemory_") + std::to_string(passmanager->getIntVar(memoryLayersCounter)++);
    if (prevLayer != nullptr) {
        gnalog() << "Inserted Memory layer " << memoryName << " after: " << prevLayer->name << "." << portId;
    } else {
        THROW_GNA_EXCEPTION << "cannot attach memory layer to nullptr, parent layer need to be specified";
    }

    auto memoryLayer = std::make_shared<CNNLayer>(LayerParams({memoryName, "Memory", Precision::FP32}));
    if (prevLayer->outData.size() <= portId) {
        THROW_GNA_EXCEPTION << "cannot attach memory layer to " << prevLayer->name << " at port " << portId
                            << ", since there are only " << prevLayer->outData.size() << " ports";
    }
    auto dataPtr = std::make_shared<Data>(memoryName, prevLayer->outData[portId]->getTensorDesc());

    auto memoryLayerWithQuant = quantized ? InferenceEngine::injectData<QuantizedLayerParams>(memoryLayer) : memoryLayer;

    InferenceEngine::getCreatorLayer(dataPtr) = memoryLayerWithQuant;
    memoryLayerWithQuant->outData.push_back(dataPtr);

    // TODO: insert layer algo doesn't support inserting between given layer and void if there are active connection
    // CNNNetworkInsertLayer(prevLayer, nullptr, memoryLayer, portId);

    InferenceEngine::getInputTo(prevLayer->outData[portId])[memoryName] = memoryLayerWithQuant;
    memoryLayerWithQuant->insData.push_back(prevLayer->outData[portId]);
    memoryLayerWithQuant->params["id"] = id;
    memoryLayerWithQuant->params["index"] = "0";
    memoryLayerWithQuant->params["size"] = "2";
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
    auto dataPtr = std::make_shared<Data>(copyName, inputData->getTensorDesc());
    auto copyWithQuant = quantized ?
                         InferenceEngine::injectData<QuantizedLayerParams>(copyLayer) :
                         copyLayer;
    getCreatorLayer(dataPtr) = copyWithQuant;
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

    auto PrevFunctionalLayer = [](CNNLayerPtr l, int idx = 0) {
        auto prevLayer = CNNNetPrevLayerSkipCertain(l, idx, [](CNNLayerPtr ptr) {
            return LayerInfo(ptr).isNonFunctional();
        });
        gnalog() << "CNNNetPrevLayerSkipCertain for :: " << l->name << "returned: " << prevLayer->name << std::endl;
        return prevLayer;
    };


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
        // TODO: skip non functional
        auto prev0 = PrevFunctionalLayer(l, 0);
        auto prev1 = PrevFunctionalLayer(l, 1);
        switch (eltwise->_operation) {
            case EltwiseLayer::Sub:
            case EltwiseLayer::Sum:
                if (!LayerInfo(prev0).has32BOutput() || !LayerInfo(prev1).has32BOutput()) {
                    return prevLayers;
                }
                // TODO: whether there are possibility to select after what layer identity gets inserted
                prevLayers.push_back(CNNNetPrevLayer(l, 0));
                break;
            case EltwiseLayer::Prod: {
                if (LayerInfo(prev0).has16BOutput() && LayerInfo(prev1).has16BOutput()) {
                    return prevLayers;
                }

                if (LayerInfo(prev0).has32BOutput()) {
                    prevLayers.push_back(CNNNetPrevLayer(l, 0));
                }

                // if layers of outdata are different
                auto prevData0 = l->insData[0].lock();
                auto prevData1 = l->insData[1].lock();

                if ((prev0 != prev1 || prevData0 != prevData1) && LayerInfo(prev1).has32BOutput()) {
                        prevLayers.push_back(CNNNetPrevLayer(l, 1));
                }

                break;
            }
            default :
                THROW_GNA_EXCEPTION << "Eltwise Layer of type: " << eltwise->_operation << " not supported";
        }
    } else if (concat != nullptr) {
        for (int i = 0; CNNNetHasPrevLayer(l.get(), i); ++i) {
            auto prev = PrevFunctionalLayer(l, i);
            if (LayerInfo(prev).has32BOutput()) {
                prevLayers.push_back(CNNNetPrevLayer(l, i));
            }
        }
    } else {
        // not eltwise or concat
        // other layers has 1 inputs - situation is easier
        // ex. activation or pooling - no need to insert identity activation.
        if (LayerInfo(l).isNonFunctional() || LayerInfo(l).has32BInput())
            return prevLayers;

        gnalog() << "CNNNetPrevLayerSkipCertain for :: " << l->name << std::endl;
        auto prevLayer = CNNNetPrevLayerSkipCertain(l, 0, [](CNNLayerPtr ptr) {
            return LayerInfo(ptr).isNonFunctional();
        });

        if (!LayerInfo(prevLayer).has32BOutput())
            return prevLayers;

        prevLayers.push_back(CNNNetPrevLayer(l, 0));
    }
    return prevLayers;
}

void InsertDiagonalLayerPass::run() {
    for (auto & l : *pLayers) {
        if (l->insData.empty()) continue;
        auto prevLayer = CNNNetPrevLayerSkipCertain(l, 0, [](CNNLayerPtr ptr) {
            return LayerInfo(ptr).isNonFunctional();
        });
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

            if (eltwise->_operation != EltwiseLayer::Sum && eltwise->_operation != EltwiseLayer::Sub)
                continue;

            auto prevLayer1 = CNNNetPrevLayerSkipCertain(l, 1, [](CNNLayerPtr ptr) {
                return LayerInfo(ptr).isNonFunctional();
            });
            if (!LayerInfo(prevLayer).has16BOutput() || !LayerInfo(prevLayer1).has16BOutput())
                continue;
        }
        auto prevDirectLayer = CNNNetPrevLayer(l, 0);
        insertDiagonalLayerBetween(prevDirectLayer, l, getPassManager(), 1.f);
    }
}

void HandleMultipleActivationsForTheLayerPass::run() {
    // found layer followed by multiple activations
    for (auto & l : *pLayers) {
        CNNLayerSet activations;

        for (auto && odata : l->outData) {
            for (auto && inputTo : getInputTo(odata)) {
                LayerInfo info(inputTo.second);

                if (info.isActivation()) {
                    activations.insert(inputTo.second);
                }
            }
        }
        // single or not activations case
        if (activations.size() < 2) continue;

        // insert diagonals one per each activation
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

void SubstituteSoftSignPass::run() {
    auto hasNChildren = [](CNNLayerPtr l, int N){
        if (l->outData.size() != 1) return false;
        if (getInputTo(l->outData.front()).size() != N) return false;
        return true;
    };
    auto getNthChild = [](CNNLayerPtr l, int N) {
        auto first = getInputTo(l->outData.front()).begin();
        std::advance(first, N);
        return first->second;
    };
    for (auto & l : *pLayers) {
        if (!hasNChildren(l, 2)) continue;
        auto mul = getNthChild(l, 0);
        auto abs = getNthChild(l, 1);

        bool cont = true;
        if (LayerInfo(mul).isEltwiseMul() && LayerInfo(abs).isAbs()) {
            cont = false;
        }
        if (cont && LayerInfo(abs).isEltwiseMul() && LayerInfo(mul).isAbs()) {
            std::swap(mul, abs);
            cont = false;
        }
        if (cont) continue;
        if (!hasNChildren(abs, 1)) continue;
        auto power = getNthChild(abs, 0);

        if (!LayerInfo(power).isPower()) continue;
        auto powerLayer = LayerInfo(power).as<PowerLayer*>();
        if (powerLayer->power != -1) continue;
        if (powerLayer->offset != 1) continue;
        if (powerLayer->scale != 1) continue;

        if (!hasNChildren(power, 1)) continue;
        auto mulSame = getNthChild(power, 0);
        if (mulSame != mul) continue;

        // pattern matched - lets substitute
        gnalog() << "SoftSign subgraph found consits of: \n"
                 << "\t" << abs->name << "\n"
                 << "\t" << power->name << "\n"
                 << "\t" << mul->name << "\n"
                 << std::endl;

        // creating softsign layer
        auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(l);
        auto layerName = std::string("Synthetic_SoftSign_")
                + std::to_string(getPassManager()->getIntVar(softSignLayersCounter)++);

        CNNLayerPtr activationLayer =
                std::make_shared<GenericLayer>(LayerParams({layerName, "SoftSign", Precision::FP32}));
        auto activationLayerWithQuant = quantized ?
                                        InferenceEngine::injectData<QuantizedLayerParams>(activationLayer) :
                                        activationLayer;

        auto mulData = mul->outData;

        // rebind outdata of mull to be outdata of softsign
        for (auto && data : mulData) {
            getCreatorLayer(data) = activationLayerWithQuant;
            data->setName("softsign_data_" + std::to_string(getPassManager()->getIntVar(softSignLayersCounter)));
            activationLayerWithQuant->outData.push_back(data);
        }

        // making connection l->softsign
        getInputTo(l->outData.front()).clear();
        getInputTo(l->outData.front())[layerName] = activationLayerWithQuant;

        // making back connection softsign->mul
        activationLayerWithQuant->insData.push_back(l->outData.front());
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
        return getInputTo(layer->outData[0]).begin()->second.get();
    };

    // TODO: unit tests for bad cases
    for (auto & l : *pLayers) {
        // assume l is starting layer, that is followed by eltwise_sum(relu, negate/relu/scale/negate)
        if (l->outData.size() != 1) continue;
        auto &outputLayers = getInputTo(l->outData[0]);
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
        if (sum->insData.size() != 2
                || sum->insData[0].lock() == nullptr
                || sum->insData[1].lock() == nullptr) continue;

        auto inData_0 = sum->insData[0].lock();
        IE_ASSERT(inData_0 != nullptr);
        auto creatorLayer_0 = getCreatorLayer(inData_0).lock();
        IE_ASSERT(creatorLayer_0 != nullptr);
        auto inData_1 = sum->insData[1].lock();
        IE_ASSERT(inData_1 != nullptr);
        auto creatorLayer_1 = getCreatorLayer(inData_1).lock();
        IE_ASSERT(creatorLayer_1 != nullptr);

        auto s1 = creatorLayer_0.get();
        auto s2 = creatorLayer_1.get();

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
        getCreatorLayer(relu1->outData[0]) = relu1;
        // pointing back to relu if any
        if (!getInputTo(relu1->outData[0]).empty()) {
            auto summOutputLayer = getInputTo(relu1->outData[0]).begin()->second;
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
            return LayerInfo(l2).isNonFunctional();
        });
    };


    std::function<CNNLayerPtr(CNNLayerPtr)> nextLayerSkipReshape = [&nextLayerSkipReshape](CNNLayerPtr layer) -> CNNLayerPtr {
        if (layer->outData.empty()) {
            return nullptr;
        }
        if (getInputTo(layer->outData.front()).size() != 1) {
            return nullptr;
        }
        auto next = getInputTo(layer->outData.front()).begin()->second;

        if (LayerInfo(next).isNonFunctional()) return nextLayerSkipReshape(next);

        return next;
    };

    auto prevConv = [&prevLayerSkipCertain](CNNLayerPtr layer) -> CNNLayerPtr {
        return prevLayerSkipCertain(layer, [] (CNNLayerPtr l2) {
            return
                LayerInfo(l2).isNonFunctional() ||
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

            auto dataPtr = std::make_shared<Data>("identity_data_" + std::to_string(numOfIdentityLayers), inputData->getTensorDesc());
            auto activationLayerWithQuant = quantized ?
                                            InferenceEngine::injectData<QuantizedLayerParams>(activationLayer) :
                                            activationLayer;
            getCreatorLayer(dataPtr) = activationLayerWithQuant;
            activationLayerWithQuant->outData.push_back(dataPtr);
            // wether 1 identity or all outputs TODO possible grouping here, need to implement special groupped inserter
            bool notAll = false;
            for (auto && nextData  : prev->outData) {
                for (auto && nextLayer : getInputTo(nextData)) {
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

void InsertCopyLayerPass::run() {
    for (auto & l : *pLayers) {
        if (l->insData.empty()) continue;
        auto prevLayers = CNNNetGetPrevLayersSkip(l, [](CNNLayerPtr origin){
            return !LayerInfo(origin).isNonFunctional();
        });

        for (int i=0; i != prevLayers.size(); i++) {
            auto & prevIndirectLayer = prevLayers[i].first;
            bool bInsert = false;
            if (LayerInfo(l).isMemory()) {
                if (LayerInfo(prevIndirectLayer).isConcat()) { bInsert = true;}
                // memory usualy preceded by either activation or split, or other layers in order to have 2b precision
                for (auto && inputto : getInputTo(prevLayers[i].first->outData[prevLayers[i].second])) {
                    // if preceding layer is common for memory and concat
                    if (LayerInfo(inputto.second).isConcat()) {
                        bInsert = true;
                        break;
                    }
                }
            }
            if (LayerInfo(l).isConcat() && LayerInfo(prevIndirectLayer).isCrop()) { bInsert = true; }

            if (bInsert) {
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

    if (getPassManager()->getPolicy().ConcatAlignmentPolicy == Policy::ConcatAlignment::DISABLED) {
        return;
    }
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
        size_t offset = 0;
        auto concatLayer = info.as<ConcatLayer*>();

        for (auto input_idx = 0; input_idx != concatLayer->insData.size(); input_idx++) {
            auto getLayerByIndex = [&concatLayer](int idx) {
                auto input = concatLayer->insData[idx];
                auto lockedInput = input.lock();
                if (!lockedInput) {
                    THROW_GNA_EXCEPTION << "cannot get insdata : "<< idx << " for layer: " << concatLayer->name;
                }
                return lockedInput;
            };

            auto concatInput = getLayerByIndex(input_idx);
            auto dims = concatInput->getDims();
            auto outputSize = details::product(++dims.begin(), dims.end()) * bytesPerConcatElement;

            auto useAlignFilterIf = [&concatLayer, &getLayerByIndex](int concat_input_idx) {
                if (concatLayer->insData.size() <= concat_input_idx) return false;

                auto nextInput = getCreatorLayer(getLayerByIndex(concat_input_idx)).lock();

                if (LayerInfo(nextInput).isInput()) return false;

                return true;
            };

            // correcting offset by copy layer insertion. This can be improved by collapsing copy and affine or diagonal later-on
            // if next concat inputs requires align filter - then current input also requires either copy or align filter
            if (ALIGN64(offset) != offset || (ALIGN64(outputSize) != outputSize && useAlignFilterIf(input_idx + 1))) {
                auto prevLayer = getCreatorLayer(concatInput).lock();
                // input layer parameters are copied not using GNA-primitives - so nothing to allign here.
                if (!useAlignFilterIf(input_idx)) continue;

                gnalog() << "Inserted Concat Aligning Layer between: " << prevLayer->name << " and " << l->name << std::endl;

                // insert the filter
                auto filterName = std::string("ConcatAlignFilter_") + std::to_string(numOfFilterLayers++);
                auto concatAligningFilter =
                    std::make_shared<WeightableLayer>(LayerParams({filterName, "ConcatAlignFilter", Precision::FP32}));

                if (dims.size() != 2) {
                    THROW_GNA_EXCEPTION << "unsupported concat input    a of dims.size()=" << dims.size() << ", layer=" << prevLayer->name;
                }

                auto num_rows_in = dims[1];
                size_t aligned64_offset = std::max(0, static_cast<int>(ALIGN64(offset) - 64));
                size_t num_rows_padded = (offset - aligned64_offset) / bytesPerConcatElement;
                size_t num_rows_out = num_rows_padded + num_rows_in;

                // encodes offset to beginning of split layer input
                concatAligningFilter->params["output_offset"] =
                        std::to_string((aligned64_offset / bytesPerConcatElement) * (quantized ? bytesPerConcatElement : 4));

                // for padded rows we cannot use copy layer - TBD how to implement
                concatAligningFilter->params["num_rows_padded"] = std::to_string(num_rows_padded);

                // encodes original output size
                concatAligningFilter->params["original_num_rows"] = std::to_string(num_rows_in);

                std::vector<float> filterWeights(num_rows_out * num_rows_in, 0.f);

                auto identityIdx = num_rows_padded * num_rows_in;
                for (int i = 0; i != num_rows_in; i++) {
                    filterWeights[identityIdx] = 1.0f;
                    identityIdx += num_rows_in + 1;
                }

                concatAligningFilter->_weights = make_shared_blob<float>(
                                        TensorDesc(
                                            concatInput->getTensorDesc().getPrecision(),
                                            SizeVector({filterWeights.size()}),
                                            Layout::C));
                concatAligningFilter->_weights->allocate();

                CopyVectorToBlob(concatAligningFilter->_weights, filterWeights);

                // modifying output rows to be used - to avoid modification to original concat we are store num of elements in params
                dims[1] = num_rows_out;

                auto outData = std::make_shared<Data>(filterName,
                                                      TensorDesc(concatInput->getPrecision(),
                                                                 dims,
                                                                 concatInput->getLayout()));

                auto filterWithQuant = quantized ?
                                       InferenceEngine::injectData<QuantizedLayerParams>(concatAligningFilter) :
                                       concatAligningFilter;
                getCreatorLayer(outData) = filterWithQuant;
                filterWithQuant->outData.push_back(outData);

                CNNNetworkInsertLayer(prevLayer, l, filterWithQuant);
            }
            offset += outputSize;
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
        if (!info.isConcatAlignFilter()) {
            continue;
        }

        // 2rd locating concat
        if (l->outData.size() != 1) {
            THROW_GNA_EXCEPTION << "no concat layer after concat aligning layer" << l->name;
        }
        auto nextLayers = getInputTo(l->outData.front());

        if (nextLayers.size() != 1) {
            THROW_GNA_EXCEPTION << "Invalid concat connection in align filter : " << l->name;
        }
        auto concat = nextLayers.begin()->second;
        if (!LayerInfo(concat).isConcat()) {
            THROW_GNA_EXCEPTION << "no concat layer after concat-aligning layer" << l->name << ", but was: " << concat->type;
        }
        // 3stage locate first input in concat
        if (concat->insData.size() < 2) {
            THROW_GNA_EXCEPTION << "Concat layer has unsupported number of incoming layers: " << concat->name;
        }
        auto inputsToConcatFirst = CNNNetGetPrevLayersSkip(concat, [](CNNLayerPtr origin){
            return !LayerInfo(origin).isNonFunctional() && !LayerInfo(origin).isSplit();
        }, 0);

        if (inputsToConcatFirst.empty()) {
            THROW_GNA_EXCEPTION << "cannot locate first input into concat layer: " << l;
        }

        auto firstInputToConcat = inputsToConcatFirst.front().first;

        // concat has first input of concat align filter - dont need to reorder it
        if (firstInputToConcat == l) {
            continue;
        }

        bool bFinish = false;
        // making a link activation possible without extra layer if first input to concat not a parent / indirect parent of second input
        // using ufs - upper first search
        gnalog() << "[UFS] searching for: "<< firstInputToConcat->name << "\n";

        CNNNetDFS(l, [&l, &firstInputToConcat, &bFinish](CNNLayerPtr layer) {
            gnalog() << "[UFS] from : "<< l->name <<" reached: " << layer->name << "\n";
            // found that direct input to concat is a indirect parent of align filter - so no link required
            if (layer.get() == firstInputToConcat.get() || LayerInfo(firstInputToConcat).isInput()) {
                gnalog() << "[UFS] copy layer insertion needed\n";
                bFinish = true;
            }
        }, true, [&bFinish](InferenceEngine::CNNLayer* from) {
            // aborting UFS once link not need
            return make_upstream_order(!bFinish ? from : nullptr);
        });

        auto linkName = std::string("link_") + std::to_string(numOfLinkLayers++);

        auto linkWithoutQuant = std::make_shared<CNNLayer>(LayerParams({linkName, "link", Precision::FP32}));

        auto link = quantized ?
                    InferenceEngine::injectData<QuantizedLayerParams>(linkWithoutQuant) :
                    linkWithoutQuant;


        auto linkOutData = std::make_shared<Data>(linkName,
                                              TensorDesc(Precision::FP32,
                                                         SizeVector({1}),
                                                         Layout::C));
        getCreatorLayer(linkOutData) = link;

        link->outData.push_back(linkOutData);
        link->insData.push_back(l->outData.front());

        getInputTo(linkOutData)[firstInputToConcat->name + ".via.link"] = firstInputToConcat;
        firstInputToConcat->insData.push_back(linkOutData);

        getInputTo(l->outData.front())[linkName] = link;
    }
}

void InsertSplitAligningFilterPass::run() {
    // currently split layer only supports 2 bytes in int16 and int8 mode. In fp32 mode this is not necessary but is useful for testing
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
                for (auto &&followingLayers : getInputTo(splitOutput)) {
                    if (getInputTo(splitOutput).size() != 1) {
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

                size_t aligned64_offset = std::max(0, static_cast<int>(ALIGN64(currentOffset) - 64));
                size_t newOutputSize = (currentOffset + ALIGN(outputSize, 8) * bytesPerSplitElement - aligned64_offset)
                                       / bytesPerSplitElement;

                IE_ASSERT(filterLayer != nullptr);

                // encodes offset to beginning of split layer input
                filterLayer->params["offset"] = std::to_string(aligned64_offset / bytesPerSplitElement);

                auto dims = splitOutput->getTensorDesc().getDims();
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

                filterLayer->_weights = make_shared_blob<float>(TensorDesc(
                        inputData->getTensorDesc().getPrecision(),
                        SizeVector({filterWeights.size()}),
                        Layout::C));
                filterLayer->_weights->allocate();
                CopyVectorToBlob(filterLayer->_weights, filterWeights);

                auto outData = std::make_shared<Data>(filterName,
                                                      TensorDesc(splitOutput->getTensorDesc().getPrecision(),
                                                                 splitOutput->getTensorDesc().getDims(),
                                                                 inputData->getTensorDesc().getLayout()));

                auto filterWithQuant = quantized ?
                                       InferenceEngine::injectData<QuantizedLayerParams>(filterLayer) :
                                       filterLayer;
                getCreatorLayer(outData) = filterWithQuant;
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

void RemoveConstPass::run() {
    auto network = getPassManager()->getNetwork();
    auto* implNetwork = dynamic_cast<details::CNNNetworkImpl*>(network.get());
    if (!implNetwork) {
        THROW_GNA_EXCEPTION << "Remove const layers pass can only work on cnnnetworkimpl type";
    }
    ConstTransformer transformer(implNetwork);
    transformer.fullTrim();
}

void FuseMultipleIdentitiesPass::run() {
    for (auto &l : *pLayers) {
        if (l->insData.empty()) continue;

        auto isNonFunctional = [](CNNLayerPtr ptr) {
            return LayerInfo(ptr).isNonFunctional();
        };
        auto eltwise = dynamic_cast<InferenceEngine::EltwiseLayer *>(l.get());
        auto concat = dynamic_cast<InferenceEngine::ConcatLayer *>(l.get());

        if (LayerInfo(l).isNonFunctional() || LayerInfo(l).has32BInput())
            continue;
        gnalog() << "CNNNetPrevLayer skip non functional from :: " << l->name;
        auto prevLayersReached = CNNNetGetPrevLayersSkip(l, [](CNNLayerPtr ptr) {
            return !LayerInfo(ptr).isNonFunctional();
        });
        prevLayersReached.erase(std::remove_if(prevLayersReached.begin(),
                                               prevLayersReached.end(),
                                               [] (const std::pair<CNNLayerPtr, int> & candidate) {
            return LayerInfo(candidate.first).isLink();
        }), prevLayersReached.end());

        if (prevLayersReached.size() != 1 && eltwise == nullptr && concat == nullptr) {
            std::stringstream layers;
            for (auto && prevLayer : prevLayersReached) {
                layers << prevLayer.first->name;
                layers << ", ";
            }
            THROW_GNA_LAYER_EXCEPTION(l) << "unsupported case: connected to "
            << (prevLayersReached.empty() ? "zero" : "multiple") << " outputs : " << layers.str();
        }
        auto prevLayer = prevLayersReached.front().first;
        auto outDataIdx = prevLayersReached.front().second;
        gnalog() << ", reached " << prevLayer->name << " at " << outDataIdx << std::endl;

        if (!LayerInfo(prevLayer).has32BOutput())
            continue;

        std::vector<CNNLayerPtr> resultSet = CNNNetGetAllNextLayersSkipCertain(prevLayer, outDataIdx, isNonFunctional);

        // now result set should have all needed layers
        // checking that result set consist of already identity
        CNNLayerPtr  alreadyIdentity;
        for (auto &&res : resultSet) {
            if (LayerInfo(res).isIdentity()) {
                alreadyIdentity = res;
                break;
            }
        }
        if (!alreadyIdentity) {
            continue;
        } else {
            // just figure out how to connect to that "already identity"
            // 1st stage - disconnect given layer from previous
            auto directPrev = getCreatorLayer(l->insData.front().lock()).lock();
            auto oDataIdx = CNNLayerFindOutDataIdx(directPrev, 0);
            auto &inputTo = getInputTo(directPrev->outData[oDataIdx]);
            for (auto inIterator = inputTo.begin(); inIterator != inputTo.end(); inIterator++) {
                if (inIterator->second == l) {
                    inputTo.erase(inIterator);
                    break;
                }
            }
            l->insData.clear();

            //2nd stage - now setting up new connection
            l->insData.push_back(alreadyIdentity->outData.front());
            getInputTo(alreadyIdentity->outData.front())[l->name] = l;
        }
    }
}

void MakeExtraMemoryLayersPass::run() {
    auto network = getPassManager()->getNetwork();
    InputsDataMap iinfo;
    network->getInputsInfo(iinfo);

    OutputsDataMap outputsInfo;
    network->getOutputsInfo(outputsInfo);

    auto* implNetwork = dynamic_cast<details::CNNNetworkImpl*>(network.get());
    if (!implNetwork) {
        THROW_GNA_EXCEPTION << "[" << getName()<< "]" << " can only work on cnnnetworkimpl type";
    }

    int idx = 0;
    for (auto & input : iinfo) {
        // we have a new input -> output pair
        pass_trace() << "input layer name="<< input.first <<"\n";
        auto outLayerName = getPassManager()->getStringVar(std::to_string(idx++));
        pass_trace() << "output layer name="<< outLayerName <<"\n";

        if (outLayerName.empty()) continue;

        auto portPos = outLayerName.rfind("/");
        if (std::string::npos == portPos) {
            THROW_GNA_EXCEPTION << "output layer port for \"" << outLayerName << "\" should placed after layer name and prefixed by \"/\"";
        }
        auto port = outLayerName.substr(portPos + 1);
        bool err = false;
        float portId;
        try {
            portId = CNNLayer::ie_parse_float(port);
            if (portId == 0.0 && port != "0") {
                err = true;
            }
        } catch(...) {
            err = true;
        }
        if (err || std::ceil(portId) != portId || portId < 0) {
            THROW_GNA_EXCEPTION << "for layer: \"" << outLayerName << "\" port value should be non negative integer, but was: " << port;
        }
        int iPortId = static_cast<int>(std::ceil(portId));
        auto layerName = outLayerName.substr(0, portPos);
        auto encodedLayerAndPort = layerName + "." + std::to_string(iPortId);

        auto outputIt = outputsInfo.find(encodedLayerAndPort);
        if (outputIt == outputsInfo.end()) {
            outputIt = outputsInfo.find(layerName);
            if (outputIt == outputsInfo.end()) {
                THROW_GNA_EXCEPTION << "Cannot create memory layer pair for \"" << layerName << "\" : \"" << port
                                    << "\" layer not an output in topology";
            }
        }

        // checking that index for given out data might changed during previous transformations
        auto parent = InferenceEngine::getCreatorLayer(outputIt->second).lock();
        auto outDataIt = std::find(parent->outData.begin(), parent->outData.end(), outputIt->second);
        auto outDataIdxActual = std::distance(parent->outData.begin(), outDataIt);
        // real insertion of memory layer
        auto id = std::string("synth_") + std::to_string(getPassManager()->getIntVar(memoryLayersCounter));
        InsertMemoryLayer(parent, outDataIdxActual, getPassManager(), id);

        // replacing type of input layer to be memory layer
        auto firtstLayerAfterInput = InferenceEngine::getInputTo(input.second->getInputData()).begin()->second;
        auto insDataIdx = CNNLayerFindInsDataIdxes(input.second->getInputData(), firtstLayerAfterInput);
        auto inputLayer = InferenceEngine::getCreatorLayer(firtstLayerAfterInput->insData[insDataIdx.front()].lock()).lock();
        inputLayer->type = "Memory";
        inputLayer->params["id"] = id;
        inputLayer->params["index"] = "1";
        inputLayer->params["size"] = "2";

        // removing input info semantics completely
        implNetwork->removeInputInfo(input.first);
    }
}

int PassManager::run(int index) {
// #define PLOT
#ifdef PLOT
    auto dumpNetworkAfterPass = [&index, this] (std::shared_ptr<Pass> pass) {
        std::string name = std::string("gna_passes_") + (index < 10 ? "0" : "") + std::to_string(index) + "_" + pass->getName();
        std::ofstream out(name + ".dot");
        saveGraphToDot(*network.get(), out, [](const CNNLayerPtr layer,
                                               ordered_properties &printed_properties,
                                               ordered_properties &node_properties) {});
        network->serialize(name + ".xml", name + ".bin", nullptr);
    };
#else
    auto dumpNetworkAfterPass = [] (std::shared_ptr<Pass> ) {};
#endif

    for (auto && pass : passes) {
        if (settings.runBeforeCopy != pass->runBeforeCopyPass()) {
            continue;
        }
        auto layers = CNNNetSortTopologically(*network.get());
        pass->attach(layers);
        gnalog() << "PASS: " << ++index << "/" << passes.size() << ":" << pass->getName() << "\n";
        pass->run();
        dumpNetworkAfterPass(pass);
    }
    return index;
}
