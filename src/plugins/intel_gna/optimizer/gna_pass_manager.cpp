// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

#include <legacy/graph_transformer.h>
#include <blob_factory.hpp>
#include <ie_memcpy.h>
#include <ie_algorithm.hpp>
#include <legacy/details/ie_cnn_network_tools.h>
#include <legacy/ie_util_internal.hpp>
#include <legacy/graph_tools.hpp>
#include <legacy/net_pass.h>
#include <layers/gna_copy_layer.hpp>

#include "backend/dnn_types.h"
#include "gna_plugin_log.hpp"
#include "frontend/quantization.h"
#include "frontend/quantized_layer_params.hpp"
#include <layers/gna_copy_layer.hpp>
#include <layers/gna_fake_quantize_layer.hpp>
#include <runtime/pwl.h>
#include "gna_graph_tools.hpp"
#include "gna_pass_manager.hpp"
#include "layers/gna_layer_info.hpp"
#include "gna_upstream_iterator.hpp"
#include "frontend/quantization.h"
#include "gna_groups.hpp"
#include "gna_graph_patterns.hpp"
#include "gna_data_types.hpp"
#include "gna_tensor_tools.hpp"
#include "gna_itt.hpp"
#include "backend/gna_limitations.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;
using namespace GNAPluginNS;

#define pass_trace() gnalog() << "[" << getName() << "] "

std::shared_ptr<IPassManager> BasePass::getPassManager() {
    auto sharedMgr = mgr.lock();
    if (!sharedMgr) {
        THROW_GNA_EXCEPTION << getName() << ": cannot get PassManager object";
    }
    return sharedMgr;
}

static Blob::Ptr convertToRWBlob(const Blob::Ptr& readOnlyBlob, const std::string& name = {}) {
    auto blob = Blob::CreateFromData(std::make_shared<Data>(name, readOnlyBlob->getTensorDesc()));
    blob->allocate();
    const auto ret = ie_memcpy(blob->buffer().as<uint8_t*>(),
        blob->size() * blob->getTensorDesc().getPrecision().size(),
        readOnlyBlob->buffer().as<uint8_t*>(),
        readOnlyBlob->size() * readOnlyBlob->getTensorDesc().getPrecision().size());
    IE_ASSERT(ret == 0);
    return blob;
}

static bool fp32eq(float p1, float p2) {
    return (std::abs(p1 - p2) <= 0.00001f * std::min(std::abs(p1), std::abs(p2)));
}
// indexes stored in pass manager
static const char identityLayersCounterName[] = "identityLayerCounter";
static const char diagonalLayersCounterName[] = "diagonalLayerCounter";
static const char copyLayersCounter[] = "numCopyLayers";
static const char softSignLayersCounter[] = "numSoftSignLayers";

/**
 * @brief helper injections of diagonal layer with certain value
 */

static void insertDiagonalLayerBetween(InferenceEngine::CNNLayerPtr prevLayer,
                                       InferenceEngine::CNNLayerPtr nextLayer,
                                       std::shared_ptr<IPassManager> passmanager,
                                       float fillValue,
                                       size_t in_data_idx = invalid_data_idx) {
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(prevLayer);
    auto diagName = std::string("SyntheticScaleShift_") + std::to_string(passmanager->getIntVar(diagonalLayersCounterName)++);
    gnalog() << "Inserted Diagonal Layer " << diagName <<" between: " << prevLayer->name << " and " << nextLayer->name << "\n" << std::flush;

    auto diagLayer = std::make_shared<ScaleShiftLayer>(LayerParams({diagName, "ScaleShift", Precision::FP32}));
    IE_ASSERT(diagLayer != nullptr);

    auto inputLayer = InferenceEngine::CNNNetPrevLayerSkipCertain(nextLayer, 0, [](InferenceEngine::CNNLayerPtr ptr) {
        return LayerInfo(ptr).isNonValuesChangable();
    });
    IE_ASSERT(inputLayer != nullptr);
    size_t weightsSize = LayerInfo(prevLayer).has32BOutput() ? nextLayer->outData[0]->getDims().back() :
        Get2DReshapedData(nextLayer->outData[0], GNALimitations::GetMinBatchToFitInBuffer(nextLayer->outData[0]), 8)->getDims()[1];
    std::vector<float> weightsValues(weightsSize, fillValue);
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
    CNNNetworkInsertLayer(prevLayer, nextLayer, diagonalWithQuant, invalid_data_idx, in_data_idx);
}

/**
 * @brief copy layer inserted by several passes
 * @returns pointer to newly created COPYLayer
 */
static CNNLayerPtr InsertCopyLayer(CNNLayerPtr prevLayer, CNNLayerPtr nextLayer, int beforeIdx,
                                   std::shared_ptr<IPassManager> passmanager,  std::string copyLayerType) {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "InsertCopyLayer");
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(prevLayer);
    std::string copyName = copyLayerType + std::string("_") + std::to_string(passmanager->getIntVar(copyLayersCounter)++);
    gnalog() << "Inserted " << copyName << " between: " << prevLayer->name << " and " << nextLayer->name << std::endl;

    CNNLayerPtr copyLayer = std::make_shared<GenericLayer>(LayerParams({copyName, copyLayerType, Precision::FP32}));

    auto inputData = nextLayer->insData[beforeIdx].lock();
    auto dataPtr = std::make_shared<Data>(copyName, inputData->getTensorDesc());
    auto copyWithQuant = quantized ?
                         InferenceEngine::injectData<QuantizedLayerParams>(copyLayer) :
                         copyLayer;
    getCreatorLayer(dataPtr) = copyWithQuant;
    copyWithQuant->outData.push_back(dataPtr);
    CNNNetworkInsertLayer(prevLayer, nextLayer, copyWithQuant, invalid_data_idx, beforeIdx);
    return copyWithQuant;
}

static bool hasNextFuncLayer(const CNNLayerPtr layer) {
    return CNNNetHasNextLayerSkipCertain(layer, 0, 0, [](CNNLayerPtr layer) {
                return LayerInfo(layer).isNonFunctional();
            });
}

static std::vector<CNNLayerPtr> getCandidatesForIdentityInsertion(const CNNLayerPtr l, std::shared_ptr<IPassManager> passmanager,
                                                                  bool skipFq = false) {
    std::vector<CNNLayerPtr> prevLayers;

    // skipping memory inputs and true inputs layers
    if (l->insData.empty()) return {};

    auto eltwise = dynamic_cast<InferenceEngine::EltwiseLayer *>(l.get());
    auto concat = dynamic_cast<InferenceEngine::ConcatLayer *>(l.get());

    auto PrevFunctionalLayer = [skipFq](CNNLayerPtr l, int idx = 0) {
        auto prevLayer = CNNNetPrevLayerSkipCertain(l, idx, [skipFq](CNNLayerPtr ptr) {
            return LayerInfo(ptr).isNonFunctional() || skipFq && LayerInfo(ptr).isFakeQuantize();
        });
        gnalog() << "CNNNetPrevLayerSkipCertain for :: " << l->name << "returned: " << prevLayer->name << std::endl;
        return prevLayer;
    };


    // eltwise
    if (eltwise != nullptr) {
        // eltwise layer has 2 inputs, so depends on situation identity should or should not be inserted

        // for sum with 16-bit input precision
        //          if we have 4-4 inputs - we will handle that by inserting identity activation case (1)
        //          if we have 4-2 inputs - OK
        //          if we have 2-2 inputs - we need to insert diagonal

        // for sum with 8-bit input precision
        //          if we have 1-1 inputs - OK
        //          if we have 4-4 inputs - there are 2 options
        //              option 1 both inputs came from single outdata - we need to insert 1 identity activation to just convert single input into 1 byte
        //              option 2 each input came from its own outdata - we need to insert 2 identity activations to convert both and feed weights and inputs

        // for mul if we have 2-2 or 1-1 (low precision case) inputs - OK
        // for mul if we have 2-4 or 1-4 (low precision case) inputs - we need to insert identity activation to make 2 bytes input
        //                                                             or 1 byte input (low precision case)
        // for mul if we have 4-4 inputs - there are 2 options
        //          option 1 both inputs came from single outdata - we need to insert 1 identity activation to just convert single input into 2 bytes
        //                                                          or 1 byte (low precision case)
        //          option 2 each input came from its own outdata - we need to insert 2 identity activations to convert both and feed weights and inputs

        auto prev0 = PrevFunctionalLayer(l, 0);
        auto prev1 = PrevFunctionalLayer(l, 1);

        switch (eltwise->_operation) {
            case EltwiseLayer::Sub:
            case EltwiseLayer::Sum:
                if (!passmanager->isLowPrecision()) {
                    if (!LayerInfo(prev0).has32BOutput() || !LayerInfo(prev1).has32BOutput()) {
                        return prevLayers;
                    }
                    // TODO: whether there are possibility to select after what layer identity gets inserted
                    prevLayers.push_back(CNNNetPrevLayer(l, 0));
                } else {
                    if (LayerInfo(prev0).has8BOr16BOutput() && LayerInfo(prev1).has8BOr16BOutput()) {
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
                }
                break;
            case EltwiseLayer::Prod: {
                if (LayerInfo(prev0).has8BOr16BOutput() && LayerInfo(prev1).has8BOr16BOutput()) {
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

        auto prevLayer = PrevFunctionalLayer(l, LayerInfo(l).isGemm() ? 1 : 0);

        // No need to instert identity activation
        // when activation was already there before pooling
        // in case of CNN -> Activation -> Pooling order
        if (LayerInfo(prevLayer).isPooling()) {
            auto prevPrevLayer = PrevFunctionalLayer(prevLayer, 0);
            if (LayerInfo(prevPrevLayer).isActivation()) {
                return prevLayers;
            }
        }

        if (!LayerInfo(prevLayer).has32BOutput())
            return prevLayers;

        prevLayers.push_back(CNNNetPrevLayer(l, 0));
    }
    return prevLayers;
}

void InsertDiagonalLayerPass::run() {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "InsertDiagonalLayerPass");
    bool lowPrecision = getPassManager()->isLowPrecision();

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
            // in case of eltwise sum in 16-bit input precision, one of inputs is 4 bytes, the other is 2 bytes
            // in case of eltwise mul in 16-bit input precision, both inputs are 2 bytes
            // in case of eltwise sum in low (8-bit) input precision, both inputs are 1 byte
            // in case of eltwise mul in low (8-bit) input precision, both inputs are 1 byte
            // for e sum if we have 4-4 inputs we will handle that by inserting identity activation
            // for e sum if we have 4-2 - OK
            // for e sum if we have 2-2 inputs we need to insert diagonal -- handling here
            // for e sum if we have 1-1 inputs in low precision mode - OK
            // for e mul if we have 2-2 - OK
            // for e mul if we have 1-1 in low precision mode - OK
            // for e mul if we have 2-4 - inputs we need to insert identity to put 4 bytes input into weights
            // for e mul if we have 4-4 - inputs we need to insert 2 identities to put both 4 bytes input into weights

            if (eltwise->_operation != EltwiseLayer::Sum && eltwise->_operation != EltwiseLayer::Sub)
                continue;

            auto prevLayer1 = CNNNetPrevLayerSkipCertain(l, 1, [](CNNLayerPtr ptr) {
                return LayerInfo(ptr).isNonFunctional();
            });
            if (!LayerInfo(prevLayer).has8BOr16BOutput() || !LayerInfo(prevLayer1).has8BOr16BOutput())
                continue;

            if (lowPrecision && LayerInfo(prevLayer).has8BOr16BOutput() && LayerInfo(prevLayer1).has8BOr16BOutput())
                continue;
        }
        auto prevDirectLayer = CNNNetPrevLayer(l, 0);
        insertDiagonalLayerBetween(prevDirectLayer, l, getPassManager(), 1.f, 0);
    }
}

void HandleMultipleActivationsForTheLayerPass::run() {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "HandleMultipleActivationsForTheLayerPass");
    // found layer followed by multiple activations
    for (auto & l : *pLayers) {
        CNNLayerSet activations;

        for (auto && odata : l->outData) {
            for (auto && inputTo : getInputTo(odata)) {
                LayerInfo info(inputTo.second);

                if (info.isActivation()) {
                    if (odata->getDims().empty()) continue;
                    if (!activations.empty() && odata->getDims()[0] != 1) {
                        THROW_GNA_EXCEPTION << "Unsupported batch size " << odata->getDims()[0]
                                            << " for diagonal layer insertion";
                    }
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

void ForbidActivationFusingPass::run() {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "ForbidActivationFusingPass");
    for (auto& l : *pLayers) {
        if (LayerInfo(l).isActivation()) {
            auto prevLayer = CNNNetPrevLayer(l);
            if (LayerInfo(prevLayer).has32BOutput()) {
                // find all layers directly connected to the outputs of the previous layer
                const auto allUsingPrev = CNNNetGetAllNextLayersSkipCertain(prevLayer, -1,
                    [&](CNNLayerPtr nextLayer) -> bool {
                        for (const auto& input : nextLayer->insData) {
                            for (const auto& output : prevLayer->outData) {
                                if (areEqualDatas(input.lock(), output) &&
                                    areEqualDatas(l->insData[0].lock(), output) &&
                                    (LayerInfo(nextLayer).isEltwiseSum() || nextLayer == l)) {
                                    return false;
                                }
                            }
                        }
                        return true;
                    });
                if (allUsingPrev.size() > 1) {
                    // the weights of MAX_VAL_2B_WEIGHT are used to enforce 1.0 scale factor
                    // so the scores are more correct
                    insertDiagonalLayerBetween(prevLayer, l, getPassManager(), MAX_VAL_2B_WEIGHT);
                }
                continue;
            }
        }
    }
}

namespace {
    template<class T>
    bool is2D(T&& vec) {
        return vec.size() >= 2 && vec[0] > 1 && vec[1] > 1;
    }
} // namespace

void ReorderMaxPoolPass::run() {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "ReorderMaxPoolPass");
    // detecting following pattern
    // conv->activation->maxpooling
    // changing it to conv->maxpooling->activation
    for (auto & l : *pLayers) {
        auto pool = LayerInfo(l);
        if (!pool.isMaxPooling()) continue;

        // don't reorder if pooling is 2D for CNN2D
        auto pooling = dynamic_cast<PoolingLayer*>(l.get());
        // todo: return the check for stride after it'll be fixed in MO for Kaldi models
        if (pooling == nullptr || (is2D(pooling->_kernel))) continue;

        // checking prev layer type
        auto actLayer = CNNNetPrevLayer(l);
        auto activation = LayerInfo(actLayer);
        if (!activation.isActivation() || actLayer->insData.size() > 1) continue;

        // if activation came from convolution
        auto convolution = LayerInfo(CNNNetPrevLayer(static_cast<InferenceEngine::CNNLayer*>(activation)));
        if (!convolution.isConvolution()) continue;

        gnalog() << "MaxPooling: " << pool << ", reordered with activation: " << activation << "\n";

        CNNNetSwapLayers(activation, pool);
    }
}

void SubstituteSoftSignPass::run() {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "SubstituteSoftSignPass");
    //detecting following pattern
    // irv7 model:          irv10 model:
    // a layer                  a layer
    // |  \                     |  \
    // abs  \                   abs  \
    // |     |                  |     |
    // |     |                  add   |
    // |     |                  |     |
    // power |                  power |
    //  \   /                    \   /
    //    mul                      mul
    auto fp32eq = [](float p1, float p2) {
        return (std::abs(p1 - p2) <= 0.00001f * std::min(std::abs(p1), std::abs(p2)));
    };

    auto hasNChildren = [](CNNLayerPtr l, int N){
        if (l->outData.size() != 1) return false;
        if (getInputTo(l->outData.front()).size() != N) return false;
        return true;
    };
    auto getNthChild = [](CNNLayerPtr l, int N) {
        auto first = getInputTo(l->outData.front()).begin();
        auto last = getInputTo(l->outData.front()).end();
        IE_ASSERT(first != last);
        IE_ASSERT(N <= std::distance(first, last));
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
        auto addition = getNthChild(abs, 0);
        InferenceEngine::CNNLayerPtr power = nullptr;

        if (!LayerInfo(addition).isPower()) continue;
        auto powerLayer = LayerInfo(addition).as<PowerLayer*>();

        // first layer after abs must have scale of 1, offset of 1 and power of either 1 or -1
        if (!fp32eq(powerLayer->scale, 1.0f) || !fp32eq(powerLayer->offset, 1.0f) || !fp32eq(std::abs(powerLayer->power), 1.0f)) continue;
        // power == -1, offset = 1, scale = 1
        if (fp32eq(powerLayer->power, -1.0f)) {
            std::swap(addition, power);
        } else { // power = 1, offset = 1, scale - 1
            power = getNthChild(addition, 0);
            if (!LayerInfo(power).isPower()) continue;
            auto powerLayer_1 = LayerInfo(power).as<PowerLayer*>();
            // layer after addition must have power of -1, offset of 0 and scale of 1
            if (!fp32eq(powerLayer_1->power, -1.0f) || !fp32eq(powerLayer_1->offset, 0.0f) || !fp32eq(powerLayer_1->scale, 1.0f)) continue;
        }

        if (!hasNChildren(power, 1)) continue;
        auto mulSame = getNthChild(power, 0);
        if (mulSame != mul) continue;

        // pattern matched - lets substitute
        gnalog() << "SoftSign subgraph found consits of: \n"
                 << "\t" << abs->name << "\n";
        if (addition != nullptr) gnalog() << "\t" << addition->name << "\n";
        gnalog() << "\t" << mul->name << "\n"
                 << std::endl;

        // creating softsign layer
        auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(l);
        auto layerName = std::string("Synthetic_SoftSign_")
                + std::to_string(getPassManager()->getIntVar(softSignLayersCounter)++);

        CNNLayerPtr activationLayer =
                std::make_shared<GenericLayer>(LayerParams({layerName, "SoftSign", Precision::FP32}));
        IE_ASSERT(activationLayer != nullptr);
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
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "SubstitutePReluPass");
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
        IE_ASSERT(sum != nullptr);
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

void RemovePermutationsNHWCToNCHWPass::run() {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "RemovePermutationsNHWCToNCHWPass");
    std::set<CNNLayerPtr> permutations_to_remove;
    std::list<std::pair<CNNLayerPtr, CNNLayerPtr>> nhwc_layout_patterns;
    for (auto& l : *pLayers) {
        if (!LayerInfo(l).isConvolution()) {
            continue;
        }

        CNNLayerPtr prev, next;
        std::tie(prev, next) = FindPermutationsAroundConvolutionInNHWCModel(l);

        if (prev == nullptr || next == nullptr) continue;

        if (LayerInfo(prev).isPermute() || LayerInfo(prev).isPermuteViaReshape()) {
            permutations_to_remove.insert(prev);
        }

        if (LayerInfo(next).isPermute() || LayerInfo(prev).isPermuteViaReshape()) {
            permutations_to_remove.insert(next);
        }

        nhwc_layout_patterns.push_back({prev, next});
    }

    for (const auto& layers : nhwc_layout_patterns) {
        auto pattern_start = layers.first;
        auto pattern_end = layers.second;

        auto getTransposedLayout = [](InferenceEngine::DataPtr data) {
            size_t dims_size = data->getDims().size();
            if (dims_size < 3 || dims_size > 4) {
                THROW_GNA_EXCEPTION << data->getName() <<
                    " unexpected dimensions size in Permute - Conv - Permute pattern";
            }
            // HWC layout enum is used here as the only available in CNNNetwork for 3D vectors,
            // but the real layout is NCW and it's the one used in order vector later
            return dims_size == 4 ? Layout::NHWC : Layout::HWC;
        };

        auto setTransposedOrder = [getTransposedLayout](InferenceEngine::DataPtr data) {
            auto layout = getTransposedLayout(data);
            if (data->getLayout() == layout) return;

            auto current_layer = getCreatorLayer(data).lock();
            if (LayerInfo(current_layer).isConcat()) {
                auto concat_layer = dynamic_cast<InferenceEngine::ConcatLayer*> (current_layer.get());
                auto dims_size = data->getDims().size();
                concat_layer->_axis = (dims_size == 4 ? GetPermuteOrder(Layout::NHWC, Layout::NCHW) :
                    std::vector<int32_t>{0, 2, 1})[concat_layer->_axis];
            }

            // NWC->NCW layouts are used here for order vector, see comments a few lines above
            auto dims = data->getDims();
            auto order = dims.size() == 4 ? GetPermuteOrder(Layout::NCHW, Layout::NHWC) :
                std::vector<int32_t>{0, 2, 1};
            InferenceEngine::SizeVector new_dims;
            for (int i = 0; i < dims.size(); ++i) {
                new_dims.push_back(dims[order[i]]);
            }
            data->setDims(new_dims);
            data->setLayout(layout);
        };

        std::function<std::list<InferenceEngine::DataPtr>(CNNLayerPtr, const std::list<InferenceEngine::DataPtr>&)> getPathBetweenTransposes =
            [pattern_start, pattern_end, &getPathBetweenTransposes](CNNLayerPtr current_layer, const std::list<InferenceEngine::DataPtr>& path) {
            if (current_layer == pattern_end) {
                // the pattern end has been found, return the full path
                return path;
            }

            // transpose is reached, the pattern end hasn't been found
            if (current_layer != pattern_start &&
                (LayerInfo(current_layer).isPermute() || LayerInfo(current_layer).isPermuteViaReshape())) {
                return std::list<InferenceEngine::DataPtr>();
            }

            auto new_path(path);
            std::list<InferenceEngine::DataPtr> mergedChildPath;
            for (const auto& output : current_layer->outData) {
                new_path.push_back(output);
                for (const auto& input : getInputTo(output)) {
                    auto childPath = getPathBetweenTransposes(input.second, new_path);
                    // only the branch with the pattern end will return not empty list
                    if (!childPath.empty()) {
                        mergedChildPath.insert(std::end(mergedChildPath), std::begin(childPath), std::end(childPath));
                        break;
                    }
                }
            }

            return mergedChildPath;
        };

        auto path = getPathBetweenTransposes(pattern_start, std::list<InferenceEngine::DataPtr>());
        for (const auto& data : path) {
            setTransposedOrder(data);
        }

        if ((LayerInfo(pattern_start).isPermute() || LayerInfo(pattern_start).isPermuteViaReshape()) &&
         !getInputTo(pattern_start->outData.front()).empty()) {
            auto layer_before_permute = CNNNetPrevLayer(pattern_start);
            DataPtr output = nullptr;
            for (auto before_output : layer_before_permute->outData) {
                if (areEqualDatas(pattern_start->input(), before_output)) {
                    output = before_output;
                    output->setLayout(getTransposedLayout(output));
                    break;
                }
            }
            if (output == nullptr) {
                THROW_GNA_EXCEPTION << "Could not find correct data link between " << pattern_start->name << " and " << layer_before_permute->name;
            }
        }

        if (!pattern_end->outData.empty() && !getInputTo(pattern_end->outData.front()).empty()) {
            auto layer_after_permute = getInputTo(pattern_end->outData.front()).begin()->second;
            layer_after_permute->input()->setLayout(getTransposedLayout(layer_after_permute->input()));
        }
    }

    for (auto&& to_remove : permutations_to_remove) {
        gnalog() << to_remove->type << " layer '" << to_remove->name << "' will be removed" << '\n';
        CNNNetworkRemoveLayer(to_remove, false);
    }
}

void InsertIdentityLayerPass::run() {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "InsertIdentityLayerPass");
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(pLayers->front());
    auto createIdentityLayer = [quantized, this](const TensorDesc& tensorDesc) {
        int numOfIdentityLayers = this->getPassManager()->getIntVar(identityLayersCounterName)++;
        auto activationName = std::string("identity_") + std::to_string(numOfIdentityLayers);
        CNNLayerPtr activationLayer =
            std::make_shared<GenericLayer>(LayerParams({activationName, "identity", Precision::FP32}));
        CNNLayerPtr activationLayerWithQuant = quantized ?
                                        InferenceEngine::injectData<QuantizedLayerParams>(activationLayer) :
                                        activationLayer;
        auto dataPtr = std::make_shared<Data>("identity_data_" + std::to_string(numOfIdentityLayers), tensorDesc);
        getCreatorLayer(dataPtr) = activationLayerWithQuant;
        activationLayerWithQuant->outData.push_back(dataPtr);
        return activationLayerWithQuant;
    };

    for (auto & l : *pLayers) {
        if (LayerInfo(l).isPooling()) {
            // Identity should be inserted after 1D pooling if it's the last functional layer.
            auto pooling = LayerInfo(l).as<PoolingLayer*>();
            IE_ASSERT(pooling != nullptr);
            if (is2D(pooling->_kernel)) continue;

            auto hasNextFuncLayer = CNNNetHasNextLayerSkipCertain(l, 0, 0, [](CNNLayerPtr layer) {
                return LayerInfo(layer).isNonFunctional();
            });
            if (hasNextFuncLayer) continue;

            auto identityLayer = createIdentityLayer(l->outData[0]->getTensorDesc());
            gnalog() << "Inserted "<< identityLayer->name << " after " << l->name << std::endl;

            auto nextLayer = CNNNetCheckNextLayerSkipCertain(l, 0, 0, true, [](CNNLayerPtr layer) { return false; }).first;
            CNNNetworkInsertLayer(l, nextLayer, identityLayer);
        }

        for (auto && prev : getCandidatesForIdentityInsertion(l, getPassManager())) {
            // Do an upstream search until Functional layer is found
            auto original_prev_layer = prev;
            auto true_layer = l;
            while (LayerInfo(prev).isNonFunctional()) {
                if (CNNNetHasPrevLayer(prev.get()) && prev->outData.size() == 1) {
                    true_layer = prev;
                    prev = CNNNetPrevLayer(prev);
                } else {
                    gnawarn() << "Could not find Functional parent for " << original_prev_layer->name << ", using original layer";
                    prev = original_prev_layer;
                    true_layer = l;
                    break;
                }
            }
            // check if prev layer have id layer already connected to output
            // if so reuse it instead of create new one
            bool reconnected = false;
            for (auto prev_layer_output : prev->outData) {
                // prev ---------+--> identity --> layer XYZ
                //               |
                //               |  <= here we want to inject identity
                //               |
                //               +--> l layer
                // but we may just connect l layer with existing identity
                for (auto&& next_layer : getInputTo(prev_layer_output)) {
                    auto child_of_prev_layer = next_layer.second;
                    if (child_of_prev_layer.get() == true_layer.get()) {
                        continue;
                    } else if (LayerInfo(child_of_prev_layer).isIdentity()) {
                        CNNNetworkReconnectLayer(prev, child_of_prev_layer, true_layer);
                        reconnected = true;
                        break;
                    }
                }
            }
            if (reconnected)
                continue;

            // TODO: why index is 0 ? - better use direct indexing in getCandidateFunction
            // detecting ins-data-idx
            size_t insDataIdx = std::numeric_limits<size_t>::max();
            for (size_t i = 0; i != true_layer->insData.size(); i++) {
                if (getCreatorLayer(true_layer->insData[i].lock()).lock() == prev) {
                    insDataIdx = i;
                    break;
                }
            }
            if (insDataIdx == std::numeric_limits<size_t>::max()) {
                THROW_GNA_EXCEPTION << "cannot insert identity layer after" << prev->name << " and before " << true_layer->name;
            }

            auto inputData = true_layer->insData[insDataIdx].lock();
            auto identityLayer = createIdentityLayer(inputData->getTensorDesc());

            gnalog() << "Inserted "<< identityLayer->name << " between: " << prev->name << " and " << true_layer->name << "\n" << std::flush;

            // copy offset - to be used while connecting outputs
            if (prev->params.find("output_offset") != prev->params.end()) {
                identityLayer->params["output_offset"] = prev->params["output_offset"];
            }
            // copy offset - to be used while connecting outputs
            if (prev->params.find("original_num_rows") != prev->params.end()) {
                identityLayer->params["original_num_rows"] = prev->params["original_num_rows"];
            }

            CNNNetworkInsertLayer(prev, CNNLayerPtr(nullptr), identityLayer);
        }
    }
}

void InsertCopyLayerPass::run() {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "InsertCopyLayerPass");
    using FuncChildrenInfo = std::tuple<
        CNNLayerPtr,   // parent layer
        CNNLayerPtr,   // child layer
        int32_t        // input index
    >;
    // recursively searches for children functional layers skipping non-functional ones
    std::function<std::vector<FuncChildrenInfo>(CNNLayerPtr, CNNLayerPtr, int32_t)> find_func_layers =
        [&find_func_layers](CNNLayerPtr currentLayer, CNNLayerPtr parentLayer, int32_t input_idx) {
        if (!LayerInfo(currentLayer).isNonFunctional() ||
            currentLayer->outData.size() == 0 ||
            getInputTo(currentLayer->outData[0]).size() == 0) {
            return std::vector<FuncChildrenInfo>{std::make_tuple(parentLayer, currentLayer, input_idx)};
        }
        std::vector<FuncChildrenInfo> results;
        for (size_t i = 0; i < getInputTo(currentLayer->outData[0]).size(); ++i) {
            auto next_layer = CNNNetGetNextLayerSkipCertain(currentLayer, 0, i,
                [](CNNLayerPtr origin) {return false; }).first;
            auto result = find_func_layers(next_layer, currentLayer,
                CNNLayerFindInsDataIdxes(currentLayer->outData[0], next_layer)[0]);
            results.insert(std::end(results), std::begin(result), std::end(result));
        }
        return results;
    };

    // Copy layer insertion happens in few cases:
    // Crop output goes to concat layer -> copy layer insertion
    // Splitted part of input goes to concat layer -> copy layer insertion
    // Concat|Split|Crop layer goes to memory layer -> delayed copy layer insertion
    // One output goes to multiple concat and/or memory layers -> delayed copies before memory layers
    // and copies before concat layers (one less copy than outputs)
    // Concat has multiple connections to the same input
    // Subgraph has only non-functional layers
    for (auto & l : *pLayers) {
        if (!LayerInfo(l).isConcat()) continue;

        // Insert copy layers after concat inputs with multiple connections to concat
        std::set<DataPtr> parents;
        for (size_t input_idx = 0; input_idx < l->insData.size(); ++input_idx) {
            IE_ASSERT(l->insData[input_idx].lock() != nullptr);
            auto inputData = l->insData[input_idx].lock();
            if (parents.find(inputData) != std::end(parents)) {
                auto parent = getCreatorLayer(inputData);
                IE_ASSERT(parent.lock() != nullptr);
                InsertCopyLayer(parent.lock(), l, input_idx, this->getPassManager(), CopyLayerName);
            } else {
                parents.insert(inputData);
            }
        }
    }

    for (auto & l : *pLayers) {
        if (LayerInfo(l).isNonFunctional()) continue;

        // Crop -> Concat, Input -> Split -> Concat and Concat -> Memory cases
        if ((LayerInfo(l).isCrop() && !LayerInfo(l).isCropAffined()) || LayerInfo(l).isConcat() || LayerInfo(l).isSplit()) {
            std::vector<FuncChildrenInfo> copy_insertion_tuples;
            std::vector<FuncChildrenInfo> delayed_copy_insertion_tuples;
            for (auto output : l->outData) {
                auto& inputTo = getInputTo(output);
                for (auto& childLayer : inputTo) {
                    std::vector<int> connections = CNNLayerFindInsDataIdxes(output, childLayer.second);
                    for (auto input_idx : connections) {
                        auto children_info = find_func_layers(childLayer.second, l, input_idx);
                        for (const auto &child_info : children_info) {
                            CNNLayerPtr child = std::get<1>(child_info);
                            if ((LayerInfo(l).isConcat() || LayerInfo(l).isCrop() || LayerInfo(l).isSplit()) && LayerInfo(child).isMemory()) {
                                // Concat|Split|Crop -> Memory case
                                delayed_copy_insertion_tuples.push_back(child_info);
                            } else if ((LayerInfo(l).isSplit() || LayerInfo(l).isCrop()) && LayerInfo(child).isConcat()) {
                                // Split|Crop -> Concat case
                                // concat may be connected to previous layer with multiple connections
                                copy_insertion_tuples.push_back(child_info);
                            }
                        }
                    }
                }
            }

            for (auto& tuple : delayed_copy_insertion_tuples) {
                // Concat -> Memory case
                InsertCopyLayer(std::get<0>(tuple), std::get<1>(tuple), std::get<2>(tuple), this->getPassManager(), DelayedCopyLayerName);
            }
            for (auto& tuple : copy_insertion_tuples) {
                // Crop -> Concat case
                InsertCopyLayer(std::get<0>(tuple), std::get<1>(tuple), std::get<2>(tuple), this->getPassManager(), CopyLayerName);
            }
        }

        // Layer -> multiple concat/memory case
        for (auto output : l->outData) {
            std::vector<std::pair<CNNLayerPtr, size_t>> MemoryLayers;
            std::vector<std::pair<CNNLayerPtr, size_t>> ConcatLayers;
            auto& inputTo = getInputTo(output);
            if (inputTo.size() < 2) continue;
            for (auto& childLayer : inputTo) {
                auto layer_to_insert = childLayer.second;
                auto current_layer = childLayer.second;
                auto previous_layer = l;
                size_t input_idx = CNNLayerFindInsDataIdxes(output, current_layer)[0];

                while (LayerInfo(current_layer).isNonFunctional()) {
                    if (current_layer->outData.size() == 0) break;
                    if (getInputTo(current_layer->outData[0]).size() == 0) break;
                    previous_layer = current_layer;
                    current_layer = CNNNetGetNextLayerSkipCertain(current_layer, 0, 0, [](CNNLayerPtr origin){return false;}).first;
                }
                if (LayerInfo(current_layer).isConcat()) {
                    ConcatLayers.push_back(make_pair(layer_to_insert, input_idx));
                } else if (LayerInfo(current_layer).isMemory()) {
                    MemoryLayers.push_back(make_pair(layer_to_insert, input_idx));
                }
            }
            if (MemoryLayers.empty() && ConcatLayers.empty()) continue;
            auto toCopyCount = MemoryLayers.size() + ConcatLayers.size() - (LayerInfo(l).isInput() ? 0 : 1);
            size_t currentCopyIdx = 0;
            while (currentCopyIdx < toCopyCount) {
                if (currentCopyIdx < MemoryLayers.size()) {
                    size_t memoryIdx = currentCopyIdx;
                    auto memoryLayer = MemoryLayers[memoryIdx].first;
                    auto inputIdx = MemoryLayers[memoryIdx].second;
                    InsertCopyLayer(l, memoryLayer, inputIdx, this->getPassManager(), DelayedCopyLayerName);
                } else {
                    size_t concatIdx = currentCopyIdx - MemoryLayers.size();
                    auto concatLayer = ConcatLayers[concatIdx].first;
                    auto inputIdx = ConcatLayers[concatIdx].second;
                    InsertCopyLayer(l, concatLayer, inputIdx, this->getPassManager(), CopyLayerName);
                }
                currentCopyIdx++;
            }
        }
    }

    for (auto & l : *pLayers) {
        if (l->outData.size() > 0 &&
            getInputTo(l->outData[0]).size() > 0) continue;

        bool bNeedInsertCopyLayer = true;
        CNNNetDFS(l, [&l, &bNeedInsertCopyLayer](CNNLayerPtr layer) {
            if (!(LayerInfo(layer).isNonFunctional() || LayerInfo(layer).isSplit() || LayerInfo(layer).isCrop() || LayerInfo(layer).isInput())) {
                bNeedInsertCopyLayer = false;
            }
            }, true, [&bNeedInsertCopyLayer](InferenceEngine::CNNLayer* from) {
                    // aborting UFS if we found functional layer (excluding Splits and Crops)
                    return make_upstream_order(bNeedInsertCopyLayer ? from : nullptr);
            });

        if (bNeedInsertCopyLayer) {
            for (size_t inputIdx = 0; inputIdx < l->insData.size(); ++inputIdx) {
                IE_ASSERT(l->insData[inputIdx].lock() != nullptr);
                auto inputData = l->insData[inputIdx].lock();
                auto parentLayer = getCreatorLayer(inputData);
                IE_ASSERT(parentLayer.lock() != nullptr);
                InsertCopyLayer(parentLayer.lock(), l, inputIdx, this->getPassManager(), CopyLayerName);
            }
        }
    }
}

void FlattenTrivialConcatPass::run() {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "FlattenTrivialConcatPass");
    // change all trivial concatenations (concatenation where output buffer is a buffer made by appending input buffers)
    // by reshaping its inputs to 1 x total_input_size and its output to 1 x total_concat_size and changing the axis to 1
    // for example if 4D concat have unaligned inputs then ConcatAlignFilters need to be used; if sizes before concat
    // axis are all ones then concat can be changed to 2D for example, let's say all inputs have the same shape equal to:
    // 1, 1, 5, 3 then for axis 0, 1, 2 the change will be made and inputs will be reshaped to 1, 15,
    // but for shape 2, 1, 5, 3 only axis 0 is valid and in such case inputs will be reshaped to 1, 30
    // TODO: detection of trivial cases could be moved to one common place when all transformations are migrated to ngraph.
    // See as well code for detection of unsupported concat
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(pLayers->front());

    auto getLayerByIndex = [](int idx, ConcatLayer* concatLayer) {
        auto input = concatLayer->insData[idx];
        auto lockedInput = input.lock();
        if (!lockedInput) {
            THROW_GNA_EXCEPTION << "cannot get insdata : "<< idx << " for layer: " << concatLayer->name;
        }
        return lockedInput;
    };

    for (auto & l : *pLayers) {
        LayerInfo info(l);
        auto concatLayer = info.as<ConcatLayer*>();
        if (!concatLayer) continue;
        if (concatLayer->insData.size() < 1) continue;

        // Skip obvious supported cases
        auto dims_size = concatLayer->insData[0].lock()->getDims().size();
        if (dims_size < 2 || concatLayer->_axis == dims_size - 1) continue;

        // Skip cases which cannot be flattened (these might be unsupported at all)
        auto axis = concatLayer->_axis;
        bool skip_layer = false;
        for (unsigned int i = 0; i < axis; i++) {
            if (concatLayer->insData[0].lock()->getDims()[i] != 1) skip_layer = true;
        }
        if (skip_layer) continue;

        // Calculate total input sizes
        std::vector<size_t> total_sizes;
        for (auto& input : concatLayer->insData) {
            auto input_dims = input.lock()->getDims();
            total_sizes.push_back(std::accumulate(input_dims.begin(), input_dims.end(), size_t(1), std::multiplies<size_t>()));
        }

        // Reshape concat inputs
        for (size_t input_idx = 0; input_idx != concatLayer->insData.size(); input_idx++) {
            auto concatInput = getLayerByIndex(input_idx, concatLayer);

            auto tensor = InferenceEngine::TensorDesc(concatInput->getTensorDesc());
            tensor.reshape(SizeVector({1, total_sizes[input_idx]}), Layout::NC);
            auto reshapeName = l->name + "_input_"+ std::to_string(input_idx) +"_reshape";
            auto reshape = CNNNetworkCreateReshape(tensor, reshapeName, quantized);

            CNNNetworkInsertLayer(getCreatorLayer(concatInput).lock(), l, reshape);
            gnalog() << "\tInserted " << reshapeName << " between " << getCreatorLayer(concatInput).lock()->name << " and " << l->name << std::endl;
        }

        // Reshape concat outputs back to the original size
        for (auto output_idx = 0; output_idx != concatLayer->outData.size(); output_idx++) {
            auto output = concatLayer->outData[output_idx];
            auto output_tensor_copy = TensorDesc(output->getTensorDesc());

            auto dims = output_tensor_copy.getDims();
            auto total_size = std::accumulate(dims.begin(), dims.end(), size_t(1), std::multiplies<size_t>());

            auto new_tensor = output->getTensorDesc();
            new_tensor.reshape(SizeVector({1, total_size}), Layout::NC);

            auto new_output = CNNReplaceDataWithChangedTensorDescription(output, new_tensor);
            gnalog() << "\tChanged " << output->getName() << " dims to 2D" << std::endl;

            auto reshapeName = l->name + "_output_"+ std::to_string(output_idx) +"_reshape";

            auto reshape = CNNNetworkCreateReshape(output_tensor_copy, reshapeName, quantized);
            if (getInputTo(new_output).empty()) {
                reshape->insData.push_back(new_output);
                getInputTo(new_output)[reshape->name] = reshape;
            } else {
                CNNNetworkInsertLayer(l, nullptr, reshape, output_idx);
            }
            gnalog() << "\tInserted " << reshapeName << " after " << l->name << std::endl;
        }

        concatLayer->_axis = 1;
    }
}

void InsertConcatAligningFilterPass::run() {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "InsertConcatAligningFilterPass");
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(pLayers->front());
    // currently concat layer only supports 2 bytes in int16 and int8 mode. In fp32 mode this no necessary but usefull for testing
    const int bytesPerConcatElement = 2;

    int numOfFilterLayers = 0;

    for (auto & l : *pLayers) {
        LayerInfo info(l);
        if (!info.isConcat()) continue;
        size_t offset = 0;
        auto concatLayer = info.as<ConcatLayer*>();
        IE_ASSERT(concatLayer != nullptr);

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
                    THROW_GNA_EXCEPTION << "unsupported concat input of dims.size()=" << dims.size() << ", layer=" << prevLayer->name;
                }

                auto num_rows_in = dims[1];
                size_t aligned64_offset = std::max(0, static_cast<int>(ALIGN64(offset) - 64));
                size_t num_rows_padded = (offset - aligned64_offset) / bytesPerConcatElement;
                size_t num_rows_out = num_rows_padded + num_rows_in;

                // encodes offset to beginning of split layer input
                size_t bytesOffset = (aligned64_offset / bytesPerConcatElement) * (quantized ? bytesPerConcatElement : 4);
                concatAligningFilter->params["output_offset"] =
                        std::to_string(bytesOffset);

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
                if (!concatAligningFilter->_weights->buffer().as<float*>()) {
                    THROW_GNA_EXCEPTION << "Failed to allocate weights of size " << filterWeights.size() << " for " << filterName;
                }

                CopyVectorToBlob(concatAligningFilter->_weights, filterWeights);

                // modifying output rows to be used - to avoid modification to original concat we are store num of elements in params
                dims[1] = num_rows_out;

                if ((concatInput->getLayout() == Layout::NC && dims[0] > 8) ||
                    (concatInput->getLayout() == Layout::CN && dims[1] > 8)) {
                    THROW_GNA_EXCEPTION << "unsupported batch number '" <<
                        (concatInput->getLayout() == Layout::NC ? dims[0] : dims[1]) <<
                        "' in layer '" << concatLayer->name << "'";
                }

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
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "ReorderConcatInputsPass");
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(pLayers->front());
    int numOfLinkLayers = 0;

    for (auto& l : *pLayers) {
        // 1st stage locate concat
        LayerInfo info(l);
        if (!info.isConcat()) {
            continue;
        }

        // 2nd stage locate first input in concat
        if (l->insData.size() < 2) {
            THROW_GNA_EXCEPTION << "Concat layer has unsupported number of incoming layers: " << l->name;
        }

        auto concatLayer = info.as<ConcatLayer*>();
        auto getLayerByIndex = [&concatLayer](int idx) {
            auto input = concatLayer->insData[idx];
            auto lockedInput = input.lock();
            if (!lockedInput) {
                THROW_GNA_EXCEPTION << "cannot get insdata : " << idx << " for layer: " << concatLayer->name;
            }
            return lockedInput;
        };

        for (auto input_idx = 1; input_idx != concatLayer->insData.size(); input_idx++) {
            auto concatInput = getLayerByIndex(input_idx);
            auto currConcatLayer = getCreatorLayer(concatInput).lock();

            LayerInfo infoConcatInput(currConcatLayer);
            if (!infoConcatInput.isConcatAlignFilter()) {
                continue;
            }

            auto inputsToConcatPrev = CNNNetGetPrevLayersSkip(l, [](CNNLayerPtr origin) {
                return !LayerInfo(origin).isNonFunctional() && !LayerInfo(origin).isSplit();
                }, input_idx - 1);

            if (inputsToConcatPrev.empty()) {
                THROW_GNA_EXCEPTION << "cannot locate first input into concat layer: " << currConcatLayer;
            }

            auto prevInputToConcat = inputsToConcatPrev.front().first;

            // concat has first input of concat align filter - dont need to reorder it
            if (prevInputToConcat == currConcatLayer) {
                continue;
            }

            bool bFinish = false;
            // making a link activation possible without extra layer if first input to concat not a parent / indirect parent of second input
            // using ufs - upper first search
            gnalog() << "[UFS] searching for: " << prevInputToConcat->name << "\n";

            CNNNetDFS(currConcatLayer, [&currConcatLayer, &prevInputToConcat, &bFinish](CNNLayerPtr layer) {
                gnalog() << "[UFS] from : " << currConcatLayer->name << " reached: " << layer->name << "\n";
                // found that direct input to concat is a indirect parent of align filter - so no link required
                if (layer.get() == prevInputToConcat.get() || LayerInfo(prevInputToConcat).isInput()) {
                    gnalog() << "[UFS] copy layer insertion needed\n";
                    bFinish = true;
                }
                }, true, [&bFinish](InferenceEngine::CNNLayer* from) {
                    // aborting UFS once link not needed
                    return make_upstream_order(!bFinish ? from : nullptr);
                });

            auto linkName = std::string("link_") + std::to_string(numOfLinkLayers++);

            auto linkWithoutQuant = std::make_shared<CNNLayer>(LayerParams({ linkName, "link", Precision::FP32 }));

            auto link = quantized ?
                InferenceEngine::injectData<QuantizedLayerParams>(linkWithoutQuant) :
                linkWithoutQuant;


            auto linkOutData = std::make_shared<Data>(linkName,
                TensorDesc(Precision::FP32,
                    SizeVector({ 1 }),
                    Layout::C));
            getCreatorLayer(linkOutData) = link;

            link->outData.push_back(linkOutData);
            link->insData.push_back(currConcatLayer->outData.front());

            getInputTo(linkOutData)[prevInputToConcat->name + ".via.link"] = prevInputToConcat;
            prevInputToConcat->insData.push_back(linkOutData);

            getInputTo(currConcatLayer->outData.front())[linkName] = link;
        }
    }
}

void InsertSplitAligningFilterPass::run() {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "InsertSplitAligningFilterPass");

    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(pLayers->front());

    int numOfFilterLayers = 0;
    for (auto &l : *pLayers) {
        auto info = LayerInfo(l);
        if (!info.isSplit() && !info.isSlice()) {
            continue;
        }

        auto outFunctionalLayers = CNNNetGetAllNextLayersSkipCertain(l, -1, [](CNNLayerPtr next_layer) {
            return GNAPluginNS::LayerInfo(next_layer).isNonFunctional();
        });
        size_t padding = 0;
        for (auto &&outFunctionalLayer : outFunctionalLayers) {
            padding = std::max(padding, LayerInfo(outFunctionalLayer).paddingSize());
        }

        size_t currentOffset = 0;
        int splitOutIndex = 0;
        for (auto &&splitOutput  : l->outData) {
            auto outputSize = product(begin(splitOutput->getDims()), end(splitOutput->getDims()));

            if ((currentOffset != ALIGN64(currentOffset)) || (padding != 0)) {
                // check that this split output actually connected to further layers
                if (getInputTo(splitOutput).empty()) {
                    gnalog() << "Output port: " << splitOutIndex << " of " << l->name << " unconnected, skipping\n";
                } else {
                    if (splitOutput->getDims().size() > 1 && splitOutput->getDims().front() > 1) {
                        THROW_GNA_EXCEPTION << l->name << " Convolution Filter doesn't support batch="
                            << splitOutput->getDims().front();
                    }

                    // this split output not beginning from 64 bytes aligned boundary - need to correct by aligning filter layer
                    // insert the filter
                    auto filterName = std::string("AlignFilter_") + std::to_string(numOfFilterLayers++);

#ifdef PLOT
                    // getting list of layers attached to current split output
                    gnalog() << "Inserted Affine Filter: " << filterName << " between: " << l->name << " and ";
                    for (auto &&followingLayers : getInputTo(splitOutput)) {
                        if (getInputTo(splitOutput).size() != 1) {
                            gnalog() << "\n    ";
                        }
                        gnalog() << followingLayers.second->name;
                    }
                    gnalog() << std::endl;
#endif
                    auto filterLayer =
                            std::make_shared<ConvolutionLayer>(LayerParams({filterName, "ConvolutionFilter", Precision::FP32}));

                    auto inputData = splitOutput;

                    size_t aligned64_offset = std::max(0, static_cast<int>(ALIGN64(currentOffset) - 64));

                    IE_ASSERT(filterLayer != nullptr);

                    // encodes offset to beginning of split layer input
                    filterLayer->params["offset"] = std::to_string(aligned64_offset / GNALimitations::bytesPerSplitElement);
                    auto dims = splitOutput->getTensorDesc().getDims();
                    if (dims.size() > 3) {
                        THROW_GNA_EXCEPTION << "unsupported split layer dims size: " << dims.size();
                    }

                    const auto offsetOfUnalignment = (currentOffset - aligned64_offset) / GNALimitations::bytesPerSplitElement;
                    // TODO consider to use a different number of filters do decrese the number of trailing zeros (additionalPaddingOfFilter)
                    const auto numberOfFilters = GNALimitations::convMinFiltersNum;
                    const auto filterSize = ALIGN(offsetOfUnalignment + numberOfFilters, GNALimitations::convFilterSizeDivider);

                    // filterWeights: numberOfFilters X (offsetOfUnalignment + additionalPaddingOfFilter + numberOfFilters)
                    // offsetOfUnalignment - the leading zeros in the filter
                    //       |
                    //       |             additionalPaddingOfFilter = filterSize - offsetOfUnalignment - numberOfFilters
                    //   ____|___         ___|___
                    //  |        |       |       |
                    //  0 0 ... 0 1 0 0 0 0 ... 0
                    //  0 0 ... 0 0 1 0 0 0 ... 0
                    //  0 0 ... 0 0 0 1 0 0 ... 0
                    //  0 0 ... 0 0 0 0 1 0 ... 0
                    std::vector<float> filterWeights(filterSize * 4, 0.f);
                    for (auto f = 0u; f < numberOfFilters; f++) {
                        filterWeights[f * filterSize + f + offsetOfUnalignment] = 1;
                    }

                    filterLayer->_out_depth = numberOfFilters;
                    filterLayer->_stride_x = numberOfFilters;
                    filterLayer->_stride_y = 1;
                    filterLayer->_kernel_x = filterSize;
                    filterLayer->_kernel_y = 1;
                    filterLayer->_padding_x = 0;
                    filterLayer->_padding_y = 0;

                    filterLayer->_weights = make_shared_blob<float>(TensorDesc(
                            inputData->getTensorDesc().getPrecision(),
                            SizeVector({filterWeights.size()}),
                            Layout::C));
                    filterLayer->_weights->allocate();
                    CopyVectorToBlob(filterLayer->_weights, filterWeights);

                    std::vector<float> biasWeights(numberOfFilters, 0.f);

                    filterLayer->_biases = make_shared_blob<float>(TensorDesc(
                        inputData->getTensorDesc().getPrecision(),
                        SizeVector({ biasWeights.size() }),
                        Layout::C));
                    filterLayer->_biases->allocate();
                    CopyVectorToBlob(filterLayer->_biases, biasWeights);

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
            }

            // search data that starts from unaligned location
            currentOffset += outputSize * GNALimitations::bytesPerSplitElement;
            splitOutIndex++;
        }
    }
}

static InferenceEngine::Blob::Ptr tileBlob(Blob::Ptr& blob, size_t TileTo) {
    auto weightsElements = blob->size();
    auto weightsBytes = blob->byteSize();
    if (weightsElements == 0) {
        IE_THROW() << "Blob size is 0";
    }

    auto tiledBlob = make_plain_blob(blob->getTensorDesc().getPrecision(), { TileTo });
    tiledBlob->allocate();

    for (int i = 0; i < (TileTo / weightsElements); ++i) {
        ie_memcpy(tiledBlob->buffer().as<uint8_t*>() + i * weightsBytes, weightsBytes, blob->cbuffer(), weightsBytes);
    }
    return tiledBlob;
}

void EltwiseSplitOverChannelsPass::run() {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "EltwiseSplitOverChannelsPass");

    for (auto & l : *pLayers) {
        if (!LayerInfo(l).isEltwise()) {
            continue;
        }
        auto masterEltwise = std::dynamic_pointer_cast<EltwiseLayer>(l);
        IE_ASSERT(masterEltwise != nullptr);

        if (l->outData.size() != 1) {
            THROW_GNA_LAYER_EXCEPTION(l) << "number of outputs expected to be 1";
        }
        auto oData = l->outData.front();
        auto oDims = oData->getDims();
        auto totalElementsSize = details::product(std::begin(oDims), std::end(oDims));
        if (totalElementsSize <= GNALimitations::bufferMaxSize) {
            continue;
        }

        auto firstValuableDim = std::find_if(std::begin(oDims), std::end(oDims), [](size_t val) { return val > 1; });
        IE_ASSERT(firstValuableDim != std::end(oDims));
        auto splittedElementsSize = *firstValuableDim;
        auto splittedDimIx = std::distance(std::begin(oDims), firstValuableDim);
        auto alignment = GNALimitations::inputByteAlignment;

        // Split output size should be multiple by 64 to avoid align filters insertion,
        // but we need to check if our input size to split exceeds 64; if not we can always
        // split if the remaining size is aligned
        if (splittedElementsSize <= 64) {
            if ((totalElementsSize / splittedElementsSize) % alignment == 0) {
                alignment = 1;
            } else {
                THROW_GNA_LAYER_EXCEPTION(l) << "splitting didn't succeed\n";
            }
        }

        auto splitSizes = GetAlignedSplitSizes(splittedElementsSize,
            GNALimitations::bufferMaxSize * splittedElementsSize / totalElementsSize, alignment);

        pass_trace() << "transforming " << LAYER_NAME(l) << " by splitting it to multiple eltwise operations\n";
        auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(l);

        bool sameInputs = l->insData[0].lock() == l->insData[1].lock();
        std::vector<CNNLayerPtr> splitLayers(2);
        for (size_t kThEltwiseInput = 0; kThEltwiseInput != 2; kThEltwiseInput++) {
            // create split layer
            auto splitRaw = std::make_shared<SplitLayer>(
                    LayerParams{l->name + "/split/" + std::to_string(kThEltwiseInput), "Split", Precision::FP32});
            auto split = quantized ? InferenceEngine::injectData<QuantizedLayerParams>(splitRaw) : splitRaw;
            splitLayers[kThEltwiseInput] = split;

            split->insData.push_back(l->insData[kThEltwiseInput]);
            auto inputDesc = l->insData[kThEltwiseInput].lock()->getTensorDesc();

            // create split layer outputs
            for (auto elementsNum : splitSizes) {
                auto newDims = oDims;
                newDims[splittedDimIx] = elementsNum;
                auto newDesc = TensorDesc(inputDesc.getPrecision(), newDims, inputDesc.getLayout());
                auto data = std::make_shared<Data>(l->name + "/" + std::to_string(kThEltwiseInput) + "/1", newDesc);
                getCreatorLayer(data) = split;
                split->outData.push_back(data);
            }
            // replacing connection X->eltwise to X->split
            auto oData = CNNLayerFindOutData(l, kThEltwiseInput);
            oData.second->second = split;

            if (sameInputs) {
                splitLayers[1] = splitLayers[0];
                break;
            }
        }

        // create concatlayer
        auto concatRaw = std::make_shared<ConcatLayer>(
                LayerParams{l->name + "/concat", "Concat", Precision::FP32});
        auto concat = quantized ? InferenceEngine::injectData<QuantizedLayerParams>(concatRaw) : concatRaw;

        concat->outData.push_back(masterEltwise->outData.front());
        getCreatorLayer(masterEltwise->outData.front()) = concat;

        for (size_t k = 0; k != splitSizes.size(); k++) {
            auto eltwiseRaw = std::make_shared<EltwiseLayer>(
                    LayerParams{l->name + "/eltwise/" + std::to_string(k), "Eltwise", Precision::FP32});
            IE_ASSERT(eltwiseRaw != nullptr);
            eltwiseRaw->_operation = masterEltwise->_operation;
            eltwiseRaw->coeff = masterEltwise->coeff;
            auto eltwise = quantized ? InferenceEngine::injectData<QuantizedLayerParams>(eltwiseRaw) : eltwiseRaw;

            eltwise->insData.push_back(splitLayers[0]->outData[k]);
            eltwise->insData.push_back(splitLayers[1]->outData[k]);
            getInputTo(splitLayers[0]->outData[k])[eltwise->name] = eltwise;
            getInputTo(splitLayers[1]->outData[k])[eltwise->name] = eltwise;

            SizeVector newDims = splitLayers[1]->outData[k]->getDims();
            auto newDesc = TensorDesc(splitLayers[1]->outData[k]->getPrecision(), newDims,
                    splitLayers[1]->outData[k]->getLayout());
            auto data = std::make_shared<Data>(l->name + "/elwise/out/" + std::to_string(k), newDesc);
            getCreatorLayer(data) = eltwise;
            eltwise->outData.push_back(data);
            if (quantized) {
                auto eltwiseQuant = InferenceEngine::getInjectedData<QuantizedLayerParams>(eltwise);
                if (quantized->_src_quant.IsStatsSet()) {
                    eltwiseQuant->_src_quant.CopyStats(quantized->_src_quant);
                }
                if (quantized->_dst_quant.IsStatsSet()) {
                    eltwiseQuant->_dst_quant.CopyStats(quantized->_dst_quant);
                }
            }
            getInputTo(data)[concat->name] = concat;
            concat->insData.push_back(data);
        }
    }
}

void SubstituteScaleShiftBroadCastPass::run() {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "SubstituteScaleShiftBroadCastPass");
    std::map<std::string, InferenceEngine::SizeVector> reshaped_data;
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(pLayers->front());

    for (auto & l : *pLayers) {
        LayerInfo layerInfo(l);

        if (!layerInfo.isScaleShift()) {
            continue;
        }

        auto scaleShift = layerInfo.as<ScaleShiftLayer*>();
        IE_ASSERT(scaleShift != nullptr);

        auto insData = scaleShift->insData.front().lock();
        if (!insData) {
            THROW_GNA_EXCEPTION << "Cannot get inputs data for layer: " << l->name;
        }

        bool was_reshaped = reshaped_data.count(insData->getName()) != 0;
        bool reshape_batch = HasTo2DReshapeData(l);
        InferenceEngine::SizeVector dataDims;
        if (was_reshaped) {
            dataDims = reshaped_data[insData->getName()];
        } else {
            dataDims = HasTo2DReshapeData(l) ?
                Get2DReshapedData(insData, GNALimitations::GetMinBatchToFitInBuffer(insData), 8)->getDims() :
                insData->getDims();
        }

        if (dataDims.size() <= 2) {
            // NC or C cannot do broadcast
            continue;
        }

        auto batchSize = dataDims[0];
        auto nElements = product(begin(dataDims), end(dataDims)) / batchSize;
        auto weightsElements = scaleShift->_weights->size();

        if (!reshape_batch && nElements == weightsElements) {
            continue;
        }

        // TODO: add broadcasting rules checks

        gnalog() << "Substitution ScaleShift broadcast for layer: " << l->name << "\n";
        if (nElements % scaleShift->_weights->size()) {
            THROW_GNA_EXCEPTION << "Cannot tile weights for layer: " << l->name << ", due to weights size not GCD of dims product";
        }
        scaleShift->_weights = tileBlob(scaleShift->_weights, nElements);
        if (scaleShift->_biases) {
            if (nElements % scaleShift->_biases->size()) {
                THROW_GNA_EXCEPTION << "Cannot tile biases for layer: " << l->name << ", due to biases size not GCD of dims product";
            }
            scaleShift->_biases = tileBlob(scaleShift->_biases, nElements);
        }

        auto tensor = InferenceEngine::TensorDesc(insData->getTensorDesc());
        tensor.reshape(SizeVector{ batchSize, nElements }, Layout::NC);
        auto reshapeName = scaleShift->name + "_input_" + std::to_string(0) + "_reshape";
        auto reshape = CNNNetworkCreateReshape(tensor, reshapeName, quantized);
        auto layer_before_scale_shift = getCreatorLayer(insData);

        CNNNetworkInsertLayer(layer_before_scale_shift.lock(), l, reshape);
        gnalog() << "\tInserted " << reshapeName << " between " << layer_before_scale_shift.lock()->name << " and " << l->name << std::endl;
    }
}

void BroadcastConstPass::run() {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "BroadcastConstPass");
    for (auto constLayer : *pLayers) {
        if (!LayerInfo(constLayer).isConst()) {
            continue;
        }

        auto isNonFunctional = [](CNNLayerPtr layer) {
            return LayerInfo(layer).isNonFunctional();
        };

        auto nextLayer = CNNNetCheckNextLayerSkipCertain(constLayer, 0, 0, true, isNonFunctional).first;
        if (!nextLayer || (!LayerInfo(nextLayer).isEltwise() && !LayerInfo(nextLayer).isFakeQuantize())) {
            continue;
        }

        auto prevLayer = nextLayer;
        if (LayerInfo(nextLayer).isFakeQuantize()) {
            if (CNNNetPrevLayer(nextLayer, 0) != constLayer) {
                continue;
            }

            nextLayer = CNNNetCheckNextLayerSkipCertain(nextLayer, 0, 0, true, isNonFunctional).first;
            if (!nextLayer || !LayerInfo(nextLayer).isEltwise()) {
                continue;
            }
        }

        auto constDims = constLayer->outData.front()->getTensorDesc().getDims();
        auto constDimsSize = product(constDims.begin(), constDims.end());
        auto eltwiseDims = nextLayer->outData.front()->getTensorDesc().getDims();
        auto eltwiseDimsSize = product(eltwiseDims.begin(), eltwiseDims.end());
        if (constDimsSize == eltwiseDimsSize || eltwiseDimsSize % constDimsSize) {
            continue;
        }

        auto blobsIter = constLayer->blobs.find("custom");
        if (blobsIter == constLayer->blobs.end()) {
            THROW_GNA_LAYER_EXCEPTION(constLayer) << "Const layer " << constLayer->name << " is missing 'custom' parameter";
        }

        auto currentConstBlob = blobsIter->second;
        blobsIter->second = tileBlob(currentConstBlob, eltwiseDimsSize);
        constLayer->outData.front()->setDims(nextLayer->outData.front()->getDims());
        constLayer->outData.front()->setLayout(nextLayer->outData.front()->getLayout());
        if (prevLayer != nextLayer) {
            prevLayer->outData.front()->setDims(nextLayer->outData.front()->getDims());
            prevLayer->outData.front()->setLayout(nextLayer->outData.front()->getLayout());
        }
        gnalog() << "Const layer '" << constLayer->name << "' was changed to match output of '" << nextLayer->name << "'\n";
    }
}

void BreakFusingOfOutputLayersPass::run() {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "BreakFusingOfOutputLayersPass");
    OutputsDataMap outputsMap = this->getPassManager()->getNetwork().getOutputsInfo();
    for (auto layer : *pLayers) {
        /* Inserion of the second activation after pooling will break Conv - Pooling - Activation component
         * since scaleshift layers will be inserted between the pooling and activations
         */
        if (LayerInfo(layer).isPooling()) continue;
        for (int output_idx = 0; output_idx < layer->outData.size(); output_idx++) {
            auto& output = layer->outData[output_idx];
            auto& input_to = getInputTo(output);

            auto output_name = output->getName();
            auto is_network_output = outputsMap.find(output_name) != outputsMap.end();
            // In cases that this layer is network output you cannot use identity as sole output on
            // it since it will possibly be fused and layer outputs will be unavailable
            if (is_network_output) {
                if (input_to.size() != 1) continue;
                if (!LayerInfo(input_to.begin()->second).isActivation()) continue;

                CNNLayerPtr additional_output =
                    std::make_shared<GenericLayer>(LayerParams({output_name + "_side_identity", "identity", InferenceEngine::Precision::FP32}));

                auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);
                auto additional_output_quant = quantized ? InferenceEngine::injectData<QuantizedLayerParams>(additional_output) : additional_output;

                additional_output_quant->insData.resize(1);
                additional_output_quant->outData.resize(1);

                auto out_data = DataPtr(new Data(output_name + "_side_identity_data", output->getTensorDesc()));
                getCreatorLayer(out_data) = additional_output_quant;

                additional_output_quant->outData[0] = out_data;

                input_to[additional_output_quant->name] = additional_output_quant;
                additional_output_quant->insData[0] = output;
            }
        }
    }
}

void UnrollLSTMCellPass::run() {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "UnrollLSTMCellPass");
    InferenceEngine::NetPass::UnrollRNN_if(getPassManager()->getNetwork(), [] (const RNNCellBase& rnn) -> bool {
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
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "UnrollTIPass");
    auto sts = InferenceEngine::NetPass::UnrollTI(getPassManager()->getNetwork());
    if (!sts) {
        THROW_GNA_EXCEPTION << "TensorIterator layer cannot be unrolled!";
    }
}

void RemoveConstPass::run() {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "RemoveConstPass");
    auto network = getPassManager()->getNetwork();
    IE_SUPPRESS_DEPRECATED_START
    auto & icnnnet = static_cast<ICNNNetwork &>(network);
    IE_SUPPRESS_DEPRECATED_END
    auto* implNetwork = dynamic_cast<details::CNNNetworkImpl*>(&icnnnet);
    if (!implNetwork) {
        THROW_GNA_EXCEPTION << "Remove const layers pass can only work on cnnnetworkimpl type";
    }
    ConstTransformer transformer(implNetwork);
    transformer.fullTrim();
}

void RemoveSingleInputConcatPass::run() {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "RemoveSingleInputConcatPass");
    for (auto &l : *pLayers) {
        if (l->type == "Concat") {
            auto concat = dynamic_cast<ConcatLayer*>(l.get());
            if (concat == nullptr) {
                THROW_GNA_EXCEPTION << "Layer has type Concat but faild during casting to ConcatLayer";
            }
            if (concat->insData.size() == 1 && concat->outData.size() > 0) {
                auto in = concat->insData[0];
                auto in_layer = getCreatorLayer(in.lock());

                auto out = concat->outData[0];

                for (auto out_layer : getInputTo(out)) {
                    for (int i = 0; i < out_layer.second->insData.size(); i++) {
                        if (out_layer.second->insData[i].lock() == out) {
                            out_layer.second->insData[i] = in;
                            getInputTo(in.lock())[out_layer.second->name] = out_layer.second;
                        }
                    }
                }
                getInputTo(in.lock()).erase(concat->name);
                getInputTo(out).clear();
                concat->insData.clear();
                concat->outData.clear();
            }
        }
    }
}

void FuseMultipleIdentitiesPass::run() {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "FuseMultipleIdentitiesPass");
    for (auto &l : *pLayers) {
        if (l->insData.empty()) continue;

        auto isNonFunctional = [](CNNLayerPtr ptr) {
            return LayerInfo(ptr).isNonFunctional();
        };
        if (LayerInfo(l).hasMultipleInputs()) {
            continue;
        }
        if (LayerInfo(l).isNonFunctional() || LayerInfo(l).has32BInput()) {
            continue;
        }
        gnalog() << "CNNNetPrevLayer skip non functional from :: " << l->name;
        auto isFunctional = [](CNNLayerPtr ptr) {
            return !LayerInfo(ptr).isNonFunctional();
        };

        auto prevLayersReached = CNNNetGetPrevLayersSkip(l, isFunctional);
        if (!prevLayersReached.empty()) {
            prevLayersReached.erase(std::remove_if(prevLayersReached.begin(),
                                                prevLayersReached.end(),
                                                [] (const std::pair<CNNLayerPtr, int> & candidate) {
                return LayerInfo(candidate.first).isLink();
            }), prevLayersReached.end());
            if (prevLayersReached.empty()) {
                gnalog() << ", connected to link output only" << std::endl;
                continue;
            }
        }

        if (prevLayersReached.size() != 1) {
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

void FuseFQIntoWeightsPass::run() {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "FuseFQIntoWeightsPass");
    auto isNonFunctional = [](CNNLayerPtr ptr) {
        return LayerInfo(ptr).isNonFunctional();
    };

    auto assignWeightsAndBiases = [](CNNLayerPtr layer, Blob::Ptr weights, Blob::Ptr biases) {
        auto weigtableLayer = std::dynamic_pointer_cast<WeightableLayer>(layer);
        if (nullptr == weigtableLayer) {
            THROW_GNA_LAYER_EXCEPTION(layer) << " not a weightable layer";
        }
        weigtableLayer->_weights = weights;
        weigtableLayer->_biases = biases;
        weigtableLayer->blobs["weights"] = weights;
        weigtableLayer->blobs["biases"] = biases;
    };

    for (auto& l : *pLayers) {
        if (!LayerInfo(l).isFakeQuantize()) {
            continue;
        }
        // determine whether this FQ is actually ends into weigtable layer
        auto fqLayer = l;
        if (!CNNNetHasNextLayerSkipCertain(fqLayer, 0, 0, isNonFunctional)) {
            continue;
        }

        GNAFakeQuantizeLayer gnaFakeQuantizeLayer(fqLayer);
        auto inputTo = getInputTo(fqLayer->outData[0]);
        size_t layers_connected_to_fq_count = inputTo.size();
        auto layerBeforeWeightable = fqLayer;
        while (layers_connected_to_fq_count == 1 && LayerInfo(inputTo.begin()->second).isNonFunctional()) {
            layerBeforeWeightable = inputTo.begin()->second;
            inputTo = getInputTo(layerBeforeWeightable->outData[0]);
            layers_connected_to_fq_count = inputTo.size();
        }
        for (int index = 0; index < layers_connected_to_fq_count; index++) {
            auto weightableLayer = CNNNetCheckNextLayerSkipCertain(layerBeforeWeightable, 0, index, true, isNonFunctional).first;
            if (!weightableLayer || !LayerInfo(weightableLayer).isWeightable()) {
                continue;
            }
            if (weightableLayer->insData.size() < 2) {
                continue;
            }

            // check whether this FQ represents weights - it need to be at index 1 of weightable layer
            const size_t weightsIdx = 1;
            const size_t biasesIdx = 2;
            auto prevLayerAt1 = CNNNetPrevLayerSkipCertain(weightableLayer, weightsIdx, isNonFunctional);

            if (prevLayerAt1 != fqLayer) {
                continue;
            }

            // now this FQ layer represents weights - lets apply it and fuse to given weightable layer.
            pass_trace() << "found " << LAYER_NAME(fqLayer) << " that will be converted to weights of "
                << LAYER_NAME(weightableLayer) << "\n";

            auto biases = weightableLayer->insData.size() == 3 ?
                LayerUtils::getParamFromInputAsBlob(weightableLayer, biasesIdx) : nullptr;
            auto quantizedWeights = gnaFakeQuantizeLayer.getConstInputData();

            // 1. broke existing connections - by detaching fq subgraph from rest of graph
            auto prevData = weightableLayer->insData[weightsIdx].lock();
            auto prevLayer = getCreatorLayer(prevData).lock();
            auto weightDims = prevLayer->outData.front()->getDims();
            // input related to FakeQuantize layer is removed
            auto& relatedInputs = getInputTo(prevLayer->outData[0]);
            auto relatedInputsIter = relatedInputs.begin();
            while (relatedInputsIter != relatedInputs.end()) {
                if (LayerInfo(relatedInputsIter->second).isFakeQuantize()) {
                    relatedInputsIter = relatedInputs.erase(relatedInputsIter);
                } else {
                    ++relatedInputsIter;
                }
            }

            weightableLayer->insData.resize(1);

            // 2. running FQ function for given layer
            auto outputSize = details::product(weightDims.begin(), weightDims.end());

            // depending on compute precision weights will be recreated
            // for integer mode - weights might be simply copied - to avoid furter quantisations overhead
            auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(weightableLayer);
            if (quantized) {
                // assign already quantized Weights
                assignWeightsAndBiases(weightableLayer, quantizedWeights, biases);

                // modify scale factors for quantized component
                auto levels = gnaFakeQuantizeLayer.getLevels();
                auto inputRange = gnaFakeQuantizeLayer.getInputRange();
                auto outputRange = gnaFakeQuantizeLayer.getOutputRange();
                if (outputRange.first.size() != outputRange.second.size()) {
                    THROW_GNA_LAYER_EXCEPTION(fqLayer) << " number of min and max data must be equal, min size: "
                        << outputRange.first.size() << ", max size: " << outputRange.second.size();
                }

                if (inputRange.first.size() != outputRange.first.size() ||
                    inputRange.second.size() != outputRange.second.size()) {
                    THROW_GNA_LAYER_EXCEPTION(fqLayer) << " size of input and output range differs. "
                        << "input min size: " << inputRange.first.size() << ", "
                        << "output min size: " << outputRange.first.size() << ", "
                        << "input max size: " << inputRange.second.size() << ", "
                        << "output max size: " << outputRange.second.size();
                }

                if (levels > std::numeric_limits<uint8_t>::max() && outputRange.first.size() > 1) {
                    // find min and max for input and output range
                    THROW_GNA_LAYER_EXCEPTION(fqLayer) << " unsupported per-channel quantization for int16 weights."
                        << " Per-channel quantization ";
                }

                // check if
                // - weights were float values and need to be quantized,
                // - weights are integer values and quantization can be skipped
                quantized->_weights_quant.SetMinValues(inputRange.first, true);
                quantized->_weights_quant.SetMaxValues(inputRange.second, true);
                quantized->_weights_quant.SetMinValues(outputRange.first, false);
                quantized->_weights_quant.SetMaxValues(outputRange.second, false);
                quantized->_weights_quant.SetLevels(levels);

                // lets find out minimum scale factor among channels
                if (!quantized->_weights_quant.IsStatsSet()) {
                    THROW_GNA_LAYER_EXCEPTION(fqLayer) << " per channel/tensor weigths scales are missed";
                }
                continue;
            }

            size_t depth = 1;
            intel_dnn_component_t component;
            component.num_columns_in = weightDims[1];
            component.num_rows_in = weightDims[0];

            if (LayerInfo(weightableLayer).isConvolution()) {
                depth = (weightDims.size() == 4) ? weightDims[2] * weightDims[3] : 1;
            }

            intel_piecewiselinear_t* transform = reinterpret_cast<intel_piecewiselinear_t*>(&component.op.pwl);
            transform->func_id = gnaFakeQuantizeLayer.parseAsActivation();

            auto quantizedWeightsData = quantizedWeights->buffer();
            auto dequantizedWeights = make_shared_blob<float>(TensorDesc(Precision::FP32, { outputSize }, Layout::C));
            dequantizedWeights->allocate();

            auto resultBuffer = dequantizedWeights->buffer();
            for (size_t i = 0; i < depth; ++i) {
                component.ptr_inputs = quantizedWeightsData.as<float*>() + i * component.num_columns_in * component.num_rows_in;
                component.ptr_outputs = resultBuffer.as<float*>() + i * component.num_columns_in * component.num_rows_in;

                PwlApply32(&component, 0, component.num_rows_in - 1, 0, component.num_columns_in - 1);
            }

            // 3. assign dequantized const blob to weightable layer
            assignWeightsAndBiases(weightableLayer, dequantizedWeights, biases);
        }
    }
}

void MoveFakeQuantizeLayerIntoQuantParamsPass :: run() {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "MoveFakeQuantizeLayerIntoQuantParamsPass");
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(pLayers->front());
    if (!quantized) {
        return;
    }

    auto donotSkip = [](CNNLayerPtr) {
        return false;
    };

    auto allowFQFuse = [this](CNNLayerPtr layer) -> bool {
        auto skipNonFunctionalOrMemory = [](CNNLayerPtr layer) {
            return LayerInfo(layer).isNonFunctional() || LayerInfo(layer).isMemory();
        };
        auto skipNonFunctional = [](CNNLayerPtr layer) {
            return LayerInfo(layer).isNonFunctional();
        };
        // Don't fuse FQ if it's the output layer for the network
        if (CNNNetGetAllNextLayersSkipCertain(layer, -1, skipNonFunctionalOrMemory).empty()) {
            return false;
        }
        // Fuse FQ if it's not required to change precision from int32 to int16
        auto nextLayers = CNNNetGetAllNextLayersSkipCertain(layer, -1, skipNonFunctional);
        for (auto& l : nextLayers) {
            if (getCandidatesForIdentityInsertion(l, getPassManager(), true).empty()) {
                return true;
            }
        }
        return false;
    };

    std::function<void(QuantizedLayerParams*, CNNLayerPtr)> propagateStatistics =
        [&propagateStatistics](QuantizedLayerParams* srcQuantParams, CNNLayerPtr layer) {
        if (LayerInfo(layer).isFakeQuantize()) {
            return;
        }

        auto donotSkip = [](CNNLayerPtr) {
            return false;
        };

        auto quantParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);
        IE_ASSERT(quantParams != nullptr);

        // Find all output layers connected to FQ
        auto nextLayers = CNNNetGetAllNextLayersSkipCertain(layer.get(), -1, donotSkip);
        if (nextLayers.empty()) {
            quantParams->_src_quant.CopyStats(srcQuantParams->_dst_quant);
            if (LayerInfo(layer).isNonFunctional()) {
                quantParams->_dst_quant.CopyStats(srcQuantParams->_dst_quant);
            }
            return;
        }

        auto srcMinVals = srcQuantParams->_dst_quant.GetMinValues().front();
        auto srcMaxVals = srcQuantParams->_dst_quant.GetMaxValues().front();
        // If a next layer is concat, find minimum nad maximum statistics
        if (LayerInfo(layer).isConcat() && quantParams->_src_quant.IsStatsSet()) {
            auto concatMinVal = quantParams->_src_quant.GetMinValues().front();
            auto concatMaxVal = quantParams->_src_quant.GetMaxValues().front();
            quantParams->_src_quant.SetMinValues({ std::min(srcMinVals, concatMinVal) });
            quantParams->_src_quant.SetMaxValues({ std::max(srcMaxVals, concatMaxVal) });
        } else if (quantParams->_src_quant.IsStatsSet()) {
            return;
        } else {
            quantParams->_src_quant.CopyStats(srcQuantParams->_dst_quant);
        }

        if (!LayerInfo(layer).isWeightable() && !LayerInfo(layer).isEltwise() &&
            !LayerInfo(layer).isActivation() && !LayerInfo(layer).isFakeQuantize()) {
            auto doNotSetDstStats = false;
            for (auto& l : nextLayers) {
                if (LayerInfo(l).isFakeQuantize()) {
                    doNotSetDstStats = true;
                    continue;
                }
            }

            if (doNotSetDstStats) {
                return;
            }

            quantParams->_dst_quant.CopyStats(quantParams->_src_quant);

            for (auto& l : nextLayers) {
                if (LayerInfo(l).isFakeQuantize()) {
                    continue;
                }

                propagateStatistics(quantParams, l);
            }
        }
    };

    for (auto &&l : *pLayers) {
        if (!LayerInfo(l).isFakeQuantize()) {
            continue;
        }
        GNAFakeQuantizeLayer fqLayer(l);
        auto prevLayer = CNNNetPrevLayerSkipCertain(*fqLayer, 0, donotSkip);
        auto prevDataIt = std::find_if(std::begin(prevLayer->outData), std::end(prevLayer->outData), [l](DataPtr data) {
            return getInputTo(data).find(l->name) != std::end(getInputTo(data));
        });
        if (prevDataIt == std::end(prevLayer->outData)) {
            THROW_GNA_LAYER_EXCEPTION(fqLayer) << "Invalid connection between " << prevLayer->name << " and " << l->name;
        }

        auto inputRange = fqLayer.getInputRange();
        auto outputRange = fqLayer.getOutputRange();
        if (inputRange.first.size() != 1 || inputRange.second.size() != 1 ||
            outputRange.first.size() != 1 || outputRange.second.size() != 1) {
            THROW_GNA_LAYER_EXCEPTION(fqLayer) << " unsupported per-channel quantisation";
        }

        if (!LayerInfo(prevLayer).isConst() &&
            !fp32eq(inputRange.first.front(), outputRange.first.front()) &&
            !fp32eq(inputRange.second.front(), outputRange.second.front())) {
            THROW_GNA_LAYER_EXCEPTION(fqLayer) << " unsupported data range conversion. Input: (" <<
                inputRange.first.front() << "," << inputRange.second.front() << "), output: (" <<
                outputRange.first.front() << "," << outputRange.second.front() << ")";
        }

        auto fqLevels = fqLayer.getLevels();
        if (fqLevels == 0) {
            THROW_GNA_LAYER_EXCEPTION(fqLayer) << "Zero levels";
        }

        // Before FQ layer is removed, the previous functional layer has to be updated with its quantization data
        auto prevFuncLayer = CNNNetPrevLayerSkipCertain(*fqLayer, 0, [](CNNLayerPtr layer) {
            return LayerInfo(layer).isNonFunctional() || LayerInfo(layer).isPooling();
        });
        auto quantParamsPrevLayer = InferenceEngine::getInjectedData<QuantizedLayerParams>(prevFuncLayer);
        quantParamsPrevLayer->_dst_quant.SetLevels(fqLevels);
        quantParamsPrevLayer->_dst_quant.SetMinValues({ inputRange.first[0] }, true);
        quantParamsPrevLayer->_dst_quant.SetMaxValues({ inputRange.second[0] }, true);
        quantParamsPrevLayer->_dst_quant.SetMinValues({ outputRange.first[0] }, false);
        quantParamsPrevLayer->_dst_quant.SetMaxValues({ outputRange.second[0] }, false);

        // Propogate destination statistics to multiply layer if it's set for the next sum/sub layer (is considered as bias)
        if (LayerInfo(prevLayer).isEltwiseSum() || LayerInfo(prevLayer).isEltwiseSub()) {
            auto eltwPrevLayer = CNNNetPrevLayerSkipCertain(prevLayer, 0, donotSkip);
            auto constLayer = CNNNetPrevLayerSkipCertain(prevLayer, 1, donotSkip);
            if (LayerInfo(eltwPrevLayer).isEltwise() && LayerInfo(constLayer).isConst()) {
                auto quantParamsEltwLayer = InferenceEngine::getInjectedData<QuantizedLayerParams>(eltwPrevLayer);
                quantParamsEltwLayer->_dst_quant.CopyStats(quantParamsPrevLayer->_dst_quant);
            }
        }

        auto fqQauntParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(l);
        fqQauntParams->_dst_quant.SetLevels(UINT16_MAX);
        fqQauntParams->_dst_quant.SetMinValues({ inputRange.first[0] }, true);
        fqQauntParams->_dst_quant.SetMaxValues({ inputRange.second[0] }, true);
        fqQauntParams->_dst_quant.SetMinValues({ outputRange.first[0] }, false);
        fqQauntParams->_dst_quant.SetMaxValues({ outputRange.second[0] }, false);
        fqQauntParams->_src_quant = fqQauntParams->_dst_quant;

        l->insData.resize(1);
        if (!CNNNetHasPrevLayer(prevLayer.get())) {
            quantParamsPrevLayer->_src_quant = quantParamsPrevLayer->_dst_quant;
        }

        // Allow FQ Fuse checks if FQ layer can be fused to a layer before or after.
        // FQ Layer is fused if it's not required for precision change.
        bool isFQFuseAllowed = allowFQFuse(l);
        auto prevData = *prevDataIt;

        // Find all output layers connected to FQ
        auto nextLayers = CNNNetGetAllNextLayersSkipCertain(*fqLayer, -1, donotSkip);
        if (nextLayers.empty()) {
            continue;
        }

        if (isFQFuseAllowed) {
            getInputTo(prevData).erase(l->name);
        }

        // Connect all next layers after FQ to the layer that is before FQ
        // and propagate quantization data
        for (size_t i = 0; i < nextLayers.size(); ++i) {
            if (isFQFuseAllowed) {
                auto insDatas = CNNLayerFindInsDataIdxes(fqLayer->outData.front(), nextLayers[i]);
                if (insDatas.empty()) {
                    THROW_GNA_LAYER_EXCEPTION(fqLayer) << " fake quantize connection to layer: "
                        << LAYER_NAME(nextLayers[i]) << " is not correct";
                }
                for (int insDataIdx : insDatas) {
                    nextLayers[i]->insData[insDataIdx] = prevData;
                }
                getInputTo(prevData)[nextLayers[i]->name] = nextLayers[i];
            }

            propagateStatistics(quantParamsPrevLayer, nextLayers[i]);
        }
    }
}

void TransposeWeightsFromNCHWToNHWCPass::run() {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "TransposeWeightsFromNCHWToNHWCPass");
    if (!MustBeConvertedFromNCHWToNHWC(*pLayers)) return;

    auto printTranspositionInfo = [](const std::vector<TranspositionInfo> &transpositionInfo) {
        for (const auto &transpositionInfoPart : transpositionInfo) {
            gnalog() << "transpose=" << transpositionInfoPart.transpose << " rows_num=" << transpositionInfoPart.num_transpose_rows
                     << " columns_num=" << transpositionInfoPart.num_transpose_columns << "\n";
        }
    };

    auto transpInfoMatchWeightsSize = [](const std::vector<TranspositionInfo> &transpositionInfo, size_t weightsSize, const std::string &layerName) {
        size_t totalElements = 0;
        for (auto && transpositionInfoPart : transpositionInfo) {
            totalElements += transpositionInfoPart.num_transpose_rows * transpositionInfoPart.num_transpose_columns;
        }
        if (totalElements != weightsSize) {
            THROW_GNA_EXCEPTION << layerName << " weights elements from transposition info (" << totalElements
                                << ") don't match input dimensions (" << weightsSize << ")";
        }
    };

    for (auto &&l : *pLayers) {
        if (LayerInfo(l).isScaleShift()) {
            std::vector<TranspositionInfo> transpositionInfo;
            // Try to find a convolution in previous layers
            if (InferenceEngine::CNNNetHasPrevLayer(l.get())) {
                transpositionInfo = FindTranspositionInfoFromPrevLayers(InferenceEngine::CNNNetPrevLayer(l));
                // If no convolutions are found try to find them in next layers
                if (!FoundPartToTranspose(transpositionInfo) && !l->outData.empty() && !getInputTo(l->outData[0]).empty()) {
                    transpositionInfo = FindTranspositionInfoFromNextLayers(getInputTo(l->outData[0]).begin()->second);
                }
            }
            if (FoundPartToTranspose(transpositionInfo)) {
                if (l->input()->getDims().front() > 1) {
                    THROW_GNA_EXCEPTION << l->name << " Weights transposition is not supported for a layer with batch size > 1";
                }
                auto weightable = dynamic_cast<WeightableLayer*>(l.get());
                IE_ASSERT(weightable != nullptr);

                size_t totalWeights = weightable->_weights->size();
                transpInfoMatchWeightsSize(transpositionInfo, totalWeights, l->name);

                ConvertTensorFromNCHWToNHWC(weightable->precision.size(), 1, weightable->_weights->size(),
                    weightable->_weights->cbuffer().as<uint8_t*>(), true, transpositionInfo);
                if (weightable->_biases) {
                    ConvertTensorFromNCHWToNHWC(weightable->precision.size(), 1, weightable->_biases->size(),
                        weightable->_biases->cbuffer().as<uint8_t*>(), true, transpositionInfo);
                }
                gnalog() << l->name << " weights and biases rows transposition info:\n";
                printTranspositionInfo(transpositionInfo);
            }
        }

        if (LayerInfo(l).isFullyConnected()) {
            auto weightable = dynamic_cast<WeightableLayer*>(l.get());
            IE_ASSERT(weightable != nullptr);
            IE_ASSERT(weightable->_weights != nullptr);
            auto precision = weightable->precision.size();
            auto out_dims = l->outData[0]->getDims();
            auto in_dims = l->input()->getDims();
            auto weightsRows = InferenceEngine::details::product(std::begin(out_dims) + 1, std::end(out_dims));
            auto weightsColumns = InferenceEngine::details::product(std::begin(in_dims) + 1, std::end(in_dims));
            // Find a convolution in previous layers to rotate weights rows
            if (InferenceEngine::CNNNetHasPrevLayer(l.get())) {
                std::vector<TranspositionInfo> transpositionInfo;
                auto prevLayer = InferenceEngine::CNNNetPrevLayer(l);
                transpositionInfo = FindTranspositionInfoFromPrevLayers(prevLayer);
                if (FoundPartToTranspose(transpositionInfo)) {
                    if (l->input()->getDims().front() > 1) {
                        THROW_GNA_EXCEPTION << l->name << " Weights transposition is not supported for a layer with batch size > 1";
                    }
                    if (LayerInfo(prevLayer).isSplit()) {
                        // If we found a split it's not possible to rotate data
                        THROW_GNA_EXCEPTION << l->name << " won't be transposed due to a split before it";
                    }

                    transpInfoMatchWeightsSize(transpositionInfo, weightsColumns, l->name);

                    weightable->_weights = convertToRWBlob(weightable->_weights);

                    ConvertTensorFromNCHWToNHWC(precision, weightsRows, weightsColumns, weightable->_weights->buffer().as<uint8_t*>(),
                                                true, transpositionInfo);
                    gnalog() << l->name << " weights rows transposition info:\n";
                    printTranspositionInfo(transpositionInfo);
                }
            }
            // Find a convolution in next layers to rotate weights columns
            if (!l->outData.empty() && !getInputTo(l->outData[0]).empty()) {
                std::vector<TranspositionInfo> transpositionInfo;
                auto nextLayer = getInputTo(l->outData[0]).begin()->second;
                transpositionInfo = FindTranspositionInfoFromNextLayers(nextLayer);
                if (FoundPartToTranspose(transpositionInfo)) {
                    if (l->outData[0]->getDims().front() > 1) {
                        THROW_GNA_EXCEPTION << l->name << " Weights transposition is not supported for a layer with batch size > 1";
                    }
                    if (LayerInfo(nextLayer).isConcat()) {
                        // If we found a concat it's not possible to rotate data
                        THROW_GNA_EXCEPTION << l->name << " won't be transposed due to a concat after it";
                    }

                    transpInfoMatchWeightsSize(transpositionInfo, weightsRows, l->name);

                    weightable->_weights = convertToRWBlob(weightable->_weights);

                    ConvertTensorFromNCHWToNHWC(precision, weightsRows, weightsColumns, weightable->_weights->cbuffer().as<uint8_t*>(),
                                                false, transpositionInfo);
                    gnalog() << l->name << " weights columns transposition info:\n";
                    printTranspositionInfo(transpositionInfo);
                }
            }
        }

        if (LayerInfo(l).isEltwise()) {
            // We need to transpose a constant which is an eltwise input
            auto firstInput = InferenceEngine::CNNNetPrevLayer(l, 0);
            auto secondInput = InferenceEngine::CNNNetPrevLayer(l, 1);
            if (!LayerInfo(firstInput).isConst() && !LayerInfo(secondInput).isConst()) {
                continue;
            }
            // Let a constant to be the second input
            if (LayerInfo(firstInput).isConst()) {
                std::swap(firstInput, secondInput);
            }
            // Find a convolution in previous or next layers
            auto transpositionInfo = FindTranspositionInfoFromPrevLayers(firstInput);
            if (!FoundPartToTranspose(transpositionInfo) && !l->outData.empty() && !getInputTo(l->outData[0]).empty()) {
                transpositionInfo = FindTranspositionInfoFromNextLayers(getInputTo(l->outData[0]).begin()->second);
            }
            if (FoundPartToTranspose(transpositionInfo)) {
                auto blob = secondInput->blobs["custom"];
                ConvertTensorFromNCHWToNHWC(blob->getTensorDesc().getPrecision().size(), 1, blob->size(),
                                            blob->buffer().as<uint8_t*>(), true, transpositionInfo);
                gnalog() << secondInput->name << " data transposition info:\n";
                printTranspositionInfo(transpositionInfo);
            }
        }

        if (LayerInfo(l).isConcat()) {
            auto concatLayer = LayerInfo(l).as<InferenceEngine::ConcatLayer*>();
            IE_ASSERT(concatLayer != nullptr);
            // If concatenation is along channel axis constant input transposition isn't required
            if (concatLayer->_axis <= 1) continue;

            std::vector<InferenceEngine::CNNLayerPtr> constInputs;
            bool transpose = false;
            int nonConstInputIx = 0;
            // Check if non-const inputs are transposed
            for (int i = 0; InferenceEngine::CNNNetHasPrevLayer(l.get(), i); ++i) {
                auto input = InferenceEngine::CNNNetPrevLayer(l, i);
                if (LayerInfo(input).isConst()) {
                    constInputs.push_back(input);
                    continue;
                }
                auto transpositionInfo = FindTranspositionInfoFromPrevLayers(input);
                bool transposeInput = FoundPartToTranspose(transpositionInfo);
                if (nonConstInputIx == 0) {
                    transpose = transposeInput;
                } else if (transposeInput != transpose) {
                    THROW_GNA_EXCEPTION << "Concat layer " << l->name << " inputs have different layouts";
                }
                ++nonConstInputIx;
            }
            if (!transpose) continue;

            // Transpose all constant inputs
            for (auto && input : constInputs) {
                auto rows = GetDataDimSize(input->outData[0], DataDimName::C);
                auto columns = GetDataDimSize(input->outData[0], DataDimName::H) * GetDataDimSize(input->outData[0], DataDimName::W);

                auto blob = convertToRWBlob(input->blobs["custom"]);
                input->blobs["custom"] = blob;

                // A constant should have the same number of channels since concatenation will be in height/weight dimension
                TranspositionInfo concatTranspositionInfo{true, rows, columns};
                ConvertTensorFromNCHWToNHWC(blob->getTensorDesc().getPrecision().size(), 1, blob->size(),
                                            blob->buffer().as<uint8_t*>(), true, {concatTranspositionInfo});
                gnalog() << input->name << " data transposition info:\n";
                printTranspositionInfo({concatTranspositionInfo});
            }
        }
    }
}

int PassManager::run(int index) {
#if defined PLOT || defined ENABLE_V7_SERIALIZE
    auto dumpNetworkAfterPass = [&index, this] (std::shared_ptr<Pass> pass) {
        std::string name = std::string("gna_passes_") + (index < 10 ? "0" : "") + std::to_string(index) + "_" + pass->getName();
#ifdef PLOT
        std::ofstream out(name + ".dot");
        saveGraphToDot(network, out, [](const CNNLayerPtr layer,
                                        ordered_properties &printed_properties,
                                        ordered_properties &node_properties) {});
#endif
#ifdef ENABLE_V7_SERIALIZE
        network.serialize(name + ".xml", name + ".bin");
#endif
    };
#else
    auto dumpNetworkAfterPass = [] (std::shared_ptr<Pass> ) {};
#endif

    for (auto && pass : passes) {
        if (settings.runBeforeCopy != pass->runBeforeCopyPass()) {
            continue;
        }
        auto layers = CNNNetSortTopologically(network);
        pass->attach(layers);
        gnalog() << "PASS: " << ++index << "/" << passes.size() << ":" << pass->getName() << "\n";
        pass->run();
        dumpNetworkAfterPass(pass);
    }
    return index;
}
