// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <runtime/pwl.h>
#include "nnet_base_matcher.hpp"

#include "gna_lib_ver_selector.hpp"

extern void PwlApply16(intel_dnn_component_t *component, uint32_t num_subset_size);

class PWLMatcher : public ::testing::MatcherInterface<const gna_nnet_type_t*> {
    bool matchInserted;
    int matchQuantity;
    mutable int timesInserted = 0;
    mutable std::vector<DnnActivationType> activationsToLookFor;
    mutable std::string notFoundLayer;

    template <typename Key>
    using HashType = typename std::conditional<std::is_enum<Key>::value, std::hash<typename std::underlying_type<Key>::type>, std::hash<Key>>::type;

 public:
    PWLMatcher(bool inserted, int matchQuantity, const std::vector<DnnActivationType> & particularActivations)
        : matchInserted(inserted), matchQuantity(matchQuantity), activationsToLookFor(particularActivations) {
    }

    bool MatchAndExplain(const gna_nnet_type_t *foo, ::testing::MatchResultListener *listener) const override {
        if (foo == nullptr)
            return false;
        timesInserted = 0;
        std::unordered_map<DnnActivationType, int, HashType<DnnActivationType>> foundActivations;

        for(int i = 0; i < foo->nLayers; i++) {
            if (foo->pLayers[i].nLayerKind != INTEL_AFFINE &&
                foo->pLayers[i].nLayerKind != INTEL_AFFINE_DIAGONAL &&
                foo->pLayers[i].nLayerKind != INTEL_CONVOLUTIONAL) continue;
            auto affine = reinterpret_cast<gna_affine_layer_t*>(foo->pLayers[i].pLayerStruct);
            if (affine == nullptr) continue;

            bool hasPwl = affine->pwl.nSegments != 0 && affine->pwl.pSegments != nullptr;

            if (hasPwl) {
                if (matchQuantity == -1) {
                    if (activationsToLookFor.empty())
                        return matchInserted;
                    // detection of particular activation type
                    foundActivations[detectPwlType(foo->pLayers + i)]++;
                } else {
                    timesInserted++;
                }
            }
        }
        if (!activationsToLookFor.empty()) {
            for (auto & activation : activationsToLookFor) {
                if (!foundActivations.count(activation)) {
                    notFoundLayer = intel_dnn_activation_name[activation];
                    return false;
                }
                if (!--foundActivations[activation]) {
                    foundActivations.erase(activation);
                }
            }
            return true;
        }
        if (matchInserted) {
            if (matchQuantity != -1) {
                return timesInserted == matchQuantity;
            }
            return timesInserted != 0;
        }

        return timesInserted == 0;
    };

    DnnActivationType detectPwlType(gna_nnet_layer_t *layer) const {
        intel_dnn_component_t comp{};
        comp.ptr_outputs = layer->pOutputs;
        comp.num_columns_in = layer->nInputColumns;
        comp.num_rows_in = layer->nInputRows;

        if (layer->nLayerKind == INTEL_AFFINE ||
            layer->nLayerKind == INTEL_AFFINE_DIAGONAL) {
            auto pAffineLayer = reinterpret_cast<gna_affine_layer_t *>(layer->pLayerStruct);
            comp.op.pwl.num_segments = pAffineLayer->pwl.nSegments;
            comp.op.pwl.ptr_segments = pAffineLayer->pwl.pSegments;
        } else if (layer->nLayerKind == INTEL_CONVOLUTIONAL) {
            auto pConvolutionalLayer = reinterpret_cast<gna_convolutional_layer_t *>(layer->pLayerStruct);
            comp.op.pwl.num_segments = pConvolutionalLayer->pwl.nSegments;
            comp.op.pwl.ptr_segments = pConvolutionalLayer->pwl.pSegments;
        } else {
            return kActNone;
        }

        int16_t prevSlope = 0;
        int slopeChangedTimes = 0;
        for (int i = 0; i != comp.op.pwl.num_segments; i++) {
            if  (!i || prevSlope != comp.op.pwl.ptr_segments[i].slope) {
                slopeChangedTimes++;
                prevSlope = comp.op.pwl.ptr_segments[i].slope;
            }
        }

        switch (slopeChangedTimes) {
            case 3 :
                if (comp.op.pwl.num_segments == 4) {
                    // ReLU has y=0 segment while identity doesn't have
                    // 2 segments are added: one at the begining and one at the end, due to saturation errata
                    return kActRelu;
                } else {
                    return kActIdentity;
                }
            default:
                // currently cannot determine between sigmoid or tanh etc
                if (slopeChangedTimes > 3) {
                    return kActSigmoid;
                }
                return kActNone;
        }
    }

    void DescribeTo(::std::ostream *os) const override {
        if (!matchInserted ) {
            *os << "should not have PWL layer as part of nnet structure, but was found " << timesInserted <<" times" ;
        } else {
            if (matchQuantity == -1) {
                *os << "should have PWL layer " << notFoundLayer << " as part of nnet structure, but it was not found " ;
            } else {
                *os << "should have PWL layer as part of nnet structure, for " << matchQuantity <<" times, but was for " << timesInserted ;
            }
        }
    }
};
