// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include"gna-api.h"
#include "nnet_base_matcher.hpp"
#include "quantization/quantization.h"

class DiagLayerMatcher : public ::testing::MatcherInterface<const intel_nnet_type_t*> {
    bool matchInserted;
    int matchQuantity;
 public:
    DiagLayerMatcher(bool matchInserted, int matchQuantity) : matchInserted(matchInserted), matchQuantity(matchQuantity) {}
    bool MatchAndExplain(const intel_nnet_type_t *foo, ::testing::MatchResultListener *listener) const override {
        if (foo == nullptr)
            return false;
        for(int i = 0; i < foo->nLayers; i++) {
            if (foo->pLayers[i].nLayerKind != INTEL_AFFINE_DIAGONAL) continue;
            // diagonal layer has to have 1 for weights and 0 for biases

            auto diag = (intel_affine_func_t*)foo->pLayers[i].pLayerStruct;

            bool bWeightsOK = true;
            for (int j =0; j < foo->pLayers[i].nOutputRows; j++) {
                auto weights = (int16_t*)diag->pWeights;    
                auto biases = (int32_t*)diag->pBiases;
                // identity matrix tansformed to 16384 values
                if (weights[j] != MAX_VAL_2B_WEIGHT || biases[j] != 0) {
                    bWeightsOK = false;
                    break;
                }
            }
            if (!bWeightsOK) continue;

            return matchInserted;
        }
        return !matchInserted;
    };
    void DescribeTo(::std::ostream *os) const override {
        *os << "should "<< (matchInserted ? "" : "not ") << "have Identity Diagonal Primitive primitive as part of nnet structure";
    }
};

inline ::testing::Matcher<const intel_nnet_type_t*> HasDiagonalLayer(bool matchInserted = false, int matchQuantity = -1) {
    std::unique_ptr<NNetComponentMatcher> c (new NNetComponentMatcher());
    c->add(new DiagLayerMatcher(matchInserted, matchQuantity));
    return ::testing::MakeMatcher(c.release());
}


