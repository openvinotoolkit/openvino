// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "backend/gna_types.h"
#include "nnet_base_matcher.hpp"
#include "frontend/quantization.h"

class DiagLayerMatcher : public ::testing::MatcherInterface<const gna_nnet_type_t*> {
    bool matchInserted;
    int  matchQuantity;
    mutable int  actualQuantity;
public:
    DiagLayerMatcher(bool matchInserted, int matchQuantity) : matchInserted(matchInserted), matchQuantity(matchQuantity) {}
    bool MatchAndExplain(const gna_nnet_type_t *foo, ::testing::MatchResultListener *listener) const override {
        if (foo == nullptr)
            return false;
        actualQuantity = 0;
        for(int i = 0; i < foo->nLayers; i++) {
            if (foo->pLayers[i].nLayerKind != INTEL_AFFINE_DIAGONAL) continue;
            // diagonal layer has to have 1 for weights and 0 for biases

            auto diag = (gna_affine_func_t*)foo->pLayers[i].pLayerStruct;
            bool bWeightsOK = true;

            int beforePadding = 0;
            for (int j =0; j < foo->pLayers[i].nOutputRows; j++) {
                auto weights = (int16_t*)diag->pWeights;
                auto biases = (int32_t*)diag->pBiases;
                // identity matrix transformed to 16384 values
                if (weights[j] != MAX_VAL_2B_WEIGHT || biases[j] != 0) {
                    beforePadding = j;
                    break;
                }
            }

            for (int j = beforePadding; j < foo->pLayers[i].nOutputRows; j++) {
                auto weights = (int16_t*)diag->pWeights;
                auto biases = (int32_t*)diag->pBiases;
                // identity matrix transformed to 16384 values
                if (weights[j] != 0 || biases[j] != 0) {
                    bWeightsOK = false;
                    break;
                }
            }

            // if all weights are zero, or zero value doesn't look like padding
            if (!bWeightsOK && beforePadding == -1) continue;
            actualQuantity ++;
        }
        // means any quantity > 0
        if (matchQuantity == -1) {
            if (actualQuantity > 0)
                return matchInserted;
            else
                return !matchInserted;
        }
        if (actualQuantity == matchQuantity)
            return matchInserted;
        else
            return !matchInserted;

    };
    void DescribeTo(::std::ostream *os) const override {
        *os << "should "<< (matchInserted ? "" : "not ") << "have "
            << (matchQuantity == -1 ? "any" : std::to_string(matchQuantity))
            << " Identity Diagonal Primitive primitive as part of nnet structure, but was " << actualQuantity;
    }
};

inline ::testing::Matcher<const gna_nnet_type_t*> HasDiagonalLayer(bool matchInserted = false, int matchQuantity = -1) {
    std::unique_ptr<NNetComponentMatcher> c (new NNetComponentMatcher());
    c->add(new DiagLayerMatcher(matchInserted, matchQuantity));
    return ::testing::MakeMatcher(c.release());
}


