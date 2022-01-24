// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "backend/gna_types.h"
#include "nnet_base_matcher.hpp"
#include "frontend/quantization.h"

class ConvoluionLayerMatcher : public ::testing::MatcherInterface<const gna_nnet_type_t*> {
    bool matchInserted;
    int matchQuantity;
 public:
    ConvoluionLayerMatcher(bool matchInserted, int matchQuantity) : matchInserted(matchInserted), matchQuantity(matchQuantity) {}
    bool MatchAndExplain(const gna_nnet_type_t *foo, ::testing::MatchResultListener *listener) const override {
        if (foo == nullptr)
            return false;
        for(int i = 0; i < foo->nLayers; i++) {
            if (foo->pLayers[i].nLayerKind != INTEL_CONVOLUTIONAL)
                continue;

            return matchInserted;
        }
        return !matchInserted;
    };
    void DescribeTo(::std::ostream *os) const override {
        *os << "should "<< (matchInserted ? "" : "not ") << "have Convolution primitive as part of nnet structure";
    }
};



