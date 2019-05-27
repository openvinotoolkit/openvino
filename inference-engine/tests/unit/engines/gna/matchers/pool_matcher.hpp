// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include"gna-api.h"
#include "nnet_base_matcher.hpp"
#include "quantization/quantization.h"

class PoolingLayerMatcher : public ::testing::MatcherInterface<const intel_nnet_type_t*> {
    bool matchInserted;
    int matchQuantity;
    bool bMaxPool;
 public:
    PoolingLayerMatcher(bool matchInserted, int matchQuantity, bool bMaxPool)
        : matchInserted(matchInserted), matchQuantity(matchQuantity), bMaxPool(bMaxPool) {}
    bool MatchAndExplain(const intel_nnet_type_t *foo, ::testing::MatchResultListener *listener) const override {
        if (foo == nullptr)
            return false;
        for(int i = 0; i < foo->nLayers; i++) {
            if (foo->pLayers[i].nLayerKind != INTEL_CONVOLUTIONAL) continue;

            auto conv = (intel_convolutional_layer_t*)foo->pLayers[i].pLayerStruct;
            if (conv->poolType != INTEL_MAX_POOLING) continue;

            return matchInserted;
        }
        return !matchInserted;
    };
    void DescribeTo(::std::ostream *os) const override {
        *os << "should "<< (matchInserted ? "" : "not ") << "have MaxPooling primitive as part of nnet structure";
    }
};



