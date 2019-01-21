// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "nnet_base_matcher.hpp"

class PWLMatcher : public ::testing::MatcherInterface<const intel_nnet_type_t*> {
    bool matchInserted;
    const int matchQuantity;
    mutable int timesInserted = 0;
 public:
    PWLMatcher(bool inserted, int matchQuantity) : matchInserted(inserted), matchQuantity(matchQuantity) {}

    bool MatchAndExplain(const intel_nnet_type_t *foo, ::testing::MatchResultListener *listener) const override {
        if (foo == nullptr)
            return false;
        timesInserted = 0;
        for(int i = 0; i < foo->nLayers; i++) {
            if (foo->pLayers[i].nLayerKind != INTEL_AFFINE &&
                foo->pLayers[i].nLayerKind != INTEL_AFFINE_DIAGONAL &&
                foo->pLayers[i].nLayerKind != INTEL_CONVOLUTIONAL) continue;
            auto affine = reinterpret_cast<intel_affine_layer_t*>(foo->pLayers[i].pLayerStruct);
            if (affine == nullptr) continue;

            bool hasPwl = affine->pwl.nSegments != 0 && affine->pwl.pSegments != nullptr;

            if (hasPwl) {
                if (matchQuantity == -1)
                    return matchInserted;
                else
                    timesInserted ++;
            }
        }
        if (matchInserted) {
            if (matchQuantity != -1) {
                return timesInserted == matchQuantity;
            }
            return timesInserted != 0;
        }

        return timesInserted == 0;
    };
    void DescribeTo(::std::ostream *os) const override {
        if (!matchInserted ) {
            *os << "should not have PWL layer as part of nnet structure, but was found " << timesInserted <<" times" ;
        } else {
            if (matchQuantity == -1) {
                *os << "should have PWL layer as part of nnet structure, but it was not found " ;
            } else {
                *os << "should have PWL layer as part of nnet structure, for " << matchQuantity <<" times, but was found only " << timesInserted ;
            }
        }
    }
};

inline ::testing::Matcher<const intel_nnet_type_t*> HasPwlLayer(bool inserted = true, int matchQuantity = -1) {
    std::unique_ptr<NNetComponentMatcher> c (new NNetComponentMatcher());
    c->add(new PWLMatcher(inserted, matchQuantity));
    return ::testing::MakeMatcher(c.release());
}
