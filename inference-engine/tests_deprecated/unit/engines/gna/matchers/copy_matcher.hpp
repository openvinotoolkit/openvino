// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "nnet_base_matcher.hpp"
class CopyLayerMatcher : public ::testing::MatcherInterface<const intel_nnet_type_t*> {
    bool matchInserted;
    const int matchQuantity;
 public:
    CopyLayerMatcher(bool matchInserted, int matchQuantity) : matchInserted(matchInserted), matchQuantity(matchQuantity) {}
    bool MatchAndExplain(const intel_nnet_type_t *foo, ::testing::MatchResultListener *listener) const override {
        if (foo == nullptr)
            return false;
        for(int i = 0; i < foo->nLayers; i++) {
            if (foo->pLayers[i].nLayerKind != INTEL_COPY) continue;
            return matchInserted;
        }
        return !matchInserted;
    };
    void DescribeTo(::std::ostream *os) const override {
        *os << "should "<< (matchInserted ? "" : "not ") << "have Copy primitive as part of nnet structure";
    }
};

inline ::testing::Matcher<const intel_nnet_type_t*> HasCopyLayer(bool matchInserted = false, int matchQuantity = -1) {
    std::unique_ptr<NNetComponentMatcher> c (new NNetComponentMatcher());
    c->add(new CopyLayerMatcher(matchInserted, matchQuantity));
    return ::testing::MakeMatcher(c.release());
}


