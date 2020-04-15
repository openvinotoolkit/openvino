// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "nnet_base_matcher.hpp"
class CopyLayerMatcher : public ::testing::MatcherInterface<const intel_nnet_type_t*> {
    bool matchInserted;
    const int matchQuantity;
    mutable int actualNumberOfCopyLayers;
 public:
    CopyLayerMatcher(bool matchInserted, int matchQuantity) : matchInserted(matchInserted), matchQuantity(matchQuantity) {}
    bool MatchAndExplain(const intel_nnet_type_t *foo, ::testing::MatchResultListener *listener) const override {
        if (foo == nullptr)
            return false;
        actualNumberOfCopyLayers = 0;

        for(int i = 0; i < foo->nLayers; i++) {
            if (foo->pLayers[i].nLayerKind != INTEL_COPY) continue;

            if (!matchInserted) {
                return false;
            }
            actualNumberOfCopyLayers ++;
        }
        if (matchQuantity == -1) {
            if (actualNumberOfCopyLayers > 0) {
                return true;
            }
            return false;
        }
        if (actualNumberOfCopyLayers != matchQuantity) {
            return false;
        }
        return true;
    };
    void DescribeTo(::std::ostream *os) const override {
        *os << "should "<< (matchInserted ? "" : "not ") << "have " << (matchInserted ? std::to_string(matchQuantity) : "" )
            << " Copy primitives as part of nnet structure" << (matchInserted ? std::string(" but was only: ") + std::to_string(actualNumberOfCopyLayers) + " copy layers" : "" );
    }
};

inline ::testing::Matcher<const intel_nnet_type_t*> HasCopyLayer(bool matchInserted = false, int matchQuantity = -1) {
    std::unique_ptr<NNetComponentMatcher> c (new NNetComponentMatcher());
    c->add(new CopyLayerMatcher(matchInserted, matchQuantity));
    return ::testing::MakeMatcher(c.release());
}


