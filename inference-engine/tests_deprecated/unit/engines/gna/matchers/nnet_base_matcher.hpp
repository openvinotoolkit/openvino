// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "backend/gna_types.h"
#include "gna_lib_ver_selector.hpp"

class NNetComponentMatcher : public ::testing::MatcherInterface<const gna_nnet_type_t*> {
    std::vector<std::shared_ptr<::testing::MatcherInterface<const gna_nnet_type_t*>>> matchers;
    mutable int failIdx = -1;
    mutable std::stringstream reason;
    int bitness;
 public:
    NNetComponentMatcher(int bitness  = 16) : bitness(bitness) {}
    NNetComponentMatcher& add(::testing::MatcherInterface<const gna_nnet_type_t*> * p) {
        matchers.push_back(std::shared_ptr<::testing::MatcherInterface<const gna_nnet_type_t*>>(p));
        return *this;
    }
    bool empty() const {
        return matchers.empty();
    }
    bool MatchAndExplain(const gna_nnet_type_t* foo, ::testing::MatchResultListener* listener) const override {
        if (foo == nullptr)
            return false;
        reason.str("");
        // checking pointers are set
        for (int i=0; i < foo->nLayers; i++) {
            if (nullptr == foo->pLayers[i].pInputs ||
                nullptr == foo->pLayers[i].pOutputs) {
                reason << "input/output pointers in pLayers[" << i << "] shouldn't be null NULL";
                return false;
            }
            if (foo->pLayers[i].nBytesPerInput * 8 != bitness) {
                reason << "numberOfBytes per input in pLayers[" << i << "] should be " << (bitness/8) << ", but was "
                    << foo->pLayers[i].nBytesPerInput;
                return false;
            }

            if (foo->pLayers[i].nBytesPerOutput * 8 != bitness) {
                // if this output is a output to a bias this is fine
                // also if this output is defacto network output - other words this whouldnt use in inputs,
                for (int j=0; j < foo->nLayers; j++) {
                    // bad
                    if (foo->pLayers[j].pInputs == foo->pLayers[i].pOutputs) {
                        reason << "numberOfBytes per output int pLayers[" << i << "] should be " << (bitness/8) << ", but was "
                               << foo->pLayers[i].nBytesPerOutput << " cannot use this output as inputs for layer :" << j;
                        return false;
                    }
                    if (foo->pLayers[j].nLayerKind == INTEL_AFFINE ||
                        foo->pLayers[j].nLayerKind == INTEL_AFFINE_DIAGONAL) {
                        auto pAffine = reinterpret_cast<gna_affine_func_t*>(foo->pLayers[j].pLayerStruct);

                        if (pAffine->pWeights == foo->pLayers[i].pOutputs) {
                            reason << "numberOfBytes per output int pLayers[" << i << "] should be " << (bitness/8) << ", but was "
                                   << foo->pLayers[i].nBytesPerOutput << " cannot use this output as weights for affine layer :" << j;
                            return false;
                        }
                    }
                }
            }
        }

        int i = 0;
        for (auto && matcher : matchers) {
            bool res = matcher->MatchAndExplain(foo, listener);
            if (!res) {
                failIdx = i;
                return false;
            }
            i++;
        }
        return true;
    }

    void DescribeTo(::std::ostream* os) const override {

        if (failIdx != -1) {
            matchers[failIdx]->DescribeTo(os);
            return;
        }

        *os << reason.str();
    }

};

