// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "backend/gna_types.h"

class OutputFiller : public ::testing::MatcherInterface<const gna_nnet_type_t*> {
    mutable std::stringstream reason;
    int32_t fill32BValue;
    int16_t fill16BValue;

 public:
    OutputFiller(int32_t fill32BValue, int16_t fill16BValue) : fill32BValue(fill32BValue), fill16BValue(fill16BValue) {}


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
            auto nElements = foo->pLayers[i].nOutputColumns * foo->pLayers[i].nOutputRows;
            if (foo->pLayers[i].nBytesPerOutput == 2) {
                std::fill_n((int16_t *) foo->pLayers[i].pOutputs, nElements, fill16BValue);
            } else if (foo->pLayers[i].nBytesPerOutput == 4) {
                std::fill_n((int32_t *) foo->pLayers[i].pOutputs, nElements, fill32BValue);
            } else {
                reason << "output bitness of layer [" << i << "] shouldn't be 16 or 32, but was " << foo->pLayers[i].nBytesPerOutput;
                return false;
            }
        }
        return true;
    }

    void DescribeTo(::std::ostream *os) const override {
        *os << "Not a Matcher but a fake, but error happened anyway: " << reason.str();
    }

};

