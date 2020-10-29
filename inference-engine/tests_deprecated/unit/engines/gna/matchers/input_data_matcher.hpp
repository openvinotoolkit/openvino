// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#pragma once

#include <gmock/gmock-matchers.h>
#include "nnet_base_matcher.hpp"

class InputDataMatcher : public ::testing::MatcherInterface<const intel_nnet_type_t *> {
    std::vector<int16_t> refInput;
public:

    explicit InputDataMatcher(const std::vector<int16_t> &_refInput) : refInput(_refInput) {}

    bool MatchAndExplain(const intel_nnet_type_t *foo, ::testing::MatchResultListener *listener) const override {
        if (foo->pLayers == nullptr) {
            *listener << "Address of the first layer descriptor is NULL";
            return false;
        }
        auto firstLayer = foo->pLayers[0];
        auto actualInput = firstLayer.pInputs;
        if (!actualInput) {
            *listener << "Input of the first layer is NULL";
            return false;
        }

        auto *actualInputI16 = reinterpret_cast<int16_t *>(actualInput);
        for (int i = 0; i < refInput.size(); i++) {
            if (actualInputI16[i] != refInput[i]) {
                *listener << "Actual and reference value of input doesn't match: " << actualInputI16[i] << " vs "
                          << refInput[i];
            }
        }
        return true;
    }

    void DescribeTo(::std::ostream *os) const override {}
};
