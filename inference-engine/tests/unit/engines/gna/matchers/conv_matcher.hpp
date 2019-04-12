/*
 * INTEL CONFIDENTIAL
 * Copyright (C) 2018-2019 Intel Corporation.
 *
 * The source code contained or described herein and all documents
 * related to the source code ("Material") are owned by Intel Corporation
 * or its suppliers or licensors. Title to the Material remains with
 * Intel Corporation or its suppliers and licensors. The Material may
 * contain trade secrets and proprietary and confidential information
 * of Intel Corporation and its suppliers and licensors, and is protected
 * by worldwide copyright and trade secret laws and treaty provisions.
 * No part of the Material may be used, copied, reproduced, modified,
 * published, uploaded, posted, transmitted, distributed, or disclosed
 * in any way without Intel's prior express written permission.
 *
 * No license under any patent, copyright, trade secret or other
 * intellectual property right is granted to or conferred upon you by
 * disclosure or delivery of the Materials, either expressly, by implication,
 * inducement, estoppel or otherwise. Any license under such intellectual
 * property rights must be express and approved by Intel in writing.
 *
 * Include any supplier copyright notices as supplier requires Intel to use.
 *
 * Include supplier trademarks or logos as supplier requires Intel to use,
 * preceded by an asterisk. An asterisked footnote can be added as follows:
 * *Third Party trademarks are the property of their respective owners.
 *
 * Unless otherwise agreed by Intel in writing, you may not remove or alter
 * this notice or any other notice embedded in Materials by Intel or Intel's
 * suppliers or licensors in any way.
 */

#pragma once

#include"gna-api.h"
#include "nnet_base_matcher.hpp"
#include "quantization/quantization.h"

class ConvoluionLayerMatcher : public ::testing::MatcherInterface<const intel_nnet_type_t*> {
    bool matchInserted;
    int matchQuantity;
 public:
    ConvoluionLayerMatcher(bool matchInserted, int matchQuantity) : matchInserted(matchInserted), matchQuantity(matchQuantity) {}
    bool MatchAndExplain(const intel_nnet_type_t *foo, ::testing::MatchResultListener *listener) const override {
        if (foo == nullptr)
            return false;
        for(int i = 0; i < foo->nLayers; i++) {
            if (foo->pLayers[i].nLayerKind != INTEL_CONVOLUTIONAL) continue;

            auto conv = (intel_convolutional_layer_t*)foo->pLayers[i].pLayerStruct;

            return matchInserted;
        }
        return !matchInserted;
    };
    void DescribeTo(::std::ostream *os) const override {
        *os << "should "<< (matchInserted ? "" : "not ") << "have Convolution primitive as part of nnet structure";
    }
};



