// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base_matcher.hpp"
#include <cmath>
#include <cpp/ie_cnn_network.h>

IE_SUPPRESS_DEPRECATED_START
namespace Regression { namespace Matchers {

using namespace InferenceEngine;

class CustomMatcher : public BaseMatcher {
 protected:
    InferenceEngine::CNNNetwork network;
    InferenceContext ctx;
    bool match_in_dctor = false;
    int precision;

 public:

    explicit CustomMatcher(const RegressionConfig &config, bool match_in_dctor = false)
            : BaseMatcher(config),
              match_in_dctor(match_in_dctor),
              precision(4) {
        if (!match_in_dctor) {
            matchCustom();
            checkResult();
        }
    }
    ~CustomMatcher() {
        if (match_in_dctor) {
            matchCustom();
            checkResult();
        }
    }

    CustomMatcher& withAvgDelta(float value) {
        BaseMatcher::config.nearAvgValue = value;
        return *this;
    }

    CustomMatcher& withDelta(float value) {
        BaseMatcher::config.nearValue = value;
        return *this;
    }

    CustomMatcher& setPrecision(int precision) {
        this->precision = precision;
        return *this;
    }

    void matchCustom();

    template<typename TReal>
    inline bool isApproximatelyEqual(TReal a, TReal b, TReal tolerance = std::numeric_limits<TReal>::epsilon())
    {
        TReal diff = std::fabs(a - b);
        if (diff <= tolerance)
            return true;

        if (diff < std::fmax(std::fabs(a), std::fabs(b)) * tolerance)
            return true;

        return false;
    }

    void checkResult();

 protected:
    InferenceEngine::ExecutableNetwork createExecutableNetworkFromIR();
    InferenceEngine::ExecutableNetwork createExecutableNetworkFromAOT();
};

}
} //  namespace Matchers
IE_SUPPRESS_DEPRECATED_END
