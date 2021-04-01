// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "custom_matcher.hpp"

namespace Regression { namespace Matchers {

using namespace InferenceEngine;

class OptimizedNetworkMatcher : public CustomMatcher {
 protected:
    std::string path_to_reference_dump;
    std::vector<uint8_t> firmware;
    InferenceEngine::ExecutableNetwork executableApi;
 public:

    explicit OptimizedNetworkMatcher(const RegressionConfig &config)
        : CustomMatcher(config, true) {
    }
    ~OptimizedNetworkMatcher() {
        if (match_in_dctor) {
            matchCustom();
            checkResult();
            //not allow base matcher to match one more time
            match_in_dctor = false;
        }
    }

    void matchCustom();

    void to(std::string path_to_reference_dump);
    std::vector<uint8_t> readDumpFromFile(std::string path);
    void checkResult();
};

class OptimizedNetworkDumper : public OptimizedNetworkMatcher {
 public:
    using OptimizedNetworkMatcher::OptimizedNetworkMatcher;

    ~OptimizedNetworkDumper() {
        if (match_in_dctor) {
            dump();
            //not allow base matcher to match one more time
            match_in_dctor = false;
        }
    }

    void match() {}

    void dump();

};

} //  namespace Regression
} //  namespace Matchers
