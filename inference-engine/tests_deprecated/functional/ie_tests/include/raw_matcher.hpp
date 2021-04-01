// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <string>
#include <map>
#include "base_matcher.hpp"
#include <ie_blob.h>

namespace Regression {
namespace Matchers {

class RawMatcher : public BaseMatcher {
    InferenceEngine::BlobMap outputBlobs;
public:
    RawMatcher(const RegressionConfig &config)
            : BaseMatcher(config) {
    }

    virtual void match();

    void checkResult(const std::map<std::string, std::map<size_t, float>> &allExpected);

    void to(const std::map<std::string, std::map<size_t, float>> &allExpected) {
        ASSERT_NO_FATAL_FAILURE(match());
        ASSERT_NO_FATAL_FAILURE(checkResult(allExpected));
    }

};

}
} //  namespace matchers
