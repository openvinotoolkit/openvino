// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "regression_config.hpp"
#include <tests_common.hpp>

namespace Regression { namespace Matchers {

using namespace InferenceEngine;

class BaseMatcher {
protected:
    RegressionConfig config;
public:
    explicit BaseMatcher(const RegressionConfig &config) : config(config) {
#ifndef NDEBUG
        std::cout << "Matching on " << config._device_name << std::endl;
#endif
    }

    void checkImgNumber(int dynBatch = -1);
};

void loadImage(const std::string &imageFilename, InferenceEngine::Blob::Ptr &blob, bool bgr = true, int batchNumber = 1);

}  // namepspace Matchers
}  // namespace Regression
