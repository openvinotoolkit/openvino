// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"

namespace LowPrecisionTestDefinitions {

typedef std::tuple<
    InferenceEngine::Precision,                                 // Net precision
    LayerTestsUtils::TargetDevice,                              // Device name
    std::pair<std::string, std::map<std::string, std::string>>  // Configuration
> lowPrecisionTestParamsSet;

class LowPrecisionTest : public testing::WithParamInterface<lowPrecisionTestParamsSet>,
    virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<lowPrecisionTestParamsSet>& obj);

protected:
    void SetUp() override;
};
// ! [test_low_precision:definition]

}  // namespace LowPrecisionTestDefinitions
