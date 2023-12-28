// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"

namespace ExecutionGraphTests {

using ExecGraphDisableLowingPrecisionSpecificParams = std::tuple<
    bool,                               //Disable lowering precision on device
    std::string                         // Target Device
>;

class ExecGraphDisableLoweringPrecision : public testing::WithParamInterface<ExecGraphDisableLowingPrecisionSpecificParams>,
                                 public ov::test::TestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ExecGraphDisableLowingPrecisionSpecificParams> obj);
    std::string targetDevice;

protected:
    void SetUp() override;
    void TearDown() override;
    void create_model();
    void checkInferPrecision();
    bool disableLoweringPrecision;
    std::shared_ptr<ov::Model> funcPtr = nullptr;
};

}  // namespace ExecutionGraphTests
