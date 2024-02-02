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

using ExecGraphDisableLoweringPrecisionSpecificParams = std::tuple<
    bool,                               //Disable lowering precision on device
    std::string,                        //Target Device
    ov::element::Type                   //Infer precision on target device
>;

class ExecGraphDisableLoweringPrecision : public testing::WithParamInterface<ExecGraphDisableLoweringPrecisionSpecificParams>,
                                 public ov::test::TestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ExecGraphDisableLoweringPrecisionSpecificParams> obj);

protected:
    void SetUp() override;
    void TearDown() override;
    void create_model();
    void checkInferPrecision();
    bool disableLoweringPrecision;
    std::string targetDevice;
    ov::element::Type loweringPrecision;
    std::shared_ptr<ov::Model> funcPtr = nullptr;
};

}  // namespace ExecutionGraphTests
