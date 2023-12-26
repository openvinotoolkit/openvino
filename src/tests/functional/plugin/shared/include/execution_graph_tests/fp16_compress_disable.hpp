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

using ExecGraphDisableFP16CompressSpecificParams = std::tuple<
    bool,
    bool,
    std::string                         // Target Device
>;

class ExecGraphDisableFP16Compress : public testing::WithParamInterface<ExecGraphDisableFP16CompressSpecificParams>,
                                 public ov::test::TestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ExecGraphDisableFP16CompressSpecificParams> obj);
    std::string targetDevice;

protected:
    void SetUp() override;
    void TearDown() override;
    void create_model();
    void checkInferPrecision();
    bool matmulFP16Disabled;
    bool FCFP16Disabled;
    std::string matmulPrecision;
    std::string fcPrecision;
    std::shared_ptr<ov::Model> function = nullptr;
    std::string enforcedPrecision;
};

}  // namespace ExecutionGraphTests
