// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"


namespace LayerTestsDefinitions {

typedef std::tuple<
    LayerTestsUtils::TargetDevice, // Device name
    std::map<std::string, std::string>
> VpuScaleParams;

class VpuScaleTest : public testing::WithParamInterface<VpuScaleParams>,
                        virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<VpuScaleParams>& obj);

protected:
    void SetUp() override;
protected:
    std::map<std::string, std::string> additionalConfig = {};
};

}  // namespace LayerTestsDefinitions
