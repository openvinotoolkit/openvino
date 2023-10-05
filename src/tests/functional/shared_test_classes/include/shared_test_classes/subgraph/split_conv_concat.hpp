// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"

namespace SubgraphTestsDefinitions {

class SplitConvConcat : public testing::WithParamInterface<LayerTestsUtils::basicParams>,
                        virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LayerTestsUtils::basicParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace SubgraphTestsDefinitions
