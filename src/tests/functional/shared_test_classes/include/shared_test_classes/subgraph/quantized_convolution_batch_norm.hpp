// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace SubgraphTestsDefinitions {

class QuantizedConvolutionBatchNorm : public testing::WithParamInterface<std::string>,
                                            virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::string>& obj);

protected:
    void SetUp() override;
    void TearDown() override;
};

}  // namespace SubgraphTestsDefinitions
