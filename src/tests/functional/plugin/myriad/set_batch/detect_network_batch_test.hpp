// Copyright (C) 2018-2022 Intel Corporation
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
    unsigned int                   // Batch size
> DetectNetworkBatchParams;

class DetectNetworkBatch : public testing::WithParamInterface<DetectNetworkBatchParams>,
                        virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DetectNetworkBatchParams>& obj);

protected:
    void SetUp() override;
    void LoadNetwork() override;

protected:
    unsigned int  m_batchSize = 1;
};

}  // namespace LayerTestsDefinitions
