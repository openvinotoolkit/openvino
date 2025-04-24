// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
typedef std::tuple<
        std::vector<InputShape>,               // input shape
        int64_t ,                              // Max rois
        ElementType,                           // Model type
        std::string                            // Device name
> ExperimentalDetectronTopKROIsTestParams;

class ExperimentalDetectronTopKROIsLayerTest : public testing::WithParamInterface<ExperimentalDetectronTopKROIsTestParams>,
                                               virtual public SubgraphBaseTest {
protected:
    void SetUp() override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<ExperimentalDetectronTopKROIsTestParams>& obj);
};
} // namespace test
} // namespace ov
